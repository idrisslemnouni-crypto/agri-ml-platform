"""Strict local research-demo API and dashboard."""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, StrictFloat

from agri_platform.data import FEATURES
from agri_platform.monitoring import record, summarize


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, StrictFloat | None] = Field(min_length=1, max_length=16)
    source: Literal["manual", "historical-replay"] = "manual"


def validate_features(payload):
    if set(payload) != set(FEATURES):
        raise ValueError("Exact feature schema required")
    if payload["SOIL_MOISTURE_10_DAILY"] is None:
        raise ValueError("Current observed 10 cm moisture is required")
    for name, value in payload.items():
        if value is None:
            continue
        if isinstance(value, bool) or not np.isfinite(value):
            raise ValueError("Finite numeric values required")
        if (name.startswith("SOIL_MOISTURE") or name.startswith("soil10_")) and not 0 <= value <= 1:
            raise ValueError("Moisture must be m3/m3 in [0,1]")
        if name.startswith("RH_") and not 0 <= value <= 100:
            raise ValueError("Relative humidity must be percent in [0,100]")
        if name.startswith("T_DAILY") and not -80 <= value <= 60:
            raise ValueError("Unsupported Celsius temperature")
        if (name.startswith("P_DAILY") or name.startswith("SOLARAD")) and value < 0:
            raise ValueError("Rainfall and solar radiation must be nonnegative")
        if name.startswith("doy_") and not -1 <= value <= 1:
            raise ValueError("Cyclic feature outside [-1,1]")


def load_release(root):
    release = json.loads((root / "models/release.json").read_text())
    artifact = (root / release["artifact"]).resolve()
    if not artifact.is_relative_to(root.resolve()) or not release.get("validation_gate_passed"):
        raise ValueError("Invalid local release")
    if (
        release["features"] != FEATURES
        or release["feature_schema_sha256"]
        != hashlib.sha256(json.dumps(FEATURES).encode()).hexdigest()
    ):
        raise ValueError("Feature schema mismatch")
    if hashlib.sha256(artifact.read_bytes()).hexdigest() != release["sha256"]:
        raise ValueError("Model checksum mismatch")
    state = joblib.load(artifact)  # Only checksum-bound artifacts trained locally are trusted.
    if state["kind"] != "random_forest" or state["features"] != FEATURES:
        raise ValueError("Unsupported artifact contract")
    return release, state


def create_app(root: Path):
    app = FastAPI(
        title="Agri ML local platform",
        description="Measured-soil research screening; no validated irrigation prescription",
    )
    app.state.release = None
    app.state.model = None
    try:
        app.state.release, app.state.model = load_release(root)
    except (OSError, ValueError, KeyError) as exc:
        app.state.release_error = str(exc)

    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return (Path(__file__).parent / "templates/index.html").read_text(encoding="utf-8")

    @app.get("/health")
    def health():
        if app.state.release is None:
            raise HTTPException(503, "No verified local release; run prepare first")
        return {
            "status": "ready",
            "release_id": app.state.release["release_id"],
            "scope": app.state.release["scope"],
        }

    @app.get("/example")
    def example():
        path = root / "configs/example-input.json"
        if not path.exists():
            raise HTTPException(503, "Run local preparation first")
        return {"features": json.loads(path.read_text()), "source": "manual"}

    @app.post("/predict")
    def inference(request: Request):
        if app.state.release is None:
            raise HTTPException(503, "No verified local release")
        try:
            validate_features(request.features)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        start = time.perf_counter()
        x = pd.DataFrame([request.features], columns=FEATURES).astype(float)
        p = float(app.state.model["model"].predict_proba(x)[0, 1])
        elapsed = (time.perf_counter() - start) * 1000
        record(
            root / "models/telemetry.sqlite",
            app.state.release["release_id"],
            p,
            elapsed,
            request.source,
            request.features,
        )
        return {
            "release_id": app.state.release["release_id"],
            "next_day_below_proxy_probability": p,
            "screening_flag": p >= app.state.model["probability_threshold"],
            "threshold_m3_m3": 0.20,
            "scope": app.state.release["scope"],
        }

    @app.get("/monitoring")
    def monitoring(source: Literal["all", "manual", "historical-replay"] = "all"):
        if app.state.release is None:
            raise HTTPException(503, "No verified local release")
        return summarize(
            root / "models/telemetry.sqlite",
            app.state.release["reference_distributions"],
            app.state.release["release_id"],
            source=source,
        )

    return app


app = create_app(Path(os.environ.get("AGRI_ROOT", Path.cwd())))
