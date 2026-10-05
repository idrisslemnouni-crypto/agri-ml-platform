import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from agri_platform.api import create_app, load_release, validate_features
from agri_platform.data import FEATURES
from agri_platform.monitoring import psi


def test_no_release_health_and_dashboard(tmp_path):
    client = TestClient(create_app(tmp_path))
    assert client.get("/health").status_code == 503
    assert client.get("/").status_code == 200


def test_psi_equal_distribution_and_shift():
    reference = {"boundaries": [0.5], "proportions": [0.5, 0.5]}
    assert psi(np.r_[np.zeros(30), np.ones(30)], reference) == pytest.approx(0)
    assert psi(np.ones(60), reference) > 0.2
    assert psi(np.ones(2), reference) is None


def test_feature_contract():
    sample = {name: None for name in FEATURES}
    sample["SOIL_MOISTURE_10_DAILY"] = 0.3
    validate_features(sample)
    with pytest.raises(ValueError):
        validate_features({**sample, "extra": 1})
    with pytest.raises(ValueError):
        validate_features({**sample, "soil10_lag1": 3})
    with pytest.raises(ValueError):
        validate_features({**sample, "T_DAILY_AVG": float("nan")})


def test_checksum_rejects_changed_model(tmp_path):
    (tmp_path / "models").mkdir()
    model = tmp_path / "models/selected.joblib"
    model.write_bytes(b"changed")
    release = {
        "artifact": "models/selected.joblib",
        "validation_gate_passed": True,
        "features": FEATURES,
        "feature_schema_sha256": hashlib.sha256(json.dumps(FEATURES).encode()).hexdigest(),
        "sha256": "incorrect",
    }
    (tmp_path / "models/release.json").write_text(json.dumps(release))
    with pytest.raises(ValueError, match="checksum"):
        load_release(tmp_path)


def test_actual_service_and_mlflow_release(tmp_path):
    import shutil

    root = Path(__file__).resolve().parents[1]
    if not (root / "models/release.json").exists():
        pytest.skip("Real local release is not committed")
    (tmp_path / "models").mkdir()
    for name in ["selected.joblib", "release.json"]:
        shutil.copy(root / "models" / name, tmp_path / "models" / name)
    release, state = load_release(tmp_path)
    assert state["kind"] == "random_forest" and release["mlflow_run_id"]
    client = TestClient(create_app(tmp_path))
    payload = json.loads((root / "configs/replay-inputs.json").read_text())[0]
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert 0 <= response.json()["next_day_below_proxy_probability"] <= 1
    assert client.post("/predict", json={**payload, "extra": 1}).status_code == 422
    bad = {**payload, "features": {**payload["features"], "RH_DAILY_AVG": 101}}
    assert client.post("/predict", json=bad).status_code == 422
    bad = {**payload, "features": {**payload["features"], "T_DAILY_AVG": True}}
    assert client.post("/predict", json=bad).status_code == 422
    assert client.get("/monitoring").json()["total_accepted"] == 1
