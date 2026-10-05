"""Reproduce the fixed study, track its run and create a checksum-bound local release."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import mlflow

from agri_platform.benchmark import run, split_data
from agri_platform.data import FEATURES, features, read_sources
from agri_platform.monitoring import distributions


def prepare(root: Path):
    result = run(root)
    if (
        result["selected_on_validation"] != "random_forest"
        or result["validation"]["random_forest"]["positive_f1"]
        < result["validation"]["persistence"]["positive_f1"]
    ):
        raise ValueError("Previously selected validation-gated model did not reproduce")
    data, _ = read_sources(root)
    table = features(data, 1).dropna(subset=["target", "SOIL_MOISTURE_10_DAILY"])
    table["label"] = (table.target < result["config"]["proxy_threshold_m3_m3"]).astype(int)
    train, _, test = split_data(table, result["config"])
    tracking = root / "models/mlruns"
    mlflow.set_tracking_uri(tracking.resolve().as_uri())
    mlflow.set_experiment("measured-soil-screening")
    with mlflow.start_run(run_name="reproduce-and-release") as tracked:
        mlflow.log_params(
            {
                "method": "random_forest",
                "trees": 160,
                "max_depth": 10,
                "seed": 42,
                "features": len(FEATURES),
                "train_rows": len(train),
                "held_out_station": result["config"]["holdout_station"],
            }
        )
        mlflow.log_metrics(
            {
                "validation_f1": result["validation"]["random_forest"]["positive_f1"],
                "test_f1": result["test"]["random_forest"]["positive_f1"],
                "test_average_precision": result["test"]["random_forest"]["pr_auc"],
            }
        )
        for file in [
            root / "models/selected.joblib",
            root / "reports/metrics.json",
            root / "data/source-manifest.json",
            root / "configs/default.json",
        ]:
            mlflow.log_artifact(str(file), artifact_path="evidence")
        release = {
            "release_id": "soil-screening-v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact": "models/selected.joblib",
            "sha256": hashlib.sha256((root / "models/selected.joblib").read_bytes()).hexdigest(),
            "feature_schema_sha256": hashlib.sha256(json.dumps(FEATURES).encode()).hexdigest(),
            "features": FEATURES,
            "validation_gate_passed": True,
            "mlflow_run_id": tracked.info.run_id,
            "reference_distributions": distributions(train, FEATURES),
            "scope": "Measured shallow-soil threshold proxy; no irrigation ground truth or production validity",
        }
        (root / "models/release.json").write_text(json.dumps(release, indent=2, allow_nan=False))
        mlflow.log_artifact(str(root / "models/release.json"), artifact_path="release")
    replay = test.head(40)[FEATURES].copy()
    payloads = []
    for row in replay.to_dict(orient="records"):
        payloads.append(
            {
                "features": {k: float(v) if v == v else None for k, v in row.items()},
                "source": "historical-replay",
            }
        )
    (root / "configs/replay-inputs.json").write_text(
        json.dumps(payloads, indent=2, allow_nan=False)
    )
    return release


if __name__ == "__main__":
    release = prepare(Path.cwd())
    print(
        json.dumps(
            {
                "release_id": release["release_id"],
                "mlflow_run_id": release["mlflow_run_id"],
                "validation_gate": release["validation_gate_passed"],
            },
            indent=2,
        )
    )
