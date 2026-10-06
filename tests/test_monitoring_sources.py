import pytest
from fastapi.testclient import TestClient

from agri_platform.api import create_app
from agri_platform.monitoring import record, summarize

REFERENCE = {"moisture": {"boundaries": [0.5], "proportions": [0.5, 0.5]}}


def test_replay_cannot_displace_manual_window_or_change_its_psi(tmp_path):
    database = tmp_path / "telemetry.sqlite"
    for i in range(30):
        record(database, "release-A", 0.5, 2, "manual", {"moisture": i % 2})
    before = summarize(database, REFERENCE, "release-A", source="manual")
    for _ in range(305):
        record(database, "release-A", 0.9, 10, "historical-replay", {"moisture": 1})
    record(database, "other-release", 0.9, 50, "manual", {"moisture": 1})
    manual = summarize(database, REFERENCE, "release-A", source="manual")
    replay = summarize(database, REFERENCE, "release-A", source="historical-replay")
    combined = summarize(database, REFERENCE, "release-A")
    assert manual == before
    assert manual["total_accepted"] == manual["window_n"] == 30
    assert manual["source_counts"] == {"manual": 30}
    assert manual["feature_psi"]["moisture"] == pytest.approx(0)
    assert replay["total_accepted"] == 305 and replay["window_n"] == 300
    assert replay["feature_psi"]["moisture"] > 0.2
    assert combined["total_accepted"] == 335 and combined["window_n"] == 300
    assert combined["source_counts"] == {"historical-replay": 300}


def test_empty_filtered_population_and_invalid_source(tmp_path):
    database = tmp_path / "telemetry.sqlite"
    record(database, "A", 0.9, 2, "historical-replay", {"moisture": 1})
    result = summarize(database, REFERENCE, "A", "manual")
    assert result["selected_source"] == "manual"
    assert result["window_n"] == result["total_accepted"] == 0
    assert result["feature_psi"] == {}
    with pytest.raises(ValueError, match="source"):
        summarize(database, REFERENCE, "A", "untrusted")


def test_monitoring_api_filters_and_validates_source_without_a_model(tmp_path):
    app = create_app(tmp_path)
    app.state.release = {"release_id": "A", "reference_distributions": REFERENCE}
    record(tmp_path / "models/telemetry.sqlite", "A", 0.9, 2, "historical-replay", {})
    client = TestClient(app)
    assert client.get("/monitoring?source=manual").json()["total_accepted"] == 0
    response = client.get("/monitoring?source=historical-replay")
    assert response.status_code == 200
    assert response.json()["selected_source"] == "historical-replay"
    assert response.json()["total_accepted"] == 1
    assert client.get("/monitoring?source=all").json()["total_accepted"] == 1
    assert client.get("/monitoring?source=invalid").status_code == 422
