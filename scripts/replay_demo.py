"""Explicit historical-data replay through the real local application."""

import json
from pathlib import Path

from fastapi.testclient import TestClient

from agri_platform.api import create_app

root = Path.cwd()
client = TestClient(create_app(root))
for payload in json.loads((root / "configs/replay-inputs.json").read_text()):
    response = client.post("/predict", json=payload)
    response.raise_for_status()
result = client.get("/monitoring").json()
(root / "reports/replay-monitoring.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
