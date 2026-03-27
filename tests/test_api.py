import importlib

import numpy as np
from fastapi.testclient import TestClient

from src.tracking.telemetry import TelemetryStore


class DummyProcessor:
    def transform(self, df):
        return df.values, None


class DummyModel:
    def predict(self, X):
        return np.array([1] * len(X))


def _load_api_module(monkeypatch):
    monkeypatch.setenv("API_SECRET_KEY", "test-key")
    import api

    return importlib.reload(api)


def test_health_live(monkeypatch):
    api_module = _load_api_module(monkeypatch)
    client = TestClient(api_module.app)

    response = client.get("/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"


def test_predict_requires_valid_api_key(monkeypatch):
    api_module = _load_api_module(monkeypatch)
    client = TestClient(api_module.app)

    response = client.post("/predict", json={"data": [{"x": 1}]})
    assert response.status_code == 422

    response = client.post("/predict", headers={"x-api-key": "wrong"}, json={"data": [{"x": 1}]})
    assert response.status_code == 403


def test_predict_success_and_sqlite_telemetry(monkeypatch, tmp_path):
    api_module = _load_api_module(monkeypatch)
    api_module.model_assets["processor"] = DummyProcessor()
    api_module.model_assets["model"] = DummyModel()
    api_module.model_assets["model_name"] = "dummy.pkl"
    api_module.telemetry_store = TelemetryStore(str(tmp_path / "telemetry.db"))

    client = TestClient(api_module.app)
    response = client.post(
        "/predict",
        headers={"x-api-key": "test-key"},
        json={"data": [{"x": 1}, {"x": 2}]},
    )

    assert response.status_code == 200
    assert response.json()["predictions"] == [1, 1]
