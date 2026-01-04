from __future__ import annotations

from fastapi.testclient import TestClient

from free_mlops.api import create_app
from free_mlops.service import run_experiment


def test_api_health(settings) -> None:
    app = create_app(settings=settings)
    client = TestClient(app)

    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_api_predict_without_model_returns_404(settings) -> None:
    app = create_app(settings=settings)
    client = TestClient(app)

    resp = client.post("/predict", json={"records": [{"a": 1}]})
    assert resp.status_code == 404


def test_api_predict_with_trained_model(settings, classification_csv) -> None:
    record = run_experiment(
        dataset_path=classification_csv,
        target_column="target",
        problem_type="classification",
        settings=settings,
    )

    app = create_app(settings=settings)
    client = TestClient(app)

    models_resp = client.get("/models")
    assert models_resp.status_code == 200
    models = models_resp.json()
    assert len(models) >= 1

    details_resp = client.get(f"/models/{record['id']}")
    assert details_resp.status_code == 200
    details = details_resp.json()
    assert details["id"] == record["id"]

    predict_resp = client.post(
        "/predict",
        json={
            "records": [{"num1": 0.1, "num2": 3, "cat": "A"}],
        },
    )
    assert predict_resp.status_code == 200
    payload = predict_resp.json()
    assert "predictions" in payload
    assert len(payload["predictions"]) == 1
