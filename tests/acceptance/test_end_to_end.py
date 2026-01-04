from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from free_mlops.api import create_app
from free_mlops.service import run_experiment
from free_mlops.service import save_uploaded_bytes


def test_end_to_end_upload_train_and_predict(settings, classification_csv: Path) -> None:
    file_bytes = classification_csv.read_bytes()

    dataset_path = save_uploaded_bytes(
        file_bytes=file_bytes,
        original_filename="dataset.csv",
        settings=settings,
    )

    record = run_experiment(
        dataset_path=dataset_path,
        target_column="target",
        problem_type="classification",
        settings=settings,
    )

    model_path = Path(record["model_path"])
    assert model_path.exists()

    app = create_app(settings=settings)
    client = TestClient(app)

    resp = client.post(
        "/predict",
        json={
            "experiment_id": record["id"],
            "records": [{"num1": 0.2, "num2": 1, "cat": "B"}],
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["experiment_id"] == record["id"]
    assert len(body["predictions"]) == 1
