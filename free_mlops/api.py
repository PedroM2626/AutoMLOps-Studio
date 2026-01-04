from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi import HTTPException

from free_mlops.config import Settings
from free_mlops.config import get_settings
from free_mlops.db import get_experiment
from free_mlops.db import get_latest_experiment
from free_mlops.db import init_db
from free_mlops.db import list_experiments
from free_mlops.schemas import ExperimentDetails
from free_mlops.schemas import ExperimentSummary
from free_mlops.schemas import HealthResponse
from free_mlops.schemas import PredictRequest
from free_mlops.schemas import PredictResponse
from free_mlops.service import align_features
from free_mlops.service import load_model


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    init_db(settings.db_path)

    app = FastAPI(title="Free MLOps API", version="0.1.0")

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.get("/models", response_model=list[ExperimentSummary])
    def models() -> list[ExperimentSummary]:
        records = list_experiments(settings.db_path, limit=200, offset=0)
        return [
            ExperimentSummary(
                id=r["id"],
                created_at=r["created_at"],
                problem_type=r["problem_type"],
                target_column=r["target_column"],
                best_model_name=r["best_model_name"],
            )
            for r in records
        ]

    @app.get("/models/{experiment_id}", response_model=ExperimentDetails)
    def model_details(experiment_id: str) -> ExperimentDetails:
        record = get_experiment(settings.db_path, experiment_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Experiment not found")
        return ExperimentDetails(**record)

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        if not payload.records:
            raise HTTPException(status_code=400, detail="records must not be empty")

        record: dict[str, Any] | None
        if payload.experiment_id:
            record = get_experiment(settings.db_path, payload.experiment_id)
        else:
            record = get_latest_experiment(settings.db_path)

        if record is None:
            raise HTTPException(status_code=404, detail="No trained model found")

        model_path = Path(record["model_path"]) 
        if not model_path.exists():
            raise HTTPException(status_code=500, detail="Model artifact not found on disk")

        feature_columns = list(record.get("feature_columns", []))
        if not feature_columns:
            raise HTTPException(status_code=500, detail="Model metadata missing feature_columns")

        X = align_features(payload.records, feature_columns=feature_columns)

        model = load_model(model_path)
        try:
            preds = model.predict(X)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

        return PredictResponse(
            experiment_id=str(record["id"]),
            model_name=str(record["best_model_name"]),
            predictions=list(pd.Series(preds).tolist()),
        )

    return app


def main() -> None:
    settings = get_settings()
    app = create_app(settings=settings)
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)


if __name__ == "__main__":
    main()
