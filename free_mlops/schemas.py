from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(default="ok")


class ExperimentSummary(BaseModel):
    id: str
    created_at: str
    problem_type: str
    target_column: str
    best_model_name: str


class ExperimentDetails(ExperimentSummary):
    dataset_path: str
    test_size: float
    random_state: int
    n_rows: int
    n_cols: int
    feature_columns: list[str]
    best_metrics: dict[str, Any]
    leaderboard: list[dict[str, Any]]
    model_path: str


class PredictRequest(BaseModel):
    records: list[dict[str, Any]]
    experiment_id: str | None = None


class PredictResponse(BaseModel):
    experiment_id: str
    model_name: str
    predictions: list[Any]
