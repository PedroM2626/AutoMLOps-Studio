from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from free_mlops.automl import ProblemType
from free_mlops.automl import run_automl
from free_mlops.finetune import run_finetune
from free_mlops.config import Settings
from free_mlops.db import get_experiment
from free_mlops.db import get_latest_experiment
from free_mlops.db import insert_experiment
from free_mlops.db import list_experiments


def save_uploaded_bytes(file_bytes: bytes, original_filename: str, settings: Settings) -> Path:
    settings.data_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(original_filename).name
    file_id = uuid4().hex
    out_path = settings.data_dir / f"{file_id}_{safe_name}"
    out_path.write_bytes(file_bytes)
    return out_path


def load_csv(dataset_path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(dataset_path)
    except Exception:
        return pd.read_csv(dataset_path, sep=None, engine="python")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_problem_setup(df: pd.DataFrame, target_column: str, problem_type: ProblemType) -> None:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataset")

    if df.shape[0] < 10:
        raise ValueError("Dataset must have at least 10 rows")

    y = df[target_column]
    y_non_null = y.dropna()

    if y_non_null.empty:
        raise ValueError("Target column has only null values")

    if problem_type == "classification":
        n_unique = int(y_non_null.nunique())
        if n_unique <= 1:
            raise ValueError("Classification target must have at least 2 classes")

    if problem_type == "regression":
        y_numeric = pd.to_numeric(y_non_null, errors="coerce")
        if y_numeric.isna().any():
            raise ValueError("Regression requires a numeric target column")


def run_experiment(
    dataset_path: Path,
    target_column: str,
    problem_type: ProblemType,
    settings: Settings,
) -> dict[str, Any]:
    df = load_csv(dataset_path)

    validate_problem_setup(df, target_column=target_column, problem_type=problem_type)

    df = df.dropna(subset=[target_column]).reset_index(drop=True)

    feature_columns = [c for c in df.columns if c != target_column]
    if not feature_columns:
        raise ValueError("Dataset must have at least 1 feature column")

    X = df[feature_columns]
    y = df[target_column]

    if problem_type == "regression":
        y = pd.to_numeric(y, errors="raise")

    stratify = None
    if problem_type == "classification":
        try:
            counts = y.value_counts(dropna=False)
            if (counts >= 2).all() and int(y.nunique()) >= 2:
                stratify = y
        except Exception:
            stratify = None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=settings.default_test_size,
            random_state=settings.random_state,
            stratify=stratify,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=settings.default_test_size,
            random_state=settings.random_state,
            stratify=None,
        )

    automl_result = run_automl(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        problem_type=problem_type,
        random_state=settings.random_state,
    )

    automl_result.best_pipeline.fit(X, y)

    experiment_id = uuid4().hex
    created_at = datetime.now(timezone.utc).isoformat()

    artifact_dir = settings.artifacts_dir / experiment_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "model.pkl"
    joblib.dump(automl_result.best_pipeline, model_path)

    leaderboard_path = artifact_dir / "leaderboard.json"
    leaderboard_path.write_text(
        json.dumps(automl_result.leaderboard, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    best_metrics_path = artifact_dir / "best_metrics.json"
    best_metrics_path.write_text(
        json.dumps(automl_result.best_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    model_version = "v1.0.0"
    model_metadata = {
        "version": model_version,
        "framework": "scikit-learn",
        "problem_type": problem_type,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "model_name": automl_result.best_model_name,
        "created_at": created_at,
        "dataset_path": str(dataset_path),
        "test_size": settings.default_test_size,
        "random_state": settings.random_state,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "best_metrics": automl_result.best_metrics,
        "leaderboard": automl_result.leaderboard,
    }

    metadata_path = artifact_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(model_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    record: dict[str, Any] = {
        "id": experiment_id,
        "created_at": created_at,
        "dataset_path": str(dataset_path),
        "target_column": str(target_column),
        "problem_type": str(problem_type),
        "test_size": float(settings.default_test_size),
        "random_state": int(settings.random_state),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "feature_columns": [str(c) for c in feature_columns],
        "leaderboard": automl_result.leaderboard,
        "best_model_name": automl_result.best_model_name,
        "best_metrics": automl_result.best_metrics,
        "model_path": str(model_path),
        "model_version": model_version,
        "model_metadata": model_metadata,
    }

    insert_experiment(settings.db_path, record)

    return record


def get_experiment_record(settings: Settings, experiment_id: str) -> dict[str, Any] | None:
    return get_experiment(settings.db_path, experiment_id)


def list_experiment_records(settings: Settings, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    return list_experiments(settings.db_path, limit=limit, offset=offset)


def get_latest_experiment_record(settings: Settings) -> dict[str, Any] | None:
    return get_latest_experiment(settings.db_path)


def load_model(model_path: Path) -> Any:
    return joblib.load(model_path)


def align_features(records: list[dict[str, Any]], feature_columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame(records)

    for col in feature_columns:
        if col not in df.columns:
            df[col] = np.nan

    return df[feature_columns]


def hash_uploaded_file(file_bytes: bytes) -> str:
    return _sha256_bytes(file_bytes)
