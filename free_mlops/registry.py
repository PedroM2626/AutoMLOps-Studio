from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

from free_mlops.config import Settings
from free_mlops.db import get_experiment
from free_mlops.db import init_db
from free_mlops.db import insert_experiment
from free_mlops.service import load_model


def register_model(
    settings: Settings,
    experiment_id: str,
    new_version: str,
    description: str = "",
) -> dict[str, Any]:
    init_db(settings.db_path)

    record = get_experiment(settings.db_path, experiment_id)
    if record is None:
        raise ValueError(f"Experiment {experiment_id} not found")

    old_version = record.get("model_version", "v1.0.0")
    if new_version == old_version:
        raise ValueError("New version must be different from current version")

    old_artifact_dir = settings.artifacts_dir / experiment_id
    new_experiment_id = uuid4().hex
    new_artifact_dir = settings.artifacts_dir / new_experiment_id

    new_artifact_dir.mkdir(parents=True, exist_ok=True)

    for src in old_artifact_dir.glob("*"):
        if src.is_file():
            shutil.copy2(src, new_artifact_dir / src.name)

    new_metadata = record["model_metadata"].copy()
    new_metadata["version"] = new_version
    new_metadata["description"] = description
    new_metadata["registered_from_experiment_id"] = experiment_id
    new_metadata["registered_at"] = record["created_at"]

    metadata_path = new_artifact_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(new_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    new_record = record.copy()
    new_record["id"] = new_experiment_id
    new_record["model_version"] = new_version
    new_record["model_metadata"] = new_metadata
    new_record["model_path"] = str(new_artifact_dir / "model.pkl")

    insert_experiment(settings.db_path, new_record)

    return new_record


def list_registered_models(settings: Settings, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
    init_db(settings.db_path)

    from free_mlops.db import list_experiments

    records = list_experiments(settings.db_path, limit=limit, offset=offset)

    registered = [
        r
        for r in records
        if r.get("model_metadata", {}).get("registered_from_experiment_id") is not None
    ]

    return registered


def get_registered_model(settings: Settings, experiment_id: str) -> dict[str, Any] | None:
    record = get_experiment(settings.db_path, experiment_id)
    if record is None:
        return None

    if record.get("model_metadata", {}).get("registered_from_experiment_id") is None:
        return None

    return record


def download_model_package(settings: Settings, experiment_id: str, dest_dir: Path) -> Path:
    record = get_registered_model(settings, experiment_id)
    if record is None:
        raise ValueError(f"Registered model {experiment_id} not found")

    artifact_dir = Path(record["model_path"]).parent
    dest_dir.mkdir(parents=True, exist_ok=True)

    package_name = f"model_{experiment_id}_{record['model_version']}"
    package_path = dest_dir / package_name
    package_path.mkdir(exist_ok=True)

    for src in artifact_dir.glob("*"):
        if src.is_file():
            shutil.copy2(src, package_path / src.name)

    readme_path = package_path / "README.txt"
    readme_path.write_text(
        f"""Free MLOps Model Package

Experiment ID: {experiment_id}
Version: {record['model_version']}
Model Name: {record['best_model_name']}
Problem Type: {record['problem_type']}
Target Column: {record['target_column']}

Files:
- model.pkl: scikit-learn pipeline (preprocess + model)
- metadata.json: full metadata and version info
- leaderboard.json: AutoML leaderboard from training
- best_metrics.json: metrics on test set

To load:
    import joblib
    pipeline = joblib.load('model.pkl')
    preds = pipeline.predict(X)
""",
        encoding="utf-8",
    )

    return package_path


def load_registered_model(settings: Settings, experiment_id: str) -> Any:
    record = get_registered_model(settings, experiment_id)
    if record is None:
        raise ValueError(f"Registered model {experiment_id} not found")

    model_path = Path(record["model_path"])
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    return load_model(model_path)
