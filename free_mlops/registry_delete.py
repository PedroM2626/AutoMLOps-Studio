from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from free_mlops.db import get_experiment
from free_mlops.db import init_db


def delete_registered_model(db_path: Path, experiment_id: str) -> None:
    init_db(db_path)

    record = get_experiment(db_path, experiment_id)
    if record is None:
        raise ValueError(f"Registered model {experiment_id} not found")

    if record.get("model_metadata", {}).get("registered_from_experiment_id") is None:
        raise ValueError(f"Experiment {experiment_id} is not a registered model")

    import sqlite3

    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
        conn.commit()

    artifact_dir = Path(record["model_path"]).parent
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
