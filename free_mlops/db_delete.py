from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from free_mlops.db import get_experiment
from free_mlops.db import init_db


def delete_experiment(db_path: Path, experiment_id: str) -> None:
    init_db(db_path)

    record = get_experiment(db_path, experiment_id)
    if record is None:
        raise ValueError(f"Experiment {experiment_id} not found")

    import sqlite3

    with sqlite3.connect(db_path) as conn:
        conn.execute("DELETE FROM experiments WHERE id = ?", (experiment_id,))
        conn.commit()

    artifact_dir = Path(record["model_path"]).parent
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)


def list_experiment_ids(db_path: Path) -> list[str]:
    init_db(db_path)

    import sqlite3

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT id FROM experiments ORDER BY created_at DESC").fetchall()
        return [row[0] for row in rows]
