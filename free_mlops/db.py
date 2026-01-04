from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                dataset_path TEXT NOT NULL,
                target_column TEXT NOT NULL,
                problem_type TEXT NOT NULL,
                test_size REAL NOT NULL,
                random_state INTEGER NOT NULL,
                n_rows INTEGER NOT NULL,
                n_cols INTEGER NOT NULL,
                feature_columns_json TEXT NOT NULL,
                leaderboard_json TEXT NOT NULL,
                best_model_name TEXT NOT NULL,
                best_metrics_json TEXT NOT NULL,
                model_path TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_experiments_created_at ON experiments(created_at);"
        )


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def insert_experiment(db_path: Path, record: dict[str, Any]) -> None:
    init_db(db_path)

    feature_columns_json = json.dumps(record["feature_columns"], ensure_ascii=False)
    leaderboard_json = json.dumps(record["leaderboard"], ensure_ascii=False)
    best_metrics_json = json.dumps(record["best_metrics"], ensure_ascii=False)

    with _connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO experiments (
                id,
                created_at,
                dataset_path,
                target_column,
                problem_type,
                test_size,
                random_state,
                n_rows,
                n_cols,
                feature_columns_json,
                leaderboard_json,
                best_model_name,
                best_metrics_json,
                model_path
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["id"],
                record["created_at"],
                record["dataset_path"],
                record["target_column"],
                record["problem_type"],
                record["test_size"],
                record["random_state"],
                record["n_rows"],
                record["n_cols"],
                feature_columns_json,
                leaderboard_json,
                record["best_model_name"],
                best_metrics_json,
                record["model_path"],
            ),
        )
        conn.commit()


def _row_to_record(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "dataset_path": row["dataset_path"],
        "target_column": row["target_column"],
        "problem_type": row["problem_type"],
        "test_size": float(row["test_size"]),
        "random_state": int(row["random_state"]),
        "n_rows": int(row["n_rows"]),
        "n_cols": int(row["n_cols"]),
        "feature_columns": json.loads(row["feature_columns_json"]),
        "leaderboard": json.loads(row["leaderboard_json"]),
        "best_model_name": row["best_model_name"],
        "best_metrics": json.loads(row["best_metrics_json"]),
        "model_path": row["model_path"],
    }


def list_experiments(db_path: Path, limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    init_db(db_path)

    with _connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM experiments
            ORDER BY created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()

    return [_row_to_record(r) for r in rows]


def get_experiment(db_path: Path, experiment_id: str) -> dict[str, Any] | None:
    init_db(db_path)

    with _connect(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM experiments WHERE id = ?",
            (experiment_id,),
        ).fetchone()

    if row is None:
        return None

    return _row_to_record(row)


def get_latest_experiment(db_path: Path) -> dict[str, Any] | None:
    init_db(db_path)

    with _connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT *
            FROM experiments
            ORDER BY created_at DESC
            LIMIT 1
            """
        ).fetchone()

    if row is None:
        return None

    return _row_to_record(row)
