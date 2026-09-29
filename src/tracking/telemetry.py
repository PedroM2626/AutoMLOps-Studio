import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


class TelemetryStore:
    """SQLite-backed telemetry store for API inference events."""

    def __init__(self, db_path: str = "data_lake/monitoring/telemetry.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS inference_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp_utc TEXT NOT NULL,
                        model_version TEXT NOT NULL,
                        row_count INTEGER NOT NULL,
                        request_json TEXT NOT NULL,
                        predictions_json TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_inference_logs_ts ON inference_logs(timestamp_utc)"
                )

    def to_dataframe(self, since_days: int | None = None):
        """Flatten the logged requests into one row per payload row, with its prediction.

        Rows are stored as JSON blobs because the writer must stay cheap and
        concurrency-safe; the drift analysis on the Monitoring page needs a frame.
        """
        import pandas as pd

        query = "SELECT timestamp_utc, model_version, request_json, predictions_json FROM inference_logs"
        params: tuple = ()
        if since_days:
            cutoff = datetime.now(timezone.utc).timestamp() - since_days * 86400
            query += " WHERE timestamp_utc >= ?"
            params = (datetime.fromtimestamp(cutoff, timezone.utc).isoformat(),)

        with self._lock:
            with self._get_connection() as conn:
                records = conn.execute(query, params).fetchall()

        rows = []
        for timestamp_utc, model_version, request_json, predictions_json in records:
            try:
                payload = json.loads(request_json)
                predictions = json.loads(predictions_json)
            except json.JSONDecodeError:
                continue
            for index, feature_row in enumerate(payload):
                if not isinstance(feature_row, dict):
                    continue
                flat = dict(feature_row)
                flat["__timestamp"] = timestamp_utc
                flat["__model_version"] = model_version
                flat["__prediction"] = predictions[index] if index < len(predictions) else None
                rows.append(flat)
        return pd.DataFrame(rows)

    def log_inference(self, payload_rows: list[dict[str, Any]], predictions: list[Any], model_version: str) -> None:
        record = (
            datetime.now(timezone.utc).isoformat(),
            model_version,
            len(payload_rows),
            json.dumps(payload_rows, ensure_ascii=True),
            json.dumps(predictions, ensure_ascii=True),
        )

        with self._lock:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO inference_logs (
                        timestamp_utc,
                        model_version,
                        row_count,
                        request_json,
                        predictions_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    record,
                )
