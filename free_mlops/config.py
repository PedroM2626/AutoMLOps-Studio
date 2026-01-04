from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    artifacts_dir: Path
    db_path: Path
    default_test_size: float
    random_state: int
    api_host: str
    api_port: int


def _get_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Invalid float for {name}: {value}") from exc


def _get_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid int for {name}: {value}") from exc


def _get_env_path(name: str, default: str) -> Path:
    value = os.environ.get(name)
    if value is None or value == "":
        value = default
    return Path(value).expanduser().resolve() if value.startswith("~") else Path(value)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    load_dotenv(override=False)

    data_dir = _get_env_path("FREE_MLOPS_DATA_DIR", "./data")
    artifacts_dir = _get_env_path("FREE_MLOPS_ARTIFACTS_DIR", "./artifacts")
    db_path = _get_env_path("FREE_MLOPS_DB_PATH", "./free_mlops.db")

    default_test_size = _get_env_float("FREE_MLOPS_DEFAULT_TEST_SIZE", 0.2)
    if not 0.05 <= default_test_size <= 0.95:
        raise ValueError("FREE_MLOPS_DEFAULT_TEST_SIZE must be between 0.05 and 0.95")

    random_state = _get_env_int("FREE_MLOPS_RANDOM_STATE", 42)

    api_host = os.environ.get("FREE_MLOPS_API_HOST", "127.0.0.1")
    api_port = _get_env_int("FREE_MLOPS_API_PORT", 8000)

    data_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    return Settings(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        db_path=db_path,
        default_test_size=default_test_size,
        random_state=random_state,
        api_host=api_host,
        api_port=api_port,
    )
