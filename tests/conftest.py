from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from free_mlops.config import Settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    artifacts_dir = tmp_path / "artifacts"
    db_path = tmp_path / "free_mlops.db"

    data_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        db_path=db_path,
        default_test_size=0.2,
        random_state=42,
        api_host="127.0.0.1",
        api_port=8000,
    )


def _make_classification_df(n_rows: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(
        {
            "num1": rng.normal(size=n_rows),
            "num2": rng.integers(0, 10, size=n_rows),
            "cat": rng.choice(["A", "B", "C"], size=n_rows),
        }
    )

    score = df["num1"] + df["num2"] * 0.1
    df["target"] = (score > score.median()).astype(int)

    df.loc[0, "num1"] = np.nan
    df.loc[1, "cat"] = None

    return df


@pytest.fixture
def classification_csv(tmp_path: Path) -> Path:
    df = _make_classification_df()
    path = tmp_path / "classification.csv"
    df.to_csv(path, index=False)
    return path


def _make_regression_df(n_rows: int = 120, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(
        {
            "num1": rng.normal(size=n_rows),
            "cat": rng.choice(["A", "B"], size=n_rows),
        }
    )

    df.loc[0, "cat"] = None
    df["target"] = df["num1"] * 3.0 + rng.normal(scale=0.1, size=n_rows)

    return df


@pytest.fixture
def regression_csv(tmp_path: Path) -> Path:
    df = _make_regression_df()
    path = tmp_path / "regression.csv"
    df.to_csv(path, index=False)
    return path
