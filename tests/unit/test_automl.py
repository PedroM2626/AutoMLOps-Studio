from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from free_mlops.automl import run_automl


def test_run_automl_classification_returns_result() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "num1": rng.normal(size=120),
            "num2": rng.integers(0, 10, size=120),
            "cat": rng.choice(["A", "B", "C"], size=120),
        }
    )
    score = df["num1"] + df["num2"] * 0.1
    df["target"] = (score > score.median()).astype(int)

    X = df[["num1", "num2", "cat"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    result = run_automl(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        problem_type="classification",
        random_state=42,
    )

    assert isinstance(result.best_model_name, str)
    assert result.best_model_name
    assert isinstance(result.leaderboard, list)
    assert any(row.get("success") is True for row in result.leaderboard)
    assert isinstance(result.best_metrics, dict)
    assert "f1_weighted" in result.best_metrics


def test_run_automl_regression_returns_result() -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "num1": rng.normal(size=120),
            "cat": rng.choice(["A", "B"], size=120),
        }
    )
    df["target"] = df["num1"] * 3.0 + rng.normal(scale=0.1, size=120)

    X = df[["num1", "cat"]]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    result = run_automl(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        problem_type="regression",
        random_state=42,
    )

    assert isinstance(result.best_model_name, str)
    assert result.best_model_name
    assert isinstance(result.leaderboard, list)
    assert any(row.get("success") is True for row in result.leaderboard)
    assert isinstance(result.best_metrics, dict)
    assert "rmse" in result.best_metrics
