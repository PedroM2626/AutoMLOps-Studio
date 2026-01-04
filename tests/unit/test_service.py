from __future__ import annotations

import pandas as pd
import pytest

from free_mlops.service import align_features
from free_mlops.service import validate_problem_setup


def test_align_features_adds_missing_columns() -> None:
    records = [{"a": 1}, {"a": 2, "b": "x"}]
    df = align_features(records, feature_columns=["a", "b", "c"])

    assert list(df.columns) == ["a", "b", "c"]
    assert df["c"].isna().all()


def test_validate_problem_setup_errors_on_missing_target() -> None:
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]})

    with pytest.raises(ValueError):
        validate_problem_setup(df, target_column="missing", problem_type="classification")


def test_validate_problem_setup_classification_requires_two_classes() -> None:
    df = pd.DataFrame({"x": list(range(10)), "y": [1] * 10})

    with pytest.raises(ValueError):
        validate_problem_setup(df, target_column="y", problem_type="classification")


def test_validate_problem_setup_regression_requires_numeric_target() -> None:
    df = pd.DataFrame({"x": list(range(20)), "y": ["a"] * 20})

    with pytest.raises(ValueError):
        validate_problem_setup(df, target_column="y", problem_type="regression")
