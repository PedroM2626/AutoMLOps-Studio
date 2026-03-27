import pandas as pd

from src.core.drift import DriftDetector


def test_drift_numeric_and_categorical_signals():
    reference = pd.DataFrame(
        {
            "num": [1, 2, 3, 4, 5],
            "cat": ["a", "a", "b", "b", "b"],
        }
    )
    current = pd.DataFrame(
        {
            "num": [10, 20, 30, 40, 50],
            "cat": ["a", "c", "c", "c", "c"],
        }
    )

    drifts = DriftDetector.detect_drift(reference, current)

    assert "num" in drifts
    assert drifts["num"]["feature_type"] == "numeric"
    assert "cat" in drifts
    assert drifts["cat"]["feature_type"] == "categorical"
    assert "p_value" in drifts["cat"]
