import numpy as np
import pandas as pd
import pytest

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


def test_identical_frames_report_no_drift():
    frame = pd.DataFrame({"num": np.arange(200), "cat": (["a", "b", "c"] * 67)[:200]})

    drifts = DriftDetector.detect_drift(frame, frame.copy())

    assert set(drifts) == {"num", "cat"}
    assert all(report["drift_detected"] is False for report in drifts.values())


def test_shifted_numeric_column_is_flagged():
    rng = np.random.default_rng(7)
    reference = pd.DataFrame({"num": rng.normal(0.0, 1.0, 400)})
    current = pd.DataFrame({"num": rng.normal(6.0, 1.0, 400)})

    drifts = DriftDetector.detect_drift(reference, current)

    assert drifts["num"]["drift_detected"] is True
    assert drifts["num"]["statistic"] == pytest.approx(1.0, abs=0.05)
    assert drifts["num"]["current_mean"] - drifts["num"]["reference_mean"] > 5.0


def test_category_missing_from_one_frame_does_not_raise():
    # A level that only exists in production is the common case for new enum values.
    # scipy's chi2_contingency has to survive the resulting one-sided contingency table.
    reference = pd.DataFrame({"cat": ["a"] * 30 + ["b"] * 30})
    current = pd.DataFrame({"cat": ["a"] * 25 + ["b"] * 25 + ["new_level"] * 25})

    drifts = DriftDetector.detect_drift(reference, current)

    assert drifts["cat"]["categories_compared"] == 3
    assert drifts["cat"]["drift_detected"] is True


def test_all_nan_numeric_column_reports_a_usable_p_value():
    # ks_2samp returns a NaN p-value for an empty sample, which the UI would render as a
    # blank statistic next to a silent "Stable" verdict.
    reference = pd.DataFrame({"num": range(40), "blank": [np.nan] * 40})
    current = pd.DataFrame({"num": range(100, 140), "blank": [np.nan] * 40})

    drifts = DriftDetector.detect_drift(reference, current)

    assert np.isfinite(drifts["blank"]["p_value"])
    assert drifts["blank"]["p_value"] == 1.0
    assert drifts["blank"]["reference_mean"] is None
    assert drifts["blank"]["drift_detected"] is False
    assert drifts["num"]["drift_detected"] is True


def test_mixed_type_object_column_is_comparable():
    # sorted() cannot compare the int and str values a CSV can drop into one object column.
    reference = pd.DataFrame({"mixed": pd.Series([1, "2", 1, "2"], dtype=object)})
    current = pd.DataFrame({"mixed": pd.Series(["2", "2", "2", 1], dtype=object)})

    drifts = DriftDetector.detect_drift(reference, current)

    assert drifts["mixed"]["feature_type"] == "categorical"
    assert drifts["mixed"]["categories_compared"] == 2


def test_only_shared_columns_are_tested_in_a_stable_order():
    reference = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 1, 2, 2], "gone": [1, 2, 3, 4]})
    current = pd.DataFrame({"b": [1, 2, 1, 2], "a": [4, 3, 2, 1], "added": [1, 1, 1, 1]})

    first = DriftDetector.detect_drift(reference, current)
    second = DriftDetector.detect_drift(reference, current)

    assert list(first) == ["a", "b"]
    assert list(first) == list(second)


def test_threshold_controls_the_verdict():
    rng = np.random.default_rng(11)
    reference = pd.DataFrame({"num": rng.normal(0.0, 1.0, 300)})
    current = pd.DataFrame({"num": rng.normal(0.06, 1.0, 300)})

    strict = DriftDetector.detect_drift(reference, current, threshold=0.999)
    relaxed = DriftDetector.detect_drift(reference, current, threshold=1e-9)

    assert strict["num"]["drift_detected"] is True
    assert relaxed["num"]["drift_detected"] is False


def test_free_text_columns_are_declared_untested_not_stable():
    # A near-unique column gives chi-square one observation per cell, which returns
    # p=1.0. Reporting that as "Stable" would be a claim the test cannot support.
    levels = [f"tweet number {i}" for i in range(400)]
    reference = pd.DataFrame({"text": levels, "sentiment": ["pos"] * 200 + ["neg"] * 200})
    current = pd.DataFrame({"text": levels, "sentiment": ["pos"] * 200 + ["neg"] * 200})

    drifts = DriftDetector.detect_drift(reference, current)

    assert drifts["text"]["test"] == "skipped_high_cardinality"
    assert drifts["text"]["drift_detected"] is False
    assert "400 distinct levels" in drifts["text"]["skipped_reason"]
    assert drifts["sentiment"]["test"] == "chi2_contingency"
    assert drifts["sentiment"]["drift_detected"] is False
