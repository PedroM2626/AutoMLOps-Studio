"""End-to-end coverage for the task types that shipped broken.

These tests call AutoMLTrainer.train() and evaluate() rather than only checking
that a model name exists in the catalog, which is the gap that let
dimensionality_reduction, survival_analysis, uplift_modeling and multi_task be
advertised while they could not finish training.
"""
import numpy as np
import pandas as pd
import pytest

from src.core.processor import AutoMLDataProcessor
from src.engines.classical import (
    AutoMLTrainer,
    ConstrainedLDA,
    QuantileBundleRegressor,
    SLearner,
    SurvivalTimeRegressor,
    TLearner,
    calculate_c_index,
    calculate_qini_score,
)

FAST = dict(n_trials=1, timeout=240, validation_strategy="holdout")


def _features(n=120, k=4, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0, 1, size=(n, k)), columns=[f"f{i}" for i in range(k)])


def _survival_frame(n=150, seed=7):
    df = _features(n=n, seed=seed)
    rng = np.random.default_rng(seed)
    df["duration"] = np.exp(0.9 * df["f0"] + rng.normal(0, 0.35, n))
    df["event"] = rng.integers(0, 2, n)
    return df


def _uplift_frame(n=180, seed=7):
    df = _features(n=n, seed=seed)
    rng = np.random.default_rng(seed)
    treatment = rng.integers(0, 2, n)
    persuadable = (df["f0"] > 0.3).astype(float)
    df["treatment"] = treatment
    df["outcome"] = ((rng.random(n) + treatment * persuadable) > 0.62).astype(float)
    return df


def _train(task, df, target, model_name, **overrides):
    X, y = AutoMLDataProcessor(target_column=target, task_type=task).fit_transform(df)
    trainer = AutoMLTrainer(task_type=task, preset="test", use_deep_learning=False)
    kwargs = dict(FAST)
    kwargs.update(overrides)
    trainer.train(X, y, selected_models=[model_name], experiment_name=f"ci_{task}", **kwargs)
    return trainer, X, y


@pytest.mark.parametrize("reducer", ["pca", "truncated_svd", "lda", "nca", "pls"])
def test_dimensionality_reduction_trains_and_projects(reducer):
    """Each reducer has to produce a champion and a projection, not just a catalog entry."""
    df = _features()
    df["label"] = np.random.default_rng(7).integers(0, 3, len(df))
    trainer, X, y = _train("dimensionality_reduction", df, "label", reducer)

    assert trainer.best_model is not None
    metrics, projected = trainer.evaluate(X[:60], y[:60])
    assert np.asarray(projected).shape[0] == 60
    assert "supervised_separability" in metrics
    if reducer in ("pca", "truncated_svd", "lda"):
        assert 0.0 < metrics["explained_variance"] <= 1.0 + 1e-9


def test_lda_clamps_components_to_the_class_bound():
    """LDA rejects n_components above n_classes - 1, which used to prune the whole trial."""
    rng = np.random.default_rng(2)
    X = rng.normal(size=(90, 6))
    y = rng.integers(0, 3, 90)          # at most 2 meaningful components
    lda = ConstrainedLDA(n_components=5, solver="svd")
    lda.fit(X, y)
    assert lda.n_components <= len(np.unique(y)) - 1
    assert lda.transform(X).shape[1] == lda.n_components


@pytest.mark.parametrize("model_name", ["survival_cox_ph", "survival_random_forest", "survival_gradient_boosting"])
def test_survival_analysis_trains_on_two_column_target(model_name):
    trainer, X, y = _train("survival_analysis", _survival_frame(), ["duration", "event"], model_name)

    assert isinstance(trainer.best_model, SurvivalTimeRegressor)
    metrics, preds = trainer.evaluate(X[:70], y[:70])
    assert 0.0 <= metrics["c_index"] <= 1.0
    # duration is driven by f0, so the fitted model must rank better than noise
    assert metrics["c_index"] > 0.55


def test_survival_rejects_a_single_column_target():
    df = _survival_frame()
    model = SurvivalTimeRegressor()
    with pytest.raises(ValueError, match="two-column|two target columns"):
        model.fit(df[["f0", "f1"]], df["duration"])

    # Through the AutoML path the same fault fails every trial, and train() must say why
    # instead of surfacing Optuna's generic "No trials are completed yet".
    X, y = AutoMLDataProcessor(target_column="duration", task_type="survival_analysis").fit_transform(df)
    trainer = AutoMLTrainer(task_type="survival_analysis", preset="test", use_deep_learning=False)
    with pytest.raises(ValueError, match="Every trial failed"):
        trainer.train(X, y, selected_models=["survival_cox_ph"], experiment_name="ci_bad_survival", **FAST)


@pytest.mark.parametrize("model_name", ["s_learner", "t_learner"])
def test_uplift_modeling_trains_and_scores(model_name):
    trainer, X, y = _train("uplift_modeling", _uplift_frame(), ["treatment", "outcome"], model_name)

    assert isinstance(trainer.best_model, (SLearner, TLearner))
    metrics, uplift = trainer.evaluate(X[:80], y[:80])
    assert np.isfinite(metrics["qini_score"])
    # The old implementation clipped every ranking to exactly 1.0.
    assert metrics["qini_score"] < 1.0


def test_qini_scores_rank_useful_targeting_above_random():
    rng = np.random.default_rng(5)
    n = 500
    treatment = rng.integers(0, 2, n)
    effect = (rng.random(n) < 0.30).astype(float)
    outcome = (rng.random(n) + treatment * effect > 0.55).astype(float)

    oracle = calculate_qini_score(treatment, outcome, effect)
    anti = calculate_qini_score(treatment, outcome, -effect)
    random_spread = [calculate_qini_score(treatment, outcome, rng.random(n)) for _ in range(15)]

    assert oracle > 0.05
    assert anti < 0.0
    assert abs(float(np.mean(random_spread))) < 0.15


def test_c_index_reacts_to_the_ranking_direction():
    rng = np.random.default_rng(9)
    n = 300
    duration = np.exp(rng.normal(0, 1, n))
    event = rng.integers(0, 2, n)

    assert calculate_c_index(event, duration, -duration) > 0.9    # oracle risk
    assert calculate_c_index(event, duration, duration) < 0.1     # inverted risk
    assert abs(calculate_c_index(event, duration, rng.random(n)) - 0.5) < 0.1


def test_multi_task_handles_a_multiclass_output_column():
    """hamming_loss does not accept multiclass-multioutput, so it is averaged per column."""
    df = _features()
    rng = np.random.default_rng(7)
    df["t1"] = rng.integers(0, 2, len(df))
    df["t2"] = rng.integers(0, 3, len(df))
    trainer, X, y = _train("multi_task", df, ["t1", "t2"], "random_forest")

    metrics, preds = trainer.evaluate(X[:60], y[:60])
    assert 0.0 <= metrics["hamming_loss"] <= 1.0
    assert metrics["accuracy"] > 0.4
    assert np.asarray(preds).shape == (60, 2)


@pytest.mark.parametrize("base", ["hist_gradient_boosting", "lightgbm"])
def test_quantile_band_reaches_its_nominal_coverage(base):
    """Uncalibrated quantile fits under-cover badly; the conformal step must fix that."""
    rng = np.random.default_rng(13)
    n = 1600
    X = pd.DataFrame(rng.normal(size=(n, 4)), columns=list("abcd"))
    y = 1.5 * X["a"] - X["b"] + rng.normal(0, 1.0, n)
    cut = int(n * 0.75)

    calibrated = QuantileBundleRegressor(base=base, n_estimators=200, random_state=0).fit(X[:cut], y[:cut])
    low, high = calibrated.predict_intervals(X[cut:])
    truth = np.asarray(y[cut:])
    coverage = float(np.mean((truth >= low) & (truth <= high)))

    raw = QuantileBundleRegressor(base=base, n_estimators=200, random_state=0, calibrate=False)
    raw.fit(X[:cut], y[:cut])
    raw_low, raw_high = raw.predict_intervals(X[cut:])
    raw_coverage = float(np.mean((truth >= raw_low) & (truth <= raw_high)))

    assert np.all(low <= high)
    assert 0.65 <= coverage <= 0.97
    assert coverage > raw_coverage


def test_quantile_models_train_inside_the_regression_task():
    rng = np.random.default_rng(7)
    df = _features(n=200)
    df["target"] = 2 * df["f0"] - df["f1"] + rng.normal(0, 1, len(df)) * (0.3 + 1.2 * np.abs(df["f0"]))
    trainer, X, y = _train("regression", df, "target", "quantile_hgb")

    metrics, preds = trainer.evaluate(X[:80], y[:80])
    assert isinstance(trainer.best_model, QuantileBundleRegressor)
    assert metrics["interval_mean_width"] > 0.0
    assert 0.0 <= metrics["interval_coverage"] <= 1.0
    assert np.asarray(preds).shape == (80,)


class _RecordingTrial:
    """Collects the parameter names a catalog lambda suggests, feeding back safe values."""

    def __init__(self):
        self.seen = set()

    def suggest_int(self, name, low, high, *args, **kwargs):
        self.seen.add(name)
        return low

    def suggest_float(self, name, low, high, *args, **kwargs):
        self.seen.add(name)
        return (low + high) / 2.0

    def suggest_categorical(self, name, choices, *args, **kwargs):
        self.seen.add(name)
        return choices[0]


@pytest.mark.parametrize("task,model_name", [
    ("dimensionality_reduction", "pca"),
    ("dimensionality_reduction", "lda"),
    ("dimensionality_reduction", "nca"),
    ("survival_analysis", "survival_cox_ph"),
    ("survival_analysis", "survival_random_forest"),
    ("uplift_modeling", "s_learner"),
    ("uplift_modeling", "t_learner"),
    ("regression", "quantile_hgb"),
    ("regression", "quantile_gbm"),
    ("regression", "hist_gradient_boosting"),
    ("classification", "hist_gradient_boosting"),
])
def test_manual_hyperparameter_names_match_the_search_space(task, model_name):
    """A schema key the catalog never suggests would make the wizard's manual mode a no-op."""
    trainer = AutoMLTrainer(task_type=task, preset="test", use_deep_learning=False)
    trial = _RecordingTrial()
    assert trainer._get_models(trial=trial, name=model_name, random_state=42) is not None
    schema = set(trainer.get_model_params_schema(model_name))
    assert schema, f"{model_name} exposes no tunable parameters"
    assert schema <= trial.seen
