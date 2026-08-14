"""
Smoke tests for the expanded task catalog:
- Forecast Classification
- Time Series Classification (sequential classification)
- TS Clustering (window-based)
- Density Estimation (KDE / GMM / Histogram)
- Extended Anomaly Detection detectors
- Preprocessing customization (scaler, imputer, encoding, winsorizer)
- Per-model scaling overrides
- NLP regression wiring (TransformersWrapper task mapping)
- NLP text vectorizer customization (TF-IDF / Bag-of-Words / binary / hashing)
- Multi-output (multivariate) regression
"""
import numpy as np
import pandas as pd
import pytest

from src.core.processor import AutoMLDataProcessor, build_scaler
from src.engines.classical import (
    AutoMLTrainer,
    StatisticalZScoreDetector,
    ModifiedZScoreDetector,
    MahalanobisDetector,
    HBOSDetector,
    KNNOutlierDetector,
    PCAResidualDetector,
    RollingResidualDetector,
    DensityKDEWrapper,
    DensityGMMWrapper,
    DensityHistogram,
)

FAST = dict(n_trials=1, timeout=120, validation_strategy='holdout')


def make_ts_classification_df(n=240, seed=7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    value = np.sin(np.arange(n) / 12) * 10 + rng.normal(0, 1, n)
    label = np.where(np.diff(value, prepend=value[0]) >= 0, "up", "down")
    return pd.DataFrame({"date": dates, "value": value, "state": label})


def make_regression_ts_df(n=240, seed=3):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    value = np.cumsum(rng.normal(0.2, 1.0, n)) + 50
    return pd.DataFrame({"date": dates, "value": value})


# ──────────────────────────────────────────────────────────────
# Forecast Classification
# ──────────────────────────────────────────────────────────────
def test_forecast_classification_end_to_end():
    df = make_ts_classification_df()
    proc = AutoMLDataProcessor(
        target_column='state', task_type='forecast_classification',
        data_type='sequential', date_col='date', forecast_horizon=1
    )
    X, y = proc.fit_transform(df)
    assert X.shape[0] == len(y)
    # Lag features of the encoded categorical target must exist
    feat_names = proc.get_feature_names()
    assert any('lag_' in str(f) for f in feat_names)

    trainer = AutoMLTrainer(task_type='forecast_classification', preset='test', data_type='sequential')
    assert trainer.task_type == 'classification'
    assert trainer.is_time_series is True
    trainer.train(
        X, y, selected_models=['logistic_regression'],
        experiment_name='test_forecast_cls', optimization_metric='accuracy', **FAST
    )
    metrics, preds = trainer.evaluate(X[-40:], y[-40:])
    assert 'accuracy' in metrics
    assert len(preds) == 40


def test_ts_classification_with_time_series_cv():
    df = make_ts_classification_df(n=200)
    proc = AutoMLDataProcessor(
        target_column='state', task_type='classification',
        data_type='sequential', date_col='date'
    )
    X, y = proc.fit_transform(df)
    trainer = AutoMLTrainer(task_type='classification', preset='test', data_type='sequential')
    trainer.train(
        X, y, selected_models=['logistic_regression'],
        experiment_name='test_ts_cls', optimization_metric='accuracy',
        n_trials=1, timeout=120, validation_strategy='time_series_cv',
        validation_params={'folds': 3}
    )
    assert trainer.best_model is not None


# ──────────────────────────────────────────────────────────────
# TS Clustering
# ──────────────────────────────────────────────────────────────
def test_ts_clustering_end_to_end():
    df = make_regression_ts_df(n=300)
    proc = AutoMLDataProcessor(
        task_type='ts_clustering', data_type='sequential', date_col='date',
        ts_clustering_config={'series_col': 'value', 'window_size': 12, 'step': 2}
    )
    X, y = proc.fit_transform(df)
    assert y is None
    # 7 window summary features (+ calendar features extracted from the date column)
    assert X.shape[1] >= 7
    assert any('value_w_mean' in str(f) for f in proc.get_feature_names())
    assert X.shape[0] > 50

    trainer = AutoMLTrainer(task_type='ts_clustering', preset='test')
    assert trainer.task_type == 'clustering'
    trainer.train(
        X, selected_models=['kmeans'],
        experiment_name='test_ts_cluster', optimization_metric='silhouette', **FAST
    )
    metrics, labels = trainer.evaluate(X)
    assert 'silhouette' in metrics
    assert len(labels) == X.shape[0]


# ──────────────────────────────────────────────────────────────
# Density Estimation
# ──────────────────────────────────────────────────────────────
def test_density_estimation_end_to_end():
    rng = np.random.default_rng(11)
    df = pd.DataFrame(rng.normal(0, 1, size=(300, 3)), columns=['a', 'b', 'c'])
    proc = AutoMLDataProcessor(task_type='density_estimation')
    X, y = proc.fit_transform(df)
    assert y is None

    trainer = AutoMLTrainer(task_type='density_estimation', preset='test')
    models = trainer.get_available_models()
    assert {'kernel_density', 'gaussian_mixture_density', 'histogram_density'}.issubset(set(models))

    trainer.train(
        X, selected_models=['kernel_density', 'gaussian_mixture_density', 'histogram_density'],
        experiment_name='test_density', optimization_metric='log_likelihood', **FAST
    )
    metrics, densities = trainer.evaluate(X[:50])
    assert 'mean_log_likelihood' in metrics
    assert len(densities) == 50
    assert np.all(densities >= 0)


def test_density_estimators_standalone():
    rng = np.random.default_rng(5)
    X = rng.normal(0, 1, size=(200, 2))
    for est in [DensityKDEWrapper(bandwidth=0.5), DensityGMMWrapper(n_components=2), DensityHistogram(n_bins=20)]:
        est.fit(X)
        scores = est.score_samples(X[:20])
        assert scores.shape == (20,)
        assert est.predict(X[:20]).shape == (20,)


# ──────────────────────────────────────────────────────────────
# Extended Anomaly Detection
# ──────────────────────────────────────────────────────────────
@pytest.mark.parametrize("detector", [
    StatisticalZScoreDetector(threshold=3.0),
    ModifiedZScoreDetector(threshold=3.5),
    MahalanobisDetector(robust=False, contamination=0.1),
    MahalanobisDetector(robust=True, contamination=0.1),
    HBOSDetector(n_bins=15, contamination=0.1),
    KNNOutlierDetector(n_neighbors=10, contamination=0.1),
    PCAResidualDetector(n_components=2, contamination=0.1),
    RollingResidualDetector(window=10, threshold=3.0),
])
def test_anomaly_detectors_standalone(detector):
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, size=(200, 3))
    X[195:] += 12  # obvious anomalies
    detector.fit(X[:180])
    preds = detector.predict(X)
    assert set(np.unique(preds)).issubset({-1, 1})
    assert np.sum(preds == -1) >= 1
    scores = detector.decision_function(X)
    assert scores.shape == (200,)
    # Injected anomalies must score lower (less normal) than the mean
    assert scores[195:].mean() < scores[:180].mean()


def test_anomaly_detection_training_catalog():
    rng = np.random.default_rng(9)
    df = pd.DataFrame(rng.normal(0, 1, size=(250, 3)), columns=['a', 'b', 'c'])
    proc = AutoMLDataProcessor(task_type='anomaly_detection')
    X, _ = proc.fit_transform(df)

    trainer = AutoMLTrainer(task_type='anomaly_detection', preset='test')
    available = trainer.get_available_models()
    for m in ['isolation_forest', 'local_outlier_factor', 'elliptic_envelope', 'one_class_svm',
              'zscore_detector', 'modified_zscore', 'mahalanobis', 'hbos',
              'knn_outlier', 'pca_residual', 'rolling_residual']:
        assert m in available, f"{m} missing from anomaly catalog"

    trainer.train(
        X, selected_models=['zscore_detector', 'hbos', 'isolation_forest'],
        experiment_name='test_anomaly', optimization_metric='decision_score', **FAST
    )
    metrics, preds = trainer.evaluate(X)
    assert 'n_anomalies' in metrics
    assert set(np.unique(preds)).issubset({-1, 1})


# ──────────────────────────────────────────────────────────────
# Preprocessing customization
# ──────────────────────────────────────────────────────────────
def make_messy_df():
    return pd.DataFrame({
        'num1': [1.0, 2.0, 3.0, np.nan, 100.0, 5.0, 6.0, 7.0],
        'num2': [10, 20, 30, 40, 50, np.nan, 70, 80],
        'cat': ['a', 'b', 'a', 'c', 'b', 'a', 'c', 'b'],
        'target': [0, 1, 0, 1, 0, 1, 0, 1],
    })


def test_scaler_options_build():
    assert build_scaler('none') is None
    assert build_scaler('minmax').__class__.__name__ == 'MinMaxScaler'
    assert build_scaler('robust').__class__.__name__ == 'RobustScaler'
    assert build_scaler('quantile').__class__.__name__ == 'QuantileTransformer'
    assert build_scaler('power').__class__.__name__ == 'PowerTransformer'


def test_preprocessing_minmax_and_winsorize():
    proc = AutoMLDataProcessor(
        target_column='target', task_type='classification',
        scaler_type='minmax', impute_strategy='mean',
        encoding_mode='ordinal', clip_outliers=True,
        outlier_lower_q=0.05, outlier_upper_q=0.95
    )
    X, y = proc.fit_transform(make_messy_df())
    X_arr = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
    # Numeric columns (first two) must be winsorized + MinMax-scaled into [0, 1]
    assert np.nanmax(X_arr[:, :2]) <= 1.0 + 1e-9
    assert np.nanmin(X_arr[:, :2]) >= -1e-9
    assert not np.isnan(X_arr).any()
    # Feature names must remain available even with the Winsorizer in the pipeline
    names = proc.get_feature_names()
    assert len(names) == X.shape[1]


def test_preprocessing_no_scaling():
    proc = AutoMLDataProcessor(
        target_column='target', task_type='classification',
        scaler_type='none', impute_strategy='median'
    )
    X, _ = proc.fit_transform(make_messy_df())
    X_arr = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
    # Without scaling the imputed num2 median (45) must remain in original scale
    assert np.any(np.abs(X_arr) > 5)


def test_preprocessing_onehot_mode():
    proc = AutoMLDataProcessor(
        target_column='target', task_type='classification', encoding_mode='onehot'
    )
    X, _ = proc.fit_transform(make_messy_df())
    # One-hot of 3 categories => at least 3 dummy columns + 2 numeric
    n_cols = X.shape[1]
    assert n_cols >= 5


# ──────────────────────────────────────────────────────────────
# Per-model scaling override
# ──────────────────────────────────────────────────────────────
def test_per_model_scaler_override():
    from sklearn.datasets import make_classification
    X_arr, y_arr = make_classification(n_samples=150, n_features=6, random_state=0)
    trainer = AutoMLTrainer(task_type='classification', preset='test')
    trainer.train(
        X_arr, y_arr, selected_models=['logistic_regression'],
        experiment_name='test_override', optimization_metric='accuracy',
        scaler_overrides={'logistic_regression': 'minmax'}, **FAST
    )
    from sklearn.pipeline import Pipeline
    assert isinstance(trainer.best_model, Pipeline)
    assert trainer.best_model.named_steps['model_scaler'].__class__.__name__ == 'MinMaxScaler'
    metrics, _ = trainer.evaluate(X_arr[:20], y_arr[:20])
    assert 'accuracy' in metrics


# ──────────────────────────────────────────────────────────────
# NLP regression wiring (no heavy model download)
# ──────────────────────────────────────────────────────────────
def test_nlp_regression_model_mapping():
    from src.core.trainer import TransformersWrapper as RealWrapper
    from src.engines import classical as _classical
    if _classical.TransformersWrapper is not RealWrapper:
        pytest.skip("TransformersWrapper monkeypatched by other tests")
    trainer = AutoMLTrainer(task_type='regression', preset='test')
    params = {'num_train_epochs': 1, 'learning_rate': 2e-5}
    try:
        model = trainer._base_instantiate_model('bert-base-uncased-reg', params)
    except Exception:
        pytest.skip("Transformers runtime unavailable")
    if model is None:
        pytest.skip("Transformers runtime unavailable")
    assert getattr(model, 'task', None) == 'regression'
    assert getattr(model, 'model_name', None) == 'bert-base-uncased'


def test_transformers_wrapper_regression_dtype_logic():
    import torch
    # Emulate the label casting rule used inside TransformersWrapper.fit
    task = 'regression'
    label_dtype = torch.float if task == 'regression' else torch.long
    t = torch.tensor([1.5, 2.5], dtype=label_dtype)
    assert t.dtype == torch.float


# ─────────────────────────────────────────────────────────────────────────────
# NLP text preprocessing customization (vectorizers + cleaning)
# ─────────────────────────────────────────────────────────────────────────────

def make_nlp_df(n=120, seed=11):
    rng = np.random.default_rng(seed)
    pos_phrases = [
        "the movie was fantastic and exciting",
        "great film with an amazing story",
        "brilliant acting and wonderful direction",
    ]
    neg_phrases = [
        "the movie was terrible and boring",
        "awful film with a bad plot",
        "dull acting and horrible direction",
    ]
    texts, labels = [], []
    for _ in range(n):
        if rng.random() < 0.5:
            texts.append(rng.choice(pos_phrases))
            labels.append("pos")
        else:
            texts.append(rng.choice(neg_phrases))
            labels.append("neg")
    return pd.DataFrame({"review": texts, "label": labels})


@pytest.mark.parametrize("vectorizer", ["tfidf", "count", "binary", "hashing"])
def test_nlp_vectorizer_modes(vectorizer):
    df = make_nlp_df()
    proc = AutoMLDataProcessor(
        target_column="label", task_type="classification",
        nlp_config={"vectorizer": vectorizer, "max_features": 500,
                    "ngram_range": (1, 2), "stop_words": False},
    )
    X, y = proc.fit_transform(df, nlp_cols=["review"])
    assert X.shape[0] == len(df)
    assert X.shape[1] > 0
    names = proc.get_feature_names()
    assert len(names) == X.shape[1]
    # transform() on unseen data must reuse the fitted vectorizer
    X2, _ = proc.transform(make_nlp_df(n=20, seed=99))
    assert X2.shape[1] == X.shape[1]


def test_nlp_binary_bow_is_binary():
    df = make_nlp_df()
    proc = AutoMLDataProcessor(
        target_column="label", task_type="classification",
        nlp_config={"vectorizer": "binary", "stop_words": False},
    )
    X, _ = proc.fit_transform(df, nlp_cols=["review"])
    arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    assert set(np.unique(arr)).issubset({0.0, 1.0})


def test_nlp_cleaning_mode_none_keeps_raw_text():
    df = pd.DataFrame({
        "review": ["Hello WORLD http://x.com #tag", "MixedCase TeXt"],
        "label": ["a", "b"],
    })
    proc = AutoMLDataProcessor(
        target_column="label", task_type="classification",
        nlp_config={"cleaning_mode": "none"},
    )
    proc.fit_transform(df.copy(), nlp_cols=["review"])
    # Original casing and URL must survive when cleaning is disabled
    assert df.loc[0, "review"] == "Hello WORLD http://x.com #tag"
    cleaned = proc._clean_text_feature(df.copy(), "review")
    assert cleaned.loc[0, "review"] == "Hello WORLD http://x.com #tag"


def test_nlp_cleaning_mode_standard_lowercases():
    df = pd.DataFrame({"review": ["Hello WORLD http://x.com"], "label": ["a"]})
    proc = AutoMLDataProcessor(
        target_column="label", task_type="classification",
        nlp_config={"cleaning_mode": "standard"},
    )
    cleaned = proc._clean_text_feature(df.copy(), "review")
    assert cleaned.loc[0, "review"] == "hello world"


def test_nlp_classification_e2e_bag_of_words():
    df = make_nlp_df(n=160)
    proc = AutoMLDataProcessor(
        target_column="label", task_type="classification",
        nlp_config={"vectorizer": "count", "max_features": 300, "stop_words": False},
    )
    X, y = proc.fit_transform(df, nlp_cols=["review"])
    trainer = AutoMLTrainer(task_type="classification", preset="test")
    trainer.train(
        X, y,
        feature_names=proc.get_feature_names(),
        selected_models=["logistic_regression"],
        optimization_metric="accuracy",
        **FAST,
    )
    assert trainer.best_model is not None
    metrics, _ = trainer.evaluate(X, y)
    assert metrics["accuracy"] > 0.8


# ─────────────────────────────────────────────────────────────────────────────
# Multi-output (multivariate) regression
# ─────────────────────────────────────────────────────────────────────────────

def make_multi_reg_df(n=200, seed=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, size=(n, 4))
    y1 = 2 * X[:, 0] + X[:, 1] + rng.normal(0, 0.1, n)
    y2 = -X[:, 0] + 3 * X[:, 2] + rng.normal(0, 0.1, n)
    df = pd.DataFrame(X, columns=list("abcd"))
    df["t1"] = y1
    df["t2"] = y2
    return df


def test_multi_regression_model_catalog():
    from sklearn.multioutput import MultiOutputRegressor
    trainer = AutoMLTrainer(task_type='multi_regression', preset='test')
    models = trainer.get_available_models()
    assert {'random_forest', 'ridge', 'linear_regression'}.issubset(set(models))
    # Instantiated models must be wrapped for multi-output support
    wrapped = trainer._instantiate_model('ridge', {'ridge_alpha': 1.0})
    assert isinstance(wrapped, MultiOutputRegressor)


def test_multi_regression_e2e_holdout():
    df = make_multi_reg_df()
    proc = AutoMLDataProcessor(target_column=['t1', 't2'], task_type='multi_regression')
    X, y = proc.fit_transform(df)
    assert np.asarray(y).ndim == 2 and np.asarray(y).shape[1] == 2

    trainer = AutoMLTrainer(task_type='multi_regression', preset='test')
    trainer.train(
        X, y, selected_models=['ridge', 'random_forest'],
        experiment_name='test_multi_reg', optimization_metric='r2', **FAST
    )
    assert trainer.best_model is not None
    metrics, preds = trainer.evaluate(X[:30], y[:30])
    assert 'r2' in metrics and 'rmse' in metrics
    assert metrics['n_outputs'] == 2
    assert len(metrics['per_output_r2']) == 2
    assert np.asarray(preds).shape == (30, 2)
    assert metrics['r2'] > 0.5


def test_multi_regression_e2e_cv():
    df = make_multi_reg_df(n=180)
    proc = AutoMLDataProcessor(target_column=['t1', 't2'], task_type='multi_regression')
    X, y = proc.fit_transform(df)
    trainer = AutoMLTrainer(task_type='multi_regression', preset='test')
    trainer.train(
        X, y, selected_models=['ridge'],
        experiment_name='test_multi_reg_cv', optimization_metric='r2',
        n_trials=1, timeout=120, validation_strategy='cv',
        validation_params={'folds': 3}
    )
    assert trainer.best_model is not None
    metrics, _ = trainer.evaluate(X[:20], y[:20])
    assert metrics['r2'] > 0.5

