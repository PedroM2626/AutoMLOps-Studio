"""Coverage for src/utils/explainers.py::ModelExplainer.

ModelExplainer is the SHAP wrapper behind the explainability reports produced by
src/engines/classical.py (plot_importance) and imported by app.py.  Every test here
drives it the way the product does: a real estimator fitted on a real encoded frame,
then get_shap_values / plot_importance on a held-out slice.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import OrdinalEncoder

from src.utils.explainers import ModelExplainer

N_ROWS = 48
INFORMATIVE_FEATURE = "num_a"
NOISE_FEATURE = "noise"

@pytest.fixture(scope="module")
def tiny_frame():
    """Numeric columns plus a categorical one, encoded exactly the way
    AutoMLDataProcessor encodes it before training (one-hot indicators for the
    low-cardinality column, ordinal codes for the high-cardinality one), because
    that encoded frame is what ModelExplainer receives in production."""
    rng = np.random.default_rng(7)
    n = N_ROWS
    num_a = rng.normal(0.0, 1.0, n)
    num_b = rng.normal(0.0, 1.0, n)
    noise = rng.normal(0.0, 1.0, n)
    colour = rng.choice(["red", "green", "blue"], size=n)
    size = rng.choice(["small", "medium", "large"], size=n)

    X = pd.DataFrame(
        {
            "num_a": num_a,
            "num_b": num_b,
            "noise": noise,
            "colour_green": (colour == "green").astype("int64"),
            "colour_red": (colour == "red").astype("int64"),
            "size_ordinal": OrdinalEncoder()
            .fit_transform(pd.DataFrame({"size": size}))
            .ravel()
            .astype("int64"),
        }
    )

    score = 2.0 * num_a - 1.5 * num_b + 0.8 * X["colour_green"]
    y_binary = (score > np.median(score)).astype("int64")
    y_multi = pd.qcut(score, 3, labels=[0, 1, 2]).astype("int64")
    y_reg = 2.0 * num_a - 1.5 * num_b + 0.4 * X["size_ordinal"] + 0.05 * noise
    return X, y_binary, y_multi, y_reg


@pytest.fixture(scope="module")
def random_forest(tiny_frame):
    X, y, _, _ = tiny_frame
    return RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y)


@pytest.fixture(scope="module")
def random_forest_multi(tiny_frame):
    X, _, y_multi, _ = tiny_frame
    return RandomForestClassifier(n_estimators=20, random_state=0).fit(X, y_multi)


@pytest.fixture(scope="module")
def extra_trees_regressor(tiny_frame):
    X, _, _, y_reg = tiny_frame
    return ExtraTreesRegressor(n_estimators=20, random_state=0).fit(X, y_reg)


@pytest.fixture(scope="module")
def logistic_clf(tiny_frame):
    X, y, _, _ = tiny_frame
    return LogisticRegression(max_iter=500).fit(X, y)


@pytest.fixture(scope="module")
def logistic_multi(tiny_frame):
    X, _, y_multi, _ = tiny_frame
    return LogisticRegression(max_iter=500).fit(X, y_multi)


@pytest.fixture(scope="module")
def linear_reg(tiny_frame):
    X, _, _, y_reg = tiny_frame
    return LinearRegression().fit(X, y_reg)


def _values_matrix(shap_output):
    """Normalise the several shapes ModelExplainer can hand back into an ndarray."""
    if isinstance(shap_output, list):
        return np.stack([np.asarray(v, dtype=float) for v in shap_output], axis=-1)
    if hasattr(shap_output, "values"):
        return np.asarray(shap_output.values, dtype=float)
    return np.asarray(shap_output, dtype=float)


def _assert_matches_input(values, n_samples, n_features):
    assert values.ndim in (2, 3), f"unexpected SHAP tensor rank: {values.shape}"
    assert values.shape[0] == n_samples
    assert values.shape[1] == n_features
    assert not np.isnan(values).any(), "SHAP values contain NaNs"
    assert np.isfinite(values).all(), "SHAP values contain non-finite entries"


def _background_rows(explainer):
    """Background size kept by the explainer; the internal layout differs between
    the explainer classes, so probe the two shap 0.50 spellings."""
    for attr in ("masker", "data"):
        candidate = getattr(explainer, attr, None)
        if candidate is None:
            continue
        shape = getattr(candidate, "shape", None)
        if shape is not None:
            return shape[0]
        inner = getattr(candidate, "data", None)
        inner_shape = getattr(inner, "shape", None)
        if inner_shape is not None:
            return inner_shape[0]
    return None


def test_task_type_defaults_to_classification(random_forest, tiny_frame):
    X, _, _, _ = tiny_frame
    explainer = ModelExplainer(random_forest, X)
    assert explainer.task_type == "classification"
    assert explainer.model is random_forest
    assert explainer.use_native_catboost is False


@pytest.mark.parametrize("model_name", ["random_forest", "extra_trees_regressor"])
def test_tree_models_take_the_tree_explainer_branch(model_name, request, tiny_frame):
    X, _, _, _ = tiny_frame
    model = request.getfixturevalue(model_name)
    explainer = ModelExplainer(model, X, task_type="classification")
    assert "TreeExplainer" in type(explainer.explainer).__name__
    assert explainer.use_native_catboost is False


def test_xgboost_takes_the_tree_explainer_branch(tiny_frame):
    xgboost = pytest.importorskip("xgboost")
    X, y, _, _ = tiny_frame
    model = xgboost.XGBClassifier(
        n_estimators=20, max_depth=3, eval_metric="logloss", random_state=0
    ).fit(X, y)
    explainer = ModelExplainer(model, X, task_type="classification")
    assert "TreeExplainer" in type(explainer.explainer).__name__


def test_sklearn_boosting_outside_allowlist_is_still_explained(tiny_frame, logistic_clf):
    X, y, _, _ = tiny_frame
    model = GradientBoostingClassifier(n_estimators=20, random_state=0).fit(X, y)
    explainer = ModelExplainer(model, X, task_type="classification")
    values = _values_matrix(explainer.get_shap_values(X.head(8)))
    _assert_matches_input(values, 8, X.shape[1])


def test_tree_binary_shap_values_shape_and_finiteness(random_forest, tiny_frame):
    X, _, _, _ = tiny_frame
    explainer = ModelExplainer(random_forest, X, task_type="classification")
    probe = X.head(12)
    values = _values_matrix(explainer.get_shap_values(probe))
    _assert_matches_input(values, 12, X.shape[1])
    if values.ndim == 3:
        assert values.shape[2] == 2, "binary classifier should expose one axis per class"


def test_tree_multiclass_shap_values_are_per_class(random_forest_multi, tiny_frame):
    X, _, _, _ = tiny_frame
    n_classes = 3
    explainer = ModelExplainer(random_forest_multi, X, task_type="classification")
    probe = X.head(12)
    values = _values_matrix(explainer.get_shap_values(probe))
    assert values.ndim == 3, f"expected per-class tensor, got {values.shape}"
    assert values.shape == (12, X.shape[1], n_classes)
    assert not np.isnan(values).any()


def test_tree_regression_shap_values_shape_and_finiteness(extra_trees_regressor, tiny_frame):
    X, _, _, _ = tiny_frame
    explainer = ModelExplainer(extra_trees_regressor, X, task_type="regression")
    probe = X.head(12)
    values = _values_matrix(explainer.get_shap_values(probe))
    assert values.ndim == 2
    _assert_matches_input(values, 12, X.shape[1])


def test_informative_feature_dominates_noise_feature(linear_reg, tiny_frame):
    X, _, _, _ = tiny_frame
    explainer = ModelExplainer(linear_reg, X, task_type="regression")
    values = _values_matrix(explainer.get_shap_values(X))
    if values.ndim == 3:
        values = values[..., -1]
    importance = np.abs(values).mean(axis=0)
    by_name = dict(zip(X.columns, importance))
    assert by_name[INFORMATIVE_FEATURE] > by_name[NOISE_FEATURE], (
        "SHAP failed to rank the driver above the pure-noise column: "
        f"{ {k: round(v, 4) for k, v in by_name.items()} }"
    )


def test_logistic_regression_does_not_use_the_tree_explainer(logistic_clf, tiny_frame):
    X, _, _, _ = tiny_frame
    explainer = ModelExplainer(logistic_clf, X, task_type="classification")
    assert "TreeExplainer" not in type(explainer.explainer).__name__
    assert type(explainer.explainer).__name__ in {
        "LinearExplainer",
        "KernelExplainer",
        "Permutation",
        "Exact",
    }


def test_binary_logistic_shap_values_shape_and_finiteness(logistic_clf, tiny_frame):
    X, _, _, _ = tiny_frame
    explainer = ModelExplainer(logistic_clf, X, task_type="classification")
    probe = X.head(12)
    values = _values_matrix(explainer.get_shap_values(probe))
    _assert_matches_input(values, 12, X.shape[1])


def test_multiclass_logistic_shap_values_are_per_class(logistic_multi, tiny_frame):
    X, _, _, _ = tiny_frame
    explainer = ModelExplainer(logistic_multi, X, task_type="classification")
    probe = X.head(12)
    values = _values_matrix(explainer.get_shap_values(probe))
    assert values.ndim == 3
    assert values.shape == (12, X.shape[1], 3)
    assert not np.isnan(values).any()


def test_linear_regression_fallback_explainer_path(linear_reg, tiny_frame):
    X, _, _, _ = tiny_frame
    explainer = ModelExplainer(linear_reg, X, task_type="regression")
    assert "TreeExplainer" not in type(explainer.explainer).__name__
    probe = X.head(12)
    values = _values_matrix(explainer.get_shap_values(probe))
    assert values.ndim == 2
    _assert_matches_input(values, 12, X.shape[1])


@pytest.mark.parametrize(
    "model_name,task_type,expected_class_dim",
    [
        ("knn_classifier", "classification", 2),
        ("knn_regressor", "regression", None),
    ],
)
def test_kernel_explainer_fallback_branch(
    model_name, task_type, expected_class_dim, request, tiny_frame
):
    """shap.Explainer cannot analyse neighbours models, so the class must land on its
    last-resort KernelExplainer and still return usable values."""
    X, _, _, _ = tiny_frame
    model = request.getfixturevalue(model_name)
    explainer = ModelExplainer(model, X, task_type=task_type)
    assert type(explainer.explainer).__name__ == "KernelExplainer"

    probe = X.head(6)
    values = _values_matrix(explainer.get_shap_values(probe))
    _assert_matches_input(values, 6, X.shape[1])
    if expected_class_dim is not None:
        assert values.ndim == 3, "classification fallback must explain the probability output"


@pytest.fixture(scope="module")
def knn_classifier(tiny_frame):
    X, y, _, _ = tiny_frame
    return KNeighborsClassifier(n_neighbors=3).fit(X, y)


@pytest.fixture(scope="module")
def knn_regressor(tiny_frame):
    X, _, _, y_reg = tiny_frame
    return KNeighborsRegressor(n_neighbors=3).fit(X, y_reg)


def test_background_is_sampled_for_large_training_sets(knn_classifier, tiny_frame):
    X, y, _, _ = tiny_frame
    # The sampling branch only triggers above 100 training rows, so this one test
    # repeats the 48-row frame instead of keeping every fixture that large.
    big_X = pd.concat([X] * 3, ignore_index=True)
    big_y = pd.concat([y] * 3, ignore_index=True)
    model = KNeighborsClassifier(n_neighbors=3).fit(big_X, big_y)

    explainer = ModelExplainer(model, big_X, task_type="classification")
    assert len(explainer.X_train) == len(big_X), "training frame must not be mutated"
    rows = _background_rows(explainer.explainer)
    assert rows == 100, f"expected the background to be capped at 100 rows, got {rows}"

    values = _values_matrix(explainer.get_shap_values(big_X.head(6)))
    _assert_matches_input(values, 6, X.shape[1])


def test_numpy_training_matrix_is_supported(logistic_clf, tiny_frame):
    X, _, _, _ = tiny_frame
    X_np = X.to_numpy(dtype="float64")
    explainer = ModelExplainer(logistic_clf, X_np, task_type="classification")
    values = _values_matrix(explainer.get_shap_values(X_np[:8]))
    assert values.shape[0] == 8
    assert values.shape[1] == X.shape[1]
    assert not np.isnan(values).any()


@pytest.mark.parametrize("plot_type", ["summary", "bar"])
def test_plot_importance_returns_figure_for_tree_model(
    plot_type, random_forest, tiny_frame
):
    X, _, _, _ = tiny_frame
    explainer = ModelExplainer(random_forest, X, task_type="classification")
    fig = None
    try:
        fig = explainer.plot_importance(X.head(12), plot_type=plot_type)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert fig.axes, "SHAP plot was rendered without any axes"
    finally:
        if fig is not None:
            plt.close(fig)


@pytest.mark.parametrize("plot_type", ["summary", "bar"])
def test_plot_importance_returns_figure_for_linear_model(
    plot_type, logistic_clf, tiny_frame
):
    X, _, _, _ = tiny_frame
    explainer = ModelExplainer(logistic_clf, X, task_type="classification")
    fig = None
    try:
        fig = explainer.plot_importance(X.head(12), plot_type=plot_type)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert fig.axes
    finally:
        if fig is not None:
            plt.close(fig)


def test_plot_importance_returns_figure_for_multiclass_tree_model(
    random_forest_multi, tiny_frame
):
    X, _, _, _ = tiny_frame
    explainer = ModelExplainer(random_forest_multi, X, task_type="classification")
    fig = None
    try:
        fig = explainer.plot_importance(X.head(12))
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        if fig is not None:
            plt.close(fig)


def test_plot_importance_returns_figure_for_regression_model(extra_trees_regressor, tiny_frame):
    X, _, _, _ = tiny_frame
    explainer = ModelExplainer(extra_trees_regressor, X, task_type="regression")
    fig = None
    try:
        fig = explainer.plot_importance(X.head(12))
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        if fig is not None:
            plt.close(fig)


def test_catboost_native_path(tiny_frame):
    catboost = pytest.importorskip("catboost")
    X, y, _, _ = tiny_frame
    model = catboost.CatBoostClassifier(
        iterations=20, depth=3, verbose=0, random_seed=0
    ).fit(X, y)

    explainer = ModelExplainer(model, X, task_type="classification")
    assert explainer.use_native_catboost is True
    assert explainer.explainer is None, "native path must not build a shap explainer"

    values = _values_matrix(explainer.get_shap_values(X.head(10)))
    # Bias column is stripped, so the matrix is exactly (samples, features).
    assert values.shape == (10, X.shape[1])
    assert not np.isnan(values).any()

    fig = None
    try:
        fig = explainer.plot_importance(X.head(10))
        assert isinstance(fig, matplotlib.figure.Figure)
    finally:
        if fig is not None:
            plt.close(fig)


def test_catboost_native_path_for_regression(tiny_frame):
    catboost = pytest.importorskip("catboost")
    X, _, _, y_reg = tiny_frame
    model = catboost.CatBoostRegressor(
        iterations=20, depth=3, verbose=0, random_seed=0
    ).fit(X, y_reg)

    explainer = ModelExplainer(model, X, task_type="regression")
    assert explainer.use_native_catboost is True
    values = _values_matrix(explainer.get_shap_values(X.head(10)))
    assert values.shape == (10, X.shape[1])
    assert not np.isnan(values).any()


def test_catboost_native_path_returns_none_when_features_do_not_match(tiny_frame):
    """Pins the current contract: a feature-name mismatch is swallowed and reported as
    None, so src/engines/classical.py:2920 silently drops the SHAP chart instead of
    surfacing an error to the user."""
    catboost = pytest.importorskip("catboost")
    X, y, _, _ = tiny_frame
    model = catboost.CatBoostClassifier(
        iterations=20, depth=3, verbose=0, random_seed=0
    ).fit(X, y)
    explainer = ModelExplainer(model, X, task_type="classification")

    mismatched = X.drop(columns=["size_ordinal"]).rename(columns={"num_a": "renamed_a"})
    assert explainer.get_shap_values(mismatched.head(6)) is None
    assert explainer.plot_importance(mismatched.head(6)) is None


# ---------------------------------------------------------------------------
# build_waterfall: the What-If Simulator's SHAP path
# ---------------------------------------------------------------------------
def _binary_forest():
    from sklearn.ensemble import RandomForestClassifier

    rng = np.random.default_rng(3)
    X = rng.normal(0.0, 1.0, (60, 3))
    y = (X[:, 0] > 0).astype(int)
    frame = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    return RandomForestClassifier(n_estimators=8, random_state=0).fit(frame, y), frame


def test_waterfall_plots_one_class_vector_for_a_binary_tree_model():
    import shap
    from src.utils.explainers import build_waterfall

    model, frame = _binary_forest()
    row = frame.head(1)
    explainer = shap.TreeExplainer(model)

    explanation = build_waterfall(
        explainer.shap_values(row), explainer.expected_value, row.iloc[0],
        frame.columns, focus_class=1,
    )

    assert explanation.values.shape == (frame.shape[1],)
    assert explanation.base_values == float(np.asarray(explainer.expected_value).ravel()[1])
    assert list(explanation.feature_names) == list(frame.columns)

    fig = _render(explanation)
    assert len(fig.axes) >= 1


def test_waterfall_handles_the_regression_shape():
    import shap
    from sklearn.ensemble import RandomForestRegressor
    from src.utils.explainers import build_waterfall

    rng = np.random.default_rng(4)
    X = rng.normal(0.0, 1.0, (60, 3))
    frame = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    model = RandomForestRegressor(n_estimators=8, random_state=0).fit(frame, X[:, 0] * 2)
    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(frame.head(1))
    assert np.asarray(raw).ndim == 2

    explanation = build_waterfall(raw, explainer.expected_value, frame.iloc[0], frame.columns)

    assert explanation.values.shape == (3,)


def test_waterfall_accepts_the_per_class_list_form():
    from src.utils.explainers import build_waterfall

    per_class = [np.zeros((1, 2)), np.ones((1, 2))]
    row = pd.Series([0.5, -0.5], index=["a", "b"])

    explanation = build_waterfall(per_class, [0.1, 0.9], row, ["a", "b"], focus_class=1)

    assert explanation.values.tolist() == [1.0, 1.0]
    assert explanation.base_values == 0.9


def test_waterfall_falls_back_to_class_zero_when_the_focus_is_out_of_range():
    from src.utils.explainers import build_waterfall

    raw = np.zeros((1, 2, 2))
    raw[0, :, 1] = 5.0
    row = pd.Series([0.0, 0.0], index=["a", "b"])

    explanation = build_waterfall(raw, [0.0, 1.0], row, ["a", "b"], focus_class=9)

    assert explanation.values.tolist() == [0.0, 0.0]
    assert explanation.base_values == 0.0


def test_waterfall_rejects_a_shape_it_cannot_plot():
    import pytest
    from src.utils.explainers import build_waterfall

    with pytest.raises(ValueError, match="Unsupported SHAP value shape"):
        build_waterfall(np.zeros((1, 2, 2, 2)), [0.0], pd.Series([0.0, 0.0], index=["a", "b"]), ["a", "b"])


def _render(explanation):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import shap

    plt.close("all")
    shap.waterfall_plot(explanation, show=False)
    return plt.gcf()




def test_xgboost_explanation_stays_on_the_fast_tree_path(tiny_frame):
    """shap 0.49.1 could not read an xgboost 3.1.1 tree - it raised while parsing the leaf
    values, so every boosted-tree explanation fell through to KernelExplainer and the
    Experiments page silently lost its SHAP chart. The pinned shap has to keep this fast."""
    import shap
    xgboost = pytest.importorskip("xgboost")
    X, y, _, _ = tiny_frame
    model = xgboost.XGBClassifier(
        n_estimators=20, max_depth=3, eval_metric="logloss", random_state=0
    ).fit(X, y)

    explainer = ModelExplainer(model, X, task_type="classification")

    assert "TreeExplainer" in type(explainer.explainer).__name__
    values = _values_matrix(explainer.get_shap_values(X.head(8)))
    _assert_matches_input(values, 8, X.shape[1])
    assert np.isfinite(values).all()
