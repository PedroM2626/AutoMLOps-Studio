from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from free_mlops.automl import _build_preprocessor
from free_mlops.automl import _infer_columns
from free_mlops.automl import _classification_metrics
from free_mlops.automl import _regression_metrics
from free_mlops.automl import ProblemType


def _default_param_grid(model_name: str, problem_type: ProblemType) -> dict[str, list[Any]]:
    grids: dict[str, dict[str, list[Any]]] = {
        "logistic_regression": {
            "C": [0.1, 1.0, 10.0],
            "penalty": ["l2"],
        },
        "linear_svc": {
            "C": [0.1, 1.0, 10.0],
        },
        "random_forest": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
        },
        "extra_trees": {
            "n_estimators": [200, 300],
            "max_depth": [None, 10, 20],
        },
        "gradient_boosting": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
        },
        "decision_tree": {
            "max_depth": [None, 5, 10],
        },
        "knn": {
            "n_neighbors": [3, 5, 7],
        },
        "ridge": {
            "alpha": [0.1, 1.0, 10.0],
        },
        "lasso": {
            "alpha": [0.01, 0.1, 1.0],
        },
        "elastic_net": {
            "alpha": [0.01, 0.1, 1.0],
            "l1_ratio": [0.2, 0.5, 0.8],
        },
        "svr": {
            "C": [0.1, 1.0, 10.0],
            "kernel": ["linear", "rbf"],
        },
        "random_forest": {
            "n_estimators": [100, 200],
            "max_depth": [None, 10, 20],
        },
        "extra_trees": {
            "n_estimators": [200, 300],
            "max_depth": [None, 10, 20],
        },
        "gradient_boosting": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
        },
        "decision_tree": {
            "max_depth": [None, 5, 10],
        },
        "knn": {
            "n_neighbors": [3, 5, 7],
        },
    }

    return grids.get(model_name, {})


def run_finetune(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: ProblemType,
    model_name: str,
    search_type: Literal["grid", "random"],
    random_state: int,
    cv_folds: int = 3,
) -> dict[str, Any]:
    numeric_cols, categorical_cols = _infer_columns(X_train)
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)

    X_train_pre = preprocessor.fit_transform(X_train)
    X_test_pre = preprocessor.transform(X_test)

    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import LogisticRegression
    from sklearn.linear_model import Ridge
    from sklearn.svm import LinearSVC
    from sklearn.svm import SVR
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neighbors import KNeighborsRegressor

    model_map: dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=2000),
        "linear_svc": LinearSVC(max_iter=5000, dual=False),
        "random_forest": RandomForestClassifier(random_state=random_state, n_jobs=-1),
        "extra_trees": ExtraTreesClassifier(random_state=random_state, n_jobs=-1),
        "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "knn": KNeighborsClassifier(),
        "ridge": Ridge(),
        "lasso": Lasso(max_iter=5000),
        "elastic_net": ElasticNet(max_iter=5000),
        "svr": SVR(),
        "random_forest": RandomForestRegressor(random_state=random_state, n_jobs=-1),
        "extra_trees": ExtraTreesRegressor(random_state=random_state, n_jobs=-1),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
        "decision_tree": DecisionTreeRegressor(random_state=random_state),
        "knn": KNeighborsRegressor(),
    }

    base_model = model_map.get(model_name)
    if base_model is None:
        raise ValueError(f"Unsupported model_name for finetune: {model_name}")

    param_grid = _default_param_grid(model_name, problem_type)
    if not param_grid:
        raise ValueError(f"No parameter grid defined for model: {model_name}")

    if search_type == "grid":
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring="f1_weighted" if problem_type == "classification" else "neg_root_mean_squared_error",
            cv=cv_folds,
            n_jobs=-1,
            verbose=0,
        )
    elif search_type == "random":
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_grid,
            n_iter=10,
            scoring="f1_weighted" if problem_type == "classification" else "neg_root_mean_squared_error",
            cv=cv_folds,
            n_jobs=-1,
            random_state=random_state,
            verbose=0,
        )
    else:
        raise ValueError("search_type must be 'grid' or 'random'")

    search.fit(X_train_pre, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test_pre)

    if problem_type == "classification":
        metrics = _classification_metrics(y_test, y_pred, None)
    else:
        metrics = _regression_metrics(y_test, y_pred)

    best_pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", best_model)])

    return {
        "model_name": model_name,
        "search_type": search_type,
        "best_params": search.best_params_,
        "best_score": float(search.best_score_),
        "metrics": metrics,
        "pipeline": best_pipeline,
        "cv_results": search.cv_results_,
    }
