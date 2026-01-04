from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

ProblemType = Literal["classification", "regression"]


@dataclass
class AutoMLResult:
    best_model_name: str
    best_pipeline: Any
    best_metrics: dict[str, Any]
    leaderboard: list[dict[str, Any]]


def _infer_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def _build_preprocessor(numeric_cols: list[str], categorical_cols: list[str]) -> ColumnTransformer:
    transformers: list[tuple[str, Any, list[str]]] = []

    if numeric_cols:
        numeric_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", numeric_transformer, numeric_cols))

    if categorical_cols:
        categorical_transformer = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, max_categories=50)),
            ]
        )
        transformers.append(("cat", categorical_transformer, categorical_cols))

    if not transformers:
        raise ValueError("No feature columns found after preprocessing.")

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _classification_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray | None) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }

    if y_proba is not None:
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_proba))
        except Exception:
            pass

        try:
            unique = pd.Series(y_true).unique()
            n_classes = len(unique)
            if n_classes == 2:
                positive_proba = y_proba[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_true, positive_proba))
            elif n_classes > 2:
                metrics["roc_auc_ovr_weighted"] = float(
                    roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
                )
        except Exception:
            pass

    return metrics


def _regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, Any]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def run_automl(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: ProblemType,
    random_state: int,
) -> AutoMLResult:
    numeric_cols, categorical_cols = _infer_columns(X_train)
    preprocessor = _build_preprocessor(numeric_cols, categorical_cols)

    X_train_pre = preprocessor.fit_transform(X_train)
    X_test_pre = preprocessor.transform(X_test)

    candidates: dict[str, Any]

    if problem_type == "classification":
        candidates = {
            "logistic_regression": LogisticRegression(max_iter=2000),
            "linear_svc": LinearSVC(max_iter=5000, dual=False),
            "random_forest": RandomForestClassifier(
                n_estimators=200, random_state=random_state, n_jobs=-1
            ),
            "extra_trees": ExtraTreesClassifier(
                n_estimators=300, random_state=random_state, n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingClassifier(random_state=random_state),
            "decision_tree": DecisionTreeClassifier(random_state=random_state),
            "knn": KNeighborsClassifier(),
        }
    elif problem_type == "regression":
        candidates = {
            "ridge": Ridge(),
            "lasso": Lasso(max_iter=5000),
            "elastic_net": ElasticNet(max_iter=5000),
            "svr": SVR(),
            "random_forest": RandomForestRegressor(
                n_estimators=200, random_state=random_state, n_jobs=-1
            ),
            "extra_trees": ExtraTreesRegressor(
                n_estimators=300, random_state=random_state, n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
            "decision_tree": DecisionTreeRegressor(random_state=random_state),
            "knn": KNeighborsRegressor(),
        }
    else:
        raise ValueError(f"Unsupported problem_type: {problem_type}")

    leaderboard: list[dict[str, Any]] = []

    for model_name, model in candidates.items():
        try:
            model.fit(X_train_pre, y_train)

            y_pred = model.predict(X_test_pre)

            y_proba: np.ndarray | None = None
            if problem_type == "classification" and hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test_pre)
                except Exception:
                    y_proba = None

            if problem_type == "classification":
                metrics = _classification_metrics(y_test, y_pred, y_proba)
                score = float(metrics.get("f1_weighted", 0.0))
            else:
                metrics = _regression_metrics(y_test, y_pred)
                score = -float(metrics.get("rmse", float("inf")))

            leaderboard.append(
                {
                    "model_name": model_name,
                    "success": True,
                    "score": float(score),
                    "metrics": metrics,
                }
            )
        except Exception as exc:
            leaderboard.append(
                {
                    "model_name": model_name,
                    "success": False,
                    "score": None,
                    "error": str(exc),
                }
            )

    successful = [r for r in leaderboard if r.get("success") is True]
    if not successful:
        errors = [r.get("error") for r in leaderboard if r.get("error")]
        joined = " | ".join([e for e in errors if isinstance(e, str)])
        raise RuntimeError(f"AutoML failed for all candidates. Errors: {joined}")

    best_row = max(successful, key=lambda r: float(r.get("score", float("-inf"))))
    best_model_name = str(best_row["model_name"])

    best_model = candidates[best_model_name]
    best_model.fit(X_train_pre, y_train)

    best_metrics = dict(best_row.get("metrics", {}))

    best_pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", best_model)])

    return AutoMLResult(
        best_model_name=best_model_name,
        best_pipeline=best_pipeline,
        best_metrics=best_metrics,
        leaderboard=leaderboard,
    )
