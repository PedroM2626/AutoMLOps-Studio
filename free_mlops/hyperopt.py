from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from free_mlops.config import get_settings


class HyperparameterOptimizer:
    """Otimizador de hiperparâmetros usando Optuna."""
    
    def __init__(self, settings: Optional[Any] = None):
        self.settings = settings or get_settings()
        self.study_dir = self.settings.artifacts_dir / "hyperopt"
        self.study_dir.mkdir(parents=True, exist_ok=True)
        
        # Definições de espaços de busca para cada modelo
        self.search_spaces = {
            "logistic_regression": {
                "C": ("loguniform", 1e-4, 1e4),
                "penalty": ("categorical", ["l1", "l2"]),
                "solver": ("categorical", ["liblinear", "saga"]),
                "max_iter": ("int", 100, 2000),
            },
            "ridge": {
                "alpha": ("loguniform", 1e-4, 1e4),
                "solver": ("categorical", ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]),
            },
            "lasso": {
                "alpha": ("loguniform", 1e-4, 1e4),
                "max_iter": ("int", 100, 2000),
                "selection": ("categorical", ["cyclic", "random"]),
            },
            "elastic_net": {
                "alpha": ("loguniform", 1e-4, 1e4),
                "l1_ratio": ("uniform", 0, 1),
                "max_iter": ("int", 100, 2000),
                "selection": ("categorical", ["cyclic", "random"]),
            },
            "random_forest_classifier": {
                "n_estimators": ("int", 50, 500),
                "max_depth": ("int", 3, 20) or ("categorical", [None]),
                "min_samples_split": ("int", 2, 20),
                "min_samples_leaf": ("int", 1, 20),
                "max_features": ("categorical", ["sqrt", "log2", None]),
            },
            "random_forest_regressor": {
                "n_estimators": ("int", 50, 500),
                "max_depth": ("int", 3, 20) or ("categorical", [None]),
                "min_samples_split": ("int", 2, 20),
                "min_samples_leaf": ("int", 1, 20),
                "max_features": ("categorical", ["sqrt", "log2", None]),
            },
            "gradient_boosting_classifier": {
                "n_estimators": ("int", 50, 300),
                "learning_rate": ("loguniform", 1e-4, 1e-1),
                "max_depth": ("int", 3, 10),
                "min_samples_split": ("int", 2, 20),
                "min_samples_leaf": ("int", 1, 20),
                "subsample": ("uniform", 0.6, 1.0),
            },
            "gradient_boosting_regressor": {
                "n_estimators": ("int", 50, 300),
                "learning_rate": ("loguniform", 1e-4, 1e-1),
                "max_depth": ("int", 3, 10),
                "min_samples_split": ("int", 2, 20),
                "min_samples_leaf": ("int", 1, 20),
                "subsample": ("uniform", 0.6, 1.0),
            },
            "decision_tree_classifier": {
                "max_depth": ("int", 3, 20) or ("categorical", [None]),
                "min_samples_split": ("int", 2, 20),
                "min_samples_leaf": ("int", 1, 20),
                "max_features": ("categorical", ["sqrt", "log2", None]),
            },
            "decision_tree_regressor": {
                "max_depth": ("int", 3, 20) or ("categorical", [None]),
                "min_samples_split": ("int", 2, 20),
                "min_samples_leaf": ("int", 1, 20),
                "max_features": ("categorical", ["sqrt", "log2", None]),
            },
            "knn_classifier": {
                "n_neighbors": ("int", 1, 50),
                "weights": ("categorical", ["uniform", "distance"]),
                "algorithm": ("categorical", ["auto", "ball_tree", "kd_tree", "brute"]),
                "p": ("int", 1, 2),
            },
            "knn_regressor": {
                "n_neighbors": ("int", 1, 50),
                "weights": ("categorical", ["uniform", "distance"]),
                "algorithm": ("categorical", ["auto", "ball_tree", "kd_tree", "brute"]),
                "p": ("int", 1, 2),
            },
            "svr": {
                "C": ("loguniform", 1e-3, 1e3),
                "epsilon": ("loguniform", 1e-4, 1e-1),
                "kernel": ("categorical", ["linear", "poly", "rbf", "sigmoid"]),
                "gamma": ("categorical", ["scale", "auto"]) or ("loguniform", 1e-4, 1e-1),
            },
        }
        
        # Mapeamento de modelos
        self.model_classes = {
            "logistic_regression": LogisticRegression,
            "ridge": Ridge,
            "lasso": Lasso,
            "elastic_net": ElasticNet,
            "random_forest_classifier": RandomForestClassifier,
            "random_forest_regressor": RandomForestRegressor,
            "gradient_boosting_classifier": GradientBoostingClassifier,
            "gradient_boosting_regressor": GradientBoostingRegressor,
            "decision_tree_classifier": DecisionTreeClassifier,
            "decision_tree_regressor": DecisionTreeRegressor,
            "knn_classifier": KNeighborsClassifier,
            "knn_regressor": KNeighborsRegressor,
            "svr": SVR,
        }
    
    def optimize_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_name: str,
        problem_type: str,
        n_trials: int = 100,
        timeout: Optional[int] = None,
        cv_folds: int = 5,
        random_state: int = 42,
    ) -> Dict[str, Any]:
        """Otimiza hiperparâmetros para um modelo específico."""
        
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna não está instalado. Instale com: pip install optuna")
        
        # Preparar dados
        X_train_processed, X_val_processed, preprocessor = self._preprocess_data(
            X_train, X_val, problem_type
        )
        
        # Criar study
        study_name = f"{model_name}_{problem_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=random_state),
        )
        
        # Função objetivo
        def objective(trial):
            try:
                # Sugerir hiperparâmetros
                params = self._suggest_params(trial, model_name)
                
                # Criar modelo
                model = self._create_model(model_name, params, random_state)
                
                # Criar pipeline
                pipeline = Pipeline([
                    ("preprocess", preprocessor),
                    ("model", model),
                ])
                
                # Avaliar modelo
                if problem_type in ["classification", "multiclass_classification", "binary_classification"]:
                    # Usar validação cruzada para classificação
                    scores = cross_val_score(
                        pipeline, X_train, y_train,
                        cv=cv_folds,
                        scoring="f1_weighted",
                        n_jobs=-1,
                    )
                    score = scores.mean()
                else:
                    # Usar validação cruzada para regressão
                    scores = cross_val_score(
                        pipeline, X_train, y_train,
                        cv=cv_folds,
                        scoring="neg_mean_squared_error",
                        n_jobs=-1,
                    )
                    score = -scores.mean()  # Negativo porque otimização é maximize
                
                return score
                
            except Exception as e:
                # Retornar valor ruim se falhar
                return -1.0 if problem_type == "regression" else 0.0
        
        # Otimizar
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
        
        # Treinar melhor modelo com dados completos
        best_params = study.best_params
        best_model = self._create_model(model_name, best_params, random_state)
        best_pipeline = Pipeline([
            ("preprocess", preprocessor),
            ("model", best_model),
        ])
        
        # Treinar em todos os dados de treino
        best_pipeline.fit(X_train, y_train)
        
        # Avaliar em validação
        y_pred = best_pipeline.predict(X_val_processed)
        
        if problem_type in ["classification", "multiclass_classification", "binary_classification"]:
            metrics = {
                "accuracy": float(accuracy_score(y_val, y_pred)),
                "f1_weighted": float(f1_score(y_val, y_pred, average="weighted")),
            }
        else:
            metrics = {
                "rmse": float(np.sqrt(mean_squared_error(y_val, y_pred))),
                "mae": float(mean_absolute_error(y_val, y_pred)),
                "r2": float(r2_score(y_val, y_pred)),
            }
        
        # Preparar resultado
        result = {
            "model_name": model_name,
            "problem_type": problem_type,
            "study_name": study_name,
            "best_params": best_params,
            "best_score": study.best_value,
            "n_trials": len(study.trials),
            "validation_metrics": metrics,
            "optimization_time": time.time(),
            "preprocessor": preprocessor,
            "best_pipeline": best_pipeline,
        }
        
        # Salvar estudo
        self._save_study(study, result)
        
        return result
    
    def _preprocess_data(
        self,
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        problem_type: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, ColumnTransformer]:
        """Preprocessa dados para otimização."""
        
        # Identificar colunas
        numeric_cols = []
        categorical_cols = []
        
        for col in X_train.columns:
            if pd.api.types.is_numeric_dtype(X_train[col]):
                numeric_cols.append(col)
            else:
                categorical_cols.append(col)
        
        # Criar preprocessador
        transformers = []
        
        if numeric_cols:
            numeric_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("num", numeric_transformer, numeric_cols))
        
        if categorical_cols:
            categorical_transformer = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True, max_categories=50)),
            ])
            transformers.append(("cat", categorical_transformer, categorical_cols))
        
        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
        
        # Preprocessar dados
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        
        return X_train_processed, X_val_processed, preprocessor
    
    def _suggest_params(self, trial, model_name: str) -> Dict[str, Any]:
        """Sugere hiperparâmetros usando Optuna trial."""
        
        search_space = self.search_spaces.get(model_name, {})
        params = {}
        
        for param_name, param_config in search_space.items():
            param_type = param_config[0]
            
            if param_type == "uniform":
                params[param_name] = trial.suggest_uniform(param_name, param_config[1], param_config[2])
            elif param_type == "loguniform":
                params[param_name] = trial.suggest_loguniform(param_name, param_config[1], param_config[2])
            elif param_type == "int":
                params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
            elif param_type == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, param_config[1])
            elif param_type == "discrete_uniform":
                params[param_name] = trial.suggest_discrete_uniform(param_name, param_config[1], param_config[2], param_config[3])
        
        return params
    
    def _create_model(self, model_name: str, params: Dict[str, Any], random_state: int) -> Any:
        """Cria instância do modelo com parâmetros."""
        
        model_class = self.model_classes.get(model_name)
        if not model_class:
            raise ValueError(f"Modelo não suportado: {model_name}")
        
        # Adicionar random_state se aplicável
        if hasattr(model_class(), "random_state"):
            params["random_state"] = random_state
        
        # Adicionar n_jobs para modelos que suportam
        if hasattr(model_class(), "n_jobs") and model_name not in ["knn_classifier", "knn_regressor"]:
            params["n_jobs"] = -1
        
        return model_class(**params)
    
    def _save_study(self, study, result: Dict[str, Any]) -> None:
        """Salva estudo e resultados."""
        
        study_file = self.study_dir / f"{result['study_name']}.json"
        
        # Preparar dados para salvar (sem objetos não serializáveis)
        save_data = {
            "study_name": result["study_name"],
            "model_name": result["model_name"],
            "problem_type": result["problem_type"],
            "best_params": result["best_params"],
            "best_score": result["best_score"],
            "n_trials": result["n_trials"],
            "validation_metrics": result["validation_metrics"],
            "optimization_time": result["optimization_time"],
            "trials": [
                {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": trial.state.name,
                }
                for trial in study.trials
            ],
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        
        study_file.write_text(
            json.dumps(save_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
    
    def get_optimization_history(self, study_name: str) -> List[Dict[str, Any]]:
        """Retorna histórico de otimização."""
        
        study_file = self.study_dir / f"{study_name}.json"
        if not study_file.exists():
            return []
        
        try:
            data = json.loads(study_file.read_text(encoding="utf-8"))
            return data.get("trials", [])
        except Exception:
            return []
    
    def list_studies(self) -> List[Dict[str, Any]]:
        """Lista todos os estudos salvos."""
        
        studies = []
        
        for study_file in self.study_dir.glob("*.json"):
            try:
                data = json.loads(study_file.read_text(encoding="utf-8"))
                studies.append({
                    "study_name": data.get("study_name", study_file.stem),
                    "model_name": data.get("model_name", "unknown"),
                    "problem_type": data.get("problem_type", "unknown"),
                    "best_score": data.get("best_score", 0),
                    "n_trials": data.get("n_trials", 0),
                    "saved_at": data.get("saved_at", ""),
                })
            except Exception:
                continue
        
        return sorted(studies, key=lambda x: x["saved_at"], reverse=True)


def get_hyperparameter_optimizer() -> HyperparameterOptimizer:
    """Factory function para HyperparameterOptimizer."""
    return HyperparameterOptimizer()
