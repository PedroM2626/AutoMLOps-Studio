import pandas as pd
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging

logger = logging.getLogger(__name__)

class StabilityAnalyzer:
    def __init__(self, base_model, X, y, task_type='classification', random_state=42):
        """
        Initialize Stability Analyzer.
        
        Args:
            base_model: Scikit-learn estimator (fitted or unfitted).
            X: Feature data (pandas DataFrame or numpy array).
            y: Target data.
            task_type: 'classification' or 'regression'.
            random_state: Base seed for reproducibility.
        """
        self.base_model = base_model
        self.X = X
        self.y = y
        self.task_type = task_type
        self.random_state = random_state

    def _get_metrics(self, y_true, y_pred, y_proba=None):
        metrics = {}
        if self.task_type == 'classification':
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            if y_proba is not None:
                # Handle multiclass or binary AUC
                if len(np.unique(y_true)) == 2:
                    if y_proba.shape[1] == 2:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                    else:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
                else:
                     try:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
                     except:
                        pass
        elif self.task_type == 'regression':
            metrics['mse'] = mean_squared_error(y_true, y_pred)
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
        
        return metrics

    def run_seed_stability(self, n_iterations=10):
        """
        Test stability by varying the model's random_state (initialization).
        Keeps train/test split constant to isolate initialization effect.
        """
        metrics_history = []
        
        # Fixed split for this test to isolate model seed effect
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state
        )
        
        for i in range(n_iterations):
            seed = self.random_state + i
            
            # Clone model to reset it
            model = clone(self.base_model)
            
            # Set seed if model supports it
            if hasattr(model, 'random_state'):
                model.set_params(random_state=seed)
            elif hasattr(model, 'random_seed'): # CatBoost sometimes uses random_seed
                model.set_params(random_seed=seed)
            
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
                
                iter_metrics = self._get_metrics(y_val, y_pred, y_proba)
                iter_metrics['iteration'] = i
                iter_metrics['seed'] = seed
                metrics_history.append(iter_metrics)
            except Exception as e:
                logger.error(f"Error in seed stability iteration {i}: {e}")
            
        return pd.DataFrame(metrics_history)

    def run_split_stability(self, n_splits=10, test_size=0.2):
        """
        Test stability by varying the train/test split (data variation).
        Keeps model seed constant (if possible) to isolate data effect.
        """
        metrics_history = []
        
        for i in range(n_splits):
            split_seed = self.random_state + i
            
            X_train, X_val, y_train, y_val = train_test_split(
                self.X, self.y, test_size=test_size, random_state=split_seed
            )
            
            model = clone(self.base_model)
            # Fix model seed to isolate split effect
            if hasattr(model, 'random_state'):
                model.set_params(random_state=self.random_state)
            elif hasattr(model, 'random_seed'):
                model.set_params(random_seed=self.random_state)
                
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
                
                iter_metrics = self._get_metrics(y_val, y_pred, y_proba)
                iter_metrics['iteration'] = i
                iter_metrics['split_seed'] = split_seed
                metrics_history.append(iter_metrics)
            except Exception as e:
                logger.error(f"Error in split stability iteration {i}: {e}")
            
        return pd.DataFrame(metrics_history)

    def run_hyperparameter_stability(self, param_name, param_values):
        """
        Test stability by varying a specific hyperparameter.
        Keeps split and seed constant.
        """
        metrics_history = []
        
        # Fixed split
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state
        )
        
        for val in param_values:
            model = clone(self.base_model)
            
            # Set parameter
            try:
                params = {param_name: val}
                model.set_params(**params)
                
                if hasattr(model, 'random_state'):
                    model.set_params(random_state=self.random_state)
            except Exception as e:
                logger.error(f"Failed to set param {param_name}={val}: {e}")
                continue
                
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
                
                iter_metrics = self._get_metrics(y_val, y_pred, y_proba)
                iter_metrics['param_value'] = val
                metrics_history.append(iter_metrics)
            except Exception as e:
                logger.error(f"Error in hyperparam stability for {param_name}={val}: {e}")
            
        return pd.DataFrame(metrics_history)

    def run_general_stability_check(self, n_iterations=10):
        """
        Runs a combined stability check (Seed + Split) to give a general assessment.
        Returns a dictionary with summarized metrics.
        """
        # 1. Seed Stability
        seed_results = self.run_seed_stability(n_iterations=n_iterations)
        seed_metrics = self.calculate_stability_metrics(seed_results)
        
        # 2. Split Stability (Monte Carlo)
        split_results = self.run_stability_test(n_iterations=n_iterations, cv_strategy='monte_carlo')
        split_metrics = self.calculate_stability_metrics(split_results)
        
        combined_report = {
            'seed_stability': seed_metrics,
            'split_stability': split_metrics,
            'raw_seed': seed_results,
            'raw_split': split_results
        }
        
        return combined_report

    def calculate_stability_metrics(self, df_results):
        """
        Calculates aggregate stability metrics (mean, std, stability score) from raw results.
        """
        if df_results.empty:
            return pd.DataFrame()

        summary = {}
        # Identify metric columns (exclude metadata like iteration, seed, etc.)
        metric_cols = [c for c in df_results.columns if c not in ['iteration', 'seed', 'split_seed', 'param_value']]
        
        for col in metric_cols:
            try:
                series = df_results[col]
                mean_val = series.mean()
                std_val = series.std()
                min_val = series.min()
                max_val = series.max()
                
                # Stability Score: Higher is better (0 to 1). 
                # Formula: 1 / (1 + std_dev)
                # If std is 0, score is 1. If std is high, score drops.
                stability_score = 1.0 / (1.0 + (std_val * 10)) # Multiplied by 10 to be more sensitive to small stds in accuracy (e.g. 0.01)
                
                # Alternative score considering mean performance too?
                # User proposed: "1 / (1 + desvio_padrao)"
                user_stability_score = 1.0 / (1.0 + std_val)

                summary[col] = {
                    'mean': mean_val,
                    'std': std_val,
                    'min': min_val,
                    'max': max_val,
                    'range': max_val - min_val,
                    'cv': (std_val / mean_val) if mean_val != 0 else 0,
                    'stability_score': user_stability_score
                }
            except Exception as e:
                pass
                
        return pd.DataFrame(summary).T

    def run_stability_test(self, n_iterations=10, test_size=0.2, perturbation=0.0, cv_strategy='monte_carlo'):
        """
        Runs a comprehensive stability test.
        Varies data split (Monte Carlo CV, K-Fold, etc.) and optionally adds noise (perturbation).
        
        Args:
            cv_strategy: 'monte_carlo', 'kfold', 'stratified_kfold', 'time_series_split'
        """
        metrics_history = []
        splits = []

        # Generate Splits
        if cv_strategy == 'kfold':
            kf = KFold(n_splits=n_iterations, shuffle=True, random_state=self.random_state)
            splits = list(kf.split(self.X, self.y))
        elif cv_strategy == 'stratified_kfold':
            if self.task_type == 'classification':
                skf = StratifiedKFold(n_splits=n_iterations, shuffle=True, random_state=self.random_state)
                # Stratified requires y for splitting
                splits = list(skf.split(self.X, self.y))
            else:
                # Fallback to KFold for regression/clustering
                kf = KFold(n_splits=n_iterations, shuffle=True, random_state=self.random_state)
                splits = list(kf.split(self.X, self.y))
        elif cv_strategy == 'time_series_split':
            tscv = TimeSeriesSplit(n_splits=n_iterations)
            splits = list(tscv.split(self.X, self.y))
        else: # monte_carlo
            pass
        
        for i in range(n_iterations):
            seed = self.random_state + i
            
            # 1. Data Splitting
            if cv_strategy == 'monte_carlo':
                # Monte Carlo Shuffle Split
                try:
                    stratify = self.y if (self.task_type == 'classification' and self.y is not None) else None
                    if stratify is not None:
                         # Check if enough samples per class
                        if isinstance(stratify, pd.Series):
                             counts = stratify.value_counts()
                             if counts.min() < 2:
                                 stratify = None
                    
                    X_train, X_val, y_train, y_val = train_test_split(
                        self.X, self.y, test_size=test_size, random_state=seed, stratify=stratify
                    )
                except ValueError:
                    # Fallback if stratify fails (e.g. too few samples)
                    X_train, X_val, y_train, y_val = train_test_split(
                        self.X, self.y, test_size=test_size, random_state=seed
                    )
            else:
                # K-Fold / Stratified / TimeSeries
                if i >= len(splits): break
                train_idx, val_idx = splits[i]
                
                # Handle DataFrame vs Numpy
                if isinstance(self.X, pd.DataFrame):
                    X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                else:
                    X_train, X_val = self.X[train_idx], self.X[val_idx]
                    
                if isinstance(self.y, pd.Series) or isinstance(self.y, pd.DataFrame):
                    y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
                else:
                    y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            # 2. Model Training & Evaluation
            # Clone model to ensure fresh start
            model = clone(self.base_model)
            if hasattr(model, 'random_state'):
                model.set_params(random_state=self.random_state)
            elif hasattr(model, 'random_seed'):
                model.set_params(random_seed=self.random_state)
                
            try:
                model.fit(X_train, y_train)
                
                # Apply perturbation if requested
                X_test_final = X_val
                if perturbation > 0:
                    # Determine numeric columns for noise addition
                    if isinstance(X_val, pd.DataFrame):
                        numeric_cols = X_val.select_dtypes(include=[np.number]).columns
                        if not numeric_cols.empty:
                            X_test_final = X_val.copy()
                            # Add noise
                            noise = np.random.normal(0, perturbation, X_val[numeric_cols].shape)
                            X_test_final[numeric_cols] += noise
                    else:
                        # Numpy array
                        noise = np.random.normal(0, perturbation, X_val.shape)
                        X_test_final = X_val + noise
                    
                y_pred = model.predict(X_test_final)
                y_proba = model.predict_proba(X_test_final) if hasattr(model, "predict_proba") else None
                
                iter_metrics = self._get_metrics(y_val, y_pred, y_proba)
                iter_metrics['iteration'] = i
                metrics_history.append(iter_metrics)
                
            except Exception as e:
                logger.error(f"Error in stability test iteration {i}: {e}")
                
        return pd.DataFrame(metrics_history)
