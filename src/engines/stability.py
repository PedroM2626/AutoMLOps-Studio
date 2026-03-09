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
        if base_model is None:
            raise ValueError("StabilityAnalyzer requires a non-None base_model.")
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
                import traceback
                error_msg = f"Error in seed stability iteration {i}: {e}\n{traceback.format_exc()}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
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
                import traceback
                error_msg = f"Error in split stability iteration {i}: {e}\n{traceback.format_exc()}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
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

    def run_noise_injection_stability(self, noise_level=0.05, n_iterations=5):
        """
        Test stability against random noise (Gaussian for numeric, random flipping for categorical).
        noise_level: Fraction of standard deviation for numeric noise, or fraction of rows to flip for categorical.
        """
        metrics_history = []
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state
        )
        
        # Identify types
        is_df = isinstance(X_val, pd.DataFrame)
        numeric_cols = []
        cat_cols = []
        if is_df:
            numeric_cols = X_val.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = X_val.select_dtypes(exclude=[np.number]).columns.tolist()
        else:
            # Assume all numeric if it's a numpy array
            numeric_cols = list(range(X_val.shape[1]))
            
        std_devs = None
        if len(numeric_cols) > 0:
            std_devs = X_val[numeric_cols].std() if is_df else np.std(X_val, axis=0)
            # handle zero std
            if is_df:
                std_devs = std_devs.replace(0, 1e-6)
            else:
                std_devs[std_devs == 0] = 1e-6
                
        # Train base model once
        model = clone(self.base_model)
        if hasattr(model, 'random_state'): model.set_params(random_state=self.random_state)
        elif hasattr(model, 'random_seed'): model.set_params(random_seed=self.random_state)
        model.fit(X_train, y_train)

        # Baseline (No Noise)
        try:
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
            baseline_metrics = self._get_metrics(y_val, y_pred, y_proba)
            baseline_metrics['iteration'] = -1
            baseline_metrics['noise_type'] = 'baseline'
            metrics_history.append(baseline_metrics)
        except Exception as e:
            logger.error(f"Error calculating baseline metrics for noise stability: {e}")
            metrics_history.append({'iteration': -1, 'noise_type': 'baseline', 'error': str(e)})
        
        for i in range(n_iterations):
            np.random.seed(self.random_state + i)
            X_val_noisy = X_val.copy() if is_df else np.copy(X_val)
            
            # Numeric noise
            if len(numeric_cols) > 0:
                noise = np.random.normal(0, std_devs * noise_level, size=(X_val.shape[0], len(numeric_cols)))
                if is_df:
                    X_val_noisy[numeric_cols] = X_val_noisy[numeric_cols] + noise
                else:
                    X_val_noisy += noise
                    
            # Categorical noise (flipping)
            if is_df and len(cat_cols) > 0:
                for col in cat_cols:
                    unique_vals = X_val[col].dropna().unique()
                    if len(unique_vals) > 1:
                        mask = np.random.rand(len(X_val_noisy)) < noise_level
                        mask_sum = np.sum(mask)
                        # Assign random categories to the masked rows
                        random_cats = np.random.choice(unique_vals, size=mask_sum)
                        X_val_noisy.loc[mask, col] = random_cats
                        
            # Predict on noisy data
            y_pred_noisy = model.predict(X_val_noisy)
            y_proba_noisy = model.predict_proba(X_val_noisy) if hasattr(model, "predict_proba") else None
            
            iter_metrics = self._get_metrics(y_val, y_pred_noisy, y_proba_noisy)
            iter_metrics['iteration'] = i
            iter_metrics['noise_type'] = f'noise_level_{noise_level}'
            metrics_history.append(iter_metrics)
            
        return pd.DataFrame(metrics_history)

    def run_slice_stability(self, slice_column):
        """
        Evaluate model performance across different groups (slices) of a categorical feature to test Fairness/Bias.
        """
        if not isinstance(self.X, pd.DataFrame):
            raise ValueError("Slice stability requires X to be a pandas DataFrame.")
        if slice_column not in self.X.columns:
            raise ValueError(f"Column '{slice_column}' not found in X.")
            
        metrics_history = []
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state
        )
        
        # Train once
        model = clone(self.base_model)
        if hasattr(model, 'random_state'): model.set_params(random_state=self.random_state)
        elif hasattr(model, 'random_seed'): model.set_params(random_seed=self.random_state)
        model.fit(X_train, y_train)

        # Baseline (Overall)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
        baseline_metrics = self._get_metrics(y_val, y_pred, y_proba)
        baseline_metrics['slice'] = 'OVERALL (Baseline)'
        metrics_history.append(baseline_metrics)
        
        unique_slices = X_val[slice_column].unique()
        
        for val in unique_slices:
            if pd.isna(val):
                mask = X_val[slice_column].isna()
            else:
                mask = X_val[slice_column] == val
                
            mask_sum = np.sum(mask)
            if mask_sum < 5:
                # Skip slices that are too small
                continue
                
            X_slice = X_val[mask]
            y_slice = y_val[mask]
            
            y_pred_slice = model.predict(X_slice)
            y_proba_slice = model.predict_proba(X_slice) if hasattr(model, "predict_proba") else None
            
            slice_metrics = self._get_metrics(y_slice, y_pred_slice, y_proba_slice)
            slice_metrics['slice'] = str(val)
            slice_metrics['support'] = int(mask_sum)
            metrics_history.append(slice_metrics)
            
        df = pd.DataFrame(metrics_history)
        # Reorder to make slice first
        cols = ['slice'] + [c for c in df.columns if c != 'slice']
        return df[cols]

    def run_missing_value_robustness(self, missing_fractions=[0.05, 0.10, 0.20]):
        """
        Randomly drops values (NaNs) in the evaluation set to test imputation resilience.
        """
        metrics_history = []
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state
        )
        
        # Train once
        model = clone(self.base_model)
        if hasattr(model, 'random_state'): model.set_params(random_state=self.random_state)
        elif hasattr(model, 'random_seed'): model.set_params(random_seed=self.random_state)
        model.fit(X_train, y_train)

        # Baseline
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
        baseline_metrics = self._get_metrics(y_val, y_pred, y_proba)
        baseline_metrics['missing_fraction'] = 0.0
        metrics_history.append(baseline_metrics)
        
        is_df = isinstance(X_val, pd.DataFrame)
        
        for frac in missing_fractions:
            np.random.seed(self.random_state)
            X_val_missing = X_val.copy() if is_df else np.copy(X_val)
            
            # Create a random mask for the entire matrix
            # Only apply to numeric columns to avoid breaking text pipelines completely
            target_cols = []
            if is_df:
                target_cols = [c for c in X_val.columns if pd.api.types.is_numeric_dtype(X_val[c])]
            else:
                target_cols = list(range(X_val.shape[1]))
            
            if len(target_cols) > 0:
                 if is_df:
                     mask = np.random.rand(X_val.shape[0], len(target_cols)) < frac
                     # Use numpy assignment for speed
                     vals = X_val_missing[target_cols].values
                     # Cast vals to float so np.nan can be assigned without breaking ints
                     vals = vals.astype(float)
                     vals[mask] = np.nan
                     X_val_missing[target_cols] = vals
                 else:
                     X_val_missing = X_val_missing.astype(float)
                     mask = np.random.rand(*X_val.shape) < frac
                     X_val_missing[mask] = np.nan

            try:
                y_pred_miss = model.predict(X_val_missing)
                y_proba_miss = model.predict_proba(X_val_missing) if hasattr(model, "predict_proba") else None
                
                iter_metrics = self._get_metrics(y_val, y_pred_miss, y_proba_miss)
                iter_metrics['missing_fraction'] = frac
                metrics_history.append(iter_metrics)
            except Exception as e:
                # The model/pipeline lacks imputation logic
                 iter_metrics = {'missing_fraction': frac, 'error': str(e)}
                 metrics_history.append(iter_metrics)
                 
        return pd.DataFrame(metrics_history)

    def run_calibration_stability(self, n_splits=5):
        """
        Evaluate Brier Score and calibration across CV splits (Classification only).
        """
        if self.task_type != 'classification':
            raise ValueError("Calibration stability is only for classification tasks.")
            
        from sklearn.metrics import brier_score_loss
            
        metrics_history = []
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        
        # Check if binary
        is_binary = len(np.unique(self.y)) == 2
        
        for i, (train_idx, val_idx) in enumerate(cv.split(self.X, self.y)):
            if isinstance(self.X, pd.DataFrame):
                X_train, X_val = self.X.iloc[train_idx], self.X.iloc[val_idx]
                y_train, y_val = self.y.iloc[train_idx], self.y.iloc[val_idx]
            else:
                X_train, X_val = self.X[train_idx], self.X[val_idx]
                y_train, y_val = self.y[train_idx], self.y[val_idx]
                
            model = clone(self.base_model)
            if hasattr(model, 'random_state'): model.set_params(random_state=self.random_state)
            
            model.fit(X_train, y_train)
            
            # Check for predict_proba, handling Pipelines
            has_proba = hasattr(model, "predict_proba")
            if not has_proba and hasattr(model, "steps"):
                has_proba = hasattr(model.steps[-1][1], "predict_proba")
                
            if not has_proba:
                logger.warning("Model does not support predict_proba. Cannot calculate Brier score.")
                return pd.DataFrame([{'split': 'N/A', 'brier_score': None, 'error': 'predict_proba() unsupported by this model type.'}])
                
            y_proba = model.predict_proba(X_val)
            
            iter_metrics = {'split': i}
            if is_binary:
                iter_metrics['brier_score'] = brier_score_loss(y_val, y_proba[:, 1])
            else:
                # Multiclass Brier score is the sum of binary Brier scores over classes
                brier_sum = 0
                for c in range(y_proba.shape[1]):
                    # Binarize label
                    y_val_bin = (y_val == model.classes_[c]).astype(int)
                    brier_sum += brier_score_loss(y_val_bin, y_proba[:, c])
                iter_metrics['brier_score'] = brier_sum / y_proba.shape[1]
                
            metrics_history.append(iter_metrics)
            
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

    def run_nlp_robustness(self, n_iterations=3, typo_probability=0.1, X_raw=None, transform_func=None):
        """
        Simulates typos and character dropping in text columns to test NLP robustness.
        """
        metrics_history = []
        X_to_use = X_raw if X_raw is not None else self.X
        is_df = isinstance(X_to_use, pd.DataFrame)
        
        if not is_df:
            return pd.DataFrame([{'iteration': -1, 'error': 'NLP Robustness requires a Pandas DataFrame with explicit text columns.'}])
            
        # Ensure X_to_use and self.y have the same number of rows
        if len(X_to_use) != len(self.y):
            logger.warning(f"Length mismatch in run_nlp_robustness: X_raw ({len(X_to_use)}) != y ({len(self.y)}). Attempting to align...")
            if len(X_to_use) > len(self.y):
                # This often happens if the processor dropped some rows (e.g. NaNs)
                # If X_to_use is a DataFrame, we try to match indices if self.y is a Series
                if isinstance(self.y, (pd.Series, pd.DataFrame)):
                    X_to_use = X_to_use.loc[self.y.index]
                else:
                    # If self.y is a numpy array, we assume the first N rows correspond (risky but better than crashing)
                    X_to_use = X_to_use.iloc[:len(self.y)]
            else:
                return pd.DataFrame([{'iteration': -1, 'error': f'Inconsistent number of samples: X={len(X_to_use)}, y={len(self.y)}'}])

        object_cols = [c for c in X_to_use.columns if X_to_use[c].dtype == 'object' or X_to_use[c].dtype.name == 'category']
        
        # We consider all string/object columns as valid targets for typo injection
        text_cols = list(object_cols)
                
        if len(text_cols) == 0:
            return pd.DataFrame([{'iteration': -1, 'error': 'No suitable text columns detected for NLP Robustness.'}])

        X_train_raw, X_val_raw, y_train, y_val = train_test_split(
            X_to_use, self.y, test_size=0.2, random_state=self.random_state
        )
        
        X_train, X_val, _, _ = train_test_split(
            self.X, self.y, test_size=0.2, random_state=self.random_state
        )
        
        model = clone(self.base_model)
        if hasattr(model, 'random_state'): model.set_params(random_state=self.random_state)
        elif hasattr(model, 'random_seed'): model.set_params(random_seed=self.random_state)
        
        model.fit(X_train, y_train)

        # Baseline
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None
        baseline_metrics = self._get_metrics(y_val, y_pred, y_proba)
        baseline_metrics['iteration'] = -1
        baseline_metrics['noise_type'] = 'None'
        metrics_history.append(baseline_metrics)
        
        def inject_typos(text, p=0.1):
            if not isinstance(text, str):
                return text
            chars = list(text)
            for i in range(len(chars)):
                if np.random.rand() < p and chars[i].isalpha():
                    # random swap with another letter or drop
                    action = np.random.choice(['drop', 'swap'])
                    if action == 'swap':
                        chars[i] = chr(np.random.randint(97, 123)) # a-z
                    else:
                        chars[i] = ""
            return "".join(chars)

        for i in range(n_iterations):
            np.random.seed(self.random_state + i)
            X_val_noisy_raw = X_val_raw.copy()
            
            for c in text_cols:
                X_val_noisy_raw[c] = X_val_noisy_raw[c].apply(lambda x: inject_typos(x, p=typo_probability))
                
            if transform_func is not None:
                try:
                    out = transform_func(X_val_noisy_raw)
                    if isinstance(out, tuple) and len(out) == 2:
                        X_val_noisy = out[0]
                    else:
                        X_val_noisy = out
                except Exception as e:
                    return pd.DataFrame([{'iteration': i, 'error': f'Preprocessor transform failed on noisy data: {e}'}])
            else:
                X_val_noisy = X_val_noisy_raw
                
            y_pred_noisy = model.predict(X_val_noisy)
            y_proba_noisy = model.predict_proba(X_val_noisy) if hasattr(model, "predict_proba") else None
            
            iter_metrics = self._get_metrics(y_val, y_pred_noisy, y_proba_noisy)
            iter_metrics['iteration'] = i
            iter_metrics['noise_type'] = f'Text Typos {typo_probability*100}%'
            metrics_history.append(iter_metrics)
            
        return pd.DataFrame(metrics_history)

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
