import pandas as pd
import numpy as np
import logging
import re
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler, QuantileTransformer,
    PowerTransformer, OneHotEncoder, LabelEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

# Map of user-facing scaler identifiers to sklearn scaler factories.
# 'auto' keeps the historical default (StandardScaler).
SCALER_REGISTRY = {
    'auto': lambda: StandardScaler(),
    'standard': lambda: StandardScaler(),
    'minmax': lambda: MinMaxScaler(),
    'robust': lambda: RobustScaler(),
    'maxabs': lambda: MaxAbsScaler(),
    'quantile': lambda: QuantileTransformer(output_distribution='normal', random_state=42),
    'power': lambda: PowerTransformer(method='yeo-johnson'),
}


def build_scaler(scaler_type, sparse_input=False):
    """Builds a scaler instance from a string identifier.

    When the input is a sparse matrix, scalers that require densification or
    centering (standard/quantile/power) are swapped for sparse-safe variants.
    Returns None when scaling is disabled.
    """
    key = (scaler_type or 'auto').lower()
    if key in ('none', 'off', 'disabled'):
        return None
    if sparse_input and key in ('standard', 'auto'):
        return StandardScaler(with_mean=False)
    if sparse_input and key in ('quantile', 'power'):
        # These estimators do not support sparse input; fall back gracefully.
        logger.warning(f"Scaler '{key}' does not support sparse matrices. Falling back to MaxAbsScaler.")
        return MaxAbsScaler()
    factory = SCALER_REGISTRY.get(key, SCALER_REGISTRY['auto'])
    return factory()


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clips numeric features to [lower_quantile, upper_quantile] bounds learned at fit time."""

    def __init__(self, lower_q=0.01, upper_q=0.99):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        X_arr = X.toarray() if hasattr(X, 'toarray') else np.asarray(X, dtype=float)
        self.lower_bounds_ = np.nanquantile(X_arr, self.lower_q, axis=0)
        self.upper_bounds_ = np.nanquantile(X_arr, self.upper_q, axis=0)
        return self

    def transform(self, X):
        if hasattr(X, 'toarray'):
            X = X.toarray()
        X_arr = np.asarray(X, dtype=float)
        return np.clip(X_arr, self.lower_bounds_, self.upper_bounds_)

    def get_feature_names_out(self, input_features=None):
        # Clipping does not change the feature set — passthrough names.
        return np.asarray(input_features)


class AutoMLDataProcessor:
    def __init__(self, target_column=None, task_type=None, data_type='tabular', date_col=None, forecast_horizon=1, nlp_config=None, scaler_type='auto', semi_supervised=False, enable_dfs=False, dfs_depth=1, impute_strategy='median', impute_fill_value=0.0, encoding_mode='auto', onehot_cardinality_threshold=15, clip_outliers=False, outlier_lower_q=0.01, outlier_upper_q=0.99, ts_clustering_config=None):
        self.target_column = target_column
        self.task_type = task_type
        self.data_type = data_type
        self.date_col = date_col
        self.forecast_horizon = forecast_horizon
        self.nlp_config = nlp_config if nlp_config else {}
        self.scaler_type = scaler_type or 'auto'
        self.preprocessor = None
        self.nlp_cols = []
        self.is_time_series = (data_type == 'sequential')
        self.semi_supervised = semi_supervised
        self.enable_dfs = enable_dfs
        self.dfs_depth = dfs_depth
        # Customizable preprocessing knobs
        self.impute_strategy = impute_strategy or 'median'
        self.impute_fill_value = impute_fill_value
        self.encoding_mode = (encoding_mode or 'auto').lower()
        self.onehot_cardinality_threshold = int(onehot_cardinality_threshold or 15)
        self.clip_outliers = bool(clip_outliers)
        self.outlier_lower_q = float(outlier_lower_q)
        self.outlier_upper_q = float(outlier_upper_q)
        self.ts_clustering_config = ts_clustering_config or {}
        self._ts_target_encoder = None

    def _resolve_target_columns(self, df):
        """Resolve target column(s) present in the given DataFrame."""
        if isinstance(self.target_column, (list, tuple)):
            return [c for c in self.target_column if c in df.columns]
        if isinstance(self.target_column, str) and self.target_column in df.columns:
            return [self.target_column]
        return []

    def _clean_text_feature(self, df, col):
        """Applies text cleaning to a specific column in DataFrame."""
        if col in df.columns:
            cleaning_mode = self.nlp_config.get('cleaning_mode', 'standard')
            if cleaning_mode == 'none':
                # Raw mode: keep the text untouched for vectorizers/models
                # that perform their own tokenization.
                return df
            logger.info(f"Cleaning text from column: {col} (Mode: {cleaning_mode})")
            
            def clean_text_optimized(text):
                text = str(text).lower()
                text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
                text = re.sub(r'\@\w+|\#','', text)
                
                if cleaning_mode == 'god_mode':
                    text = re.sub(r'(.)\1+', r'\1\1', text)
                    text = re.sub(r'[^a-z\s\!\?]', '', text)
                else:
                    text = re.sub(r'[^a-z\s]', '', text)
                
                text = " ".join(text.split())
                return text
            
            df[col] = df[col].apply(clean_text_optimized)
        return df

    def _apply_ts_features(self, df, y=None):
        """Applies time-series specific feature engineering to a DataFrame."""
        df = df.copy()
        if self.date_col and self.date_col in df.columns:
            try:
                df[self.date_col] = pd.to_datetime(df[self.date_col])
                df['hour'] = df[self.date_col].dt.hour
                df['dayofweek'] = df[self.date_col].dt.dayofweek
                df['quarter'] = df[self.date_col].dt.quarter
                df['month'] = df[self.date_col].dt.month
                df['year'] = df[self.date_col].dt.year
                df['dayofyear'] = df[self.date_col].dt.dayofyear
                df['dayofmonth'] = df[self.date_col].dt.day
                df['weekofyear'] = df[self.date_col].dt.isocalendar().week.astype(int)
            except Exception as e:
                logger.warning(f"Could not extract temporal features: {e}")

        target_vals = None
        if y is not None:
            target_vals = y
        elif self.target_column and self.target_column in df.columns:
            target_vals = df[self.target_column]
            
        if target_vals is not None:
            target_vals_numeric = pd.to_numeric(target_vals, errors='coerce')
            if target_vals_numeric.isna().all() and not pd.isna(target_vals).all():
                # Categorical target (e.g. Forecast Classification / TS Classification):
                # encode labels consistently so lag features can be derived from them.
                try:
                    if self._ts_target_encoder is None:
                        self._ts_target_encoder = LabelEncoder()
                        clean_vals = target_vals.dropna().astype(str)
                        self._ts_target_encoder.fit(clean_vals)
                    target_vals_numeric = pd.Series(
                        self._ts_target_encoder.transform(target_vals.astype(str)),
                        index=target_vals.index
                    )
                except Exception as e:
                    logger.warning(f"Could not encode categorical target for TS lags: {e}")
                    target_vals_numeric = None
            if target_vals_numeric is not None and not target_vals_numeric.isna().all():
                target_vals = target_vals_numeric
                if self.target_column and self.target_column in df.columns and pd.to_numeric(df[self.target_column], errors='coerce').isna().all():
                    # Only overwrite when the column was purely categorical
                    df[self.target_column] = target_vals
                
                for i in range(self.forecast_horizon, self.forecast_horizon + 5):
                    df[f'lag_{i}'] = target_vals.shift(i)
                
                df[f'rolling_mean_{self.forecast_horizon}'] = target_vals.shift(self.forecast_horizon).rolling(window=3).mean()
                df[f'rolling_std_{self.forecast_horizon}'] = target_vals.shift(self.forecast_horizon).rolling(window=3).std()
                lag_rolling_cols = [c for c in df.columns if c.startswith(('lag_', 'rolling_'))]
                if lag_rolling_cols:
                    df = df.dropna(subset=lag_rolling_cols)
            
        return df

    def _apply_ts_windows(self, df):
        """Segments time series into sliding windows and extracts summary statistics.

        Used by the TS Clustering task: each output row describes one temporal
        window (regime) so that standard clustering algorithms can group
        similar behaviors together.
        """
        cfg = self.ts_clustering_config or {}
        window_size = max(2, int(cfg.get('window_size', 12)))
        step = max(1, int(cfg.get('step', 1)))
        series_col = cfg.get('series_col')

        df = df.copy()
        if series_col and series_col in df.columns:
            series_cols = [series_col]
        else:
            series_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
            if self.date_col and self.date_col in series_cols:
                series_cols.remove(self.date_col)

        if not series_cols:
            logger.warning("TS Clustering: no numeric series column found. Returning raw frame.")
            return df

        logger.info(f"TS Clustering: windowing {series_cols} (window={window_size}, step={step})")
        feature_frames = []
        for col in series_cols:
            s = pd.to_numeric(df[col], errors='coerce')
            r = s.rolling(window=window_size, step=step, min_periods=window_size)
            feats = pd.DataFrame(index=r.mean().index)
            feats[f'{col}_w_mean'] = r.mean()
            feats[f'{col}_w_std'] = r.std().fillna(0.0)
            feats[f'{col}_w_min'] = r.min()
            feats[f'{col}_w_max'] = r.max()
            feats[f'{col}_w_median'] = r.median()
            feats[f'{col}_w_skew'] = r.skew().fillna(0.0)
            # Trend within the window: current value minus the value one window ago
            feats[f'{col}_w_trend'] = s.loc[feats.index] - s.shift(window_size).loc[feats.index]
            feature_frames.append(feats.dropna())

        if not feature_frames:
            return df

        windowed = pd.concat(feature_frames, axis=1).dropna()
        # Keep the date column aligned when possible for traceability
        if self.date_col and self.date_col in df.columns:
            try:
                windowed[self.date_col] = df[self.date_col].loc[windowed.index].values
            except Exception:
                pass
        logger.info(f"TS Clustering: generated {windowed.shape[1]} window features over {windowed.shape[0]} segments.")
        return windowed


    def fit_transform(self, df, nlp_cols=None):
        self.nlp_cols = nlp_cols if nlp_cols else []
        self.quality_report_html = None
        try:
            from deepchecks.tabular import Dataset as DeepDataset
            from deepchecks.tabular.suites import data_integrity
            label = self.target_column if self.target_column in df.columns else None
            if len(df) > 10:
                logger.info("Running Data Integrity check with Deepchecks...")
                ds = DeepDataset(df, label=label, cat_features=df.select_dtypes(include=['object', 'category']).columns.tolist())
                integ_suite = data_integrity()
                suite_result = integ_suite.run(ds)
                # Fixed: Use lower case np.inf for NumPy 2.0 compatibility.
                # The warning in Deepchecks might still appear if the library uses the old alias internally.
                self.quality_report_html = suite_result.save_as_html(render_static=True)
                logger.info("Data Integrity check completed.")
        except Exception as e:
            logger.warning(f"Deepchecks failed: {e}")

        if self.nlp_cols:
            for col in self.nlp_cols:
                df = self._clean_text_feature(df, col)
                if col in df.columns:
                     df[col] = df[col].fillna("")

        # TS Clustering: convert the series into window-summary features first.
        # The task becomes an ordinary clustering problem over temporal regimes.
        if self.task_type == 'ts_clustering':
            df = self._apply_ts_windows(df)

        if self.is_time_series:
            df = self._apply_ts_features(df)

        target_cols = self._resolve_target_columns(df)
        if target_cols:
            X = df.drop(columns=target_cols)
            y = df[target_cols[0]] if len(target_cols) == 1 else df[target_cols].copy()
        else:
            X = df
            y = None
            
        if self.enable_dfs and not self.is_time_series and not self.nlp_cols:
            try:
                import featuretools as ft
                logger.info("Applying Deep Feature Synthesis (DFS)...")
                dfs_df = X.copy()
                es = ft.EntitySet(id="dataset")
                dfs_df['_dfs_id'] = range(len(dfs_df))
                es = es.add_dataframe(dataframe_name="data", dataframe=dfs_df, index="_dfs_id")
                
                feature_matrix, _ = ft.dfs(
                    entityset=es,
                    target_dataframe_name="data",
                    trans_primitives=['add_numeric', 'multiply_numeric', 'subtract_numeric'],
                    max_depth=self.dfs_depth,
                    features_only=False,
                    verbose=False
                )
                
                if '_dfs_id' in feature_matrix.columns:
                    feature_matrix = feature_matrix.drop(columns=['_dfs_id'])
                
                X = feature_matrix
                logger.info(f"DFS completed. New feature matrix shape: {X.shape}")
            except ImportError:
                logger.warning("DFS failed: 'featuretools' is not installed. Please add it to requirements.")
            except Exception as e:
                logger.warning(f"DFS failed: {e}")
        
        process_cols = [c for c in X.columns if c != self.date_col]
        nlp_features = [c for c in self.nlp_cols if c in process_cols]
        non_nlp_cols = [c for c in process_cols if c not in nlp_features]
        X_to_process = X[non_nlp_cols]
        
        numeric_features = X_to_process.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        all_categorical = X_to_process.select_dtypes(include=['object', 'category']).columns.tolist()

        constant_cols = [col for col in X_to_process.columns if X_to_process[col].nunique() <= 1]
        if constant_cols:
            numeric_features = [c for c in numeric_features if c not in constant_cols]
            all_categorical = [c for c in all_categorical if c not in constant_cols]
        
        low_card_features = []
        high_card_features = []
        for col in all_categorical:
            if X_to_process[col].nunique() <= self.onehot_cardinality_threshold:
                low_card_features.append(col)
            else:
                high_card_features.append(col)

        # Encoding mode overrides the automatic cardinality-based routing
        if self.encoding_mode == 'onehot':
            low_card_features = all_categorical
            high_card_features = []
        elif self.encoding_mode == 'ordinal':
            low_card_features = []
            high_card_features = all_categorical

        scaler = build_scaler(self.scaler_type)

        impute_strategies_numeric = ['mean', 'median', 'constant']
        num_impute_strategy = self.impute_strategy if self.impute_strategy in impute_strategies_numeric else 'median'
        num_imputer_kwargs = {'fill_value': self.impute_fill_value} if num_impute_strategy == 'constant' else {}

        numeric_steps = [('imputer', SimpleImputer(strategy=num_impute_strategy, **num_imputer_kwargs))]
        if self.clip_outliers:
            numeric_steps.append(('winsorizer', Winsorizer(lower_q=self.outlier_lower_q, upper_q=self.outlier_upper_q)))
        if scaler is not None:
            numeric_steps.append(('scaler', scaler))
        numeric_transformer = Pipeline(steps=numeric_steps)

        cat_impute_strategy = self.impute_strategy if self.impute_strategy in ['most_frequent', 'constant'] else 'most_frequent'
        cat_imputer_kwargs = {'fill_value': str(self.impute_fill_value)} if cat_impute_strategy == 'constant' else {}

        low_card_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=cat_impute_strategy, **cat_imputer_kwargs)),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
        ])
        
        high_card_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy=cat_impute_strategy, **cat_imputer_kwargs)),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        transformers = []
        if numeric_features:
            transformers.append(('num', numeric_transformer, numeric_features))
        if low_card_features:
            transformers.append(('cat_low', low_card_transformer, low_card_features))
        if high_card_features:
            transformers.append(('cat_high', high_card_transformer, high_card_features))

        if nlp_features:
            from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
            vectorizer_type = self.nlp_config.get('vectorizer', 'tfidf')
            ngram_range = self.nlp_config.get('ngram_range', (1, 3))
            max_features = self.nlp_config.get('max_features', 5000)
            
            # Optimization: reduce max_features if many NLP columns to prevent explosion
            effective_max_features = max_features
            if len(nlp_features) > 3:
                 effective_max_features = min(max_features, 2000)
            elif len(nlp_features) > 1:
                 effective_max_features = min(max_features, 3000)
            chosen_language = self.nlp_config.get('language', 'english').lower()
            stop_words = chosen_language if self.nlp_config.get('stop_words', True) else None
            
            for col in nlp_features:
                if vectorizer_type == 'embeddings':
                    try:
                        from sentence_transformers import SentenceTransformer
                        from sklearn.base import BaseEstimator, TransformerMixin
                        class STTransformer(BaseEstimator, TransformerMixin):
                            def __init__(self, model_name='all-MiniLM-L6-v2'):
                                self.model_name = model_name
                                self.model = None
                            def fit(self, X, y=None):
                                if self.model is None:
                                    self.model = SentenceTransformer(self.model_name)
                                return self
                            def transform(self, X):
                                texts = [str(t) for t in X]
                                return self.model.encode(texts, show_progress_bar=False)
                            def get_feature_names_out(self, input_features=None):
                                return [f"ST_emb_{i}" for i in range(384)]
                        vectorizer = STTransformer(model_name=self.nlp_config.get('embedding_model', 'all-MiniLM-L6-v2'))
                    except ImportError:
                        vectorizer = TfidfVectorizer(max_features=effective_max_features, ngram_range=ngram_range, stop_words=stop_words)
                elif vectorizer_type == 'count':
                    vectorizer = CountVectorizer(max_features=effective_max_features, ngram_range=ngram_range, stop_words=stop_words)
                elif vectorizer_type == 'binary':
                    # Binary Bag-of-Words: presence flags instead of raw counts
                    vectorizer = CountVectorizer(binary=True, max_features=effective_max_features, ngram_range=ngram_range, stop_words=stop_words)
                elif vectorizer_type == 'hashing':
                    # Feature hashing: stateless, fixed-width, memory-friendly for huge vocabularies.
                    # HashingVectorizer lacks get_feature_names_out in sklearn, so wrap it to keep
                    # the ColumnTransformer feature-name chain intact.
                    from sklearn.base import BaseEstimator, TransformerMixin

                    class HashingVectorizerAdapter(BaseEstimator, TransformerMixin):
                        def __init__(self, n_features=5000, ngram_range=(1, 3), stop_words=None):
                            self.n_features = n_features
                            self.ngram_range = ngram_range
                            self.stop_words = stop_words
                            self.vectorizer = None
                        def fit(self, X, y=None):
                            self.vectorizer = HashingVectorizer(
                                n_features=self.n_features, ngram_range=self.ngram_range,
                                stop_words=self.stop_words, alternate_sign=True
                            )
                            self.vectorizer.fit(X, y)
                            return self
                        def transform(self, X):
                            return self.vectorizer.transform(X)
                        def get_feature_names_out(self, input_features=None):
                            return np.array([f"hash_{i}" for i in range(self.n_features)])

                    vectorizer = HashingVectorizerAdapter(
                        n_features=effective_max_features, ngram_range=ngram_range, stop_words=stop_words
                    )
                elif vectorizer_type == 'passthrough':
                     def pass_text(x):
                         if hasattr(x, 'values'): x = x.values
                         if hasattr(x, 'to_numpy'): x = x.to_numpy()
                         return x.reshape(-1, 1)
                     vectorizer = FunctionTransformer(pass_text, validate=False)
                else:
                    is_god_mode = self.nlp_config.get('cleaning_mode') == 'god_mode'
                    vectorizer = TfidfVectorizer(
                        max_features=effective_max_features, ngram_range=ngram_range, stop_words=stop_words,
                        sublinear_tf=self.nlp_config.get('sublinear_tf', True),
                        strip_accents='unicode' if is_god_mode else None
                    )
                transformers.append((f'nlp_{col}', vectorizer, col))

        # Favor sparse if we have NLP or many features (prevents memory explosion)
        sparse_thresh = 1.0 if (nlp_features or len(transformers) > 5) else 0.3
        self.preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=sparse_thresh)
        X_processed = self.preprocessor.fit_transform(X)
        
        if not nlp_features and hasattr(X_processed, "toarray"):
            # Check if dense matrix would be too large (> 10 million elements)
            n_elements = X_processed.shape[0] * X_processed.shape[1]
            if n_elements < 10_000_000:
                X_processed = X_processed.toarray()
            else:
                logger.info(f"Keep sparse: Matrix size ({X_processed.shape}) too large for dense conversion.")
            
        y_processed = None
        if y is not None:
            if isinstance(y, pd.DataFrame) and y.shape[1] > 1:
                y_processed = y.to_numpy()
            else:
                y_series = y.iloc[:, 0] if isinstance(y, pd.DataFrame) else pd.Series(y)
                unlabeled_mask = y_series.isna() | (y_series == -1) | (y_series == '-1') | (y_series == '')
                
                if self.task_type == 'classification' and self.semi_supervised:
                    labeled_y = y_series[~unlabeled_mask]
                    self.label_encoder = LabelEncoder()
                    if len(labeled_y) > 0:
                        encoded_labeled = self.label_encoder.fit_transform(labeled_y)
                    else:
                        encoded_labeled = []
                    
                    y_processed = np.full(len(y_series), -1, dtype=int)
                    y_processed[~unlabeled_mask] = encoded_labeled
                else:
                    if y_series.dtype == 'object' or y_series.dtype.name == 'category':
                        self.label_encoder = LabelEncoder()
                        y_processed = self.label_encoder.fit_transform(y_series)
                    else:
                        y_processed = y_series.to_numpy()

        return X_processed, y_processed

    def transform(self, df):
        if self.nlp_cols:
            for col in self.nlp_cols:
                df = self._clean_text_feature(df, col)
                if col in df.columns:
                     df[col] = df[col].fillna("")

        if self.task_type == 'ts_clustering':
            df = self._apply_ts_windows(df)

        if self.is_time_series:
            df = self._apply_ts_features(df)

        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return None, None

        target_cols = self._resolve_target_columns(df)
        if target_cols:
            X = df.drop(columns=target_cols)
            y = df[target_cols[0]] if len(target_cols) == 1 else df[target_cols].copy()
            if not isinstance(y, pd.DataFrame) and hasattr(self, 'label_encoder') and self.label_encoder:
                try:
                    y = self.label_encoder.transform(y)
                except ValueError as e:
                    import logging
                    logging.getLogger(__name__).warning(f"Label encoder transform failed (unseen labels?): {e}. Returning raw labels.")
                except Exception:
                    pass
            elif isinstance(y, pd.DataFrame):
                y = y.to_numpy()
        else:
            X = df
            y = None
        
        if not isinstance(X, pd.DataFrame): X = pd.DataFrame(X)
        X_processed = self.preprocessor.transform(X)

        if not self.nlp_cols and hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()
            
        return X_processed, y

    def get_feature_names(self):
        if self.preprocessor is None: return []
        feature_names = []
        for name, transformer, columns in self.preprocessor.transformers_:
            if name == 'remainder' and transformer == 'drop': continue
            if hasattr(transformer, 'get_feature_names_out'):
                names = transformer.get_feature_names_out(columns)
                feature_names.extend(names)
            else:
                feature_names.extend(columns)
        return feature_names
