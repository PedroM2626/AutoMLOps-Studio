import pandas as pd
import numpy as np
import logging
import re
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

logger = logging.getLogger(__name__)

class AutoMLDataProcessor:
    def __init__(self, target_column=None, task_type=None, data_type='tabular', date_col=None, forecast_horizon=1, nlp_config=None, scaler_type='standard', semi_supervised=False, enable_dfs=False, dfs_depth=1):
        self.target_column = target_column
        self.task_type = task_type
        self.data_type = data_type
        self.date_col = date_col
        self.forecast_horizon = forecast_horizon
        self.nlp_config = nlp_config if nlp_config else {}
        self.scaler_type = scaler_type
        self.preprocessor = None
        self.nlp_cols = []
        self.is_time_series = (data_type == 'sequential')
        self.semi_supervised = semi_supervised
        self.enable_dfs = enable_dfs
        self.dfs_depth = dfs_depth

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
            if not target_vals_numeric.isna().all():
                target_vals = target_vals_numeric
                if self.target_column and self.target_column in df.columns:
                    df[self.target_column] = target_vals
                
                for i in range(self.forecast_horizon, self.forecast_horizon + 5):
                    df[f'lag_{i}'] = target_vals.shift(i)
                
                df[f'rolling_mean_{self.forecast_horizon}'] = target_vals.shift(self.forecast_horizon).rolling(window=3).mean()
                df[f'rolling_std_{self.forecast_horizon}'] = target_vals.shift(self.forecast_horizon).rolling(window=3).std()
                df = df.dropna()
            
        return df

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
            if X_to_process[col].nunique() <= 15:
                low_card_features.append(col)
            else:
                high_card_features.append(col)

        if self.scaler_type == 'minmax':
            scaler = MinMaxScaler()
        elif self.scaler_type == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', scaler)
        ])

        low_card_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
        ])
        
        high_card_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
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
            from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
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
                except:
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
