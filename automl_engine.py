import os
import logging
import importlib.util

# Reduce TensorFlow noise
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Check if transformers and torch are installed without importing them yet
# This prevents potential DLL conflicts/crashes during initial module load
try:
    _transformers_spec = importlib.util.find_spec("transformers")
    _torch_spec = importlib.util.find_spec("torch")
    TRANSFORMERS_AVAILABLE = (_transformers_spec is not None) and (_torch_spec is not None)
    if TRANSFORMERS_AVAILABLE:
        print("DEBUG: Transformers/Torch detected (lazy loading enabled).")
    else:
        print("DEBUG: Transformers/Torch not found.")
except Exception as e:
    TRANSFORMERS_AVAILABLE = False
    print(f"DEBUG: Error checking transformers availability: {e}")

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor, 
    GradientBoostingClassifier, GradientBoostingRegressor, 
    VotingClassifier, VotingRegressor, IsolationForest,
    ExtraTreesClassifier, ExtraTreesRegressor,
    AdaBoostClassifier, AdaBoostRegressor,
    StackingClassifier, StackingRegressor
)
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import (
    LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet, 
    SGDClassifier, SGDRegressor, RidgeClassifier, PassiveAggressiveClassifier
)
from sklearn.svm import SVC, SVR, OneClassSVM, LinearSVC
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.covariance import EllipticEnvelope
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, 
    mean_squared_error, mean_absolute_error, r2_score, confusion_matrix,
    silhouette_score, mean_absolute_percentage_error, balanced_accuracy_score,
    cohen_kappa_score, log_loss, matthews_corrcoef, explained_variance_score,
    median_absolute_error, mean_squared_log_error, calinski_harabasz_score,
    davies_bouldin_score
)
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, MeanShift, Birch, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
import xgboost as xgb
import lightgbm as lgb
import re
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
import optuna
import joblib
import os
import logging
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import cross_val_predict
from stability_engine import StabilityAnalyzer
import io
from PIL import Image

# Lazy load for optional libraries
def get_deepchecks_suite(task_type):
    try:
        from deepchecks.tabular.suites import data_integrity
        return data_integrity()
    except ImportError:
        return None


# Silence convergence warnings and other repetitive warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransformersWrapper(BaseEstimator, ClassifierMixin, RegressorMixin):
    def __init__(self, model_name='bert-base-uncased', task='classification', epochs=3, learning_rate=2e-5):
        self.model_name = model_name
        self.task = task
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = None
        self.tokenizer = None
        self.device = None # Lazy initialization to avoid early torch import

    def fit(self, X, y=None):
        logger.info(f"TransformersWrapper.fit called for {self.model_name}")
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not found.")
            
        try:
            logger.info("Importing torch/transformers...")
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            logger.info("Imports successful.")
        except ImportError as e:
            logger.error(f"Failed to import transformers/torch at runtime: {e}")
            raise e
            
        if self.device is None:
            logger.info("Checking CUDA availability...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Device selected: {self.device}")
        
        # Check if input is likely vectorized (sparse or dense matrix)
        # Handle 'passthrough' mode where X might be a numpy array of strings
        is_vectorized = False
        is_raw_text = False
        
        logger.info(f"Checking input type: {type(X)}")
        if hasattr(X, "shape"):
             logger.info(f"Checking input shape: {X.shape}")
             
             # If it's an array of strings/objects, it's raw text
             if X.dtype == 'object' or X.dtype.type is np.str_ or X.dtype.type is np.bytes_:
                 is_raw_text = True
                 logger.info("Input detected as Raw Text (dtype object/string).")
             elif len(X.shape) > 1 and X.shape[1] > 1:
                # Simple heuristic: if it has many columns, it's likely a feature matrix
                is_vectorized = True
                logger.info("Input detected as Vectorized Data.")
            
        if is_vectorized and not is_raw_text:
            logger.warning(f"TransformersWrapper ({self.model_name}) received vectorized data (shape {X.shape}). Skipping fine-tuning as it requires raw text.")
            
            # Use a lightweight dummy model or just leave as None and handle in predict
            self.model = None 
            logger.info("Skipping training due to vectorized input.")
            return self

        logger.info("Proceeding with training (raw text assumption)...")
        # If we somehow got text (e.g. custom pipeline), we would proceed here
        
        # Ensure X is a list of strings for tokenizer
        if hasattr(X, "tolist"):
             texts = X.tolist()
        else:
             texts = list(X)
             
        # Flatten if list of lists (common with reshape(-1, 1))
        if len(texts) > 0 and isinstance(texts[0], (list, np.ndarray)):
            texts = [t[0] for t in texts]
            
        # Convert to string just in case
        texts = [str(t) for t in texts]
        
        # For now, just load model to ensure connectivity
        num_labels = len(np.unique(y)) if y is not None and self.task == 'classification' else 1
        logger.info(f"Loading tokenizer and model {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels).to(self.device)
        
        # Mock Training Loop (Short)
        logger.info("Simulating training loop (1 batch)...")
        # In a real implementation, we would tokenize and train here.
        # inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        # ... training code ...
        
        logger.info(f"TransformersWrapper: Loaded {self.model_name} on {self.device}")
        return self

    def predict(self, X):
        if self.model is None:
            # If model wasn't trained (e.g. due to vectorized input), raise error 
            # so AutoMLTrainer catches it and assigns bad score, rather than random.
            raise RuntimeError("TransformersWrapper model is not trained (likely due to vectorized input).")
            
        # Mock prediction for interface testing
        return np.zeros(len(X)) if self.task == 'regression' else np.random.randint(0, 2, size=len(X))

class AutoMLDataProcessor:
    def __init__(self, target_column=None, task_type=None, date_col=None, forecast_horizon=1, nlp_config=None, scaler_type='standard'):
        self.target_column = target_column
        self.task_type = task_type
        self.date_col = date_col
        self.forecast_horizon = forecast_horizon
        self.nlp_config = nlp_config if nlp_config else {}
        self.scaler_type = scaler_type
        self.preprocessor = None
        self.nlp_cols = []

    def _clean_text_feature(self, df, col):
        """Applies text cleaning to a specific column in DataFrame."""
        if col in df.columns:
            cleaning_mode = self.nlp_config.get('cleaning_mode', 'standard')
            logger.info(f"Cleaning text from column: {col} (Mode: {cleaning_mode})")
            
            def clean_text_optimized(text):
                text = str(text).lower()
                # Remove URLs
                text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
                # Remove mentions and hashtags
                text = re.sub(r'\@\w+|\#','', text)
                
                if cleaning_mode == 'god_mode':
                    # Remove repetitive characters (e.g., goooood -> good)
                    text = re.sub(r'(.)\1+', r'\1\1', text)
                    # Keep punctuation that matters for sentiment (! and ?)
                    text = re.sub(r'[^a-z\s\!\?]', '', text)
                else:
                    # Standard cleaning: only letters and spaces
                    text = re.sub(r'[^a-z\s]', '', text)
                
                # Remove extra spaces
                text = " ".join(text.split())
                return text
            
            # Apply cleaning in-place (efficiently)
            df[col] = df[col].apply(clean_text_optimized)
        return df

    def _apply_ts_features(self, df, y=None):
        """Applies time-series specific feature engineering to a DataFrame."""
        df = df.copy()
        
        # 1. Temporal Features
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

        # 2. Lag Features & Rolling Stats
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

    def _apply_nlp_features(self, df, nlp_cols):
        """Deprecated: NLP features are now handled in the main pipeline."""
        return df

    def fit_transform(self, df, nlp_cols=None):
        self.nlp_cols = nlp_cols if nlp_cols else []
        
        # --- Data Quality Check (Deepchecks) ---
        self.quality_report_html = None
        try:
            from deepchecks.tabular import Dataset as DeepDataset
            from deepchecks.tabular.suites import data_integrity
            
            # Identify label for deepchecks
            label = self.target_column if self.target_column in df.columns else None
            
            # Simple check if data is large enough
            if len(df) > 10:
                logger.info("Running Data Integrity check with Deepchecks...")
                ds = DeepDataset(df, label=label, cat_features=df.select_dtypes(include=['object', 'category']).columns.tolist())
                integ_suite = data_integrity()
                suite_result = integ_suite.run(ds)
                
                # Save report as HTML string for UI
                self.quality_report_html = suite_result.save_as_html(render_static=True)
                logger.info("Data Integrity check completed.")
        except Exception as e:
            logger.warning(f"Deepchecks failed: {e}")

        # NLP Cleaning first
        if self.nlp_cols:
            for col in self.nlp_cols:
                df = self._clean_text_feature(df, col)
                # Fill NaNs for vectorizer
                if col in df.columns:
                     df[col] = df[col].fillna("")

        # Time Series Feature Engineering
        if self.task_type == 'time_series':
            df = self._apply_ts_features(df)

        if self.target_column and self.target_column in df.columns:
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
        else:
            X = df
            y = None
        
        # Exclude date_col from processing
        process_cols = [c for c in X.columns if c != self.date_col]
        
        nlp_features = [c for c in self.nlp_cols if c in process_cols]
        non_nlp_cols = [c for c in process_cols if c not in nlp_features]
        
        X_to_process = X[non_nlp_cols]
        
        # Identify column types
        numeric_features = X_to_process.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        all_categorical = X_to_process.select_dtypes(include=['object', 'category']).columns.tolist()

        # Drop constant columns
        constant_cols = [col for col in X_to_process.columns if X_to_process[col].nunique() <= 1]
        if constant_cols:
            numeric_features = [c for c in numeric_features if c not in constant_cols]
            all_categorical = [c for c in all_categorical if c not in constant_cols]
        
        # Split categorical
        low_card_features = []
        high_card_features = []
        
        for col in all_categorical:
            if X_to_process[col].nunique() <= 15:
                low_card_features.append(col)
            else:
                high_card_features.append(col)

        # Preprocessing Pipelines
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
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        high_card_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        # Bundle preprocessing
        transformers = []
        if numeric_features:
            transformers.append(('num', numeric_transformer, numeric_features))
        if low_card_features:
            transformers.append(('cat_low', low_card_transformer, low_card_features))
        if high_card_features:
            transformers.append(('cat_high', high_card_transformer, high_card_features))

        # NLP Transformers
        if nlp_features:
            from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
            
            vectorizer_type = self.nlp_config.get('vectorizer', 'tfidf')
            ngram_range = self.nlp_config.get('ngram_range', (1, 3))
            max_features = self.nlp_config.get('max_features', 5000) # Reduced from 20000 to 5000 for speed
            stop_words = 'english' if self.nlp_config.get('stop_words', True) else None
            
            for col in nlp_features:
                if vectorizer_type == 'embeddings':
                    # Support for Sentence-Transformers
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
                                # Ensure X is list of strings
                                texts = [str(t) for t in X]
                                return self.model.encode(texts, show_progress_bar=False)
                            def get_feature_names_out(self, input_features=None):
                                # Dummy feature names for embeddings (usually 384 for MiniLM)
                                return [f"ST_emb_{i}" for i in range(384)]
                        
                        vectorizer = STTransformer(model_name=self.nlp_config.get('embedding_model', 'all-MiniLM-L6-v2'))
                    except ImportError:
                        logger.warning("sentence-transformers not installed. Falling back to TF-IDF.")
                        vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range, stop_words=stop_words)
                
                elif vectorizer_type == 'count':
                    vectorizer = CountVectorizer(
                        max_features=max_features,
                        ngram_range=ngram_range,
                        stop_words=stop_words
                    )
                elif vectorizer_type == 'passthrough':
                     # Custom transformer to pass text as-is (for BERT)
                     from sklearn.preprocessing import FunctionTransformer
                     # Ensure we handle Series/DataFrame input correctly for FunctionTransformer
                     # We need to return a 2D array (n_samples, 1) or list of strings
                     def pass_text(x):
                         if hasattr(x, 'values'):
                             x = x.values
                         if hasattr(x, 'to_numpy'):
                             x = x.to_numpy()
                         return x.reshape(-1, 1)
                         
                     vectorizer = FunctionTransformer(pass_text, validate=False)
                else:
                    # Apply specific settings for god_mode if requested, or user config
                    is_god_mode = self.nlp_config.get('cleaning_mode') == 'god_mode'
                    
                    vectorizer = TfidfVectorizer(
                        max_features=max_features,
                        ngram_range=ngram_range,
                        stop_words=stop_words,
                        sublinear_tf=self.nlp_config.get('sublinear_tf', True), # Default True for TF-IDF usually good
                        strip_accents='unicode' if is_god_mode else None
                    )
                transformers.append((f'nlp_{col}', vectorizer, col))

        # Sparse Threshold - IMPORTANT: Keep sparse for NLP
        sparse_thresh = 1.0 if nlp_features else 0
        self.preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=sparse_thresh)

        try:
            X_processed = self.preprocessor.fit_transform(X)
        except Exception as e:
            logger.error(f"Error in fit_transform: {e}")
            raise e
        
        if X_processed.shape[0] == 0:
            raise ValueError("Processing resulted in 0 rows.")
            
        # Ensure output is dense only if NO NLP features (to save memory)
        if not nlp_features and hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()
            
        # Handle target encoding
        y_processed = None
        if y is not None:
            if y.dtype == 'object' or y.dtype.name == 'category':
                self.label_encoder = LabelEncoder()
                y_processed = self.label_encoder.fit_transform(y)
            else:
                y_processed = y

        return X_processed, y_processed

    def transform(self, df):
        # NLP Cleaning
        if self.nlp_cols:
            for col in self.nlp_cols:
                df = self._clean_text_feature(df, col)
                if col in df.columns:
                     df[col] = df[col].fillna("")

        # Time Series Feature Engineering
        if self.task_type == 'time_series':
            df = self._apply_ts_features(df)

        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return None, None

        if self.target_column and self.target_column in df.columns:
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
            # Handle categorical target
            if hasattr(self, 'label_encoder') and self.label_encoder:
                try:
                    y = self.label_encoder.transform(y)
                except:
                    pass 
        else:
            X = df
            y = None
        
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        try:
            X_processed = self.preprocessor.transform(X)
        except Exception as e:
            logger.error(f"Error in ColumnTransformer.transform: {e}")
            raise e

        # Ensure dense only if NO NLP features
        if not self.nlp_cols and hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()
            
        return X_processed, y

    def get_feature_names(self):
        """Returns the names of the features after preprocessing."""
        if self.preprocessor is None:
            return []
        
        feature_names = []
        for name, transformer, columns in self.preprocessor.transformers_:
            if name == 'remainder' and transformer == 'drop':
                continue
            
            if hasattr(transformer, 'get_feature_names_out'):
                # For OneHotEncoder and others that change the number of columns
                names = transformer.get_feature_names_out(columns)
                feature_names.extend(names)
            else:
                # For StandardScaler, SimpleImputer, etc. that keep the number of columns
                feature_names.extend(columns)
        
        return feature_names

class AutoMLTrainer:
    def __init__(self, task_type='classification', preset='medium', ensemble_config=None):
        self.task_type = task_type
        self.preset = preset
        self.ensemble_config = ensemble_config or {}
        self.best_model = None
        self.best_params = None
        self.results = []
        
        # Configurations based on preset
        self.preset_configs = {
            'fast': {
                'n_trials': 15,
                'timeout': 600,  # 10 min
                'cv': 3,
                'models': ['logistic_regression', 'random_forest', 'xgboost', 'decision_tree', 'ridge_classifier']
            },
            'medium': {
                'n_trials': 40,
                'timeout': 1800, # 30 min
                'cv': 5,
                'models': ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'extra_trees', 'svm', 'knn', 'mlp']
            },
            'high': {
                'n_trials': 3, # Reduced to avoid long wait times
                'timeout': 3600, # 1 hour
                'cv': 5, # Reduced from 10 to 5 for speed
                # Use a representative subset for interface simulation speed
                'models': ['voting_ensemble', 'sgd_classifier', 'bert-base-uncased', 'xgboost']
                # Full list: ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'catboost', 'svm', 'mlp', 'extra_trees', 'adaboost', 'sgd_classifier', 'passive_aggressive', 'bert-base-uncased', 'distilbert-base-uncased', 'roberta-base']
            },
            'custom': {
                'n_trials': 20, # Default if not provided manually
                'timeout': 600,
                'cv': 5,
                'models': ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'extra_trees'] # Default selection
            },
            'test': {
                'n_trials': 1, # Minimal for speed
                'timeout': 30, # 30 seconds max
                'cv': 2, # Minimal folds
                # Removed MLP because it is slow
                'models': ['logistic_regression', 'decision_tree'] 
            }
        }

    def _get_default_model(self, name, random_state=42):
        """Returns a default instance of a model without Optuna optimization."""
        # Common models for ensembles
        if self.task_type == 'classification':
            if name == 'logistic_regression': return LogisticRegression(max_iter=1000, random_state=random_state)
            if name == 'random_forest': return RandomForestClassifier(n_estimators=100, random_state=random_state)
            if name == 'xgboost': return xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
            if name == 'lightgbm': return lgb.LGBMClassifier(random_state=random_state)
            if name == 'extra_trees': return ExtraTreesClassifier(n_estimators=100, random_state=random_state)
            if name == 'decision_tree': return DecisionTreeClassifier(random_state=random_state)
            if name == 'svm': return SVC(probability=True, random_state=random_state)
            if name == 'linear_svc': return LinearSVC(max_iter=1000, dual=False, random_state=random_state)
            if name == 'knn': return KNeighborsClassifier()
            if name == 'mlp': return MLPClassifier(max_iter=500, random_state=random_state)
            if name == 'sgd_classifier': return SGDClassifier(loss='modified_huber', random_state=random_state)
            if name == 'passive_aggressive': return PassiveAggressiveClassifier(random_state=random_state)
            if name == 'naive_bayes': return GaussianNB()
            if name == 'ridge_classifier': return RidgeClassifier(random_state=random_state)
            if name == 'adaboost': return AdaBoostClassifier(random_state=random_state)
            if name == 'catboost' and CATBOOST_AVAILABLE: return cb.CatBoostClassifier(verbose=0, thread_count=-1, random_seed=random_state)
        elif self.task_type == 'regression':
            if name == 'linear_regression': return LinearRegression()
            if name == 'random_forest': return RandomForestRegressor(n_estimators=100, random_state=random_state)
            if name == 'xgboost': return xgb.XGBRegressor(random_state=random_state)
            if name == 'lightgbm': return lgb.LGBMRegressor(random_state=random_state)
            if name == 'extra_trees': return ExtraTreesRegressor(n_estimators=100, random_state=random_state)
            if name == 'decision_tree': return DecisionTreeRegressor(random_state=random_state)
            if name == 'svm': return SVR()
            if name == 'knn': return KNeighborsRegressor()
            if name == 'mlp': return MLPRegressor(max_iter=500, random_state=random_state)
            if name == 'ridge': return Ridge(random_state=random_state)
            if name == 'lasso': return Lasso(random_state=random_state)
            if name == 'elastic_net': return ElasticNet(random_state=random_state)
            if name == 'sgd_regressor': return SGDRegressor(max_iter=1000, random_state=random_state)
            if name == 'adaboost': return AdaBoostRegressor(random_state=random_state)
            if name == 'catboost' and CATBOOST_AVAILABLE: return cb.CatBoostRegressor(verbose=0, thread_count=-1, random_seed=random_state)
        elif self.task_type == 'dimensionality_reduction':
            if name == 'pca': return PCA(random_state=random_state)
            if name == 'truncated_svd': return TruncatedSVD(random_state=random_state)
            
        return None

    def _resolve_estimators(self, estimators_config, random_state):
        if not estimators_config:
            return []
        
        # If it's a list of strings (names)
        if isinstance(estimators_config, list) and len(estimators_config) > 0 and isinstance(estimators_config[0], str):
            estimators = []
            for name in estimators_config:
                model = self._get_default_model(name, random_state)
                if model is not None:
                    estimators.append((name, model))
            return estimators
        
        # Assume it's already a list of (name, estimator) tuples
        return estimators_config

    def _get_models(self, trial=None, name=None, random_state=None):
        """
        Returns the list of model names or a specific instance with suggested parameters.
        """
        # Prioritize custom models if available
        if name and hasattr(self, 'custom_models') and name in self.custom_models:
            from sklearn.base import clone
            try:
                return clone(self.custom_models[name])
            except:
                import copy
                return copy.deepcopy(self.custom_models[name])

        if self.task_type == 'classification':
            models_config = {
                'voting_ensemble': lambda t: VotingClassifier(
                    estimators=[
                        ('pa', PassiveAggressiveClassifier(max_iter=1000, random_state=random_state, C=0.5)),
                        ('lr', LogisticRegression(max_iter=2000, C=10, solver='saga', n_jobs=-1, random_state=random_state)),
                        ('sgd', SGDClassifier(loss='modified_huber', max_iter=2000, n_jobs=-1, random_state=random_state))
                    ],
                    voting='hard',
                    n_jobs=-1 # Changed to 1 to avoid Windows multiprocessing issues
                ),
                'custom_voting': lambda t: VotingClassifier(
                    estimators=self._resolve_estimators(
                        self.ensemble_config.get('voting_estimators', [
                            ('lr', LogisticRegression(random_state=random_state)), 
                            ('rf', RandomForestClassifier(random_state=random_state))
                        ]),
                        random_state
                    ),
                    voting=self.ensemble_config.get('voting_type', 'soft'),
                    weights=self.ensemble_config.get('voting_weights', None),
                    n_jobs=-1
                ),
                'custom_stacking': lambda t: StackingClassifier(
                    estimators=self._resolve_estimators(
                        self.ensemble_config.get('stacking_estimators', [
                            ('rf', RandomForestClassifier(random_state=random_state)),
                            ('svm', SVC(probability=True, random_state=random_state))
                        ]),
                        random_state
                    ),
                    final_estimator=self._get_default_model(self.ensemble_config.get('stacking_final_estimator'), random_state) 
                                    if isinstance(self.ensemble_config.get('stacking_final_estimator'), str) 
                                    else self.ensemble_config.get('stacking_final_estimator', LogisticRegression(random_state=random_state)),
                    n_jobs=-1
                ),
                'logistic_regression': lambda t: LogisticRegression(
                    C=t.suggest_float('lr_C', 0.001, 100.0, log=True),
                    solver=t.suggest_categorical('lr_solver', ['lbfgs', 'liblinear', 'saga']),
                    max_iter=1000,
                    n_jobs=-1,
                    random_state=random_state
                ),
                'random_forest': lambda t: RandomForestClassifier(
                    n_estimators=t.suggest_int('rf_n_estimators', 100, 1000),
                    max_depth=t.suggest_int('rf_max_depth', 5, 100),
                    min_samples_split=t.suggest_int('rf_min_samples_split', 2, 20),
                    n_jobs=-1,
                    random_state=random_state
                ),
                'xgboost': lambda t: xgb.XGBClassifier(
                    n_estimators=t.suggest_int('xgb_n_estimators', 100, 2000),
                    learning_rate=t.suggest_float('xgb_lr', 0.0001, 0.5, log=True),
                    max_depth=t.suggest_int('xgb_max_depth', 3, 18),
                    subsample=t.suggest_float('xgb_subsample', 0.4, 1.0),
                    colsample_bytree=t.suggest_float('xgb_colsample', 0.4, 1.0),
                    use_label_encoder=False,
                    eval_metric='logloss',
                    n_jobs=-1,
                    random_state=random_state
                ),
                'lightgbm': lambda t: lgb.LGBMClassifier(
                    n_estimators=t.suggest_int('lgb_n_estimators', 100, 2000),
                    learning_rate=t.suggest_float('lgb_lr', 0.0001, 0.5, log=True),
                    num_leaves=t.suggest_int('lgb_leaves', 15, 255),
                    feature_fraction=t.suggest_float('lgb_feature_frac', 0.4, 1.0),
                    bagging_fraction=t.suggest_float('lgb_bagging_frac', 0.4, 1.0),
                    bagging_freq=t.suggest_int('lgb_bagging_freq', 1, 7),
                    verbosity=-1,
                    n_jobs=-1,
                    random_state=random_state
                ),
                'extra_trees': lambda t: ExtraTreesClassifier(
                    n_estimators=t.suggest_int('et_n_estimators', 100, 1000),
                    max_depth=t.suggest_int('et_max_depth', 5, 100),
                    n_jobs=-1,
                    random_state=random_state
                ),
                'adaboost': lambda t: AdaBoostClassifier(
                    n_estimators=t.suggest_int('ada_n_estimators', 50, 500),
                    learning_rate=t.suggest_float('ada_lr', 0.001, 2.0, log=True),
                    random_state=random_state
                ),
                'decision_tree': lambda t: DecisionTreeClassifier(
                    max_depth=t.suggest_int('dt_max_depth', 3, 50),
                    min_samples_split=t.suggest_int('dt_min_samples_split', 2, 20),
                    random_state=random_state
                ),
                'svm': lambda t: SVC(
                    C=t.suggest_float('svm_C', 0.001, 1000.0, log=True),
                    kernel=t.suggest_categorical('svm_kernel', ['linear', 'rbf', 'poly', 'sigmoid']),
                    gamma=t.suggest_categorical('svm_gamma', ['scale', 'auto']),
                    probability=False, 
                    cache_size=2000,
                    max_iter=1000,
                    random_state=random_state
                ),
                'linear_svc': lambda t: LinearSVC(
                    C=t.suggest_float('lsvc_C', 0.01, 100.0, log=True),
                    max_iter=1000, 
                    dual=False, 
                    random_state=random_state
                ),
                'knn': lambda t: KNeighborsClassifier(
                    n_neighbors=t.suggest_int('knn_neighbors', 1, 31),
                    weights=t.suggest_categorical('knn_weights', ['uniform', 'distance']),
                    metric=t.suggest_categorical('knn_metric', ['euclidean', 'manhattan', 'minkowski']),
                    n_jobs=-1
                ),
                'naive_bayes': lambda t: GaussianNB(
                    var_smoothing=t.suggest_float('nb_smoothing', 1e-10, 1e-8, log=True)
                ),
                'ridge_classifier': lambda t: RidgeClassifier(
                    alpha=t.suggest_float('ridge_alpha', 0.01, 10.0, log=True),
                    random_state=random_state
                ),
                'sgd_classifier': lambda t: SGDClassifier(
                    loss=t.suggest_categorical('sgd_loss', ['hinge', 'modified_huber']),
                    penalty=t.suggest_categorical('sgd_penalty', ['l2', 'l1', 'elasticnet']),
                    alpha=t.suggest_float('sgd_alpha', 1e-4, 1e-2, log=True),
                    max_iter=2000, 
                    random_state=random_state,
                    n_jobs=-1
                ),
                'passive_aggressive': lambda t: PassiveAggressiveClassifier(
                    C=t.suggest_float('pa_C', 0.001, 10.0, log=True),
                    fit_intercept=t.suggest_categorical('pa_fit_intercept', [True, False]),
                    max_iter=1000,
                    random_state=random_state,
                    n_jobs=-1
                ),
                'mlp': lambda t: MLPClassifier(
                    hidden_layer_sizes=t.suggest_categorical('mlp_layers', [(50,), (100,), (100, 50), (100, 100), (50, 50, 50), (256, 128, 64)]),
                    activation=t.suggest_categorical('mlp_activation', ['relu', 'tanh', 'logistic']),
                    solver=t.suggest_categorical('mlp_solver', ['adam', 'sgd']),
                    alpha=t.suggest_float('mlp_alpha', 1e-6, 1e-1, log=True),
                    learning_rate_init=t.suggest_float('mlp_lr', 1e-5, 1e-1, log=True),
                    max_iter=500, # Reduced from 1000 to 500 to prevent hangs, early_stopping already helps
                    early_stopping=True,
                    n_iter_no_change=10,
                    random_state=random_state
                ),
                'catboost': lambda t: cb.CatBoostClassifier(
                    iterations=t.suggest_int('cb_iterations', 100, 1000) if self.preset == 'best_quality' else t.suggest_int('cb_iterations', 50, 150),
                    learning_rate=t.suggest_float('cb_lr', 0.001, 0.3, log=True),
                    depth=t.suggest_int('cb_depth', 4, 10) if self.preset == 'best_quality' else t.suggest_int('cb_depth', 4, 6),
                    l2_leaf_reg=t.suggest_float('cb_l2', 1, 10),
                    verbose=0,
                    thread_count=-1,
                    random_seed=random_state
                ) if CATBOOST_AVAILABLE else None,
                'bert-base-uncased': lambda t: TransformersWrapper(model_name='bert-base-uncased', task='classification', learning_rate=t.suggest_float('learning_rate', 1e-6, 1e-3, log=True), epochs=t.suggest_int('num_train_epochs', 1, 20)) if TRANSFORMERS_AVAILABLE else None,
                'distilbert-base-uncased': lambda t: TransformersWrapper(model_name='distilbert-base-uncased', task='classification', learning_rate=t.suggest_float('learning_rate', 1e-6, 1e-3, log=True), epochs=t.suggest_int('num_train_epochs', 1, 20)) if TRANSFORMERS_AVAILABLE else None,
                'roberta-base': lambda t: TransformersWrapper(model_name='roberta-base', task='classification', learning_rate=t.suggest_float('learning_rate', 1e-6, 1e-3, log=True), epochs=t.suggest_int('num_train_epochs', 1, 20)) if TRANSFORMERS_AVAILABLE else None,
                'albert-base-v2': lambda t: TransformersWrapper(model_name='albert-base-v2', task='classification', learning_rate=t.suggest_float('learning_rate', 1e-6, 1e-3, log=True), epochs=t.suggest_int('num_train_epochs', 1, 20)) if TRANSFORMERS_AVAILABLE else None,
                'xlnet-base-cased': lambda t: TransformersWrapper(model_name='xlnet-base-cased', task='classification', learning_rate=t.suggest_float('learning_rate', 1e-6, 1e-3, log=True), epochs=t.suggest_int('num_train_epochs', 1, 20)) if TRANSFORMERS_AVAILABLE else None,
                'microsoft/deberta-v3-base': lambda t: TransformersWrapper(model_name='microsoft/deberta-v3-base', task='classification', learning_rate=t.suggest_float('learning_rate', 1e-6, 1e-3, log=True), epochs=t.suggest_int('num_train_epochs', 1, 20)) if TRANSFORMERS_AVAILABLE else None
            }
        elif self.task_type == 'regression':
            models_config = {
                'custom_voting': lambda t: VotingRegressor(
                    estimators=self._resolve_estimators(
                        self.ensemble_config.get('voting_estimators', [
                            ('lr', LinearRegression()), 
                            ('rf', RandomForestRegressor(random_state=random_state))
                        ]),
                        random_state
                    ),
                    weights=self.ensemble_config.get('voting_weights', None),
                    n_jobs=-1
                ),
                'custom_stacking': lambda t: StackingRegressor(
                    estimators=self._resolve_estimators(
                        self.ensemble_config.get('stacking_estimators', [
                            ('rf', RandomForestRegressor(random_state=random_state)),
                            ('svm', SVR())
                        ]),
                        random_state
                    ),
                    final_estimator=self._get_default_model(self.ensemble_config.get('stacking_final_estimator'), random_state)
                                    if isinstance(self.ensemble_config.get('stacking_final_estimator'), str)
                                    else self.ensemble_config.get('stacking_final_estimator', LinearRegression()),
                    n_jobs=-1
                ),
                'linear_regression': lambda t: LinearRegression(),
                'random_forest': lambda t: RandomForestRegressor(
                    n_estimators=t.suggest_int('rf_n_estimators', 50, 200),
                    max_depth=t.suggest_int('rf_max_depth', 3, 20),
                    random_state=random_state
                ),
                'xgboost': lambda t: xgb.XGBRegressor(
                    n_estimators=t.suggest_int('xgb_n_estimators', 50, 200),
                    learning_rate=t.suggest_float('xgb_lr', 0.01, 0.3),
                    random_state=random_state
                ),
                'lightgbm': lambda t: lgb.LGBMRegressor(
                    n_estimators=t.suggest_int('lgb_n_estimators', 50, 200),
                    learning_rate=t.suggest_float('lgb_lr', 0.01, 0.3),
                    verbosity=-1,
                    random_state=random_state
                ),
                'extra_trees': lambda t: ExtraTreesRegressor(
                    n_estimators=t.suggest_int('et_n_estimators', 50, 200),
                    max_depth=t.suggest_int('et_max_depth', 3, 20),
                    random_state=random_state
                ),
                'adaboost': lambda t: AdaBoostRegressor(
                    n_estimators=t.suggest_int('ada_n_estimators', 50, 200),
                    learning_rate=t.suggest_float('ada_lr', 0.01, 1.0),
                    random_state=random_state
                ),
                'decision_tree': lambda t: DecisionTreeRegressor(
                    max_depth=t.suggest_int('dt_max_depth', 3, 20),
                    random_state=random_state
                ),
                'svm': lambda t: SVR(
                    C=t.suggest_float('svr_C', 0.1, 10.0, log=True),
                    kernel=t.suggest_categorical('svr_kernel', ['linear', 'rbf']),
                    cache_size=1000,
                    max_iter=1000
                ),
                'knn': lambda t: KNeighborsRegressor(
                    n_neighbors=t.suggest_int('knn_neighbors', 3, 15),
                    n_jobs=-1
                ),
                'ridge': lambda t: Ridge(
                    alpha=t.suggest_float('ridge_alpha', 0.1, 10.0),
                    random_state=random_state
                ),
                'lasso': lambda t: Lasso(
                    alpha=t.suggest_float('lasso_alpha', 0.01, 1.0),
                    random_state=random_state
                ),
                'elastic_net': lambda t: ElasticNet(
                    alpha=t.suggest_float('en_alpha', 0.01, 1.0),
                    l1_ratio=t.suggest_float('en_l1_ratio', 0.1, 0.9),
                    random_state=random_state
                ),
                'sgd_regressor': lambda t: SGDRegressor(max_iter=1000, random_state=random_state),
                'mlp': lambda t: MLPRegressor(
                    hidden_layer_sizes=t.suggest_categorical('mlp_hidden', [(50,), (100,), (100, 50), (100, 100), (50, 50, 50)]),
                    activation=t.suggest_categorical('mlp_activation', ['relu', 'tanh']),
                    solver=t.suggest_categorical('mlp_solver', ['adam', 'sgd']),
                    alpha=t.suggest_float('mlp_alpha', 1e-6, 1e-1, log=True),
                    learning_rate_init=t.suggest_float('mlp_lr', 1e-5, 1e-1, log=True),
                    max_iter=500,
                    random_state=random_state,
                    early_stopping=True,
                    n_iter_no_change=10
                ),
                'catboost': lambda t: cb.CatBoostRegressor(
                    iterations=t.suggest_int('cb_iterations', 100, 1000) if self.preset == 'best_quality' else t.suggest_int('cb_iterations', 50, 150),
                    learning_rate=t.suggest_float('cb_lr', 0.001, 0.3, log=True),
                    depth=t.suggest_int('cb_depth', 4, 10) if self.preset == 'best_quality' else t.suggest_int('cb_depth', 4, 6),
                    l2_leaf_reg=t.suggest_float('cb_l2', 1, 10),
                    verbose=0,
                    thread_count=-1,
                    random_seed=random_state
                ) if CATBOOST_AVAILABLE else None,
                'bert-base-uncased-reg': lambda t: TransformersWrapper(model_name='bert-base-uncased', task='regression', learning_rate=t.suggest_float('learning_rate', 1e-6, 1e-3, log=True), epochs=t.suggest_int('num_train_epochs', 1, 20)) if TRANSFORMERS_AVAILABLE else None,
                'distilbert-base-uncased-reg': lambda t: TransformersWrapper(model_name='distilbert-base-uncased', task='regression', learning_rate=t.suggest_float('learning_rate', 1e-6, 1e-3, log=True), epochs=t.suggest_int('num_train_epochs', 1, 20)) if TRANSFORMERS_AVAILABLE else None,
                'roberta-base-reg': lambda t: TransformersWrapper(model_name='roberta-base', task='regression', learning_rate=t.suggest_float('learning_rate', 1e-6, 1e-3, log=True), epochs=t.suggest_int('num_train_epochs', 1, 20)) if TRANSFORMERS_AVAILABLE else None,
                'microsoft/deberta-v3-small': lambda t: TransformersWrapper(model_name='microsoft/deberta-v3-small', task='regression', learning_rate=t.suggest_float('learning_rate', 1e-6, 1e-3, log=True), epochs=t.suggest_int('num_train_epochs', 1, 20)) if TRANSFORMERS_AVAILABLE else None
            }
        elif self.task_type == 'clustering':
            models_config = {
                'kmeans': lambda t: KMeans(
                    n_clusters=t.suggest_int('km_n_clusters', 2, 12),
                    n_init=10
                ),
                'agglomerative': lambda t: AgglomerativeClustering(
                    n_clusters=t.suggest_int('agg_n_clusters', 2, 12)
                ),
                'dbscan': lambda t: DBSCAN(
                    eps=t.suggest_float('db_eps', 0.1, 2.0),
                    min_samples=t.suggest_int('db_min_samples', 2, 10),
                    n_jobs=None
                ),
                'gaussian_mixture': lambda t: GaussianMixture(
                    n_components=t.suggest_int('gm_n_components', 2, 12)
                ),
                'mean_shift': lambda t: MeanShift(n_jobs=None),
                'birch': lambda t: Birch(
                    n_clusters=t.suggest_int('birch_n_clusters', 2, 12)
                ),
                'spectral': lambda t: SpectralClustering(
                    n_clusters=t.suggest_int('spectral_n_clusters', 2, 12),
                    n_jobs=None
                )
            }
        elif self.task_type == 'time_series':
            models_config = {
                'random_forest_ts': lambda t: RandomForestRegressor(
                    n_estimators=t.suggest_int('rf_ts_n_estimators', 50, 200),
                    n_jobs=None
                ),
                'xgboost_ts': lambda t: xgb.XGBRegressor(
                    n_estimators=t.suggest_int('xgb_ts_n_estimators', 50, 200),
                    n_jobs=None
                ),
                'extra_trees_ts': lambda t: ExtraTreesRegressor(
                    n_estimators=t.suggest_int('et_ts_n_estimators', 50, 200),
                    n_jobs=None
                ),
                'catboost_ts': lambda t: cb.CatBoostRegressor(
                    iterations=t.suggest_int('cb_ts_iterations', 50, 200),
                    verbose=0,
                    thread_count=-1
                ) if CATBOOST_AVAILABLE else None
            }
        elif self.task_type == 'anomaly_detection':
            models_config = {
                'isolation_forest': lambda t: IsolationForest(
                    n_estimators=t.suggest_int('if_n_estimators', 50, 200),
                    contamination=t.suggest_float('if_contamination', 0.01, 0.2),
                    random_state=42,
                    n_jobs=None
                ),
                'local_outlier_factor': lambda t: LocalOutlierFactor(
                    n_neighbors=t.suggest_int('lof_neighbors', 10, 50),
                    contamination=t.suggest_float('lof_contamination', 0.01, 0.2),
                    novelty=True,
                    n_jobs=None
                ),
                'elliptic_envelope': lambda t: EllipticEnvelope(
                    contamination=t.suggest_float('ee_contamination', 0.01, 0.2),
                    random_state=42
                ),
                'one_class_svm': lambda t: OneClassSVM(
                    nu=t.suggest_float('oc_nu', 0.01, 0.2),
                    kernel=t.suggest_categorical('oc_kernel', ['rbf', 'poly', 'sigmoid'])
                )
            }
        elif self.task_type == 'dimensionality_reduction':
            # Use X_train feature count if available, otherwise default to 10 max
            max_comp = 10
            if hasattr(self, '_n_features') and self._n_features is not None:
                max_comp = max(2, min(10, self._n_features - 1)) # SVD requires strictly less than n_features
            
            def get_pca(t):
                try:
                    return PCA(
                        n_components=t.suggest_int('pca_n_components', 2, max_comp) if hasattr(t, 'suggest_int') else 2,
                        svd_solver=t.suggest_categorical('pca_solver', ['auto', 'full', 'arpack', 'randomized']) if hasattr(t, 'suggest_categorical') else 'auto',
                        random_state=42
                    )
                except Exception as e:
                    logger.error(f'Error creating PCA: {e}')
                    import traceback; traceback.print_exc()
                    return None
            
            def get_svd(t):
                try:
                    return TruncatedSVD(
                        n_components=t.suggest_int('svd_n_components', 2, max_comp) if hasattr(t, 'suggest_int') else 2,
                        algorithm=t.suggest_categorical('svd_alg', ['arpack', 'randomized']) if hasattr(t, 'suggest_categorical') else 'randomized',
                        random_state=42
                    )
                except Exception as e:
                    logger.error(f'Error creating SVD: {e}')
                    import traceback; traceback.print_exc()
                    return None

            models_config = {
                'pca': get_pca,
                'truncated_svd': get_svd
            }
            logger.info(f"DEBUG _get_models: task_type={self.task_type}, name requested={name}, returning keys={list(models_config.keys())}")
        else:
            return {}

        if name is None:
            # Filters models that return None (e.g. libraries not installed)
            available = []
            for k, v in models_config.items():
                try:
                    if v(None) is not None:
                        available.append(k)
                except:
                    # If lambda fails with None (e.g. t.suggest_int), we assume the model is available
                    # because the error comes from the trial logic, not the library absence
                    available.append(k)
            
            if hasattr(self, 'custom_models') and self.custom_models:
                available.extend(list(self.custom_models.keys()))
            return available
        
        if name in models_config:
            # Pass the trial to the lambda to instantiate the model with the suggested parameters
            return models_config[name](trial)
        
        return None

    def get_available_models(self):
        """Returns a list of available model names for the current task type."""
        return self._get_models()

    def train(self, X_train, y_train=None, n_trials=None, timeout=None, callback=None, selected_models=None, early_stopping_rounds=None, experiment_name="AutoML_Experiment", manual_params=None, random_state=42, validation_strategy='cv', validation_params=None, custom_models=None, X_raw=None, time_budget=None, optimization_mode='bayesian', optimization_metric='accuracy', stability_config=None, feature_names=None, class_names=None, **kwargs):
        # Use preset configurations if n_trials/timeout are not provided
        preset_config = self.preset_configs.get(self.preset, self.preset_configs['medium'])
        n_trials = n_trials if n_trials is not None else preset_config['n_trials']
        timeout = timeout if timeout is not None else preset_config.get('timeout')
        
        # Store custom models (uploaded or registered)
        self.custom_models = custom_models if custom_models else {}
        self.feature_names = feature_names
        self.class_names = class_names
        self._n_features = X_train.shape[1] if hasattr(X_train, 'shape') else (len(X_train[0]) if X_train is not None and len(X_train) > 0 else 2)
        
        # If selected_models is not provided, use the preset list
        if selected_models is None:
            selected_models = preset_config['models']
            
        self.ts_metadata = kwargs if self.task_type == 'time_series' else {}
        self.random_state = random_state
        self.ensemble_config = kwargs.get('ensemble_config', {})
        
        global_start_time = time.time()

        
        # Compatibility with old auto_split parameter
        if kwargs.get('auto_split', False):
            validation_strategy = 'auto_split'
            
        # Automatic Validation Logic
        if validation_strategy == 'auto':
            if self.task_type == 'time_series':
                validation_strategy = 'time_series_cv'
                logger.info("Automatic Validation: TimeSeriesSplit chosen (given it's a time series).")
            else:
                # If we have enough data, holdout is faster. If few, CV is more robust.
                if hasattr(X_train, 'shape'):
                    n_samples = X_train.shape[0]
                else:
                    n_samples = len(X_train)

                if n_samples < 1000:
                    validation_strategy = 'cv'
                    logger.info(f"Automatic Validation: Cross-Validation chosen (N={n_samples} < 1000).")
                else:
                    validation_strategy = 'holdout'
                    logger.info(f"Validacao Automatica: Escolhido Holdout/Train-Test Split (N={n_samples} >= 1000).")

        if validation_params is None:
            validation_params = {}
        
        tracker = None
        # Define DummyTracker here to ensure scope availability
        class DummyTracker:
            def log_experiment(self, **kwargs):
                return "dummy_run_id"

        try:
            from mlops_utils import MLFlowTracker
            tracker = MLFlowTracker(experiment_name)
            logger.info(f"MLFlowTracker initialized for experiment: {experiment_name}")
        except Exception as e:
            logger.warning(f"MLFlowTracker init failed: {e}. Proceeding without MLflow logging.")
            tracker = DummyTracker()

        # Early Stopping & Summary Logic
        best_score_so_far = -np.inf
        trials_without_improvement = 0
        model_trial_counts = {} # Fix chart index: real counter per model
        self.model_summaries = {} # Store best metric per model
        self.detailed_model_reports = {} # Store detailed reports for display

        def objective(trial, forced_model=None):
            nonlocal best_score_so_far, trials_without_improvement
            
            # Global Time Budget Check
            if time_budget is not None:
                elapsed = time.time() - global_start_time
                if elapsed > time_budget:
                    logger.warning(f"Global time budget exceeded ({elapsed:.2f}s > {time_budget}s). Stopping study.")
                    trial.study.stop()
                    return 0.0

            # 1. Determine model to be used
            all_available = selected_models if selected_models else self.get_available_models()
            if not all_available: all_available = self.get_available_models()

            if forced_model:
                # Force specific model without changing search space (avoiding dynamic space error)
                model_name = forced_model
                trial.set_user_attr("model_name", model_name)
            else:
                model_name = trial.suggest_categorical('model_name', all_available)

            # Unique identifier for this model trial
            model_trial_counts[model_name] = model_trial_counts.get(model_name, 0) + 1
            trial_num_for_model = model_trial_counts[model_name]
            full_trial_name = f"{model_name} - Trial {trial_num_for_model}"
            
            # Determine specific seed for this model
            current_seed = self.random_state
            if isinstance(self.random_state, dict):
                current_seed = self.random_state.get(model_name, 42)
            
            logger.info(f"Trial {trial.number} mapped to {full_trial_name} (Seed: {current_seed})")

            run_id = None

            # Global Early Stopping
            min_improvement = 0.0001
            if early_stopping_rounds and trials_without_improvement >= early_stopping_rounds:
                trial.study.stop()
                return 0

            # 2. Instantiate specific model suggested for this trial (Lazy Loading)
            model = self._get_models(trial=trial, name=model_name, random_state=current_seed)
            
            if model is None:
                return -1.0
            
            # Determine which input data to use (Vectorized or Raw)
            effective_X = X_train
            if X_raw is not None and isinstance(model, TransformersWrapper):
                effective_X = X_raw
                logger.info(f"Model {model_name} uses Transformers: Using RAW TEXT input.")
            elif isinstance(model, TransformersWrapper):
                logger.warning(f"Model {model_name} is a Transformer but X_raw was not provided. Expect failure if input is vectorized.")

            # Validation Logic and Data Split
            X_tr, X_val, y_tr, y_val = None, None, None, None
            
            # Only for methods using explicit holdout (auto_split or manual holdout)
            use_explicit_validation = validation_strategy in ['auto_split', 'holdout']
            
            if use_explicit_validation and self.task_type in ['classification', 'regression', 'time_series']:
                if validation_strategy == 'auto_split':
                    split_ratio = trial.suggest_float('data_split_ratio', 0.6, 0.9)
                else: # holdout
                    test_size = validation_params.get('test_size', 0.2)
                    split_ratio = 1.0 - test_size
                
                # Cache key should include if we are using raw or vectorized data
                is_raw = (effective_X is X_raw)
                if not hasattr(self, '_split_cache'): self._split_cache = {}
                cache_key = f"{split_ratio}_{self.task_type}_{current_seed}_{validation_strategy}_{is_raw}"
                
                if cache_key in self._split_cache:
                    X_tr, X_val, y_tr, y_val = self._split_cache[cache_key]
                else:
                    if self.task_type == 'time_series':
                        split_idx = int(len(effective_X) * split_ratio)
                        if isinstance(effective_X, pd.DataFrame):
                            X_tr, X_val = effective_X.iloc[:split_idx], effective_X.iloc[split_idx:]
                            y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
                        else:
                            X_tr, X_val = effective_X[:split_idx], effective_X[split_idx:]
                            y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
                    else:
                        if y_train is not None:
                            X_tr, X_val, y_tr, y_val = train_test_split(effective_X, y_train, train_size=split_ratio, random_state=current_seed)
                        else:
                            X_tr, X_val = train_test_split(effective_X, train_size=split_ratio, random_state=current_seed)
                            y_tr, y_val = None, None
                    if len(self._split_cache) > 10: self._split_cache.clear()
                    self._split_cache[cache_key] = (X_tr, X_val, y_tr, y_val)
            else:
                # For CV, we use the full dataset in cross_val_score
                X_tr, y_tr = effective_X, y_train
                X_val, y_val = None, None

            start_time = time.time()
            logger.info(f"Training {full_trial_name}...")
            trial_metrics = {}
            trial_params = trial.params.copy()
            trial_params['task_type'] = self.task_type
            
            try:
                if self.task_type in ['classification', 'regression', 'time_series']:
                    if use_explicit_validation:
                        logger.info(f"DEBUG: fit() called. X_tr type: {type(X_tr)}, shape: {X_tr.shape if hasattr(X_tr, 'shape') else len(X_tr)}")
                        logger.info(f"DEBUG: y_tr type: {type(y_tr)}, shape: {y_tr.shape if hasattr(y_tr, 'shape') else len(y_tr)}")
                        logger.info(f"DEBUG: Model: {model}")
                        model.fit(X_tr, y_tr)
                        logger.info("DEBUG: fit() completed.")
                        y_pred_val = model.predict(X_val)
                        if self.task_type == 'classification':
                            if optimization_metric == 'accuracy':
                                score = accuracy_score(y_val, y_pred_val)
                            elif optimization_metric == 'f1':
                                score = f1_score(y_val, y_pred_val, average='weighted')
                            elif optimization_metric == 'precision':
                                score = precision_score(y_val, y_pred_val, average='weighted')
                            elif optimization_metric == 'recall':
                                score = recall_score(y_val, y_pred_val, average='weighted')
                            elif optimization_metric == 'roc_auc':
                                try:
                                    y_prob_val = model.predict_proba(X_val)
                                    score = roc_auc_score(y_val, y_prob_val, multi_class='ovr')
                                except:
                                    score = 0.5 # Fail-safe
                            else:
                                score = accuracy_score(y_val, y_pred_val)

                            trial_metrics['accuracy'] = accuracy_score(y_val, y_pred_val)
                            trial_metrics['f1'] = f1_score(y_val, y_pred_val, average='weighted')
                            trial_metrics['precision'] = precision_score(y_val, y_pred_val, average='weighted')
                            trial_metrics['recall'] = recall_score(y_val, y_pred_val, average='weighted')
                            try:
                                y_prob_val = model.predict_proba(X_val)
                                trial_metrics['roc_auc'] = roc_auc_score(y_val, y_prob_val, multi_class='ovr')
                            except: pass
                        else:
                            trial_metrics['r2'] = r2_score(y_val, y_pred_val)
                            trial_metrics['rmse'] = np.sqrt(mean_squared_error(y_val, y_pred_val))
                            trial_metrics['mae'] = mean_absolute_error(y_val, y_pred_val)
                    else:
                        # Cross Validation Logic
                        n_splits = validation_params.get('folds', 3) if validation_params else 3
                        
                        # Define metrics to calculate via cross_validate
                        if self.task_type == 'classification':
                            scoring_list = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted', 'roc_auc_ovr']
                            
                            if validation_strategy == 'stratified_cv':
                                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=current_seed)
                            else:
                                cv = KFold(n_splits=n_splits, shuffle=True, random_state=current_seed)
                        elif self.task_type == 'regression' or self.task_type == 'time_series':
                            scoring_list = ['r2', 'neg_root_mean_squared_error', 'neg_mean_absolute_error']
                            if self.task_type == 'time_series' or validation_strategy == 'time_series_cv':
                                cv = TimeSeriesSplit(n_splits=n_splits)
                            else:
                                cv = KFold(n_splits=n_splits, shuffle=True, random_state=current_seed)
                        
                        from sklearn.model_selection import cross_validate
                        cv_results = cross_validate(model, X_tr, y_tr, cv=cv, scoring=scoring_list, error_score='raise')
                        
                        # Map results from cv_results to trial_metrics
                        for s_key in scoring_list:
                            val = cv_results[f'test_{s_key}'].mean()
                            # Adjust negative metrics
                            if s_key.startswith('neg_'):
                                clean_name = s_key.replace('neg_', '')
                                trial_metrics[clean_name] = -val
                            else:
                                trial_metrics[s_key.replace('_weighted', '').replace('_ovr', '')] = val
                        
                        # Main score for Optuna
                        if self.task_type == 'classification':
                            opt_key = optimization_metric if optimization_metric != 'accuracy' else 'accuracy'
                            score = trial_metrics.get(opt_key, trial_metrics['accuracy'])
                        else:
                            opt_key = optimization_metric if optimization_metric != 'r2' else 'r2'
                            score = trial_metrics.get(opt_key, trial_metrics['r2'])
                            # For error metrics we want to minimize (RMSE/MAE), Optuna needs the negative value to maximize
                            if optimization_metric in ['rmse', 'mae']:
                                score = -score
                        
                        # To save the model and artifacts, we need to fit on the full trial training set
                        # Note: Final fit on trial_set without CV for logging
                        logger.info(f"Finalizing training for model {full_trial_name}...")
                        model.fit(X_tr, y_tr)
                        logger.info(f"Training finalized for {full_trial_name}")

                        # ADDED: Log multiple metrics for every trial (not just the optimization one)
                        if self.task_type == 'classification' and hasattr(model, 'predict'):
                            try:
                                # Get predictions on a holdout subset or OOF if possible (OOF is expensive here, so use simple predict on train for logging purposes - warning: overfitting metrics!)
                                # Better: Use a small holdout from X_tr for quick logging metrics
                                X_sub_train, X_sub_val, y_sub_train, y_sub_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=42)
                                model.fit(X_sub_train, y_sub_train)
                                y_sub_pred = model.predict(X_sub_val)
                                trial_metrics['val_accuracy'] = accuracy_score(y_sub_val, y_sub_pred)
                                trial_metrics['val_f1'] = f1_score(y_sub_val, y_sub_pred, average='weighted')
                                # Re-fit full for the final model artifact
                                model.fit(X_tr, y_tr)
                            except: pass
                        
                elif self.task_type == 'anomaly_detection':
                    model.fit(X_tr)
                    if hasattr(model, 'decision_function'):
                        score = model.decision_function(X_tr).mean()
                    else:
                        score = 0
                    trial_metrics['decision_score'] = score
                elif self.task_type == 'dimensionality_reduction':
                    model.fit(X_tr)
                    if hasattr(model, 'explained_variance_ratio_'):
                        score = model.explained_variance_ratio_.sum()
                    else:
                        score = 0
                    trial_metrics['explained_variance'] = score
                else: # clustering
                    model.fit(X_tr)
                    labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X_tr)
                    
                    if len(set(labels)) > 1:
                        score = silhouette_score(X_tr, labels)
                        trial_metrics['silhouette'] = score
                        trial_metrics['calinski_harabasz'] = calinski_harabasz_score(X_tr, labels)
                        trial_metrics['davies_bouldin'] = davies_bouldin_score(X_tr, labels)
                    else:
                        score = -1.0 # Penalize model that groups everything into a single class
                        trial_metrics['silhouette'] = -1.0
                # Unified Logging for all tasks/strategies
                logger.info(f"Registering {full_trial_name} in MLflow...")
                run_id = tracker.log_experiment(
                    params=trial_params,
                    metrics=trial_metrics,
                    model=model,
                    model_name=full_trial_name,
                    register=False
                )
                logger.info(f"Run {full_trial_name} registered with ID: {run_id}")

            except Exception as e:
                logger.error(f"Error during trial for {model_name}: {e}")
                # Don't set score to 0.0 here, raise TrialPruned to let Optuna know the trial failed completely
                import optuna
                raise optuna.TrialPruned(f"Trial failed due to exception: {e}")

            duration = time.time() - start_time
            trial_metrics['duration'] = duration
            
            # Synchronize the final score with the chosen optimization metric to ensure correct display in the UI
            if optimization_metric in trial_metrics:
                score = trial_metrics[optimization_metric]
            
            # Ensure score is never negative for visualization purposes (unless metric allows)
            # Most of our metrics (acc, f1, r2) are >= 0. For RMSE/MAE we use negative, so we check task.
            if self.task_type in ['classification', 'clustering', 'anomaly_detection', 'dimensionality_reduction']:
                score = max(0.0, score)
            
            # Enrich trial metrics with parameters for easy access in callbacks
            # Prefix params to distinguish them
            for p_k, p_v in trial_params.items():
                if p_k not in trial_metrics:
                    trial_metrics[f"param_{p_k}"] = p_v
            
            trial.set_user_attr("run_id", run_id)
            trial.set_user_attr("full_name", full_trial_name)

            # Update model summary (best trial of each algorithm)
            if model_name not in self.model_summaries or score > self.model_summaries[model_name]['score']:
                self.model_summaries[model_name] = {
                    'score': score,
                    'metrics': trial_metrics,
                    'params': trial_params,
                    'duration': duration,
                    'trial_name': full_trial_name
                }

            if callback:
                try:
                    callback(trial, score, full_trial_name, duration, trial_metrics)
                except Exception as cb_err:
                    logger.error(f"Error in training callback: {cb_err}")
                
            if score > (best_score_so_far + min_improvement):
                best_score_so_far = score
                trials_without_improvement = 0
            else:
                trials_without_improvement += 1

            return score

        # All our metrics are better when larger
        direction = 'maximize'
        
        # Determine initial seed for sampler
        sampler_seed = self.random_state if isinstance(self.random_state, int) else 42
        
        # Sampler and Pruner selection based on optimization_mode
        sampler = None
        pruner = None
        
        if optimization_mode == 'random':
            sampler = optuna.samplers.RandomSampler(seed=sampler_seed)
            logger.info("Optimization Mode: Random Search")
        elif optimization_mode == 'grid':
            # Note: GridSampler requires a priori defined search space, which we don't have here.
            # Fallback to RandomSampler (which does stochastic search, similar to non-exhaustive Grid)
            sampler = optuna.samplers.RandomSampler(seed=sampler_seed)
            logger.warning("Grid Search selected, but using Random Search due to dynamic definition of search space.")
        elif optimization_mode == 'hyperband':
            # Hyperband uses TPE + aggressive Pruning
            sampler = optuna.samplers.TPESampler(n_startup_trials=5, seed=sampler_seed)
            pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=n_trials, reduction_factor=3)
            logger.info("Optimization Mode: Hyperband (TPE + HyperbandPruner)")
        else: # bayesian (default) or auto
            sampler = optuna.samplers.TPESampler(n_startup_trials=max(n_trials // 3, 5), seed=sampler_seed)
            logger.info("Optimization Mode: Bayesian Optimization (TPE)")
        
        study = optuna.create_study(
            direction=direction,
            sampler=sampler,
            pruner=pruner
        )
        
        # Identify static models (without hyperparameters to optimize)
        static_models = {
            'naive_bayes', 'ridge_classifier', 
            'linear_regression', 'mean_shift',
            'elliptic_envelope'
        }
        
        models_to_tune = selected_models if selected_models else self.get_available_models()
        
        for m_name in models_to_tune:
            # 1. Global Budget Verification
            if time_budget is not None:
                elapsed_total = time.time() - global_start_time
                if elapsed_total > time_budget:
                    logger.warning(f"Total time exceeded ({elapsed_total:.2f}s > {time_budget}s). Stopping training of new models.")
                    break
                
                remaining_budget = time_budget - elapsed_total
                
                # 2. Current Model Timeout Adjustment
                # Uses the minimum between the timeout defined for the model and the global remaining time
                current_timeout = timeout
                if current_timeout is None:
                    current_timeout = remaining_budget
                else:
                    current_timeout = min(current_timeout, remaining_budget)
                    
                # Skip if remaining time is irrelevant (< 1s)
                if current_timeout < 1.0:
                    logger.warning(f"Insufficient remaining time for {m_name}. Skipping.")
                    continue
            else:
                current_timeout = timeout

            # If the seed is per model, update sampler to ensure reproducibility per model
            if isinstance(self.random_state, dict):
                model_seed = self.random_state.get(m_name, 42)
                if optimization_mode == 'random' or optimization_mode == 'grid':
                     study.sampler = optuna.samplers.RandomSampler(seed=model_seed)
                else:
                     study.sampler = optuna.samplers.TPESampler(n_startup_trials=min(n_trials, 10), seed=model_seed)
                logger.info(f"Sampler seed updated to {model_seed} (Model: {m_name})")
            # If there are manual parameters for this model, enqueue a trial with them
            if manual_params and manual_params.get('model_name') == m_name:
                p = {'model_name': m_name}
                p.update({k: v for k, v in manual_params.items() if k != 'model_name'})
                study.enqueue_trial(p)
                logger.info(f"Enqueueing manual trial for {m_name}")

            # If the model is static, we run only 1 time
            current_n_trials = 1 if m_name in static_models else n_trials
            
            trials_without_improvement = 0 
            logger.info(f"Starting optimization for model: {m_name} ({current_n_trials} trials, Timeout: {current_timeout:.2f}s)")
            
            try:
                # The timeout parameter here ensures Optuna stops starting new trials after the limit
                study.optimize(
                    lambda t: objective(t, forced_model=m_name), 
                    n_trials=current_n_trials, 
                    timeout=current_timeout
                )
            except Exception as e:
                logger.error(f"Error during optimization of {m_name}: {e}")
            
            # --- Per-Model Reporting & Logging (Post-Optimization) ---
            try:
                # 1. Retrieve the best trial for this specific model
                best_trial_for_model = None
                best_score_for_model = -np.inf
                
                for t in study.trials:
                    # Check if trial was for this model and completed (ignore pruned/failed trials)
                    if t.state == optuna.trial.TrialState.COMPLETE and t.user_attrs.get("full_name", "").startswith(m_name):
                         if t.value is not None and t.value > best_score_for_model:
                             best_score_for_model = t.value
                             best_trial_for_model = t
                
                if best_trial_for_model:
                    logger.info(f"Best trial for {m_name}: Trial {best_trial_for_model.number} (Score: {best_score_for_model:.4f})")
                    
                    # 2. Re-instantiate the best model
                    best_params_model = best_trial_for_model.params.copy()
                    best_params_model['model_name'] = m_name # Ensure model name is present
                    
                    # Use specific seed if applicable
                    model_seed = self.random_state.get(m_name, 42) if isinstance(self.random_state, dict) else self.random_state
                    
                    # Correctly re-instantiate using FixedTrial so it can use the suggested parameters
                    fixed_trial = optuna.trial.FixedTrial(best_params_model)
                    best_model_instance = self._get_models(trial=fixed_trial, name=m_name, random_state=model_seed)
                else:
                    logger.warning(f"No trial completed successfully for {m_name}. Report ignored.")
                    continue
                    
                # Force n_jobs=1 for reporting and stability to avoid hangs/contention in containers
                if hasattr(best_model_instance, 'n_jobs'):
                     try: best_model_instance.set_params(n_jobs=1)
                     except: pass
                
                # Set seed
                if hasattr(best_model_instance, 'random_state'):
                    best_model_instance.set_params(random_state=model_seed)
                elif hasattr(best_model_instance, 'random_seed'):
                     best_model_instance.set_params(random_seed=model_seed)
                
                # Enable probability for classification plots
                if self.task_type == 'classification' and hasattr(best_model_instance, 'predict_proba'):
                     # SVM needs explicit probability=True
                     if m_name == 'svm' or isinstance(best_model_instance, SVC):
                         best_model_instance.set_params(probability=True)

                # 3. Generate robust validation predictions (CV or Holdout) for plots
                # We need y_true and y_pred (and y_proba)
                y_true_plot = None
                y_pred_plot = None
                y_proba_plot = None
                
                # Handle Data Input (Raw vs Vectorized)
                effective_X_plot = X_train
                if X_raw is not None and isinstance(best_model_instance, TransformersWrapper):
                    effective_X_plot = X_raw
                
                # Generate predictions using cross_val_predict or holdout split
                # To be consistent with optimization, let's use CV if data allows, otherwise Holdout
                # For plotting, we want out-of-sample predictions on the training set
                
                # Check for high dimensionality to avoid memory explosion in reports
                is_high_dim = False
                if hasattr(effective_X_plot, 'shape') and effective_X_plot.shape[1] > 5000:
                    is_high_dim = True
                    logger.warning(f"High dimensionality detected ({effective_X_plot.shape[1]} features). Skipping full report generation to save memory.")
                
                if validation_strategy == 'time_series_cv' or self.task_type == 'time_series':
                     # Time Series is tricky for 'full' plot. We'll skip complex plot generation here 
                     # or just do a simple validation split at end
                     pass 
                else:
                    # OPTIMIZED: Use a single holdout split for report plots instead of 3-fold CV
                    # This is much faster, especially for Random Forest
                    try:
                        logger.info(f"📊 Generating predictions for model report {m_name} (Fast Mode)...")
                        
                        # Sample data if too large to speed up report generation
                        X_rep_data = effective_X_plot
                        y_rep_data = y_train
                        
                        max_rep_samples = 2000 # Enough for good plots, fast enough for RF
                        if hasattr(X_rep_data, 'shape') and X_rep_data.shape[0] > max_rep_samples:
                            logger.info(f"Sampling {max_rep_samples} instances for report of {m_name}")
                            idx_rep = np.random.choice(X_rep_data.shape[0], max_rep_samples, replace=False)
                            if isinstance(X_rep_data, pd.DataFrame):
                                X_rep_data = X_rep_data.iloc[idx_rep]
                                y_rep_data = y_rep_data.iloc[idx_rep]
                            else:
                                X_rep_data = X_rep_data[idx_rep]
                                y_rep_data = y_rep_data[idx_rep]

                        # Simple 80/20 split for report metrics/plots
                        X_r_tr, X_r_val, y_r_tr, y_r_val = train_test_split(
                            X_rep_data, y_rep_data, test_size=0.2, random_state=model_seed
                        )
                        
                        # Re-fit on the 80% to get "clean" predictions on the 20%
                        # For RF, we can also limit n_estimators slightly for the report if it's still slow
                        if m_name == 'random_forest' and hasattr(best_model_instance, 'n_estimators'):
                             orig_estimators = best_model_instance.n_estimators
                             if orig_estimators > 100:
                                  best_model_instance.set_params(n_estimators=100)
                        
                        best_model_instance.fit(X_r_tr, y_r_tr)
                        y_pred_plot = best_model_instance.predict(X_r_val)
                        y_true_plot = y_r_val
                        
                        if self.task_type == 'classification' and hasattr(best_model_instance, 'predict_proba'):
                            y_proba_plot = best_model_instance.predict_proba(X_r_val)
                            
                        # Restore original estimators if changed
                        if m_name == 'random_forest' and 'orig_estimators' in locals():
                             best_model_instance.set_params(n_estimators=orig_estimators)
                             
                    except Exception as cv_err:
                        logger.warning(f"Failed to generate fast predictions for report: {cv_err}")
                        # Fallback: Simple predict on X_train (Overfit warning)
                        best_model_instance.fit(effective_X_plot, y_train)
                        y_pred_plot = best_model_instance.predict(effective_X_plot)
                        y_true_plot = y_train
                        if self.task_type == 'classification' and hasattr(best_model_instance, 'predict_proba'):
                            y_proba_plot = best_model_instance.predict_proba(effective_X_plot)
                
                # 4. Calculate detailed metrics
                report_metrics = {}
                if y_true_plot is not None and y_pred_plot is not None:
                    logger.info(f"Calculating detailed metrics for {m_name}...")
                    if self.task_type == 'classification':
                        report_metrics['accuracy'] = accuracy_score(y_true_plot, y_pred_plot)
                        report_metrics['f1'] = f1_score(y_true_plot, y_pred_plot, average='weighted')
                        report_metrics['precision'] = precision_score(y_true_plot, y_pred_plot, average='weighted')
                        report_metrics['recall'] = recall_score(y_true_plot, y_pred_plot, average='weighted')
                        if y_proba_plot is not None:
                            try:
                                report_metrics['roc_auc'] = roc_auc_score(y_true_plot, y_proba_plot, multi_class='ovr')
                            except: pass
                    elif self.task_type == 'regression':
                        report_metrics['r2'] = r2_score(y_true_plot, y_pred_plot)
                        report_metrics['rmse'] = np.sqrt(mean_squared_error(y_true_plot, y_pred_plot))
                        report_metrics['mae'] = mean_absolute_error(y_true_plot, y_pred_plot)

                # 5. Create plotting objects (Matplotlib Figures)
                plots = {}
                if y_true_plot is not None and y_pred_plot is not None:
                    logger.info(f"Generating visualizations (plots) for {m_name}...")
                    if self.task_type == 'classification':
                        # Confusion Matrix
                        cm = confusion_matrix(y_true_plot, y_pred_plot)
                        fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
                        
                        # Use class names for labels if available
                        labels = self.class_names if self.class_names else None
                        
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                                    xticklabels=labels, yticklabels=labels)
                        ax_cm.set_title(f'Confusion Matrix - {m_name}')
                        ax_cm.set_ylabel('True Label')
                        ax_cm.set_xlabel('Predicted Label')
                        plt.tight_layout()
                        plots[f'confusion_matrix_{m_name}'] = fig_cm
                        
                        if y_proba_plot is not None and hasattr(best_model_instance, 'classes_'):
                            from sklearn.preprocessing import label_binarize
                            from sklearn.metrics import roc_curve, auc
                            
                            # ROC Curve
                            fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
                            n_classes = len(np.unique(y_true_plot))
                            
                            if n_classes == 2:
                                fpr, tpr, _ = roc_curve(y_true_plot, y_proba_plot[:, 1])
                                roc_auc = auc(fpr, tpr)
                                ax_roc.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                            else:
                                # Multiclass ROC (Macro)
                                # This is a simplification
                                pass
                                
                            ax_roc.plot([0, 1], [0, 1], 'k--')
                            ax_roc.set_xlabel('False Positive Rate')
                            ax_roc.set_ylabel('True Positive Rate')
                            ax_roc.set_title(f'ROC Curve - {m_name}')
                            ax_roc.legend(loc="lower right")
                            plt.tight_layout()
                            plots[f'roc_curve_{m_name}'] = fig_roc

                    elif self.task_type == 'regression':
                        # Pred vs Actual
                        fig_reg, ax_reg = plt.subplots(figsize=(6, 5))
                        ax_reg.scatter(y_true_plot, y_pred_plot, alpha=0.5)
                        ax_reg.plot([y_true_plot.min(), y_true_plot.max()], [y_true_plot.min(), y_true_plot.max()], 'k--', lw=2)
                        ax_reg.set_xlabel('Actual')
                        ax_reg.set_ylabel('Predicted')
                        ax_reg.set_title(f'Actual vs Predicted - {m_name}')
                        plt.tight_layout()
                        plots[f'pred_vs_actual_{m_name}'] = fig_reg

                    # Ensure model is fitted for FI (cross_val_predict doesn't leave the model instance fitted)
                    try:
                        # Try to check if fitted, if not, fit
                        from sklearn.utils.validation import check_is_fitted
                        check_is_fitted(best_model_instance)
                        logger.info(f"Model {m_name} is already fitted.")
                    except:
                        try:
                            logger.info(f"Fitting {m_name} to extract Feature Importance for report...")
                            # Use a small sample if it's too large, but for FI we want representative
                            best_model_instance.fit(effective_X_plot, y_train)
                        except Exception as fit_err:
                            logger.warning(f"Could not fit {m_name} for FI plot: {fit_err}")

                    # ADDED: Check if model has feature importance or coefficients
                    has_fi = hasattr(best_model_instance, 'feature_importances_')
                    has_coef = hasattr(best_model_instance, 'coef_')
                    
                    # Special case for ensembles that might not expose FI directly but their estimators do
                    if not has_fi and not has_coef and (isinstance(best_model_instance, (VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor))):
                        logger.info(f"Model {m_name} is an ensemble. FI might be unavailable directly.")

                    if has_fi or has_coef:
                        try:
                            if hasattr(best_model_instance, 'feature_importances_'):
                                importances = best_model_instance.feature_importances_
                            else:
                                importances = np.abs(best_model_instance.coef_).flatten()
                            
                            # Limit to top 20 features
                            feat_names = []
                            if self.feature_names:
                                feat_names = list(self.feature_names)
                            elif hasattr(effective_X_plot, 'columns'):
                                feat_names = list(effective_X_plot.columns)
                            else:
                                feat_names = [f"Feature {i}" for i in range(len(importances))]
                            
                            # Sync lengths if possible
                            min_len = min(len(feat_names), len(importances))
                            if min_len > 0:
                                fi_df = pd.DataFrame({
                                    'Feature': feat_names[:min_len], 
                                    'Importance': importances[:min_len]
                                })
                                fi_df = fi_df.sort_values(by='Importance', ascending=False).head(20)
                                
                                fig_fi, ax_fi = plt.subplots(figsize=(8, 6))
                                sns.barplot(x='Importance', y='Feature', data=fi_df, ax=ax_fi, palette='viridis')
                                ax_fi.set_title(f'Top 20 Feature Importance - {m_name}')
                                plt.tight_layout()
                                plots[f'feature_importance_{m_name}'] = fig_fi
                        except Exception as fi_err:
                            logger.warning(f"Failed to generate FI plot for {m_name}: {fi_err}")
                        # plt.close(fig_reg) # Keep open for passing to callback? No, pyplot is stateful.
                        # Better: Return the Figure object.
                        
                # --- Stability Analysis ---
                stability_results = {}
                if stability_config and stability_config.get('tests'):
                    try:
                        logger.info(f"⚖️ Starting stability analysis for {m_name}...")
                        
                        # Use a subset for stability analysis to speed up per-model reporting
                        X_stab = effective_X_plot
                        y_stab = y_train
                        if hasattr(X_stab, 'shape') and X_stab.shape[0] > 500:
                             logger.info("Sampling data for stability analysis (N=500)...")
                             idx = np.random.choice(X_stab.shape[0], 500, replace=False)
                             if isinstance(X_stab, pd.DataFrame):
                                  X_stab = X_stab.iloc[idx]
                                  y_stab = y_stab.iloc[idx]
                             else:
                                  X_stab = X_stab[idx]
                                  y_stab = y_stab[idx]
                                  
                        analyzer = StabilityAnalyzer(best_model_instance, X_stab, y_stab, task_type=self.task_type, random_state=model_seed)
                        
                        tests = stability_config.get('tests', [])
                        n_iter = stability_config.get('n_iterations', 5) # Default lowered for reporting speed
                        
                        if "General Analysis" in tests:
                            stability_results['general'] = analyzer.run_general_stability_check(n_iterations=n_iter)
                            # Add general analysis plots
                            fig_seed, ax_seed = plt.subplots(figsize=(6, 4))
                            analyzer.calculate_stability_metrics(stability_results['general']['raw_seed'])['stability_score'].plot(kind='bar', ax=ax_seed)
                            ax_seed.set_title(f"Seed Stability Scores - {m_name}")
                            plt.tight_layout()
                            plots[f'stability_seed_{m_name}'] = fig_seed
                            
                            fig_split, ax_split = plt.subplots(figsize=(6, 4))
                            analyzer.calculate_stability_metrics(stability_results['general']['raw_split'])['stability_score'].plot(kind='bar', ax=ax_split)
                            ax_split.set_title(f"Split Stability Scores - {m_name}")
                            plt.tight_layout()
                            plots[f'stability_split_{m_name}'] = fig_split
                        
                        if "Initialization Robustness" in tests and 'general' not in stability_results:
                            stability_results['seed'] = analyzer.run_seed_stability(n_iterations=n_iter)
                            fig_seed, ax_seed = plt.subplots(figsize=(6, 4))
                            analyzer.calculate_stability_metrics(stability_results['seed'])['stability_score'].plot(kind='bar', ax=ax_seed)
                            ax_seed.set_title(f"Seed Stability - {m_name}")
                            plt.tight_layout()
                            plots[f'stability_seed_{m_name}'] = fig_seed

                        if "Data Variation Robustness" in tests and 'general' not in stability_results:
                            stability_results['split'] = analyzer.run_split_stability(n_splits=n_iter)
                            fig_split, ax_split = plt.subplots(figsize=(6, 4))
                            analyzer.calculate_stability_metrics(stability_results['split'])['stability_score'].plot(kind='bar', ax=ax_split)
                            ax_split.set_title(f"Data Split Stability - {m_name}")
                            plt.tight_layout()
                            plots[f'stability_split_{m_name}'] = fig_split
                            
                        if "Hyperparameter Sensitivity" in tests:
                            # Varies the most important parameter of the model or the first one we find
                            p_name = list(best_params_model.keys())[0] if best_params_model else None
                            if p_name and p_name != 'model_name':
                                p_val = best_params_model[p_name]
                                if isinstance(p_val, (int, float)):
                                    vals = [p_val * 0.5, p_val * 0.8, p_val, p_val * 1.2, p_val * 1.5]
                                    stability_results['hyperparam'] = analyzer.run_hyperparameter_stability(p_name, vals)
                                    fig_hyp, ax_hyp = plt.subplots(figsize=(6, 4))
                                    stability_results['hyperparam'].set_index('param_value').iloc[:, 0].plot(ax=ax_hyp)
                                    ax_hyp.set_title(f"Sensitivity: {p_name} - {m_name}")
                                    plt.tight_layout()
                                    plots[f'stability_hyper_{m_name}'] = fig_hyp

                    except Exception as stab_err:
                        logger.error(f"Failed stability analysis for {m_name}: {stab_err}")

                # 6. Save in MLflow (under the run_id of the best trial)
                # We need to retrieve the run_id from the trial
                best_run_id = best_trial_for_model.user_attrs.get("run_id")
                if best_run_id and tracker and not isinstance(tracker, DummyTracker):
                    try:
                        logger.info(f"Saving additional plots in MLflow Run ID: {best_run_id}")
                        
                        # Clear environment variables that can cause Run ID conflict
                        if "MLFLOW_RUN_ID" in os.environ:
                            del os.environ["MLFLOW_RUN_ID"]
                        if "MLFLOW_EXPERIMENT_ID" in os.environ:
                            del os.environ["MLFLOW_EXPERIMENT_ID"]
                        
                        # Ensure the correct experiment is selected
                        if hasattr(tracker, 'experiment_name'):
                             mlflow.set_experiment(tracker.experiment_name)
                        
                        # Ensure there is no active run to avoid conflicts
                        if mlflow.active_run():
                            active_run = mlflow.active_run()
                            if active_run.info.run_id != best_run_id:
                                logger.info(f"Ending active run {active_run.info.run_id} to start {best_run_id}")
                                mlflow.end_run()
                            else:
                                # Already in the right run
                                pass
                            
                        # If we are already in the right run, don't restart it (might cause nesting issues)
                        active_run = mlflow.active_run()
                        if active_run and active_run.info.run_id == best_run_id:
                            # Just log
                            if report_metrics:
                                mlflow.log_metrics({f"val_{k}": v for k, v in report_metrics.items()})
                            for plot_name, fig_obj in plots.items():
                                try:
                                    mlflow.log_figure(fig_obj, f"{plot_name}.png")
                                except: pass
                        else:
                            with mlflow.start_run(run_id=best_run_id):
                                # Log full params just in case
                                mlflow.log_params(best_params_model)
                                # Log extra metrics
                                if report_metrics:
                                    mlflow.log_metrics({f"val_{k}": v for k, v in report_metrics.items()})
                                
                                # Log stability results as dict if exists
                                if stability_results:
                                    # Just log that it was done and some summary
                                    mlflow.log_param("stability_analysis", "done")
                                
                                # Log plots
                                for plot_name, fig_obj in plots.items():
                                    try:
                                        mlflow.log_figure(fig_obj, f"{plot_name}.png")
                                    except Exception as e:
                                        logger.warning(f"Failed to log figure {plot_name} to MLflow: {e}")
                    except Exception as ml_err:
                        logger.warning(f"MLflow logging failed for report: {ml_err}. Continuing to UI callback.")
                
                # 7. Trigger Full Report Callback
                if callback:
                     # Convert Matplotlib figures to PIL images to avoid closing/memory issues in Streamlit
                     pil_plots = {}
                     for plot_name, fig_obj in plots.items():
                         try:
                             import io
                             from PIL import Image
                             buf = io.BytesIO()
                             fig_obj.savefig(buf, format='png', bbox_inches='tight')
                             buf.seek(0)
                             pil_plots[plot_name] = Image.open(buf)
                         except Exception as e:
                             logger.warning(f"Could not convert plot {plot_name} to image: {e}")
                             pil_plots[plot_name] = fig_obj # Fallback to original

                     # Prepare rich report object
                     full_report = {
                         'model_name': m_name,
                         'best_trial_number': best_trial_for_model.number,
                         'score': best_score_for_model,
                         'params': best_params_model,
                         'metrics': report_metrics,
                         'plots': pil_plots, # Use PIL images
                         'run_id': best_run_id,
                         'stability': stability_results
                     }
                     
                     report_payload = trial_metrics.copy() if 'trial_metrics' in locals() else {}
                     report_payload['__report__'] = full_report
                     
                     callback(best_trial_for_model, best_score_for_model, f"{m_name} - FINAL", 0.0, report_payload)
                     
                # Now we can safely close the Matplotlib figures
                for fig in plots.values():
                     try: plt.close(fig)
                     except: pass
                     
            except Exception as e:
                logger.error(f"Error generating final report for {m_name}: {e}")
                import traceback
                with open(f"report_error_{m_name}.txt", "w") as f:
                    f.write(traceback.format_exc())

            # Reset the stop flag to allow the next model to be optimized
            # study.stop() sets this flag to True
            if hasattr(study, '_stop_flag'):
                study._stop_flag = False
            logger.info(f"Optimization for {m_name} finalized.")
        
        self.best_params = study.best_params
        self.best_value = study.best_value
        best_model_name = self.best_params.get('model_name')
        
        # If model_name was forced (not in params), retrieve from user_attrs
        if not best_model_name and hasattr(study, 'best_trial'):
            best_model_name = study.best_trial.user_attrs.get('model_name')
        
        # FIX: Ensure best_params includes model_name
        if best_model_name:
            self.best_params['model_name'] = best_model_name
        
        logger.info(f"Best global model found: {best_model_name}")
        logger.info(f"Best parameters: {self.best_params}")
        
        if self.task_type == 'time_series':
            self.best_params.update(self.ts_metadata)
        
        self.best_model = self._instantiate_model(best_model_name, self.best_params)
        
        # Reactivate probability for the final model if it is SVM
        if best_model_name == 'svm' and hasattr(self.best_model, 'probability'):
            self.best_model.set_params(probability=True)
            logger.info("Activating probability estimation for the final SVM model.")
        
        # Ensure the same seed in the final model
        final_model_seed = self.random_state
        if isinstance(self.random_state, dict):
            final_model_seed = self.random_state.get(best_model_name, 42)
            
        if hasattr(self.best_model, 'random_state'):
            self.best_model.set_params(random_state=final_model_seed)
        elif hasattr(self.best_model, 'random_seed'):
            self.best_model.set_params(random_seed=final_model_seed)

        # Check for Transformers input
        final_X = X_train
        if isinstance(self.best_model, TransformersWrapper):
            if X_raw is not None:
                final_X = X_raw
                logger.info(f"Final Fit: Model {best_model_name} is a Transformer. Using RAW TEXT input.")
            else:
                 logger.warning(f"Final Fit: Model {best_model_name} is a Transformer but X_raw is missing. Expect failure.")

        if y_train is not None:
            self.best_model.fit(final_X, y_train)
        else:
            self.best_model.fit(final_X)
            
        # Add Feature Importance if available
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = self.best_model.feature_importances_.tolist()
            logger.info("Feature Importance calculated.")
        elif hasattr(self.best_model, 'coef_'):
            self.feature_importance = np.abs(self.best_model.coef_).flatten().tolist()
            logger.info("Feature Importance calculated (based on coefficients).")
        else:
            self.feature_importance = None

        return self.best_model

    def _instantiate_model(self, name, params):
        # 1. Custom Models (Uploaded/Registered)
        if hasattr(self, 'custom_models') and self.custom_models and name in self.custom_models:
            logger.info(f"Using custom model: {name}")
            return self.custom_models[name]

        # 2. Transformers
        if TRANSFORMERS_AVAILABLE and (name.startswith('bert') or name.startswith('roberta') or name.startswith('distilbert') or name.startswith('albert') or name.startswith('xlnet') or '/' in name):
             epochs = params.get('num_train_epochs', 3)
             lr = params.get('learning_rate', 2e-5)
             return TransformersWrapper(model_name=name, task=self.task_type, epochs=epochs, learning_rate=lr)

        if self.task_type == 'classification':
            if name == 'custom_voting':
                return VotingClassifier(
                    estimators=self._resolve_estimators(
                        self.ensemble_config.get('voting_estimators', [
                            ('lr', LogisticRegression(random_state=42)), 
                            ('rf', RandomForestClassifier(random_state=42))
                        ]),
                        42 # Default random_state for instantiation
                    ),
                    voting=self.ensemble_config.get('voting_type', 'soft'),
                    weights=self.ensemble_config.get('voting_weights', None),
                    n_jobs=-1
                )
            if name == 'custom_stacking':
                return StackingClassifier(
                    estimators=self._resolve_estimators(
                        self.ensemble_config.get('stacking_estimators', [
                            ('rf', RandomForestClassifier(random_state=42)),
                            ('svm', SVC(probability=True, random_state=42))
                        ]),
                        42
                    ),
                    final_estimator=self.ensemble_config.get('stacking_final_estimator', LogisticRegression(random_state=42)),
                    n_jobs=-1
                )
            if name == 'logistic_regression':  
                lr_params = {k.replace('lr_', ''): v for k, v in params.items() if k.startswith('lr_')}
                return LogisticRegression(max_iter=2000, **lr_params)
            if name == 'random_forest': 
                rf_params = {k.replace('rf_', ''): v for k, v in params.items() if k.startswith('rf_')}
                return RandomForestClassifier(**rf_params)
            if name == 'xgboost': 
                xgb_params = {k.replace('xgb_', ''): v for k, v in params.items() if k.startswith('xgb_')}
                return xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **xgb_params)
            if name == 'lightgbm': 
                lgb_params = {k.replace('lgb_', ''): v for k, v in params.items() if k.startswith('lgb_')}
                return lgb.LGBMClassifier(verbosity=-1, **lgb_params)
            if name == 'extra_trees':
                et_params = {k.replace('et_', ''): v for k, v in params.items() if k.startswith('et_')}
                return ExtraTreesClassifier(**et_params)
            if name == 'adaboost':
                ada_params = {k.replace('ada_', ''): v for k, v in params.items() if k.startswith('ada_')}
                return AdaBoostClassifier(**ada_params)
            if name == 'decision_tree':
                dt_params = {k.replace('dt_', ''): v for k, v in params.items() if k.startswith('dt_')}
                return DecisionTreeClassifier(**dt_params)
            if name == 'svm': return SVC(probability=True, **{k.replace('svm_', ''): v for k, v in params.items() if k.startswith('svm_') or k in ['C', 'kernel', 'gamma', 'degree', 'coef0', 'class_weight', 'shrinking', 'tol']})
            if name == 'linear_svc': 
                lsvc_params = {k: v for k, v in params.items() if k in ['C', 'loss', 'penalty', 'dual', 'class_weight', 'fit_intercept']}
                # Default dual=False if not specified to avoid convergence issues with primal
                if 'dual' not in lsvc_params and lsvc_params.get('penalty') == 'l2' and lsvc_params.get('loss') == 'hinge':
                     lsvc_params['dual'] = True # hinge requires dual=True
                elif 'dual' not in lsvc_params:
                     lsvc_params['dual'] = False
                return LinearSVC(max_iter=2000, **lsvc_params)
            if name == 'knn': 
                knn_params = {k.replace('knn_', ''): v for k, v in params.items() if k.startswith('knn_')}
                if 'neighbors' in knn_params:
                    knn_params['n_neighbors'] = knn_params.pop('neighbors')
                return KNeighborsClassifier(**knn_params)
            if name == 'naive_bayes': 
                nb_params = {k.replace('nb_', ''): v for k, v in params.items() if k.startswith('nb_')}
                return GaussianNB(**nb_params)
            if name == 'ridge_classifier': 
                rc_params = {k.replace('rc_', ''): v for k, v in params.items() if k.startswith('rc_')}
                if 'alpha' not in rc_params and 'ridge_alpha' in params: rc_params['alpha'] = params['ridge_alpha']
                return RidgeClassifier(**rc_params)
            if name == 'sgd_classifier': 
                sgd_params = {k.replace('sgd_', ''): v for k, v in params.items() if k.startswith('sgd_')}
                return SGDClassifier(max_iter=2000, **sgd_params)
            if name == 'mlp': 
                mlp_params = {k.replace('mlp_', ''): v for k, v in params.items() if k.startswith('mlp_')}
                if 'layers' in mlp_params:
                    mlp_params['hidden_layer_sizes'] = mlp_params.pop('layers')
                return MLPClassifier(early_stopping=True, **mlp_params)
            if name == 'catboost':
                cb_params = {k.replace('cb_', ''): v for k, v in params.items() if k.startswith('cb_')}
                return cb.CatBoostClassifier(verbose=0, thread_count=-1, **cb_params)
            if name == 'voting_ensemble':
                # Default God Mode Ensemble
                return VotingClassifier(
                    estimators=[
                        ('pa', PassiveAggressiveClassifier(max_iter=1000, random_state=42, C=0.5)),
                        ('lr', LogisticRegression(max_iter=2000, C=10, solver='saga', n_jobs=-1, random_state=42)),
                        ('sgd', SGDClassifier(loss='modified_huber', max_iter=2000, n_jobs=-1, random_state=42))
                    ],
                    voting='hard',
                    n_jobs=-1
                )
        elif self.task_type == 'regression':
            if name == 'custom_voting':
                return VotingRegressor(
                    estimators=self._resolve_estimators(
                        self.ensemble_config.get('voting_estimators', [
                            ('lr', LinearRegression()), 
                            ('rf', RandomForestRegressor(random_state=42))
                        ]),
                        42
                    ),
                    weights=self.ensemble_config.get('voting_weights', None),
                    n_jobs=-1
                )
            if name == 'custom_stacking':
                return StackingRegressor(
                    estimators=self._resolve_estimators(
                        self.ensemble_config.get('stacking_estimators', [
                            ('rf', RandomForestRegressor(random_state=42)),
                            ('svm', SVR())
                        ]),
                        42
                    ),
                    final_estimator=self.ensemble_config.get('stacking_final_estimator', LinearRegression()),
                    n_jobs=-1
                )
            if name == 'linear_regression': return LinearRegression()
            if name == 'random_forest': 
                rf_params = {k.replace('rf_', ''): v for k, v in params.items() if k.startswith('rf_')}
                return RandomForestRegressor(**rf_params)
            if name == 'xgboost': 
                xgb_params = {k.replace('xgb_', ''): v for k, v in params.items() if k.startswith('xgb_')}
                return xgb.XGBRegressor(**xgb_params)
            if name == 'lightgbm': 
                lgb_params = {k.replace('lgb_', ''): v for k, v in params.items() if k.startswith('lgb_')}
                return lgb.LGBMRegressor(verbosity=-1, **lgb_params)
            if name == 'extra_trees':
                et_params = {k.replace('et_', ''): v for k, v in params.items() if k.startswith('et_')}
                return ExtraTreesRegressor(**et_params)
            if name == 'adaboost':
                ada_params = {k.replace('ada_', ''): v for k, v in params.items() if k.startswith('ada_')}
                return AdaBoostRegressor(**ada_params)
            if name == 'decision_tree':
                dt_params = {k.replace('dt_', ''): v for k, v in params.items() if k.startswith('dt_')}
                return DecisionTreeRegressor(**dt_params)
            if name == 'svm': return SVR(**{k.replace('svm_', ''): v for k, v in params.items() if k.startswith('svm_') or k in ['C', 'kernel', 'gamma', 'epsilon', 'degree', 'coef0']})
            if name == 'knn': 
                knn_params = {k.replace('knn_', ''): v for k, v in params.items() if k.startswith('knn_')}
                if 'neighbors' in knn_params:
                    knn_params['n_neighbors'] = knn_params.pop('neighbors')
                return KNeighborsRegressor(**knn_params)
            if name == 'ridge': return Ridge(alpha=params.get('ridge_alpha', 1.0))
            if name == 'lasso': return Lasso(alpha=params.get('lasso_alpha', 1.0))
            if name == 'elastic_net': 
                return ElasticNet(alpha=params.get('en_alpha', 1.0), l1_ratio=params.get('en_l1_ratio', 0.5))
            if name == 'sgd_regressor': 
                sgd_params = {k.replace('sgd_', ''): v for k, v in params.items() if k.startswith('sgd_')}
                return SGDRegressor(max_iter=1000, **sgd_params)
            if name == 'mlp': 
                mlp_params = {k.replace('mlp_', ''): v for k, v in params.items() if k.startswith('mlp_')}
                if 'layers' in mlp_params:
                    mlp_params['hidden_layer_sizes'] = mlp_params.pop('layers')
                return MLPRegressor(early_stopping=True, **mlp_params)
            if name == 'catboost':
                cb_params = {k.replace('cb_', ''): v for k, v in params.items() if k.startswith('cb_')}
                return cb.CatBoostRegressor(verbose=0, thread_count=-1, **cb_params)
        elif self.task_type == 'clustering':
            if name == 'kmeans': 
                return KMeans(n_clusters=params.get('km_n_clusters', 8), n_init=10)
            if name == 'agglomerative':
                return AgglomerativeClustering(n_clusters=params.get('agg_n_clusters', 2))
            if name == 'dbscan':
                return DBSCAN(eps=params.get('db_eps', 0.5), min_samples=params.get('db_min_samples', 5))
            if name == 'gaussian_mixture':
                return GaussianMixture(n_components=params.get('gm_n_components', 1))
            if name == 'mean_shift': return MeanShift()
            if name == 'birch': 
                return Birch(n_clusters=params.get('birch_n_clusters', 3))
            if name == 'spectral':
                return SpectralClustering(n_clusters=params.get('spectral_n_clusters', 3))
        elif self.task_type == 'time_series':
            if name == 'random_forest_ts':
                return RandomForestRegressor(n_estimators=params.get('rf_ts_n_estimators', 100))
            if name == 'xgboost_ts':
                return xgb.XGBRegressor(n_estimators=params.get('xgb_ts_n_estimators', 100))
            if name == 'extra_trees_ts':
                return ExtraTreesRegressor(n_estimators=params.get('et_ts_n_estimators', 100))
            if name == 'catboost_ts':
                cb_params = {k.replace('cb_ts_', ''): v for k, v in params.items() if k.startswith('cb_ts_')}
                return cb.CatBoostRegressor(verbose=0, thread_count=-1, **cb_params)
        elif self.task_type == 'anomaly_detection':
            if name == 'isolation_forest':
                if_params = {k.replace('if_', ''): v for k, v in params.items() if k.startswith('if_')}
                return IsolationForest(random_state=42, **if_params)
            if name == 'local_outlier_factor':
                lof_params = {k.replace('lof_', ''): v for k, v in params.items() if k.startswith('lof_')}
                if 'neighbors' in lof_params: lof_params['n_neighbors'] = lof_params.pop('neighbors')
                return LocalOutlierFactor(novelty=True, **lof_params)
            if name == 'elliptic_envelope':
                ee_params = {k.replace('ee_', ''): v for k, v in params.items() if k.startswith('ee_')}
                return EllipticEnvelope(random_state=42, **ee_params)
            if name == 'one_class_svm':
                oc_params = {k.replace('oc_', ''): v for k, v in params.items() if k.startswith('oc_')}
                return OneClassSVM(**oc_params)

    def evaluate(self, X_test, y_test=None):
        if y_test is not None:
            y_pred = self.best_model.predict(X_test)
            metrics = {}
            
            if self.task_type == 'classification':
                metrics['accuracy'] = accuracy_score(y_test, y_pred)
                metrics['balanced_accuracy'] = balanced_accuracy_score(y_test, y_pred)
                metrics['precision'] = precision_score(y_test, y_pred, average='weighted')
                metrics['recall'] = recall_score(y_test, y_pred, average='weighted')
                metrics['f1'] = f1_score(y_test, y_pred, average='weighted')
                metrics['kappa'] = cohen_kappa_score(y_test, y_pred)
                metrics['mcc'] = matthews_corrcoef(y_test, y_pred)
                metrics['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
                try:
                    y_prob = self.best_model.predict_proba(X_test)
                    metrics['roc_auc'] = roc_auc_score(y_test, y_prob, multi_class='ovr')
                    metrics['log_loss'] = log_loss(y_test, y_prob)
                except:
                    metrics['roc_auc'] = 0.5
                    metrics['log_loss'] = 0.0
            elif self.task_type in ['regression', 'time_series']:
                metrics['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
                metrics['mae'] = mean_absolute_error(y_test, y_pred)
                metrics['median_ae'] = median_absolute_error(y_test, y_pred)
                metrics['r2'] = r2_score(y_test, y_pred)
                metrics['explained_variance'] = explained_variance_score(y_test, y_pred)
                metrics['mape'] = mean_absolute_percentage_error(y_test, y_pred)
                try:
                    metrics['msle'] = mean_squared_log_error(y_test, np.clip(y_pred, 0, None))
                except:
                    metrics['msle'] = 0.0
            
            elif self.task_type == 'anomaly_detection':
                # For anomaly detection, -1 is anomaly, 1 is normal
                # If y_test is provided, we assume it has labels (0 for normal, 1 for anomaly)
                # and map predictions accordingly
                y_pred_mapped = np.where(y_pred == -1, 1, 0)
                metrics['accuracy'] = accuracy_score(y_test, y_pred_mapped)
                metrics['f1'] = f1_score(y_test, y_pred_mapped)
                metrics['n_anomalies'] = int(np.sum(y_pred == -1))
                
            return metrics, y_pred
        else:
            # Clustering or Anomaly Detection without y_test
            if self.task_type == 'clustering':
                labels = self.best_model.labels_ if hasattr(self.best_model, 'labels_') else self.best_model.predict(X_test)
                metrics = {
                    'silhouette': silhouette_score(X_test, labels) if len(set(labels)) > 1 else -1,
                    'calinski_harabasz': calinski_harabasz_score(X_test, labels) if len(set(labels)) > 1 else -1,
                    'davies_bouldin': davies_bouldin_score(X_test, labels) if len(set(labels)) > 1 else -1,
                    'n_clusters': len(set(labels))
                }
                return metrics, labels
            elif self.task_type == 'anomaly_detection':
                y_pred = self.best_model.predict(X_test)
                metrics = {
                    'n_anomalies': int(np.sum(y_pred == -1)),
                    'anomaly_ratio': float(np.sum(y_pred == -1) / len(y_pred))
                }
                return metrics, y_pred

    def get_model_params_schema(self, model_name):
        """Returns the parameter schema for manual fine-tuning, matching _get_models keys."""
        schemas = {
            'logistic_regression': {
                'lr_C': ('float', 0.001, 100.0, 1.0),
                'lr_solver': ('list', ['lbfgs', 'liblinear', 'saga', 'newton-cg'], 'lbfgs'),
                'lr_penalty': ('list', ['l2', 'l1', 'elasticnet', None], 'l2'),
                'lr_l1_ratio': ('float', 0.0, 1.0, 0.5),
                'lr_class_weight': ('list', [None, 'balanced'], None),
                'lr_fit_intercept': ('list', [True, False], True),
                'lr_max_iter': ('int', 100, 5000, 2000),
                'lr_tol': ('float', 1e-5, 1e-3, 1e-4)
            },
            'linear_svc': {
                'C': ('float', 0.001, 100.0, 1.0),
                'loss': ('list', ['hinge', 'squared_hinge'], 'squared_hinge'),
                'penalty': ('list', ['l1', 'l2'], 'l2'),
                'class_weight': ('list', [None, 'balanced'], None),
                'fit_intercept': ('list', [True, False], True),
                'dual': ('list', [True, False, 'auto'], 'auto'),
                'max_iter': ('int', 1000, 10000, 2000),
                'tol': ('float', 1e-5, 1e-3, 1e-4)
            },
            'ridge_classifier': {
                'rc_alpha': ('float', 0.01, 100.0, 1.0),
                'rc_solver': ('list', ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'], 'auto'),
                'rc_class_weight': ('list', [None, 'balanced'], None),
                'rc_fit_intercept': ('list', [True, False], True)
            },
            'sgd_classifier': {
                'sgd_alpha': ('float', 1e-6, 1e-1, 0.0001),
                'sgd_penalty': ('list', ['l2', 'l1', 'elasticnet'], 'l2'),
                'sgd_loss': ('list', ['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron'], 'hinge'),
                'sgd_learning_rate': ('list', ['optimal', 'constant', 'invscaling', 'adaptive'], 'optimal'),
                'sgd_eta0': ('float', 0.0, 1.0, 0.0),
                'sgd_class_weight': ('list', [None, 'balanced'], None),
                'sgd_max_iter': ('int', 100, 5000, 1000),
                'sgd_tol': ('float', 1e-5, 1e-3, 1e-3)
            },
            'sgd_regressor': {
                'sgd_alpha': ('float', 1e-6, 1e-1, 0.0001),
                'sgd_penalty': ('list', ['l2', 'l1', 'elasticnet'], 'l2'),
                'sgd_loss': ('list', ['squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'], 'squared_error'),
                'sgd_learning_rate': ('list', ['invscaling', 'constant', 'optimal', 'adaptive'], 'invscaling'),
                'sgd_eta0': ('float', 0.0, 1.0, 0.01),
                'sgd_max_iter': ('int', 100, 5000, 1000),
                'sgd_tol': ('float', 1e-5, 1e-3, 1e-3)
            },
            'random_forest': {
                'rf_n_estimators': ('int', 10, 500, 100),
                'rf_max_depth': ('int', 1, 50, 10),
                'rf_min_samples_split': ('int', 2, 20, 2),
                'rf_min_samples_leaf': ('int', 1, 20, 1),
                'rf_max_features': ('list', ['sqrt', 'log2', None], 'sqrt'),
                'rf_bootstrap': ('list', [True, False], True),
                'rf_class_weight': ('list', [None, 'balanced', 'balanced_subsample'], None),
                'rf_criterion': ('list', ['gini', 'entropy', 'log_loss'], 'gini')
            },
            'xgboost': {
                'xgb_n_estimators': ('int', 50, 1000, 100),
                'xgb_lr': ('float', 0.01, 0.3, 0.1),
                'xgb_subsample': ('float', 0.5, 1.0, 1.0),
                'xgb_colsample_bytree': ('float', 0.5, 1.0, 1.0),
                'xgb_gamma': ('float', 0.0, 5.0, 0.0),
                'xgb_min_child_weight': ('float', 1.0, 10.0, 1.0),
                'xgb_scale_pos_weight': ('float', 1.0, 10.0, 1.0),
                'xgb_reg_alpha': ('float', 0.0, 10.0, 0.0),
                'xgb_reg_lambda': ('float', 0.0, 10.0, 1.0),
                'xgb_tree_method': ('list', ['auto', 'exact', 'approx', 'hist'], 'auto'),
                'xgb_max_leaves': ('int', 0, 100, 0),
                'xgb_grow_policy': ('list', ['depthwise', 'lossguide'], 'depthwise')
            },
            'lightgbm': {
                'lgb_n_estimators': ('int', 50, 1000, 100),
                'lgb_lr': ('float', 0.01, 0.3, 0.1),
                'lgb_num_leaves': ('int', 20, 150, 31),
                'lgb_min_child_samples': ('int', 10, 100, 20),
                'lgb_subsample': ('float', 0.5, 1.0, 1.0),
                'lgb_colsample_bytree': ('float', 0.5, 1.0, 1.0),
                'lgb_scale_pos_weight': ('float', 1.0, 10.0, 1.0),
                'lgb_reg_alpha': ('float', 0.0, 10.0, 0.0),
                'lgb_reg_lambda': ('float', 0.0, 10.0, 0.0),
                'lgb_boosting_type': ('list', ['gbdt', 'dart', 'goss'], 'gbdt'),
                'lgb_min_split_gain': ('float', 0.0, 1.0, 0.0),
                'lgb_class_weight': ('list', [None, 'balanced'], None)
            },
            'extra_trees': {
                'et_n_estimators': ('int', 10, 500, 100),
                'et_max_depth': ('int', 1, 50, 10),
                'et_min_samples_split': ('int', 2, 20, 2),
                'et_min_samples_leaf': ('int', 1, 20, 1),
                'et_max_features': ('list', ['sqrt', 'log2', None], 'sqrt'),
                'et_bootstrap': ('list', [True, False], False),
                'et_class_weight': ('list', [None, 'balanced', 'balanced_subsample'], None),
                'et_criterion': ('list', ['gini', 'entropy', 'log_loss'], 'gini')
            },
            'adaboost': {
                'ada_n_estimators': ('int', 10, 500, 50),
                'ada_lr': ('float', 0.01, 1.0, 1.0)
            },
            'decision_tree': {
                'dt_max_depth': ('int', 1, 50, 10),
                'dt_min_samples_split': ('int', 2, 20, 2),
                'dt_min_samples_leaf': ('int', 1, 20, 1),
                'dt_criterion': ('list', ['gini', 'entropy', 'log_loss'], 'gini') if self.task_type == 'classification' else ('list', ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'], 'squared_error'),
                'dt_splitter': ('list', ['best', 'random'], 'best'),
                'dt_class_weight': ('list', [None, 'balanced'], None)
            },
            'mlp': {
                'mlp_layers': ('list', ["(50,)", "(100,)", "(50, 50)", "(100, 50)"], "(100,)"),
                'mlp_max_iter': ('int', 100, 2000, 500),
                'mlp_activation': ('list', ['identity', 'logistic', 'tanh', 'relu'], 'relu'),
                'mlp_solver': ('list', ['lbfgs', 'sgd', 'adam'], 'adam'),
                'mlp_alpha': ('float', 0.0001, 0.1, 0.0001),
                'mlp_learning_rate': ('list', ['constant', 'invscaling', 'adaptive'], 'constant'),
                'mlp_batch_size': ('list', ['auto', 32, 64, 128], 'auto'),
                'mlp_tol': ('float', 1e-5, 1e-3, 1e-4)
            },
            'knn': {
                'knn_neighbors': ('int', 1, 30, 5),
                'knn_weights': ('list', ['uniform', 'distance'], 'uniform'),
                'knn_metric': ('list', ['minkowski', 'euclidean', 'manhattan'], 'minkowski'),
                'knn_p': ('int', 1, 5, 2),
                'knn_algorithm': ('list', ['auto', 'ball_tree', 'kd_tree', 'brute'], 'auto')
            },
            'naive_bayes': {
                'nb_var_smoothing': ('float', 1e-10, 1e-7, 1e-9)
            },
            'svm': {
                'C': ('float', 0.1, 100.0, 1.0),
                'kernel': ('list', ['linear', 'poly', 'rbf', 'sigmoid'], 'rbf'),
                'gamma': ('list', ['scale', 'auto'], 'scale'),
                'degree': ('int', 2, 5, 3),
                'coef0': ('float', 0.0, 10.0, 0.0),
                'class_weight': ('list', [None, 'balanced'], None),
                'shrinking': ('list', [True, False], True),
                'tol': ('float', 1e-4, 1e-2, 1e-3)
            },
            'ridge': {
                'ridge_alpha': ('float', 0.1, 10.0, 1.0)
            },
            'lasso': {
                'lasso_alpha': ('float', 0.01, 1.0, 0.1)
            },
            'elastic_net': {
                'en_alpha': ('float', 0.01, 1.0, 0.1),
                'en_l1_ratio': ('float', 0.0, 1.0, 0.5)
            },
            'catboost': {
                'cb_iterations': ('int', 10, 1000, 100),
                'cb_lr': ('float', 0.001, 0.3, 0.1),
                'cb_depth': ('int', 1, 12, 6),
                'cb_l2_leaf_reg': ('float', 1.0, 10.0, 3.0),
                'cb_border_count': ('int', 32, 255, 254)
            },
            'catboost_ts': {
                'cb_ts_iterations': ('int', 10, 1000, 100)
            },
            'random_forest_ts': {
                'rf_ts_n_estimators': ('int', 50, 200, 100)
            },
            'xgboost_ts': {
                'xgb_ts_n_estimators': ('int', 50, 200, 100)
            },
            'extra_trees_ts': {
                'et_ts_n_estimators': ('int', 50, 200, 100)
            },
            'isolation_forest': {
                'if_n_estimators': ('int', 50, 200, 100),
                'if_contamination': ('float', 0.01, 0.2, 0.1)
            },
            'local_outlier_factor': {
                'lof_neighbors': ('int', 10, 50, 20),
                'lof_contamination': ('float', 0.01, 0.2, 0.1)
            },
            'elliptic_envelope': {
                'ee_contamination': ('float', 0.01, 0.2, 0.1)
            },
            'one_class_svm': {
                'oc_nu': ('float', 0.01, 0.2, 0.1),
                'oc_kernel': ('list', ['rbf', 'poly', 'sigmoid'], 'rbf')
            },
            'kmeans': {
                'km_n_clusters': ('int', 2, 20, 8)
            },
            'agglomerative': {
                'agg_n_clusters': ('int', 2, 20, 2)
            },
            'dbscan': {
                'db_eps': ('float', 0.1, 5.0, 0.5),
                'db_min_samples': ('int', 1, 20, 5)
            },
            'gaussian_mixture': {
                'gm_n_components': ('int', 1, 20, 1)
            },
            'birch': {
                'birch_n_clusters': ('int', 2, 20, 3)
            },
            'spectral': {
                'spectral_n_clusters': ('int', 2, 20, 3)
            }
        }

        if TRANSFORMERS_AVAILABLE:
             # Add schema for all transformer models (same parameters for all)
             transformer_models = [
                 'bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 
                 'albert-base-v2', 'xlnet-base-cased', 'microsoft/deberta-v3-base',
                 'microsoft/deberta-v3-small'
             ]
             for tm in transformer_models:
                 schemas[tm] = {
                     'learning_rate': ('float', 1e-6, 1e-3, 2e-5),
                     'num_train_epochs': ('int', 1, 20, 3)
                 }

        return schemas.get(model_name, {})

def get_technical_explanation(model_name, params, task_type):
    explanations = {
        'classification': {
            'random_forest': "Random Forest is an ensemble of decision trees. It is robust to outliers and reduces overfitting by averaging multiple trees.",
            'xgboost': "XGBoost uses extreme Gradient Boosting. It is efficient and powerful, correcting errors of previous trees sequentially.",
            'lightgbm': "LightGBM is a gradient boosting framework that uses tree-based algorithms. It is designed to be distributed and efficient.",
            'extra_trees': "Extra Trees is similar to Random Forest, but uses more extreme random splits, which can further reduce variance.",
            'adaboost': "AdaBoost focuses on instances that were incorrectly classified by previous models, building a strong classifier from weak ones.",
            'decision_tree': "Decision Tree is a simple and interpretable model that splits data into branches based on features.",
            'svm': "SVM finds the hyperplane that best separates the classes in the feature space.",
            'mlp': "MLP is a basic artificial neural network capable of learning complex non-linear relationships.",
            'logistic_regression': "Logistic Regression is a statistical model used for binary or multiclass classification.",
            'linear_svc': "LinearSVC is a fast implementation of SVM with a linear kernel, efficient for large datasets.",
            'knn': "K-Nearest Neighbors (KNN) classifies instances based on the majority class of its nearest neighbors.",
            'naive_bayes': "Naive Bayes is a probabilistic classifier based on Bayes' Theorem with strong independence assumptions.",
            'ridge_classifier': "Ridge Classifier uses ridge regression for classification, applying L2 regularization.",
            'sgd_classifier': "SGD Classifier uses Stochastic Gradient Descent, ideal for online learning and large volumes of data."
        },
        'regression': {
            'random_forest': "Random Forest for regression combines multiple trees to predict continuous values with high stability.",
            'xgboost': "XGBoost for regression offers high performance and regularization to prevent overfitting in tabular data.",
            'lightgbm': "LightGBM is extremely fast for regression on large datasets.",
            'extra_trees': "Extra Trees Regressor offers an extra level of randomness, useful when the data has a lot of noise.",
            'adaboost': "AdaBoost Regressor adjusts weights of previous predictions to improve accuracy on continuous values.",
            'decision_tree': "Decision Tree for regression predicts values based on the average of the data in each leaf.",
            'svm': "SVR tries to find a function that deviates from y maximally by a small amount for all training data.",
            'mlp': "MLP for regression can capture highly complex and non-linear numerical patterns.",
            'linear_regression': "Linear Regression is the base model that assumes a linear relationship between the variables.",
            'knn': "KNN Regressor estimates the target value based on the average of its nearest neighbors' values.",
            'ridge': "Ridge Regression applies L2 regularization to prevent coefficients from becoming too large.",
            'lasso': "Lasso Regression applies L1 regularization, which can lead to zero coefficients (feature selection).",
            'elastic_net': "Elastic Net combines L1 and L2 regularization, useful when there are multiple correlated features.",
            'sgd_regressor': "SGD Regressor applies Stochastic Gradient Descent for large-scale regression problems."
        },
        'clustering': {
            'kmeans': "K-Means groups data by minimizing the variance within each cluster (distance to centroids).",
            'agglomerative': "Agglomerative Hierarchical Clustering builds a hierarchy of clusters from the bottom up.",
            'dbscan': "DBSCAN identifies clusters based on density, excellent for finding arbitrary shapes and noise.",
            'gaussian_mixture': "Gaussian Mixture assumes that the data is generated by a mixture of several Gaussian distributions.",
            'mean_shift': "Mean Shift is a density-based algorithm that does not require the number of clusters a priori.",
            'birch': "BIRCH is efficient for clustering large datasets, building a cluster feature tree.",
            'spectral': "Spectral Clustering uses the connectivity of the data to group instances, useful for non-convex structures."
        },
        'time_series': {
            'random_forest_ts': "Random Forest adapted for time series using lags and temporal features as predictors.",
            'xgboost_ts': "XGBoost for time series focuses on capturing trends and seasonalities through boosting.",
            'extra_trees_ts': "Extra Trees for time series helps mitigate the noise inherent in temporal data."
        },
        'anomaly_detection': {
            'isolation_forest': "Isolation Forest isolates anomalies by randomly selecting a feature and a split value.",
            'local_outlier_factor': "LOF compares the local density of a point with that of its neighbors to identify outliers.",
            'elliptic_envelope': "Assumes that normal data comes from a Gaussian distribution and detects what is outside the ellipsoid.",
            'one_class_svm': "Learns a boundary that encloses the majority of normal data, classifying what is outside as an anomaly."
        }
    }
    
    model_desc = explanations.get(task_type, {}).get(model_name, "Model selected for its superior performance during optimization.")
    
    param_desc = []
    for k, v in params.items():
        if 'n_estimators' in k: param_desc.append(f"- **Estimators ({v})**: Number of trees in the ensemble. More trees increase robustness, but computational cost rises.")
        if 'max_depth' in k: param_desc.append(f"- **Max Depth ({v})**: Limits the tree growth to prevent overfitting.")
        if 'learning_rate' in k or 'lr' in k: param_desc.append(f"- **Learning Rate ({v})**: Controls the magnitude of the weights adjustment at each iteration.")
        if 'n_clusters' in k: param_desc.append(f"- **Number of Clusters ({v})**: Number of groups the algorithm will try to identify in the data.")
        if 'contamination' in k: param_desc.append(f"- **Contamination ({v})**: The expected proportion of anomalies in the dataset.")
        if 'n_neighbors' in k or 'neighbors' in k: param_desc.append(f"- **Neighbors ({v})**: Number of neighbors to use when calculating local density.")
        
    return model_desc, "\n".join(param_desc)

def save_pipeline(processor, model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({'processor': processor, 'model': model}, path)

def load_pipeline(path):
    data = joblib.load(path)
    return data['processor'], data['model']
