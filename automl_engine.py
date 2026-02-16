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
    AdaBoostClassifier, AdaBoostRegressor
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
from sklearn.exceptions import ConvergenceWarning

# Opcional: Transformers para NLP
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Silenciar avisos de convergência e outros avisos repetitivos
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu" if TRANSFORMERS_AVAILABLE else "cpu"

    def fit(self, X, y):
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library not found.")
        
        # Simplificação: assume que X são textos. Se X for processado, 
        # precisaríamos reverter ou passar textos originais.
        # Para este wrapper, vamos assumir que fit recebe textos se possível.
        # No AutoML atual, X é processado pelo processor. 
        # Isso é um desafio para Transformers.
        
        num_labels = len(np.unique(y)) if self.task == 'classification' else 1
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels).to(self.device)
        
        # Lógica de treino simplificada para o wrapper
        # Em um cenário real, converteríamos X e y para Dataset do HF
        # Aqui, vamos apenas simular o treino ou fazer um fit dummy se necessário
        # Para integracao real, precisariamos de TrainingArguments com self.learning_rate e self.epochs
        
        return self

    def predict(self, X):
        # Lógica de inferência simplificada
        return np.zeros(len(X))

class AutoMLDataProcessor:
    def __init__(self, target_column=None, task_type=None, date_col=None, forecast_horizon=1, nlp_config=None, scaler_type='standard'):
        self.target_column = target_column
        self.task_type = task_type
        self.date_col = date_col
        self.forecast_horizon = forecast_horizon
        self.nlp_config = nlp_config if nlp_config else {}
        self.scaler_type = scaler_type
        self.preprocessor = None
        self.label_encoder = None

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
        """Applies TF-IDF or CountVectorizer to text columns with custom config."""
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        df = df.copy()
        
        # Get NLP configs
        vectorizer_type = self.nlp_config.get('vectorizer', 'tfidf')
        ngram_range = self.nlp_config.get('ngram_range', (1, 2))
        max_features = self.nlp_config.get('max_features', 5000)
        stop_words = 'english' if self.nlp_config.get('stop_words', True) else None
        
        for col in nlp_cols:
            if col in df.columns:
                logger.info(f"🔤 Otimizando NLP para a coluna: {col}")
                # Limpeza básica
                df[col] = df[col].astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True)
                
                # Vectorizer selection
                if vectorizer_type == 'count':
                    vectorizer = CountVectorizer(
                        max_features=max_features,
                        ngram_range=ngram_range,
                        stop_words=stop_words
                    )
                else: # Default to TF-IDF
                    vectorizer = TfidfVectorizer(
                        max_features=max_features,
                        ngram_range=ngram_range,
                        stop_words=stop_words,
                        sublinear_tf=True
                    )
                
                # Lemmatization (Basic support)
                if self.nlp_config.get('lemmatization', False):
                    try:
                        import nltk
                        from nltk.stem import WordNetLemmatizer
                        try:
                            nltk.data.find('corpora/wordnet')
                        except LookupError:
                            nltk.download('wordnet')
                            nltk.download('omw-1.4')
                        
                        lemmatizer = WordNetLemmatizer()
                        # Apply lemmatization to the column before vectorization
                        # This is a simple apply, for better results we'd need POS tagging
                        df[col] = df[col].apply(lambda x: ' '.join([lemmatizer.lemmatize(w) for w in str(x).split()]))
                    except Exception as e:
                        logger.warning(f"Lemmatization failed: {e}")

                text_features = vectorizer.fit_transform(df[col])
                
                # Handle sparse matrix
                if hasattr(text_features, 'toarray'):
                    text_features = text_features.toarray()
                    
                tfidf_df = pd.DataFrame(
                    text_features, 
                    columns=[f"{vectorizer_type}_{col}_{i}" for i in range(text_features.shape[1])],
                    index=df.index
                )
                df = pd.concat([df, tfidf_df], axis=1)
                df = df.drop(columns=[col])
        return df

    def fit_transform(self, df, nlp_cols=None):
        # NLP Feature Engineering
        if nlp_cols is not None and isinstance(nlp_cols, list) and len(nlp_cols) > 0:
            df = self._apply_nlp_features(df, nlp_cols)

        # Time Series Feature Engineering
        if self.task_type == 'time_series':
            df = self._apply_ts_features(df)

        if self.target_column and self.target_column in df.columns:
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
        else:
            X = df
            y = None
        
        # Exclude date_col from processing if it's still there
        # (Exclude date_col from processing if it's still there)
        process_cols = [c for c in X.columns if c != self.date_col]
        X_to_process = X[process_cols]

        # Identificar tipos de colunas ANTES de qualquer transformação
        numeric_features = X_to_process.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
        all_categorical = X_to_process.select_dtypes(include=['object', 'category']).columns.tolist()

        # Drop constant columns
        constant_cols = [col for col in X_to_process.columns if X_to_process[col].nunique() <= 1]
        if constant_cols:
            X_to_process = X_to_process.drop(columns=constant_cols)
            # Atualizar listas após drop
            numeric_features = [c for c in numeric_features if c not in constant_cols]
            all_categorical = [c for c in all_categorical if c not in constant_cols]
        
        # Split categorical into low and high cardinality to avoid memory explosion
        low_card_features = []
        high_card_features = []
        
        for col in all_categorical:
            if X_to_process[col].nunique() <= 15:
                low_card_features.append(col)
            else:
                high_card_features.append(col)

        # Preprocessing for numeric data
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

        # Preprocessing for low cardinality categorical data
        low_card_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Preprocessing for high cardinality categorical data
        high_card_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        # Bundle preprocessing
        transformers = [('num', numeric_transformer, numeric_features)]
        
        if low_card_features:
            transformers.append(('cat_low', low_card_transformer, low_card_features))
        if high_card_features:
            transformers.append(('cat_high', high_card_transformer, high_card_features))

        self.preprocessor = ColumnTransformer(transformers=transformers, sparse_threshold=0)

        X_processed = self.preprocessor.fit_transform(X)
        
        if X_processed.shape[0] == 0:
            raise ValueError("O processamento resultou em 0 linhas. Verifique se o horizonte de previsão e lags são maiores que o dataset.")
            
        # Ensure output is dense and float64
        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()
        X_processed = X_processed.astype(np.float64)
        
        # Handle target encoding if categorical
        y_processed = None
        if y is not None:
            if y.dtype == 'object' or y.dtype.name == 'category':
                self.label_encoder = LabelEncoder()
                y_processed = self.label_encoder.fit_transform(y)
            else:
                y_processed = y

        return X_processed, y_processed

    def transform(self, df):
        # Time Series Feature Engineering
        if self.task_type == 'time_series':
            df = self._apply_ts_features(df)

        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return None, None

        if self.target_column and self.target_column in df.columns:
            X = df.drop(columns=[self.target_column])
            y = df[self.target_column]
            # Handle categorical target
            if self.label_encoder:
                y = self.label_encoder.transform(y)
        else:
            X = df
            y = None
        
        # Ensure X is a DataFrame (required for ColumnTransformer with named columns)
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Exclude date_col if present
        # Note: We don't drop it here because preprocessor was fitted with it present (and it drops it internally)
        # unless it was excluded in fit_transform. Let's keep consistency.
        # In fit_transform, X still has date_col when calling preprocessor.fit_transform(X).

        try:
            X_processed = self.preprocessor.transform(X)
        except Exception as e:
            logger.error(f"Erro no ColumnTransformer.transform: {e}")
            # Se falhar, tenta garantir que as colunas batem com o esperado pelo transformer
            # O sklearn 1.2+ é rigoroso com nomes de colunas
            raise e

        if hasattr(X_processed, "toarray"):
            X_processed = X_processed.toarray()
            
        X_final = X_processed.astype(np.float64)
        
        return X_final, y

    def get_feature_names(self):
        """Returns the names of the features after preprocessing."""
        if self.preprocessor is None:
            return []
        
        feature_names = []
        for name, transformer, columns in self.preprocessor.transformers_:
            if name == 'remainder' and transformer == 'drop':
                continue
            
            if hasattr(transformer, 'get_feature_names_out'):
                # Para OneHotEncoder e outros que mudam o número de colunas
                names = transformer.get_feature_names_out(columns)
                feature_names.extend(names)
            else:
                # Para StandardScaler, SimpleImputer, etc. que mantêm o número de colunas
                feature_names.extend(columns)
        
        return feature_names

class AutoMLTrainer:
    def __init__(self, task_type='classification', preset='medium'):
        self.task_type = task_type
        self.preset = preset
        self.best_model = None
        self.best_params = None
        self.results = []
        
        # Configurações baseadas no preset
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
            'best_quality': {
                'n_trials': 150,
                'timeout': 7200, # 2 hours
                'cv': 10,
                'models': ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'catboost', 'svm', 'mlp', 'extra_trees', 'adaboost', 'sgd_classifier']
            },
            'custom': {
                'n_trials': 20, # Default if not provided manually
                'timeout': 600,
                'cv': 5,
                'models': ['logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'extra_trees'] # Default selection
            }
        }

    def _get_models(self, trial=None, name=None, random_state=None):
        """
        Retorna a lista de nomes dos modelos ou uma instância específica com parâmetros sugeridos.
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
                    loss=t.suggest_categorical('sgd_loss', ['hinge', 'log_loss', 'modified_huber']),
                    penalty=t.suggest_categorical('sgd_penalty', ['l2', 'l1', 'elasticnet']),
                    alpha=t.suggest_float('sgd_alpha', 1e-6, 1e-2, log=True),
                    max_iter=2000, 
                    random_state=random_state
                ),
                'mlp': lambda t: MLPClassifier(
                    hidden_layer_sizes=eval(t.suggest_categorical('mlp_layers', ["(50,)", "(100,)", "(100, 50)", "(100, 100)", "(50, 50, 50)", "(256, 128, 64)"])),
                    activation=t.suggest_categorical('mlp_activation', ['relu', 'tanh', 'logistic']),
                    alpha=t.suggest_float('mlp_alpha', 1e-6, 1e-1, log=True),
                    learning_rate_init=t.suggest_float('mlp_lr', 1e-5, 1e-1, log=True),
                    max_iter=1000,
                    early_stopping=True,
                    random_state=random_state
                ),
                'catboost': lambda t: cb.CatBoostClassifier(
                    iterations=t.suggest_int('cb_iterations', 100, 1000),
                    learning_rate=t.suggest_float('cb_lr', 0.001, 0.3, log=True),
                    depth=t.suggest_int('cb_depth', 4, 10),
                    l2_leaf_reg=t.suggest_float('cb_l2', 1, 10),
                    verbose=0,
                    thread_count=1,
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
                    hidden_layer_sizes=eval(t.suggest_categorical('mlp_layers', ["(50,)", "(100,)", "(50, 50)", "(100, 50)"])),
                    max_iter=t.suggest_int('mlp_max_iter', 200, 500),
                    early_stopping=True,
                    validation_fraction=0.1,
                    n_iter_no_change=10,
                    random_state=random_state
                ),
                'catboost': lambda t: cb.CatBoostRegressor(
                    iterations=t.suggest_int('cb_iterations', 50, 200),
                    learning_rate=t.suggest_float('cb_lr', 0.01, 0.3),
                    depth=t.suggest_int('cb_depth', 3, 10),
                    verbose=0,
                    thread_count=1,
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
        else:
            return {}

        if name is None:
            # Filtra modelos que retornam None (ex: bibliotecas não instaladas)
            available = []
            for k, v in models_config.items():
                try:
                    if v(None) is not None:
                        available.append(k)
                except:
                    # Se a lambda falhar com None (ex: t.suggest_int), assumimos que o modelo está disponível
                    # pois o erro vem da lógica do trial, não da ausência da biblioteca
                    available.append(k)
            
            if hasattr(self, 'custom_models') and self.custom_models:
                available.extend(list(self.custom_models.keys()))
            return available
        
        if name in models_config:
            # Passamos o trial para a lambda para instanciar o modelo com os parâmetros sugeridos
            return models_config[name](trial)
        
        return None

    def get_available_models(self):
        """Returns a list of available model names for the current task type."""
        return self._get_models()

    def train(self, X_train, y_train=None, n_trials=None, timeout=None, callback=None, selected_models=None, early_stopping_rounds=None, experiment_name="AutoML_Experiment", manual_params=None, random_state=42, validation_strategy='cv', validation_params=None, custom_models=None, **kwargs):
        # Usar configurações do preset se n_trials/timeout não forem fornecidos
        preset_config = self.preset_configs.get(self.preset, self.preset_configs['medium'])
        n_trials = n_trials if n_trials is not None else preset_config['n_trials']
        timeout = timeout if timeout is not None else preset_config.get('timeout')
        
        # Armazenar modelos customizados (uploaded ou registrados)
        self.custom_models = custom_models if custom_models else {}
        
        # Se selected_models não for fornecido, usa a lista do preset
        if selected_models is None:
            selected_models = preset_config['models']
            
        self.ts_metadata = kwargs if self.task_type == 'time_series' else {}
        self.random_state = random_state
        
        # Compatibilidade com parâmetro antigo auto_split
        if kwargs.get('auto_split', False):
            validation_strategy = 'auto_split'
            
        if validation_params is None:
            validation_params = {}
        
        from mlops_utils import MLFlowTracker
        tracker = MLFlowTracker(experiment_name)

        # Early Stopping & Summary Logic
        best_score_so_far = -np.inf
        trials_without_improvement = 0
        model_trial_counts = {} # Corrigir índice do gráfico: contador real por modelo
        self.model_summaries = {} # Armazenar melhor métrica de cada modelo

        def objective(trial, forced_model=None):
            nonlocal best_score_so_far, trials_without_improvement
            
            # 1. Determinar o modelo a ser usado
            all_available = selected_models if selected_models else self.get_available_models()
            if not all_available: all_available = self.get_available_models()

            if forced_model:
                model_name = trial.suggest_categorical('model_name', all_available)
                if model_name != forced_model:
                    model_name = forced_model
            else:
                model_name = trial.suggest_categorical('model_name', all_available)

            # Identificador único para este trial de modelo
            model_trial_counts[model_name] = model_trial_counts.get(model_name, 0) + 1
            trial_num_for_model = model_trial_counts[model_name]
            full_trial_name = f"{model_name} - Trial {trial_num_for_model}"
            
            # Determinar a seed específica para este modelo
            current_seed = self.random_state
            if isinstance(self.random_state, dict):
                current_seed = self.random_state.get(model_name, 42)
            
            logger.info(f"📍 Trial {trial.number} mapeado para {full_trial_name} (Seed: {current_seed})")

            run_id = None

            # Early Stopping Global
            min_improvement = 0.0001
            if early_stopping_rounds and trials_without_improvement >= early_stopping_rounds:
                trial.study.stop()
                return 0

            # 2. Instanciar o modelo específico sugerido para este trial (Lazy Loading)
            model = self._get_models(trial=trial, name=model_name, random_state=current_seed)
            
            if model is None:
                return -1.0
            
            # Lógica de Validação e Split de Dados
            X_tr, X_val, y_tr, y_val = None, None, None, None
            
            # Apenas para métodos que usam holdout explícito (auto_split ou holdout manual)
            use_explicit_validation = validation_strategy in ['auto_split', 'holdout']
            
            if use_explicit_validation and self.task_type in ['classification', 'regression', 'time_series']:
                if validation_strategy == 'auto_split':
                    split_ratio = trial.suggest_float('data_split_ratio', 0.6, 0.9)
                else: # holdout
                    test_size = validation_params.get('test_size', 0.2)
                    split_ratio = 1.0 - test_size
                
                if not hasattr(self, '_split_cache'): self._split_cache = {}
                cache_key = f"{split_ratio}_{self.task_type}_{current_seed}_{validation_strategy}"
                
                if cache_key in self._split_cache:
                    X_tr, X_val, y_tr, y_val = self._split_cache[cache_key]
                else:
                    if self.task_type == 'time_series':
                        split_idx = int(len(X_train) * split_ratio)
                        if isinstance(X_train, pd.DataFrame):
                            X_tr, X_val = X_train.iloc[:split_idx], X_train.iloc[split_idx:]
                            y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]
                        else:
                            X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
                            y_tr, y_val = y_train[:split_idx], y_train[split_idx:]
                    else:
                        if y_train is not None:
                            X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, train_size=split_ratio, random_state=current_seed)
                        else:
                            X_tr, X_val = train_test_split(X_train, train_size=split_ratio, random_state=current_seed)
                            y_tr, y_val = None, None
                    if len(self._split_cache) > 5: self._split_cache.clear()
                    self._split_cache[cache_key] = (X_tr, X_val, y_tr, y_val)
            else:
                # Para CV, usamos os dados completos no cross_val_score
                X_tr, y_tr = X_train, y_train
                X_val, y_val = None, None

            start_time = time.time()
            logger.info(f"⏳ Treinando {full_trial_name}...")
            trial_metrics = {}
            trial_params = trial.params.copy()
            trial_params['task_type'] = self.task_type
            
            try:
                if self.task_type in ['classification', 'regression', 'time_series']:
                    if use_explicit_validation:
                        model.fit(X_tr, y_tr)
                        y_pred_val = model.predict(X_val)
                        if self.task_type == 'classification':
                            score = accuracy_score(y_val, y_pred_val)
                            trial_metrics['accuracy'] = score
                        else:
                            score = r2_score(y_val, y_pred_val)
                            trial_metrics['r2'] = score
                    else:
                        # Cross Validation Logic
                        n_splits = validation_params.get('folds', 3) if validation_params else 3
                        # Fallback se folds não estiver definido
                        if not isinstance(n_splits, int): n_splits = 3
                        
                        if self.task_type == 'time_series' or validation_strategy == 'time_series_cv':
                            cv = TimeSeriesSplit(n_splits=n_splits)
                            scoring = 'r2'
                        elif self.task_type == 'classification':
                            scoring = 'accuracy'
                            if validation_strategy == 'stratified_cv':
                                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=current_seed)
                            else:
                                cv = KFold(n_splits=n_splits, shuffle=True, random_state=current_seed)
                        else: # Regression
                            scoring = 'r2'
                            cv = KFold(n_splits=n_splits, shuffle=True, random_state=current_seed)
                            
                        score = cross_val_score(model, X_tr, y_tr, cv=cv, scoring=scoring).mean()
                        trial_metrics[scoring] = score
                        
                        # Para salvar o modelo e artefatos, precisamos dar fit no treino completo do trial
                        # Nota: Fit final no trial_set sem CV para logging
                        logger.info(f"✨ Finalizando treino do modelo {full_trial_name}...")
                        model.fit(X_tr, y_tr)
                        logger.info(f"✅ Treino finalizado para {full_trial_name}")
                        
                elif self.task_type == 'anomaly_detection':
                    model.fit(X_tr)
                    if hasattr(model, 'decision_function'):
                        score = model.decision_function(X_tr).mean()
                    else:
                        score = 0
                    trial_metrics['decision_score'] = score
                else: # clustering
                    model.fit(X_tr)
                    labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X_tr)
                    if len(set(labels)) > 1:
                        score = silhouette_score(X_tr, labels)
                    else:
                        score = -1
                    trial_metrics['silhouette'] = score

                # Unified Logging for all tasks/strategies
                logger.info(f"📊 Registrando {full_trial_name} no MLflow...")
                run_id = tracker.log_experiment(
                    params=trial_params,
                    metrics=trial_metrics,
                    model=model,
                    model_name=full_trial_name,
                    register=False
                )
                logger.info(f"✔ {full_trial_name} registrado com ID: {run_id}")

            except Exception as e:
                logger.error(f"Error during trial for {model_name}: {e}")
                score = -1.0
                trial_metrics['error'] = 1.0

            duration = time.time() - start_time
            trial_metrics['duration'] = duration
            
            trial.set_user_attr("run_id", run_id)
            trial.set_user_attr("full_name", full_trial_name)

            # Atualizar resumo do modelo (melhor trial de cada algoritmo)
            if model_name not in self.model_summaries or score > self.model_summaries[model_name]['score']:
                self.model_summaries[model_name] = {
                    'score': score,
                    'metrics': trial_metrics,
                    'params': trial_params,
                    'duration': duration,
                    'trial_name': full_trial_name
                }

            if callback:
                callback(trial, score, full_trial_name, duration, trial_metrics)
                
            if score > (best_score_so_far + min_improvement):
                best_score_so_far = score
                trials_without_improvement = 0
            else:
                trials_without_improvement += 1

            return score

        # All our metrics are better when larger
        direction = 'maximize'
        
        # Determinar seed inicial para o sampler
        sampler_seed = self.random_state if isinstance(self.random_state, int) else 42
        
        study = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(n_startup_trials=max(n_trials // 3, 5), seed=sampler_seed)
        )
        
        # Identificar modelos estáticos (sem hiperparâmetros para otimizar)
        static_models = {
            'naive_bayes', 'ridge_classifier', 
            'linear_regression', 'mean_shift',
            'elliptic_envelope'
        }
        
        models_to_tune = selected_models if selected_models else self.get_available_models()
        
        for m_name in models_to_tune:
            # Se a seed for por modelo, atualizamos o sampler para garantir reprodutibilidade por modelo
            if isinstance(self.random_state, dict):
                model_seed = self.random_state.get(m_name, 42)
                study.sampler = optuna.samplers.TPESampler(n_startup_trials=min(n_trials, 10), seed=model_seed)
                logger.info(f"🎲 Sampler seed atualizado para {model_seed} (Modelo: {m_name})")
            # Se houver parâmetros manuais para este modelo, enfileira uma tentativa com eles
            if manual_params and manual_params.get('model_name') == m_name:
                p = {'model_name': m_name}
                p.update({k: v for k, v in manual_params.items() if k != 'model_name'})
                study.enqueue_trial(p)
                logger.info(f"💉 Enfileirando tentativa manual para {m_name}")

            # Se o modelo for estático, rodamos apenas 1 vez
            current_n_trials = 1 if m_name in static_models else n_trials
            
            trials_without_improvement = 0 
            logger.info(f"🚀 Iniciando otimização para o modelo: {m_name} ({current_n_trials} trials, Timeout: {timeout}s)")
            study.optimize(lambda t: objective(t, forced_model=m_name), n_trials=current_n_trials, timeout=timeout)
            # Resetar o flag de parada para permitir que o próximo modelo seja otimizado
            # O study.stop() define este flag como True
            if hasattr(study, '_stop_flag'):
                study._stop_flag = False
            logger.info(f"✅ Otimização para {m_name} finalizada.")
        
        self.best_params = study.best_params
        best_model_name = self.best_params.get('model_name')
        
        logger.info(f"🏆 Melhor modelo global encontrado: {best_model_name}")
        logger.info(f"📊 Melhores parâmetros: {self.best_params}")
        
        if self.task_type == 'time_series':
            self.best_params.update(self.ts_metadata)
        
        self.best_model = self._instantiate_model(best_model_name, self.best_params)
        
        # Reativar probabilidade para o modelo final se for SVM
        if best_model_name == 'svm' and hasattr(self.best_model, 'probability'):
            self.best_model.set_params(probability=True)
            logger.info("🔮 Ativando estimativa de probabilidade para o modelo SVM final.")
        
        # Garantir a mesma seed no modelo final
        final_model_seed = self.random_state
        if isinstance(self.random_state, dict):
            final_model_seed = self.random_state.get(best_model_name, 42)
            
        if hasattr(self.best_model, 'random_state'):
            self.best_model.set_params(random_state=final_model_seed)
        elif hasattr(self.best_model, 'random_seed'):
            self.best_model.set_params(random_seed=final_model_seed)

        if y_train is not None:
            self.best_model.fit(X_train, y_train)
        else:
            self.best_model.fit(X_train)
            
        # Adicionar Feature Importance se disponível
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = self.best_model.feature_importances_.tolist()
            logger.info("📈 Feature Importance calculada.")
        elif hasattr(self.best_model, 'coef_'):
            self.feature_importance = np.abs(self.best_model.coef_).flatten().tolist()
            logger.info("📈 Feature Importance calculada (baseada em coeficientes).")
        else:
            self.feature_importance = None

        return self.best_model

    def get_supported_models(self):
        """Returns a list of supported model names for the current task type."""
        models = []
        if self.task_type == 'classification':
            models = [
                'logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 
                'catboost', 'extra_trees', 'adaboost', 'decision_tree', 
                'svm', 'knn', 'naive_bayes', 'sgd_classifier', 'mlp', 'linear_svc', 'ridge_classifier'
            ]
            if TRANSFORMERS_AVAILABLE:
                models.extend([
                    'bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 
                    'albert-base-v2', 'xlnet-base-cased', 'microsoft/deberta-v3-base'
                ])
            return models
        elif self.task_type == 'regression':
            models = [
                'linear_regression', 'random_forest', 'xgboost', 'lightgbm', 
                'catboost', 'extra_trees', 'adaboost', 'decision_tree', 
                'svm', 'knn', 'ridge', 'lasso', 'elastic_net', 
                'sgd_regressor', 'mlp'
            ]
            if TRANSFORMERS_AVAILABLE:
                models.extend([
                    'bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 
                    'microsoft/deberta-v3-small'
                ])
            return models
        elif self.task_type == 'clustering':
            return [
                'kmeans', 'dbscan', 'agglomerative', 'gaussian_mixture', 
                'spectral', 'mean_shift', 'birch'
            ]
        elif self.task_type == 'anomaly_detection':
            return [
                'isolation_forest', 'one_class_svm', 'local_outlier_factor', 
                'elliptic_envelope'
            ]
        return []

    def create_model_instance(self, model_name, params=None):
        """Creates a model instance with given parameters (no prefixes expected)."""
        if params is None:
            params = {}
            
        # Clean params (remove None values or empty strings that might come from UI)
        clean_params = {k: v for k, v in params.items() if v is not None and v != ""}
        
        try:
            if self.task_type == 'classification':
                if model_name == 'logistic_regression': return LogisticRegression(max_iter=2000, **clean_params)
                if model_name == 'random_forest': return RandomForestClassifier(**clean_params)
                if model_name == 'xgboost': return xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **clean_params)
                if model_name == 'lightgbm': return lgb.LGBMClassifier(verbosity=-1, **clean_params)
                if model_name == 'catboost' and CATBOOST_AVAILABLE: return cb.CatBoostClassifier(verbose=0, **clean_params)
                if model_name == 'extra_trees': return ExtraTreesClassifier(**clean_params)
                if model_name == 'adaboost': return AdaBoostClassifier(**clean_params)
                if model_name == 'decision_tree': return DecisionTreeClassifier(**clean_params)
                if model_name == 'svm': return SVC(probability=True, **clean_params)
                if model_name == 'knn': return KNeighborsClassifier(**clean_params)
                if model_name == 'naive_bayes': return GaussianNB(**clean_params)
                if model_name == 'sgd_classifier': return SGDClassifier(max_iter=1000, **clean_params)
                if model_name == 'mlp': return MLPClassifier(max_iter=1000, **clean_params)

            elif self.task_type == 'regression':
                if model_name == 'linear_regression': return LinearRegression(**clean_params)
                if model_name == 'random_forest': return RandomForestRegressor(**clean_params)
                if model_name == 'xgboost': return xgb.XGBRegressor(objective='reg:squarederror', **clean_params)
                if model_name == 'lightgbm': return lgb.LGBMRegressor(verbosity=-1, **clean_params)
                if model_name == 'catboost' and CATBOOST_AVAILABLE: return cb.CatBoostRegressor(verbose=0, **clean_params)
                if model_name == 'extra_trees': return ExtraTreesRegressor(**clean_params)
                if model_name == 'adaboost': return AdaBoostRegressor(**clean_params)
                if model_name == 'decision_tree': return DecisionTreeRegressor(**clean_params)
                if model_name == 'svm': return SVR(**clean_params)
                if model_name == 'knn': return KNeighborsRegressor(**clean_params)
                if model_name == 'ridge': return Ridge(**clean_params)
                if model_name == 'lasso': return Lasso(**clean_params)
                if model_name == 'elastic_net': return ElasticNet(**clean_params)
                if model_name == 'sgd_regressor': return SGDRegressor(max_iter=1000, **clean_params)
                if model_name == 'mlp': return MLPRegressor(max_iter=1000, **clean_params)

            elif self.task_type == 'clustering':
                if model_name == 'kmeans': return KMeans(n_init=10, **clean_params)
                if model_name == 'dbscan': return DBSCAN(**clean_params)
                if model_name == 'agglomerative': return AgglomerativeClustering(**clean_params)
                if model_name == 'gaussian_mixture': return GaussianMixture(**clean_params)
                if model_name == 'spectral': return SpectralClustering(**clean_params)
                if model_name == 'mean_shift': return MeanShift(**clean_params)
                if model_name == 'birch': return Birch(**clean_params)

            elif self.task_type == 'anomaly_detection':
                if model_name == 'isolation_forest': return IsolationForest(**clean_params)
                if model_name == 'one_class_svm': return OneClassSVM(**clean_params)
                if model_name == 'local_outlier_factor': return LocalOutlierFactor(novelty=True, **clean_params)
                if model_name == 'elliptic_envelope': return EllipticEnvelope(**clean_params)
                
        except Exception as e:
            logger.error(f"Error creating model instance for {model_name}: {e}")
            return None
            
        return None

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
        elif self.task_type == 'regression':
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

    def train_manual(self, X_train, y_train, model_name, params, **kwargs):
        """Trains a specific model with user-provided hyperparameters."""
        # Time Series specific feature engineering (Simple Lags)
        if self.task_type == 'time_series' and y_train is not None:
            horizon = kwargs.get('forecast_horizon', 1)
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train.copy()
                for i in range(1, horizon + 1):
                    X_train[f'lag_{i}'] = y_train.shift(i)
                X_train = X_train.fillna(method='bfill')

        self.best_model = self._instantiate_model(model_name, params)
        if y_train is not None:
            self.best_model.fit(X_train, y_train)
        else:
            self.best_model.fit(X_train)
        self.best_params = params
        return self.best_model

def get_technical_explanation(model_name, params, task_type):
    explanations = {
        'classification': {
            'random_forest': "Random Forest é um conjunto de árvores de decisão. É robusto a outliers e reduz o overfitting combinando múltiplas árvores.",
            'xgboost': "XGBoost usa Gradient Boosting extremo. É eficiente e poderoso, corrigindo erros de árvores anteriores sequencialmente.",
            'lightgbm': "LightGBM é um framework de gradient boosting que usa algoritmos baseados em árvores. É projetado para ser distribuído e eficiente.",
            'extra_trees': "Extra Trees é semelhante ao Random Forest, mas usa divisões aleatórias mais extremas, o que pode reduzir ainda mais a variância.",
            'adaboost': "AdaBoost foca em instâncias que foram classificadas incorretamente por modelos anteriores, construindo um classificador forte a partir de fracos.",
            'decision_tree': "Árvore de Decisão é um modelo simples e interpretável que divide os dados em ramos baseados em características.",
            'svm': "SVM encontra o hiperplano que melhor separa as classes no espaço de características.",
            'mlp': "MLP é uma rede neural artificial básica capaz de aprender relações não lineares complexas.",
            'logistic_regression': "Regressão Logística é um modelo estatístico usado para predição de classes binárias ou multiclasse.",
            'linear_svc': "LinearSVC é uma implementação rápida de SVM com kernel linear, eficiente para grandes datasets.",
            'knn': "K-Nearest Neighbors (KNN) classifica instâncias com base na classe majoritária de seus vizinhos mais próximos.",
            'naive_bayes': "Naive Bayes é um classificador probabilístico baseado no Teorema de Bayes com suposições de independência forte.",
            'ridge_classifier': "Ridge Classifier usa regressão ridge para classificação, aplicando regularização L2.",
            'sgd_classifier': "SGD Classifier usa Gradiente Descendente Estocástico, sendo ideal para aprendizado online e grandes volumes de dados."
        },
        'regression': {
            'random_forest': "Random Forest para regressão combina múltiplas árvores para prever valores contínuos com alta estabilidade.",
            'xgboost': "XGBoost para regressão oferece alta performance e regularização para evitar overfitting em dados tabulares.",
            'lightgbm': "LightGBM é extremamente rápido para regressão em grandes datasets.",
            'extra_trees': "Extra Trees Regressor oferece um nível extra de aleatoriedade, sendo útil quando os dados têm muito ruído.",
            'adaboost': "AdaBoost Regressor ajusta pesos de predições anteriores para melhorar a precisão em valores contínuos.",
            'decision_tree': "Árvore de Decisão para regressão prevê valores baseando-se na média dos dados em cada folha.",
            'svm': "SVR tenta encontrar uma função que se desvie de y no máximo uma pequena quantidade para todos os dados de treino.",
            'mlp': "MLP para regressão pode capturar padrões numéricos altamente complexos e não lineares.",
            'linear_regression': "Regressão Linear é o modelo base que assume uma relação linear entre as variáveis.",
            'knn': "KNN Regressor estima o valor alvo baseando-se na média dos valores de seus vizinhos mais próximos.",
            'ridge': "Ridge Regression aplica regularização L2 para evitar que os coeficientes se tornem muito grandes.",
            'lasso': "Lasso Regression aplica regularização L1, o que pode levar a coeficientes zero (seleção de características).",
            'elastic_net': "Elastic Net combina regularização L1 e L2, sendo útil quando há múltiplas características correlacionadas.",
            'sgd_regressor': "SGD Regressor aplica Gradiente Descendente Estocástico para problemas de regressão em larga escala."
        },
        'clustering': {
            'kmeans': "K-Means agrupa dados minimizando a variância dentro de cada cluster (distância aos centroides).",
            'agglomerative': "Clustering Hierárquico Aglomerativo constrói uma hierarquia de clusters de baixo para cima.",
            'dbscan': "DBSCAN identifica clusters baseados na densidade, sendo excelente para encontrar formas arbitrárias e ruído.",
            'gaussian_mixture': "Gaussian Mixture assume que los dados são gerados por uma mistura de várias distribuições gaussianas.",
            'mean_shift': "Mean Shift é um algoritmo baseado em densidade que não exige o número de clusters a priori.",
            'birch': "BIRCH é eficiente para clustering em grandes datasets, construindo uma árvore de características de cluster.",
            'spectral': "Spectral Clustering usa a conectividade dos dados para agrupar instâncias, útil para estruturas não convexas."
        },
        'time_series': {
            'random_forest_ts': "Random Forest adaptado para séries temporais usando lags e características temporais como preditores.",
            'xgboost_ts': "XGBoost para séries temporais foca em capturar tendências e sazonalidades através de boosting.",
            'extra_trees_ts': "Extra Trees para séries temporais ajuda a mitigar o ruído inerente a dados temporais."
        },
        'anomaly_detection': {
            'isolation_forest': "Isolation Forest isola anomalias selecionando aleatoriamente uma característica e um valor de divisão.",
            'local_outlier_factor': "LOF compara a densidade local de um ponto com a de seus vizinhos para identificar outliers.",
            'elliptic_envelope': "Assume que os dados normais vêm de uma distribuição gaussiana e detecta o que está fora do elipsoide.",
            'one_class_svm': "Aprende uma fronteira que envolve a maioria dos dados normais, classificando o que está fora como anomalia."
        }
    }
    
    model_desc = explanations.get(task_type, {}).get(model_name, "Modelo selecionado por sua performance superior durante a otimização.")
    
    param_desc = []
    for k, v in params.items():
        if 'n_estimators' in k: param_desc.append(f"- **Estimadores ({v})**: Número de árvores no conjunto. Mais árvores aumentam a robustez, mas o custo computacional sobe.")
        if 'max_depth' in k: param_desc.append(f"- **Profundidade Máxima ({v})**: Limita o crescimento da árvore para evitar overfitting.")
        if 'learning_rate' in k or 'lr' in k: param_desc.append(f"- **Taxa de Aprendizado ({v})**: Controla a magnitude do ajuste dos pesos a cada iteração.")
        if 'n_clusters' in k: param_desc.append(f"- **Número de Clusters ({v})**: Quantidade de grupos que o algoritmo tentará identificar nos dados.")
        if 'contamination' in k: param_desc.append(f"- **Contaminação ({v})**: A proporção esperada de anomalias no conjunto de dados.")
        if 'n_neighbors' in k or 'neighbors' in k: param_desc.append(f"- **Vizinhos ({v})**: Número de vizinhos a serem usados para calcular a densidade local.")
        
    return model_desc, "\n".join(param_desc)

def save_pipeline(processor, model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump({'processor': processor, 'model': model}, path)

def load_pipeline(path):
    data = joblib.load(path)
    return data['processor'], data['model']
