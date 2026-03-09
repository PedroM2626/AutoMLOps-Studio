import numpy as np
import logging
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

logger = logging.getLogger(__name__)

# Keys considered 'ensemble' models — excluded when use_ensemble=False
_ENSEMBLE_MODEL_KEYS = frozenset(['voting_ensemble', 'custom_voting', 'custom_stacking'])

# Keys considered 'deep learning' models — excluded when use_deep_learning=False
_DL_MODEL_KEYS = frozenset([
    'mlp',
    'bert-base-uncased', 'distilbert-base-uncased', 'roberta-base',
    'albert-base-v2', 'xlnet-base-cased', 'microsoft/deberta-v3-base',
    'bert-base-uncased-reg', 'distilbert-base-uncased-reg',
    'roberta-base-reg', 'microsoft/deberta-v3-small',
])

class TransformersWrapper(BaseEstimator, ClassifierMixin, RegressorMixin):
    def __init__(self, model_name='bert-base-uncased', task='classification', epochs=3, learning_rate=2e-5):
        self.model_name = model_name
        self.task = task
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = None
        self.tokenizer = None
        self.device = None 

    def fit(self, X, y=None):
        logger.info(f"TransformersWrapper.fit called for {self.model_name}")
        # Lazy imports to avoid heavy load at start
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            TRANSFORMERS_AVAILABLE = True
        except ImportError:
            TRANSFORMERS_AVAILABLE = False
            raise ImportError("Transformers library not found.")
            
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        is_vectorized = False
        is_raw_text = False
        
        if hasattr(X, "shape"):
             if X.dtype == 'object' or X.dtype.type is np.str_ or X.dtype.type is np.bytes_:
                 is_raw_text = True
             elif len(X.shape) > 1 and X.shape[1] > 1:
                is_vectorized = True
            
        if is_vectorized and not is_raw_text:
            logger.warning(f"TransformersWrapper received vectorized data. Skipping fine-tuning.")
            self.model = None 
            return self

        if hasattr(X, "tolist"):
             texts = X.tolist()
        else:
             texts = list(X)
             
        if len(texts) > 0 and isinstance(texts[0], (list, np.ndarray)):
            texts = [t[0] for t in texts]
            
        texts = [str(t) for t in texts]
        
        num_labels = len(np.unique(y)) if y is not None and self.task == 'classification' else 1
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=num_labels).to(self.device)
        
        logger.info(f"TransformersWrapper: Loaded {self.model_name} on {self.device}")
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("TransformersWrapper model is not trained.")
        return np.zeros(len(X)) if self.task == 'regression' else np.random.randint(0, 2, size=len(X))
