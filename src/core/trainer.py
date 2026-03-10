import numpy as np
import logging
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

logger = logging.getLogger(__name__)

# Keys considered 'ensemble' models — excluded when use_ensemble=False
_ENSEMBLE_MODEL_KEYS = frozenset([
    'voting_ensemble', 'stacking_ensemble',
    'custom_voting', 'custom_stacking', 'custom_bagging',
    'bagging', 'adaboost', 'hist_gradient_boosting'
])

# Keys considered 'deep learning' / neural network models — excluded when use_deep_learning=False
_DL_MODEL_KEYS = frozenset([
    'bert-base-uncased', 'distilbert-base-uncased', 'roberta-base',
    'albert-base-v2', 'xlnet-base-cased', 'microsoft/deberta-v3-base',
    'bert-base-uncased-reg', 'distilbert-base-uncased-reg',
    'roberta-base-reg', 'microsoft/deberta-v3-small',
])

# Regular neural networks that run on CPU/Sklearn but are still toggled by Deep Learning flag
_NEURAL_NET_KEYS = frozenset([
    'mlp'
])

# Human-readable display names for custom ensemble models
ENSEMBLE_DISPLAY_NAMES = {
    'custom_voting':   'Custom Voting Ensemble',
    'custom_stacking': 'Custom Stacking Ensemble',
    'custom_bagging':  'Custom Bagging Ensemble',
    'voting_ensemble': 'Voting Ensemble',
    'stacking_ensemble': 'Stacking Ensemble',
}


def get_ensemble_display_name(model_key: str, estimators: list = None) -> str:
    """Return human-readable display name for ensemble model keys.
    If estimators are provided, append them in parentheses.
    """
    base_name = ENSEMBLE_DISPLAY_NAMES.get(model_key, model_key)
    if estimators:
        # Clean up names for display (strip _regressor, etc)
        names = []
        for e in estimators:
            if isinstance(e, str):
                name = e.replace('_', ' ').title()
            else:
                name = e.__class__.__name__
            names.append(name)
        return f"{base_name} ({', '.join(names)})"
    return base_name


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
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from torch.optim import AdamW
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("torch and transformers libraries are required for deep learning models.")

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Detect input type
        is_vectorized = False
        is_raw_text = False

        if hasattr(X, "shape"):
            # NumPy 2.0 compatible dtype checking
            is_str_dtype = np.issubdtype(X.dtype, np.str_) or X.dtype == object
            is_bytes_dtype = np.issubdtype(X.dtype, np.bytes_)
            
            if is_str_dtype or is_bytes_dtype:
                is_raw_text = True
            elif len(X.shape) > 1 and X.shape[1] > 1:
                is_vectorized = True

        if is_vectorized and not is_raw_text:
            logger.warning(f"TransformersWrapper received vectorized/numeric data. "
                           f"Transformers need raw text. Skipping fine-tuning and returning a zeroed model.")
            # Store labels for fallback prediction
            self._classes = np.unique(y) if y is not None else np.array([0, 1])
            self.model = None
            return self

        # Convert to list of strings
        if hasattr(X, "tolist"):
            texts = X.tolist()
        else:
            texts = list(X)

        if len(texts) > 0 and isinstance(texts[0], (list, np.ndarray)):
            texts = [t[0] for t in texts]

        texts = [str(t) for t in texts]

        num_labels = len(np.unique(y)) if y is not None and self.task == 'classification' else 1
        self._classes = np.unique(y) if y is not None else np.array([0, 1])

        logger.info(f"TransformersWrapper: Loading {self.model_name} tokenizer & model (num_labels={num_labels})...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=num_labels
        ).to(self.device)

        # --- Real training loop ---
        max_length = 128
        batch_size = min(16, len(texts))
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

        self.model.train()
        logger.info(f"TransformersWrapper: Starting training for {self.epochs} epoch(s)...")

        for epoch in range(self.epochs):
            total_loss = 0.0
            n_batches = 0
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start:start + batch_size]
                encoding = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)

                labels_batch = None
                if y is not None:
                    batch_labels = y[start:start + batch_size]
                    if hasattr(batch_labels, 'tolist'):
                        batch_labels = batch_labels.tolist()
                    labels_batch = torch.tensor(batch_labels, dtype=torch.long).to(self.device)

                optimizer.zero_grad()
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_batch
                )
                loss = outputs.loss
                if loss is not None:
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            logger.info(f"TransformersWrapper: Epoch {epoch+1}/{self.epochs} — avg loss: {avg_loss:.4f}")

        self.model.eval()
        logger.info(f"TransformersWrapper: Training complete for {self.model_name}")
        return self

    def predict(self, X):
        if self.model is None:
            # Fallback: return majority class
            n = len(X) if hasattr(X, '__len__') else 1
            return np.zeros(n, dtype=int)

        try:
            import torch
        except ImportError:
            raise ImportError("torch is required for TransformersWrapper predictions.")

        if hasattr(X, "tolist"):
            texts = X.tolist()
        else:
            texts = list(X)
        if len(texts) > 0 and isinstance(texts[0], (list, np.ndarray)):
            texts = [t[0] for t in texts]
        texts = [str(t) for t in texts]

        max_length = 128
        batch_size = 16
        all_preds = []

        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start:start + batch_size]
                encoding = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                if self.task == 'regression':
                    preds = logits.squeeze(-1).cpu().numpy()
                else:
                    preds = logits.argmax(dim=-1).cpu().numpy()
                all_preds.extend(preds.tolist())

        return np.array(all_preds)

    def predict_proba(self, X):
        if self.model is None:
            n = len(X) if hasattr(X, '__len__') else 1
            n_cls = len(self._classes) if hasattr(self, '_classes') else 2
            return np.ones((n, n_cls)) / n_cls

        try:
            import torch
            import torch.nn.functional as F
        except ImportError:
            raise ImportError("torch is required for TransformersWrapper.")

        if hasattr(X, "tolist"):
            texts = X.tolist()
        else:
            texts = list(X)
        if len(texts) > 0 and isinstance(texts[0], (list, np.ndarray)):
            texts = [t[0] for t in texts]
        texts = [str(t) for t in texts]

        max_length = 128
        batch_size = 16
        all_proba = []

        self.model.eval()
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                batch_texts = texts[start:start + batch_size]
                encoding = self.tokenizer(
                    batch_texts,
                    truncation=True,
                    padding='max_length',
                    max_length=max_length,
                    return_tensors='pt'
                )
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                proba = F.softmax(outputs.logits, dim=-1).cpu().numpy()
                all_proba.extend(proba.tolist())

        return np.array(all_proba)
