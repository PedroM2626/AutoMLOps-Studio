"""
Unified Deep Learning module for Free MLOps.
Combines traditional DL, NLP, and advanced models into a single interface.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List, Union
from pathlib import Path
import time
from datetime import datetime

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import torch.nn.functional as F
except ImportError:
    raise ImportError("PyTorch não está instalado. Instale com: pip install torch")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support
except ImportError:
    raise ImportError("Scikit-learn não está instalado. Instale com: pip install scikit-learn")

try:
    import transformers
    from transformers import AutoTokenizer, AutoModel
    from transformers import BertModel, BertTokenizer
except ImportError:
    print("Transformers não disponível. Usando apenas TF-IDF.")


class UnifiedDeepLearningAutoML:
    """
    Unified AutoML for all Deep Learning tasks.
    Supports traditional DL, NLP, and advanced models.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Default configurations for all model types
        self.default_configs = {
            # Traditional Deep Learning
            "mlp": {
                "hidden_layers": [128, 64, 32],
                "dropout_rate": 0.2,
                "batch_size": 32,
                "epochs": 50,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "activation": "relu",
            },
            "cnn": {
                "conv_filters": [32, 64, 128],
                "dense_layers": [128, 64],
                "dropout_rate": 0.2,
                "batch_size": 32,
                "epochs": 50,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "activation": "relu",
            },
            "lstm": {
                "hidden_dim": 64,
                "num_layers": 2,
                "dropout_rate": 0.2,
                "bidirectional": True,
                "batch_size": 32,
                "epochs": 50,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "sequence_length": 50,
            },
            # NLP Models
            "text_cnn": {
                "embed_dim": 128,
                "num_filters": 100,
                "filter_sizes": [3, 4, 5],
                "dropout_rate": 0.1,
                "batch_size": 32,
                "epochs": 50,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "max_length": 512,
                "max_features": 10000,
            },
            "text_lstm": {
                "embed_dim": 128,
                "hidden_dim": 64,
                "num_layers": 2,
                "dropout_rate": 0.1,
                "bidirectional": True,
                "batch_size": 32,
                "epochs": 50,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "max_length": 512,
                "max_features": 10000,
            },
            "bert_classifier": {
                "model_name": "bert-base-uncased",
                "dropout_rate": 0.1,
                "freeze_bert": False,
                "batch_size": 16,
                "epochs": 10,
                "learning_rate": 2e-5,
                "optimizer": "adam",
                "max_length": 512,
            },
            # Advanced Models
            "tab_transformer": {
                "embedding_dim": 8,
                "num_heads": 8,
                "num_layers": 6,
                "hidden_dim": 128,
                "dropout_rate": 0.1,
                "batch_size": 32,
                "epochs": 50,
                "learning_rate": 0.001,
                "optimizer": "adam",
            }
        }
    
    def create_model(
        self,
        X_train: Union[pd.DataFrame, List[str]],
        y_train: Union[pd.Series, List[str]],
        X_val: Union[pd.DataFrame, List[str]] = None,
        y_val: Union[pd.Series, List[str]] = None,
        model_type: str = "mlp",
        framework: str = "pytorch",
        problem_type: str = "classification",
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Unified method to create any type of model.
        This is the main interface that handles all model types.
        """
        
        config = config or self.default_configs.get(model_type, {})
        
        # Handle different input types
        if isinstance(X_train, list) and isinstance(y_train, list):
            # NLP models with text data
            return self._create_nlp_model(
                X_train, y_train, X_val, y_val, model_type, problem_type, config
            )
        else:
            # Traditional DL models with tabular data
            return self._create_traditional_model(
                X_train, y_train, X_val, y_val, model_type, framework, problem_type, config
            )
    
    def _create_nlp_model(
        self,
        texts_train: List[str],
        labels_train: List[str],
        texts_val: List[str] = None,
        labels_val: List[str] = None,
        model_type: str = "text_cnn",
        problem_type: str = "classification",
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create NLP models (text_cnn, text_lstm, bert_classifier)."""
        
        config = config or self.default_configs.get(model_type, {})
        
        # Split data if validation not provided
        if texts_val is None:
            texts_train, texts_val, labels_train, labels_val = train_test_split(
                texts_train, labels_train, test_size=0.2, random_state=42
            )
        
        # Process text data
        processor = NLPProcessor(method="tfidf")
        X_train_processed, y_train_encoded = processor.fit_transform(texts_train, labels_train)
        X_val_processed, y_val_encoded = processor.transform(texts_val), processor.transform_labels(labels_val)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_processed)
        y_train_tensor = torch.LongTensor(y_train_encoded)
        X_val_tensor = torch.FloatTensor(X_val_processed)
        y_val_tensor = torch.LongTensor(y_val_encoded)
        
        # Create datasets and loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 32), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.get("batch_size", 32), shuffle=False)
        
        # Create model based on type
        vocab_size = processor.vocab_size
        num_classes = len(processor.label_encoder.classes_)
        
        if model_type == "text_cnn":
            model = TextCNN(vocab_size, num_classes, config)
        elif model_type == "text_lstm":
            model = TextLSTM(vocab_size, num_classes, config)
        elif model_type == "bert_classifier":
            model = BERTClassifier(num_classes, config)
        else:
            raise ValueError(f"NLP model type not supported: {model_type}")
        
        # Train model
        result = self._train_pytorch_model(
            model, train_loader, val_loader, problem_type, config
        )
        
        # Add NLP-specific metadata
        result["vocab_size"] = vocab_size
        result["class_names"] = processor.label_encoder.classes_.tolist()
        result["processor"] = processor
        
        return result
    
    def _create_traditional_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        model_type: str = "mlp",
        framework: str = "pytorch",
        problem_type: str = "classification",
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create traditional DL models (mlp, cnn, lstm, tab_transformer)."""
        
        config = config or self.default_configs.get(model_type, {})
        
        # Prepare data
        X_train_clean = X_train.fillna(X_train.mean()).select_dtypes(include=[np.number])
        X_val_clean = X_val.fillna(X_val.mean()).select_dtypes(include=[np.number])
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train_clean.values)
        y_train_tensor = torch.LongTensor(y_train.values) if problem_type == "classification" else torch.FloatTensor(y_train.values)
        X_val_tensor = torch.FloatTensor(X_val_clean.values)
        y_val_tensor = torch.LongTensor(y_val.values) if problem_type == "classification" else torch.FloatTensor(y_val.values)
        
        # Create datasets and loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=config.get("batch_size", 32), shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.get("batch_size", 32), shuffle=False)
        
        # Create model based on type
        input_shape = X_train_clean.shape[1]
        num_classes = len(y_train.unique()) if problem_type == "classification" else 1
        
        if model_type == "mlp":
            model = MLP(input_shape, num_classes, config)
        elif model_type == "cnn":
            model = CNN1D(input_shape, num_classes, config)
        elif model_type == "lstm":
            model = LSTM(input_shape, num_classes, config)
        elif model_type == "tab_transformer":
            # For tab transformer, we need to identify categorical features
            categorical_features = []
            categorical_cardinalities = []
            continuous_features = 0
            
            for col in X_train_clean.columns:
                if X_train_clean[col].nunique() < 20 and X_train_clean[col].dtype == 'int64':
                    categorical_features.append(continuous_features)
                    categorical_cardinalities.append(X_train_clean[col].nunique())
                else:
                    continuous_features += 1
            
            model = TabTransformer(
                continuous_features, categorical_features, categorical_cardinalities, num_classes, config
            )
        else:
            raise ValueError(f"Model type not supported: {model_type}")
        
        # Train model
        result = self._train_pytorch_model(
            model, train_loader, val_loader, problem_type, config
        )
        
        # Add traditional model metadata
        result["input_shape"] = input_shape
        result["feature_names"] = X_train_clean.columns.tolist()
        
        return result
    
    def _train_pytorch_model(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        problem_type: str,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Unified training method for all PyTorch models."""
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        
        # Setup training
        criterion = self._get_criterion(problem_type)
        optimizer = self._get_optimizer(model, config)
        
        # Training loop
        start_time = time.time()
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        for epoch in range(config.get("epochs", 50)):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                if problem_type == "classification":
                    if outputs.shape[1] == 1:  # Binary classification
                        loss = criterion(outputs.squeeze(), batch_y.float())
                        predicted = (outputs.squeeze() > 0.5).long()
                    else:  # Multi-class
                        loss = criterion(outputs, batch_y)
                        predicted = outputs.argmax(dim=1)
                else:  # Regression
                    loss = criterion(outputs.squeeze(), batch_y)
                    predicted = None
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                if predicted is not None:
                    train_correct += (predicted == batch_y).sum().item()
                    train_total += batch_y.size(0)
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    
                    if problem_type == "classification":
                        if outputs.shape[1] == 1:  # Binary
                            loss = criterion(outputs.squeeze(), batch_y.float())
                            predicted = (outputs.squeeze() > 0.5).long()
                        else:  # Multi-class
                            loss = criterion(outputs, batch_y)
                            predicted = outputs.argmax(dim=1)
                    else:  # Regression
                        loss = criterion(outputs.squeeze(), batch_y)
                        predicted = None
                    
                    val_loss += loss.item()
                    
                    if predicted is not None:
                        val_correct += (predicted == batch_y).sum().item()
                        val_total += batch_y.size(0)
                        
                        # Collect predictions for metrics
                        pred_np = predicted.cpu().numpy().flatten()
                        labels_np = batch_y.cpu().numpy().flatten()
                        all_predictions.extend(pred_np)
                        all_labels.extend(labels_np)
                    else:
                        # For regression, collect continuous values
                        if problem_type == "regression":
                            pred_np = outputs.squeeze().cpu().numpy().flatten()
                            labels_np = batch_y.cpu().numpy().flatten()
                            all_predictions.extend(pred_np)
                            all_labels.extend(labels_np)
            
            # Calculate metrics
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            train_acc = train_correct / train_total if train_total > 0 else 0
            val_acc = val_correct / val_total if val_total > 0 else 0
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Load best model
        model.load_state_dict(best_model_state)
        training_time = time.time() - start_time
        
        # Calculate final metrics
        metrics = {"val_loss": best_val_loss}
        
        if problem_type == "classification" and len(all_predictions) > 0:
            from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_fscore_support
            
            accuracy = accuracy_score(all_labels, all_predictions)
            precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
            
            metrics.update({
                "val_accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "precision_weighted": precision,
                "recall_weighted": recall,
                "f1_weighted": f1,
            })
            
            # Confusion matrix
            all_classes = sorted(list(set(all_labels) | set(all_predictions)))
            cm = confusion_matrix(all_labels, all_predictions, labels=all_classes)
            metrics["confusion_matrix"] = cm.tolist()
            
            # Detailed metrics per class
            precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
                all_labels, all_predictions, average=None, zero_division=0
            )
            
            metrics["classification_report"] = {
                "precision": {str(all_classes[i]): float(precision_per_class[i]) for i in range(len(all_classes))},
                "recall": {str(all_classes[i]): float(recall_per_class[i]) for i in range(len(all_classes))},
                "f1_score": {str(all_classes[i]): float(f1_per_class[i]) for i in range(len(all_classes))},
                "support": {str(all_classes[i]): int(support_per_class[i]) for i in range(len(all_classes))},
                "class_names": [str(cls) for cls in all_classes]
            }
        
        return {
            "success": True,
            "model": model,
            "metrics": metrics,
            "training_time": training_time,
            "config": config,
        }
    
    def _get_criterion(self, problem_type: str) -> nn.Module:
        """Get appropriate loss function."""
        if problem_type == "classification":
            return nn.CrossEntropyLoss()
        else:
            return nn.MSELoss()
    
    def _get_optimizer(self, model: nn.Module, config: Dict[str, Any]) -> optim.Optimizer:
        """Get appropriate optimizer."""
        optimizer_name = config.get("optimizer", "adam").lower()
        lr = config.get("learning_rate", 0.001)
        
        if optimizer_name == "adam":
            return optim.Adam(model.parameters(), lr=lr)
        elif optimizer_name == "sgd":
            return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif optimizer_name == "rmsprop":
            return optim.RMSprop(model.parameters(), lr=lr)
        else:
            return optim.Adam(model.parameters(), lr=lr)


# Model Classes (simplified versions from original files)

class MLP(nn.Module):
    """Multi-Layer Perceptron."""
    
    def __init__(self, input_dim: int, num_classes: int, config: Dict[str, Any]):
        super(MLP, self).__init__()
        hidden_layers = config.get("hidden_layers", [128, 64, 32])
        dropout_rate = config.get("dropout_rate", 0.2)
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class CNN1D(nn.Module):
    """1D CNN for tabular data."""
    
    def __init__(self, input_dim: int, num_classes: int, config: Dict[str, Any]):
        super(CNN1D, self).__init__()
        conv_filters = config.get("conv_filters", [32, 64, 128])
        dense_layers = config.get("dense_layers", [128, 64])
        dropout_rate = config.get("dropout_rate", 0.2)
        
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        
        for filters in conv_filters:
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_channels, filters, kernel_size=3, padding=1),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.MaxPool1d(2)
            ))
            in_channels = filters
        
        # Calculate flattened size
        self.flattened_size = (input_dim // (2 ** len(conv_filters))) * conv_filters[-1]
        
        # Dense layers
        layers = []
        prev_dim = self.flattened_size
        
        for dense_dim in dense_layers:
            layers.extend([
                nn.Linear(prev_dim, dense_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = dense_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.dense_layers = nn.Sequential(*layers)
    
    def forward(self, x):
        # Reshape for 1D CNN: (batch, 1, features)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Flatten and dense layers
        x = x.view(x.size(0), -1)
        return self.dense_layers(x)


class LSTM(nn.Module):
    """LSTM for sequence data."""
    
    def __init__(self, input_dim: int, num_classes: int, config: Dict[str, Any]):
        super(LSTM, self).__init__()
        hidden_dim = config.get("hidden_dim", 64)
        num_layers = config.get("num_layers", 2)
        dropout_rate = config.get("dropout_rate", 0.2)
        bidirectional = config.get("bidirectional", True)
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout_rate, bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_output_dim, num_classes)
    
    def forward(self, x):
        # Reshape for LSTM: (batch, 1, features)
        x = x.unsqueeze(1)
        
        lstm_out, _ = self.lstm(x)
        # Take last output
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


class TextCNN(nn.Module):
    """CNN for text classification."""
    
    def __init__(self, vocab_size: int, num_classes: int, config: Dict[str, Any]):
        super(TextCNN, self).__init__()
        embed_dim = config.get("embed_dim", 128)
        num_filters = config.get("num_filters", 100)
        filter_sizes = config.get("filter_sizes", [3, 4, 5])
        dropout_rate = config.get("dropout_rate", 0.1)
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in filter_sizes
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        embedded = embedded.unsqueeze(1)  # (batch, 1, seq_len, embed_dim)
        
        # Convolution + ReLU + MaxPool
        conved = [F.relu(conv(embedded)).squeeze(3) for conv in self.convs]
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        cat = torch.cat(pooled, 1)
        
        cat = self.dropout(cat)
        return self.fc(cat)


class TextLSTM(nn.Module):
    """LSTM for text classification."""
    
    def __init__(self, vocab_size: int, num_classes: int, config: Dict[str, Any]):
        super(TextLSTM, self).__init__()
        embed_dim = config.get("embed_dim", 128)
        hidden_dim = config.get("hidden_dim", 64)
        num_layers = config.get("num_layers", 2)
        dropout_rate = config.get("dropout_rate", 0.1)
        bidirectional = config.get("bidirectional", True)
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout_rate, bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_output_dim, num_classes)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        out = lstm_out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


class BERTClassifier(nn.Module):
    """BERT-based classifier."""
    
    def __init__(self, num_classes: int, config: Dict[str, Any]):
        super(BERTClassifier, self).__init__()
        model_name = config.get("model_name", "bert-base-uncased")
        dropout_rate = config.get("dropout_rate", 0.1)
        
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)


class TabTransformer(nn.Module):
    """Transformer for tabular data."""
    
    def __init__(self, continuous_features: int, categorical_features: List[int], 
                 categorical_cardinalities: List[int], num_classes: int, config: Dict[str, Any]):
        super(TabTransformer, self).__init__()
        embedding_dim = config.get("embedding_dim", 8)
        num_heads = config.get("num_heads", 8)
        num_layers = config.get("num_layers", 6)
        hidden_dim = config.get("hidden_dim", 128)
        dropout_rate = config.get("dropout_rate", 0.1)
        
        self.continuous_features = continuous_features
        self.categorical_features = categorical_features
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        
        # Embeddings for categorical features
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embedding_dim)
            for cardinality in categorical_cardinalities
        ])
        
        # Layer normalization for embeddings
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout_rate,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Continuous feature processing
        self.continuous_norm = nn.LayerNorm(continuous_features)
        
        # Final classifier
        total_features = continuous_features + len(categorical_features) * embedding_dim
        self.classifier = nn.Sequential(
            nn.Linear(total_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        # Split continuous and categorical features
        x_continuous = x[:, :self.continuous_features]
        x_categorical = x[:, self.continuous_features:]
        
        # Process continuous features
        x_continuous = self.continuous_norm(x_continuous)
        
        # Process categorical features
        embedded_features = []
        for i, (cat_idx, embedding) in enumerate(zip(self.categorical_features, self.embeddings)):
            cat_values = x_categorical[:, i].long()
            embedded = embedding(cat_values)
            embedded_features.append(embedded)
        
        # Combine all features
        if embedded_features:
            x_categorical = torch.cat(embedded_features, dim=1)
            x_categorical = self.embedding_norm(x_categorical)
            x_combined = torch.cat([x_continuous, x_categorical], dim=1)
        else:
            x_combined = x_continuous
        
        # Add sequence dimension for transformer
        x_combined = x_combined.unsqueeze(1)
        
        # Apply transformer
        transformer_out = self.transformer(x_combined)
        
        # Take first token output
        out = transformer_out[:, 0, :]
        return self.classifier(out)


class NLPProcessor:
    """Text preprocessing and feature extraction for NLP tasks."""
    
    def __init__(self, method: str = "tfidf"):
        self.method = method
        self.vectorizer = None
        self.label_encoder = None
        self.vocab_size = None
        self.max_length = 512
        
    def fit_transform(self, texts: List[str], labels: List[str]):
        """Fit processor and transform texts."""
        if self.method == "tfidf":
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english'
            )
        else:
            self.vectorizer = CountVectorizer(max_features=10000)
        
        X = self.vectorizer.fit_transform(texts).toarray()
        
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)
        
        self.vocab_size = len(self.vectorizer.vocabulary_)
        return X, y
    
    def transform(self, texts: List[str]):
        """Transform new texts using fitted processor."""
        if self.vectorizer is None:
            raise ValueError("Processor not fitted. Call fit_transform first.")
        
        X = self.vectorizer.transform(texts).toarray()
        return X
    
    def inverse_transform_labels(self, y_encoded):
        """Convert encoded labels back to original labels."""
        if self.label_encoder is None:
            raise ValueError("Label encoder not fitted.")
        return self.label_encoder.inverse_transform(y_encoded)


def get_unified_deep_learning_automl() -> UnifiedDeepLearningAutoML:
    """Factory function for UnifiedDeepLearningAutoML."""
    return UnifiedDeepLearningAutoML()
