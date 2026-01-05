"""
Natural Language Processing (NLP) module for Free MLOps.
Supports text classification, sentiment analysis, and other NLP tasks.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
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
except ImportError:
    raise ImportError("Scikit-learn não está instalado. Instale com: pip install scikit-learn")

try:
    import transformers
    from transformers import AutoTokenizer, AutoModel
    from transformers import BertModel, BertTokenizer
except ImportError:
    print("Transformers não disponível. Usando apenas TF-IDF.")

try:
    import mlflow
    import mlflow.pytorch
except ImportError:
    print("MLflow não disponível. Tracking será desabilitado.")


class TextCNN(nn.Module):
    """
    CNN model for text classification.
    Uses convolutional layers on text embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        num_filters: int = 100,
        filter_sizes: List[int] = [3, 4, 5],
        num_classes: int = 2,
        dropout_rate: float = 0.1,
        max_length: int = 512
    ):
        super(TextCNN, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, kernel_size=filter_size)
            for filter_size in filter_sizes
        ])
        
        # Dropout and output
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        embedded = embedded.transpose(1, 2)  # (batch_size, embed_dim, seq_len)
        
        # Apply convolutions and pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = F.relu(conv(embedded))  # (batch_size, num_filters, new_seq_len)
            pooled = F.max_pool1d(conv_out, conv_out.size(2))  # (batch_size, num_filters, 1)
            conv_outputs.append(pooled.squeeze(2))  # (batch_size, num_filters)
        
        # Concatenate all conv outputs
        output = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes) * num_filters)
        
        # Apply dropout and fully connected layer
        output = self.dropout(output)
        output = self.fc(output)
        
        return output


class TextLSTM(nn.Module):
    """
    LSTM model for text classification.
    """
    
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_classes: int = 2,
        dropout_rate: float = 0.1,
        bidirectional: bool = True
    ):
        super(TextLSTM, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        # Output layer
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(lstm_output_dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state
        if self.bidirectional:
            # Concatenate forward and backward hidden states
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        # Apply dropout and fully connected layer
        output = self.dropout(hidden)
        output = self.fc(output)
        
        return output


class BERTClassifier(nn.Module):
    """
    BERT-based classifier for text classification.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        dropout_rate: float = 0.1,
        freeze_bert: bool = False
    ):
        super(BERTClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        # Load BERT model and tokenizer
        try:
            self.bert = BertModel.from_pretrained(model_name)
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
        except Exception as e:
            print(f"Erro ao carregar BERT: {e}")
            raise
        
        # Freeze BERT parameters if specified
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use the [CLS] token representation
        pooled_output = outputs.pooler_output
        
        # Apply dropout and classification
        output = self.dropout(pooled_output)
        logits = self.classifier(output)
        
        return logits
    
    def tokenize_texts(self, texts: List[str], max_length: int = 512):
        """Tokenize a list of texts."""
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )


class NLPProcessor:
    """
    Text preprocessing and feature extraction for NLP tasks.
    """
    
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
                stop_words='english' if any(word.isascii() for word in texts[:10]) else None
            )
            X = self.vectorizer.fit_transform(texts).toarray()
            self.vocab_size = len(self.vectorizer.vocabulary_)
            
        elif self.method == "count":
            self.vectorizer = CountVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english' if any(word.isascii() for word in texts[:10]) else None
            )
            X = self.vectorizer.fit_transform(texts).toarray()
            self.vocab_size = len(self.vectorizer.vocabulary_)
            
        else:
            raise ValueError(f"Method {self.method} not supported")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(labels)
        
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


class NLPDeepLearningAutoML:
    """
    AutoML for Natural Language Processing tasks.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Default configurations
        self.default_configs = {
            "pytorch": {
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
                }
            }
        }
    
    def create_text_cnn(
        self,
        texts: List[str],
        labels: List[str],
        val_texts: List[str],
        val_labels: List[str],
        config: Optional[Dict[str, Any]] = None,
        problem_type: str = "classification"
    ) -> Dict[str, Any]:
        """Create and train Text CNN model."""
        
        config = config or self.default_configs["pytorch"]["text_cnn"]
        
        # Process text data
        processor = NLPProcessor(method="tfidf")
        X_train, y_train = processor.fit_transform(texts, labels)
        X_val, y_val = processor.transform(val_texts), processor.transform(val_texts)
        
        # Convert to tensors - ensure proper shapes
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train.flatten())
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val.flatten())
        
        # Create model
        model = TextCNN(
            vocab_size=processor.vocab_size,
            embed_dim=config["embed_dim"],
            num_filters=config["num_filters"],
            filter_sizes=config["filter_sizes"],
            num_classes=len(processor.label_encoder.classes_),
            dropout_rate=config["dropout_rate"],
            max_length=config["max_length"]
        )
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        
        # Data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
        
        # Training loop
        start_time = time.time()
        best_val_loss = float('inf')
        
        for epoch in range(config["epochs"]):
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            # Coletar predições para métricas detalhadas
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                    
                    # Coletar para métricas detalhadas
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
        
        training_time = time.time() - start_time
        
        # Calcular métricas detalhadas
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        validation_metrics = {
            "val_loss": best_val_loss,
            "val_accuracy": val_accuracy,
            "precision": precision_score(all_labels, all_predictions, average='weighted', zero_division=0),
            "recall": recall_score(all_labels, all_predictions, average='weighted', zero_division=0),
            "f1_score": f1_score(all_labels, all_predictions, average='weighted', zero_division=0),
            "confusion_matrix": confusion_matrix(all_labels, all_predictions).tolist()
        }
        
        return {
            "model": model,
            "framework": "pytorch",
            "model_type": "text_cnn",
            "training_time": training_time,
            "validation_metrics": validation_metrics,
            "config": config,
            "processor": processor,
            "num_classes": len(processor.label_encoder.classes_),
            "class_names": processor.label_encoder.classes_.tolist()
        }
    
    def create_text_lstm(
        self,
        texts: List[str],
        labels: List[str],
        val_texts: List[str],
        val_labels: List[str],
        config: Optional[Dict[str, Any]] = None,
        problem_type: str = "classification"
    ) -> Dict[str, Any]:
        """Create and train Text LSTM model."""
        
        config = config or self.default_configs["pytorch"]["text_lstm"]
        
        # Process text data
        processor = NLPProcessor(method="tfidf")
        X_train, y_train = processor.fit_transform(texts, labels)
        X_val, y_val = processor.transform(val_texts), processor.transform(val_texts)
        
        # Convert to tensors - ensure proper shapes
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train.flatten())
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val.flatten())
        
        # Create model
        model = TextLSTM(
            vocab_size=processor.vocab_size,
            embed_dim=config["embed_dim"],
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            dropout_rate=config["dropout_rate"],
            bidirectional=config["bidirectional"],
            num_classes=len(processor.label_encoder.classes_)
        )
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        
        # Data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
        
        # Training loop
        start_time = time.time()
        best_val_loss = float('inf')
        
        for epoch in range(config["epochs"]):
            model.train()
            train_loss = 0.0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            # Coletar predições para métricas detalhadas
            all_predictions = []
            all_labels = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
                    
                    # Coletar para métricas detalhadas
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
        
        training_time = time.time() - start_time
        
        # Calcular métricas detalhadas
        from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        
        validation_metrics = {
            "val_loss": best_val_loss,
            "val_accuracy": val_accuracy,
            "precision": precision_score(all_labels, all_predictions, average='weighted', zero_division=0),
            "recall": recall_score(all_labels, all_predictions, average='weighted', zero_division=0),
            "f1_score": f1_score(all_labels, all_predictions, average='weighted', zero_division=0),
            "confusion_matrix": confusion_matrix(all_labels, all_predictions).tolist()
        }
        
        return {
            "model": model,
            "framework": "pytorch",
            "model_type": "text_lstm",
            "training_time": training_time,
            "validation_metrics": validation_metrics,
            "config": config,
            "processor": processor,
            "num_classes": len(processor.label_encoder.classes_),
            "class_names": processor.label_encoder.classes_.tolist()
        }
    
    def create_bert_classifier(
        self,
        texts: List[str],
        labels: List[str],
        val_texts: List[str],
        val_labels: List[str],
        config: Optional[Dict[str, Any]] = None,
        problem_type: str = "classification"
    ) -> Dict[str, Any]:
        """Create and train BERT classifier."""
        
        config = config or self.default_configs["pytorch"]["bert_classifier"]
        
        # Process labels
        processor = NLPProcessor(method="tfidf")  # Just for label encoding
        _, y_train = processor.fit_transform(texts, labels)
        _, y_val = processor.transform(val_texts), processor.transform(val_texts)
        
        # Create BERT model
        model = BERTClassifier(
            model_name=config["model_name"],
            num_classes=len(processor.label_encoder.classes_),
            dropout_rate=config["dropout_rate"],
            freeze_bert=config["freeze_bert"]
        )
        
        # Tokenize texts
        train_encodings = model.tokenize_texts(texts, max_length=config["max_length"])
        val_encodings = model.tokenize_texts(val_texts, max_length=config["max_length"])
        
        # Create datasets
        train_dataset = TensorDataset(
            train_encodings['input_ids'],
            train_encodings['attention_mask'],
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        
        val_dataset = TensorDataset(
            val_encodings['input_ids'],
            val_encodings['attention_mask'],
            torch.LongTensor(y_val)
        )
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
        
        # Training setup
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        
        # Training loop
        start_time = time.time()
        best_val_loss = float('inf')
        
        for epoch in range(config["epochs"]):
            model.train()
            train_loss = 0.0
            
            for batch_input_ids, batch_attention_mask, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_input_ids, batch_attention_mask)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_input_ids, batch_attention_mask, batch_y in val_loader:
                    outputs = model(batch_input_ids, batch_attention_mask)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += batch_y.size(0)
                    correct += (predicted == batch_y).sum().item()
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
        
        training_time = time.time() - start_time
        
        return {
            "model": model,
            "framework": "pytorch",
            "model_type": "bert_classifier",
            "training_time": training_time,
            "validation_metrics": {
                "val_loss": best_val_loss,
                "val_accuracy": val_accuracy,
            },
            "config": config,
            "processor": processor,
            "num_classes": len(processor.label_encoder.classes_),
            "class_names": processor.label_encoder.classes_.tolist(),
            "tokenizer": model.tokenizer
        }


def get_nlp_deep_learning_automl() -> NLPDeepLearningAutoML:
    """Factory function for NLPDeepLearningAutoML."""
    return NLPDeepLearningAutoML()
