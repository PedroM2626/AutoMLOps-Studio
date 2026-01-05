from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from free_mlops.config import get_settings


class DeepLearningAutoML:
    """AutoML para Deep Learning com suporte a TensorFlow e PyTorch."""
    
    def __init__(self, settings: Optional[Any] = None):
        self.settings = settings or get_settings()
        self.models_dir = self.settings.artifacts_dir / "deep_learning"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurações padrão
        self.default_configs = {
            "tensorflow": {
                "mlp": {
                    "hidden_layers": [128, 64, 32],
                    "activation": "relu",
                    "dropout_rate": 0.2,
                    "batch_size": 32,
                    "epochs": 100,
                    "learning_rate": 0.001,
                    "optimizer": "adam",
                },
                "cnn": {
                    "conv_layers": [{"filters": 32, "kernel_size": 3}, {"filters": 64, "kernel_size": 3}],
                    "dense_layers": [128, 64],
                    "activation": "relu",
                    "dropout_rate": 0.3,
                    "batch_size": 32,
                    "epochs": 100,
                    "learning_rate": 0.001,
                    "optimizer": "adam",
                },
                "lstm": {
                    "lstm_layers": [64, 32],
                    "dense_layers": [32],
                    "activation": "tanh",
                    "dropout_rate": 0.2,
                    "batch_size": 32,
                    "epochs": 100,
                    "learning_rate": 0.001,
                    "optimizer": "adam",
                }
            },
            "pytorch": {
                "mlp": {
                    "hidden_layers": [128, 64, 32],
                    "activation": "relu",
                    "dropout_rate": 0.2,
                    "batch_size": 32,
                    "epochs": 100,
                    "learning_rate": 0.001,
                    "optimizer": "adam",
                },
                "cnn": {
                    "conv_layers": [{"filters": 32, "kernel_size": 3}, {"filters": 64, "kernel_size": 3}],
                    "dense_layers": [128, 64],
                    "activation": "relu",
                    "dropout_rate": 0.3,
                    "batch_size": 32,
                    "epochs": 100,
                    "learning_rate": 0.001,
                    "optimizer": "adam",
                },
                "lstm": {
                    "lstm_layers": [64, 32],
                    "dense_layers": [32],
                    "activation": "tanh",
                    "dropout_rate": 0.2,
                    "batch_size": 32,
                    "epochs": 100,
                    "learning_rate": 0.001,
                    "optimizer": "adam",
                }
            }
        }
    
    def create_tensorflow_mlp(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        input_shape: Tuple[int, ...],
        num_classes: int,
        config: Optional[Dict[str, Any]] = None,
        problem_type: str = "classification",
    ) -> Dict[str, Any]:
        """Cria e treina MLP com TensorFlow."""
        
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models, optimizers, callbacks
        except ImportError:
            raise ImportError("TensorFlow não está instalado. Instale com: pip install tensorflow")
        
        config = config or self.default_configs["tensorflow"]["mlp"]
        
        # Garantir que config tem as chaves necessárias
        if "hidden_layers" not in config:
            config = self.default_configs["tensorflow"]["mlp"]
        
        # Criar modelo
        model = models.Sequential()
        
        # Input layer
        model.add(layers.Input(shape=input_shape))
        
        # Hidden layers
        for units in config["hidden_layers"]:
            model.add(layers.Dense(units, activation=config["activation"]))
            if config["dropout_rate"] > 0:
                model.add(layers.Dropout(config["dropout_rate"]))
        
        # Output layer
        if problem_type == "classification":
            if num_classes == 2:
                model.add(layers.Dense(1, activation="sigmoid"))
                loss_fn = "binary_crossentropy"
                metrics = ["accuracy"]
            else:
                model.add(layers.Dense(num_classes, activation="softmax"))
                loss_fn = "sparse_categorical_crossentropy"
                metrics = ["accuracy"]
        else:
            model.add(layers.Dense(1, activation="linear"))
            loss_fn = "mse"
            metrics = ["mae"]
        
        # Compilar
        optimizer_config = config["optimizer"]
        if optimizer_config == "adam":
            optimizer = optimizers.Adam(learning_rate=config["learning_rate"])
        elif optimizer_config == "sgd":
            optimizer = optimizers.SGD(learning_rate=config["learning_rate"])
        else:
            optimizer = optimizers.Adam(learning_rate=config["learning_rate"])
        
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics
        )
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        ]
        
        # Treinar
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            callbacks=callback_list,
            verbose=0,
        )
        training_time = time.time() - start_time
        
        # Avaliar
        val_results = model.evaluate(X_val, y_val, verbose=0)
        
        # Preparar resultados
        result = {
            "framework": "tensorflow",
            "model_type": "mlp",
            "problem_type": problem_type,
            "model": model,
            "config": config,
            "training_time": training_time,
            "history": {
                "loss": history.history["loss"],
                "val_loss": history.history["val_loss"],
                "accuracy": history.history.get("accuracy", []),
                "val_accuracy": history.history.get("val_accuracy", []),
                "mae": history.history.get("mae", []),
                "val_mae": history.history.get("val_mae", []),
            },
            "validation_metrics": dict(zip(model.metrics_names, val_results)),
            "input_shape": input_shape,
            "num_classes": num_classes,
        }
        
        return result
    
    def create_pytorch_mlp(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        input_shape: Tuple[int, ...],
        num_classes: int,
        config: Optional[Dict[str, Any]] = None,
        problem_type: str = "classification",
    ) -> Dict[str, Any]:
        """Cria e treina MLP com PyTorch."""
        
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch não está instalado. Instale com: pip install torch")
        
        config = config or self.default_configs["pytorch"]["mlp"]
        
        # Garantir que config tem as chaves necessárias
        if "hidden_layers" not in config:
            config = self.default_configs["pytorch"]["mlp"]
        
        # Converter para tensores com tipos corretos
        X_train_clean = X_train.astype(np.float32)
        y_train_clean = y_train.astype(np.int32) if problem_type == "classification" else y_train.astype(np.float32)
        X_val_clean = X_val.astype(np.float32)
        y_val_clean = y_val.astype(np.int32) if problem_type == "classification" else y_val.astype(np.float32)
        
        X_train_tensor = torch.FloatTensor(X_train_clean)
        y_train_tensor = torch.LongTensor(y_train_clean) if problem_type == "classification" else torch.FloatTensor(y_train_clean)
        X_val_tensor = torch.FloatTensor(X_val_clean)
        y_val_tensor = torch.LongTensor(y_val_clean) if problem_type == "classification" else torch.FloatTensor(y_val_clean)
        
        # Data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
        
        # Definir modelo
        class MLP(nn.Module):
            def __init__(self, input_dim, hidden_layers, dropout_rate, num_classes, problem_type):
                super(MLP, self).__init__()
                
                layers = []
                prev_dim = input_dim
                
                for units in hidden_layers:
                    layers.append(nn.Linear(prev_dim, units))
                    layers.append(nn.ReLU())
                    if dropout_rate > 0:
                        layers.append(nn.Dropout(dropout_rate))
                    prev_dim = units
                
                # Output layer
                if problem_type == "classification":
                    if num_classes == 2:
                        layers.append(nn.Linear(prev_dim, 1))
                        layers.append(nn.Sigmoid())
                    else:
                        layers.append(nn.Linear(prev_dim, num_classes))
                        layers.append(nn.Softmax(dim=1))
                else:
                    layers.append(nn.Linear(prev_dim, 1))
                
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        model = MLP(input_shape[0], config["hidden_layers"], config["dropout_rate"], num_classes, problem_type)
        
        # Loss e optimizer
        if problem_type == "classification":
            if num_classes == 2:
                criterion = nn.BCELoss()
            else:
                criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        if config["optimizer"] == "adam":
            optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        elif config["optimizer"] == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
        else:
            optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        
        # Treinar
        start_time = time.time()
        history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config["epochs"]):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                if problem_type == "classification" and num_classes == 2:
                    loss = criterion(outputs.squeeze(), batch_y.float())
                    predicted = (outputs.squeeze() > 0.5).long()
                elif problem_type == "classification":
                    loss = criterion(outputs, batch_y)
                    predicted = outputs.argmax(dim=1)
                else:
                    loss = criterion(outputs.squeeze(), batch_y)
                    predicted = None
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if predicted is not None:
                    train_correct += (predicted == batch_y).sum().item()
                    train_total += batch_y.size(0)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    
                    if problem_type == "classification" and num_classes == 2:
                        loss = criterion(outputs.squeeze(), batch_y.float())
                        predicted = (outputs.squeeze() > 0.5).long()
                    elif problem_type == "classification":
                        loss = criterion(outputs, batch_y)
                        predicted = outputs.argmax(dim=1)
                    else:
                        loss = criterion(outputs.squeeze(), batch_y)
                        predicted = None
                    
                    val_loss += loss.item()
                    
                    if predicted is not None:
                        val_correct += (predicted == batch_y).sum().item()
                        val_total += batch_y.size(0)
            
            # Record history
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            history["loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            
            if problem_type == "classification":
                train_acc = train_correct / train_total if train_total > 0 else 0
                val_acc = val_correct / val_total if val_total > 0 else 0
                history["accuracy"].append(train_acc)
                history["val_accuracy"].append(val_acc)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break
        
        # Load best model
        model.load_state_dict(best_model_state)
        training_time = time.time() - start_time
        
        # Final evaluation
        model.eval()
        final_val_loss = 0.0
        final_val_correct = 0
        final_val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                
                if problem_type == "classification" and num_classes == 2:
                    loss = criterion(outputs.squeeze(), batch_y.float())
                    predicted = (outputs.squeeze() > 0.5).long()
                elif problem_type == "classification":
                    loss = criterion(outputs, batch_y)
                    predicted = outputs.argmax(dim=1)
                else:
                    loss = criterion(outputs.squeeze(), batch_y)
                    predicted = None
                
                final_val_loss += loss.item()
                
                if predicted is not None:
                    final_val_correct += (predicted == batch_y).sum().item()
                    final_val_total += batch_y.size(0)
        
        avg_final_val_loss = final_val_loss / len(val_loader)
        
        # Preparar resultados
        result = {
            "framework": "pytorch",
            "model_type": "mlp",
            "problem_type": problem_type,
            "model": model,
            "config": config,
            "training_time": training_time,
            "history": history,
            "validation_metrics": {
                "val_loss": avg_final_val_loss,
                "val_accuracy": final_val_correct / final_val_total if problem_type == "classification" else None,
            },
            "input_shape": input_shape,
            "num_classes": num_classes,
        }
        
        return result
    
    def create_tensorflow_cnn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        input_shape: Tuple[int, ...],
        num_classes: int,
        config: Optional[Dict[str, Any]] = None,
        problem_type: str = "classification",
    ) -> Dict[str, Any]:
        """Cria e treina CNN com TensorFlow."""
        
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models, optimizers, callbacks
        except ImportError:
            raise ImportError("TensorFlow não está instalado. Instale com: pip install tensorflow")
        
        config = config or self.default_configs["tensorflow"]["cnn"]
        
        # Garantir que config tem as chaves necessárias
        if "conv_layers" not in config:
            config = self.default_configs["tensorflow"]["cnn"]
        
        # Verificar se input_shape é adequado para CNN
        if len(input_shape) < 2:
            raise ValueError("CNN requer input_shape com pelo menos 2 dimensões (height, width)")
        
        # Criar modelo
        model = models.Sequential()
        
        # Conv layers
        for i, conv_layer in enumerate(config["conv_layers"]):
            if i == 0:
                model.add(layers.Conv2D(
                    conv_layer["filters"],
                    conv_layer["kernel_size"],
                    activation=config["activation"],
                    input_shape=input_shape
                ))
            else:
                model.add(layers.Conv2D(
                    conv_layer["filters"],
                    conv_layer["kernel_size"],
                    activation=config["activation"]
                ))
            model.add(layers.MaxPooling2D((2, 2)))
        
        model.add(layers.Flatten())
        
        # Dense layers
        for units in config["dense_layers"]:
            model.add(layers.Dense(units, activation=config["activation"]))
            if config["dropout_rate"] > 0:
                model.add(layers.Dropout(config["dropout_rate"]))
        
        # Output layer
        if problem_type == "classification":
            if num_classes == 2:
                model.add(layers.Dense(1, activation="sigmoid"))
                loss_fn = "binary_crossentropy"
                metrics = ["accuracy"]
            else:
                model.add(layers.Dense(num_classes, activation="softmax"))
                loss_fn = "sparse_categorical_crossentropy"
                metrics = ["accuracy"]
        else:
            model.add(layers.Dense(1, activation="linear"))
            loss_fn = "mse"
            metrics = ["mae"]
        
        # Compilar
        optimizer_config = config["optimizer"]
        if optimizer_config == "adam":
            optimizer = optimizers.Adam(learning_rate=config["learning_rate"])
        elif optimizer_config == "sgd":
            optimizer = optimizers.SGD(learning_rate=config["learning_rate"])
        else:
            optimizer = optimizers.Adam(learning_rate=config["learning_rate"])
        
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics
        )
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        ]
        
        # Treinar
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            callbacks=callback_list,
            verbose=0,
        )
        training_time = time.time() - start_time
        
        # Avaliar
        val_results = model.evaluate(X_val, y_val, verbose=0)
        
        # Preparar resultados
        result = {
            "framework": "tensorflow",
            "model_type": "cnn",
            "problem_type": problem_type,
            "model": model,
            "config": config,
            "training_time": training_time,
            "history": {
                "loss": history.history["loss"],
                "val_loss": history.history["val_loss"],
                "accuracy": history.history.get("accuracy", []),
                "val_accuracy": history.history.get("val_accuracy", []),
                "mae": history.history.get("mae", []),
                "val_mae": history.history.get("val_mae", []),
            },
            "validation_metrics": dict(zip(model.metrics_names, val_results)),
            "input_shape": input_shape,
            "num_classes": num_classes,
        }
        
        return result
    
    def create_tensorflow_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        input_shape: Tuple[int, ...],
        num_classes: int,
        config: Optional[Dict[str, Any]] = None,
        problem_type: str = "classification",
    ) -> Dict[str, Any]:
        """Cria e treina LSTM com TensorFlow."""
        
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models, optimizers, callbacks
        except ImportError:
            raise ImportError("TensorFlow não está instalado. Instale com: pip install tensorflow")
        
        config = config or self.default_configs["tensorflow"]["lstm"]
        
        # Garantir que config tem as chaves necessárias
        if "lstm_layers" not in config:
            config = self.default_configs["tensorflow"]["lstm"]
        
        # Verificar se input_shape é adequado para LSTM
        if len(input_shape) != 2:
            raise ValueError("LSTM requer input_shape com 2 dimensões (timesteps, features)")
        
        # Criar modelo
        model = models.Sequential()
        
        # LSTM layers
        for i, units in enumerate(config["lstm_layers"]):
            return_sequences = i < len(config["lstm_layers"]) - 1
            model.add(layers.LSTM(
                units,
                activation=config["activation"],
                return_sequences=return_sequences,
                input_shape=input_shape if i == 0 else None
            ))
        
        # Dense layers
        for units in config["dense_layers"]:
            model.add(layers.Dense(units, activation=config["activation"]))
            if config["dropout_rate"] > 0:
                model.add(layers.Dropout(config["dropout_rate"]))
        
        # Output layer
        if problem_type == "classification":
            if num_classes == 2:
                model.add(layers.Dense(1, activation="sigmoid"))
                loss_fn = "binary_crossentropy"
                metrics = ["accuracy"]
            else:
                model.add(layers.Dense(num_classes, activation="softmax"))
                loss_fn = "sparse_categorical_crossentropy"
                metrics = ["accuracy"]
        else:
            model.add(layers.Dense(1, activation="linear"))
            loss_fn = "mse"
            metrics = ["mae"]
        
        # Compilar
        optimizer_config = config["optimizer"]
        if optimizer_config == "adam":
            optimizer = optimizers.Adam(learning_rate=config["learning_rate"])
        elif optimizer_config == "sgd":
            optimizer = optimizers.SGD(learning_rate=config["learning_rate"])
        else:
            optimizer = optimizers.Adam(learning_rate=config["learning_rate"])
        
        model.compile(
            optimizer=optimizer,
            loss=loss_fn,
            metrics=metrics
        )
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        ]
        
        # Treinar
        start_time = time.time()
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            callbacks=callback_list,
            verbose=0,
        )
        training_time = time.time() - start_time
        
        # Avaliar
        val_results = model.evaluate(X_val, y_val, verbose=0)
        
        # Preparar resultados
        result = {
            "framework": "tensorflow",
            "model_type": "lstm",
            "problem_type": problem_type,
            "model": model,
            "config": config,
            "training_time": training_time,
            "history": {
                "loss": history.history["loss"],
                "val_loss": history.history["val_loss"],
                "accuracy": history.history.get("accuracy", []),
                "val_accuracy": history.history.get("val_accuracy", []),
                "mae": history.history.get("mae", []),
                "val_mae": history.history.get("val_mae", []),
            },
            "validation_metrics": dict(zip(model.metrics_names, val_results)),
            "input_shape": input_shape,
            "num_classes": num_classes,
        }
        
        return result
    
    def create_pytorch_cnn(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        input_shape: Tuple[int, ...],
        num_classes: int,
        config: Optional[Dict[str, Any]] = None,
        problem_type: str = "classification",
    ) -> Dict[str, Any]:
        """Cria e treina CNN com PyTorch."""
        
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch não está instalado. Instale com: pip install torch")
        
        config = config or self.default_configs["pytorch"]["cnn"]
        
        # Garantir que config tem as chaves necessárias
        if "conv_layers" not in config:
            config = self.default_configs["pytorch"]["cnn"]
        
        # Para CNN, precisamos reshape os dados para (batch, channels, height, width)
        # Para dados tabulares, vamos tratar como (batch, channels, 1, features)
        X_train_reshaped = X_train.reshape(X_train.shape[0], 1, 1, X_train.shape[1])
        X_val_reshaped = X_val.reshape(X_val.shape[0], 1, 1, X_val.shape[1])
        
        # Converter para tensores com tipos corretos
        X_train_clean = X_train_reshaped.astype(np.float32)
        y_train_clean = y_train.astype(np.int32) if problem_type == "classification" else y_train.astype(np.float32)
        X_val_clean = X_val_reshaped.astype(np.float32)
        y_val_clean = y_val.astype(np.int32) if problem_type == "classification" else y_val.astype(np.float32)
        
        X_train_tensor = torch.FloatTensor(X_train_clean)
        y_train_tensor = torch.LongTensor(y_train_clean) if problem_type == "classification" else torch.FloatTensor(y_train_clean)
        X_val_tensor = torch.FloatTensor(X_val_clean)
        y_val_tensor = torch.LongTensor(y_val_clean) if problem_type == "classification" else torch.FloatTensor(y_val_clean)
        
        # Data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
        
        # Definir modelo CNN
        class CNN(nn.Module):
            def __init__(self, input_channels, conv_layers, dense_layers, dropout_rate, num_classes, problem_type):
                super(CNN, self).__init__()
                
                layers = []
                prev_channels = input_channels
                
                # Camadas convolucionais
                for conv_layer in conv_layers:
                    layers.append(nn.Conv2d(prev_channels, conv_layer["filters"], conv_layer["kernel_size"], padding=1))
                    layers.append(nn.ReLU())
                    layers.append(nn.MaxPool2d(2, 2))
                    prev_channels = conv_layer["filters"]
                
                layers.append(nn.Flatten())
                
                # Camadas densas
                for units in dense_layers:
                    layers.append(nn.Linear(prev_channels, units))
                    layers.append(nn.ReLU())
                    if dropout_rate > 0:
                        layers.append(nn.Dropout(dropout_rate))
                    prev_channels = units
                
                # Camada de saída
                if problem_type == "classification":
                    if num_classes == 2:
                        layers.append(nn.Linear(prev_channels, 1))
                        layers.append(nn.Sigmoid())
                    else:
                        layers.append(nn.Linear(prev_channels, num_classes))
                        layers.append(nn.Softmax(dim=1))
                else:
                    layers.append(nn.Linear(prev_channels, 1))
                
                self.network = nn.Sequential(*layers)
            
            def forward(self, x):
                return self.network(x)
        
        model = CNN(1, config["conv_layers"], config["dense_layers"], config["dropout_rate"], num_classes, problem_type)
        
        # Loss e optimizer
        if problem_type == "classification":
            if num_classes == 2:
                criterion = nn.BCELoss()
            else:
                criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        if config["optimizer"] == "adam":
            optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        elif config["optimizer"] == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
        else:
            optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        
        # Treinar
        start_time = time.time()
        history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config["epochs"]):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                if problem_type == "classification" and num_classes == 2:
                    loss = criterion(outputs.squeeze(), batch_y.float())
                    predicted = (outputs.squeeze() > 0.5).long()
                elif problem_type == "classification":
                    loss = criterion(outputs, batch_y)
                    predicted = outputs.argmax(dim=1)
                else:
                    loss = criterion(outputs.squeeze(), batch_y)
                    predicted = None
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if predicted is not None:
                    train_correct += (predicted == batch_y).sum().item()
                    train_total += batch_y.size(0)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    
                    if problem_type == "classification" and num_classes == 2:
                        loss = criterion(outputs.squeeze(), batch_y.float())
                        predicted = (outputs.squeeze() > 0.5).long()
                    elif problem_type == "classification":
                        loss = criterion(outputs, batch_y)
                        predicted = outputs.argmax(dim=1)
                    else:
                        loss = criterion(outputs.squeeze(), batch_y)
                        predicted = None
                    
                    val_loss += loss.item()
                    
                    if predicted is not None:
                        val_correct += (predicted == batch_y).sum().item()
                        val_total += batch_y.size(0)
            
            # Record history
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            history["loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            
            if problem_type == "classification":
                train_acc = train_correct / train_total if train_total > 0 else 0
                val_acc = val_correct / val_total if val_total > 0 else 0
                history["accuracy"].append(train_acc)
                history["val_accuracy"].append(val_acc)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break
        
        # Load best model
        model.load_state_dict(best_model_state)
        training_time = time.time() - start_time
        
        # Final evaluation
        model.eval()
        final_val_loss = 0.0
        final_val_correct = 0
        final_val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                
                if problem_type == "classification" and num_classes == 2:
                    loss = criterion(outputs.squeeze(), batch_y.float())
                    predicted = (outputs.squeeze() > 0.5).long()
                elif problem_type == "classification":
                    loss = criterion(outputs, batch_y)
                    predicted = outputs.argmax(dim=1)
                else:
                    loss = criterion(outputs.squeeze(), batch_y)
                    predicted = None
                
                final_val_loss += loss.item()
                
                if predicted is not None:
                    final_val_correct += (predicted == batch_y).sum().item()
                    final_val_total += batch_y.size(0)
        
        avg_final_val_loss = final_val_loss / len(val_loader)
        
        # Preparar resultados
        result = {
            "framework": "pytorch",
            "model_type": "cnn",
            "problem_type": problem_type,
            "model": model,
            "config": config,
            "training_time": training_time,
            "history": history,
            "validation_metrics": {
                "val_loss": avg_final_val_loss,
                "val_accuracy": final_val_correct / final_val_total if problem_type == "classification" else None,
            },
            "input_shape": input_shape,
            "num_classes": num_classes,
        }
        
        return result
    
    def create_pytorch_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        input_shape: Tuple[int, ...],
        num_classes: int,
        config: Optional[Dict[str, Any]] = None,
        problem_type: str = "classification",
    ) -> Dict[str, Any]:
        """Cria e treina LSTM com PyTorch."""
        
        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError("PyTorch não está instalado. Instale com: pip install torch")
        
        config = config or self.default_configs["pytorch"]["lstm"]
        
        # Garantir que config tem as chaves necessárias
        if "lstm_layers" not in config:
            config = self.default_configs["pytorch"]["lstm"]
        
        # Para LSTM, precisamos reshape para (batch, sequence_length, features)
        sequence_length = 1  # Simplificado para dados tabulares
        X_train_reshaped = X_train.reshape(X_train.shape[0], sequence_length, X_train.shape[1])
        X_val_reshaped = X_val.reshape(X_val.shape[0], sequence_length, X_val.shape[1])
        
        # Converter para tensores com tipos corretos
        X_train_clean = X_train_reshaped.astype(np.float32)
        y_train_clean = y_train.astype(np.int32) if problem_type == "classification" else y_train.astype(np.float32)
        X_val_clean = X_val_reshaped.astype(np.float32)
        y_val_clean = y_val.astype(np.int32) if problem_type == "classification" else y_val.astype(np.float32)
        
        X_train_tensor = torch.FloatTensor(X_train_clean)
        y_train_tensor = torch.LongTensor(y_train_clean) if problem_type == "classification" else torch.FloatTensor(y_train_clean)
        X_val_tensor = torch.FloatTensor(X_val_clean)
        y_val_tensor = torch.LongTensor(y_val_clean) if problem_type == "classification" else torch.FloatTensor(y_val_clean)
        
        # Data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
        
        # Definir modelo LSTM
        class LSTM(nn.Module):
            def __init__(self, input_size, lstm_layers, dense_layers, dropout_rate, num_classes, problem_type):
                super(LSTM, self).__init__()
                
                # Camada LSTM
                self.lstm = nn.LSTM(input_size, lstm_layers[0], lstm_layers[1], batch_first=True, dropout=dropout_rate)
                
                # Camadas densas
                layers = []
                prev_size = lstm_layers[1]
                for units in dense_layers:
                    layers.append(nn.Linear(prev_size, units))
                    layers.append(nn.ReLU())
                    if dropout_rate > 0:
                        layers.append(nn.Dropout(dropout_rate))
                    prev_size = units
                
                # Camada de saída
                if problem_type == "classification":
                    if num_classes == 2:
                        layers.append(nn.Linear(prev_size, 1))
                        layers.append(nn.Sigmoid())
                    else:
                        layers.append(nn.Linear(prev_size, num_classes))
                        layers.append(nn.Softmax(dim=1))
                else:
                    layers.append(nn.Linear(prev_size, 1))
                
                self.dense_layers = nn.Sequential(*layers)
            
            def forward(self, x):
                # Passar através da LSTM
                lstm_out, _ = self.lstm(x)
                
                # Pegar apenas a última saída da LSTM para cada sequência
                if lstm_out.dim() == 3:  # (batch, seq, features)
                    last_output = lstm_out[:, -1, :]  # (batch, features)
                else:
                    last_output = lstm_out[:, -1]  # (batch, features)
                
                # Passar através das camadas densas
                return self.dense_layers(last_output)
        
        model = LSTM(X_train.shape[1], config["lstm_layers"], config["dense_layers"], config["dropout_rate"], num_classes, problem_type)
        
        # Loss e optimizer
        if problem_type == "classification":
            if num_classes == 2:
                criterion = nn.BCELoss()
            else:
                criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
        
        if config["optimizer"] == "adam":
            optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        elif config["optimizer"] == "sgd":
            optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
        else:
            optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        
        # Treinar
        start_time = time.time()
        history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config["epochs"]):
            # Training
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                if problem_type == "classification" and num_classes == 2:
                    loss = criterion(outputs.squeeze(), batch_y.float())
                    predicted = (outputs.squeeze() > 0.5).long()
                elif problem_type == "classification":
                    loss = criterion(outputs, batch_y)
                    predicted = outputs.argmax(dim=1)
                else:
                    loss = criterion(outputs.squeeze(), batch_y)
                    predicted = None
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                
                if predicted is not None:
                    train_correct += (predicted == batch_y).sum().item()
                    train_total += batch_y.size(0)
            
            # Validation
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    
                    if problem_type == "classification" and num_classes == 2:
                        loss = criterion(outputs.squeeze(), batch_y.float())
                        predicted = (outputs.squeeze() > 0.5).long()
                    elif problem_type == "classification":
                        loss = criterion(outputs, batch_y)
                        predicted = outputs.argmax(dim=1)
                    else:
                        loss = criterion(outputs.squeeze(), batch_y)
                        predicted = None
                    
                    val_loss += loss.item()
                    
                    if predicted is not None:
                        val_correct += (predicted == batch_y).sum().item()
                        val_total += batch_y.size(0)
            
            # Record history
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            history["loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)
            
            if problem_type == "classification":
                train_acc = train_correct / train_total if train_total > 0 else 0
                val_acc = val_correct / val_total if val_total > 0 else 0
                history["accuracy"].append(train_acc)
                history["val_accuracy"].append(val_acc)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    break
        
        # Load best model
        model.load_state_dict(best_model_state)
        training_time = time.time() - start_time
        
        # Final evaluation
        model.eval()
        final_val_loss = 0.0
        final_val_correct = 0
        final_val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                
                if problem_type == "classification" and num_classes == 2:
                    loss = criterion(outputs.squeeze(), batch_y.float())
                    predicted = (outputs.squeeze() > 0.5).long()
                elif problem_type == "classification":
                    loss = criterion(outputs, batch_y)
                    predicted = outputs.argmax(dim=1)
                else:
                    loss = criterion(outputs.squeeze(), batch_y)
                    predicted = None
                
                final_val_loss += loss.item()
                
                if predicted is not None:
                    final_val_correct += (predicted == batch_y).sum().item()
                    final_val_total += batch_y.size(0)
        
        avg_final_val_loss = final_val_loss / len(val_loader)
        
        # Preparar resultados
        result = {
            "framework": "pytorch",
            "model_type": "lstm",
            "problem_type": problem_type,
            "model": model,
            "config": config,
            "training_time": training_time,
            "history": history,
            "validation_metrics": {
                "val_loss": avg_final_val_loss,
                "val_accuracy": final_val_correct / final_val_total if problem_type == "classification" else None,
            },
            "input_shape": input_shape,
            "num_classes": num_classes,
        }
        
        return result
    
    def _clean_numeric_data(self, data: np.ndarray) -> np.ndarray:
        """Limpa dados numéricos removendo colunas não-numéricas e convertendo para float32."""
        import pandas as pd
        
        # Converter para DataFrame se for array
        if isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Identificar colunas numéricas
        numeric_cols = []
        for col in df.columns:
            try:
                # Tentar converter para numérico
                pd.to_numeric(df[col], errors='raise')
                numeric_cols.append(col)
            except (ValueError, TypeError):
                # Pular colunas não-numéricas
                continue
        
        if not numeric_cols:
            raise ValueError("Nenhuma coluna numérica encontrada nos dados. Para Deep Learning, todos os features devem ser numéricos.")
        
        # Manter apenas colunas numéricas
        df_numeric = df[numeric_cols]
        
        # Converter para float32
        return df_numeric.astype(np.float32).values
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """Carrega modelo salvo."""
        
        model_dir = Path(model_path)
        
        # Carregar metadados
        with open(model_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        framework = metadata["framework"]
        model_type = metadata["model_type"]
        
        if framework == "tensorflow":
            try:
                import tensorflow as tf
                model = tf.keras.models.load_model(model_dir / "model")
            except Exception as e:
                raise RuntimeError(f"Erro ao carregar modelo TensorFlow: {e}")
        
        elif framework == "pytorch":
            try:
                import torch
                import torch.nn as nn
                
                # Reconstruir modelo (simplificado - na prática precisaríamos salvar a arquitetura)
                # Aqui estamos apenas carregando os pesos
                model = None  # Precisaria da classe original
                # model.load_state_dict(torch.load(model_dir / "model.pth"))
                
                raise NotImplementedError("Carregamento de modelos PyTorch requer implementação específica")
            except Exception as e:
                raise RuntimeError(f"Erro ao carregar modelo PyTorch: {e}")
        
        return {
            "model": model,
            "metadata": metadata,
        }
    
    def list_models(self) -> List[Dict[str, Any]]:
        """Lista todos os modelos salvos."""
        
        models = []
        
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir():
                metadata_file = model_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                        
                        models.append({
                            "name": metadata.get("model_name", model_dir.name),
                            "framework": metadata.get("framework", "unknown"),
                            "model_type": metadata.get("model_type", "unknown"),
                            "problem_type": metadata.get("problem_type", "unknown"),
                            "saved_at": metadata.get("saved_at", ""),
                            "training_time": metadata.get("training_time", 0),
                            "validation_metrics": metadata.get("validation_metrics", {}),
                            "path": str(model_dir),
                        })
                    except Exception:
                        continue
        
        return sorted(models, key=lambda x: x["saved_at"], reverse=True)
    
    def predict(
        self,
        model_path: str,
        X: np.ndarray,
        batch_size: int = 32,
    ) -> Dict[str, Any]:
        """Realiza predições com modelo salvo."""
        
        loaded = self.load_model(model_path)
        model = loaded["model"]
        metadata = loaded["metadata"]
        
        framework = metadata["framework"]
        problem_type = metadata["problem_type"]
        
        try:
            if framework == "tensorflow":
                import tensorflow as tf
                
                predictions = model.predict(X, batch_size=batch_size, verbose=0)
                
                if problem_type == "classification":
                    if metadata["num_classes"] == 2:
                        # Binary classification
                        pred_classes = (predictions > 0.5).astype(int).flatten()
                        pred_proba = predictions.flatten()
                    else:
                        # Multi-class classification
                        pred_classes = predictions.argmax(axis=1)
                        pred_proba = predictions
                else:
                    # Regression
                    pred_classes = predictions.flatten()
                    pred_proba = None
                
            elif framework == "pytorch":
                import torch
                
                model.eval()
                X_tensor = torch.FloatTensor(X)
                
                with torch.no_grad():
                    outputs = model(X_tensor)
                    
                    if problem_type == "classification":
                        if metadata["num_classes"] == 2:
                            predictions = torch.sigmoid(outputs).numpy().flatten()
                            pred_classes = (predictions > 0.5).astype(int)
                            pred_proba = predictions
                        else:
                            predictions = torch.softmax(outputs, dim=1).numpy()
                            pred_classes = predictions.argmax(axis=1)
                            pred_proba = predictions
                    else:
                        predictions = outputs.numpy().flatten()
                        pred_classes = predictions
                        pred_proba = None
            
            return {
                "success": True,
                "predictions": pred_classes.tolist(),
                "probabilities": pred_proba.tolist() if pred_proba is not None else None,
                "metadata": metadata,
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "metadata": metadata,
            }

    def create_model(
            self,
            X_train: pd.DataFrame,
            y_train: pd.Series,
            X_val: pd.DataFrame,
            y_val: pd.Series,
            model_type: str,
            framework: str,
            problem_type: str = "classification",
            config: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
        """Método wrapper para criar modelos de forma simplificada."""
        
        # Preparar input shape inicial (será atualizado após limpeza)
        if model_type == "mlp":
            input_shape = (X_train.shape[1],)
        elif model_type == "cnn":
            input_shape = (X_train.shape[1], 1)
        elif model_type == "lstm":
            input_shape = (1, X_train.shape[1])
        else:
            raise ValueError(f"Model type not supported: {model_type}")
        
        # Determinar número de classes
        if problem_type == "classification":
            num_classes = len(y_train.unique())
        else:
            num_classes = 1
        
        # Converter para arrays numpy e limpar dados
        X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
        X_val_np = X_val.values if hasattr(X_val, 'values') else X_val
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        y_val_np = y_val.values if hasattr(y_val, 'values') else y_val
        
        # Detectar se temos dados de texto (NLP) ou dados tabulares
        if self._is_text_data(X_train):
            # Processamento para NLP
            X_train_clean, y_train_clean = self._process_text_data(X_train, y_train, problem_type)
            X_val_clean, y_val_clean = self._process_text_data(X_val, y_val, problem_type, is_fit=False)
        else:
            # Processamento para dados tabulares numéricos
            X_train_clean = self._clean_numeric_data(X_train_np)
            X_val_clean = self._clean_numeric_data(X_val_np)
            y_train_clean = self._clean_target_data(y_train_np, problem_type)
            y_val_clean = self._clean_target_data(y_val_np, problem_type)
        
        # Atualizar input_shape baseado nos dados limpos
        if model_type == "mlp":
            input_shape = (X_train_clean.shape[1],)
        elif model_type == "cnn":
            input_shape = (X_train_clean.shape[1], 1)
        elif model_type == "lstm":
            input_shape = (1, X_train_clean.shape[1])
        
        # Usar configurações padrão se custom_config for None
        if config is None:
            config = self.default_configs[framework][model_type]
        
        # Chamar método específico
        if framework == "tensorflow":
            if model_type == "mlp":
                return self.create_tensorflow_mlp(
                    X_train_clean, y_train_clean, X_val_clean, y_val_clean,
                    input_shape, num_classes, config, problem_type
                )
            elif model_type == "cnn":
                return self.create_tensorflow_cnn(
                    X_train_clean, y_train_clean, X_val_clean, y_val_clean,
                    input_shape, num_classes, config, problem_type
                )
            elif model_type == "lstm":
                return self.create_tensorflow_lstm(
                    X_train_clean, y_train_clean, X_val_clean, y_val_clean,
                    input_shape, num_classes, config, problem_type
                )
        elif framework == "pytorch":
            if model_type == "mlp":
                return self.create_pytorch_mlp(
                    X_train_clean, y_train_clean, X_val_clean, y_val_clean,
                    input_shape, num_classes, config, problem_type
                )
            elif model_type == "cnn":
                return self.create_pytorch_cnn(
                    X_train_clean, y_train_clean, X_val_clean, y_val_clean,
                    input_shape, num_classes, config, problem_type
                )
            elif model_type == "lstm":
                return self.create_pytorch_lstm(
                    X_train_clean, y_train_clean, X_val_clean, y_val_clean,
                    input_shape, num_classes, config, problem_type
                )
        
        raise ValueError(f"Framework not supported: {framework}")


def get_deep_learning_automl() -> DeepLearningAutoML:
    """Factory function para DeepLearningAutoML."""
    return DeepLearningAutoML()
