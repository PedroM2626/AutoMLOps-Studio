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
                    "conv_layers": [{"out_channels": 32, "kernel_size": 3}, {"out_channels": 64, "kernel_size": 3}],
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
        
        # Converter para tensores
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train) if problem_type == "classification" else torch.FloatTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val) if problem_type == "classification" else torch.FloatTensor(y_val)
        
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
    
    def save_model(self, result: Dict[str, Any], model_name: str) -> str:
        """Salva modelo treinado."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.models_dir / f"{model_name}_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        framework = result["framework"]
        model_type = result["model_type"]
        model = result["model"]
        
        if framework == "tensorflow":
            try:
                import tensorflow as tf
                model.save(model_dir / "model")
            except Exception as e:
                raise RuntimeError(f"Erro ao salvar modelo TensorFlow: {e}")
        
        elif framework == "pytorch":
            try:
                import torch
                torch.save(model.state_dict(), model_dir / "model.pth")
                
                # Salvar arquitetura
                with open(model_dir / "architecture.txt", "w") as f:
                    f.write(str(model))
            except Exception as e:
                raise RuntimeError(f"Erro ao salvar modelo PyTorch: {e}")
        
        # Salvar metadados
        metadata = {
            "framework": framework,
            "model_type": model_type,
            "problem_type": result["problem_type"],
            "config": result["config"],
            "training_time": result["training_time"],
            "validation_metrics": result["validation_metrics"],
            "input_shape": result["input_shape"],
            "num_classes": result["num_classes"],
            "saved_at": datetime.now(timezone.utc).isoformat(),
            "model_name": model_name,
        }
        
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
        
        return str(model_dir)
    
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


def get_deep_learning_automl() -> DeepLearningAutoML:
    """Factory function para DeepLearningAutoML."""
    return DeepLearningAutoML()
