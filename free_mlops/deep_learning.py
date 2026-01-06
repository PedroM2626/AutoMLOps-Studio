from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from free_mlops.config import get_settings

# Import PyTorch no nível do módulo para evitar problemas de escopo
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False


class MLPModule(nn.Module):
    """MLP Module for PyTorch - definido como classe de nível superior."""
    
    def __init__(self, input_dim, hidden_layers, dropout_rate, num_classes, problem_type, activation="relu"):
        super(MLPModule, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        # Mapear activations
        activation_map = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "leaky_relu": nn.LeakyReLU,
            "elu": nn.ELU,
            "gelu": nn.GELU
        }
        
        activation_fn = activation_map.get(activation, nn.ReLU)
        
        for units in hidden_layers:
            layers.append(nn.Linear(prev_dim, units))
            layers.append(activation_fn())
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
        y_train_tensor = torch.LongTensor(y_train_clean.flatten()) if problem_type == "classification" else torch.FloatTensor(y_train_clean.flatten())
        X_val_tensor = torch.FloatTensor(X_val_clean)
        y_val_tensor = torch.LongTensor(y_val_clean.flatten()) if problem_type == "classification" else torch.FloatTensor(y_val_clean.flatten())
        
        # Data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
        
        # Usar a classe MLPModule definida no nível do módulo
        model = MLPModule(input_shape[0], config["hidden_layers"], config["dropout_rate"], num_classes, problem_type, config.get("activation", "relu"))
        
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
        max_training_seconds = config.get("max_training_time", 30) * 60  # Converter minutos para segundos
        experiment_id = config.get("experiment_id", f"exp_{int(start_time)}")
        
        print(f"Starting training - Experiment ID: {experiment_id}")
        print(f"Max training time: {max_training_seconds/60:.1f} minutes")
        
        history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None  # Initialize best model state
        
        for epoch in range(config["epochs"]):
            # Verificar tempo máximo de treinamento
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            if elapsed_time >= max_training_seconds:
                print(f"Training stopped: Maximum time limit reached ({elapsed_time/60:.1f} minutes)")
                break
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
                    
                    # Verificar se outputs contém NaN
                    if torch.isnan(outputs).any():
                        print("Warning: NaN detected in model outputs, skipping batch")
                        continue
                    
                    if problem_type == "classification" and num_classes == 2:
                        # Para classificação binária
                        outputs_squeezed = outputs.squeeze()
                        if outputs_squeezed.dim() == 0:
                            outputs_squeezed = outputs_squeezed.unsqueeze(0)
                        loss = criterion(outputs_squeezed, batch_y.float())
                        predicted = (outputs_squeezed > 0.5).long()
                    elif problem_type == "classification":
                        # Para classificação multiclasse
                        loss = criterion(outputs, batch_y.long())
                        predicted = outputs.argmax(dim=1)
                    else:
                        # Para regressão
                        outputs_squeezed = outputs.squeeze()
                        if outputs_squeezed.dim() == 0:
                            outputs_squeezed = outputs_squeezed.unsqueeze(0)
                        loss = criterion(outputs_squeezed, batch_y.float())
                        predicted = None
                    
                    # Verificar se loss é válido
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        val_loss += loss.item()
                    else:
                        print("Warning: Invalid loss detected, skipping batch")
                        continue
                    
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
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        training_time = time.time() - start_time
        
        # Final evaluation
        model.eval()
        final_val_loss = 0.0
        final_val_correct = 0
        final_val_total = 0
        all_predictions = []
        all_labels = []
        
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
                    
                    # Coletar predições para métricas detalhadas
                    pred_np = predicted.cpu().numpy().flatten()
                    labels_np = batch_y.cpu().numpy().flatten()
                    
                    all_predictions.extend(pred_np)
                    all_labels.extend(labels_np)
                else:
                    # Para regressão, coletar valores contínuos
                    if problem_type == "regression":
                        pred_np = outputs.squeeze().cpu().numpy().flatten()
                        labels_np = batch_y.cpu().numpy().flatten()
                        all_predictions.extend(pred_np)
                        all_labels.extend(labels_np)
        
        avg_final_val_loss = final_val_loss / len(val_loader)
        
        # Calcular métricas finais usando as predições já coletadas no último epoch
        validation_metrics = {
            "val_loss": avg_final_val_loss,
        }
        
        if problem_type == "classification":
            val_accuracy = final_val_correct / final_val_total if final_val_total > 0 else 0
            validation_metrics["val_accuracy"] = val_accuracy
            
            # Coletar predições durante a avaliação final (já foi feito acima)
            # Se não coletamos predições suficientes, usar uma abordagem simples
            if len(all_predictions) == 0:
                print("⚠️ No predictions collected, using simple evaluation...")
                model.eval()
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        
                        # Verificar se outputs contém NaN
                        if torch.isnan(outputs).any():
                            continue
                        
                        if num_classes == 2:
                            outputs_squeezed = outputs.squeeze()
                            if outputs_squeezed.dim() == 0:
                                outputs_squeezed = outputs_squeezed.unsqueeze(0)
                            predicted = (outputs_squeezed > 0.5).long()
                        else:
                            predicted = outputs.argmax(dim=1)
                        
                        # Garantir arrays 1D
                        pred_np = predicted.cpu().numpy().flatten()
                        labels_np = batch_y.cpu().numpy().flatten()
                        
                        all_predictions.extend(pred_np)
                        all_labels.extend(labels_np)
            
            # Calcular métricas detalhadas
            print(f"Total predictions collected: {len(all_predictions)}")
            print(f"Total labels collected: {len(all_labels)}")
            print(f"Sample predictions: {all_predictions[:5] if all_predictions else 'None'}")
            print(f"Sample labels: {all_labels[:5] if all_labels else 'None'}")
            
            if len(all_predictions) > 0 and len(all_labels) > 0:
                from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
                
                all_predictions = np.array(all_predictions)
                all_labels = np.array(all_labels)
                
                print(f"Predictions array shape: {all_predictions.shape}")
                print(f"Labels array shape: {all_labels.shape}")
                print(f"Unique predictions: {np.unique(all_predictions)}")
                print(f"Unique labels: {np.unique(all_labels)}")
                
                try:
                    # Métricas principais
                    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
                    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
                    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
                    
                    validation_metrics["precision"] = float(precision)
                    validation_metrics["recall"] = float(recall)
                    validation_metrics["f1_score"] = float(f1)
                    validation_metrics["precision_weighted"] = float(precision)
                    validation_metrics["recall_weighted"] = float(recall)
                    validation_metrics["f1_weighted"] = float(f1)
                    
                    # Matriz de confusão completa
                    all_classes = sorted(list(set(all_labels) | set(all_predictions)))
                    cm = confusion_matrix(all_labels, all_predictions, labels=all_classes)
                    validation_metrics["confusion_matrix"] = cm.tolist()
                    
                    # Adicionar métricas por classe para análise detalhada
                    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
                        all_labels, all_predictions, average=None, zero_division=0
                    )
                    
                    # Métricas detalhadas por classe
                    validation_metrics["classification_report"] = {
                        "precision": {str(all_classes[i]): float(precision_per_class[i]) for i in range(len(all_classes))},
                        "recall": {str(all_classes[i]): float(recall_per_class[i]) for i in range(len(all_classes))},
                        "f1_score": {str(all_classes[i]): float(f1_per_class[i]) for i in range(len(all_classes))},
                        "support": {str(all_classes[i]): int(support_per_class[i]) for i in range(len(all_classes))},
                        "class_names": [str(cls) for cls in all_classes]
                    }
                    
                    print(f"✅ Final metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                    print(f"✅ Confusion matrix shape: {cm.shape}")
                    print(f"✅ Confusion matrix:\n{cm}")
                    
                except Exception as e:
                    print(f"❌ Error calculating metrics: {e}")
                    import traceback
                    traceback.print_exc()
                    validation_metrics["precision"] = 0.0
                    validation_metrics["recall"] = 0.0
                    validation_metrics["f1_score"] = 0.0
                    validation_metrics["confusion_matrix"] = [[0]]
            else:
                print("❌ No valid predictions collected")
                validation_metrics["precision"] = 0.0
                validation_metrics["recall"] = 0.0
                validation_metrics["f1_score"] = 0.0
                validation_metrics["confusion_matrix"] = [[0]]
            
        else:  # regressão
            # Coletar todas as predições e verdadeiros para métricas de regressão
            all_predictions = []
            all_labels = []
            
            model.eval()
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    predicted = outputs.squeeze()
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())
            
            # Calcular métricas de regressão
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            
            validation_metrics["mae"] = mean_absolute_error(all_labels, all_predictions)
            validation_metrics["mse"] = mean_squared_error(all_labels, all_predictions)
            validation_metrics["rmse"] = np.sqrt(validation_metrics["mse"])
            validation_metrics["r2_score"] = r2_score(all_labels, all_predictions)
        
        # Preparar resultados
        result = {
            "framework": "pytorch",
            "model_type": "mlp",
            "problem_type": problem_type,
            "model": model,
            "config": config,
            "training_time": training_time,
            "history": history,
            "validation_metrics": validation_metrics,
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
        y_train_tensor = torch.LongTensor(y_train_clean.flatten()) if problem_type == "classification" else torch.FloatTensor(y_train_clean.flatten())
        X_val_tensor = torch.FloatTensor(X_val_clean)
        y_val_tensor = torch.LongTensor(y_val_clean.flatten()) if problem_type == "classification" else torch.FloatTensor(y_val_clean.flatten())
        
        # Data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
        
        # Definir modelo CNN
        class CNN(nn.Module):
            def __init__(self, input_channels, conv_layers, dense_layers, dropout_rate, num_classes, problem_type, activation="relu"):
                super(CNN, self).__init__()
                
                layers = []
                prev_channels = input_channels
                
                # Mapear activations
                activation_map = {
                    "relu": nn.ReLU,
                    "tanh": nn.Tanh,
                    "sigmoid": nn.Sigmoid,
                    "leaky_relu": nn.LeakyReLU,
                    "elu": nn.ELU,
                    "gelu": nn.GELU
                }
                
                activation_fn = activation_map.get(activation, nn.ReLU)
                
                # Camadas convolucionais
                for conv_layer in conv_layers:
                    layers.append(nn.Conv2d(prev_channels, conv_layer["filters"], conv_layer["kernel_size"], padding=1))
                    layers.append(activation_fn())
                    layers.append(nn.MaxPool2d(2, 2))
                    prev_channels = conv_layer["filters"]
                
                layers.append(nn.Flatten())
                
                # Camadas densas
                for units in dense_layers:
                    layers.append(nn.Linear(prev_channels, units))
                    layers.append(activation_fn())
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
        
        model = CNN(1, config["conv_layers"], config["dense_layers"], config["dropout_rate"], num_classes, problem_type, config.get("activation", "relu"))
        
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
        max_training_seconds = config.get("max_training_time", 30) * 60  # Converter minutos para segundos
        experiment_id = config.get("experiment_id", f"exp_{int(start_time)}")
        
        print(f"Starting training - Experiment ID: {experiment_id}")
        print(f"Max training time: {max_training_seconds/60:.1f} minutes")
        
        history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None  # Initialize best model state
        
        for epoch in range(config["epochs"]):
            # Verificar tempo máximo de treinamento
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            if elapsed_time >= max_training_seconds:
                print(f"Training stopped: Maximum time limit reached ({elapsed_time/60:.1f} minutes)")
                break
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
                    
                    # Verificar se outputs contém NaN
                    if torch.isnan(outputs).any():
                        print("Warning: NaN detected in model outputs, skipping batch")
                        continue
                    
                    if problem_type == "classification" and num_classes == 2:
                        # Para classificação binária
                        outputs_squeezed = outputs.squeeze()
                        if outputs_squeezed.dim() == 0:
                            outputs_squeezed = outputs_squeezed.unsqueeze(0)
                        loss = criterion(outputs_squeezed, batch_y.float())
                        predicted = (outputs_squeezed > 0.5).long()
                    elif problem_type == "classification":
                        # Para classificação multiclasse
                        loss = criterion(outputs, batch_y.long())
                        predicted = outputs.argmax(dim=1)
                    else:
                        # Para regressão
                        outputs_squeezed = outputs.squeeze()
                        if outputs_squeezed.dim() == 0:
                            outputs_squeezed = outputs_squeezed.unsqueeze(0)
                        loss = criterion(outputs_squeezed, batch_y.float())
                        predicted = None
                    
                    # Verificar se loss é válido
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        val_loss += loss.item()
                    else:
                        print("Warning: Invalid loss detected, skipping batch")
                        continue
                    
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
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        training_time = time.time() - start_time
        
        # Final evaluation
        model.eval()
        final_val_loss = 0.0
        final_val_correct = 0
        final_val_total = 0
        all_predictions = []
        all_labels = []
        
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
                    
                    # Coletar predições para métricas detalhadas
                    pred_np = predicted.cpu().numpy().flatten()
                    labels_np = batch_y.cpu().numpy().flatten()
                    
                    all_predictions.extend(pred_np)
                    all_labels.extend(labels_np)
                else:
                    # Para regressão, coletar valores contínuos
                    if problem_type == "regression":
                        pred_np = outputs.squeeze().cpu().numpy().flatten()
                        labels_np = batch_y.cpu().numpy().flatten()
                        all_predictions.extend(pred_np)
                        all_labels.extend(labels_np)
        
        avg_final_val_loss = final_val_loss / len(val_loader)
        
        # Calcular métricas finais usando as predições já coletadas no último epoch
        validation_metrics = {
            "val_loss": avg_final_val_loss,
        }
        
        if problem_type == "classification":
            val_accuracy = final_val_correct / final_val_total if final_val_total > 0 else 0
            validation_metrics["val_accuracy"] = val_accuracy
            
            # Coletar predições durante a avaliação final (já foi feito acima)
            # Se não coletamos predições suficientes, usar uma abordagem simples
            if len(all_predictions) == 0:
                print("⚠️ No predictions collected, using simple evaluation...")
                model.eval()
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        
                        # Verificar se outputs contém NaN
                        if torch.isnan(outputs).any():
                            continue
                        
                        if num_classes == 2:
                            outputs_squeezed = outputs.squeeze()
                            if outputs_squeezed.dim() == 0:
                                outputs_squeezed = outputs_squeezed.unsqueeze(0)
                            predicted = (outputs_squeezed > 0.5).long()
                        else:
                            predicted = outputs.argmax(dim=1)
                        
                        # Garantir arrays 1D
                        pred_np = predicted.cpu().numpy().flatten()
                        labels_np = batch_y.cpu().numpy().flatten()
                        
                        all_predictions.extend(pred_np)
                        all_labels.extend(labels_np)
            
            # Calcular métricas detalhadas
            print(f"Total predictions collected: {len(all_predictions)}")
            print(f"Total labels collected: {len(all_labels)}")
            print(f"Sample predictions: {all_predictions[:5] if all_predictions else 'None'}")
            print(f"Sample labels: {all_labels[:5] if all_labels else 'None'}")
            
            if len(all_predictions) > 0 and len(all_labels) > 0:
                from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
                
                all_predictions = np.array(all_predictions)
                all_labels = np.array(all_labels)
                
                print(f"Predictions array shape: {all_predictions.shape}")
                print(f"Labels array shape: {all_labels.shape}")
                print(f"Unique predictions: {np.unique(all_predictions)}")
                print(f"Unique labels: {np.unique(all_labels)}")
                
                try:
                    # Métricas principais
                    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
                    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
                    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
                    
                    validation_metrics["precision"] = float(precision)
                    validation_metrics["recall"] = float(recall)
                    validation_metrics["f1_score"] = float(f1)
                    validation_metrics["precision_weighted"] = float(precision)
                    validation_metrics["recall_weighted"] = float(recall)
                    validation_metrics["f1_weighted"] = float(f1)
                    
                    # Matriz de confusão completa
                    all_classes = sorted(list(set(all_labels) | set(all_predictions)))
                    cm = confusion_matrix(all_labels, all_predictions, labels=all_classes)
                    validation_metrics["confusion_matrix"] = cm.tolist()
                    
                    # Adicionar métricas por classe para análise detalhada
                    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
                        all_labels, all_predictions, average=None, zero_division=0
                    )
                    
                    # Métricas detalhadas por classe
                    validation_metrics["classification_report"] = {
                        "precision": {str(all_classes[i]): float(precision_per_class[i]) for i in range(len(all_classes))},
                        "recall": {str(all_classes[i]): float(recall_per_class[i]) for i in range(len(all_classes))},
                        "f1_score": {str(all_classes[i]): float(f1_per_class[i]) for i in range(len(all_classes))},
                        "support": {str(all_classes[i]): int(support_per_class[i]) for i in range(len(all_classes))},
                        "class_names": [str(cls) for cls in all_classes]
                    }
                    
                    print(f"✅ Final metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                    print(f"✅ Confusion matrix shape: {cm.shape}")
                    print(f"✅ Confusion matrix:\n{cm}")
                    
                except Exception as e:
                    print(f"❌ Error calculating metrics: {e}")
                    import traceback
                    traceback.print_exc()
                    validation_metrics["precision"] = 0.0
                    validation_metrics["recall"] = 0.0
                    validation_metrics["f1_score"] = 0.0
                    validation_metrics["confusion_matrix"] = [[0]]
            else:
                print("❌ No valid predictions collected")
                validation_metrics["precision"] = 0.0
                validation_metrics["recall"] = 0.0
                validation_metrics["f1_score"] = 0.0
                validation_metrics["confusion_matrix"] = [[0]]
            
        else:  # regressão
            # Coletar todas as predições e verdadeiros para métricas de regressão
            all_predictions = []
            all_labels = []
            
            model.eval()
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    predicted = outputs.squeeze()
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())
            
            # Calcular métricas de regressão
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            
            validation_metrics["mae"] = mean_absolute_error(all_labels, all_predictions)
            validation_metrics["mse"] = mean_squared_error(all_labels, all_predictions)
            validation_metrics["rmse"] = np.sqrt(validation_metrics["mse"])
            validation_metrics["r2_score"] = r2_score(all_labels, all_predictions)
        
        # Preparar resultados
        result = {
            "framework": "pytorch",
            "model_type": "cnn",
            "problem_type": problem_type,
            "model": model,
            "config": config,
            "training_time": training_time,
            "history": history,
            "validation_metrics": validation_metrics,
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
        y_train_tensor = torch.LongTensor(y_train_clean.flatten()) if problem_type == "classification" else torch.FloatTensor(y_train_clean.flatten())
        X_val_tensor = torch.FloatTensor(X_val_clean)
        y_val_tensor = torch.LongTensor(y_val_clean.flatten()) if problem_type == "classification" else torch.FloatTensor(y_val_clean.flatten())
        
        # Data loaders
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
        
        # Definir modelo LSTM
        class LSTM(nn.Module):
            def __init__(self, input_size, lstm_layers, dense_layers, dropout_rate, num_classes, problem_type, activation="relu"):
                super(LSTM, self).__init__()
                
                # Mapear activations
                activation_map = {
                    "relu": nn.ReLU,
                    "tanh": nn.Tanh,
                    "sigmoid": nn.Sigmoid,
                    "leaky_relu": nn.LeakyReLU,
                    "elu": nn.ELU,
                    "gelu": nn.GELU
                }
                
                activation_fn = activation_map.get(activation, nn.ReLU)
                
                # Camada LSTM
                self.lstm = nn.LSTM(input_size, lstm_layers[0], lstm_layers[1], batch_first=True, dropout=dropout_rate)
                
                # Camadas densas
                layers = []
                prev_size = lstm_layers[1]
                for units in dense_layers:
                    layers.append(nn.Linear(prev_size, units))
                    layers.append(activation_fn())
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
        
        model = LSTM(X_train.shape[1], config["lstm_layers"], config["dense_layers"], config["dropout_rate"], num_classes, problem_type, config.get("activation", "relu"))
        
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
        max_training_seconds = config.get("max_training_time", 30) * 60  # Converter minutos para segundos
        experiment_id = config.get("experiment_id", f"exp_{int(start_time)}")
        
        print(f"Starting training - Experiment ID: {experiment_id}")
        print(f"Max training time: {max_training_seconds/60:.1f} minutes")
        
        history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None  # Initialize best model state
        
        for epoch in range(config["epochs"]):
            # Verificar tempo máximo de treinamento
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            if elapsed_time >= max_training_seconds:
                print(f"Training stopped: Maximum time limit reached ({elapsed_time/60:.1f} minutes)")
                break
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
                    
                    # Verificar se outputs contém NaN
                    if torch.isnan(outputs).any():
                        print("Warning: NaN detected in model outputs, skipping batch")
                        continue
                    
                    if problem_type == "classification" and num_classes == 2:
                        # Para classificação binária
                        outputs_squeezed = outputs.squeeze()
                        if outputs_squeezed.dim() == 0:
                            outputs_squeezed = outputs_squeezed.unsqueeze(0)
                        loss = criterion(outputs_squeezed, batch_y.float())
                        predicted = (outputs_squeezed > 0.5).long()
                    elif problem_type == "classification":
                        # Para classificação multiclasse
                        loss = criterion(outputs, batch_y.long())
                        predicted = outputs.argmax(dim=1)
                    else:
                        # Para regressão
                        outputs_squeezed = outputs.squeeze()
                        if outputs_squeezed.dim() == 0:
                            outputs_squeezed = outputs_squeezed.unsqueeze(0)
                        loss = criterion(outputs_squeezed, batch_y.float())
                        predicted = None
                    
                    # Verificar se loss é válido
                    if not torch.isnan(loss) and not torch.isinf(loss):
                        val_loss += loss.item()
                    else:
                        print("Warning: Invalid loss detected, skipping batch")
                        continue
                    
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
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        training_time = time.time() - start_time
        
        # Final evaluation
        model.eval()
        final_val_loss = 0.0
        final_val_correct = 0
        final_val_total = 0
        all_predictions = []
        all_labels = []
        
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
                    
                    # Coletar predições para métricas detalhadas
                    pred_np = predicted.cpu().numpy().flatten()
                    labels_np = batch_y.cpu().numpy().flatten()
                    
                    all_predictions.extend(pred_np)
                    all_labels.extend(labels_np)
                else:
                    # Para regressão, coletar valores contínuos
                    if problem_type == "regression":
                        pred_np = outputs.squeeze().cpu().numpy().flatten()
                        labels_np = batch_y.cpu().numpy().flatten()
                        all_predictions.extend(pred_np)
                        all_labels.extend(labels_np)
        
        avg_final_val_loss = final_val_loss / len(val_loader)
        
        # Calcular métricas finais usando as predições já coletadas no último epoch
        validation_metrics = {
            "val_loss": avg_final_val_loss,
        }
        
        if problem_type == "classification":
            val_accuracy = final_val_correct / final_val_total if final_val_total > 0 else 0
            validation_metrics["val_accuracy"] = val_accuracy
            
            # Coletar predições durante a avaliação final (já foi feito acima)
            # Se não coletamos predições suficientes, usar uma abordagem simples
            if len(all_predictions) == 0:
                print("⚠️ No predictions collected, using simple evaluation...")
                model.eval()
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        
                        # Verificar se outputs contém NaN
                        if torch.isnan(outputs).any():
                            continue
                        
                        if num_classes == 2:
                            outputs_squeezed = outputs.squeeze()
                            if outputs_squeezed.dim() == 0:
                                outputs_squeezed = outputs_squeezed.unsqueeze(0)
                            predicted = (outputs_squeezed > 0.5).long()
                        else:
                            predicted = outputs.argmax(dim=1)
                        
                        # Garantir arrays 1D
                        pred_np = predicted.cpu().numpy().flatten()
                        labels_np = batch_y.cpu().numpy().flatten()
                        
                        all_predictions.extend(pred_np)
                        all_labels.extend(labels_np)
            
            # Calcular métricas detalhadas
            print(f"Total predictions collected: {len(all_predictions)}")
            print(f"Total labels collected: {len(all_labels)}")
            print(f"Sample predictions: {all_predictions[:5] if all_predictions else 'None'}")
            print(f"Sample labels: {all_labels[:5] if all_labels else 'None'}")
            
            if len(all_predictions) > 0 and len(all_labels) > 0:
                from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
                
                all_predictions = np.array(all_predictions)
                all_labels = np.array(all_labels)
                
                print(f"Predictions array shape: {all_predictions.shape}")
                print(f"Labels array shape: {all_labels.shape}")
                print(f"Unique predictions: {np.unique(all_predictions)}")
                print(f"Unique labels: {np.unique(all_labels)}")
                
                try:
                    # Métricas principais
                    precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
                    recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
                    f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
                    
                    validation_metrics["precision"] = float(precision)
                    validation_metrics["recall"] = float(recall)
                    validation_metrics["f1_score"] = float(f1)
                    validation_metrics["precision_weighted"] = float(precision)
                    validation_metrics["recall_weighted"] = float(recall)
                    validation_metrics["f1_weighted"] = float(f1)
                    
                    # Matriz de confusão completa
                    all_classes = sorted(list(set(all_labels) | set(all_predictions)))
                    cm = confusion_matrix(all_labels, all_predictions, labels=all_classes)
                    validation_metrics["confusion_matrix"] = cm.tolist()
                    
                    # Adicionar métricas por classe para análise detalhada
                    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
                        all_labels, all_predictions, average=None, zero_division=0
                    )
                    
                    # Métricas detalhadas por classe
                    validation_metrics["classification_report"] = {
                        "precision": {str(all_classes[i]): float(precision_per_class[i]) for i in range(len(all_classes))},
                        "recall": {str(all_classes[i]): float(recall_per_class[i]) for i in range(len(all_classes))},
                        "f1_score": {str(all_classes[i]): float(f1_per_class[i]) for i in range(len(all_classes))},
                        "support": {str(all_classes[i]): int(support_per_class[i]) for i in range(len(all_classes))},
                        "class_names": [str(cls) for cls in all_classes]
                    }
                    
                    print(f"✅ Final metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
                    print(f"✅ Confusion matrix shape: {cm.shape}")
                    print(f"✅ Confusion matrix:\n{cm}")
                    
                except Exception as e:
                    print(f"❌ Error calculating metrics: {e}")
                    import traceback
                    traceback.print_exc()
                    validation_metrics["precision"] = 0.0
                    validation_metrics["recall"] = 0.0
                    validation_metrics["f1_score"] = 0.0
                    validation_metrics["confusion_matrix"] = [[0]]
            else:
                print("❌ No valid predictions collected")
                validation_metrics["precision"] = 0.0
                validation_metrics["recall"] = 0.0
                validation_metrics["f1_score"] = 0.0
                validation_metrics["confusion_matrix"] = [[0]]
            
        else:  # regressão
            # Coletar todas as predições e verdadeiros para métricas de regressão
            all_predictions = []
            all_labels = []
            
            model.eval()
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = model(batch_X)
                    predicted = outputs.squeeze()
                    
                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(batch_y.cpu().numpy())
            
            # Calcular métricas de regressão
            from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
            
            all_predictions = np.array(all_predictions)
            all_labels = np.array(all_labels)
            
            validation_metrics["mae"] = mean_absolute_error(all_labels, all_predictions)
            validation_metrics["mse"] = mean_squared_error(all_labels, all_predictions)
            validation_metrics["rmse"] = np.sqrt(validation_metrics["mse"])
            validation_metrics["r2_score"] = r2_score(all_labels, all_predictions)
        
        # Preparar resultados
        result = {
            "framework": "pytorch",
            "model_type": "lstm",
            "problem_type": problem_type,
            "model": model,
            "config": config,
            "training_time": training_time,
            "history": history,
            "validation_metrics": validation_metrics,
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
    
    def _clean_target_data(self, data: np.ndarray, problem_type: str) -> np.ndarray:
        """Clean target data."""
        import pandas as pd
        
        if isinstance(data, np.ndarray):
            if len(data.shape) == 1:
                series = pd.Series(data)
            else:
                series = pd.Series(data.flatten())
        else:
            series = data.copy()
        
        if problem_type == "classification":
            try:
                return series.astype(np.int32).values
            except (ValueError, TypeError):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                encoded = le.fit_transform(series.astype(str))
                return encoded.astype(np.int32)
        else:
            try:
                return pd.to_numeric(series, errors='raise').astype(np.float32).values
            except (ValueError, TypeError):
                raise ValueError("Para regressão, os valores de target devem ser numéricos.")
    
    def _is_text_data(self, X: pd.DataFrame) -> bool:
        """Detect if data is primarily text (NLP task)."""
        import pandas as pd
        
        # Check if we have string columns
        text_cols = []
        if hasattr(X, 'columns'):
            for col in X.columns:
                if X[col].dtype == 'object':
                    # Check if it's actually text (not categorical codes)
                    sample_values = X[col].dropna().head(10).astype(str)
                    avg_length = sample_values.str.len().mean()
                    if avg_length > 10:  # Likely text, not categorical codes
                        text_cols.append(col)
        
        # If we have text columns and they contain substantial text, treat as NLP
        if text_cols:
            return True
        
        # Also check if all columns are non-numeric (likely text data)
        if hasattr(X, 'select_dtypes'):
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) == 0:
                return True
        else:
            # Se for numpy array, verificar se todos os dados são numéricos
            try:
                np.array(X, dtype=np.float64)
                return False  # Se conseguir converter para float, é numérico
            except (ValueError, TypeError):
                return True  # Se não conseguir, é texto
        
        return False
    
    def _process_text_data(self, X: pd.DataFrame, y: pd.Series, problem_type: str, is_fit: bool = True):
        """Process text data for NLP tasks."""
        try:
            from .nlp_deep_learning import NLPProcessor
            
            # Get text columns
            text_cols = []
            if hasattr(X, 'columns'):
                for col in X.columns:
                    if X[col].dtype == 'object':
                        text_cols.append(col)
            
            if not text_cols:
                raise ValueError("Nenhuma coluna de texto encontrada para processamento NLP.")
            
            # Combine text columns (if multiple)
            if len(text_cols) == 1:
                texts = X[text_cols[0]].astype(str).tolist()
            else:
                # Combine multiple text columns with spaces
                texts = X[text_cols].astype(str).apply(' '.join, axis=1).tolist()
            
            # Get labels
            labels = y.astype(str).tolist()
            
            # Process with NLP processor
            if is_fit:
                processor = NLPProcessor(method="tfidf")
                X_processed, y_processed = processor.fit_transform(texts, labels)
                # Store processor for later use
                self.nlp_processor = processor
                # Ensure y_processed is 1D array
                if len(y_processed.shape) > 1 and y_processed.shape[1] == 1:
                    y_processed = y_processed.flatten()
            else:
                if not hasattr(self, 'nlp_processor'):
                    raise ValueError("NLP processor not fitted. Call with is_fit=True first.")
                X_processed = self.nlp_processor.transform(texts)
                y_processed = self.nlp_processor.transform(labels) if hasattr(labels, '__iter__') else self.nlp_processor.transform([labels])
                # Ensure y_processed is 1D array
                if len(y_processed.shape) > 1 and y_processed.shape[1] == 1:
                    y_processed = y_processed.flatten()
            
            return X_processed, y_processed
            
        except ImportError:
            # Fallback: simple text processing
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import LabelEncoder
            
            # Get text columns
            if hasattr(X, 'columns'):
                text_cols = [col for col in X.columns if X[col].dtype == 'object']
            else:
                text_cols = []
            if not text_cols:
                raise ValueError("Nenhuma coluna de texto encontrada.")
            
            # Combine text
            if len(text_cols) == 1:
                texts = X[text_cols[0]].astype(str).tolist()
            else:
                texts = X[text_cols].astype(str).apply(' '.join, axis=1).tolist()
            
            # Process labels
            labels = y.astype(str).tolist()
            
            if is_fit:
                self.text_vectorizer = TfidfVectorizer(max_features=10000)
                X_processed = self.text_vectorizer.fit_transform(texts).toarray()
                
                self.label_encoder = LabelEncoder()
                y_processed = self.label_encoder.fit_transform(labels)
                # Ensure y_processed is 1D array
                if len(y_processed.shape) > 1 and y_processed.shape[1] == 1:
                    y_processed = y_processed.flatten()
            else:
                if not hasattr(self, 'text_vectorizer'):
                    raise ValueError("Text processor not fitted.")
                X_processed = self.text_vectorizer.transform(texts).toarray()
                y_processed = self.label_encoder.transform(labels)
                # Ensure y_processed is 1D array
                if len(y_processed.shape) > 1 and y_processed.shape[1] == 1:
                    y_processed = y_processed.flatten()
            
            return X_processed, y_processed
    
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
            if hasattr(y_train, 'unique'):
                num_classes = len(y_train.unique())
            else:
                # Se for numpy array, converter para pandas Series
                import pandas as pd
                num_classes = len(pd.Series(y_train).unique())
        else:
            num_classes = 1
        
        # Converter para arrays numpy e limpar dados
        X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
        X_val_np = X_val.values if hasattr(X_val, 'values') else X_val
        y_train_np = y_train.values if hasattr(y_train, 'values') else y_train
        y_val_np = y_val.values if hasattr(y_val, 'values') else y_val
        
        # Limpar dados - remover NaN e infinitos
        X_train_np = np.nan_to_num(X_train_np, nan=0.0, posinf=1e6, neginf=-1e6)
        X_val_np = np.nan_to_num(X_val_np, nan=0.0, posinf=1e6, neginf=-1e6)
        y_train_np = np.nan_to_num(y_train_np, nan=0.0, posinf=1e6, neginf=-1e6)
        y_val_np = np.nan_to_num(y_val_np, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Verificar se há dados válidos
        if np.any(np.isnan(X_train_np)) or np.any(np.isnan(y_train_np)):
            raise ValueError("Dados de treinamento contêm NaN após limpeza")
        if np.any(np.isinf(X_train_np)) or np.any(np.isinf(y_train_np)):
            raise ValueError("Dados de treinamento contêm valores infinitos após limpeza")
        
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
