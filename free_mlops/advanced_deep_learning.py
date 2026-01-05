"""
Advanced Deep Learning models including Transformers and attention mechanisms.
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
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers, callbacks
except ImportError:
    raise ImportError("TensorFlow não está instalado. Instale com: pip install tensorflow")

try:
    import mlflow
    import mlflow.pytorch
    import mlflow.tensorflow
except ImportError:
    print("MLflow não disponível. Tracking será desabilitado.")

try:
    import shap
    import captum.attr
    from captum.attr import IntegratedGradients, DeepLift, GradientShap
except ImportError:
    print("SHAP/Captum não disponível. Interpretabilidade será limitada.")


class TabTransformer(nn.Module):
    """
    TabTransformer: Transformer-based model for tabular data.
    Combines embeddings for categorical features with attention mechanisms.
    """
    
    def __init__(
        self,
        continuous_features: int,
        categorical_features: List[int],
        categorical_cardinalities: List[int],
        embedding_dim: int = 8,
        num_heads: int = 8,
        num_layers: int = 6,
        hidden_dim: int = 128,
        dropout_rate: float = 0.1,
        num_classes: int = 2
    ):
        super(TabTransformer, self).__init__()
        
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
        
    def forward(self, x_continuous, x_categorical):
        # Process categorical features
        embedded_features = []
        for i, embedding in enumerate(self.embeddings):
            embedded = embedding(x_categorical[:, i])
            embedded = self.embedding_norm(embedded)
            embedded_features.append(embedded)
        
        # Stack embeddings and apply transformer
        cat_embeddings = torch.stack(embedded_features, dim=1)  # [batch, num_cat, embedding_dim]
        cat_transformed = self.transformer(cat_embeddings)  # [batch, num_cat, embedding_dim]
        cat_flattened = cat_transformed.view(cat_transformed.size(0), -1)  # [batch, num_cat * embedding_dim]
        
        # Process continuous features
        cont_normalized = self.continuous_norm(x_continuous)
        
        # Concatenate and classify
        combined = torch.cat([cont_normalized, cat_flattened], dim=1)
        output = self.classifier(combined)
        
        return output


class VisionTransformer(nn.Module):
    """
    Simplified Vision Transformer for image-like data.
    Can be used for tabular data reshaped as "images".
    """
    
    def __init__(
        self,
        input_dim: int,
        patch_size: int = 4,
        embedding_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 4,
        hidden_dim: int = 128,
        dropout_rate: float = 0.1,
        num_classes: int = 2
    ):
        super(VisionTransformer, self).__init__()
        
        self.input_dim = input_dim
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.num_patches = (input_dim // patch_size) ** 2
        
        # Patch embedding
        self.patch_embedding = nn.Linear(patch_size, embedding_dim)
        
        # Position embedding
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, embedding_dim)
        )
        
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
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        # Reshape input into patches
        batch_size = x.size(0)
        patches = x.view(batch_size, self.num_patches, self.patch_size)
        
        # Patch embedding
        patch_embeddings = self.patch_embedding(patches)
        
        # Add position embedding
        embeddings = patch_embeddings + self.position_embedding
        
        # Apply transformer
        transformed = self.transformer(embeddings)
        
        # Global average pooling and classification
        pooled = transformed.mean(dim=1)
        output = self.classifier(pooled)
        
        return output


class AdvancedLearningRateScheduler:
    """
    Advanced learning rate scheduling strategies.
    """
    
    @staticmethod
    def get_cosine_scheduler(optimizer, total_steps: int, warmup_steps: int = 0):
        """Cosine annealing with warmup."""
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    @staticmethod
    def get_reduce_on_plateau(optimizer, mode: str = 'min', factor: float = 0.5, patience: int = 5):
        """Reduce LR on plateau."""
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, verbose=True
        )
    
    @staticmethod
    def get_cyclic_scheduler(optimizer, base_lr: float, max_lr: float, step_size_up: int):
        """Cyclic learning rate."""
        return torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=base_lr, max_lr=max_lr, 
            step_size_up=step_size_up, mode='triangular'
        )


class ModelExplainer:
    """
    Model explanation using SHAP and Captum.
    """
    
    def __init__(self, model, framework: str = 'pytorch'):
        self.model = model
        self.framework = framework
        self.explainer = None
        
    def explain_pytorch_model(self, X_background, feature_names=None):
        """Generate SHAP explanations for PyTorch models."""
        try:
            # Use DeepLift for PyTorch models
            self.explainer = captum.attr.DeepLift(self.model)
            
            # Convert to tensor
            if isinstance(X_background, pd.DataFrame):
                X_tensor = torch.FloatTensor(X_background.values)
            else:
                X_tensor = torch.FloatTensor(X_background)
            
            # Generate attributions
            attributions = self.explainer.attribute(X_tensor)
            
            # Convert to SHAP values format
            shap_values = attributions.detach().numpy()
            
            return {
                'shap_values': shap_values,
                'feature_names': feature_names or list(range(X_background.shape[1])),
                'method': 'DeepLift'
            }
        except Exception as e:
            print(f"Erro na explicação: {e}")
            return None
    
    def plot_feature_importance(self, shap_values, feature_names, max_features: int = 20):
        """Plot feature importance using SHAP."""
        try:
            import matplotlib.pyplot as plt
            
            # Calculate mean absolute SHAP values
            mean_shap = np.mean(np.abs(shap_values), axis=0)
            
            # Get top features
            top_indices = np.argsort(mean_shap)[-max_features:]
            top_values = mean_shap[top_indices]
            top_names = [feature_names[i] for i in top_indices]
            
            # Create plot
            plt.figure(figsize=(10, 8))
            plt.barh(top_names, top_values)
            plt.title('Feature Importance (SHAP Values)')
            plt.xlabel('Mean |SHAP Value|')
            plt.tight_layout()
            
            return plt.gcf()
        except Exception as e:
            print(f"Erro no plot: {e}")
            return None


class MLflowTracker:
    """
    MLflow integration for experiment tracking.
    """
    
    def __init__(self, experiment_name: str = "deep_learning"):
        self.experiment_name = experiment_name
        self.run = None
        
        try:
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            print(f"MLflow não disponível: {e}")
    
    def start_run(self, run_name: str = None):
        """Start MLflow run."""
        try:
            self.run = mlflow.start_run(run_name=run_name)
            return self.run
        except Exception as e:
            print(f"Erro ao iniciar run MLflow: {e}")
            return None
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters."""
        try:
            mlflow.log_params(params)
        except Exception as e:
            print(f"Erro ao logar parâmetros: {e}")
    
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Log metrics."""
        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            print(f"Erro ao logar métricas: {e}")
    
    def log_model(self, model, framework: str, model_name: str = "model"):
        """Log model to MLflow."""
        try:
            if framework == "pytorch":
                mlflow.pytorch.log_model(model, model_name)
            elif framework == "tensorflow":
                mlflow.tensorflow.log_model(model, model_name)
        except Exception as e:
            print(f"Erro ao logar modelo: {e}")
    
    def end_run(self):
        """End MLflow run."""
        try:
            if self.run:
                mlflow.end_run()
        except Exception as e:
            print(f"Erro ao finalizar run: {e}")


class AdvancedDeepLearningAutoML:
    """
    Advanced Deep Learning AutoML with Transformers and modern techniques.
    """
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Initialize MLflow tracker
        self.mlflow_tracker = MLflowTracker()
        
        # Default configurations
        self.default_configs = {
            "pytorch": {
                "tabtransformer": {
                    "embedding_dim": 8,
                    "num_heads": 8,
                    "num_layers": 6,
                    "hidden_dim": 128,
                    "dropout_rate": 0.1,
                    "batch_size": 32,
                    "epochs": 100,
                    "learning_rate": 0.001,
                    "optimizer": "adam",
                    "scheduler": "cosine",
                    "warmup_steps": 100,
                },
                "vision_transformer": {
                    "patch_size": 4,
                    "embedding_dim": 64,
                    "num_heads": 4,
                    "num_layers": 4,
                    "hidden_dim": 128,
                    "dropout_rate": 0.1,
                    "batch_size": 32,
                    "epochs": 100,
                    "learning_rate": 0.001,
                    "optimizer": "adam",
                    "scheduler": "cosine",
                }
            }
        }
    
    def create_tabtransformer(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        config: Optional[Dict[str, Any]] = None,
        problem_type: str = "classification"
    ) -> Dict[str, Any]:
        """Create and train TabTransformer model."""
        
        config = config or self.default_configs["pytorch"]["tabtransformer"]
        
        # Start MLflow run
        run_name = f"tabtransformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.mlflow_tracker.start_run(run_name)
        
        try:
            # Identify categorical and continuous features
            categorical_cols = []
            continuous_cols = []
            
            if hasattr(X_train, 'columns'):
                for col in X_train.columns:
                    if X_train[col].dtype == 'object' or X_train[col].dtype.name == 'category':
                        categorical_cols.append(col)
                    else:
                        continuous_cols.append(col)
            else:
                # Se for numpy array, verificar tipos de dados
                try:
                    # Tentar converter para diferentes tipos
                    for i in range(X_train.shape[1]):
                        col_data = X_train[:, i]
                        if col_data.dtype == 'object':
                            categorical_cols.append(f'col_{i}')
                        else:
                            continuous_cols.append(f'col_{i}')
                except:
                    # Se falhar, assumir que todas são contínuas
                    continuous_cols = [f'col_{i}' for i in range(X_train.shape[1])]
            
            # Prepare data
            X_train_processed = X_train.copy()
            X_val_processed = X_val.copy()
            
            # Encode categorical features
            from sklearn.preprocessing import LabelEncoder
            categorical_encoders = {}
            categorical_cardinalities = []
            
            for col in categorical_cols:
                le = LabelEncoder()
                X_train_processed[col] = le.fit_transform(X_train_processed[col].astype(str))
                X_val_processed[col] = le.transform(X_val_processed[col].astype(str))
                categorical_encoders[col] = le
                categorical_cardinalities.append(len(le.classes_))
            
            # Convert to tensors
            X_continuous = torch.FloatTensor(X_train_processed[continuous_cols].values)
            X_categorical = torch.LongTensor(X_train_processed[categorical_cols].values)
            y_tensor = torch.LongTensor(y_train.values) if problem_type == "classification" else torch.FloatTensor(y_train.values)
            
            X_val_continuous = torch.FloatTensor(X_val_processed[continuous_cols].values)
            X_val_categorical = torch.LongTensor(X_val_processed[categorical_cols].values)
            y_val_tensor = torch.LongTensor(y_val.values) if problem_type == "classification" else torch.FloatTensor(y_val.values)
            
            # Create model
            if problem_type == "classification":
                if hasattr(y_train, 'unique'):
                    num_classes = len(y_train.unique())
                else:
                    import pandas as pd
                    num_classes = len(pd.Series(y_train).unique())
            else:
                num_classes = 1
            model = TabTransformer(
                continuous_features=len(continuous_cols),
                categorical_features=len(categorical_cols),
                categorical_cardinalities=categorical_cardinalities,
                embedding_dim=config["embedding_dim"],
                num_heads=config["num_heads"],
                num_layers=config["num_layers"],
                hidden_dim=config["hidden_dim"],
                dropout_rate=config["dropout_rate"],
                num_classes=num_classes
            )
            
            # Setup training
            criterion = nn.CrossEntropyLoss() if problem_type == "classification" else nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
            
            # Add learning rate scheduler
            if config["scheduler"] == "cosine":
                scheduler = AdvancedLearningRateScheduler.get_cosine_scheduler(
                    optimizer, config["epochs"] * len(X_continuous) // config["batch_size"], 
                    config["warmup_steps"]
                )
            else:
                scheduler = None
            
            # Training loop
            train_dataset = TensorDataset(X_continuous, X_categorical, y_tensor)
            train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
            
            val_dataset = TensorDataset(X_val_continuous, X_val_categorical, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
            
            # Log parameters
            self.mlflow_tracker.log_params(config)
            
            # Training
            start_time = time.time()
            best_val_loss = float('inf')
            
            for epoch in range(config["epochs"]):
                model.train()
                train_loss = 0.0
                
                for batch_cont, batch_cat, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = model(batch_cont, batch_cat)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch_cont, batch_cat, batch_y in val_loader:
                        outputs = model(batch_cont, batch_cat)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        if problem_type == "classification":
                            _, predicted = torch.max(outputs.data, 1)
                            total += batch_y.size(0)
                            correct += (predicted == batch_y).sum().item()
                
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = correct / total if problem_type == "classification" else 0.0
                
                # Log metrics
                metrics = {
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_accuracy,
                    "epoch": epoch
                }
                self.mlflow_tracker.log_metrics(metrics, step=epoch)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
            
            training_time = time.time() - start_time
            
            # Create explainer
            explainer = ModelExplainer(model, 'pytorch')
            
            result = {
                "model": model,
                "framework": "pytorch",
                "model_type": "tabtransformer",
                "training_time": training_time,
                "validation_metrics": {
                    "val_loss": best_val_loss,
                    "val_accuracy": val_accuracy,
                },
                "config": config,
                "explainer": explainer,
                "categorical_encoders": categorical_encoders,
                "continuous_cols": continuous_cols,
                "categorical_cols": categorical_cols,
            }
            
            # Log model
            self.mlflow_tracker.log_model(model, "pytorch", "tabtransformer")
            
            return result
            
        finally:
            self.mlflow_tracker.end_run()
    
    def create_vision_transformer(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        config: Optional[Dict[str, Any]] = None,
        problem_type: str = "classification"
    ) -> Dict[str, Any]:
        """Create and train Vision Transformer model."""
        
        config = config or self.default_configs["pytorch"]["vision_transformer"]
        
        # Start MLflow run
        run_name = f"vision_transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.mlflow_tracker.start_run(run_name)
        
        try:
            # Clean and prepare data
            X_train_clean = self._clean_numeric_data(X_train.values)
            X_val_clean = self._clean_numeric_data(X_val.values)
            y_train_clean = self._clean_target_data(y_train.values, problem_type)
            y_val_clean = self._clean_target_data(y_val.values, problem_type)
            
            # Convert to tensors
            X_train_tensor = torch.FloatTensor(X_train_clean)
            y_train_tensor = torch.LongTensor(y_train_clean) if problem_type == "classification" else torch.FloatTensor(y_train_clean)
            X_val_tensor = torch.FloatTensor(X_val_clean)
            y_val_tensor = torch.LongTensor(y_val_clean) if problem_type == "classification" else torch.FloatTensor(y_val_clean)
            
            # Create model
            if problem_type == "classification":
                if hasattr(y_train, 'unique'):
                    num_classes = len(y_train.unique())
                else:
                    import pandas as pd
                    num_classes = len(pd.Series(y_train).unique())
            else:
                num_classes = 1
            model = VisionTransformer(
                input_dim=X_train_clean.shape[1],
                patch_size=config["patch_size"],
                embedding_dim=config["embedding_dim"],
                num_heads=config["num_heads"],
                num_layers=config["num_layers"],
                hidden_dim=config["hidden_dim"],
                dropout_rate=config["dropout_rate"],
                num_classes=num_classes
            )
            
            # Setup training
            criterion = nn.CrossEntropyLoss() if problem_type == "classification" else nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
            
            # Add learning rate scheduler
            if config["scheduler"] == "cosine":
                scheduler = AdvancedLearningRateScheduler.get_cosine_scheduler(
                    optimizer, config["epochs"] * len(X_train_tensor) // config["batch_size"]
                )
            else:
                scheduler = None
            
            # Training loop
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
            
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
            
            # Log parameters
            self.mlflow_tracker.log_params(config)
            
            # Training
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
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    
                    train_loss += loss.item()
                
                # Validation
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                        
                        if problem_type == "classification":
                            _, predicted = torch.max(outputs.data, 1)
                            total += batch_y.size(0)
                            correct += (predicted == batch_y).sum().item()
                
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                val_accuracy = correct / total if problem_type == "classification" else 0.0
                
                # Log metrics
                metrics = {
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "val_accuracy": val_accuracy,
                    "epoch": epoch
                }
                self.mlflow_tracker.log_metrics(metrics, step=epoch)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
            
            training_time = time.time() - start_time
            
            # Create explainer
            explainer = ModelExplainer(model, 'pytorch')
            
            result = {
                "model": model,
                "framework": "pytorch",
                "model_type": "vision_transformer",
                "training_time": training_time,
                "validation_metrics": {
                    "val_loss": best_val_loss,
                    "val_accuracy": val_accuracy,
                },
                "config": config,
                "explainer": explainer,
            }
            
            # Log model
            self.mlflow_tracker.log_model(model, "pytorch", "vision_transformer")
            
            return result
            
        finally:
            self.mlflow_tracker.end_run()
    
    def _clean_numeric_data(self, data: np.ndarray) -> np.ndarray:
        """Clean numeric data removing non-numeric columns."""
        import pandas as pd
        
        if isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        # Identify numeric columns
        numeric_cols = []
        for col in df.columns:
            try:
                pd.to_numeric(df[col], errors='raise')
                numeric_cols.append(col)
            except (ValueError, TypeError):
                continue
        
        if not numeric_cols:
            raise ValueError("Nenhuma coluna numérica encontrada nos dados.")
        
        df_numeric = df[numeric_cols]
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


def get_advanced_deep_learning_automl() -> AdvancedDeepLearningAutoML:
    """Factory function for AdvancedDeepLearningAutoML."""
    return AdvancedDeepLearningAutoML()
