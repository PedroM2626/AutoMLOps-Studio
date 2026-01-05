from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from free_mlops.config import get_settings


class TimeSeriesAutoML:
    """AutoML para Time Series com suporte a ARIMA, Prophet e LSTM."""
    
    def __init__(self, settings: Optional[Any] = None):
        self.settings = settings or get_settings()
        self.models_dir = self.settings.artifacts_dir / "time_series"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Configurações padrão
        self.default_configs = {
            "arima": {
                "order": (1, 1, 1),
                "seasonal_order": (0, 0, 0, 0),
                "auto_arima": True,
                "max_p": 5,
                "max_d": 2,
                "max_q": 5,
                "max_P": 2,
                "max_D": 1,
                "max_Q": 2,
                "max_order": 5,
                "information_criterion": "aic",
                "alpha": 0.05,
            },
            "prophet": {
                "yearly_seasonality": "auto",
                "weekly_seasonality": "auto",
                "daily_seasonality": "auto",
                "seasonality_mode": "additive",
                "changepoint_prior_scale": 0.05,
                "seasonality_prior_scale": 10.0,
                "holidays_prior_scale": 10.0,
                "mcmc_samples": 0,
                "interval_width": 0.8,
                "uncertainty_samples": 1000,
            },
            "lstm": {
                "sequence_length": 10,
                "hidden_layers": [50, 50],
                "dropout_rate": 0.2,
                "batch_size": 32,
                "epochs": 100,
                "learning_rate": 0.001,
                "optimizer": "adam",
                "early_stopping_patience": 10,
            }
        }
    
    def create_arima_model(
        self,
        data: pd.Series,
        config: Optional[Dict[str, Any]] = None,
        test_size: int = 30,
    ) -> Dict[str, Any]:
        """Cria e treina modelo ARIMA."""
        
        try:
            from statsmodels.tsa.arima.model import ARIMA
            from statsmodels.tsa.stattools import adfuller
            from statsmodels.tsa.seasonal import seasonal_decompose
        except ImportError:
            raise ImportError("Statsmodels não está instalado. Instale com: pip install statsmodels")
        
        config = config or self.default_configs["arima"]
        
        # Preparar dados
        train_data = data[:-test_size] if test_size > 0 else data
        test_data = data[-test_size:] if test_size > 0 else None
        
        start_time = time.time()
        
        # Verificar estacionariedade
        adf_result = adfuller(train_data.dropna())
        is_stationary = adf_result[1] < 0.05
        
        # Decomposição sazonal (se dados suficientes)
        decomposition = None
        if len(train_data) >= 24:  # Mínimo para decomposição
            try:
                decomposition = seasonal_decompose(train_data.dropna(), model='additive', period=12)
            except Exception:
                pass
        
        # Auto ARIMA se habilitado
        if config.get("auto_arima", True):
            try:
                import pmdarima as pm
                
                auto_model = pm.auto_arima(
                    train_data.dropna(),
                    max_p=config["max_p"],
                    max_d=config["max_d"],
                    max_q=config["max_q"],
                    max_P=config["max_P"],
                    max_D=config["max_D"],
                    max_Q=config["max_Q"],
                    max_order=config["max_order"],
                    information_criterion=config["information_criterion"],
                    alpha=config["alpha"],
                    seasonal=True,
                    stepwise=True,
                    suppress_warnings=True,
                    error_action='ignore',
                    trace=False
                )
                
                order = auto_model.order
                seasonal_order = auto_model.seasonal_order
                
            except ImportError:
                # Fallback para ARIMA manual se pmdarima não estiver disponível
                order = config["order"]
                seasonal_order = config["seasonal_order"]
        else:
            order = config["order"]
            seasonal_order = config["seasonal_order"]
        
        # Criar e treinar modelo
        model = ARIMA(train_data.dropna(), order=order, seasonal_order=seasonal_order)
        fitted_model = model.fit()
        
        # Previsões
        if test_size > 0 and test_data is not None:
            forecast = fitted_model.forecast(steps=test_size)
            forecast_ci = fitted_model.get_forecast(steps=test_size).conf_int(alpha=config["alpha"])
        else:
            forecast = None
            forecast_ci = None
        
        # Métricas
        metrics = {}
        if test_size > 0 and test_data is not None:
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            
            actual = test_data.values
            predicted = forecast.values
            
            metrics["mae"] = float(mean_absolute_error(actual, predicted))
            metrics["rmse"] = float(np.sqrt(mean_squared_error(actual, predicted)))
            metrics["mape"] = float(np.mean(np.abs((actual - predicted) / actual))) * 100
        
        # Diagnóstico do modelo
        diagnostics = {
            "aic": float(fitted_model.aic),
            "bic": float(fitted_model.bic),
            "log_likelihood": float(fitted_model.llf),
        }
        
        training_time = time.time() - start_time
        
        result = {
            "model_type": "arima",
            "model": fitted_model,
            "config": config,
            "order": order,
            "seasonal_order": seasonal_order,
            "training_data": train_data,
            "test_data": test_data,
            "forecast": forecast,
            "forecast_ci": forecast_ci,
            "metrics": metrics,
            "diagnostics": diagnostics,
            "is_stationary": is_stationary,
            "decomposition": decomposition,
            "training_time": training_time,
            "fitted_at": datetime.now(timezone.utc).isoformat(),
        }
        
        return result
    
    def create_prophet_model(
        self,
        data: pd.DataFrame,
        date_column: str,
        value_column: str,
        config: Optional[Dict[str, Any]] = None,
        test_size: int = 30,
    ) -> Dict[str, Any]:
        """Cria e treina modelo Prophet."""
        
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError("Prophet não está instalado. Instale com: pip install prophet")
        
        config = config or self.default_configs["prophet"]
        
        # Preparar dados
        df = data[[date_column, value_column]].copy()
        df.columns = ["ds", "y"]
        df["ds"] = pd.to_datetime(df["ds"])
        
        # Split treino/teste
        if test_size > 0:
            train_df = df[:-test_size]
            test_df = df[-test_size:]
        else:
            train_df = df
            test_df = None
        
        start_time = time.time()
        
        # Criar modelo
        model = Prophet(
            yearly_seasonality=config["yearly_seasonality"],
            weekly_seasonality=config["weekly_seasonality"],
            daily_seasonality=config["daily_seasonality"],
            seasonality_mode=config["seasonality_mode"],
            changepoint_prior_scale=config["changepoint_prior_scale"],
            seasonality_prior_scale=config["seasonality_prior_scale"],
            holidays_prior_scale=config["holidays_prior_scale"],
            mcmc_samples=config["mcmc_samples"],
            interval_width=config["interval_width"],
            uncertainty_samples=config["uncertainty_samples"],
        )
        
        # Treinar
        model.fit(train_df)
        
        # Criar dataframe para previsões
        if test_size > 0 and test_df is not None:
            future = model.make_future_dataframe(periods=test_size, freq="D")
            forecast = model.predict(future)
            
            # Separar previsões do teste
            test_forecast = forecast.iloc[-test_size:]
            train_forecast = forecast.iloc[:-test_size]
        else:
            # Prever apenas períodos futuros
            future = model.make_future_dataframe(periods=30, freq="D")
            forecast = model.predict(future)
            test_forecast = None
            train_forecast = forecast
        
        # Métricas
        metrics = {}
        if test_size > 0 and test_df is not None:
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            
            actual = test_df["y"].values
            predicted = test_forecast["yhat"].values
            
            metrics["mae"] = float(mean_absolute_error(actual, predicted))
            metrics["rmse"] = float(np.sqrt(mean_squared_error(actual, predicted)))
            metrics["mape"] = float(np.mean(np.abs((actual - predicted) / actual))) * 100
        
        # Componentes do modelo
        components = {
            "trend": model.params["trend"],
            "seasonalities": list(model.seasonalities.keys()),
            "changepoints": len(model.changepoints),
        }
        
        training_time = time.time() - start_time
        
        result = {
            "model_type": "prophet",
            "model": model,
            "config": config,
            "training_data": train_df,
            "test_data": test_df,
            "forecast": forecast,
            "test_forecast": test_forecast,
            "train_forecast": train_forecast,
            "metrics": metrics,
            "components": components,
            "training_time": training_time,
            "fitted_at": datetime.now(timezone.utc).isoformat(),
        }
        
        return result
    
    def create_lstm_model(
        self,
        data: pd.Series,
        config: Optional[Dict[str, Any]] = None,
        test_size: int = 30,
    ) -> Dict[str, Any]:
        """Cria e treina modelo LSTM para time series."""
        
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, models, optimizers, callbacks
        except ImportError:
            raise ImportError("TensorFlow não está instalado. Instale com: pip install tensorflow")
        
        config = config or self.default_configs["lstm"]
        
        # Preparar dados
        train_data = data[:-test_size] if test_size > 0 else data
        test_data = data[-test_size:] if test_size > 0 else None
        
        # Normalização
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(train_data.values.reshape(-1, 1))
        
        # Criar sequências
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)
        
        X_train, y_train = create_sequences(scaled_train, config["sequence_length"])
        
        start_time = time.time()
        
        # Criar modelo LSTM
        model = models.Sequential()
        
        # LSTM layers
        for i, units in enumerate(config["hidden_layers"]):
            return_sequences = i < len(config["hidden_layers"]) - 1
            model.add(layers.LSTM(
                units,
                return_sequences=return_sequences,
                input_shape=(config["sequence_length"], 1) if i == 0 else None
            ))
            if config["dropout_rate"] > 0:
                model.add(layers.Dropout(config["dropout_rate"]))
        
        # Output layer
        model.add(layers.Dense(1))
        
        # Compilar
        optimizer_config = config["optimizer"]
        if optimizer_config == "adam":
            optimizer = optimizers.Adam(learning_rate=config["learning_rate"])
        else:
            optimizer = optimizers.Adam(learning_rate=config["learning_rate"])
        
        model.compile(
            optimizer=optimizer,
            loss="mse",
            metrics=["mae"]
        )
        
        # Callbacks
        callback_list = [
            callbacks.EarlyStopping(
                patience=config["early_stopping_patience"],
                restore_best_weights=True
            ),
            callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        ]
        
        # Treinar
        history = model.fit(
            X_train, y_train,
            batch_size=config["batch_size"],
            epochs=config["epochs"],
            callbacks=callback_list,
            verbose=0,
        )
        
        # Previsões
        if test_size > 0 and test_data is not None:
            # Preparar dados de teste
            last_sequence = scaled_train[-config["sequence_length"]:]
            test_predictions = []
            
            current_sequence = last_sequence.copy()
            for _ in range(test_size):
                next_pred = model.predict(current_sequence.reshape(1, config["sequence_length"], 1), verbose=0)
                test_predictions.append(next_pred[0, 0])
                current_sequence = np.append(current_sequence[1:], next_pred[0, 0]).reshape(-1, 1)
            
            # Desnormalizar
            test_predictions = scaler.inverse_transform(np.array(test_predictions).reshape(-1, 1)).flatten()
        else:
            test_predictions = None
        
        # Métricas
        metrics = {}
        if test_size > 0 and test_data is not None:
            from sklearn.metrics import mean_absolute_error, mean_squared_error
            
            actual = test_data.values
            predicted = test_predictions
            
            metrics["mae"] = float(mean_absolute_error(actual, predicted))
            metrics["rmse"] = float(np.sqrt(mean_squared_error(actual, predicted)))
            metrics["mape"] = float(np.mean(np.abs((actual - predicted) / actual))) * 100
        
        training_time = time.time() - start_time
        
        result = {
            "model_type": "lstm",
            "model": model,
            "scaler": scaler,
            "config": config,
            "training_data": train_data,
            "test_data": test_data,
            "test_predictions": test_predictions,
            "metrics": metrics,
            "history": {
                "loss": history.history["loss"],
                "mae": history.history["mae"],
            },
            "training_time": training_time,
            "fitted_at": datetime.now(timezone.utc).isoformat(),
        }
        
        return result
    
    def save_model(self, result: Dict[str, Any], model_name: str) -> str:
        """Salva modelo treinado."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = self.models_dir / f"{model_name}_{timestamp}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_type = result["model_type"]
        
        if model_type == "arima":
            # Salvar modelo ARIMA
            import pickle
            with open(model_dir / "model.pkl", "wb") as f:
                pickle.dump(result["model"], f)
        
        elif model_type == "prophet":
            # Salvar modelo Prophet
            result["model"].save(model_dir / "prophet_model")
        
        elif model_type == "lstm":
            # Salvar modelo LSTM
            import tensorflow as tf
            result["model"].save(model_dir / "lstm_model")
            
            # Salvar scaler
            import pickle
            with open(model_dir / "scaler.pkl", "wb") as f:
                pickle.dump(result["scaler"], f)
        
        # Salvar metadados
        metadata = {
            "model_type": model_type,
            "config": result["config"],
            "metrics": result["metrics"],
            "training_time": result["training_time"],
            "fitted_at": result["fitted_at"],
            "model_name": model_name,
        }
        
        # Adicionar informações específicas do modelo
        if model_type == "arima":
            metadata.update({
                "order": result["order"],
                "seasonal_order": result["seasonal_order"],
                "diagnostics": result["diagnostics"],
            })
        elif model_type == "prophet":
            metadata.update({
                "components": result["components"],
            })
        elif model_type == "lstm":
            metadata.update({
                "sequence_length": result["config"]["sequence_length"],
            })
        
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)
        
        return str(model_dir)
    
    def load_model(self, model_path: str) -> Dict[str, Any]:
        """Carrega modelo salvo."""
        
        model_dir = Path(model_path)
        
        # Carregar metadados
        with open(model_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        model_type = metadata["model_type"]
        
        if model_type == "arima":
            import pickle
            with open(model_dir / "model.pkl", "rb") as f:
                model = pickle.load(f)
        
        elif model_type == "prophet":
            from prophet import Prophet
            model = Prophet.load(model_dir / "prophet_model")
        
        elif model_type == "lstm":
            import tensorflow as tf
            model = tf.keras.models.load_model(model_dir / "lstm_model")
            
            import pickle
            with open(model_dir / "scaler.pkl", "rb") as f:
                scaler = pickle.load(f)
            
            metadata["scaler"] = scaler
        
        return {
            "model": model,
            "metadata": metadata,
        }
    
    def forecast(
        self,
        model_path: str,
        periods: int = 30,
        frequency: str = "D",
        confidence_interval: bool = True,
    ) -> Dict[str, Any]:
        """Realiza previsões com modelo salvo."""
        
        loaded = self.load_model(model_path)
        model = loaded["model"]
        metadata = loaded["metadata"]
        model_type = metadata["model_type"]
        
        try:
            if model_type == "arima":
                forecast = model.forecast(steps=periods)
                if confidence_interval:
                    forecast_ci = model.get_forecast(steps=periods).conf_int()
                else:
                    forecast_ci = None
                
                result = {
                    "success": True,
                    "forecast": forecast.tolist(),
                    "forecast_ci": forecast_ci.values.tolist() if forecast_ci is not None else None,
                    "periods": periods,
                    "frequency": frequency,
                }
            
            elif model_type == "prophet":
                # Criar dataframe futuro
                from datetime import datetime, timedelta
                last_date = model.history["ds"].max()
                future_dates = [last_date + timedelta(days=i+1) for i in range(periods)]
                future_df = pd.DataFrame({"ds": future_dates})
                
                forecast = model.predict(future_df)
                
                result = {
                    "success": True,
                    "forecast": forecast["yhat"].tolist(),
                    "forecast_lower": forecast["yhat_lower"].tolist(),
                    "forecast_upper": forecast["yhat_upper"].tolist(),
                    "dates": forecast["ds"].dt.strftime("%Y-%m-%d").tolist(),
                    "periods": periods,
                    "frequency": frequency,
                }
            
            elif model_type == "lstm":
                # Para LSTM, precisaríamos dos dados históricos para criar sequências
                # Simplificado - retornamos erro indicando necessidade de dados
                result = {
                    "success": False,
                    "error": "LSTM forecasting requires historical data. Use predict_with_data method.",
                    "periods": periods,
                }
            
            else:
                result = {
                    "success": False,
                    "error": f"Model type {model_type} not supported for forecasting",
                }
            
            result["metadata"] = metadata
            
        except Exception as e:
            result = {
                "success": False,
                "error": str(e),
                "metadata": metadata,
            }
        
        return result
    
    def predict_with_data(
        self,
        model_path: str,
        data: pd.Series,
        periods: int = 30,
    ) -> Dict[str, Any]:
        """Realiza previsões usando dados históricos (principalmente para LSTM)."""
        
        loaded = self.load_model(model_path)
        model = loaded["model"]
        metadata = loaded["metadata"]
        model_type = metadata["model_type"]
        
        try:
            if model_type == "lstm":
                scaler = metadata["scaler"]
                sequence_length = metadata["sequence_length"]
                
                # Preparar dados
                scaled_data = scaler.transform(data.values.reshape(-1, 1))
                
                # Criar previsões
                predictions = []
                current_sequence = scaled_data[-sequence_length:].copy()
                
                for _ in range(periods):
                    next_pred = model.predict(current_sequence.reshape(1, sequence_length, 1), verbose=0)
                    predictions.append(next_pred[0, 0])
                    current_sequence = np.append(current_sequence[1:], next_pred[0, 0]).reshape(-1, 1)
                
                # Desnormalizar
                predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
                
                result = {
                    "success": True,
                    "forecast": predictions.tolist(),
                    "periods": periods,
                    "metadata": metadata,
                }
            
            else:
                # Para outros modelos, usar método forecast padrão
                result = self.forecast(model_path, periods)
                result["success"] = True
            
        except Exception as e:
            result = {
                "success": False,
                "error": str(e),
                "metadata": metadata,
            }
        
        return result
    
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
                            "model_type": metadata.get("model_type", "unknown"),
                            "fitted_at": metadata.get("fitted_at", ""),
                            "training_time": metadata.get("training_time", 0),
                            "metrics": metadata.get("metrics", {}),
                            "path": str(model_dir),
                        })
                    except Exception:
                        continue
        
        return sorted(models, key=lambda x: x["fitted_at"], reverse=True)
    
    def evaluate_model(
        self,
        model_path: str,
        test_data: pd.Series,
    ) -> Dict[str, Any]:
        """Avalia modelo com dados de teste."""
        
        loaded = self.load_model(model_path)
        model = loaded["model"]
        metadata = loaded["metadata"]
        model_type = metadata["model_type"]
        
        try:
            if model_type == "arima":
                # Usar dados de teste para previsão
                forecast = model.forecast(steps=len(test_data))
                
                from sklearn.metrics import mean_absolute_error, mean_squared_error
                
                actual = test_data.values
                predicted = forecast.values
                
                metrics = {
                    "mae": float(mean_absolute_error(actual, predicted)),
                    "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
                    "mape": float(np.mean(np.abs((actual - predicted) / actual))) * 100,
                }
            
            elif model_type == "prophet":
                # Para Prophet, criar dataframe com datas de teste
                from datetime import datetime, timedelta
                
                # Assumir datas diárias a partir da última data do treino
                last_date = model.history["ds"].max()
                test_dates = [last_date + timedelta(days=i+1) for i in range(len(test_data))]
                test_df = pd.DataFrame({"ds": test_dates, "y": test_data.values})
                
                forecast = model.predict(test_df)
                
                from sklearn.metrics import mean_absolute_error, mean_squared_error
                
                actual = test_df["y"].values
                predicted = forecast["yhat"].values
                
                metrics = {
                    "mae": float(mean_absolute_error(actual, predicted)),
                    "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
                    "mape": float(np.mean(np.abs((actual - predicted) / actual))) * 100,
                }
            
            elif model_type == "lstm":
                # Para LSTM, usar método predict_with_data
                result = self.predict_with_data(model_path, test_data, len(test_data))
                
                if result["success"]:
                    from sklearn.metrics import mean_absolute_error, mean_squared_error
                    
                    actual = test_data.values
                    predicted = result["forecast"]
                    
                    metrics = {
                        "mae": float(mean_absolute_error(actual, predicted)),
                        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
                        "mape": float(np.mean(np.abs((actual - predicted) / actual))) * 100,
                    }
                else:
                    metrics = {"error": result["error"]}
            
            else:
                metrics = {"error": f"Model type {model_type} not supported for evaluation"}
            
            return {
                "success": True,
                "metrics": metrics,
                "model_type": model_type,
                "metadata": metadata,
            }
        
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model_type": model_type,
                "metadata": metadata,
            }


def get_time_series_automl() -> TimeSeriesAutoML:
    """Factory function para TimeSeriesAutoML."""
    return TimeSeriesAutoML()
