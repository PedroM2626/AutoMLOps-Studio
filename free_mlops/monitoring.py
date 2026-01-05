from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

from free_mlops.config import get_settings


class PerformanceMonitor:
    """Monitor de performance para modelos em produção."""
    
    def __init__(self, model_id: str, settings: Optional[Any] = None):
        self.model_id = model_id
        self.settings = settings or get_settings()
        self.metrics_file = self.settings.artifacts_dir / "monitoring" / f"{model_id}_metrics.json"
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        
    def log_prediction(
        self,
        input_data: Dict[str, Any],
        prediction: Any,
        ground_truth: Optional[Any] = None,
        latency_ms: Optional[float] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Registra uma predição com métricas de performance."""
        
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        
        record = {
            "timestamp": timestamp.isoformat(),
            "input_data": input_data,
            "prediction": prediction,
            "ground_truth": ground_truth,
            "latency_ms": latency_ms,
        }
        
        # Carregar métricas existentes
        metrics = self._load_metrics()
        
        # Adicionar novo registro
        if "predictions" not in metrics:
            metrics["predictions"] = []
        metrics["predictions"].append(record)
        
        # Manter apenas últimos 1000 registros para não sobrecarregar
        if len(metrics["predictions"]) > 1000:
            metrics["predictions"] = metrics["predictions"][-1000:]
        
        # Atualizar métricas agregadas
        metrics["summary"] = self._calculate_summary_metrics(metrics["predictions"])
        
        # Salvar métricas
        self._save_metrics(metrics)
    
    def _calculate_summary_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calcula métricas agregadas."""
        if not predictions:
            return {}
        
        # Separar predições com ground truth
        with_truth = [p for p in predictions if p.get("ground_truth") is not None]
        
        summary = {
            "total_predictions": len(predictions),
            "predictions_with_truth": len(with_truth),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        
        # Métricas de latência
        latencies = [p["latency_ms"] for p in predictions if p.get("latency_ms") is not None]
        if latencies:
            summary["latency"] = {
                "avg_ms": np.mean(latencies),
                "p50_ms": np.percentile(latencies, 50),
                "p95_ms": np.percentile(latencies, 95),
                "p99_ms": np.percentile(latencies, 99),
                "min_ms": np.min(latencies),
                "max_ms": np.max(latencies),
            }
        
        # Métricas de accuracy (se tiver ground truth)
        if with_truth:
            try:
                y_true = [p["ground_truth"] for p in with_truth]
                y_pred = [p["prediction"] for p in with_truth]
                
                # Calcular métricas de regressão sempre
                summary["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                summary["r2"] = float(r2_score(y_true, y_pred))
                
                # Tentar determinar se é classificação para accuracy
                # Considerar classificação se valores são 0/1 ou inteiros pequenos
                unique_true = set(y_true)
                unique_pred = set(y_pred)
                
                is_classification = (
                    all(isinstance(val, (int, np.integer)) for val in y_true + y_pred) and
                    len(unique_true) <= 10 and len(unique_pred) <= 10 and
                    all(val in range(-10, 11) for val in list(unique_true) + list(unique_pred))
                )
                
                if is_classification:
                    summary["accuracy"] = float(accuracy_score(y_true, y_pred))
                    
            except Exception as e:
                # Em caso de erro, tentar apenas regressão
                try:
                    y_true = [p["ground_truth"] for p in with_truth]
                    y_pred = [p["prediction"] for p in with_truth]
                    summary["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
                    summary["r2"] = float(r2_score(y_true, y_pred))
                except Exception:
                    pass
        
        # Throughput (predições por hora)
        if len(predictions) > 1:
            timestamps = [datetime.fromisoformat(p["timestamp"]) for p in predictions]
            time_span = (timestamps[-1] - timestamps[0]).total_seconds()
            if time_span > 0:
                summary["throughput_per_hour"] = len(predictions) / (time_span / 3600)
        
        return summary
    
    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas atuais calculadas."""
        metrics = self._load_metrics()
        predictions = metrics.get("predictions", [])
        
        if not predictions:
            return {"predictions": [], "summary": {}}
        
        # Calcular métricas atualizadas
        return self._calculate_summary_metrics(predictions)
    
    def get_recent_predictions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Retorna predições recentes."""
        metrics = self._load_metrics()
        predictions = metrics.get("predictions", [])
        return predictions[-limit:] if predictions else []
    
    def _load_metrics(self) -> Dict[str, Any]:
        """Carrega métricas do arquivo."""
        if self.metrics_file.exists():
            try:
                return json.loads(self.metrics_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"predictions": [], "summary": {}}
    
    def _save_metrics(self, metrics: Dict[str, Any]) -> None:
        """Salva métricas no arquivo."""
        self.metrics_file.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )


class AlertManager:
    """Gerenciador de alertas para monitoramento."""
    
    def __init__(self, settings: Optional[Any] = None):
        self.settings = settings or get_settings()
        self.alerts_file = self.settings.artifacts_dir / "monitoring" / "alerts.json"
        self.alerts_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Thresholds padrão
        self.thresholds = {
            "accuracy_min": 0.8,
            "rmse_max": 1.0,
            "latency_p95_max_ms": 1000,
            "throughput_min_per_hour": 100,
        }
    
    def check_alerts(self, model_id: str, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Verifica se há alertas baseados nas métricas."""
        alerts = []
        summary = metrics.get("summary", {})
        
        # Alerta de accuracy
        if "accuracy" in summary and summary["accuracy"] < self.thresholds["accuracy_min"]:
            alerts.append({
                "model_id": model_id,
                "type": "accuracy_low",
                "severity": "warning",
                "message": f"Accuracy ({summary['accuracy']:.3f}) abaixo do threshold ({self.thresholds['accuracy_min']})",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "value": summary["accuracy"],
                "threshold": self.thresholds["accuracy_min"],
            })
        
        # Alerta de RMSE
        if "rmse" in summary and summary["rmse"] > self.thresholds["rmse_max"]:
            alerts.append({
                "model_id": model_id,
                "type": "rmse_high",
                "severity": "warning",
                "message": f"RMSE ({summary['rmse']:.3f}) acima do threshold ({self.thresholds['rmse_max']})",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "value": summary["rmse"],
                "threshold": self.thresholds["rmse_max"],
            })
        
        # Alerta de latência
        if "latency" in summary and summary["latency"]["p95_ms"] > self.thresholds["latency_p95_max_ms"]:
            alerts.append({
                "model_id": model_id,
                "type": "latency_high",
                "severity": "warning",
                "message": f"P95 Latency ({summary['latency']['p95_ms']:.1f}ms) acima do threshold ({self.thresholds['latency_p95_max_ms']}ms)",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "value": summary["latency"]["p95_ms"],
                "threshold": self.thresholds["latency_p95_max_ms"],
            })
        
        # Alerta de throughput
        if "throughput_per_hour" in summary and summary["throughput_per_hour"] < self.thresholds["throughput_min_per_hour"]:
            alerts.append({
                "model_id": model_id,
                "type": "throughput_low",
                "severity": "info",
                "message": f"Throughput ({summary['throughput_per_hour']:.1f}/h) abaixo do threshold ({self.thresholds['throughput_min_per_hour']}/h)",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "value": summary["throughput_per_hour"],
                "threshold": self.thresholds["throughput_min_per_hour"],
            })
        
        # Salvar alertas
        if alerts:
            self._save_alerts(alerts)
        
        return alerts
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retorna alertas recentes."""
        if self.alerts_file.exists():
            try:
                all_alerts = json.loads(self.alerts_file.read_text(encoding="utf-8"))
                return all_alerts[-limit:] if all_alerts else []
            except Exception:
                pass
        return []
    
    def _save_alerts(self, new_alerts: List[Dict[str, Any]]) -> None:
        """Salva alertas no arquivo."""
        existing_alerts = []
        if self.alerts_file.exists():
            try:
                existing_alerts = json.loads(self.alerts_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        
        # Adicionar novos alertas
        existing_alerts.extend(new_alerts)
        
        # Manter apenas últimos 500 alertas
        if len(existing_alerts) > 500:
            existing_alerts = existing_alerts[-500:]
        
        self.alerts_file.write_text(
            json.dumps(existing_alerts, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )


def get_performance_monitor(model_id: str) -> PerformanceMonitor:
    """Factory function para PerformanceMonitor."""
    return PerformanceMonitor(model_id)


def get_alert_manager() -> AlertManager:
    """Factory function para AlertManager."""
    return AlertManager()
