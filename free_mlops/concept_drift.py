from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error

from free_mlops.config import get_settings


class ConceptDriftDetector:
    """Detector de drift no comportamento/conceito do modelo."""
    
    def __init__(self, model_id: str, settings: Optional[Any] = None):
        self.model_id = model_id
        self.settings = settings or get_settings()
        self.concept_drift_file = self.settings.artifacts_dir / "monitoring" / f"{model_id}_concept_drift.json"
        self.concept_drift_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Thresholds para detecção de concept drift
        self.thresholds = {
            "performance_degradation_threshold": 0.1,  # 10% de degradação
            "error_increase_threshold": 0.2,           # 20% de aumento no erro
            "prediction_distribution_threshold": 0.15, # Mudança na distribuição de predições
            "min_samples_for_detection": 50,          # Mínimo de amostras para detecção
        }
    
    def save_baseline_performance(self, baseline_metrics: Dict[str, float]) -> None:
        """Salva métricas de performance baseline (treino/validação)."""
        drift_data = self._load_drift_data()
        
        drift_data["baseline"] = {
            "metrics": baseline_metrics,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        
        self._save_drift_data(drift_data)
    
    def detect_concept_drift(
        self,
        recent_predictions: List[Dict[str, Any]],
        window_size: int = 100
    ) -> Dict[str, Any]:
        """Detecta concept drift baseado em predições recentes."""
        
        if len(recent_predictions) < self.thresholds["min_samples_for_detection"]:
            return {
                "model_id": self.model_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "has_concept_drift": False,
                "reason": "Insufficient samples for detection",
                "sample_count": len(recent_predictions),
            }
        
        drift_data = self._load_drift_data()
        
        if "baseline" not in drift_data:
            raise ValueError("Nenhum baseline salvo. Use save_baseline_performance() primeiro.")
        
        baseline_metrics = drift_data["baseline"]["metrics"]
        
        # Calcular métricas atuais
        current_metrics = self._calculate_current_metrics(recent_predictions)
        
        # Detectar drift
        drift_result = self._analyze_performance_drift(baseline_metrics, current_metrics)
        
        # Analisar mudança na distribuição de predições
        distribution_drift = self._analyze_prediction_distribution_drift(recent_predictions)
        
        # Combinar resultados
        overall_drift = (
            drift_result["has_performance_drift"] or
            distribution_drift["has_distribution_drift"]
        )
        
        result = {
            "model_id": self.model_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "has_concept_drift": overall_drift,
            "sample_count": len(recent_predictions),
            "baseline_metrics": baseline_metrics,
            "current_metrics": current_metrics,
            "performance_drift": drift_result,
            "distribution_drift": distribution_drift,
            "drift_score": max(
                drift_result["drift_score"],
                distribution_drift["drift_score"]
            ),
        }
        
        # Salvar resultado
        if "drift_history" not in drift_data:
            drift_data["drift_history"] = []
        drift_data["drift_history"].append(result)
        
        # Manter apenas últimos 50 registros
        if len(drift_data["drift_history"]) > 50:
            drift_data["drift_history"] = drift_data["drift_history"][-50:]
        
        self._save_drift_data(drift_data)
        
        return result
    
    def _calculate_current_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calcula métricas de performance atuais."""
        
        # Filtrar predições com ground truth
        with_truth = [p for p in predictions if p.get("ground_truth") is not None]
        
        if not with_truth:
            return {}
        
        y_true = [p["ground_truth"] for p in with_truth]
        y_pred = [p["prediction"] for p in with_truth]
        
        metrics = {}
        
        try:
            # Tentar determinar se é classificação ou regressão
            if all(isinstance(val, (int, float)) and val == int(val) for val in y_true + y_pred):
                # Classificação
                metrics["accuracy"] = accuracy_score(y_true, y_pred)
                
                # Calcular error rate (1 - accuracy)
                metrics["error_rate"] = 1.0 - metrics["accuracy"]
            else:
                # Regressão
                mse = np.mean([(yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)])
                metrics["rmse"] = np.sqrt(mse)
                metrics["error_rate"] = metrics["rmse"]  # Usar RMSE como proxy de erro
        except Exception:
            # Em caso de erro, usar error rate simples
            errors = [1 if yt != yp else 0 for yt, yp in zip(y_true, y_pred)]
            metrics["error_rate"] = np.mean(errors)
        
        return metrics
    
    def _analyze_performance_drift(
        self,
        baseline_metrics: Dict[str, float],
        current_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """Analisa drift na performance do modelo."""
        
        drift_result = {
            "has_performance_drift": False,
            "drift_score": 0,
            "analysis": {},
        }
        
        if not current_metrics:
            return drift_result
        
        # Analisar degradação de accuracy (se aplicável)
        if "accuracy" in baseline_metrics and "accuracy" in current_metrics:
            baseline_acc = baseline_metrics["accuracy"]
            current_acc = current_metrics["accuracy"]
            
            acc_degradation = (baseline_acc - current_acc) / baseline_acc
            
            drift_result["analysis"]["accuracy"] = {
                "baseline": baseline_acc,
                "current": current_acc,
                "degradation": acc_degradation,
                "drift_detected": acc_degradation > self.thresholds["performance_degradation_threshold"],
            }
            
            if acc_degradation > self.thresholds["performance_degradation_threshold"]:
                drift_result["has_performance_drift"] = True
                drift_result["drift_score"] += 0.4
        
        # Analisar aumento de error rate
        if "error_rate" in baseline_metrics and "error_rate" in current_metrics:
            baseline_error = baseline_metrics["error_rate"]
            current_error = current_metrics["error_rate"]
            
            if baseline_error > 0:
                error_increase = (current_error - baseline_error) / baseline_error
            else:
                error_increase = current_error  # Se baseline era 0, usar valor absoluto
            
            drift_result["analysis"]["error_rate"] = {
                "baseline": baseline_error,
                "current": current_error,
                "increase": error_increase,
                "drift_detected": error_increase > self.thresholds["error_increase_threshold"],
            }
            
            if error_increase > self.thresholds["error_increase_threshold"]:
                drift_result["has_performance_drift"] = True
                drift_result["drift_score"] += 0.4
        
        # Analisar RMSE (se aplicável)
        if "rmse" in baseline_metrics and "rmse" in current_metrics:
            baseline_rmse = baseline_metrics["rmse"]
            current_rmse = current_metrics["rmse"]
            
            if baseline_rmse > 0:
                rmse_increase = (current_rmse - baseline_rmse) / baseline_rmse
            else:
                rmse_increase = current_rmse
            
            drift_result["analysis"]["rmse"] = {
                "baseline": baseline_rmse,
                "current": current_rmse,
                "increase": rmse_increase,
                "drift_detected": rmse_increase > self.thresholds["error_increase_threshold"],
            }
            
            if rmse_increase > self.thresholds["error_increase_threshold"]:
                drift_result["has_performance_drift"] = True
                drift_result["drift_score"] += 0.4
        
        # Limitar drift score a 1.0
        drift_result["drift_score"] = min(drift_result["drift_score"], 1.0)
        
        return drift_result
    
    def _analyze_prediction_distribution_drift(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analisa mudanças na distribuição de predições."""
        
        all_predictions = [p["prediction"] for p in predictions]
        
        # Calcular distribuição atual
        if all(isinstance(p, (int, float)) for p in all_predictions):
            # Numérico
            pred_array = np.array(all_predictions)
            current_stats = {
                "mean": float(np.mean(pred_array)),
                "std": float(np.std(pred_array)),
                "min": float(np.min(pred_array)),
                "max": float(np.max(pred_array)),
                "median": float(np.median(pred_array)),
            }
            
            # Comparar com baseline se existir
            drift_data = self._load_drift_data()
            baseline_stats = drift_data.get("prediction_baseline", {})
            
            drift_score = 0
            has_drift = False
            
            if baseline_stats:
                # Comparar médias
                mean_diff = abs(current_stats["mean"] - baseline_stats["mean"])
                mean_ratio = mean_diff / (abs(baseline_stats["mean"]) + 1e-8)
                
                if mean_ratio > 0.1:
                    has_drift = True
                    drift_score += 0.3
                
                # Comparar desvio padrão
                std_diff = abs(current_stats["std"] - baseline_stats["std"])
                std_ratio = std_diff / (baseline_stats["std"] + 1e-8)
                
                if std_ratio > 0.2:
                    has_drift = True
                    drift_score += 0.3
            
            # Salvar baseline se não existir
            if not baseline_stats:
                drift_data["prediction_baseline"] = current_stats
                drift_data["prediction_baseline_saved_at"] = datetime.now(timezone.utc).isoformat()
                self._save_drift_data(drift_data)
            
            return {
                "has_distribution_drift": has_drift,
                "drift_score": min(drift_score, 1.0),
                "current_stats": current_stats,
                "baseline_stats": baseline_stats,
                "type": "numeric",
            }
        
        else:
            # Categórico
            pred_series = pd.Series(all_predictions)
            value_counts = pred_series.value_counts().to_dict()
            
            # Calcular distribuição normalizada
            total = len(all_predictions)
            current_dist = {k: v/total for k, v in value_counts.items()}
            
            # Comparar com baseline
            drift_data = self._load_drift_data()
            baseline_dist = drift_data.get("prediction_baseline", {})
            
            drift_score = 0
            has_drift = False
            
            if baseline_dist:
                # Calcular mudança na distribuição
                all_categories = set(current_dist.keys()) | set(baseline_dist.keys())
                
                current_probs = np.array([current_dist.get(cat, 0) for cat in all_categories])
                baseline_probs = np.array([baseline_dist.get(cat, 0) for cat in all_categories])
                
                # Distância L1 entre distribuições
                l1_distance = np.sum(np.abs(current_probs - baseline_probs))
                
                if l1_distance > self.thresholds["prediction_distribution_threshold"]:
                    has_drift = True
                    drift_score = min(l1_distance / self.thresholds["prediction_distribution_threshold"], 1.0)
                
                # Verificar novas categorias
                new_categories = set(current_dist.keys()) - set(baseline_dist.keys())
                missing_categories = set(baseline_dist.keys()) - set(current_dist.keys())
                
                if new_categories or missing_categories:
                    has_drift = True
                    drift_score = max(drift_score, 0.3)
            
            # Salvar baseline se não existir
            if not baseline_dist:
                drift_data["prediction_baseline"] = current_dist
                drift_data["prediction_baseline_saved_at"] = datetime.now(timezone.utc).isoformat()
                self._save_drift_data(drift_data)
            
            return {
                "has_distribution_drift": has_drift,
                "drift_score": drift_score,
                "current_distribution": current_dist,
                "baseline_distribution": baseline_dist,
                "new_categories": list(set(current_dist.keys()) - set(baseline_dist.keys())) if baseline_dist else [],
                "missing_categories": list(set(baseline_dist.keys()) - set(current_dist.keys())) if baseline_dist else [],
                "type": "categorical",
            }
    
    def get_concept_drift_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retorna histórico de detecção de concept drift."""
        drift_data = self._load_drift_data()
        history = drift_data.get("drift_history", [])
        return history[-limit:] if history else []
    
    def _load_drift_data(self) -> Dict[str, Any]:
        """Carrega dados de concept drift do arquivo."""
        if self.concept_drift_file.exists():
            try:
                return json.loads(self.concept_drift_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}
    
    def _save_drift_data(self, drift_data: Dict[str, Any]) -> None:
        """Salva dados de concept drift no arquivo."""
        self.concept_drift_file.write_text(
            json.dumps(drift_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )


def get_concept_drift_detector(model_id: str) -> ConceptDriftDetector:
    """Factory function para ConceptDriftDetector."""
    return ConceptDriftDetector(model_id)
