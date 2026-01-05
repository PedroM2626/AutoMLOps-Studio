from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import LabelEncoder

from free_mlops.config import get_settings


class DataDriftDetector:
    """Detector de drift em dados de produção vs dados de treino."""
    
    def __init__(self, model_id: str, settings: Optional[Any] = None):
        self.model_id = model_id
        self.settings = settings or get_settings()
        self.drift_file = self.settings.artifacts_dir / "monitoring" / f"{model_id}_drift.json"
        self.drift_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Thresholds para detecção de drift
        self.thresholds = {
            "ks_statistic_threshold": 0.1,  # Kolmogorov-Smirnov
            "js_distance_threshold": 0.1,   # Jensen-Shannon
            "psi_threshold": 0.25,           # Population Stability Index
            "cramer_v_threshold": 0.1,       # Cramér's V para categóricas
        }
    
    def save_reference_data(self, reference_data: pd.DataFrame, column_types: Dict[str, str]) -> None:
        """Salva dados de referência (treino) para comparação."""
        reference_stats = self._calculate_data_statistics(reference_data, column_types)
        
        drift_data = self._load_drift_data()
        drift_data["reference_data"] = {
            "statistics": reference_stats,
            "column_types": column_types,
            "shape": reference_data.shape,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        
        self._save_drift_data(drift_data)
    
    def detect_drift(self, production_data: pd.DataFrame) -> Dict[str, Any]:
        """Detecta drift entre dados de produção e referência."""
        drift_data = self._load_drift_data()
        
        if "reference_data" not in drift_data:
            raise ValueError("Nenhum dado de referência salvo. Use save_reference_data() primeiro.")
        
        reference_stats = drift_data["reference_data"]["statistics"]
        column_types = drift_data["reference_data"]["column_types"]
        
        # Calcular estatísticas dos dados de produção
        production_stats = self._calculate_data_statistics(production_data, column_types)
        
        # Detectar drift para cada coluna
        drift_results = {}
        overall_drift_score = 0
        n_columns = 0
        
        for column in reference_stats.keys():
            if column not in production_stats:
                continue
            
            ref_stats = reference_stats[column]
            prod_stats = production_stats[column]
            col_type = column_types.get(column, "numeric")
            
            if col_type == "numeric":
                drift_result = self._detect_numeric_drift(column, ref_stats, prod_stats)
            else:
                drift_result = self._detect_categorical_drift(column, ref_stats, prod_stats)
            
            drift_results[column] = drift_result
            overall_drift_score += drift_result["drift_score"]
            n_columns += 1
        
        # Calcular drift geral
        overall_drift_score = overall_drift_score / n_columns if n_columns > 0 else 0
        
        result = {
            "model_id": self.model_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_drift_score": overall_drift_score,
            "has_drift": overall_drift_score > 0.5,  # Threshold geral
            "column_drift": drift_results,
            "production_shape": production_data.shape,
            "reference_shape": drift_data["reference_data"]["shape"],
        }
        
        # Salvar resultado
        if "drift_history" not in drift_data:
            drift_data["drift_history"] = []
        drift_data["drift_history"].append(result)
        
        # Manter apenas últimos 100 registros
        if len(drift_data["drift_history"]) > 100:
            drift_data["drift_history"] = drift_data["drift_history"][-100:]
        
        self._save_drift_data(drift_data)
        
        return result
    
    def _detect_numeric_drift(self, column: str, ref_stats: Dict, prod_stats: Dict) -> Dict[str, Any]:
        """Detecta drift em colunas numéricas."""
        drift_result = {
            "column": column,
            "type": "numeric",
            "drift_detected": False,
            "drift_score": 0,
            "tests": {},
        }
        
        # Kolmogorov-Smirnov test
        try:
            # Simular dados baseados nas estatísticas
            ref_sample = np.random.normal(ref_stats["mean"], ref_stats["std"], 1000)
            prod_sample = np.random.normal(prod_stats["mean"], prod_stats["std"], 1000)
            
            ks_stat, ks_pvalue = stats.ks_2samp(ref_sample, prod_sample)
            drift_result["tests"]["ks_test"] = {
                "statistic": ks_stat,
                "p_value": ks_pvalue,
                "drift_detected": ks_stat > self.thresholds["ks_statistic_threshold"],
            }
            
            if ks_stat > self.thresholds["ks_statistic_threshold"]:
                drift_result["drift_detected"] = True
                drift_result["drift_score"] += 0.3
        except Exception:
            pass
        
        # Comparação de médias
        mean_diff = abs(ref_stats["mean"] - prod_stats["mean"])
        mean_ratio = mean_diff / (abs(ref_stats["mean"]) + 1e-8)
        
        drift_result["tests"]["mean_comparison"] = {
            "reference_mean": ref_stats["mean"],
            "production_mean": prod_stats["mean"],
            "difference": mean_diff,
            "ratio": mean_ratio,
            "drift_detected": mean_ratio > 0.1,
        }
        
        if mean_ratio > 0.1:
            drift_result["drift_detected"] = True
            drift_result["drift_score"] += 0.2
        
        # Comparação de desvio padrão
        std_diff = abs(ref_stats["std"] - prod_stats["std"])
        std_ratio = std_diff / (ref_stats["std"] + 1e-8)
        
        drift_result["tests"]["std_comparison"] = {
            "reference_std": ref_stats["std"],
            "production_std": prod_stats["std"],
            "difference": std_diff,
            "ratio": std_ratio,
            "drift_detected": std_ratio > 0.2,
        }
        
        if std_ratio > 0.2:
            drift_result["drift_detected"] = True
            drift_result["drift_score"] += 0.2
        
        # Limitar drift score a 1.0
        drift_result["drift_score"] = min(drift_result["drift_score"], 1.0)
        
        return drift_result
    
    def _detect_categorical_drift(self, column: str, ref_stats: Dict, prod_stats: Dict) -> Dict[str, Any]:
        """Detecta drift em colunas categóricas."""
        drift_result = {
            "column": column,
            "type": "categorical",
            "drift_detected": False,
            "drift_score": 0,
            "tests": {},
        }
        
        # Preparar distribuições
        ref_dist = ref_stats["value_counts"]
        prod_dist = prod_stats["value_counts"]
        
        # Alinhar categorias
        all_categories = set(ref_dist.keys()) | set(prod_dist.keys())
        
        ref_probs = np.array([ref_dist.get(cat, 0) for cat in all_categories])
        prod_probs = np.array([prod_dist.get(cat, 0) for cat in all_categories])
        
        # Normalizar
        ref_probs = ref_probs / np.sum(ref_probs)
        prod_probs = prod_probs / np.sum(prod_probs)
        
        # Jensen-Shannon distance
        try:
            js_distance = jensenshannon(ref_probs, prod_probs)
            drift_result["tests"]["jensen_shannon"] = {
                "distance": js_distance,
                "drift_detected": js_distance > self.thresholds["js_distance_threshold"],
            }
            
            if js_distance > self.thresholds["js_distance_threshold"]:
                drift_result["drift_detected"] = True
                drift_result["drift_score"] += 0.4
        except Exception:
            pass
        
        # Population Stability Index (PSI)
        try:
            psi = self._calculate_psi(ref_probs, prod_probs)
            drift_result["tests"]["psi"] = {
                "psi": psi,
                "drift_detected": psi > self.thresholds["psi_threshold"],
            }
            
            if psi > self.thresholds["psi_threshold"]:
                drift_result["drift_detected"] = True
                drift_result["drift_score"] += 0.4
        except Exception:
            pass
        
        # Comparação de distribuições
        ref_top_categories = dict(sorted(ref_dist.items(), key=lambda x: x[1], reverse=True)[:5])
        prod_top_categories = dict(sorted(prod_dist.items(), key=lambda x: x[1], reverse=True)[:5])
        
        drift_result["tests"]["distribution_comparison"] = {
            "reference_top": ref_top_categories,
            "production_top": prod_top_categories,
            "new_categories": set(prod_dist.keys()) - set(ref_dist.keys()),
            "missing_categories": set(ref_dist.keys()) - set(prod_dist.keys()),
        }
        
        # Novas categorias ou categorias faltando indicam drift
        if drift_result["tests"]["distribution_comparison"]["new_categories"] or \
           drift_result["tests"]["distribution_comparison"]["missing_categories"]:
            drift_result["drift_detected"] = True
            drift_result["drift_score"] += 0.2
        
        # Limitar drift score a 1.0
        drift_result["drift_score"] = min(drift_result["drift_score"], 1.0)
        
        return drift_result
    
    def _calculate_psi(self, expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
        """Calcula Population Stability Index."""
        # Evitar divisão por zero
        expected = np.clip(expected, 1e-10, 1)
        actual = np.clip(actual, 1e-10, 1)
        
        psi = np.sum((expected - actual) * np.log(expected / actual))
        return psi
    
    def _calculate_data_statistics(self, data: pd.DataFrame, column_types: Dict[str, str]) -> Dict[str, Any]:
        """Calcula estatísticas descritivas dos dados."""
        stats = {}
        
        for column in data.columns:
            col_type = column_types.get(column, "numeric")
            
            if col_type == "numeric":
                if pd.api.types.is_numeric_dtype(data[column]):
                    stats[column] = {
                        "mean": float(data[column].mean()),
                        "std": float(data[column].std()),
                        "min": float(data[column].min()),
                        "max": float(data[column].max()),
                        "median": float(data[column].median()),
                        "q25": float(data[column].quantile(0.25)),
                        "q75": float(data[column].quantile(0.75)),
                    }
                else:
                    # Tentar converter para numérico
                    try:
                        numeric_data = pd.to_numeric(data[column], errors="coerce").dropna()
                        if len(numeric_data) > 0:
                            stats[column] = {
                                "mean": float(numeric_data.mean()),
                                "std": float(numeric_data.std()),
                                "min": float(numeric_data.min()),
                                "max": float(numeric_data.max()),
                                "median": float(numeric_data.median()),
                                "q25": float(numeric_data.quantile(0.25)),
                                "q75": float(numeric_data.quantile(0.75)),
                            }
                    except Exception:
                        pass
            else:
                # Categórico
                value_counts = data[column].value_counts().to_dict()
                stats[column] = {
                    "value_counts": value_counts,
                    "unique_count": len(value_counts),
                    "most_frequent": max(value_counts.items(), key=lambda x: x[1]) if value_counts else None,
                }
        
        return stats
    
    def get_drift_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Retorna histórico de detecção de drift."""
        drift_data = self._load_drift_data()
        history = drift_data.get("drift_history", [])
        return history[-limit:] if history else []
    
    def _load_drift_data(self) -> Dict[str, Any]:
        """Carrega dados de drift do arquivo."""
        if self.drift_file.exists():
            try:
                return json.loads(self.drift_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}
    
    def _save_drift_data(self, drift_data: Dict[str, Any]) -> None:
        """Salva dados de drift no arquivo."""
        self.drift_file.write_text(
            json.dumps(drift_data, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )


def get_drift_detector(model_id: str) -> DataDriftDetector:
    """Factory function para DataDriftDetector."""
    return DataDriftDetector(model_id)
