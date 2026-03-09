import pandas as pd
from scipy.stats import ks_2samp

class DriftDetector:
    @staticmethod
    def detect_drift(reference_data, current_data, threshold=0.05):
        """Detect drift using Kolmogorov-Smirnov test."""
        drifts = {}
        common_cols = set(reference_data.columns) & set(current_data.columns)
        
        for col in common_cols:
            if pd.api.types.is_numeric_dtype(reference_data[col]):
                stat, p_value = ks_2samp(reference_data[col].dropna(), current_data[col].dropna())
                drifts[col] = {
                    'p_value': p_value,
                    'drift_detected': p_value < threshold
                }
        return drifts
