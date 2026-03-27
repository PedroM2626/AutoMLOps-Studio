import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

class DriftDetector:
    @staticmethod
    def detect_drift(reference_data, current_data, threshold=0.05):
        """Detect drift on numeric and categorical features."""
        drifts = {}
        common_cols = set(reference_data.columns) & set(current_data.columns)
        
        for col in common_cols:
            if pd.api.types.is_numeric_dtype(reference_data[col]):
                stat, p_value = ks_2samp(reference_data[col].dropna(), current_data[col].dropna())
                drifts[col] = {
                    'feature_type': 'numeric',
                    'test': 'ks_2samp',
                    'statistic': float(stat),
                    'p_value': p_value,
                    'reference_mean': float(reference_data[col].dropna().mean()) if not reference_data[col].dropna().empty else None,
                    'current_mean': float(current_data[col].dropna().mean()) if not current_data[col].dropna().empty else None,
                    'drift_detected': p_value < threshold
                }
            else:
                ref_counts = reference_data[col].fillna('__nan__').value_counts()
                cur_counts = current_data[col].fillna('__nan__').value_counts()
                categories = sorted(set(ref_counts.index) | set(cur_counts.index))
                ref_aligned = [int(ref_counts.get(cat, 0)) for cat in categories]
                cur_aligned = [int(cur_counts.get(cat, 0)) for cat in categories]

                if sum(ref_aligned) == 0 or sum(cur_aligned) == 0:
                    p_value = 1.0
                    chi2 = 0.0
                else:
                    contingency = [ref_aligned, cur_aligned]
                    chi2, p_value, _, _ = chi2_contingency(contingency)

                drifts[col] = {
                    'feature_type': 'categorical',
                    'test': 'chi2_contingency',
                    'statistic': float(chi2),
                    'p_value': float(p_value),
                    'categories_compared': len(categories),
                    'drift_detected': p_value < threshold
                }
        return drifts
