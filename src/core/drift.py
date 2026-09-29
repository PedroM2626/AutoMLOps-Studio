import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

# A chi-square contingency test needs expected counts of at least ~5 per cell to mean
# anything. With one row per level - a text or ID column - almost every cell is a single
# observation, the statistic collapses and the test answers "no drift" without evidence.
# 50 follows deepchecks' default max_num_categories for categorical drift.
MAX_CATEGORICAL_LEVELS = 50

class DriftDetector:
    @staticmethod
    def detect_drift(reference_data, current_data, threshold=0.05):
        """Detect drift on numeric and categorical features."""
        drifts = {}
        # Sorted so repeated runs on the same pair of frames list features in the
        # same order; set iteration order changes with the interpreter's hash seed.
        common_cols = sorted(set(reference_data.columns) & set(current_data.columns))

        for col in common_cols:
            if pd.api.types.is_numeric_dtype(reference_data[col]):
                ref_series = reference_data[col].dropna()
                cur_series = current_data[col].dropna()
                if ref_series.empty or cur_series.empty:
                    # ks_2samp returns NaN for an empty sample, and a NaN p-value would
                    # render as a blank statistic while silently reporting "stable".
                    stat, p_value = 0.0, 1.0
                else:
                    stat, p_value = ks_2samp(ref_series, cur_series)
                drifts[col] = {
                    'feature_type': 'numeric',
                    'test': 'ks_2samp',
                    'statistic': float(stat),
                    'p_value': float(p_value),
                    'reference_mean': float(ref_series.mean()) if not ref_series.empty else None,
                    'current_mean': float(cur_series.mean()) if not cur_series.empty else None,
                    'drift_detected': bool(p_value < threshold)
                }
            else:
                ref_counts = reference_data[col].fillna('__nan__').value_counts()
                cur_counts = current_data[col].fillna('__nan__').value_counts()
                # key=str: an object column can mix ints and strings, which sorted()
                # cannot compare.
                categories = sorted(set(ref_counts.index) | set(cur_counts.index), key=str)
                if len(categories) > MAX_CATEGORICAL_LEVELS:
                    drifts[col] = {
                        'feature_type': 'categorical',
                        'test': 'skipped_high_cardinality',
                        'statistic': 0.0,
                        'p_value': 1.0,
                        'categories_compared': len(categories),
                        'drift_detected': False,
                        'skipped_reason': (
                            f"{len(categories)} distinct levels exceed the "
                            f"{MAX_CATEGORICAL_LEVELS}-level chi-square limit"
                        ),
                    }
                    continue

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
                    'drift_detected': bool(p_value < threshold)
                }
        return drifts
