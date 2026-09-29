import shap
import matplotlib.pyplot as plt
import numpy as np
import logging
try:
    import catboost as cb
except:
    pass

logger = logging.getLogger(__name__)

def build_waterfall(shap_values, expected_value, row, feature_names, focus_class=None):
    """Reduce one row's SHAP output to the single-vector shape a waterfall plot needs.

    TreeExplainer returns (rows, features) for regression but (rows, features, outputs) for
    classification, and shap refuses a matrix with "can currently only plot a single
    explanation" unless the output axis is picked first.
    """
    index = 0
    if isinstance(shap_values, list):
        # Older shap releases return one (rows, features) matrix per class instead of a
        # single 3-D array, so the class has to be chosen before the row is taken.
        if focus_class is not None and focus_class < len(shap_values):
            index = focus_class
        values = np.asarray(shap_values[index])
    else:
        values = np.asarray(shap_values)
    base = np.asarray(expected_value).ravel()

    if values.ndim == 3:
        values = values[0]
        outputs = values.shape[1]
        if focus_class is not None and focus_class < outputs:
            index = focus_class
        values = values[:, index]
    elif values.ndim == 2:
        values = values[0]
    elif values.ndim != 1:
        raise ValueError(f"Unsupported SHAP value shape {values.shape}")

    base_value = float(base[index]) if base.size > index else float(base[0])
    data = row.to_numpy() if hasattr(row, "to_numpy") else np.asarray(row)
    return shap.Explanation(
        values=values,
        base_values=base_value,
        data=data,
        feature_names=list(feature_names),
    )

class ModelExplainer:
    def __init__(self, model, X_train, task_type="classification"):
        self.model = model
        self.X_train = X_train
        self.task_type = task_type
        self.use_native_catboost = False
        
        model_type = str(type(model)).lower()
        logger.info(f"ModelExplainer: Initializing for {model_type}")

        # 1. Native CatBoost Optimization (Fastest & Safest)
        if "catboost" in model_type:
            try:
                logger.info("ModelExplainer: Detected CatBoost. Using Native SHAP path.")
                self.use_native_catboost = True
                self.explainer = None # Not needed for native path
                return
            except Exception as e:
                logger.warning(f"ModelExplainer: Native CatBoost SHAP failed: {e}. Falling back...")

        # 2. Prepare background data (sampled for performance and memory)
        logger.info("ModelExplainer: Preparing background data...")
        background = X_train
        if len(X_train) > 100:
            try:
                background = shap.utils.sample(X_train, 100)
            except:
                import numpy as np
                idx = np.random.choice(len(X_train), 100, replace=False)
                if hasattr(X_train, 'iloc'): background = X_train.iloc[idx]
                else: background = X_train[idx]
        logger.info("ModelExplainer: Background data ready (100 samples).")

        # 3. Choose Explainer
        try:
            if any(x in model_type for x in ["xgb", "lgbm", "randomforest", "extratrees"]):
                logger.info(f"ModelExplainer: Using TreeExplainer for {model_type}")
                try:
                    self.explainer = shap.TreeExplainer(model)
                except Exception as e:
                    logger.warning(f"TreeExplainer failed: {e}. Falling back to general Explainer.")
                    self.explainer = shap.Explainer(model, background)
            else:
                logger.info(f"ModelExplainer: Using General Explainer for {model_type}")
                self.explainer = shap.Explainer(model, background)
        except Exception as e:
            logger.warning(f"ModelExplainer: Primary explainers failed: {e}. Falling back to KernelExplainer.")
            # Fallback to KernelExplainer (slow but universal)
            if task_type == "classification" and hasattr(model, "predict_proba"):
                self.explainer = shap.KernelExplainer(model.predict_proba, background)
            else:
                self.explainer = shap.KernelExplainer(model.predict, background)
        
        logger.info("ModelExplainer: Initialization complete.")

    def get_shap_values(self, X):
        if self.use_native_catboost:
            try:
                # get_feature_importance returns [n_samples, n_features + 1] (last is bias)
                sm = self.model.get_feature_importance(cb.Pool(X), type='ShapValues')
                return sm[:, :-1]
            except Exception as e:
                logger.warning(f"Native CatBoost SHAP calculation failed: {e}")
                return None
        
        try:
            shap_values = self.explainer(X)
        except:
            shap_values = self.explainer.shap_values(X)
        return shap_values

    def plot_importance(self, X_test, plot_type="summary"):
        """
        Gera plot de importância SHAP.
        """
        final_shap_values = None
        
        if self.use_native_catboost:
            logger.info("Plotting Native CatBoost SHAP values...")
            final_shap_values = self.get_shap_values(X_test)
            if final_shap_values is None: return None
        else:
            try:
                shap_values = self.explainer(X_test)
            except:
                shap_values = self.explainer.shap_values(X_test)

            final_shap_values = shap_values
            
            if isinstance(shap_values, list):
                idx = 1 if len(shap_values) > 1 else 0
                final_shap_values = shap_values[idx]
            elif hasattr(shap_values, "values") and len(shap_values.values.shape) == 3:
                 idx = 1 if shap_values.values.shape[2] > 1 else 0
                 final_shap_values = shap_values[..., idx]
            elif hasattr(shap_values, "values"): # General shap.Explanation object
                 final_shap_values = shap_values.values

        fig = plt.figure(figsize=(10, 6))
        if plot_type == "bar":
            shap.summary_plot(final_shap_values, X_test, plot_type="bar", show=False)
        else:
            shap.summary_plot(final_shap_values, X_test, show=False)
        plt.tight_layout()
        return fig
