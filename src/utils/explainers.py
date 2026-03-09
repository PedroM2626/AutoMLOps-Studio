import shap
import matplotlib.pyplot as plt
import numpy as np

class ModelExplainer:
    def __init__(self, model, X_train, task_type="classification"):
        self.model = model
        self.X_train = X_train
        self.task_type = task_type
        
        try:
            self.explainer = shap.Explainer(model, X_train)
        except Exception:
            background = X_train
            if len(X_train) > 100:
                background = shap.utils.sample(X_train, 100)
                
            if task_type == "classification" and hasattr(model, "predict_proba"):
                self.explainer = shap.KernelExplainer(model.predict_proba, background)
            else:
                self.explainer = shap.KernelExplainer(model.predict, background)

    def get_shap_values(self, X):
        try:
            shap_values = self.explainer(X)
        except:
            shap_values = self.explainer.shap_values(X)
        return shap_values

    def plot_importance(self, X_test, plot_type="summary"):
        """
        Gera plot de importância SHAP.
        """
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

        fig = plt.figure()
        if plot_type == "bar":
            shap.summary_plot(final_shap_values, X_test, plot_type="bar", show=False)
        else:
            shap.summary_plot(final_shap_values, X_test, show=False)
        return fig
