import mlflow
import mlflow.sklearn
import os
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import shap
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

class MLFlowTracker:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def log_experiment(self, params, metrics, model, model_name, artifacts=None, register=True):
        with mlflow.start_run():
            mlflow.log_params(params)
            try:
                mlflow.log_metrics(metrics)
            except Exception as e:
                print(f"Warning: Failed to log metrics due to {e}")
            
            # Log model
            mlflow.sklearn.log_model(
                model, 
                model_name, 
                registered_model_name=model_name if register else None
            )
            
            # Log artifacts (e.g., plots, dataset samples)
            if artifacts:
                for art_path in artifacts:
                    if os.path.exists(art_path):
                        mlflow.log_artifact(art_path)
            
            run_id = mlflow.active_run().info.run_id
            return run_id

class DataLake:
    def __init__(self, base_path="./data_lake"):
        self.base_path = base_path
        if not os.path.exists(base_path):
            os.makedirs(base_path)

    def save_dataset(self, df, name):
        """Save dataset with versioning (timestamp based)."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = os.path.join(self.base_path, name)
        if not os.path.exists(version_dir):
            os.makedirs(version_dir)
        
        file_path = os.path.join(version_dir, f"v_{timestamp}.csv")
        df.to_csv(file_path, index=False)
        return file_path

    def list_datasets(self):
        if not os.path.exists(self.base_path):
            return []
        return os.listdir(self.base_path)

    def list_versions(self, name):
        version_dir = os.path.join(self.base_path, name)
        if not os.path.exists(version_dir):
            return []
        return sorted(os.listdir(version_dir), reverse=True)

    def load_version(self, name, version, **kwargs):
        file_path = os.path.join(self.base_path, name, version)
        return pd.read_csv(file_path, **kwargs)

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

class ModelExplainer:
    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train
        # Use SHAP KernelExplainer for general compatibility or TreeExplainer for trees
        if hasattr(model, "predict_proba"):
            self.explainer = shap.Explainer(model.predict, X_train)
        else:
            self.explainer = shap.Explainer(model.predict, X_train)

    def get_shap_values(self, X):
        shap_values = self.explainer(X)
        return shap_values

    def plot_importance(self, X):
        shap_values = self.explainer(X)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, X, show=False)
        return fig

def get_all_runs():
    """Retorna todos os experimentos (runs) do MLflow."""
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        experiments = client.search_experiments()
        all_runs = []
        for exp in experiments:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            if not runs.empty:
                runs['experiment_name'] = exp.name
                all_runs.append(runs)
        
        if all_runs:
            return pd.concat(all_runs, ignore_index=True)
        return pd.DataFrame()
    except Exception as e:
        print(f"Erro ao buscar runs: {e}")
        return pd.DataFrame()

def get_model_registry():
    """List all registered models in MLFlow."""
    try:
        experiments = mlflow.search_experiments()
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id for exp in experiments])
        # Add experiment name to runs
        exp_map = {exp.experiment_id: exp.name for exp in experiments}
        if not runs.empty:
            runs['experiment_name'] = runs['experiment_id'].map(exp_map)
        return runs
    except Exception as e:
        print(f"Error fetching model registry: {e}")
        return pd.DataFrame()

def register_model_from_run(run_id, model_name):
    """Register a model that was previously logged in a run."""
    try:
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, model_name)
        return True
    except Exception as e:
        print(f"Error registering model: {e}")
        return False

def get_registered_models():
    """List all models in the Model Registry."""
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        return client.search_registered_models()
    except Exception as e:
        print(f"Error fetching registered models: {e}")
        return []

def get_model_details(model_name, version=None):
    """Get details of a registered model version."""
    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        
        if version is None:
            # Get latest version
            versions = client.get_latest_versions(model_name, stages=["Production", "Staging", "None"])
            if not versions:
                return None
            model_version = versions[0]
        else:
            model_version = client.get_model_version(model_name, version)
            
        run = client.get_run(model_version.run_id)
        
        # Extract metadata
        details = {
            "name": model_name,
            "version": model_version.version,
            "run_id": model_version.run_id,
            "params": run.data.params,
            "metrics": run.data.metrics,
            "tags": run.data.tags,
            "creation_timestamp": model_version.creation_timestamp,
            "status": model_version.status,
            "description": model_version.description,
            "source": model_version.source
        }
        
        return details
    except Exception as e:
        print(f"Error fetching model details: {e}")
        return None

def load_registered_model(model_name, version=None):
    """Load a registered model from MLflow."""
    try:
        if version:
            model_uri = f"models:/{model_name}/{version}"
        else:
            model_uri = f"models:/{model_name}/latest"
            
        return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        print(f"Error loading registered model: {e}")
        return None
