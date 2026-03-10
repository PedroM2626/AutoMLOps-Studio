import mlflow
import mlflow.sklearn
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
from src.utils.helpers import get_consumption_code

class MLFlowTracker:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)

    def log_experiment(self, params, metrics, model, model_name, artifacts=None, register=True, feature_names=None):
        with mlflow.start_run():
            mlflow.log_params(params)
            try:
                mlflow.log_metrics(metrics)
            except Exception as e:
                print(f"Warning: Failed to log metrics due to {e}")
            
            # Log model
            safe_artifact_path = model_name.replace(" ", "_").replace("-", "_").replace("__", "_")
            mlflow.sklearn.log_model(
                model, 
                safe_artifact_path, 
                registered_model_name=model_name if register else None
            )
            
            # Log artifacts (e.g., plots, dataset samples)
            if artifacts:
                for art_path in artifacts:
                    if os.path.exists(art_path):
                        mlflow.log_artifact(art_path)
            
            run_id = mlflow.active_run().info.run_id
            
            # Generate and log consumption code sample
            try:
                task_type = params.get('task_type', 'classification')
                code_sample = get_consumption_code(model_name, run_id, task_type, feature_names=feature_names)
                code_filename = f"consume_{safe_artifact_path}.py"
                with open(code_filename, "w", encoding="utf-8") as f:
                    f.write(code_sample)
                mlflow.log_artifact(code_filename)
                if os.path.exists(code_filename):
                    os.remove(code_filename)
            except Exception as e:
                print(f"Warning: Failed to log code sample: {e}")

            return run_id

def get_all_runs():
    """Returns all MLflow experiments (runs)."""
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
        print(f"Error searching for runs: {e}")
        return pd.DataFrame()

def get_model_registry():
    """List all registered models in MLFlow."""
    try:
        experiments = mlflow.search_experiments()
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id for exp in experiments])
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
            versions = client.get_latest_versions(model_name, stages=["Production", "Staging", "None"])
            if not versions:
                return None
            model_version = versions[0]
        else:
            model_version = client.get_model_version(model_name, version)
            
        run = client.get_run(model_version.run_id)
        
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

def get_run_details(run_id: str) -> dict:
    """
    Returns a rich dict of all MLflow data for a specific run_id.
    """
    if not run_id or run_id == "dummy_run_id":
        return {"error": "Preview mode (dummy run) or no run ID provided. Real tracking unavailable."}

    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        run = client.get_run(run_id)

        params = dict(run.data.params)
        metrics = dict(run.data.metrics)

        metric_history = {}
        for key in metrics:
            try:
                history = client.get_metric_history(run_id, key)
                metric_history[key] = [{"step": m.step, "value": m.value, "timestamp": m.timestamp} for m in history]
            except Exception:
                pass

        tags = {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}

        try:
            artifacts = [a.path for a in client.list_artifacts(run_id)]
        except Exception:
            artifacts = []

        info = {
            "run_id": run.info.run_id,
            "experiment_id": run.info.experiment_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            "artifact_uri": run.info.artifact_uri,
            "lifecycle_stage": run.info.lifecycle_stage,
        }

        try:
            exp = client.get_experiment(run.info.experiment_id)
            info["experiment_name"] = exp.name
        except Exception:
            info["experiment_name"] = "Unknown"

        return {
            "info": info,
            "params": params,
            "metrics": metrics,
            "metric_history": metric_history,
            "tags": tags,
            "artifacts": artifacts,
        }
    except Exception as e:
        return {"error": str(e)}
