import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import json

print(f"MLflow Version: {mlflow.__version__}")
# Test with local folder first
tracking_uris = [None, "sqlite:///mlflow.db"]

for uri in tracking_uris:
    print(f"\n--- Testing URI: {uri} ---")
    if uri:
        mlflow.set_tracking_uri(uri)
    
    try:
        client = MlflowClient()
        experiments = client.search_experiments()
        print(f"Found {len(experiments)} experiments")
        
        for exp in experiments:
            print(f"  Experiment: {exp.name} (ID: {exp.experiment_id})")
            
            print("    Testing mlflow.search_runs...")
            df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            print(f"    Found {len(df)} runs via search_runs")
            
            print("    Testing client.search_runs...")
            runs = client.search_runs(experiment_ids=[exp.experiment_id])
            print(f"    Found {len(runs)} runs via client.search_runs")
            
    except Exception as e:
        print(f"    CAUGHT ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
