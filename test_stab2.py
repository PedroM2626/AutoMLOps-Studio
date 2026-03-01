import sys
sys.path.append('.')
import pandas as pd
from mlops_utils import get_datalake
import mlflow
from mlflow.tracking import MlflowClient
from stability_engine import StabilityAnalyzer
import traceback
import logging

logging.basicConfig(level=logging.INFO)

print('Starting script...')

def get_registered_models():
    client = MlflowClient()
    models = client.search_registered_models()
    return models

dl = get_datalake()
datasets = dl.list_datasets()
if datasets:
    ds_name = datasets[-1]
    print("Dataset:", ds_name)
    ver = dl.list_versions(ds_name)[0]
    df = dl.load_version(ds_name, ver)
    
    models = get_registered_models()
    if models:
        m_name = models[-1].name
        print("Model:", m_name)
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{m_name}'")
        version_num = versions[0].version
        
        model_uri = f"models:/{m_name}/{version_num}"
        loaded_pipeline = mlflow.pyfunc.load_model(model_uri)
        print("Pipeline Wrapper:", type(loaded_pipeline))
        
        actual_model = loaded_pipeline
        if hasattr(loaded_pipeline, '_model_impl'):
            if hasattr(loaded_pipeline._model_impl, 'sklearn_model'):
                actual_model = loaded_pipeline._model_impl.sklearn_model
            elif hasattr(loaded_pipeline._model_impl, 'python_model') and hasattr(loaded_pipeline._model_impl.python_model, 'pipeline'):
                actual_model = loaded_pipeline._model_impl.python_model.pipeline
                
        print("Unwrapped model:", type(actual_model))
        
        target = 'target' 
        if target in df.columns:
            X = df.drop(columns=[target])
            y = df[target]
            
            analyzer = StabilityAnalyzer(base_model=actual_model, X=X, y=y, task_type='classification')
            
            try:
                report = analyzer.run_general_stability_check(n_iterations=2)
                print('report keys:', report.keys())
                print('seed df len:', len(report['seed_stability']))
            except Exception as e:
                traceback.print_exc()
        else:
            print('TARGET NOT IN DF:', df.columns)
    else:
        print('NO MODELS')
else:
    print('NO DATASETS')
