import sys
sys.path.append('.')
import pandas as pd
from mlops_utils import get_datalake
from app import get_registered_models, load_registered_model
from stability_engine import StabilityAnalyzer
import traceback
import logging

logging.basicConfig(level=logging.INFO)

print('Starting script...')
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
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{m_name}'")
        version_num = versions[0].version
        
        loaded_pipeline = load_registered_model(m_name, version_num)
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
            
            # Run test
            try:
                report = analyzer.run_general_stability_check(n_iterations=2)
                print('report keys:', report.keys())
                print('seed raw results:', report['raw_seed'])
                print('split raw results:', report['raw_split'])
            except Exception as e:
                traceback.print_exc()
        else:
            print('TARGET NOT IN DF:', df.columns)
    else:
        print('NO MODELS')
else:
    print('NO DATASETS')
