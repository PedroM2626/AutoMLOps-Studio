import sys
sys.path.append('.')
from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from mlops_utils import get_datalake, get_registered_models, load_registered_model
from stability_engine import StabilityAnalyzer
import traceback
import logging

logging.basicConfig(level=logging.INFO)

print('Starting script...')
dl = get_datalake()
datasets = dl.list_datasets()
ds_name = datasets[-1]
ver = dl.list_versions(ds_name)[0]
df = dl.load_version(ds_name, ver)
    
models = get_registered_models()
m_name = models[-1].name

from mlflow.tracking import MlflowClient
client = MlflowClient()
versions = client.search_model_versions(f"name='{m_name}'")
version_num = versions[0].version

loaded_pipeline = load_registered_model(m_name, version_num)

actual_model = loaded_pipeline
if hasattr(loaded_pipeline, '_model_impl'):
    if hasattr(loaded_pipeline._model_impl, 'sklearn_model'):
        actual_model = loaded_pipeline._model_impl.sklearn_model
    elif hasattr(loaded_pipeline._model_impl, 'python_model') and hasattr(loaded_pipeline._model_impl.python_model, 'pipeline'):
        actual_model = loaded_pipeline._model_impl.python_model.pipeline
        
target = 'target' 
if target in df.columns:
    X = df.drop(columns=[target])
    y = df[target]
    
    analyzer = StabilityAnalyzer(base_model=actual_model, X=X, y=y, task_type='classification')
    
    try:
        print("Running run_general_stability_check...")
        report = analyzer.run_general_stability_check(n_iterations=2)
        print("Success!")
    except Exception as e:
        traceback.print_exc()
else:
    print('Target not in df', df.columns)
