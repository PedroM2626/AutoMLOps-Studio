import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from stability_engine import StabilityAnalyzer
import traceback
import logging

logging.basicConfig(level=logging.INFO)

# build a mock classification dataset
df = pd.DataFrame({
    'feature1': np.random.randn(100),
    'feature2': np.random.randn(100),
    'target': np.random.randint(0, 2, 100)
})
print("Mock dataset generated.")

client = MlflowClient()
models = client.search_registered_models()
if models:
    m_name = models[-1].name
    print('Testing model:', m_name)
    versions = client.search_model_versions(f"name='{m_name}'")
    version_num = versions[0].version
    
    model_uri = f"models:/{m_name}/{version_num}"
    loaded_pipeline = mlflow.pyfunc.load_model(model_uri)
    
    actual_model = loaded_pipeline
    if hasattr(loaded_pipeline, '_model_impl'):
        if hasattr(loaded_pipeline._model_impl, 'sklearn_model'):
            actual_model = loaded_pipeline._model_impl.sklearn_model
        elif hasattr(loaded_pipeline._model_impl, 'python_model') and hasattr(loaded_pipeline._model_impl.python_model, 'pipeline'):
            actual_model = loaded_pipeline._model_impl.python_model.pipeline
            
    print("Base Model type:", type(actual_model))
    
    # Mocking target
    target = 'target' 
    X = df.drop(columns=[target])
    y = df[target]
    
    analyzer = StabilityAnalyzer(base_model=actual_model, X=X, y=y, task_type='classification')
    
    try:
        report = analyzer.run_general_stability_check(n_iterations=2)
        print('DF Keys:', report.keys())
        print('Seed Stability Shape:', report['seed_stability'].shape)
        print('Seed Trace:', report['seed_stability'])
    except Exception as e:
        traceback.print_exc()
else:
    print("NO MODELS FOUND IN REGISTRY.")
