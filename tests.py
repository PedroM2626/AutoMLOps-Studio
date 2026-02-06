import pytest
import pandas as pd
import numpy as np
from automl_engine import AutoMLDataProcessor, AutoMLTrainer
from mlops_utils import DriftDetector
import os

@pytest.fixture
def sample_data():
    df = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': ['A', 'B', 'A', 'C'] * 25,
        'target': [0, 1, 0, 1] * 25
    })
    return df

def test_data_processor(sample_data):
    processor = AutoMLDataProcessor(target_column='target')
    X_proc, y_proc = processor.fit_transform(sample_data)
    
    assert X_proc.shape[0] == 100
    assert X_proc.shape[1] > 2 # One-hot encoding should expand columns
    assert len(y_proc) == 100

def test_automl_trainer_classification(sample_data):
    processor = AutoMLDataProcessor(target_column='target')
    X_proc, y_proc = processor.fit_transform(sample_data)
    
    trainer = AutoMLTrainer(task_type='classification')
    # Use 1 trial for speed in tests
    model = trainer.train(X_proc, y_proc, n_trials=1)
    
    assert model is not None
    metrics = trainer.evaluate(X_proc, y_proc)
    assert 'accuracy' in metrics

def test_drift_detector():
    ref = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
    curr = pd.DataFrame({'a': [10, 20, 30, 40, 50]})
    
    detector = DriftDetector()
    drifts = detector.detect_drift(ref, curr)
    
    assert drifts['a']['drift_detected'] == True

def test_model_save_load(sample_data):
    from automl_engine import save_pipeline, load_pipeline
    processor = AutoMLDataProcessor(target_column='target')
    X_proc, y_proc = processor.fit_transform(sample_data)
    trainer = AutoMLTrainer(task_type='classification')
    model = trainer.train(X_proc, y_proc, n_trials=1)
    
    path = "models/test_model.pkl"
    save_pipeline(processor, model, path)
    
    assert os.path.exists(path)
    
    proc_loaded, model_loaded = load_pipeline(path)
    assert proc_loaded is not None
    assert model_loaded is not None
    
    # Cleanup
    if os.path.exists(path):
        os.remove(path)
