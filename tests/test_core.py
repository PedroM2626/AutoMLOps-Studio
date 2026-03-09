import pytest
import pandas as pd
import numpy as np
from src.core.processor import AutoMLDataProcessor
from src.engines.classical import AutoMLTrainer
from src.core.drift import DriftDetector
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
    
    trainer = AutoMLTrainer(task_type='classification', use_ensemble=False, use_deep_learning=False)
    # Use 1 trial for speed in tests
    model = trainer.train(X_proc, y_proc, n_trials=1)
    
    assert model is not None
    metrics, preds = trainer.evaluate(X_proc, y_proc)
    assert 'accuracy' in metrics

def test_drift_detector():
    ref = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
    curr = pd.DataFrame({'a': [10, 20, 30, 40, 50]})
    
    detector = DriftDetector()
    drifts = detector.detect_drift(ref, curr)
    
    assert drifts['a']['drift_detected'] == True

def test_model_save_load(sample_data):
    from src.engines.classical import save_pipeline, load_pipeline
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

def test_automl_trainer_filtering():
    """Test that use_ensemble and use_deep_learning flags correctly filter available models."""
    # Test 1: All enabled (default)
    trainer_all = AutoMLTrainer(task_type='classification', use_ensemble=True, use_deep_learning=True)
    models_all = trainer_all.get_available_models()
    assert 'custom_voting' in models_all or 'voting_classifier' in models_all or 'Voting Classifier' in models_all or any('vot' in m.lower() or 'stack' in m.lower() for m in models_all)
    assert 'mlp' in [m.lower() for m in models_all] or 'neural_network' in [m.lower() for m in models_all]

    # Test 2: Ensembles disabled
    trainer_no_ens = AutoMLTrainer(task_type='classification', use_ensemble=False, use_deep_learning=True)
    models_no_ens = trainer_no_ens.get_available_models()
    assert not any('vot' in m.lower() or 'stack' in m.lower() for m in models_no_ens)

    # Test 3: DL disabled (using regression to check mlp_regressor if present)
    trainer_no_dl = AutoMLTrainer(task_type='regression', use_ensemble=True, use_deep_learning=False)
    models_no_dl = trainer_no_dl.get_available_models()
    assert not any('mlp' in m.lower() or 'neural' in m.lower() or 'transformer' in m.lower() for m in models_no_dl)

    # Test 4: Both disabled
    trainer_none = AutoMLTrainer(task_type='classification', use_ensemble=False, use_deep_learning=False)
    models_none = trainer_none.get_available_models()
    assert not any('vot' in m.lower() or 'stack' in m.lower() for m in models_none)
    assert not any('mlp' in m.lower() or 'neural' in m.lower() or 'transformer' in m.lower() for m in models_none)
    # Ensure standard models like RF or Logistic Regression are still there
    assert any('forest' in m.lower() or 'logistics' in m.lower() or 'logistic' in m.lower() for m in models_none)
