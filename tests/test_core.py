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
    
    trainer = AutoMLTrainer(task_type='classification', preset='test', ensemble_mode='single', use_deep_learning=False)
    model = trainer.train(X_proc, y_proc, n_trials=1, selected_models=['logistic_regression'])
    
    assert model is not None
    metrics, preds = trainer.evaluate(X_proc, y_proc)
    assert 'accuracy' in metrics

def test_automl_trainer_regression(sample_data):
    # Change target to continuous for regression
    sample_data['target'] = np.random.rand(len(sample_data))
    processor = AutoMLDataProcessor(target_column='target')
    X_proc, y_proc = processor.fit_transform(sample_data)
    
    trainer = AutoMLTrainer(task_type='regression', preset='test', ensemble_mode='single', use_deep_learning=False)
    model = trainer.train(X_proc, y_proc, n_trials=1, selected_models=['linear_regression'])
    
    assert model is not None
    metrics, preds = trainer.evaluate(X_proc, y_proc)
    assert 'r2' in metrics

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
    trainer = AutoMLTrainer(task_type='classification', preset='test', ensemble_mode='single', use_deep_learning=False)
    model = trainer.train(X_proc, y_proc, n_trials=1, selected_models=['logistic_regression'])
    
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
    """Test that use_deep_learning and ensemble_mode flags correctly filter available models."""
    # Test 1: All enabled (default - both mode)
    trainer_all = AutoMLTrainer(task_type='classification', ensemble_mode='both', use_deep_learning=True)
    models_all = trainer_all.get_available_models()
    assert 'custom_voting' in models_all or 'voting_ensemble' in models_all
    assert 'mlp' in models_all

    # Test 2: Ensemble Mode = single (no ensembles)
    trainer_single = AutoMLTrainer(task_type='classification', ensemble_mode='single', use_deep_learning=True)
    models_single = trainer_single.get_available_models()
    assert not any(m in ['custom_voting', 'custom_stacking', 'custom_bagging', 'voting_ensemble', 'stacking_ensemble'] for m in models_single)
    assert 'mlp' in models_single

    # Test 3: Ensemble Mode = ensemble_only
    trainer_ens_only = AutoMLTrainer(task_type='classification', ensemble_mode='ensemble_only', use_deep_learning=True)
    models_ens_only = trainer_ens_only.get_available_models()
    # Should only contain ensemble keys (custom or built-in)
    from src.core.trainer import _ENSEMBLE_MODEL_KEYS
    assert all(m in _ENSEMBLE_MODEL_KEYS for m in models_ens_only)
    assert len(models_ens_only) > 0

    # Test 4: DL disabled
    trainer_no_dl = AutoMLTrainer(task_type='regression', ensemble_mode='both', use_deep_learning=False)
    models_no_dl = trainer_no_dl.get_available_models()
    assert not any(m in ['mlp', 'transformer'] for m in models_no_dl)

def test_mlflow_dummy_fallback(sample_data):
    """Test that MLflow gracefully falls back when tracking URI/DB is corrupted or unavailable."""
    from src.tracking.mlflow import get_run_details
    # Match the actual implementation message from mlflow.py
    assert get_run_details("dummy_run_id") == {"error": "Preview mode (dummy run) or no run ID provided. Real tracking unavailable."}

def test_custom_ensemble_names():
    """Test that custom ensemble keys are properly mapped to display names."""
    from src.core.trainer import get_ensemble_display_name
    assert get_ensemble_display_name('custom_voting') == 'Custom Voting Ensemble'
    assert get_ensemble_display_name('custom_stacking') == 'Custom Stacking Ensemble'
    assert get_ensemble_display_name('custom_bagging') == 'Custom Bagging Ensemble'
    assert get_ensemble_display_name('random_forest') == 'random_forest'  # Non-ensemble returns key
