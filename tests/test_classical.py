import sys
import os
import numpy as np
import pandas as pd
import logging
from sklearn.datasets import make_classification, make_regression

# Add project root to path
sys.path.append(os.getcwd())

from src.engines.classical import AutoMLTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModelTester")

def test_classical_models():
    # Focused test on standard classical models
    models_to_test = [
        'logistic_regression', 'random_forest', 'xgboost', 'lightgbm', 'catboost', 
        'extra_trees', 'gradient_boosting', 'decision_tree', 'knn', 'svm', 
        'lda', 'qda', 'ada_boost'
    ]
    
    # Classification
    X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)
    trainer_clf = AutoMLTrainer(task_type="classification", preset="test")
    
    logger.info("--- Testing Classification ---")
    for m in models_to_test:
        try:
            class MockTrial:
                def suggest_float(self, *args, **kwargs): return 0.1
                def suggest_int(self, *args, **kwargs): return 1
                def suggest_categorical(self, name, choices): return choices[0]
            
            model = trainer_clf._get_models(trial=MockTrial(), name=m, random_state=42)
            if model is not None:
                model.fit(X, y)
                logger.info(f"OK: {m}")
            else:
                logger.warning(f"SKIP: {m} (Not available)")
        except Exception as e:
            logger.error(f"FAIL: {m}: {e}")

def test_supervised_dimensionality_reduction():
    logger.info("--- Testing Supervised Dimensionality Reduction ---")
    X, y = make_classification(n_samples=60, n_features=8, n_classes=3, n_informative=5, random_state=42)
    trainer_dr = AutoMLTrainer(task_type="dimensionality_reduction", preset="test")
    
    models = ['pca', 'truncated_svd', 'lda', 'nca', 'pls']
    class MockTrial:
        def suggest_float(self, *args, **kwargs): return 0.1
        def suggest_int(self, name, low, high, *args, **kwargs): return min(2, high)
        def suggest_categorical(self, name, choices): return choices[0]

    for m in models:
        try:
            model = trainer_dr._get_models(trial=MockTrial(), name=m, random_state=42)
            assert model is not None, f"Model {m} should not be None"
            if m in ['lda', 'nca', 'pls']:
                model.fit(X, y)
            else:
                model.fit(X)
            X_trans = model.transform(X)
            assert X_trans.shape[0] == 60, f"Transformed shape mismatch for {m}"
            logger.info(f"OK (Supervised DR): {m} -> Transformed shape: {X_trans.shape}")
        except Exception as e:
            logger.error(f"FAIL (Supervised DR) {m}: {e}")
            raise e

if __name__ == "__main__":
    test_classical_models()
    test_supervised_dimensionality_reduction()
