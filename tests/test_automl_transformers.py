import unittest
import sys
import os
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock transformers import BEFORE importing automl_engine
sys.modules['transformers'] = MagicMock()
sys.modules['torch'] = MagicMock()

# Now import automl_engine
import automl_engine
from automl_engine import AutoMLTrainer, AutoMLDataProcessor

# We need to monkey-patch TRANSFORMERS_AVAILABLE to True for the test
automl_engine.TRANSFORMERS_AVAILABLE = True

from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

# Mock TransformersWrapper
class MockTransformersWrapper(BaseEstimator, ClassifierMixin, RegressorMixin):
    def __init__(self, model_name='bert-base-uncased', task='classification', epochs=3, learning_rate=2e-5):
        self.model_name = model_name
        self.task = task
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.model = MagicMock()
        
    def fit(self, X, y):
        # Simulate fitting
        print(f"MockTransformersWrapper fit called with model={self.model_name}, epochs={self.epochs}, lr={self.learning_rate}")
        return self
        
    def predict(self, X):
        # Return dummy predictions
        print(f"MockTransformersWrapper predict called on {len(X)} samples")
        if self.task == 'classification':
            return np.zeros(len(X))
        else:
            return np.zeros(len(X))
            
    def get_params(self, deep=True):
        return {
            "model_name": self.model_name,
            "task": self.task,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate
        }
        
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

# Replace the real class with our mock
automl_engine.TransformersWrapper = MockTransformersWrapper

class TestAutoMLTransformers(unittest.TestCase):
    def setUp(self):
        # Create dummy classification data (text-like)
        self.df_class = pd.DataFrame({
            'text': [f"sample text {i}" for i in range(20)],
            'feature2': np.random.rand(20),
            'target': np.random.randint(0, 2, 20)
        })
        
        # Create dummy regression data
        self.df_reg = pd.DataFrame({
            'text': [f"sample text {i}" for i in range(20)],
            'feature2': np.random.rand(20),
            'target': np.random.rand(20) * 100
        })

    def test_transformer_selection_flow(self):
        """
        Simulate the 'Novo Treino' tab flow where a user selects a Transformer model.
        """
        print("\nTesting Transformer Selection Flow (Mocked)...")
        
        task_type = 'classification'
        target_col = 'target'
        
        # 1. Data Processing
        processor = AutoMLDataProcessor(target_column=target_col, task_type=task_type)
        X_train_proc, y_train_proc = processor.fit_transform(self.df_class)
        
        # 2. Training Configuration (Simulating UI inputs)
        selected_tf = "bert-base-uncased"
        lr = 2e-5
        epochs = 3
        
        manual_params = {
            'model_name': selected_tf, 
            'learning_rate': lr, 
            'num_train_epochs': epochs
        }
        selected_models = [selected_tf]
        
        trainer = AutoMLTrainer(task_type=task_type)
        
        # Note: transformers are not in get_available_models() by default, so we skip that check.
        
        print(f"Training with selected model: {selected_models} and params: {manual_params}")
        
        # 3. Train
        def mock_callback(trial, score, full_name, dur, metrics=None):
            print(f"Callback: {full_name}, Score: {score}")

        best_model = trainer.train(
            X_train_proc, 
            y_train_proc, 
            n_trials=1, 
            timeout=10,
            callback=mock_callback,
            selected_models=selected_models,
            manual_params=manual_params,
            experiment_name="test_transformer_mock"
        )
        
        self.assertIsNotNone(best_model)
        self.assertIsInstance(best_model, MockTransformersWrapper)
        self.assertEqual(best_model.model_name, selected_tf)
        self.assertEqual(best_model.learning_rate, lr)
        
        print("Transformer Selection Flow Passed!")

    def test_transformer_regression_flow(self):
        """
        Simulate Transformer for regression.
        """
        print("\nTesting Transformer Regression Flow (Mocked)...")
        
        task_type = 'regression'
        target_col = 'target'
        
        processor = AutoMLDataProcessor(target_column=target_col, task_type=task_type)
        X_train_proc, y_train_proc = processor.fit_transform(self.df_reg)
        
        selected_tf = "bert-base-uncased-reg"
        manual_params = {'model_name': selected_tf, 'learning_rate': 2e-5, 'num_train_epochs': 1}
        selected_models = [selected_tf]
        
        trainer = AutoMLTrainer(task_type=task_type)
        
        best_model = trainer.train(
            X_train_proc, 
            y_train_proc, 
            n_trials=1,
            timeout=10,
            selected_models=selected_models,
            manual_params=manual_params,
            experiment_name="test_transformer_reg_mock"
        )
        
        self.assertIsNotNone(best_model)
        self.assertIsInstance(best_model, MockTransformersWrapper)
        self.assertEqual(best_model.model_name, selected_tf)
        print("Transformer Regression Flow Passed!")

    def test_interface_simulation_unified(self):
        """
        Comprehensive test simulating the Unified 'Novo Treino' Interface flows.
        This test mimics the logic in app.py where user selects a Model Source 
        and the system configures the trainer accordingly.
        """
        print("\nTesting Unified Interface Simulation...")
        
        task_type = 'classification'
        target_col = 'target'
        
        # 0. Common Data Prep (Simulates 'Data' tab and loading)
        processor = AutoMLDataProcessor(target_column=target_col, task_type=task_type)
        X_train_proc, y_train_proc = processor.fit_transform(self.df_class)
        
        trainer = AutoMLTrainer(task_type=task_type)

        # --- SCENARIO 1: User selects "Transformers (HuggingFace)" ---
        print("Scenario 1: Transformers Source")
        # UI Inputs
        ui_model_source = "Transformers (HuggingFace)"
        ui_selected_tf = "bert-base-uncased"
        ui_lr = 5e-5
        ui_epochs = 2
        
        # Logic in app.py
        if ui_model_source == "Transformers (HuggingFace)":
            selected_models = [ui_selected_tf]
            manual_params = {'model_name': ui_selected_tf, 'learning_rate': ui_lr, 'num_train_epochs': ui_epochs}
            custom_models = {}
            
        # Backend Execution
        best_model_tf = trainer.train(
            X_train_proc, y_train_proc, n_trials=1, timeout=5,
            selected_models=selected_models, manual_params=manual_params, custom_models=custom_models,
            experiment_name="sim_interface_transformer"
        )
        
        self.assertIsInstance(best_model_tf, MockTransformersWrapper)
        self.assertEqual(best_model_tf.learning_rate, ui_lr)
        self.assertEqual(best_model_tf.epochs, ui_epochs)
        print("Scenario 1 Passed!")

        # --- SCENARIO 2: User selects "Upload Local (.pkl)" ---
        print("Scenario 2: Upload Source")
        # UI Inputs
        ui_model_source = "Upload Local (.pkl)"
        
        # Mocking the uploaded file content loading
        class UploadedModel(BaseEstimator, ClassifierMixin):
            def fit(self, X, y): return self
            def predict(self, X): return np.zeros(len(X))
        
        uploaded_model_instance = UploadedModel()
        
        # Logic in app.py
        if ui_model_source == "Upload Local (.pkl)":
            # app.py would load the pickle here
            custom_models = {"Uploaded_Model": uploaded_model_instance}
            selected_models = ["Uploaded_Model"]
            manual_params = None
            
        # Backend Execution
        best_model_up = trainer.train(
            X_train_proc, y_train_proc, n_trials=1, timeout=5,
            selected_models=selected_models, custom_models=custom_models,
            experiment_name="sim_interface_upload"
        )
        
        self.assertIsInstance(best_model_up, UploadedModel)
        print("Scenario 2 Passed!")
        
        print("Unified Interface Simulation Passed!")

if __name__ == '__main__':
    unittest.main()
