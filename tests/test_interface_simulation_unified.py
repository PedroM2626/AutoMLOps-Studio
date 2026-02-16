
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from automl_engine import AutoMLTrainer, TRANSFORMERS_AVAILABLE

class TestInterfaceSimulationUnified(unittest.TestCase):
    
    def setUp(self):
        self.trainer_cls = AutoMLTrainer(task_type='classification')
        self.trainer_reg = AutoMLTrainer(task_type='regression')

    def test_supported_models_unified(self):
        """Test if get_supported_models returns the correct unified list."""
        models_cls = self.trainer_cls.get_supported_models()
        self.assertIn('linear_svc', models_cls)
        self.assertIn('sgd_classifier', models_cls)
        
        # Check if transformers are included (if available or mocked)
        if TRANSFORMERS_AVAILABLE:
            self.assertIn('bert-base-uncased', models_cls)

        models_reg = self.trainer_reg.get_supported_models()
        self.assertIn('sgd_regressor', models_reg)
        self.assertIn('svm', models_reg) # SVR

    def test_linear_svc_params_interface(self):
        """Simulate manual parameter configuration for LinearSVC."""
        schema = self.trainer_cls.get_model_params_schema('linear_svc')
        self.assertIn('C', schema)
        self.assertIn('loss', schema)
        self.assertIn('penalty', schema)
        
        # Simulate user selecting params
        user_params = {
            'C': 0.5,
            'loss': 'hinge',
            'penalty': 'l2'
        }
        
        # Instantiate model
        model = self.trainer_cls._instantiate_model('linear_svc', user_params)
        
        self.assertEqual(model.C, 0.5)
        self.assertEqual(model.loss, 'hinge')
        self.assertEqual(model.penalty, 'l2')
        # Check default dual logic
        self.assertTrue(model.dual) # hinge requires dual=True

    def test_sgd_regressor_params_interface(self):
        """Simulate manual parameter configuration for SGDRegressor."""
        schema = self.trainer_reg.get_model_params_schema('sgd_regressor')
        self.assertIn('sgd_alpha', schema)
        
        # Simulate user selecting params
        user_params = {
            'sgd_alpha': 0.01,
            'sgd_penalty': 'l1'
        }
        
        # Instantiate model
        model = self.trainer_reg._instantiate_model('sgd_regressor', user_params)
        
        self.assertEqual(model.alpha, 0.01)
        self.assertEqual(model.penalty, 'l1')

    def test_svr_params_interface(self):
        """Simulate manual parameter configuration for SVR."""
        # SVR params are usually prefixed with svm_ in previous logic, 
        # but let's check what I implemented.
        # I implemented: k.replace('svm_', '') ... or k in ['C', 'kernel', 'gamma', 'epsilon']
        
        user_params = {
            'C': 10.0,
            'svm_kernel': 'poly',
            'epsilon': 0.2
        }
        
        # Instantiate model
        model = self.trainer_reg._instantiate_model('svm', user_params)
        
        self.assertEqual(model.C, 10.0)
        self.assertEqual(model.kernel, 'poly')
        self.assertEqual(model.epsilon, 0.2)

    @patch('automl_engine.TRANSFORMERS_AVAILABLE', True)
    def test_transformer_schema_and_instantiation(self):
        """Test transformer schema availability and instantiation simulation."""
        # Re-instantiate trainer to pick up mocked TRANSFORMERS_AVAILABLE if needed
        # (Though module level mock might need reload, but let's try just patching where it's used)
        
        # In get_model_params_schema, it checks global TRANSFORMERS_AVAILABLE
        # We need to ensure the module sees True.
        
        with patch('automl_engine.TRANSFORMERS_AVAILABLE', True):
            schema = self.trainer_cls.get_model_params_schema('bert-base-uncased')
            self.assertIn('learning_rate', schema)
            self.assertIn('num_train_epochs', schema)
            
            user_params = {
                'learning_rate': 5e-5,
                'num_train_epochs': 4
            }
            
            # Mock TransformersWrapper to avoid actual download
            with patch('automl_engine.TransformersWrapper') as MockWrapper:
                self.trainer_cls._instantiate_model('bert-base-uncased', user_params)
                
                MockWrapper.assert_called_with(
                    model_name='bert-base-uncased', 
                    task='classification', 
                    epochs=4, 
                    learning_rate=5e-5
                )

    def test_missing_hyperparameters_interface(self):
        """Test if recently added hyperparameters are present in schema."""
        # Random Forest
        rf_schema = self.trainer_cls.get_model_params_schema('random_forest')
        self.assertIn('rf_min_samples_split', rf_schema)
        self.assertIn('rf_min_samples_leaf', rf_schema)
        self.assertIn('rf_max_features', rf_schema)
        self.assertIn('rf_bootstrap', rf_schema)
        
        # XGBoost
        xgb_schema = self.trainer_cls.get_model_params_schema('xgboost')
        self.assertIn('xgb_subsample', xgb_schema)
        self.assertIn('xgb_colsample_bytree', xgb_schema)
        self.assertIn('xgb_gamma', xgb_schema)
        self.assertIn('xgb_min_child_weight', xgb_schema)
        
        # SVM
        svm_schema = self.trainer_cls.get_model_params_schema('svm')
        self.assertIn('degree', svm_schema)
        self.assertIn('coef0', svm_schema)

if __name__ == '__main__':
    unittest.main()
