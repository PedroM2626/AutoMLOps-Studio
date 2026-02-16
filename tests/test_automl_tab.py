import unittest
import sys
import os

print("Starting test_automl_tab.py...")
import pandas as pd
import numpy as np
import os
import sys
import shutil
import tempfile

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock transformers as missing to avoid heavy imports/crashes during test
sys.modules['transformers'] = None

print("Importing automl_engine...")
from automl_engine import AutoMLDataProcessor, AutoMLTrainer
print("Import successful!")

class TestAutoMLTab(unittest.TestCase):
    def setUp(self):
        # Create dummy classification data
        self.df_class = pd.DataFrame({
            'feature1': np.random.rand(50),
            'feature2': np.random.rand(50),
            'category': np.random.choice(['A', 'B'], 50),
            'target': np.random.randint(0, 2, 50)
        })
        
        # Create dummy regression data
        self.df_reg = pd.DataFrame({
            'feature1': np.random.rand(50),
            'feature2': np.random.rand(50),
            'category': np.random.choice(['X', 'Y'], 50),
            'target': np.random.rand(50) * 100
        })

    def test_classification_flow(self):
        """Simulate the AutoML flow for a classification task."""
        print("\nTesting AutoML Classification Flow...")
        
        task_type = 'classification'
        target_col = 'target'
        
        # 1. Data Processing
        processor = AutoMLDataProcessor(target_column=target_col, task_type=task_type)
        X_train_proc, y_train_proc = processor.fit_transform(self.df_class)
        
        self.assertIsNotNone(X_train_proc)
        self.assertIsNotNone(y_train_proc)
        
        # 2. Training
        print("Starting training...")
        trainer = AutoMLTrainer(task_type=task_type)
        
        # Use a very small number of trials and timeout for speed
        n_trials = 1
        timeout = 10 
        
        # Explicitly select models to run (simulating "Manual" selection in UI)
        # and to keep tests fast
        selected_models = ['logistic_regression', 'random_forest']

        # Mock callback
        callback_called = False
        def mock_callback(trial, score, full_name, dur, metrics=None):
            nonlocal callback_called
            callback_called = True
            print(f"Callback: Trial {trial.number}, Score: {score}")

        print("Calling trainer.train()...")
        best_model = trainer.train(
            X_train_proc, 
            y_train_proc, 
            n_trials=n_trials,
            timeout=timeout,
            callback=mock_callback,
            experiment_name="test_automl_classification",
            selected_models=selected_models
        )
        print("Training finished!")
        
        self.assertIsNotNone(best_model)
        self.assertTrue(callback_called, "Callback should have been called")
        
        # 3. Evaluation
        metrics, y_pred = trainer.evaluate(X_train_proc, y_train_proc)
        
        self.assertIn('accuracy', metrics)
        print(f"Classification Metrics: {metrics}")
        print("AutoML Classification Flow Passed!")

    def test_regression_flow(self):
        """Simulate the AutoML flow for a regression task."""
        print("\nTesting AutoML Regression Flow...")
        
        task_type = 'regression'
        target_col = 'target'
        
        # 1. Data Processing
        processor = AutoMLDataProcessor(target_column=target_col, task_type=task_type)
        X_train_proc, y_train_proc = processor.fit_transform(self.df_reg)
        
        self.assertIsNotNone(X_train_proc)
        self.assertIsNotNone(y_train_proc)
        
        # 2. Training
        trainer = AutoMLTrainer(task_type=task_type)
        
        n_trials = 1
        timeout = 10
        selected_models = ['linear_regression', 'random_forest']
        
        best_model = trainer.train(
            X_train_proc, 
            y_train_proc, 
            n_trials=n_trials,
            timeout=timeout,
            experiment_name="test_automl_regression",
            selected_models=selected_models
        )
        
        self.assertIsNotNone(best_model)
        
        # 3. Evaluation
        metrics, y_pred = trainer.evaluate(X_train_proc, y_train_proc)
        
        self.assertIn('r2', metrics)
        print(f"Regression Metrics: {metrics}")
        print("AutoML Regression Flow Passed!")

    def test_clustering_flow(self):
        """Simulate the AutoML flow for a clustering task."""
        print("\nTesting AutoML Clustering Flow...")
        
        task_type = 'clustering'
        
        # 1. Data Processing
        processor = AutoMLDataProcessor(target_column=None, task_type=task_type)
        X_train_proc, _ = processor.fit_transform(self.df_class.drop(columns=['target']))
        
        self.assertIsNotNone(X_train_proc)
        
        # 2. Training
        trainer = AutoMLTrainer(task_type=task_type)
        
        n_trials = 1
        timeout = 10
        selected_models = ['kmeans']
        
        best_model = trainer.train(
            X_train_proc, 
            y_train=None,
            n_trials=n_trials,
            timeout=timeout,
            experiment_name="test_automl_clustering",
            selected_models=selected_models
        )
        
        self.assertIsNotNone(best_model)
        
        # 3. Evaluation
        metrics, y_pred = trainer.evaluate(X_train_proc, None)
        
        self.assertIn('silhouette', metrics)
        print(f"Clustering Metrics: {metrics}")
        print("AutoML Clustering Flow Passed!")

    def test_validation_strategies(self):
        """Test different validation strategies (Holdout, CV, Stratified)."""
        print("\nTesting Validation Strategies...")
        
        task_type = 'classification'
        target_col = 'target'
        processor = AutoMLDataProcessor(target_column=target_col, task_type=task_type)
        X_train_proc, y_train_proc = processor.fit_transform(self.df_class)
        
        trainer = AutoMLTrainer(task_type=task_type)
        selected_models = ['logistic_regression'] # Fast model
        n_trials = 1
        timeout = 5
        
        strategies = [
            ('holdout', {'test_size': 0.2}),
            ('cv', {'folds': 2}),
            ('stratified_cv', {'folds': 2})
        ]
        
        for strategy, params in strategies:
            print(f"Testing strategy: {strategy} with params: {params}")
            best_model = trainer.train(
                X_train_proc, 
                y_train_proc, 
                n_trials=n_trials,
                timeout=timeout,
                selected_models=selected_models,
                validation_strategy=strategy,
                validation_params=params,
                experiment_name=f"test_val_{strategy}"
            )
            self.assertIsNotNone(best_model)
            print(f"Strategy {strategy} passed!")
        
        # Write success flag inside the test is not ideal as order is not guaranteed, 
        # but for now I'll rely on main block modification.

if __name__ == '__main__':
    result = unittest.main(exit=False)
    if result.result.wasSuccessful():
        with open("test_status.txt", "w") as f:
            f.write("TESTS COMPLETED SUCCESSFULLY")
