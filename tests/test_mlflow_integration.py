
import unittest
import pandas as pd
import numpy as np
import mlflow
import os
import shutil
from automl_engine import AutoMLTrainer

class TestMLFlowIntegration(unittest.TestCase):
    def setUp(self):
        # Setup temporary directory for mlruns
        self.test_dir = "test_mlflow_runs"
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
        os.makedirs(self.test_dir)
        
        # Set MLflow tracking URI to a local file for testing
        self.tracking_uri = f"file://{os.path.abspath(self.test_dir)}"
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create dummy data
        self.df = pd.DataFrame({
            'feature1': np.random.rand(20),
            'feature2': np.random.rand(20),
            'target': np.random.randint(0, 2, 20)
        })

    def tearDown(self):
        # Clean up
        mlflow.end_run()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_mlflow_logging(self):
        """Test if AutoMLTrainer logs runs to MLflow."""
        experiment_name = "test_experiment_logging"
        mlflow.set_experiment(experiment_name)
        
        trainer = AutoMLTrainer(task_type='classification')
        
        # Train with a single model and trial to be fast
        trainer.train(
            self.df[['feature1', 'feature2']], 
            self.df['target'],
            n_trials=1,
            timeout=10,
            selected_models=['logistic_regression'],
            experiment_name=experiment_name
        )
        
        # Verify if runs were created
        experiment = mlflow.get_experiment_by_name(experiment_name)
        self.assertIsNotNone(experiment, "Experiment should exist")
        
        runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
        print(f"\nFound {len(runs)} runs in experiment '{experiment_name}'")
        
        self.assertGreater(len(runs), 0, "Should have at least one run logged")
        self.assertIn('params.model_name', runs.columns, "Should log model name")
        self.assertIn('metrics.accuracy', runs.columns, "Should log accuracy metric")

if __name__ == '__main__':
    unittest.main()
