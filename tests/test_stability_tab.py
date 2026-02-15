import unittest
import pandas as pd
import numpy as np
import os
import shutil
import tempfile
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.ensemble import RandomForestClassifier
from stability_engine import StabilityAnalyzer
from mlops_utils import DataLake

class TestStabilityTab(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for DataLake
        self.test_dir = tempfile.mkdtemp()
        self.data_lake = DataLake(base_path=self.test_dir)
        
        # Create a dummy dataset
        self.df = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': np.random.randint(0, 2, 100)
        })
        
        # Save to DataLake
        self.dataset_name = "test_dataset_stability"
        self.file_path = self.data_lake.save_dataset(self.df, self.dataset_name)
        
        # Load it back to simulate "selecting from data lake"
        versions = self.data_lake.list_versions(self.dataset_name)
        if not versions:
            self.fail("Dataset not saved correctly to DataLake")
            
        self.latest_version = versions[0] # Should be the one we just saved
        
        # Load using DataLake method
        self.loaded_df = self.data_lake.load_version(self.dataset_name, self.latest_version)
        
        # Prepare for StabilityAnalyzer
        self.X = self.loaded_df[['feature1', 'feature2']]
        self.y = self.loaded_df['target']
        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.analyzer = StabilityAnalyzer(self.model, self.X, self.y, task_type='classification', random_state=42)

    def tearDown(self):
        # Clean up temp directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_general_stability_check(self):
        """Test the new General Stability Check feature."""
        print("\nTesting General Stability Check...")
        report = self.analyzer.run_general_stability_check(n_iterations=5)
        
        # Check structure
        self.assertIn('seed_stability', report)
        self.assertIn('split_stability', report)
        self.assertIn('raw_seed', report)
        self.assertIn('raw_split', report)
        
        # Check content types
        self.assertIsInstance(report['seed_stability'], pd.DataFrame)
        self.assertIsInstance(report['split_stability'], pd.DataFrame)
        
        # Check if metrics are calculated
        self.assertFalse(report['seed_stability'].empty)
        self.assertFalse(report['split_stability'].empty)
        
        # Check if raw results have correct iterations
        self.assertEqual(len(report['raw_seed']), 5)
        self.assertEqual(len(report['raw_split']), 5)
        print("General Stability Check Passed!")

    def test_dynamic_hyperparameters_simulation(self):
        """Simulate dynamic hyperparameter selection and stability test."""
        print("\nTesting Dynamic Hyperparameters Simulation...")
        # 1. Simulate UI params
        ui_params = {
            'n_estimators': 20,
            'max_depth': 5,
            'criterion': 'gini'
        }
        
        # 2. Create model with these params (as app.py would)
        model = RandomForestClassifier(**ui_params, random_state=42)
        
        # 3. Run stability test (Hyperparameter Sensitivity)
        analyzer = StabilityAnalyzer(model, self.X, self.y, task_type='classification', random_state=42)
        
        # Vary n_estimators
        param_name = 'n_estimators'
        param_values = [10, 20, 30]
        
        results = analyzer.run_hyperparameter_stability(param_name, param_values)
        
        self.assertIsInstance(results, pd.DataFrame)
        self.assertEqual(len(results), 3)
        self.assertTrue('param_value' in results.columns)
        self.assertListEqual(sorted(results['param_value'].tolist()), [10, 20, 30])
        print("Dynamic Hyperparameters Simulation Passed!")

    def test_split_stability_modes(self):
        """Test different split stability modes available in UI."""
        print("\nTesting Split Stability Modes...")
        # Monte Carlo
        res_mc = self.analyzer.run_stability_test(n_iterations=3, cv_strategy='monte_carlo')
        self.assertEqual(len(res_mc), 3)
        
        # K-Fold
        res_kf = self.analyzer.run_stability_test(n_iterations=3, cv_strategy='kfold')
        self.assertEqual(len(res_kf), 3)
        print("Split Stability Modes Passed!")

if __name__ == '__main__':
    unittest.main()
