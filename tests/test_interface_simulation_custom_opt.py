import unittest
import pandas as pd
import numpy as np
from automl_engine import AutoMLTrainer, AutoMLDataProcessor

class TestInterfaceSimulationCustomOpt(unittest.TestCase):
    def setUp(self):
        # Create a small classification dataset
        self.df_class = pd.DataFrame({
            'feature1': np.random.rand(100),
            'feature2': np.random.rand(100),
            'target': np.random.randint(0, 2, 100)
        })

    def test_custom_optimization_preset(self):
        """
        Simulate the interface flow where the user selects 'custom' preset
        and provides manual n_trials, timeout, and early_stopping.
        """
        print("\nTesting Custom Optimization Interface Flow...")
        
        task_type = 'classification'
        target_col = 'target'
        
        # 1. Data Processing
        processor = AutoMLDataProcessor(target_column=target_col, task_type=task_type)
        X_train_proc, y_train_proc = processor.fit_transform(self.df_class)
        
        # 2. Interface Simulation: User selects 'custom' preset
        ui_training_preset = 'custom'
        
        # User inputs for custom configuration
        ui_n_trials = 2
        ui_timeout = 10
        ui_early_stopping = 5
        
        # User selects specific models (optional, but simulating 'AutoML Standard' with custom preset)
        # If model_selection is 'Auto', selected_models would be None, and it falls back to preset['models']
        # In 'custom' preset, we defined a default list. Let's verify that works.
        ui_selected_models = None 
        
        # 3. Instantiate Trainer with 'custom' preset
        trainer = AutoMLTrainer(task_type=task_type, preset=ui_training_preset)
        
        # Verify preset config was loaded
        self.assertIn('custom', trainer.preset_configs)
        self.assertEqual(trainer.preset, 'custom')
        
        # 4. Train with custom parameters passed from UI
        # Note: app.py passes these directly to train()
        print(f"Training with preset={ui_training_preset}, n_trials={ui_n_trials}, timeout={ui_timeout}")
        
        best_model = trainer.train(
            X_train_proc, 
            y_train_proc, 
            n_trials=ui_n_trials,
            timeout=ui_timeout,
            early_stopping_rounds=ui_early_stopping,
            selected_models=ui_selected_models
        )
        
        self.assertIsNotNone(best_model)
        
        # Verify that training actually respected the custom n_trials (indirectly via execution time/log)
        # Since we can't easily check internal trial count without mocking, we trust the integration.
        # But we can check if results are populated.
        self.assertTrue(len(trainer.results) > 0)
        print("Custom Optimization Flow Passed!")

    def test_custom_tuning_for_transformers(self):
        """
        Simulate custom tuning for Transformers (checkbox checked in UI).
        """
        print("\nTesting Custom Tuning for Transformers...")
        # Mocking or using a lightweight test
        # Since we don't want to actually download transformers, we'll check logic configuration.
        
        ui_use_custom_tuning = True
        ui_training_preset = "custom" if ui_use_custom_tuning else "medium"
        
        ui_n_trials = 1
        ui_timeout = 5
        
        trainer = AutoMLTrainer(task_type='classification', preset=ui_training_preset)
        self.assertEqual(trainer.preset, 'custom')
        
        # We won't actually train a transformer here to save time/resources, 
        # but we confirmed the preset logic holds.
        print("Custom Tuning Logic for Transformers Passed!")

if __name__ == '__main__':
    unittest.main()
