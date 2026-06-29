import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch
from src.core.orchestrator import AutoMLOrchestrator
from src.engines.reinforcement_learning import STABLE_BASELINES_AVAILABLE

def test_orchestrator_classical_submission():
    # Setup mock configuration
    config = {
        'task': 'classification',
        'target': 'target',
        'experiment_name': 'Test_Orchestrator_Experiment',
        'n_trials': 1,
        'timeout': 10,
        'preset': 'test'
    }
    
    # Initialize orchestrator
    orchestrator = AutoMLOrchestrator(config)
    
    # Mock job manager
    mock_job_manager = MagicMock()
    mock_job_manager.submit_job.return_value = "job-1234"
    
    # Submit job
    job_id = orchestrator.submit_classical_job(mock_job_manager)
    
    assert job_id == "job-1234"
    mock_job_manager.submit_job.assert_called_once_with(config)

@pytest.mark.skipif(not STABLE_BASELINES_AVAILABLE, reason="Stable Baselines3 not installed")
def test_orchestrator_rl_training(tmp_path):
    # Setup mock configuration for RL
    config = {
        'env_id': 'CartPole-v1',
        'algorithm': 'ppo',
        'total_timesteps': 50,  # Very small for quick test
        'policy': 'MlpPolicy',
        'save_dir': str(tmp_path)
    }
    
    # Initialize orchestrator
    orchestrator = AutoMLOrchestrator(config)
    
    # Mock mlflow calls to prevent deadlock and API calls
    with patch("src.engines.reinforcement_learning.mlflow") as mock_mlflow:
        trainer, model, eval_results = orchestrator.run_rl_training()
        
        assert trainer is not None
        assert model is not None
        assert "mean_reward" in eval_results
