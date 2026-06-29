
import os
import sys
import tempfile
import pytest
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from unittest.mock import patch

from src.engines.reinforcement_learning import (
    RLTrainer,
    get_available_rl_environments,
    STABLE_BASELINES_AVAILABLE,
    compare_agents,
)

@pytest.fixture(autouse=True)
def mock_mlflow():
    with patch("src.engines.reinforcement_learning.mlflow") as mock:
        yield mock

@pytest.mark.skipif(not STABLE_BASELINES_AVAILABLE, reason="Stable Baselines3 not installed")
def test_rl_trainer_init():
    """Test that RLTrainer initializes correctly."""
    trainer = RLTrainer(
        env_id="CartPole-v1",
        algorithm="ppo",
        total_timesteps=1000,
        policy="MlpPolicy",
        verbose=0,
    )
    assert trainer.env_id == "CartPole-v1"
    assert trainer.algorithm == "ppo"
    assert trainer.total_timesteps == 1000
    assert trainer.model is None


@pytest.mark.skipif(not STABLE_BASELINES_AVAILABLE, reason="Stable Baselines3 not installed")
def test_rl_trainer_train_and_save_load():
    """Test training a small model, saving, and loading it."""
    trainer = RLTrainer(
        env_id="CartPole-v1",
        algorithm="ppo",
        total_timesteps=1000,
        policy="MlpPolicy",
        verbose=0,
    )
    model = trainer.train()
    assert model is not None
    assert trainer.model is not None

    with tempfile.TemporaryDirectory() as temp_dir:
        save_path = os.path.join(temp_dir, "test_rl_agent")
        trainer.save(save_path)

        assert os.path.exists(save_path + ".zip")
        assert os.path.exists(save_path + "_config.yaml")

        loaded_trainer = RLTrainer.load(save_path, env_id="CartPole-v1")
        assert loaded_trainer is not None
        assert loaded_trainer.model is not None


@pytest.mark.skipif(not STABLE_BASELINES_AVAILABLE, reason="Stable Baselines3 not installed")
def test_rl_trainer_evaluate():
    """Test evaluating a trained agent."""
    trainer = RLTrainer(
        env_id="CartPole-v1",
        algorithm="ppo",
        total_timesteps=1000,
        policy="MlpPolicy",
        verbose=0,
    )
    trainer.train()
    results = trainer.evaluate(n_eval_episodes=2)

    assert "mean_reward" in results
    assert "std_reward" in results
    assert "rewards" in results
    assert len(results["rewards"]) == 2


@pytest.mark.skipif(not STABLE_BASELINES_AVAILABLE, reason="Stable Baselines3 not installed")
def test_get_available_rl_environments():
    """Test that get_available_rl_environments returns a list."""
    envs = get_available_rl_environments()
    assert isinstance(envs, list)
    assert len(envs) > 0
    assert "CartPole-v1" in envs


@pytest.mark.skipif(not STABLE_BASELINES_AVAILABLE, reason="Stable Baselines3 not installed")
def test_rl_trainer_wrappers():
    """Test that RLTrainer accepts and uses wrappers."""
    wrappers = [
        {"name": "NormalizeObservation", "params": {}},
        {"name": "NormalizeReward", "params": {}},
    ]
    trainer = RLTrainer(
        env_id="CartPole-v1",
        algorithm="ppo",
        total_timesteps=1000,
        policy="MlpPolicy",
        wrappers=wrappers,
        verbose=0,
    )
    assert len(trainer.wrappers) == 2
    model = trainer.train()
    assert model is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

