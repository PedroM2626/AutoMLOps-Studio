
import os
import logging
import numpy as np
import pandas as pd
import mlflow
import joblib
from typing import Dict, Any, Optional, Tuple
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    from stable_baselines3 import PPO, DQN, A2C, SAC, TD3
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False


class MLflowRLCallback(BaseCallback):
    """Callback for logging RL training metrics to MLflow."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_training_start(self) -> None:
        pass
        
    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            reward = ep_info['r']
            length = ep_info['l']
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)
            
            mlflow.log_metric("episode_reward", reward, step=self.num_timesteps)
            mlflow.log_metric("episode_length", length, step=self.num_timesteps)
            mlflow.log_metric("mean_reward", np.mean(self.episode_rewards[-100:]), step=self.num_timesteps)
        
        return True


class RLTrainer:
    """Reinforcement Learning trainer for AutoMLOps Studio."""
    
    ALGORITHMS = {
        'ppo': PPO,
        'dqn': DQN,
        'a2c': A2C,
        'sac': SAC,
        'td3': TD3
    }
    
    ALGORITHM_DISPLAY_NAMES = {
        'ppo': 'Proximal Policy Optimization (PPO)',
        'dqn': 'Deep Q-Network (DQN)',
        'a2c': 'Advantage Actor-Critic (A2C)',
        'sac': 'Soft Actor-Critic (SAC)',
        'td3': 'Twin Delayed DDPG (TD3)'
    }
    
    def __init__(
        self,
        env_id: str = 'CartPole-v1',
        algorithm: str = 'ppo',
        total_timesteps: int = 10000,
        policy: str = 'MlpPolicy',
        verbose: int = 1,
        **kwargs
    ):
        if not STABLE_BASELINES_AVAILABLE:
            raise ImportError(
                "Stable Baselines3 and Gymnasium are required for RL. "
                "Install them with: pip install stable-baselines3[extra] gymnasium"
            )
            
        self.env_id = env_id
        self.algorithm = algorithm.lower()
        self.total_timesteps = total_timesteps
        self.policy = policy
        self.verbose = verbose
        self.kwargs = kwargs
        self.model = None
        self.env = None
        self.eval_env = None
        
        if self.algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm: {self.algorithm}. "
                f"Available: {list(self.ALGORITHMS.keys())}"
            )
    
    def _create_env(self, eval_mode: bool = False):
        """Create and wrap the environment."""
        def make_env():
            env = gym.make(self.env_id)
            env = Monitor(env)
            return env
            
        vec_env = DummyVecEnv([make_env])
        if not eval_mode:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)
        return vec_env
    
    def train(self, **kwargs):
        """Train the RL agent."""
        logger.info(f"Starting RL training for {self.env_id} with {self.algorithm}")
        
        self.env = self._create_env(eval_mode=False)
        self.eval_env = self._create_env(eval_mode=True)
        
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path="./tmp/best_model",
            log_path="./tmp/eval_logs",
            eval_freq=1000,
            deterministic=True,
            render=False
        )
        
        mlflow_callback = MLflowRLCallback()
        
        algo_class = self.ALGORITHMS[self.algorithm]
        
        self.model = algo_class(
            self.policy,
            self.env,
            verbose=self.verbose,
            **self.kwargs
        )
        
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=[eval_callback, mlflow_callback],
            **kwargs
        )
        
        logger.info("RL training completed!")
        return self.model
    
    def predict(self, observation, deterministic: bool = True):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(observation, deterministic=deterministic)
    
    def save(self, path: str):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Train first.")
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        
        if self.env is not None:
            self.env.save(f"{path}_vec_normalize.pkl")
            
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, env_id: str = None):
        """Load a trained model."""
        if not STABLE_BASELINES_AVAILABLE:
            raise ImportError("Stable Baselines3 is required.")
            
        trainer = cls(env_id=env_id or 'CartPole-v1')
        
        # Try to load with each algorithm
        for algo_name, algo_class in cls.ALGORITHMS.items():
            try:
                trainer.model = algo_class.load(path)
                trainer.algorithm = algo_name
                break
            except:
                continue
                
        # Try to load VecNormalize
        vec_norm_path = f"{path}_vec_normalize.pkl"
        if os.path.exists(vec_norm_path):
            trainer.env = VecNormalize.load(vec_norm_path, trainer._create_env(eval_mode=False))
            
        return trainer
    
    def evaluate(self, n_eval_episodes: int = 10):
        """Evaluate the trained agent."""
        if self.model is None:
            raise ValueError("Model not trained.")
            
        rewards = []
        episode_lengths = []
        
        eval_env = self._create_env(eval_mode=True)
        
        for _ in range(n_eval_episodes):
            obs = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            done = False
            
            while not done:
                action, _states = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = eval_env.step(action)
                episode_reward += reward[0]
                episode_length += 1
                
                if done:
                    rewards.append(episode_reward)
                    episode_lengths.append(episode_length)
                    
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'mean_episode_length': np.mean(episode_lengths),
            'rewards': rewards,
            'episode_lengths': episode_lengths
        }


def get_available_rl_environments():
    """Get list of available Gymnasium environments."""
    if not STABLE_BASELINES_AVAILABLE:
        return []
        
    common_envs = [
        'CartPole-v1',
        'MountainCar-v0',
        'MountainCarContinuous-v0',
        'Pendulum-v1',
        'Acrobot-v1',
        'LunarLander-v2',
        'LunarLanderContinuous-v2',
        'BipedalWalker-v3',
        'BipedalWalkerHardcore-v3'
    ]
    
    return common_envs

