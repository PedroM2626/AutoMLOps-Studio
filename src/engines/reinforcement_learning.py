
import os
import logging
import numpy as np
import pandas as pd
import mlflow
import joblib
import yaml
import json
import psutil
import gc
from typing import Dict, Any, Optional, Tuple, List, Callable
from sklearn.base import BaseEstimator
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import gymnasium as gym
    from stable_baselines3 import PPO, DQN, A2C, SAC, TD3
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import (
        BaseCallback, EvalCallback, CheckpointCallback, 
        StopTrainingOnRewardThreshold, CallbackList
    )

    import optuna
    STABLE_BASELINES_AVAILABLE = True
except ImportError:
    STABLE_BASELINES_AVAILABLE = False

try:
    from src.core.data_lake import DataLake
    DATA_LAKE_AVAILABLE = True
except ImportError:
    DATA_LAKE_AVAILABLE = False


class StreamlitRLCallback(BaseCallback):
    """Callback for real‑time metrics logging for Streamlit UI."""
    
    def __init__(self, verbose=0, save_trajectories: bool = False, data_lake: Optional[DataLake] = None, env_id: str = "unknown"):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps_history = []
        self.loss_history = []
        self.metrics = {
            'timesteps': [],
            'episode_reward': [],
            'episode_length': [],
            'mean_reward_100': [],
            'actor_loss': [],
            'critic_loss': [],
            'memory_usage': []
        }
        self.save_trajectories = save_trajectories
        self.data_lake = data_lake
        self.env_id = env_id
        self.trajectories = []
        self.last_obs = None
        
    def _on_training_start(self) -> None:
        self.start_time = datetime.now()
        self.last_obs = self.training_env.reset() if hasattr(self, 'training_env') else None
        
    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            reward = ep_info['r']
            length = ep_info['l']
            self.episode_rewards.append(reward)
            self.episode_lengths.append(length)
            
            self.metrics['timesteps'].append(self.num_timesteps)
            self.metrics['episode_reward'].append(reward)
            self.metrics['episode_length'].append(length)
            self.metrics['mean_reward_100'].append(np.mean(self.episode_rewards[-100:]))
            
            process = psutil.Process(os.getpid())
            self.metrics['memory_usage'].append(process.memory_info().rss / 1024 / 1024)
            
            mlflow.log_metric("episode_reward", reward, step=self.num_timesteps)
            mlflow.log_metric("episode_length", length, step=self.num_timesteps)
            mlflow.log_metric("mean_reward_100", np.mean(self.episode_rewards[-100:]), step=self.num_timesteps)
        
        if self.save_trajectories:
            try:
                if self.last_obs is not None:
                    actions, _states = self.model.predict(self.last_obs, deterministic=False)
                    new_obs, rewards, dones, infos = self.training_env.step(actions)
                    
                    for i in range(len(self.last_obs)):
                        trajectory = {
                            'timestep': self.num_timesteps,
                            'observation': self.last_obs[i].tolist() if isinstance(self.last_obs[i], np.ndarray) else self.last_obs[i],
                            'action': actions[i].tolist() if isinstance(actions[i], np.ndarray) else actions[i],
                            'reward': float(rewards[i]),
                            'next_observation': new_obs[i].tolist() if isinstance(new_obs[i], np.ndarray) else new_obs[i],
                            'done': bool(dones[i])
                        }
                        self.trajectories.append(trajectory)
                    
                    self.last_obs = new_obs
            except Exception as e:
                logger.debug(f"Failed to collect trajectory: {e}")
        
        return True
        
    def _on_training_end(self) -> None:
        if self.save_trajectories and self.data_lake and len(self.trajectories) > 0:
            try:
                df_trajectories = pd.DataFrame(self.trajectories)
                dataset_name = f"rl_trajectories_{self.env_id}"
                file_name = f"trajectories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                self.data_lake.save_dataframe(df_trajectories, dataset_name, file_name)
                logger.info(f"Trajectories saved to Data Lake: {dataset_name}/{file_name}")
                mlflow.log_artifact(str(self.data_lake.base_path / dataset_name / file_name))
            except Exception as e:
                logger.warning(f"Failed to save trajectories: {e}")


class RLTrainer:
    """Reinforcement Learning trainer for AutoMLOps Studio with enhanced features."""
    
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
    
    DEFAULT_HYPERPARAMS = {
        'ppo': {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'n_steps': 2048,
            'batch_size': 64,
            'clip_range': 0.2
        },
        'dqn': {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'buffer_size': 1_000_000,
            'learning_starts': 50_000,
            'target_update_interval': 10_000
        },
        'a2c': {
            'learning_rate': 7e-4,
            'gamma': 0.99,
            'n_steps': 5
        },
        'sac': {
            'learning_rate': 3e-4,
            'gamma': 0.99,
            'buffer_size': 1_000_000,
            'tau': 0.005
        },
        'td3': {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'buffer_size': 1_000_000,
            'tau': 0.005
        }
    }
    
    HYPERPARAM_SPACES = {
        'ppo': {
            'learning_rate': ('float', 1e-5, 1e-2, True),
            'gamma': ('float', 0.9, 0.999, False),
            'n_steps': ('int', 64, 4096, False),
            'batch_size': ('int', 8, 256, False),
            'clip_range': ('float', 0.1, 0.4, False)
        },
        'dqn': {
            'learning_rate': ('float', 1e-5, 1e-2, True),
            'gamma': ('float', 0.9, 0.999, False),
            'buffer_size': ('int', 10_000, 2_000_000, True),
            'learning_starts': ('int', 1000, 100_000, False),
            'target_update_interval': ('int', 1000, 50_000, False)
        }
    }
    
    def __init__(
        self,
        env_id: str = 'CartPole-v1',
        algorithm: str = 'ppo',
        total_timesteps: int = 10000,
        policy: str = 'MlpPolicy',
        verbose: int = 1,
        custom_env_path: Optional[str] = None,
        wrappers: Optional[List[Dict]] = None,
        **kwargs
    ):
        if not STABLE_BASELINES_AVAILABLE:
            raise ImportError(
                "Stable Baselines3 and Gymnasium are required for RL. "
                "Install them with: pip install stable-baselines3[extra] gymnasium optuna psutil pyyaml"
            )
            
        self.env_id = env_id
        self.algorithm = algorithm.lower()
        self.total_timesteps = total_timesteps
        self.policy = policy
        self.verbose = verbose
        self.custom_env_path = custom_env_path
        self.wrappers = wrappers or []
        self.kwargs = kwargs
        self.model = None
        self.env = None
        self.eval_env = None
        self.callback = None
        self.training_history = []
        
        if self.algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Unknown algorithm: {self.algorithm}. "
                f"Available: {list(self.ALGORITHMS.keys())}"
            )
    
    def _create_env(self, eval_mode: bool = False, custom_env_class=None):
        """Create and wrap the environment with optional wrappers."""
        
        def make_env():
            if custom_env_class:
                env = custom_env_class()
            else:
                env = gym.make(self.env_id)
            
            for wrapper_config in self.wrappers:
                wrapper_name = wrapper_config.get('name')
                wrapper_params = wrapper_config.get('params', {})
                
                if wrapper_name == 'FrameStack':
                    from gymnasium.wrappers import FrameStackObservation as FrameStack
                    env = FrameStack(env, **wrapper_params)
                elif wrapper_name == 'GrayScaleObservation':
                    from gymnasium.wrappers import GrayScaleObservation
                    env = GrayScaleObservation(env, **wrapper_params)
                elif wrapper_name == 'ResizeObservation':
                    from gymnasium.wrappers import ResizeObservation
                    env = ResizeObservation(env, **wrapper_params)
            
            env = Monitor(env)
            return env
            
        vec_env = DummyVecEnv([make_env])
        
        if not eval_mode:
            normalize_obs = any(w.get('name') == 'NormalizeObservation' for w in self.wrappers)
            normalize_rew = any(w.get('name') == 'NormalizeReward' for w in self.wrappers)
            
            if normalize_obs or normalize_rew:
                vec_env = VecNormalize(
                    vec_env, 
                    norm_obs=normalize_obs, 
                    norm_reward=normalize_rew
                )
                
        return vec_env
    
    def _load_custom_env(self, env_path: str):
        """Dynamically load a custom environment from a Python file."""
        import importlib.util
        
        spec = importlib.util.spec_from_file_location("custom_env", env_path)
        custom_env_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(custom_env_module)
        
        env_classes = [
            cls for name, cls in custom_env_module.__dict__.items()
            if isinstance(cls, type) and issubclass(cls, gym.Env) and cls != gym.Env
        ]
        
        if len(env_classes) == 0:
            raise ValueError(f"No gym.Env subclass found in {env_path}")
        elif len(env_classes) > 1:
            logger.warning(f"Multiple env classes found, using first one: {env_classes[0].__name__}")
            
        return env_classes[0]
    
    def train(self, use_optuna: bool = False, optuna_trials: int = 20, save_trajectories: bool = False, data_lake: Optional[DataLake] = None, **kwargs):
        """Train the RL agent with optional Optuna hyperparameter tuning."""
        logger.info(f"Starting RL training for {self.env_id} with {self.algorithm}")
        
        custom_env_class = None
        if self.custom_env_path and os.path.exists(self.custom_env_path):
            custom_env_class = self._load_custom_env(self.custom_env_path)
        
        self.env = self._create_env(eval_mode=False, custom_env_class=custom_env_class)
        self.eval_env = self._create_env(eval_mode=True, custom_env_class=custom_env_class)
        
        if use_optuna:
            return self._train_with_optuna(optuna_trials=optuna_trials, **kwargs)
        
        eval_callback = EvalCallback(
            self.eval_env,
            best_model_save_path="./tmp/best_model",
            log_path="./tmp/eval_logs",
            eval_freq=1000,
            deterministic=True,
            render=False,
            verbose=self.verbose
        )
        
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path="./tmp/checkpoints",
            name_prefix=f"rl_model_{self.algorithm}",
            verbose=self.verbose
        )
        
        self.callback = StreamlitRLCallback(
            verbose=self.verbose,
            save_trajectories=save_trajectories,
            data_lake=data_lake,
            env_id=self.env_id
        )
        
        callback_list = CallbackList([eval_callback, checkpoint_callback, self.callback])
        
        algo_class = self.ALGORITHMS[self.algorithm]
        all_hyperparams = {**self.DEFAULT_HYPERPARAMS.get(self.algorithm, {}), **self.kwargs}
        
        self.model = algo_class(
            self.policy,
            self.env,
            verbose=self.verbose,
            **all_hyperparams
        )
        
        config_dict = {
            'env_id': self.env_id,
            'algorithm': self.algorithm,
            'policy': self.policy,
            'total_timesteps': self.total_timesteps,
            'hyperparameters': all_hyperparams,
            'wrappers': self.wrappers,
            'timestamp': datetime.now().isoformat()
        }
        
        os.makedirs("tmp", exist_ok=True)
        with open("tmp/rl_config.yaml", "w") as f:
            yaml.dump(config_dict, f)
        
        with open("tmp/rl_config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
            
        mlflow.log_artifact("tmp/rl_config.yaml")
        mlflow.log_artifact("tmp/rl_config.json")
        
        for key, value in all_hyperparams.items():
            mlflow.log_param(key, value)
        
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=callback_list,
            **kwargs
        )
        
        logger.info("RL training completed!")
        return self.model
    
    def _objective(self, trial):
        """Optuna objective function for hyperparameter optimization."""
        import optuna
        hp_space = self.HYPERPARAM_SPACES.get(self.algorithm, {})
        params = {}
        
        for name, (hp_type, min_val, max_val, log_scale) in hp_space.items():
            if hp_type == 'float':
                params[name] = trial.suggest_float(name, min_val, max_val, log=log_scale)
            elif hp_type == 'int':
                params[name] = trial.suggest_int(name, min_val, max_val, log=log_scale)
        
        algo_class = self.ALGORITHMS[self.algorithm]
        model = algo_class(self.policy, self.env, verbose=0, **params)
        
        eval_callback = EvalCallback(
            self.eval_env,
            n_eval_episodes=3,
            eval_freq=5000,
            deterministic=True,
            render=False,
            verbose=0
        )
        
        model.learn(total_timesteps=min(50000, self.total_timesteps // 2), callback=eval_callback)
        
        mean_reward = np.mean(eval_callback.best_mean_reward) if eval_callback.best_mean_reward.size > 0 else -np.inf
        
        return mean_reward
    
    def _train_with_optuna(self, optuna_trials: int = 20, **kwargs):
        """Train using Optuna for hyperparameter optimization."""
        import optuna
        logger.info(f"Starting Optuna hyperparameter optimization with {optuna_trials} trials")
        
        study = optuna.create_study(direction='maximize', study_name=f"rl_{self.env_id}_{self.algorithm}")
        study.optimize(self._objective, n_trials=optuna_trials, show_progress_bar=True)
        
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best reward: {study.best_value}")
        logger.info(f"Best hyperparameters: {study.best_params}")
        
        for key, value in study.best_params.items():
            mlflow.log_param(f"best_{key}", value)
        mlflow.log_metric("best_reward", study.best_value)
        
        self.kwargs.update(study.best_params)
        
        return self.train(use_optuna=False, **kwargs)
    
    def predict(self, observation, deterministic: bool = True):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(observation, deterministic=deterministic)
    
    def save(self, path: str):
        """Save the trained model and config."""
        if self.model is None:
            raise ValueError("No model to save. Train first.")
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        
        if self.env and hasattr(self.env, 'save'):
            self.env.save(f"{path}_vec_normalize.pkl")
            
        config_dict = {
            'env_id': self.env_id,
            'algorithm': self.algorithm,
            'policy': self.policy,
            'wrappers': self.wrappers
        }
        
        with open(f"{path}_config.yaml", "w") as f:
            yaml.dump(config_dict, f)
            
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str, env_id: str = None):
        """Load a trained model."""
        if not STABLE_BASELINES_AVAILABLE:
            raise ImportError("Stable Baselines3 is required.")
            
        config_path = f"{path}_config.yaml"
        config = {}
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                
        trainer = cls(
            env_id=env_id or config.get('env_id', 'CartPole-v1'),
            algorithm=config.get('algorithm', 'ppo'),
            policy=config.get('policy', 'MlpPolicy'),
            wrappers=config.get('wrappers', [])
        )
        
        for algo_name, algo_class in cls.ALGORITHMS.items():
            try:
                trainer.model = algo_class.load(path)
                trainer.algorithm = algo_name
                break
            except:
                continue
                
        vec_norm_path = f"{path}_vec_normalize.pkl"
        if os.path.exists(vec_norm_path):
            trainer.env = VecNormalize.load(vec_norm_path, trainer._create_env(eval_mode=False))
            
        return trainer
    
    def evaluate(self, n_eval_episodes: int = 10):
        """Evaluate the trained agent and return detailed metrics."""
        if self.model is None:
            raise ValueError("Model not trained.")
            
        rewards = []
        episode_lengths = []
        
        custom_env_class = None
        if self.custom_env_path and os.path.exists(self.custom_env_path):
            custom_env_class = self._load_custom_env(self.custom_env_path)
            
        eval_env = self._create_env(eval_mode=True, custom_env_class=custom_env_class)
        
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
            'min_reward': np.min(rewards),
            'max_reward': np.max(rewards),
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


try:
    import d3rlpy
    D3RLPY_AVAILABLE = True
except ImportError:
    D3RLPY_AVAILABLE = False


class OfflineRLTrainer:
    """Offline Reinforcement Learning trainer using d3rlpy (BCQ, CQL, etc.)."""
    
    ALGORITHMS = {
        'bcq': 'BCQ',
        'cql': 'CQL',
        'td3_bc': 'TD3PlusBC',
        'iwbc': 'IWBC'
    }
    
    ALGORITHM_DISPLAY_NAMES = {
        'bcq': 'Batch-Constrained Q-Learning (BCQ)',
        'cql': 'Conservative Q-Learning (CQL)',
        'td3_bc': 'TD3 Plus Behavior Cloning (TD3+BC)',
        'iwbc': 'Importance Weighted Behavior Cloning (IWBC)'
    }
    
    def __init__(
        self,
        algorithm: str = 'bcq',
        observation_shape: Optional[Tuple[int, ...]] = None,
        action_size: Optional[int] = None,
        action_scaler: Optional[str] = None,
        **kwargs
    ):
        if not D3RLPY_AVAILABLE:
            raise ImportError(
                "d3rlpy is required for Offline RL. "
                "Install it with: pip install d3rlpy==2.3.0"
            )
            
        self.algorithm = algorithm.lower()
        self.observation_shape = observation_shape
        self.action_size = action_size
        self.action_scaler = action_scaler
        self.kwargs = kwargs
        self.model = None
        
        if self.algorithm not in self.ALGORITHMS:
            raise ValueError(
                f"Unknown offline RL algorithm: {self.algorithm}. "
                f"Available: {list(self.ALGORITHMS.keys())}"
            )
            
    def _prepare_dataset(self, df_trajectories: pd.DataFrame):
        """Prepare d3rlpy dataset from trajectories DataFrame."""
        observations = []
        actions = []
        rewards = []
        next_observations = []
        terminals = []
        
        for _, row in df_trajectories.iterrows():
            observations.append(np.array(row['observation']))
            actions.append(np.array(row['action']))
            rewards.append(row['reward'])
            next_observations.append(np.array(row['next_observation']))
            terminals.append(1.0 if row['done'] else 0.0)
            
        observations = np.array(observations)
        actions = np.array(actions)
        rewards = np.array(rewards)
        next_observations = np.array(next_observations)
        terminals = np.array(terminals)
        
        if self.action_scaler:
            from d3rlpy.preprocessing import MinMaxActionScaler
            action_scaler = MinMaxActionScaler()
            action_scaler.fit(actions)
        else:
            action_scaler = None
            
        dataset = d3rlpy.dataset.MDPDataset(
            observations=observations,
            actions=actions,
            rewards=rewards,
            terminals=terminals,
            action_scaler=action_scaler
        )
        
        return dataset
        
    def train(
        self,
        df_trajectories: pd.DataFrame,
        n_epochs: int = 100,
        n_steps_per_epoch: int = 1000,
        **kwargs
    ):
        """Train the offline RL agent on trajectories."""
        logger.info(f"Starting Offline RL training with {self.algorithm}")
        
        dataset = self._prepare_dataset(df_trajectories)
        
        algo_class = getattr(d3rlpy.algos, self.ALGORITHMS[self.algorithm])
        
        if self.observation_shape is None:
            self.observation_shape = dataset.observation_shape
        if self.action_size is None:
            self.action_size = dataset.action_size
            
        self.model = algo_class(
            observation_shape=self.observation_shape,
            action_size=self.action_size,
            **self.kwargs
        )
        
        self.model.fit(
            dataset,
            n_epochs=n_epochs,
            n_steps_per_epoch=n_steps_per_epoch,
            **kwargs
        )
        
        logger.info("Offline RL training completed!")
        return self.model
        
    def predict(self, observation, deterministic: bool = True):
        """Make predictions using the trained offline RL model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        if isinstance(observation, list) or isinstance(observation, np.ndarray):
            observation = np.array(observation)
        if len(observation.shape) == 1:
            observation = observation.reshape(1, -1)
            
        return self.model.predict(observation)
        
    def save(self, path: str):
        """Save the trained offline RL model."""
        if self.model is None:
            raise ValueError("No model to save. Train first.")
            
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info(f"Offline RL model saved to {path}")
        
    @classmethod
    def load(cls, path: str, algorithm: str = 'bcq'):
        """Load a trained offline RL model."""
        if not D3RLPY_AVAILABLE:
            raise ImportError("d3rlpy is required.")
            
        trainer = cls(algorithm=algorithm)
        algo_class = getattr(d3rlpy.algos, cls.ALGORITHMS[algorithm])
        trainer.model = algo_class.load(path)
        return trainer


def compare_agents(agents: List[RLTrainer], env_id: str, n_eval_episodes: int = 10):
    """Compare multiple agents on the same environment."""
    from scipy import stats
    
    results = []
    
    for i, agent in enumerate(agents):
        eval_results = agent.evaluate(n_eval_episodes=n_eval_episodes)
        results.append({
            'agent_index': i,
            'algorithm': agent.algorithm,
            **eval_results
        })
        
    comparisons = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            t_stat, p_value = stats.ttest_ind(
                results[i]['rewards'],
                results[j]['rewards']
            )
            comparisons.append({
                'agent1': i,
                'agent2': j,
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            })
            
    return results, comparisons

