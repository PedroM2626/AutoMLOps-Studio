import os
import logging
from typing import Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class AutoMLOrchestrator:
    """
    Universal orchestrator for AutoMLOps-Studio.
    Encapsulates training orchestration for Classical ML, Vision, and Reinforcement Learning.
    Allows running experiments and managing training state independently of any UI.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def submit_classical_job(self, job_manager) -> str:
        """
        Submit a classical AutoML training job to the job manager (background process).
        
        Args:
            job_manager: An instance of TrainingJobManager.
            
        Returns:
            str: The generated job ID.
        """
        logger.info(f"Submitting classical AutoML job for experiment: {self.config.get('experiment_name')}")
        return job_manager.submit_job(self.config)



    def run_vision_training(self, epoch_callback=None, mask_dir=None, augmentation_config=None, label_csv=None, val_split=0.2, optimizer_name='adam') -> Tuple[Any, Any]:
        """
        Run vision training (Image Classification, Object Detection, etc.) synchronously.
        
        Args:
            epoch_callback: Optional callback function triggered at the end of each epoch.
            mask_dir: Optional directory with segmentation masks.
            augmentation_config: Optional dictionary with image augmentation settings.
            label_csv: Optional path to label CSV file for multi-label tasks.
            val_split: Validation split fraction.
            optimizer_name: Optimizer name ('adam', 'sgd', 'rmsprop').
            
        Returns:
            Tuple[Any, Any]: (trainer_instance, best_model)
        """
        logger.info("Initializing Vision AutoML training...")
        from src.engines.vision import CVAutoMLTrainer
        
        cv_task = self.config.get('task_type', 'image_classification')
        backbone = self.config.get('selected_backbone', 'resnet18')
        epochs = self.config.get('epochs', 5)
        batch_size = self.config.get('batch_size', 32)
        lr = self.config.get('lr', 1e-3)
        dataset_path = self.config.get('dataset_path')
        
        trainer = CVAutoMLTrainer(task_type=cv_task, backbone=backbone)
        
        logger.info(f"Training {backbone} model on dataset: {dataset_path}")
        best_model = trainer.train(
            data_dir=dataset_path,
            n_epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            callback=epoch_callback,
            mask_dir=mask_dir,
            augmentation_config=augmentation_config,
            label_csv=label_csv,
            val_split=val_split,
            optimizer_name=optimizer_name
        )
        return trainer, best_model

    def run_rl_training(self) -> Tuple[Any, Any, Dict[str, Any]]:
        """
        Run Reinforcement Learning training synchronously.
        
        Returns:
            Tuple[Any, Any, Dict[str, Any]]: (trainer_instance, trained_model, evaluation_results)
        """
        logger.info("Initializing Reinforcement Learning training...")
        from src.engines.reinforcement_learning import RLTrainer
        from src.core.data_lake import DataLake
        
        env_id = self.config.get('env_id', 'CartPole-v1')
        algorithm = self.config.get('algorithm', 'ppo')
        total_timesteps = self.config.get('total_timesteps', 10000)
        policy = self.config.get('policy', 'MlpPolicy')
        use_optuna = self.config.get('use_optuna', False)
        optuna_trials = self.config.get('optuna_trials', 20)
        save_trajectories = self.config.get('save_trajectories', False)
        wrappers = self.config.get('wrappers', [])
        custom_env_path = self.config.get('custom_env_path', None)
        data_lake_base_path = self.config.get('data_lake_base_path', None)
        
        data_lake = None
        if save_trajectories and data_lake_base_path:
            data_lake = DataLake(base_path=data_lake_base_path)
            
        trainer = RLTrainer(
            env_id=env_id,
            algorithm=algorithm,
            total_timesteps=total_timesteps,
            policy=policy,
            wrappers=wrappers,
            custom_env_path=custom_env_path,
            verbose=1
        )
        
        # Train the model
        logger.info(f"Training RL agent ({algorithm}) on environment: {env_id}")
        model = trainer.train(
            use_optuna=use_optuna,
            optuna_trials=optuna_trials,
            save_trajectories=save_trajectories,
            data_lake=data_lake
        )
        
        # Evaluate model
        logger.info("Evaluating trained RL agent...")
        eval_results = trainer.evaluate(n_eval_episodes=5)
        
        # Save model
        save_dir = self.config.get('save_dir', os.path.join(os.getcwd(), 'models', 'rl'))
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"rl_agent_{env_id}_{algorithm}")
        trainer.save(save_path)
        logger.info(f"RL agent saved successfully to: {save_path}")
        
        return trainer, model, eval_results
