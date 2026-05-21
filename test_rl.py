
"""Test script for Reinforcement Learning module."""

import os
import sys

# Add project root to path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.engines.reinforcement_learning import (
    RLTrainer,
    get_available_rl_environments,
    STABLE_BASELINES_AVAILABLE
)

def test_rl_basic():
    """Test basic RL functionality."""
    print("=" * 60)
    print("Testing Reinforcement Learning Module")
    print("=" * 60)
    
    if not STABLE_BASELINES_AVAILABLE:
        print("❌ Stable Baselines3 and/or Gymnasium not installed!")
        print("Please install them with: pip install stable-baselines3[extra] gymnasium")
        return
    
    print(f"✅ Stable Baselines3 is available!")
    print(f"\nAvailable environments: {get_available_rl_environments()}")
    
    # Test with CartPole-v1
    print("\n" + "=" * 60)
    print("Testing PPO on CartPole-v1 (small training run)")
    print("=" * 60)
    
    try:
        trainer = RLTrainer(
            env_id='CartPole-v1',
            algorithm='ppo',
            total_timesteps=1000,  # Small number for quick test
            policy='MlpPolicy',
            verbose=1
        )
        
        # Train the agent
        model = trainer.train()
        print("\n✅ Training completed successfully!")
        
        # Evaluate the agent
        print("\nEvaluating trained agent...")
        results = trainer.evaluate(n_eval_episodes=3)
        print(f"Mean reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Mean episode length: {results['mean_episode_length']:.2f}")
        
        # Save and load test
        print("\nTesting save/load functionality...")
        save_path = os.path.join(ROOT_DIR, "tmp", "test_rl_model")
        trainer.save(save_path)
        print(f"✅ Model saved to {save_path}")
        
        loaded_trainer = RLTrainer.load(save_path, env_id='CartPole-v1')
        print("✅ Model loaded successfully!")
        
        # Evaluate loaded model
        print("\nEvaluating loaded model...")
        loaded_results = loaded_trainer.evaluate(n_eval_episodes=2)
        print(f"Loaded model - Mean reward: {loaded_results['mean_reward']:.2f}")
        
        print("\n" + "=" * 60)
        print("🎉 All RL tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error during RL test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_rl_basic()

