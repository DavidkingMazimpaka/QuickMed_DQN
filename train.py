import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import CheckpointCallback

from pharmacy_env import PharmacyAccessEnv

def create_env():
    """
    Create and wrap the pharmacy environment
    """
    return PharmacyAccessEnv()

def train_dqn_agent(total_timesteps=50000, log_dir='./logs', model_save_path='./models'):
    """
    Train a DQN agent for the pharmacy access environment
    """
    # Create directories if they don't exist
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)
    
    # Create environment
    env = create_env()
    
    # Create checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,  # Save every 5000 steps
        save_path=model_save_path,
        name_prefix='quickmed_dqn'
    )
    
    # Initialize DQN model
    model = DQN(
        "MlpPolicy",  # Multi-layer perceptron policy
        env,
        verbose=1,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05
    )
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        log_interval=10
    )
    
    # Save the final model
    final_model_path = os.path.join(model_save_path, 'quickmed_dqn_final')
    model.save(final_model_path)
    
    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(
        model, 
        env, 
        n_eval_episodes=10,
        render=False
    )
    
    print(f"Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    return model, mean_reward

def main():
    """
    Main training script
    """
    model, mean_reward = train_dqn_agent()
    
    # Optional: Save training results
    with open('./training_results.txt', 'w') as f:
        f.write(f"Training Complete\nMean Reward: {mean_reward:.2f}")

if __name__ == "__main__":
    main()