import gym
import numpy as np
from stable_baselines3 import DQN
from pharmacy_env import PharmacyEnv
import matplotlib.pyplot as plt

def train_agent():
    # Create environment
    env = PharmacyEnv()
    
    # Initialize DQN model with pharmacy-specific hyperparameters
    model = DQN(
        "MlpPolicy",  # Multi-layer perceptron policy
        env, 
        verbose=1,
        learning_rate=0.001,
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
    
    # Train the agent
    model.learn(total_timesteps=50000)
    
    # Save the trained model
    model.save("pharmacy_logistics_agent")
    
    # Close the environment
    env.close()

def test_trained_agent():
    # Create environment
    env = PharmacyEnv()
    
    # Load trained model
    model = DQN.load("pharmacy_logistics_agent")
    
    # Tracking variables for analysis
    total_rewards = []
    
    # Run multiple episodes
    for episode in range(10):
        obs = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            env.render()
        
        total_rewards.append(episode_reward)
        print(f"Episode {episode + 1} Reward: {episode_reward}")
    
    # Visualize rewards
    plt.figure(figsize=(10, 5))
    plt.plot(total_rewards, marker='o')
    plt.title('Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_agent()
    test_trained_agent()
