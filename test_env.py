import gym
import numpy as np
from stable_baselines3 import DQN
from pharmacy_env import PharmacyEnv

def test_environment():
    """
    Comprehensive testing of the Pharmacy Environment
    """
    # Create environment
    env = PharmacyEnv()

    # Test reset method
    print("Testing reset() method:")
    initial_state = env.reset()
    print("Initial state shape:", initial_state.shape)
    print("Initial state:", initial_state)

    # Test action space
    print("\nTesting action space:")
    print("Action space type:", type(env.action_space))
    print("Number of possible actions:", env.action_space.n)

    # Test observation space
    print("\nTesting observation space:")
    print("Observation space shape:", env.observation_space.shape)
    print("Observation space low:", env.observation_space.low)
    print("Observation space high:", env.observation_space.high)

    # Test step method with random actions
    print("\nTesting step() method:")
    for _ in range(3):
        # Sample a random action
        action = env.action_space.sample()
        print(f"\nAction: {action}")
        
        # Take a step
        next_state, reward, done, info = env.step(action)
        
        print("Next state shape:", next_state.shape)
        print("Reward:", reward)
        print("Done:", done)
        print("Info:", info)

def validate_trained_model():
    """
    Validate a trained model's performance
    """
    try:
        # Load trained model
        model = DQN.load("pharmacy_logistics_agent")
        env = PharmacyEnv()

        # Run multiple episodes
        total_rewards = []
        for episode in range(5):
            obs = env.reset()
            episode_reward = 0
            done = False

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _ = env.step(action)
                episode_reward += reward

            total_rewards.append(episode_reward)
            print(f"Episode {episode + 1} Reward: {episode_reward}")

        # Basic performance check
        print("\nPerformance Analysis:")
        print(f"Average Reward: {np.mean(total_rewards)}")
        print(f"Reward Variance: {np.var(total_rewards)}")

    except FileNotFoundError:
        print("Trained model not found. Please train the model first.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    print("=== Environment Validation ===")
    test_environment()
    
    print("\n=== Trained Model Validation ===")
    validate_trained_model()

if __name__ == "__main__":
    main()