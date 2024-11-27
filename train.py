import gym
import numpy as np
from stable_baselines3 import DQN
from pharmacy_env import PharmacyEnv

# Initialize custom environment
env = PharmacyEnv()

# Initialize DQN model
model = DQN("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=10000)

# Save the trained model
model.save("pharmacy_dqn")

# Test the trained model
obs = env.reset()
for _ in range(20):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
