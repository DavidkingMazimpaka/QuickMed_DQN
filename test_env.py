from stable_baselines3 import DQN
from pharmacy_env import PharmacyEnv

# Load the trained model
model = DQN.load("pharmacy_dqn")

# Initialize environment
env = PharmacyEnv()

# Test the model
obs = env.reset()
for _ in range(20):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
