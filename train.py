import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from pharmacy_env import PharmacyEnv

# Create the environment
env = PharmacyEnv()

# Wrap in DummyVecEnv for compatibility with Stable Baselines
vec_env = DummyVecEnv([lambda: env])

# Optional: Check if the environment follows Gym's API
check_env(env)

# Define callback for early stopping and evaluation
eval_callback = EvalCallback(
    vec_env, 
    best_model_save_path='./logs/', 
    log_path='./logs/', 
    eval_freq=1000, 
    deterministic=True, 
    render=False
)

# Define the PPO model with more robust hyperparameters
model = PPO(
    "MultiInputPolicy",  # Changed to match new observation space
    vec_env,
    learning_rate=1e-3,  # Slightly increased learning rate
    n_steps=2048,        # Large batch for stable updates
    batch_size=64,       
    n_epochs=10,         # Reduced epochs to prevent overfitting
    gamma=0.99,          # Discount factor
    gae_lambda=0.95,     # Generalized Advantage Estimation
    clip_range=0.2,      # PPO clipping
    ent_coef=0.01,       # Entropy coefficient for exploration
    vf_coef=0.5,         # Value function loss coefficient
    verbose=1,
    tensorboard_log="./ppo_pharmacy_tensorboard/"
)

# Train the model
print("Training the model...")
model.learn(
    total_timesteps=200000,  # Increased total training steps
    callback=eval_callback
)
print("Training complete!")

# Save the model
model.save("pharmacy_agent_model")
print("Model saved as 'pharmacy_agent_model'")
env.close()