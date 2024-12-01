import time
from stable_baselines3 import PPO
from pharmacy_env import PharmacyEnv

# Load the trained model
model = PPO.load("pharmacy_agent_model")

# Initialize the environment
env = PharmacyEnv()

# Reset the environment
observation, _ = env.reset()
done = False
total_reward = 0

print("Starting environment testing...\n")
print("Initial Observation:\n", observation)

# Run the agent for a series of steps using the trained model
for step in range(100):  # Increased max steps
    if done:
        print("Stopping the simulation as the episode is done.")
        break

    # Get action from the trained model
    action, _ = model.predict(observation, deterministic=True)

    # Take the action in the environment
    observation, reward, done, truncated, info = env.step(action)
    total_reward += reward

    # Render the environment
    env.render()

    # Wait for 0.5 seconds to make visualization smoother
    time.sleep(0.5)

    # Print step information
    print(f"Step {step + 1}: Action = {action}, Reward = {reward}, Done = {done}")

    # Stop if the game is done
    if done:
        print("Episode ended!")
        env.close("Congrats !!!")  # Display the completion message
        break

# Final output
print("Testing complete!")
print(f"Total Reward: {total_reward}")
env.close("Simulation ended")  # Display message if the simulation ends