import os
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from termcolor import colored

from pharmacy_env import PharmacyAccessEnv

class QuickMedModelPlayer:
    def __init__(self, model_path='./models/quickmed_dqn_final.zip'):
        """
        Initialize the model player with enhanced visualization and logging
        """
        self.model_path = model_path
        self.model = None
        self.env = None
        
    def load_model(self):
        """
        Load the trained model with enhanced error handling
        """
        try:
            if not os.path.exists(self.model_path):
                print(colored("❌ Model not found!", "red"))
                print(colored("Suggestion: Run train.py first to generate a model.", "yellow"))
                return False
            
            self.model = DQN.load(self.model_path)
            print(colored(f"✅ Model successfully loaded from {self.model_path}", "green"))
            return True
        
        except Exception as e:
            print(colored(f"❌ Error loading model: {e}", "red"))
            return False
    
    def play_episode_with_visualization(self, render=True, verbose=True):
        """
        Play a single episode with detailed visualization and logging
        """
        if not self.model:
            print(colored("Model not loaded. Call load_model() first.", "red"))
            return None
        
        # Create a new environment for each episode
        env = PharmacyAccessEnv()
        obs = env.reset()
        total_reward = 0
        done = False
        steps = 0
        
        episode_log = []
        step_details = []
        
        # Visualization setup
        plt.figure(figsize=(12, 6))
        plt.ion()  # Turn on interactive mode
        
        while not done:
            steps += 1
            
            # Predict action
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            total_reward += reward
            
            # Detailed action names
            action_names = [
                "Travel to Pharmacy", 
                "Check Stock", 
                "Pickup Medication", 
                "Travel to Patient", 
                "Skip Episode"
            ]
            
            # Log step details
            step_info = {
                'step': steps,
                'action': action_names[action],
                'reward': reward,
                'cumulative_reward': total_reward
            }
            step_details.append(step_info)
            
            # Verbose logging
            if verbose:
                print(colored(f"Step {steps}:", "cyan"))
                print(colored(f"  Action: {action_names[action]}", "blue"))
                print(colored(f"  Immediate Reward: {reward:.2f}", "green"))
                print(colored(f"  Cumulative Reward: {total_reward:.2f}", "magenta"))
                print("-" * 40)
            
            # Optional: Add a small delay to visualize steps
            if verbose:
                time.sleep(0.5)
            
            if render:
                env.render()
        
        # Visualization of episode performance
        self._visualize_episode_performance(step_details)
        
        # Final episode summary
        if verbose:
            status = "✅ Successful" if total_reward > 0 else "❌ Failed"
            print(colored(f"\n{status} Episode", "green" if total_reward > 0 else "red"))
            print(colored(f"Total Steps: {steps}", "blue"))
            print(colored(f"Total Reward: {total_reward:.2f}", "magenta"))
        
        return {
            'total_reward': total_reward,
            'steps': steps,
            'step_details': step_details
        }
    
    def _visualize_episode_performance(self, step_details):
        """
        Create a comprehensive visualization of episode performance
        """
        plt.figure(figsize=(15, 10))
        
        # Reward progression subplot
        plt.subplot(2, 2, 1)
        rewards = [step['reward'] for step in step_details]
        plt.bar(range(1, len(rewards) + 1), rewards)
        plt.title('Step Rewards')
        plt.xlabel('Step Number')
        plt.ylabel('Immediate Reward')
        
        # Cumulative reward subplot
        plt.subplot(2, 2, 2)
        cumulative_rewards = [step['cumulative_reward'] for step in step_details]
        plt.plot(range(1, len(cumulative_rewards) + 1), cumulative_rewards, marker='o')
        plt.title('Cumulative Reward Progression')
        plt.xlabel('Step Number')
        plt.ylabel('Cumulative Reward')
        
        # Action distribution subplot
        plt.subplot(2, 2, 3)
        action_names = [step['action'] for step in step_details]
        unique_actions = list(set(action_names))
        action_counts = [action_names.count(action) for action in unique_actions]
        plt.bar(unique_actions, action_counts)
        plt.title('Action Distribution')
        plt.xlabel('Action')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        
        # Reward heatmap subplot
        plt.subplot(2, 2, 4)
        plt.imshow([rewards], cmap='RdYlGn', aspect='auto')
        plt.title('Reward Heatmap')
        plt.xlabel('Step Number')
        plt.colorbar(label='Reward')
        
        plt.tight_layout()
        plt.savefig('./episode_performance.png')
        plt.close()
        
        print(colored("\n📊 Episode Performance Visualization saved as episode_performance.png", "blue"))

def main():
    """
    Main script to demonstrate model play with visualization
    """
    print(colored("🏥 QuickMed Model Play Visualization", "blue", attrs=['bold']))
    
    # Initialize player
    player = QuickMedModelPlayer()
    
    # Load model
    if not player.load_model():
        return
    
    # Interactive menu
    while True:
        print("\nChoose an option:")
        print("1. Play Single Episode with Visualization")
        print("2. Exit")
        
        choice = input("Enter your choice (1-2): ").strip()
        
        if choice == '1':
            player.play_episode_with_visualization(verbose=True)
        elif choice == '2':
            print(colored("👋 Goodbye!", "green"))
            break
        else:
            print(colored("Invalid choice. Please try again.", "red"))

if __name__ == "__main__":
    main()