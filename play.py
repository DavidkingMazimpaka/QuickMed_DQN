import gym
import numpy as np
from stable_baselines3 import DQN
from pharmacy_env import PharmacyEnv
import pygame
import matplotlib.pyplot as plt

class PharmacySimulation:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        
        # Screen dimensions
        self.screen_width = 800
        self.screen_height = 600
        
        # Create screen
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Pharmacy Logistics Simulation")
        
        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GREEN = (0, 255, 0)
        self.BLUE = (0, 0, 255)
        
        # Font
        self.font = pygame.font.Font(None, 36)
        
        # Load trained agent
        self.env = PharmacyEnv()
        self.model = DQN.load("pharmacy_logistics_agent")
        
    def run_simulation(self):
        # Reset environment
        obs = self.env.reset()
        
        # Simulation loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            
            # Agent action
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, _ = self.env.step(action)
            
            # Clear screen
            self.screen.fill(self.WHITE)
            
            # Draw pharmacy stocks
            self._draw_pharmacy_stocks(obs[:15].reshape(5, 3))
            
            # Draw patient location
            self._draw_patient_location(obs[15:17])
            
            # Draw vehicle positions
            self._draw_vehicle_positions(obs[17:].reshape(2, 2))
            
            # Update display
            pygame.display.flip()
            
            # Check if mission is complete
            if done:
                print(f"Simulation Complete. Reward: {reward}")
                break
        
        pygame.quit()
    
    def _draw_pharmacy_stocks(self, stocks):
        # Draw pharmacies with stock levels
        for i, pharmacy_stock in enumerate(stocks):
            x = 50 + i * 150
            y = 500
            
            # Pharmacy rectangle
            pygame.draw.rect(self.screen, self.BLUE, (x, y - 50, 100, 50))
            
            # Stock levels
            for j, stock in enumerate(pharmacy_stock):
                stock_height = min(stock, 50)
                color = self.GREEN if stock > 0 else self.RED
                pygame.draw.rect(self.screen, color, (x + j*30, y - stock_height, 20, stock_height))
                
                # Stock text
                stock_text = self.font.render(str(int(stock)), True, self.BLACK)
                self.screen.blit(stock_text, (x + j*30, y + 10))
    
    def _draw_patient_location(self, location):
        # Draw patient location
        x = location[1] * 70 + 50
        y = location[0] * 50 + 50
        pygame.draw.circle(self.screen, self.RED, (x, y), 20)
        patient_text = self.font.render("Patient", True, self.BLACK)
        self.screen.blit(patient_text, (x - 30, y + 30))
    
    def _draw_vehicle_positions(self, positions):
        # Draw vehicle positions
        for i, position in enumerate(positions):
            x = position[1] * 70 + 50
            y = position[0] * 50 + 50
            pygame.draw.circle(self.screen, self.BLACK, (x, y), 15)
            vehicle_text = self.font.render(f"V{i+1}", True, self.WHITE)
            self.screen.blit(vehicle_text, (x - 15, y - 15))

def main():
    sim = PharmacySimulation()
    sim.run_simulation()

if __name__ == "__main__":
    main()