import gym
from gym import spaces
import numpy as np

class PharmacyEnv(gym.Env):
    def __init__(self):
        super(PharmacyEnv, self).__init__()
        
        # State: [stock levels, patient requests, delivery vehicle positions]
        self.num_pharmacies = 5
        self.num_medications = 3
        self.max_stock = 50
        self.num_vehicles = 2
        self.grid_size = 10
        
        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(self.num_pharmacies * self.num_medications + self.num_vehicles * 2 + 2,),
            dtype=np.float32
        )
        
        # Action: [Assign pharmacy, dispatch vehicle]
        self.action_space = spaces.Discrete(self.num_pharmacies * self.num_vehicles)
        
        # Initialize state
        self.reset()

    def reset(self):
        self.pharmacy_stocks = np.random.randint(10, self.max_stock, (self.num_pharmacies, self.num_medications))
        self.patient_request = np.random.randint(0, self.num_medications)
        self.patient_location = np.random.randint(0, self.grid_size, 2)
        self.vehicle_positions = np.random.randint(0, self.grid_size, (self.num_vehicles, 2))
        self.done = False

        # Flatten state
        self.state = np.concatenate([
            self.pharmacy_stocks.flatten(),
            self.patient_location,
            self.vehicle_positions.flatten()
        ])

        return self.state

    def step(self, action):
        pharmacy_id = action % self.num_pharmacies
        vehicle_id = action // self.num_pharmacies

        # Compute the cost (distance) to fulfill request
        vehicle_position = self.vehicle_positions[vehicle_id]
        pharmacy_stock = self.pharmacy_stocks[pharmacy_id]
        
        distance_to_pharmacy = np.linalg.norm(vehicle_position - np.array(self.patient_location))
        
        # Check if stock is available
        if pharmacy_stock[self.patient_request] > 0:
            pharmacy_stock[self.patient_request] -= 1
            reward = 10 - distance_to_pharmacy
        else:
            reward = -15  # Penalty for stockout

        # Move vehicle closer to patient
        self.vehicle_positions[vehicle_id] = self.patient_location
        
        # Update state
        self.state = np.concatenate([
            self.pharmacy_stocks.flatten(),
            self.patient_location,
            self.vehicle_positions.flatten()
        ])

        # End episode after one step
        self.done = True

        return self.state, reward, self.done, {}

    def render(self, mode='human'):
        print(f"Pharmacy stocks: {self.pharmacy_stocks}")
        print(f"Patient request: {self.patient_request} at {self.patient_location}")
        print(f"Vehicle positions: {self.vehicle_positions}")
