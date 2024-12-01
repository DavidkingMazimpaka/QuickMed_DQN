import gym
import numpy as np
from gym import spaces

class PharmacyAccessEnv(gym.Env):
    """
    Custom gym environment for the QuickMed pharmacy access scenario
    """
    def __init__(self, num_pharmacies=5, num_medicines=10):
        super(PharmacyAccessEnv, self).__init__()
        
        # Environment configuration
        self.num_pharmacies = num_pharmacies
        self.num_medicines = num_medicines
        self.max_steps = 50
        
        # Action space
        # 0: Travel to Pharmacy
        # 1: Check Stock
        # 2: Pickup Medication
        # 3: Travel to Patient
        # 4: Skip/End Episode
        self.action_space = spaces.Discrete(5)
        
        # Calculate the total size of the observation vector
        inventory_size = num_pharmacies * num_medicines
        patient_requests_size = num_medicines
        patient_location_size = 2
        pharmacy_locations_size = num_pharmacies * 2
        agent_location_size = 2
        
        total_size = (
            inventory_size + 
            patient_requests_size + 
            patient_location_size + 
            pharmacy_locations_size + 
            agent_location_size
        )
        
        # State space
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(total_size,), 
            dtype=np.float32
        )
        
        # State variables
        self.pharmacy_inventory = None
        self.patient_requests = None
        self.patient_location = None
        self.pharmacy_locations = None
        self.agent_location = None
        self.current_step = None
        self.delivered_medicines = None
        
    def reset(self):
        """
        Reset the environment to initial state
        """
        # Initialize random pharmacy inventory (0-10 units)
        self.pharmacy_inventory = np.random.randint(0, 11, 
            size=(self.num_pharmacies, self.num_medicines))
        
        # Initialize patient requests (1-3 units of random medicines)
        self.patient_requests = np.random.randint(1, 4, size=self.num_medicines)
        
        # Random locations on a 100x100 grid
        self.patient_location = np.random.uniform(0, 100, size=2)
        self.pharmacy_locations = np.random.uniform(0, 100, size=(self.num_pharmacies, 2))
        self.agent_location = np.array([0.0, 0.0])  # Start at (0, 0)
        
        # Reset tracking variables
        self.current_step = 0
        self.delivered_medicines = np.zeros(self.num_medicines)
        
        return self._get_state()
    
    def step(self, action):
        """
        Execute an action and return next state, reward, done flag, info
        """
        self.current_step += 1
        reward = 0
        done = False
        
        if action == 0:  # Travel to Pharmacy
            # Select a random pharmacy to travel to
            target_pharmacy = np.random.randint(0, self.num_pharmacies)
            self.agent_location = self.pharmacy_locations[target_pharmacy]
            reward -= 5  # Penalty for traveling
        
        elif action == 1:  # Check Stock
            current_pharmacy_index = np.argmin(np.linalg.norm(
                self.pharmacy_locations - self.agent_location, axis=1))
            
            # Check if any requested medicine is in stock
            stock_available = np.any(
                (self.pharmacy_inventory[current_pharmacy_index] > 0) & 
                (self.patient_requests > 0)
            )
            
            if stock_available:
                reward += 10  # Reward for finding potential stock
        
        elif action == 2:  # Pickup Medication
            current_pharmacy_index = np.argmin(np.linalg.norm(
                self.pharmacy_locations - self.agent_location, axis=1))
            
            # Find medicines that can be picked up
            pickup_indices = np.where(
                (self.pharmacy_inventory[current_pharmacy_index] > 0) & 
                (self.patient_requests > 0)
            )[0]
            
            for med_index in pickup_indices:
                pickup_amount = min(
                    self.pharmacy_inventory[current_pharmacy_index][med_index],
                    self.patient_requests[med_index]
                )
                
                self.pharmacy_inventory[current_pharmacy_index][med_index] -= pickup_amount
                self.patient_requests[med_index] -= pickup_amount
                self.delivered_medicines[med_index] += pickup_amount
        
        elif action == 3:  # Travel to Patient
            self.agent_location = self.patient_location
            
            # Check if all patient requests are fulfilled
            if np.all(self.patient_requests == 0):
                reward += 100  # Large reward for successful delivery
                done = True
        
        elif action == 4:  # Skip/End Episode
            # Penalty if medication is still needed
            if np.any(self.patient_requests > 0):
                reward -= 50
            done = True
        
        # Check for timeout
        if self.current_step >= self.max_steps:
            reward -= 20
            done = True
        
        return self._get_state(), reward, done, {}
    
    def _get_state(self):
        """
        Construct the state vector
        """
        state = np.concatenate([
            self.pharmacy_inventory.flatten(),     # Pharmacy Inventory
            self.patient_requests,                 # Patient Requests
            self.patient_location,                 # Patient Location
            self.pharmacy_locations.flatten(),     # Pharmacy Locations
            self.agent_location                    # Current Agent Position
        ])
        
        return state
    
    def render(self, mode='human'):
        """
        Optional rendering method (console output)
        """
        print("Current Agent Location:", self.agent_location)
        print("Patient Requests:", self.patient_requests)
        print("Delivered Medicines:", self.delivered_medicines)

# Optional: Demonstration
if __name__ == "__main__":
    env = PharmacyAccessEnv()
    state = env.reset()
    print("Initial State Shape:", state.shape)
    print("Initial State:", state)
    
    for _ in range(10):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        
        if done:
            break