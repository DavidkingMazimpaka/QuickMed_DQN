import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import time

class PharmacyEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_size=5, max_steps=100):
        super(PharmacyEnv, self).__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.current_step = 0

        # More complex observation space to capture more information
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32),
            "goals": spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32),
            "obstacles": spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32)
        })

        # Four directional actions plus diagonal movements
        self.action_space = spaces.Discrete(8)  # 0:Up, 1:Down, 2:Left, 3:Right, 4:UpLeft, 5:UpRight, 6:DownLeft, 7:DownRight

        # Initialize environment variables
        self.agent_pos = None
        self.goal_positions = []
        self.obstacle_positions = []
        self.visited_goals = set()

        # Rendering
        self.cell_size = 100
        self.window_size = self.grid_size * self.cell_size
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomize start, goal, and obstacle positions
        np.random.seed(seed)
        
        # Reset agent to a random starting position
        self.agent_pos = [np.random.randint(0, self.grid_size), 
                          np.random.randint(0, self.grid_size)]
        
        # Reset goals and ensure they don't overlap with agent or each other
        self.goal_positions = []
        while len(self.goal_positions) < 3:
            goal = [np.random.randint(0, self.grid_size), 
                    np.random.randint(0, self.grid_size)]
            if goal not in self.goal_positions and goal != self.agent_pos:
                self.goal_positions.append(goal)
        
        # Add some randomized obstacles
        self.obstacle_positions = []
        while len(self.obstacle_positions) < 3:
            obstacle = [np.random.randint(0, self.grid_size), 
                        np.random.randint(0, self.grid_size)]
            if (obstacle not in self.goal_positions and 
                obstacle != self.agent_pos and 
                obstacle not in self.obstacle_positions):
                self.obstacle_positions.append(obstacle)
        
        # Reset tracking variables
        self.visited_goals = set()
        self.current_step = 0

        return self._get_obs(), {}

    def _get_obs(self):
        # Create goals and obstacles masks
        goals_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for goal in self.goal_positions:
            goals_mask[goal[0], goal[1]] = 1.0

        obstacles_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        for obstacle in self.obstacle_positions:
            obstacles_mask[obstacle[0], obstacle[1]] = 1.0

        return {
            "agent": np.array(self.agent_pos, dtype=np.int32),
            "goals": goals_mask,
            "obstacles": obstacles_mask
        }

    def step(self, action):
        self.current_step += 1
        
        # More complex movement logic with diagonal moves
        moves = [
            (-1, 0),   # Up
            (1, 0),    # Down
            (0, -1),   # Left
            (0, 1),    # Right
            (-1, -1),  # Up-Left
            (-1, 1),   # Up-Right
            (1, -1),   # Down-Left
            (1, 1)     # Down-Right
        ]
        
        # Calculate new position
        move = moves[action]
        new_pos = [
            max(0, min(self.grid_size - 1, self.agent_pos[0] + move[0])),
            max(0, min(self.grid_size - 1, self.agent_pos[1] + move[1]))
        ]

        # Check for obstacle collision
        if new_pos in self.obstacle_positions:
            return self._get_obs(), -20, False, False, {}

        # Update agent position
        self.agent_pos = new_pos

        # Reward calculations
        reward = -1  # Small negative reward for each step to encourage efficiency

        # Goal collection logic
        if self.agent_pos in self.goal_positions:
            self.visited_goals.add(tuple(self.agent_pos))
            self.goal_positions.remove(self.agent_pos)
            reward += 50  # Significant reward for collecting a goal

        # Termination conditions
        done = (len(self.visited_goals) == 3) or (self.current_step >= self.max_steps)
        
        if done and len(self.visited_goals) == 3:
            reward += 100  # Bonus for collecting all goals

        return self._get_obs(), reward, done, False, {}

    def render(self, mode="human"):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Pharmacy Environment")
            self.clock = pygame.time.Clock()

        self.window.fill((255, 255, 255))  # White background
        
        # Draw grid
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size)
                pygame.draw.rect(self.window, (200, 200, 200), rect, 1)

                # Draw agent
                if [x, y] == self.agent_pos:
                    pygame.draw.circle(self.window, (0, 0, 255), rect.center, self.cell_size // 3)
                
                # Draw goals
                elif [x, y] in self.goal_positions:
                    pygame.draw.rect(self.window, (0, 255, 0), rect)
                
                # Draw obstacles
                elif [x, y] in self.obstacle_positions:
                    pygame.draw.rect(self.window, (255, 0, 0), rect)

        pygame.display.flip()
        self.clock.tick(30)

    def close(self, message=None):
        if self.window:
            if message:
                font = pygame.font.SysFont(None, 55)
                text = font.render(message, True, (0, 0, 0))
                self.window.blit(text, (self.window_size // 4, self.window_size // 2))
                pygame.display.flip()
                time.sleep(10)  # Keep message on screen for 10 seconds
            pygame.quit()
            self.window = None