# QuickMed with Deep Q-Learning

## QuickMed Description

QuickMed Innovations is a digital healthcare platform designed to connect patients with nearby pharmacies that stock their prescribed medications in real-time. Our mission is to eliminate patients' frustration and delays when searching for specific medicines by providing an efficient, reliable, and centralized digital solution. This platform ensures faster access to essential medications, ultimately improving patient outcomes and enhancing the overall healthcare experience in Rwanda.

## Environment Name: PharmacyAccessEnv

### Description

This environment simulates a scenario where a patient is searching for a specific medication among multiple pharmacies. The environment challenges an agent to efficiently navigate the pharmacies, check their stock, and deliver the required medication to the patient in the shortest possible time while minimizing costs.

## State Space

The state is a vector representing the following components:

1. **Pharmacy Inventory**: A matrix of stock levels across multiple pharmacies for N different medicines (e.g., 5 pharmacies × 10 medicines = 50 values).
2. **Patient Requests**: A list of medicines and quantities needed by the patient.
3. **Patient Location**: 2D coordinates of the patient (e.g., (x, y)).
4. **Pharmacy Locations**: 2D coordinates of all pharmacies.
5. **Current Agent Position**: 2D coordinates of the agent (vehicle/delivery personnel).

## Action Space

The agent can perform the following actions:

1. **Travel to a Pharmacy**: Move the agent to a selected pharmacy.
2. **Check Stock at Pharmacy**: Query the stock of the medicine needed.
3. **Pickup Medication**: Collect the required medication if available.
4. **Travel to Patient**: Deliver the medication to the patient.
5. **Skip (or End Episode)**: End the search if no medication is available (penalty for incorrect skips).

## Rewards

### Positive Rewards

- Successfully delivering medication to the patient: **+100**
- Checking a pharmacy with a stock of requested medication: **+10**

### Negative Rewards

- Traveling unnecessarily (e.g., to pharmacies without stock): **-5 per move**
- Skipping when medication is available: **-50**
- Taking too many steps (episode timeout): **-20**

### Termination Criteria

**Episode Ends:**

- Medication is successfully delivered to the patient.
- The agent exceeds the maximum number of allowed steps (timeout).
- The agent skips when medication is available.

## Use Case Example

### Scenario

A patient in Kigali requests 3 units of Medicine X. There are 5 pharmacies with the following inventory:

- **Pharmacy 1**: 0 units
- **Pharmacy 2**: 2 units
- **Pharmacy 3**: 5 units
- **Pharmacy 4**: 0 units
- **Pharmacy 5**: 3 units

The agent starts at (0, 0) and must decide the optimal sequence of actions (e.g., traveling to Pharmacy 3 first).

### [Link to Video Demo and Zipped File](https://drive.google.com/drive/folders/16BH7mt8_bpDArLjQX7cr8L86ffuQ55vX?usp=sharing)

## Project Structure

```
QuickMed_DQN/
│
├── train.py          # Script for training the DQN agent
├── play.py           # Script for evaluating the trained agent
├── README.md         # This file
│
├── models/           # Directory for saved models (created during training)
│   └── quickmed_dqn_final.zip   # Trained model file (created after training)
│
└── logs/            # Training logs for tensorboard (created during training)
```

## Prerequisites

### System Requirements
- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

### Required Python Packages
- gymnasium
- stable-baselines3


## Installation

1. Clone this repository:
```bash
git clone https://github.com/DavidkingMazimpaka/QuickMed_DQN.git
cd QuickMed_DQN
```

2. Create and activate a virtual environment (recommended):
```bash
# On Windows
python -m venv env
venv\Scripts\activate

# On macOS/Linux
python3 -m venv env
source env/bin/activate
```

3. Install required packages:
```bash
pip install gymnasium stable-baselines3[extra] numpy
```

4. Install Atari ROMs:
```bash
pip install autorom[accept-rom-license]
```

## Usage

### Training the Agent

To train the DQN agent, run:
```bash
python train.py
```

The training script will:
- Create a 'models/' directory to save checkpoints
- Create a 'logs/' directory for tensorboard logs
- Train the agent for 50,000 timesteps (configurable in the script)
- Save the final model as 'models/quickmed_dqn_final.zip'
- Save periodic checkpoints during training

Training progress will be displayed in the console with metrics such as:
- Episode reward mean
- Episode length mean
- Learning rate
- Exploration rate

### Playing with the Trained Agent

To watch the trained agent play, run:
```bash
python play.py
```

This will:
- Load the trained model from 'models/quickmed_dqn_final.zip'
- Run 50 episodes of the game with visualization
- Display the total reward and steps for each episode

## Monitoring Training Progress

You can monitor the training progress using Tensorboard:

1. Install tensorboard if not already installed:
```bash
pip install tensorboard
```

2. Run tensorboard:
```bash
tensorboard --logdir logs/
```


## Configuration

### Training Parameters
You can modify the following parameters in `train.py`:
- `total_timesteps`: Number of training steps (default: 50,000)
- `learning_rate`: Learning rate for the neural network (default: 1e-4)
- `buffer_size`: Size of the replay buffer (default: 50,000)
- `learning_starts`: Number of steps before starting training (default: 1,000)
- `batch_size`: Size of training batches (default: 32)
- `exploration_fraction`: Fraction of total timesteps for exploration (default: 0.1)

### Evaluation Parameters
In `play.py`, you can modify:
- `n_episodes`: Number of episodes to play (default: 5)

## Troubleshooting

1. If you encounter ROM-related errors:
```bash
pip install autorom[accept-rom-license]
```

2. If you get rendering errors:
- Make sure you have the required system dependencies for OpenGL
- Try updating your graphics drivers

3. If the model file isn't found:
- Ensure you've run `train.py` before `play.py`
- Check that the model file exists in the 'models/' directory

## Performance Notes

- The default training duration (50,000 steps) is relatively short for optimal performance
- For better results, consider increasing `total_timesteps` to 500,000 or more
- Training time will vary depending on your hardware
- GPU acceleration is supported and recommended for faster training

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) for the DQN implementation
- [Gymnasium](https://gymnasium.farama.org/) for the Breakout environment
- Atari and Breakout are trademarks of Atari Interactive Inc.

## Author
- [KingDavid Mazimpaka](https://github.com/DavidkingMazimpaka/)
