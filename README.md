# QuickMed with Deep Q-Learning

## QuickMed Description

QuickMed Innovations is a digital healthcare platform designed to connect patients with nearby pharmacies that stock their prescribed medications in real time. Our mission is to eliminate the frustration and delays patients face when searching for specific medicines by providing an efficient, reliable, and centralized digital solution. This platform ensures faster access to essential medications, ultimately improving patient outcomes and enhancing the overall healthcare experience in Rwanda.

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
- Checking a pharmacy with stock of requested medication: **+10**

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

### Alignment with Mission

- **Real-Time Decision-Making**: Mirrors the challenge of identifying which pharmacy stocks the needed medication.
- **Efficient Routing**: Models the logistics of delivering medication promptly to the patient.
- **Minimizing Frustration**: Penalizes unnecessary travel or failed attempts to incentivize optimized behavior.

