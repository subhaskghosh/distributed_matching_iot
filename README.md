# Distributed Matching for IoT Resource Allocation

## Overview
This repository contains an implementation of distributed matching algorithms for IoT-based resource allocation. The project simulates a complex environment of users, access points, and cloudlets to study the efficiency of different resource allocation strategies in terms of delay tolerance, energy efficiency, and computational load.

The key components of the system include:
- **Network Simulation**: Models users, access points (APs), and cloudlets.
- **Bipartite Graph Reduction**: Reduces the network to a bipartite graph for matching.
- **Distributed Matching Algorithms**: Implements both greedy and maximum weight matching algorithms.
- **Performance Metrics**: Utility calculation, delay analysis, and statistical evaluations.

## Features
- Simulation of IoT environments with customizable parameters.
- Distributed maximum weight matching for resource allocation.
- Greedy matching algorithm for baseline comparisons.
- Statistical analysis of performance metrics including delay, utility, and resource efficiency.
- Visualization tools for network topology, task distributions, and matching results.

## Getting Started
### Prerequisites
- Python 3.8+
- Required libraries:
  ```bash
  pip install numpy pandas matplotlib seaborn networkx scipy
  ```

### Running the Simulation
```bash
python simulation/simulation.py
```
This will run simulations based on the parameters specified in `NetworkSimulatorRunner` and save the results in `simulation_results.csv`.

### Key Parameters
- `w`: Area width
- `n`: Number of users
- `m`: Number of access points
- `p`: Percentage of APs with cloudlets
- `u_sample`: Percentage of users sampled for matching
- `dist_type`: Task distribution type (uniform, normal, log-normal, exponential, beta, mixture)

### Customizing Simulations
Modify the `generate_combinations` method in `NetworkSimulatorRunner` to adjust the range of parameters for simulations.

## Algorithms
### Distributed Maximum Weight Matching
The core algorithm focuses on distributed maximum weight matching using:
- **Fractional Matching Stage:** Computes initial fractional matches.
- **Reduction to Multigraph:** Converts fractional matches to multigraph for gradual rounding.
- **Gradual Rounding:** Converts fractional solutions to integral matchings.

### Greedy Matching
A baseline greedy algorithm is implemented to compare against the maximum weight matching algorithm.

## Performance Metrics
- **Utility:** Measures the efficiency of resource allocation.
- **Delay Analysis:** Identifies tasks exceeding delay tolerance.
- **Resource Utilization:** Compares allocation efficiency across algorithms.

## Visualization
- **Network Topology:** Displays user and AP deployment.
- **Bipartite Graph:** Shows matching between users and cloudlet-enabled APs.
- **Performance Metrics:** Graphs for task distributions, uplink/downlink statistics, and utility analysis.

## Results
Simulation results are saved in `simulation_results.csv`, containing:
- Utility values
- Percentage of tasks above delay tolerance
- Maximum delay miss
- Statistical analysis of missed tasks

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/fooBar`)
3. Commit your changes (`git commit -am 'Add some fooBar'`)
4. Push to the branch (`git push origin feature/fooBar`)
5. Create a new Pull Request

## License
Distributed under the MIT License. See `LICENSE` for more information.

---
For issues, please raise an issue in the repository or contact the maintainer.

