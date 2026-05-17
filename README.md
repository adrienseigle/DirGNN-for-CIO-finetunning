# Mobile Environment Local

A mobile network optimization framework for antenna traffic forecasting and energy efficiency using Directed Graph Neural Networks (DirGNN) for CIO (Cell Individual Offset) optimization.

## Purpose of the Project

The primary goal of this project is to leverage Directed Graph Neural Networks (DirGNN) to find optimal values for the Cell Individual Offset (CIO) matrix in 4G and 5G cellular networks.

Mobile networks are fundamentally directional and asymmetrical. User equipment moves from a serving base station to a target base station under specific spatial and signal conditions. By modeling the cellular topology as a directed graph where base stations act as nodes and handovers act as directed edges, this framework aims to intelligently predict handover failure risks. It targets specific issues like early failures, late failures, and ping-pong effects, then automatically finetunes CIO parameters to minimize these problems and optimize overall network performance.

The project orchestrates a closed-loop, iterative process. It runs a Gymnasium-based mobile environment (`mobile-env`) to generate synthetic handover metrics under current CIO configurations, feeds these directed graph snapshots into a custom DirGNN to learn edge-specific failure risks, and then dynamically adjusts the CIO matrix to balance the network load and mitigate risky handovers.

### Understanding Cell Individual Offset (CIO)

In 4G (LTE) and 5G (NR) networks, a Cell Individual Offset is a configurable parameter used to manage and bias handover decisions between neighboring base stations.

When a mobile device moves through a network, it continuously measures the signal strength of its serving cell against neighboring cells. According to 3GPP standards, a handover is triggered only when a target cell's signal strength exceeds the serving cell's strength by a specific threshold.

The CIO acts as a correction factor added to or subtracted from these measurements for a specific pair of cells. A positive CIO artificially inflates the perceived signal strength of a target cell, causing the device to hand over to it earlier than it normally would. Conversely, a negative CIO de-prioritizes a target cell, delaying the handover and forcing the device to stay connected to its current serving cell longer. By fine-tuning the CIO matrix across the entire network, operators can proactively manage cell boundaries, balance traffic loads, and eliminate costly handover failures.

The goal of the DirGNN modelisation is to follow this implementation:

 <img width="437" height="828" alt="image" src="https://github.com/user-attachments/assets/50d5894e-01a6-4016-81cd-faad5fa98c61" />

## Project Structure & Script Details

### Core Simulation
- **`mobile_environment.py`** - Simulation Engine & Data Generation
  - Simulates user equipment (UE) movement across a mobile network using the Gymnasium environment
  - Collects handover data and implements 3GPP-compliant handover detection (Eq. 29)
  - Generates node features: per-BS load, velocity, SNR metrics
  - Generates edge features: handover counts, failure types (early/late/ping-pong), CIO values
  - Extracts transition probabilities across base stations for graph construction
  - Key functions: `simulate_dataset()`, `build_action_for_ue()`, `get_serving_bs()`

### Support & Utilities
- **`env_loader.py`** - Environment Package Resolution
  - Resolves import conflicts between local `mobile_env_local` directory and installed `mobile-env` package
  - Uses importlib to explicitly load the installed package from site-packages
  - Prevents shadowing issues during runtime

- **`data_utils.py`** - Data Processing & Feature Engineering
  - Utilities for manipulating handover data and failure metrics
  - Functions: `add_edge_cio()` (attach CIO values), `compute_failure_counts_by_category()` (aggregate failures), `build_failure_target()` (format training targets)
  - Supports multiple failure targets: early_failures, late_failures, ping_pongs, failure_count, any_failure

### Machine Learning Pipeline
- **`dirgnn_cio.py`** - Directed GNN Models for Edge Risk Prediction
  - Implements three directed graph convolution models:
    - **DirGCNConv**: Directed Graph Convolution with separate in/out-degree handling
    - **DirSageConv**: Directed GraphSAGE for neighborhood aggregation
    - **DirGATConv**: Directed Graph Attention with directed normalization
  - Architecture: DirGNNEncoder (node feature extraction) → edge embeddings → DirGNNEdgePredictor (risk scores)
  - Key functions: `load_graph_snapshots()`, `train_epoch()`, `evaluate()`
  - Predicts binary failures (any_failure) or regression targets (failure_count, ping_pongs)

- **`iterative_cio_training.py`** - Main Training Pipeline Orchestrator
  - Implements complete iterative optimization loop:
    1. **SIMULATE**: Run mobile-env with current CIO matrix
    2. **TRAIN**: Train Directed GNN to predict edge failures
    3. **UPDATE**: Adjust CIO matrix based on predicted risks
    4. **REPEAT**: Iterate until convergence
  - Outputs: CIO matrices per iteration (NPZ + CSV), training metrics, visualization plots
  - Command-line interface with configurable parameters (iterations, episodes, environment)

- **`results/`** - Output & Evaluation Results
  - Stores CIO matrices, training metrics, and visualizations from each iteration

## Requirements

- Python 3.8+
- PyTorch >= 1.9
- PyTorch Geometric
- Gymnasium
- mobile-env (for simulation)
- NumPy
- Pandas
- Matplotlib
- Seaborn

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Workflow

Run the complete iterative CIO training pipeline:

```bash
python iterative_cio_training.py \
  --data-dir results/experiment_1 \
  --iterations 4 \
  --num-episodes 20 \
  --env-name mobile-small-central-v0
```

### Key Parameters

- `--data-dir`: Directory to store results (CIO matrices, metrics, plots)
- `--iterations`: Number of training iterations (simulate → train → update → repeat)
- `--num-episodes`: Number of simulation episodes per iteration
- `--env-name`: Gymnasium environment name to use
- `--model`: GNN architecture (default: "DirGAT")
- `--target`: Failure target to optimize (default: "any_failure")

### Pipeline Details

1. **Initialization**: Random CIO matrix generated
2. **Each Iteration**:
   - Simulation generates handover data with current CIO
   - GNN trains on handover failures to predict edge risks
   - CIO matrix updated based on predicted failure risks
   - Metrics and visualizations saved
3. **Output**: Convergence analysis, failure reduction trends, optimized CIO matrix

## References

- 3GPP Handover Criterion: Equation 29 (Mt - Ms > OI_t,j + hysteresis + CIO_i,j)
- Directed normalization for graph convolutions on directed graphs
- Research on CIO optimization for reducing handover failures

## License

This project is part of the CentraleSupélec Pole Projet S8 research initiative.
