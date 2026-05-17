# Mobile Environment Local

A mobile network optimization framework for antenna traffic forecasting and energy efficiency.

## Overview

This project implements local optimization for mobile network environments, including:
- Antenna traffic data loading and preprocessing
- Graph Neural Network (GNN) based optimization
- CIO (Coordinated Inter-cell Interference Coordination) training
- Performance evaluation and result analysis

## Project Structure

- `mobile_environment.py` - Core mobile environment implementation
- `env_loader.py` - Data loading utilities
- `data_utils.py` - Data preprocessing and utilities
- `dirgnn_cio.py` - Directory-based GNN for CIO optimization
- `iterative_cio_training.py` - Iterative training pipeline for CIO
- `results/` - Output and evaluation results

## Requirements

- Python 3.8+
- PyTorch
- PyTorch Geometric
- NumPy
- Pandas

## Installation

```bash
pip install -r requirements.txt
```

## Usage

See individual module docstrings for detailed usage instructions.

## License

This project is part of the CENTRALE Nantes Pole Projet S8 research initiative.
