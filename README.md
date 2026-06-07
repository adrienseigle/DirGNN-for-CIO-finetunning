# Recursive GNN CIO Optimization

This repository implements a simulator-in-the-loop optimization pipeline for **Cell Individual Offset (CIO)** tuning in mobile networks. It uses a **Directed Graph Neural Network (DirGNN)** to learn directed handover failure risk from `mobile-env` simulations, then recursively adapts the CIO matrix and re-simulates the network.

The current project state is focused on the recursive GNN CIO optimizer only. The older dynamic step-level CIO prototype was removed after its useful utilities were integrated directly into the recursive pipeline.

## Objective

The goal is to improve handover behavior by learning which directed base-station transitions are risky and adapting the CIO matrix accordingly.

The pipeline targets:

- **Too-early handovers**
- **Too-late handovers**
- **Ping-pong handovers**
- **Directed CIO adaptation per source-target base-station pair**
- **Stable handover preservation under safety constraints**

## Core Idea

Each base station is represented as a graph node. Each possible handover direction `i -> j` is represented as a directed edge with edge features, including the current CIO value.

At each recursive iteration:

1. Run a `mobile-env` simulation with the current CIO matrix.
2. Extract node features, directed edge features, and handover events.
3. Train a Directed GNN to predict directed-edge failure risk.
4. Convert predicted risk into a row-wise, mean-centered CIO update.
5. Safety-check the candidate CIO by re-simulating it.
6. Save metrics, matrices, plots, and repeat.

## Main Script

```bash
python recursive_gnn_cio_optimization.py
```

## Important Files

| File | Role |
|---|---|
| `recursive_gnn_cio_optimization.py` | Main recursive simulator-in-the-loop GNN CIO optimizer |
| `dirgnn_cio.py` | Directed GNN model definitions and graph training helpers |
| `iterative_cio_training.py` | Legacy/support CIO training utilities reused by the recursive optimizer |
| `env_loader.py` | Loads the installed `mobile-env` package without local import shadowing |
| `regenerate_cio_matrix_plots.py` | Utility to regenerate CIO/risk matrix plots |
| `REPORT_RECURSIVE_CIO_STATE.md` | Detailed technical state report and development notes |
| `REPORT/report.tex` | LaTeX project report |

## Model Inputs

### Node features

Each base station node contains:

```text
load
avg_velocity
ues_in_range
avg_snr
min_snr
```

### Directed edge features

Each directed edge `i -> j` contains:

```text
handover_count
transition_prob
normal_handovers
cio
early_severity
late_severity
ping_pong_severity
failure_severity
```

The CIO value is therefore part of the GNN edge attributes.

## Prediction Target

The GNN predicts directed-edge failure risk, not CIO directly.

The observed risk target is based on detected failures normalized by handover activity:

```text
risk_i,j = failure_count_i,j / max(handover_count_i,j, 1)
```

The model output is passed through a sigmoid and trained with weighted mean squared error. Active handover edges receive higher weight than inactive edges.

## CIO Update Rule

The current CIO update is row-wise and mean-centered. For each source base station `i`, the optimizer compares risks among all outgoing targets `j`.

High-risk target directions receive higher CIO values, while lower-risk target directions receive lower CIO values. The row is then centered to avoid globally suppressing all outgoing handovers from the same source cell.

Safety backtracking tests several scales of the candidate update:

```text
1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0
```

A candidate is accepted only if it preserves enough handovers relative to the baseline.

## A3 Handover Rule

Handover decisions are still made by the simulator using a 3GPP-inspired A3 condition. The GNN only changes the CIO matrix.

Conceptually:

```text
M_t - M_s > Off_A3 + Hys_A3 + cio_scale * CIO_i,j
```

where:

```text
M_t       target-cell SNR/RSRP proxy
M_s       serving-cell SNR/RSRP proxy
Off_A3    global A3 offset
Hys_A3    hysteresis
CIO_i,j   directed CIO from source i to target j
```

The stored CIO values are intentionally scaled to be human-readable, currently around `0.1` to `1`, while `cio_scale` controls their effective impact inside the A3 condition.

## Parameter Presets

The optimizer supports presets:

```text
--parameter-preset auto
--parameter-preset small-balanced
--parameter-preset large-balanced
--parameter-preset none
```

Default:

```text
--parameter-preset auto
```

This selects `small-balanced` for non-large environments and `large-balanced` for environment names containing `large`.

Current preset scaling:

```text
cio_scale = 0.000001
cio_lr = 2000.0
max_cio_step = 2.0
cio_clip = 5.0
min_handover_ratio = 0.9
```

Use `--parameter-preset none` to avoid preset overrides when manually tuning parameters.

## Example Runs

### Small environment

```bash
python recursive_gnn_cio_optimization.py \
  --output-dir results_recursive/final_small_cio_fixed \
  --env-name mobile-small-central-v0 \
  --num-episodes 1 \
  --max-steps 30 \
  --iterations 4 \
  --min-train-graphs 5 \
  --epochs-per-iteration 2 \
  --hidden-dim 16 \
  --num-layers 2 \
  --batch-size 8 \
  --parameter-preset auto \
  --fixed-scenario \
  --device cpu
```

### Large environment

```bash
python recursive_gnn_cio_optimization.py \
  --output-dir results_recursive/final_large_cio_fixed \
  --env-name mobile-large-central-v0 \
  --num-episodes 1 \
  --max-steps 80 \
  --iterations 5 \
  --min-train-graphs 10 \
  --epochs-per-iteration 3 \
  --hidden-dim 16 \
  --num-layers 2 \
  --batch-size 16 \
  --parameter-preset auto \
  --fixed-scenario \
  --device cpu
```

## Final Validated Results

### Small environment

```text
normal: 85.7%
too_late_handover: 14.3%
too_early_handover: 0.0%
ping_pong_overlay: 10.7%
final handovers: 28 / 29 baseline
CIO range: [-0.101, 0.101]
```

### Large environment

```text
normal: 74.5%
too_late_handover: 25.5%
too_early_handover: 0.0%
ping_pong_overlay: 12.4%
final handovers: 773 / 812 baseline
CIO range: [-0.180, 0.367]
```

## Main Outputs

Each run writes:

```text
recursive_iteration_metrics.csv
recursive_optimization_metrics.png
recursive_edge_features.csv
recursive_node_features.csv
recursive_handover_events.csv
cio_matrix_iteration_XXX.csv
predicted_risk_matrix_iteration_XXX.csv
final_cio_matrix.csv
final_cio_matrix.png
final_predicted_risk_matrix.csv
final_predicted_risk_matrix.png
failure_type_distribution.png
gnn_vs_baseline_comparison.csv
recursive_risk_gnn_model.pt
```

## Requirements

Typical dependencies include:

- Python 3.10+
- PyTorch
- PyTorch Geometric
- Gymnasium
- mobile-env
- NumPy
- Pandas
- Matplotlib
- Seaborn

Install project requirements with:

```bash
pip install -r requirements.txt
```

## Current Repository State

The commit-ready state keeps the recursive optimizer and final result folders:

```text
results_recursive/final_small_cio_fixed
results_recursive/final_large_cio_fixed
```

Obsolete dynamic CIO prototype files and results have been removed.

## License

This project is part of the CentraleSupélec Pôle Projet S8 research initiative.
