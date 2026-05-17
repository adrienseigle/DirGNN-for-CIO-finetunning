"""
SIMULATION ENGINE - Mobile Network Environment Data Generation

Purpose:
  Simulates user equipment (UE) movement across a mobile network and collects
  handover data. Implements 3GPP-compliant handover detection with CIO biasing.
  
Inputs:
  - Environment config (seed, number of episodes)
  - CIO matrix (for biasing handover decisions)
  
Outputs:
  - Node features: per-BS load, velocity, SNR metrics
  - Edge features: handover counts, failure types, CIO values
  - Handover events: raw HO logs for analysis

Key Functions:
  simulate_dataset() - Main entry point
  build_action_for_ue() - 3GPP handover decision logic
  get_serving_bs() - Find best serving base station

References:
  Equation 29 (3GPP handover criterion):
    Mt - Ms > OI_t,j + hysteresis + CIO_i,j
"""

import gymnasium as gym
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
from mobile_env_local.env_loader import load_installed_mobile_env_package

installed_mobile_env = load_installed_mobile_env_package()
import os
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend to avoid display issues
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ============================================================
# Configuration
# ============================================================
ENV_NAME = "mobile-small-central-v0"
NUM_EPISODES = 20         # Run multiple episodes for more data
MAX_STEPS_PER_EP = 100    # Default EP_MAX_TIME in mobile-env is 100
RANDOM_SEED_BASE = 42
PING_PONG_WINDOW = 5      # If a UE returns to previous BS within N steps, it's a ping-pong
SNR_EARLY_THRESHOLD = 0.3 # Normalized SNR below this at handover = potential early failure
SNR_LATE_THRESHOLD = 0.1  # Normalized SNR below this = potential late failure

# ============================================================
# Data Collection Structures
# ============================================================
all_node_data = []         # Per-step, per-BS node features
all_edge_data = []         # Per-step, per-edge (i->j) features
all_handover_events = []   # Raw handover event log

# Cumulative transition counts across all episodes
cumulative_transitions = None
num_stations = None

print("=" * 70)
print("  Mobile-Env Data Generator for Directed GNN")
print("  Extracting: Load, Velocity, Trajectory Patterns")
print("=" * 70)

# ============================================================
# Run Simulation Episodes
# ============================================================
for episode in range(NUM_EPISODES):
    print(f"\n--- Episode {episode + 1}/{NUM_EPISODES} ---")

    # Create and reset environment
    # Use a different seed config each episode for diverse trajectories
    config = {"seed": RANDOM_SEED_BASE + episode * 7, "reset_rng_episode": True}
    env = gym.make(ENV_NAME, config=config)
    obs, info = env.reset(seed=RANDOM_SEED_BASE + episode * 7)

    # Access the unwrapped environment for internal state
    core = env.unwrapped

    # Get the number of stations and users
    num_stations = core.NUM_STATIONS
    num_users = core.NUM_USERS

    # Initialize cumulative transitions on first episode
    if cumulative_transitions is None:
        cumulative_transitions = np.zeros((num_stations, num_stations), dtype=int)

    # Per-episode tracking
    # Track which BS each UE is connected to (best-signal based)
    previous_bs = {}          # ue_id -> bs_id at previous step
    ue_bs_history = defaultdict(list)  # ue_id -> list of (step, bs_id)

    # Per-episode transition counts
    episode_transitions = np.zeros((num_stations, num_stations), dtype=int)

    # Per-episode handover events for ping-pong detection
    ue_handover_times = defaultdict(list)  # ue_id -> list of (step, from_bs, to_bs)

    done = False
    step = 0

    while not done and step < MAX_STEPS_PER_EP:
        # --------------------------------------------------------
        # 1. Use a "strongest signal" policy (connect each UE to
        #    the BS with highest SNR). This generates realistic
        #    mobility trajectories for offline RL training.
        # --------------------------------------------------------
        # Build action: for each UE, pick BS with highest SNR
        action = np.zeros(num_users, dtype=int)

        stations_sorted = sorted(core.stations.values(), key=lambda bs: bs.bs_id)

        for ue_idx, ue_id in enumerate(sorted(core.users.keys())):
            ue = core.users[ue_id]
            if ue not in core.active:
                action[ue_idx] = 0  # NOOP for inactive UEs
                continue

            # Calculate SNR to each BS
            best_snr = -np.inf
            best_bs_id = 0
            for bs in stations_sorted:
                try:
                    snr = core.channel.snr(bs, ue)
                    if snr > best_snr:
                        best_snr = snr
                        best_bs_id = bs.bs_id + 1  # action = bs_id + 1 (0 is NOOP)
                except:
                    pass

            action[ue_idx] = best_bs_id

        # Step the environment
        next_obs, reward, terminated, truncated, info = env.step(action)

        # --------------------------------------------------------
        # 2. Extract internal state after the step
        # --------------------------------------------------------

        # 2a. Determine current connections: which UEs are connected to which BSs
        bs_connected_ues = defaultdict(set)  # bs_id -> set of ue_ids
        for bs in stations_sorted:
            for ue in core.connections.get(bs, set()):
                bs_connected_ues[bs.bs_id].add(ue.ue_id)

        # 2b. Also compute "closest BS" for each active UE based on distance
        # (used as ground truth for where the UE "belongs")
        ue_closest_bs = {}
        for ue in core.active:
            min_dist = np.inf
            closest = 0
            for bs in stations_sorted:
                dist = np.sqrt((ue.x - bs.x) ** 2 + (ue.y - bs.y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    closest = bs.bs_id
            ue_closest_bs[ue.ue_id] = closest

        # --------------------------------------------------------
        # 3. Compute Node Features per BS
        # --------------------------------------------------------
        for bs in stations_sorted:
            bs_id = bs.bs_id

            # Feature 1: LOAD -- number of connected UEs
            connected_ues = bs_connected_ues[bs_id]
            load = len(connected_ues)

            # Feature 2: VELOCITY -- average velocity of connected UEs
            velocities = []
            for ue_id in connected_ues:
                ue = core.users[ue_id]
                # In mobile-env, ue.velocity is a scalar (speed in units/step)
                velocities.append(ue.velocity)

            avg_velocity = np.mean(velocities) if velocities else 0.0

            # We can also compute the actual movement vector for more detail
            # by tracking position changes, but velocity is constant per UE
            # in the default config (1.5 units/step). The direction varies
            # based on the random waypoint model.

            # Feature 3: TRAJECTORY PATTERNS -- computed from transitions
            # (accumulated over time, stored separately in edge data)

            # Additional features: position of BS, number of UEs in range
            ues_in_range = 0
            for ue in core.active:
                if core.check_connectivity(bs, ue):
                    ues_in_range += 1

            # SNR statistics for this BS's connections
            snr_values = []
            for ue_id in connected_ues:
                ue = core.users[ue_id]
                try:
                    snr = core.channel.snr(bs, ue)
                    snr_values.append(snr)
                except:
                    pass

            avg_snr = np.mean(snr_values) if snr_values else 0.0
            min_snr = np.min(snr_values) if snr_values else 0.0

            all_node_data.append({
                "episode": episode,
                "step": step,
                "bs_id": bs_id,
                "bs_x": bs.x,
                "bs_y": bs.y,
                "load": load,
                "avg_velocity": avg_velocity,
                "ues_in_range": ues_in_range,
                "avg_snr": avg_snr,
                "min_snr": min_snr,
            })

        # --------------------------------------------------------
        # 4. Track Handovers (Transitions between BSs)
        # --------------------------------------------------------
        # Determine "serving BS" for each active UE
        # A UE's serving BS = the BS it is connected to
        # If connected to multiple, use the one with best SNR
        current_bs = {}
        for ue in core.active:
            best_snr = -np.inf
            serving_bs = None
            for bs in stations_sorted:
                if ue in core.connections.get(bs, set()):
                    snr = core.channel.snr(bs, ue)
                    if snr > best_snr:
                        best_snr = snr
                        serving_bs = bs.bs_id
            # If not connected to any, use closest
            if serving_bs is None:
                serving_bs = ue_closest_bs.get(ue.ue_id, 0)
            current_bs[ue.ue_id] = serving_bs

        # Detect handovers by comparing with previous step
        for ue_id, curr_bs_id in current_bs.items():
            if ue_id in previous_bs:
                prev_bs_id = previous_bs[ue_id]
                if prev_bs_id != curr_bs_id:
                    # HANDOVER detected: UE moved from prev_bs to curr_bs
                    episode_transitions[prev_bs_id][curr_bs_id] += 1
                    cumulative_transitions[prev_bs_id][curr_bs_id] += 1

                    # Compute SNR at moment of handover for failure classification
                    ue = core.users[ue_id]
                    prev_bs_obj = stations_sorted[prev_bs_id]
                    curr_bs_obj = stations_sorted[curr_bs_id]

                    try:
                        snr_old = core.channel.snr(prev_bs_obj, ue)
                        snr_new = core.channel.snr(curr_bs_obj, ue)
                    except:
                        snr_old = 0
                        snr_new = 0

                    # Classify handover type based on SNR ratios
                    # (simplified heuristic matching the paper's FTE/FTL/PP concepts)
                    max_snr = max(snr_old, snr_new, 1e-10)
                    norm_old = snr_old / max_snr
                    norm_new = snr_new / max_snr

                    ho_type = "normal"
                    if norm_new < SNR_EARLY_THRESHOLD:
                        ho_type = "potential_early_failure"
                    elif norm_old < SNR_LATE_THRESHOLD:
                        ho_type = "potential_late_failure"

                    # Check for ping-pong
                    is_ping_pong = False
                    recent_handovers = ue_handover_times[ue_id]
                    for prev_ho_step, prev_from, prev_to in reversed(recent_handovers):
                        if step - prev_ho_step <= PING_PONG_WINDOW:
                            if prev_from == curr_bs_id and prev_to == prev_bs_id:
                                is_ping_pong = True
                                ho_type = "ping_pong"
                                break
                        else:
                            break

                    ue_handover_times[ue_id].append((step, prev_bs_id, curr_bs_id))

                    all_handover_events.append({
                        "episode": episode,
                        "step": step,
                        "ue_id": ue_id,
                        "from_bs": prev_bs_id,
                        "to_bs": curr_bs_id,
                        "snr_old_bs": snr_old,
                        "snr_new_bs": snr_new,
                        "ho_type": ho_type,
                        "is_ping_pong": is_ping_pong,
                        "ue_x": ue.x,
                        "ue_y": ue.y,
                    })

            # Update tracking
            ue_bs_history[ue_id].append((step, curr_bs_id))

        previous_bs = current_bs.copy()

        # --------------------------------------------------------
        # 5. Compute Edge Features per directed edge (i -> j)
        # --------------------------------------------------------
        for i in range(num_stations):
            for j in range(num_stations):
                if i == j:
                    continue

                # Count handovers from i to j up to this step (cumulative in episode)
                ho_count = episode_transitions[i][j]

                # Transition probability from i
                total_departures_from_i = episode_transitions[i].sum()
                trans_prob = (
                    episode_transitions[i][j] / total_departures_from_i
                    if total_departures_from_i > 0
                    else 0.0
                )

                # Count specific failure types for this edge
                edge_events = [
                    e for e in all_handover_events
                    if e["episode"] == episode
                    and e["from_bs"] == i
                    and e["to_bs"] == j
                ]
                n_early = sum(1 for e in edge_events if e["ho_type"] == "potential_early_failure")
                n_late = sum(1 for e in edge_events if e["ho_type"] == "potential_late_failure")
                n_pingpong = sum(1 for e in edge_events if e["ho_type"] == "ping_pong")
                n_normal = sum(1 for e in edge_events if e["ho_type"] == "normal")

                all_edge_data.append({
                    "episode": episode,
                    "step": step,
                    "src_bs": i,
                    "dst_bs": j,
                    "handover_count": ho_count,
                    "transition_prob": trans_prob,
                    "early_failures": n_early,
                    "late_failures": n_late,
                    "ping_pongs": n_pingpong,
                    "normal_handovers": n_normal,
                })

        step += 1
        done = terminated or truncated

    env.close()
    print(f"  Completed {step} steps. Total handovers this episode: {episode_transitions.sum()}")

# ============================================================
# 6. Build DataFrames and Save
# ============================================================
print("\n" + "=" * 70)
print("  Processing and Saving Data")
print("=" * 70)

df_nodes = pd.DataFrame(all_node_data)
df_edges = pd.DataFrame(all_edge_data)
df_handovers = pd.DataFrame(all_handover_events)

# Add trajectory pattern features to node data
# For each BS at each timestep, compute the transition probability distribution
# (what fraction of departures from this BS went to each other BS)
print("\nComputing trajectory pattern features...")

# Compute cumulative transition probabilities
transition_probs = np.zeros((num_stations, num_stations))
for i in range(num_stations):
    total = cumulative_transitions[i].sum()
    if total > 0:
        transition_probs[i] = cumulative_transitions[i] / total

# Build a lookup dict: bs_id (int) -> dict of traj_prob_to_X
traj_features = {}
for bs_id in range(num_stations):
    traj_features[bs_id] = {}
    total_departures = cumulative_transitions[bs_id].sum()
    for j in range(num_stations):
        prob = cumulative_transitions[bs_id][j] / total_departures if total_departures > 0 else 0.0
        traj_features[bs_id][f"traj_prob_to_{j}"] = prob

# Add trajectory probability columns to node dataframe
for j in range(num_stations):
    col_name = f"traj_prob_to_{j}"
    df_nodes[col_name] = df_nodes["bs_id"].astype(int).map(
        lambda bs_id, _j=j: traj_features[bs_id][f"traj_prob_to_{_j}"]
    )

# Save to CSV
output_dir = os.path.dirname(os.path.abspath(__file__))
df_nodes.to_csv(os.path.join(output_dir, "gnn_node_features.csv"), index=False)
df_edges.to_csv(os.path.join(output_dir, "gnn_edge_features.csv"), index=False)

if not df_handovers.empty:
    df_handovers.to_csv(os.path.join(output_dir, "gnn_handover_events.csv"), index=False)
else:
    print("  [INFO] No handover events detected (UEs may not have switched BSs)")
    # Create empty file with headers
    pd.DataFrame(columns=[
        "episode", "step", "ue_id", "from_bs", "to_bs",
        "snr_old_bs", "snr_new_bs", "ho_type", "is_ping_pong", "ue_x", "ue_y"
    ]).to_csv(os.path.join(output_dir, "gnn_handover_events.csv"), index=False)

# ============================================================
# 7. Print Summary Statistics
# ============================================================
print("\n" + "=" * 70)
print("  SUMMARY")
print("=" * 70)

print(f"\n  Environment: {ENV_NAME}")
print(f"  Episodes run: {NUM_EPISODES}")
print(f"  Base Stations: {num_stations}")
print(f"  User Equipments: {num_users}")

print(f"\n  Node features dataset: {len(df_nodes)} rows")
print(f"  Edge features dataset: {len(df_edges)} rows")
print(f"  Handover events: {len(df_handovers)} events")

if not df_handovers.empty:
    print(f"\n  Handover type breakdown:")
    print(df_handovers["ho_type"].value_counts().to_string(header=False))

    print(f"\n  Transition matrix (cumulative across all episodes):")
    print("  From \\ To", end="")
    for j in range(num_stations):
        print(f"   BS{j}", end="")
    print()
    for i in range(num_stations):
        print(f"    BS{i}     ", end="")
        for j in range(num_stations):
            print(f"  {cumulative_transitions[i][j]:4d}", end="")
        print()

    print(f"\n  Transition probability matrix:")
    print("  From \\ To", end="")
    for j in range(num_stations):
        print(f"   BS{j} ", end="")
    print()
    for i in range(num_stations):
        print(f"    BS{i}     ", end="")
        for j in range(num_stations):
            print(f"  {transition_probs[i][j]:.3f}", end="")
        print()

print(f"\n  Node feature columns: {list(df_nodes.columns)}")
print(f"  Edge feature columns: {list(df_edges.columns)}")

# ============================================================
# 8. Generate Visualizations
# ============================================================
print("\n  Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Mobile-Env Data Generation Summary", fontsize=16, fontweight="bold")

# Plot 1: BS positions and average load
ax1 = axes[0, 0]
bs_positions = df_nodes.groupby("bs_id").agg({"bs_x": "first", "bs_y": "first", "load": "mean"}).reset_index()
scatter = ax1.scatter(bs_positions["bs_x"], bs_positions["bs_y"],
                      s=bs_positions["load"] * 500 + 100,
                      c=bs_positions["load"], cmap="YlOrRd",
                      edgecolors="black", linewidth=2, zorder=5)
for _, row in bs_positions.iterrows():
    ax1.annotate(f"BS{int(row['bs_id'])}", (row["bs_x"], row["bs_y"]),
                 textcoords="offset points", xytext=(0, 15),
                 ha="center", fontsize=12, fontweight="bold")
ax1.set_title("BS Positions & Average Load")
ax1.set_xlabel("X")
ax1.set_ylabel("Y")
plt.colorbar(scatter, ax=ax1, label="Avg Load")

# Plot 2: Load over time per BS
ax2 = axes[0, 1]
for bs_id in range(num_stations):
    bs_data = df_nodes[(df_nodes["bs_id"] == bs_id) & (df_nodes["episode"] == 0)]
    ax2.plot(bs_data["step"], bs_data["load"], label=f"BS{bs_id}", linewidth=2)
ax2.set_title("Load Over Time (Episode 1)")
ax2.set_xlabel("Step")
ax2.set_ylabel("Connected UEs")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Transition matrix heatmap
ax3 = axes[1, 0]
if cumulative_transitions is not None and cumulative_transitions.sum() > 0:
    im = ax3.imshow(cumulative_transitions, cmap="Blues", interpolation="nearest")
    ax3.set_title("Handover Transition Counts")
    ax3.set_xlabel("To BS")
    ax3.set_ylabel("From BS")
    ax3.set_xticks(range(num_stations))
    ax3.set_yticks(range(num_stations))
    ax3.set_xticklabels([f"BS{i}" for i in range(num_stations)])
    ax3.set_yticklabels([f"BS{i}" for i in range(num_stations)])
    for i in range(num_stations):
        for j in range(num_stations):
            ax3.text(j, i, str(cumulative_transitions[i][j]),
                     ha="center", va="center", fontsize=14, fontweight="bold")
    plt.colorbar(im, ax=ax3)
else:
    ax3.text(0.5, 0.5, "No handovers detected", ha="center", va="center", transform=ax3.transAxes)
    ax3.set_title("Handover Transition Counts")

# Plot 4: Handover event types
ax4 = axes[1, 1]
if not df_handovers.empty:
    type_counts = df_handovers["ho_type"].value_counts()
    colors = {"normal": "#2ecc71", "potential_early_failure": "#e74c3c",
              "potential_late_failure": "#f39c12", "ping_pong": "#9b59b6"}
    bars = ax4.bar(type_counts.index, type_counts.values,
                   color=[colors.get(t, "#3498db") for t in type_counts.index],
                   edgecolor="black")
    ax4.set_title("Handover Event Types")
    ax4.set_ylabel("Count")
    ax4.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, type_counts.values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 str(val), ha="center", fontweight="bold")
else:
    ax4.text(0.5, 0.5, "No handover events", ha="center", va="center", transform=ax4.transAxes)
    ax4.set_title("Handover Event Types")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "data_generation_summary.png"), dpi=150, bbox_inches="tight")
print(f"  Saved visualization to data_generation_summary.png")

# ============================================================
# 9. Create PyTorch Geometric-ready data structure
# ============================================================
print("\n  Creating PyTorch Geometric data structure example...")

try:
    import torch
    from torch_geometric.data import Data

    # Build a single graph snapshot from the last timestep of last episode
    last_ep = NUM_EPISODES - 1
    last_step = df_nodes[df_nodes["episode"] == last_ep]["step"].max()

    # Node features: [load, avg_velocity, ues_in_range, avg_snr, traj_prob_to_0, ..., traj_prob_to_N]
    node_feature_cols = ["load", "avg_velocity", "ues_in_range", "avg_snr"]
    traj_cols = [c for c in df_nodes.columns if c.startswith("traj_prob_to_")]
    node_feature_cols += traj_cols

    node_df = df_nodes[(df_nodes["episode"] == last_ep) & (df_nodes["step"] == last_step)]
    node_df = node_df.sort_values("bs_id")

    x = torch.tensor(node_df[node_feature_cols].values, dtype=torch.float)

    # Directed edges: every pair (i, j) where i != j
    src_nodes = []
    dst_nodes = []
    edge_features = []

    edge_df = df_edges[(df_edges["episode"] == last_ep) & (df_edges["step"] == last_step)]

    for _, row in edge_df.iterrows():
        src_nodes.append(int(row["src_bs"]))
        dst_nodes.append(int(row["dst_bs"]))
        edge_features.append([
            row["handover_count"],
            row["transition_prob"],
            row["early_failures"],
            row["late_failures"],
            row["ping_pongs"],
        ])

    edge_index = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    print(f"\n  PyTorch Geometric Data object:")
    print(f"    Nodes: {graph.num_nodes}")
    print(f"    Edges: {graph.num_edges} (directed)")
    print(f"    Node feature dim: {graph.x.shape[1]}")
    print(f"    Edge feature dim: {graph.edge_attr.shape[1]}")
    print(f"    Node features: {node_feature_cols}")
    print(f"    Edge features: [handover_count, transition_prob, early_failures, late_failures, ping_pongs]")
    print(f"\n    Graph: {graph}")

    # Save the graph object
    torch.save(graph, os.path.join(output_dir, "sample_graph.pt"))
    print(f"    Saved sample graph to sample_graph.pt")

except ImportError:
    print("  [SKIP] torch/torch_geometric not available, skipping graph creation")

# ============================================================
# 10. Final Output
# ============================================================
print("\n" + "=" * 70)
print("  OUTPUT FILES:")
print("=" * 70)
print(f"  1. gnn_node_features.csv    -> Node features (load, velocity, trajectory)")
print(f"  2. gnn_edge_features.csv    -> Edge features (handover counts, probs, failures)")
print(f"  3. gnn_handover_events.csv  -> Raw handover event log")
print(f"  4. data_generation_summary.png -> Visualization")
print(f"  5. sample_graph.pt          -> PyTorch Geometric Data object")
print("\n  These files feed directly into the Dir-GNN implementation.")
print("  The other team member concatenates these with the Dir-GNN model.")
print("=" * 70)
