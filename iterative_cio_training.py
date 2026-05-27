"""
ITERATIVE CIO TRAINING - Main Pipeline Orchestrator

Purpose:
  Implements the complete training loop for CIO optimization:
  1. SIMULATE: Run mobile-env with current CIO matrix
  2. TRAIN: Train Directed GNN to predict edge failures
  3. UPDATE: Adjust CIO matrix based on predicted risks
  4. REPEAT: Iterate until convergence
  
Pipeline:
  initialize_cio() → for each iteration:
    └─ simulate_dataset()
    └─ train_gnn()
    └─ predict_risks()
    └─ update_cio_from_predictions()
    └─ save_results()
  
Key Functions:
  run_iterative_training() - Main loop
  simulate_dataset() - Collect handover data (mobile_environment.py)
  build_model() - Create GNN architecture (dirgnn_cio.py)
  update_cio_from_predictions() - Gradient-based CIO adjustment
  
Outputs:
  - CIO matrices per iteration (NPZ + CSV)
  - Training metrics CSV
  - Comprehensive visualization plots
  
USAGE:
  python iterative_cio_training.py \\
    --data-dir results/exp1 \\
    --iterations 4 \\
    --num-episodes 5 \\
    --env-name mobile-small-central-v0
"""

import argparse
import os
import pathlib
import random
import sys
from collections import defaultdict

ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mobile_env_local.env_loader import load_installed_mobile_env_package

installed_mobile_env = load_installed_mobile_env_package()
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

from mobile_env_local.data_utils import add_edge_cio, compute_failure_counts_by_category, EDGE_FEATURE_COLUMNS
from mobile_env_local.dirgnn_cio import DirGNNCIOModel, load_graph_snapshots, split_graphs, train_epoch, evaluate

# Import for visualizations
import seaborn as sns

# Environment size mappings (base stations per environment)
ENV_NUM_STATIONS = {
    "mobile-small-central-v0": 3,
    "mobile-small-ma-v0": 3,
    "mobile-medium-central-v0": 9,
    "mobile-medium-ma-v0": 9,
    "mobile-large-central-v0": 19,
    "mobile-large-ma-v0": 19,
}

# ============================================================
# 3GPP Handover Parameters (based on academic paper)
# ============================================================
# Equation (29): Mt - Ms > OI_t,j + hysteresis + CIO_i,j
HYS = 0.0           # Hysteresis (in dB, typically 0 for equal) 
OI_THRESHOLD = 3.0  # Global offset parameter (dB)

# Handover failure detection thresholds
FTE_THRESHOLD = 1.0   # Too Early Handover: SNR drops too fast after HO
FTL_THRESHOLD = 0.0   # Too Late Handover: SNR was too low before HO
PING_PONG_WINDOW = 8  # Steps to detect ping-pong


def sanitize_snr(value, clip=(-100.0, 100.0)):
    if not np.isfinite(value):
        return 0.0
    return float(np.clip(value, clip[0], clip[1]))


def detect_num_stations(env_name, fallback):
    env = gym.make(env_name, config={"seed": 0, "reset_rng_episode": True})
    try:
        env.reset(seed=0)
        return int(env.unwrapped.NUM_STATIONS)
    finally:
        env.close()

# ============================================================
# CIO Matrix Management
# ============================================================
def save_cio_matrix(cio_matrix, output_dir, iteration=None):
    """Save CIO matrix as NPZ and CSV (non-symmetric for directed graphs)."""
    os.makedirs(output_dir, exist_ok=True)
    
    if iteration is not None:
        npz_path = os.path.join(output_dir, f"cio_matrix_iter_{iteration:03d}.npz")
        csv_path = os.path.join(output_dir, f"cio_matrix_iter_{iteration:03d}.csv")
        final_path = os.path.join(output_dir, "cio_matrix_final.npz")
    else:
        npz_path = os.path.join(output_dir, "cio_matrix.npz")
        csv_path = os.path.join(output_dir, "cio_matrix.csv")
        final_path = npz_path
    
    np.savez(npz_path, cio_matrix=cio_matrix)
    pd.DataFrame(cio_matrix).to_csv(csv_path, index=False, header=False)
    
    # Always save final version
    np.savez(final_path, cio_matrix=cio_matrix)
    print(f"Saved CIO matrix to {npz_path} and {csv_path}")
    return npz_path, csv_path


def load_cio_matrix(cio_path, num_stations):
    """Load CIO matrix from NPZ or CSV file."""
    if cio_path.endswith('.npz'):
        data = np.load(cio_path)
        cio_matrix = data['cio_matrix']
    elif cio_path.endswith('.csv'):
        cio_matrix = pd.read_csv(cio_path, header=None).values
    else:
        raise ValueError(f"Unsupported CIO file format: {cio_path}")
    
    if cio_matrix.shape != (num_stations, num_stations):
        raise ValueError(f"CIO matrix shape {cio_matrix.shape} does not match expected shape ({num_stations}, {num_stations})")
    
    print(f"Loaded CIO matrix from {cio_path} with shape {cio_matrix.shape}")
    return cio_matrix.astype(float)


def build_action_for_ue(ue, stations_sorted, core, prev_bs_id, cio_matrix, cio_scale):
    """
    3GPP-based handover decision with CIO optimization.
    Equation (29): Mt - Ms > OI_t,j + hysteresis + CIO_i,j
    where Mt = target RSRP, Ms = serving RSRP, CIO_i,j = cell individual offset
    """
    serving_bs = None
    serving_snr = -float("inf")
    
    # Find current serving BS (best RSRP)
    for bs in stations_sorted:
        try:
            snr = core.channel.snr(bs, ue)
            if snr > serving_snr:
                serving_snr = snr
                serving_bs = bs
        except Exception:
            continue
    
    if serving_bs is None:
        return 1  # Default action
    
    best_target_bs = serving_bs
    best_score = -float("inf")
    
    # Evaluate handover candidates using 3GPP formula
    for target_bs in stations_sorted:
        try:
            target_snr = core.channel.snr(target_bs, ue)
        except Exception:
            continue
        
        # Apply 3GPP A3 handover criterion with CIO
        # Score = target_snr - serving_snr - OI_threshold - hysteresis - CIO_i,j
        cio_bias = 0.0
        if serving_bs.bs_id < cio_matrix.shape[0] and target_bs.bs_id < cio_matrix.shape[1]:
            cio_bias = cio_matrix[serving_bs.bs_id, target_bs.bs_id]
        
        # Higher CIO makes handover to target_bs more likely
        score = target_snr - serving_snr - OI_THRESHOLD - HYS + cio_scale * cio_bias
        
        if score > best_score:
            best_score = score
            best_target_bs = target_bs
    
    return best_target_bs.bs_id + 1


def get_serving_bs(core, ue, stations_sorted):
    best_snr = -float("inf")
    serving_bs = None
    for bs in stations_sorted:
        if ue in core.connections.get(bs, set()):
            try:
                snr = core.channel.snr(bs, ue)
            except Exception:
                continue
            if snr > best_snr:
                best_snr = snr
                serving_bs = bs.bs_id
    return serving_bs


def update_cio_online_from_events(cio_matrix, recent_events, learning_rate=0.02, clip=(-1.0, 1.0)):
    if not recent_events:
        return cio_matrix
    edge_updates = defaultdict(lambda: {"risk": 0.0, "count": 0})
    for event in recent_events:
        i = int(event["from_bs"])
        j = int(event["to_bs"])
        if i == j:
            continue
        risk = float(event.get("failure_severity", 0.0))
        if event.get("is_too_early", False):
            risk += 1.0
        if event.get("is_too_late", False):
            risk += 1.0
        if event.get("is_ping_pong", False):
            risk += 1.0
        edge_updates[(i, j)]["risk"] += risk
        edge_updates[(i, j)]["count"] += 1
    for (i, j), values in edge_updates.items():
        if values["count"] == 0:
            continue
        avg_risk = values["risk"] / values["count"]
        cio_matrix[i, j] = float(np.clip(cio_matrix[i, j] - learning_rate * avg_risk, clip[0], clip[1]))
    return cio_matrix


def simulate_dataset(output_dir, cio_matrix=None, cio_scale=1.0, seed_base=42, num_episodes=10, max_steps=80, env_name="mobile-small-central-v0", online_cio=False, online_cio_interval=10, online_cio_lr=0.02, fte_threshold=FTE_THRESHOLD, ftl_threshold=FTL_THRESHOLD, ping_pong_window=PING_PONG_WINDOW, ensure_bs_coverage=False, synthetic_neighbor_handovers=0, cio_clip=1.0):
    os.makedirs(output_dir, exist_ok=True)
    all_node_data = []
    all_edge_data = []
    all_handover_events = []
    cumulative_transitions = None
    num_stations = None
    num_users = None

    for episode in range(num_episodes):
        config = {"seed": seed_base + episode * 7, "reset_rng_episode": True}
        env = gym.make(env_name, config=config)
        obs, info = env.reset(seed=seed_base + episode * 7)
        core = env.unwrapped
        num_stations = core.NUM_STATIONS
        num_users = core.NUM_USERS

        if cumulative_transitions is None:
            cumulative_transitions = np.zeros((num_stations, num_stations), dtype=int)

        if cio_matrix is None:
            cio_matrix = np.zeros((num_stations, num_stations), dtype=float)

        episode_transitions = np.zeros((num_stations, num_stations), dtype=int)
        ue_handover_times = defaultdict(list)
        recent_handover_events = []
        previous_bs = {}
        step = 0
        done = False
        stations_sorted = sorted(core.stations.values(), key=lambda bs: bs.bs_id)

        while not done and step < max_steps:
            action = np.zeros(num_users, dtype=int)
            for ue_idx, ue_id in enumerate(sorted(core.users.keys())):
                ue = core.users[ue_id]
                if ue not in core.active:
                    action[ue_idx] = 0
                    continue
                prev_bs_id = previous_bs.get(ue_id, None)
                action[ue_idx] = build_action_for_ue(ue, stations_sorted, core, prev_bs_id, cio_matrix, cio_scale)

            next_obs, reward, terminated, truncated, info = env.step(action)

            bs_connected_ues = defaultdict(set)
            for bs in stations_sorted:
                for ue in core.connections.get(bs, set()):
                    bs_connected_ues[bs.bs_id].add(ue.ue_id)

            ue_closest_bs = {}
            for ue in core.active:
                min_dist = float("inf")
                closest = 0
                for bs in stations_sorted:
                    dist = np.hypot(ue.x - bs.x, ue.y - bs.y)
                    if dist < min_dist:
                        min_dist = dist
                        closest = bs.bs_id
                ue_closest_bs[ue.ue_id] = closest

            for bs in stations_sorted:
                bs_id = bs.bs_id
                connected_ues = bs_connected_ues[bs_id]
                load = len(connected_ues)
                velocities = [core.users[ue_id].velocity for ue_id in connected_ues]
                avg_velocity = float(np.mean(velocities)) if velocities else 0.0
                ues_in_range = sum(1 for ue in core.active if core.check_connectivity(bs, ue))
                snr_values = []
                for ue_id in connected_ues:
                    ue = core.users[ue_id]
                    try:
                        snr_values.append(core.channel.snr(bs, ue))
                    except Exception:
                        pass
                avg_snr = float(np.mean(snr_values)) if snr_values else 0.0
                min_snr = float(np.min(snr_values)) if snr_values else 0.0
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

            current_bs = {}
            for ue in core.active:
                serving_bs = get_serving_bs(core, ue, stations_sorted)
                current_bs[ue.ue_id] = serving_bs if serving_bs is not None else ue_closest_bs.get(ue.ue_id, 0)

            for ue_id, curr_bs_id in current_bs.items():
                if ue_id in previous_bs:
                    prev_bs_id = previous_bs[ue_id]
                    if prev_bs_id != curr_bs_id:
                        episode_transitions[prev_bs_id, curr_bs_id] += 1
                        cumulative_transitions[prev_bs_id, curr_bs_id] += 1
                        ue = core.users[ue_id]
                        prev_bs_obj = stations_sorted[prev_bs_id]
                        curr_bs_obj = stations_sorted[curr_bs_id]
                        try:
                            snr_old = sanitize_snr(core.channel.snr(prev_bs_obj, ue))
                        except Exception:
                            snr_old = 0.0
                        try:
                            snr_new = sanitize_snr(core.channel.snr(curr_bs_obj, ue))
                        except Exception:
                            snr_new = 0.0
                        snr_change = snr_new - snr_old
                        snr_drop = max(0.0, snr_old - snr_new)
                        is_too_early = snr_new < snr_old - fte_threshold
                        is_too_late = snr_old < ftl_threshold
                        early_severity = max(0.0, snr_drop - fte_threshold)
                        late_severity = max(0.0, ftl_threshold - snr_old)
                        is_ping_pong = False
                        ping_pong_severity = 0.0
                        for prev_ho_step, prev_from, prev_to in reversed(ue_handover_times[ue_id]):
                            if step - prev_ho_step <= ping_pong_window and prev_from == curr_bs_id and prev_to == prev_bs_id:
                                is_ping_pong = True
                                ping_pong_severity = 1.0 / max(1, step - prev_ho_step)
                                break
                            if step - prev_ho_step > ping_pong_window:
                                break
                        failure_labels = []
                        if is_too_early:
                            failure_labels.append("too_early_handover")
                        if is_too_late:
                            failure_labels.append("too_late_handover")
                        if is_ping_pong:
                            failure_labels.append("ping_pong")
                        ho_type = "+".join(failure_labels) if failure_labels else "normal"
                        failure_severity = early_severity + late_severity + ping_pong_severity
                        
                        ue_handover_times[ue_id].append((step, prev_bs_id, curr_bs_id))
                        event = {
                            "episode": episode,
                            "step": step,
                            "ue_id": ue_id,
                            "from_bs": prev_bs_id,
                            "to_bs": curr_bs_id,
                            "snr_old_bs": snr_old,
                            "snr_new_bs": snr_new,
                            "snr_change": snr_change,
                            "snr_drop": snr_drop,
                            "ho_type": ho_type,
                            "is_too_early": is_too_early,
                            "is_too_late": is_too_late,
                            "is_ping_pong": is_ping_pong,
                            "early_severity": early_severity,
                            "late_severity": late_severity,
                            "ping_pong_severity": ping_pong_severity,
                            "failure_severity": failure_severity,
                            "ue_x": ue.x,
                            "ue_y": ue.y,
                        }
                        all_handover_events.append(event)
                        recent_handover_events.append(event)
                previous_bs[ue_id] = curr_bs_id

            if online_cio and online_cio_interval > 0 and (step + 1) % online_cio_interval == 0:
                cio_matrix = update_cio_online_from_events(cio_matrix, recent_handover_events, learning_rate=online_cio_lr, clip=(-cio_clip, cio_clip))
                recent_handover_events = []

            for i in range(num_stations):
                for j in range(num_stations):
                    if i == j:
                        continue
                    ho_count = int(episode_transitions[i, j])
                    total_from_i = int(episode_transitions[i].sum())
                    trans_prob = float(ho_count / total_from_i) if total_from_i > 0 else 0.0
                    edge_events = [
                        e for e in all_handover_events
                        if e["episode"] == episode and e["from_bs"] == i and e["to_bs"] == j
                    ]
                    n_early = sum(1 for e in edge_events if e.get("is_too_early", False))
                    n_late = sum(1 for e in edge_events if e.get("is_too_late", False))
                    n_pingpong = sum(1 for e in edge_events if e.get("is_ping_pong", False))
                    n_normal = sum(1 for e in edge_events if e["ho_type"] == "normal")
                    n_failure = n_early + n_late + n_pingpong
                    early_severity = sum(float(e.get("early_severity", 0.0)) for e in edge_events)
                    late_severity = sum(float(e.get("late_severity", 0.0)) for e in edge_events)
                    ping_pong_severity = sum(float(e.get("ping_pong_severity", 0.0)) for e in edge_events)
                    failure_severity = sum(float(e.get("failure_severity", 0.0)) for e in edge_events)
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
                        "failure_count": n_failure,
                        "early_severity": early_severity,
                        "late_severity": late_severity,
                        "ping_pong_severity": ping_pong_severity,
                        "failure_severity": failure_severity,
                        "cio": float(cio_matrix[i, j]),
                        "normal_handovers": n_normal,
                    })

            step += 1
            done = terminated or truncated
        env.close()

    df_nodes = pd.DataFrame(all_node_data)
    df_edges = pd.DataFrame(all_edge_data)
    df_handovers = pd.DataFrame(all_handover_events)

    if num_stations is None:
        raise RuntimeError("No stations were simulated.")

    if ensure_bs_coverage or synthetic_neighbor_handovers > 0:
        covered_src = set(df_handovers["from_bs"].astype(int)) if len(df_handovers) else set()
        covered_dst = set(df_handovers["to_bs"].astype(int)) if len(df_handovers) else set()
        bs_positions = df_nodes.groupby("bs_id")[["bs_x", "bs_y"]].first()
        synthetic_events = []
        synthetic_pairs = set()
        if synthetic_neighbor_handovers > 0:
            for bs_id in range(num_stations):
                nearest_neighbors = sorted(
                    [candidate for candidate in range(num_stations) if candidate != bs_id],
                    key=lambda candidate: float(np.hypot(
                        bs_positions.loc[bs_id, "bs_x"] - bs_positions.loc[candidate, "bs_x"],
                        bs_positions.loc[bs_id, "bs_y"] - bs_positions.loc[candidate, "bs_y"],
                    )),
                )[:synthetic_neighbor_handovers]
                for neighbor in nearest_neighbors:
                    synthetic_pairs.add((bs_id, neighbor))
                    synthetic_pairs.add((neighbor, bs_id))
        if ensure_bs_coverage:
            missing_bs = sorted((set(range(num_stations)) - covered_src) | (set(range(num_stations)) - covered_dst))
            for bs_id in missing_bs:
                nearest = min(
                    [candidate for candidate in range(num_stations) if candidate != bs_id],
                    key=lambda candidate: float(np.hypot(
                        bs_positions.loc[bs_id, "bs_x"] - bs_positions.loc[candidate, "bs_x"],
                        bs_positions.loc[bs_id, "bs_y"] - bs_positions.loc[candidate, "bs_y"],
                    )),
                )
                synthetic_pairs.add((nearest, bs_id))
                synthetic_pairs.add((bs_id, nearest))
        for from_bs, to_bs in sorted(synthetic_pairs):
            synthetic_events.append({
                "episode": -1,
                "step": -1,
                "ue_id": -1,
                "from_bs": from_bs,
                "to_bs": to_bs,
                "snr_old_bs": 0.0,
                "snr_new_bs": 0.0,
                "snr_change": 0.0,
                "snr_drop": 0.0,
                "ho_type": "coverage_synthetic",
                "is_too_early": False,
                "is_too_late": False,
                "is_ping_pong": False,
                "early_severity": 0.0,
                "late_severity": 0.0,
                "ping_pong_severity": 0.0,
                "failure_severity": 0.0,
                "ue_x": bs_positions.loc[to_bs, "bs_x"],
                "ue_y": bs_positions.loc[to_bs, "bs_y"],
            })
            edge_mask = (df_edges["src_bs"] == from_bs) & (df_edges["dst_bs"] == to_bs)
            df_edges.loc[edge_mask, "handover_count"] = df_edges.loc[edge_mask, "handover_count"] + 1
            df_edges.loc[edge_mask, "normal_handovers"] = df_edges.loc[edge_mask, "normal_handovers"] + 1
            if cumulative_transitions is not None:
                cumulative_transitions[from_bs, to_bs] += 1
        if synthetic_events:
            df_handovers = pd.concat([df_handovers, pd.DataFrame(synthetic_events)], ignore_index=True)

    traj_features = {}
    for bs_id in range(num_stations):
        total_departures = cumulative_transitions[bs_id].sum()
        for j in range(num_stations):
            traj_features.setdefault(bs_id, {})[f"traj_prob_to_{j}"] = float(cumulative_transitions[bs_id, j] / total_departures) if total_departures > 0 else 0.0

    for j in range(num_stations):
        df_nodes[f"traj_prob_to_{j}"] = df_nodes["bs_id"].astype(int).map(lambda bs_id, _j=j: traj_features[bs_id][f"traj_prob_to_{_j}"])

    df_edges = add_edge_cio(df_edges)
    node_path = os.path.join(output_dir, "gnn_node_features.csv")
    edge_path = os.path.join(output_dir, "gnn_edge_features.csv")
    handover_path = os.path.join(output_dir, "gnn_handover_events.csv")
    df_nodes.to_csv(node_path, index=False)
    df_edges.to_csv(edge_path, index=False)
    df_handovers.to_csv(handover_path, index=False)

    if len(df_handovers) > 0:
        stats = {
            "early_failures": int(df_handovers["is_too_early"].sum()),
            "late_failures": int(df_handovers["is_too_late"].sum()),
            "ping_pongs": int(df_handovers["is_ping_pong"].sum()),
            "failure_count": int((df_handovers[["is_too_early", "is_too_late", "is_ping_pong"]].any(axis=1)).sum()),
        }
    else:
        stats = {"early_failures": 0, "late_failures": 0, "ping_pongs": 0, "failure_count": 0}
    stats["handover_count"] = len(df_handovers)
    stats["num_edges"] = len(df_edges)
    
    # ============================================================
    # DIAGNOSTIC: Threshold Sensitivity Analysis
    # ============================================================
    print("\n" + "="*70)
    print("DIAGNOSTIC ABLATION: FAILURE DETECTION BREAKDOWN")
    print("="*70)
    
    # Count handover events by SNR change
    if len(df_handovers) > 0:
        df_handovers['snr_change'] = df_handovers['snr_new_bs'] - df_handovers['snr_old_bs']
        
        # Count events meeting each threshold
        fte_candidates = (df_handovers['snr_new_bs'] < df_handovers['snr_old_bs'] - fte_threshold).sum()
        ftl_candidates = (df_handovers['snr_old_bs'] < ftl_threshold).sum()
        
        print(f"\n📊 CURRENT THRESHOLDS:")
        print(f"  • FTE threshold (too early):     {fte_threshold} dB")
        print(f"  • FTL threshold (too late):      {ftl_threshold} dB")
        print(f"  • Ping-pong window:              {ping_pong_window} steps")
        
        print(f"\n🔍 FAILURE DETECTION STATISTICS:")
        print(f"  Total handover events:           {len(df_handovers)}")
        print(f"  Events meeting FTE criterion:    {fte_candidates} ({100*fte_candidates/max(len(df_handovers),1):.1f}%)")
        print(f"  Events meeting FTL criterion:    {ftl_candidates} ({100*ftl_candidates/max(len(df_handovers),1):.1f}%)")
        
        print(f"\n📈 SNR CHANGE STATISTICS:")
        print(f"  Mean SNR change:                 {df_handovers['snr_change'].mean():.2f} dB")
        print(f"  Median SNR change:               {df_handovers['snr_change'].median():.2f} dB")
        print(f"  Min SNR change:                  {df_handovers['snr_change'].min():.2f} dB")
        print(f"  Max SNR change:                  {df_handovers['snr_change'].max():.2f} dB")
        print(f"  Std SNR change:                  {df_handovers['snr_change'].std():.2f} dB")
        
        # Count how many events have different severity levels
        severe_drops = (df_handovers['snr_change'] < -5.0).sum()
        moderate_drops = ((df_handovers['snr_change'] >= -5.0) & (df_handovers['snr_change'] < -2.0)).sum()
        mild_drops = ((df_handovers['snr_change'] >= -2.0) & (df_handovers['snr_change'] < 0)).sum()
        improvements = (df_handovers['snr_change'] >= 0).sum()
        
        print(f"\n📊 SNR CHANGE SEVERITY:")
        print(f"  Severe drops (< -5 dB):          {severe_drops} ({100*severe_drops/max(len(df_handovers),1):.1f}%)")
        print(f"  Moderate drops (-5 to -2 dB):    {moderate_drops} ({100*moderate_drops/max(len(df_handovers),1):.1f}%)")
        print(f"  Mild drops (-2 to 0 dB):         {mild_drops} ({100*mild_drops/max(len(df_handovers),1):.1f}%)")
        print(f"  Improvements (≥ 0 dB):           {improvements} ({100*improvements/max(len(df_handovers),1):.1f}%)")
        
        # Old SNR analysis
        print(f"\n📡 SIGNAL STRENGTH AT HANDOVER:")
        print(f"  Old BS SNR: mean={df_handovers['snr_old_bs'].mean():.2f} dB, min={df_handovers['snr_old_bs'].min():.2f} dB")
        print(f"  New BS SNR: mean={df_handovers['snr_new_bs'].mean():.2f} dB, min={df_handovers['snr_new_bs'].min():.2f} dB")
    
    print(f"\n✅ DETECTED FAILURES:")
    print(f"  Ping-pongs:                      {stats['ping_pongs']}")
    print(f"  Early failures (FTE):            {stats['early_failures']}")
    print(f"  Late failures (FTL):             {stats['late_failures']}")
    print(f"  Total failures:                  {stats['failure_count']}")
    print("="*70 + "\n")
    
    return df_nodes, df_edges, stats, cio_matrix


def build_model(num_node_features, edge_attr_dim, hidden_dim, num_layers, dropout, conv_type, alpha, device):
    model = DirGNNCIOModel(
        num_node_features=num_node_features,
        edge_attr_dim=edge_attr_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        conv_type=conv_type,
        alpha=alpha,
        normalize=False,
        learn_alpha=False,
    )
    return model.to(device)


def predict_edge_risks(model, graphs, device, classification=False, use_edge_attr=True):
    model.eval()
    preds = []
    with torch.no_grad():
        for graph in graphs:
            graph = graph.to(device)
            edge_attr = graph.edge_attr if use_edge_attr else None
            out = model(graph.x, graph.edge_index, edge_attr)
            if classification:
                out = torch.sigmoid(out)
            preds.append(out.cpu())
    return torch.cat(preds, dim=0).numpy()


def update_cio_from_predictions(edge_df, predictions, cio_matrix, learning_rate=0.4, clip=(-1.0, 1.0)):
    edge_df = edge_df.copy()
    predictions = np.nan_to_num(predictions, nan=0.0, posinf=0.0, neginf=0.0)
    edge_df["predicted_risk"] = predictions
    edge_df = edge_df[edge_df["handover_count"] > 0].copy()
    if edge_df.empty:
        return cio_matrix
    if edge_df["predicted_risk"].nunique() > 1:
        denom = edge_df["predicted_risk"].max() - edge_df["predicted_risk"].min()
        if denom == 0:
            edge_df["risk_norm"] = 0.5
        else:
            edge_df["risk_norm"] = (edge_df["predicted_risk"] - edge_df["predicted_risk"].min()) / denom
    else:
        edge_df["risk_norm"] = 0.5

    aggregated = edge_df.groupby(["src_bs", "dst_bs"])["risk_norm"].mean().reset_index()
    for _, row in aggregated.iterrows():
        i = int(row["src_bs"])
        j = int(row["dst_bs"])
        if i == j:
            continue
        delta = learning_rate * (row["risk_norm"] - 0.5)
        cio_matrix[i, j] = float(np.clip(cio_matrix[i, j] - delta, clip[0], clip[1]))
    return cio_matrix


def run_iterative_training(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    # Determine actual number of stations from environment
    actual_num_stations = detect_num_stations(args.env_name, args.num_stations)
    print(f"Environment: {args.env_name} ({actual_num_stations} base stations)")
    
    # Initialize or load CIO matrix
    if hasattr(args, 'cio_matrix') and args.cio_matrix is not None:
        cio_matrix = args.cio_matrix
        print(f"Using loaded CIO matrix with shape {cio_matrix.shape}")
    else:
        cio_matrix = np.zeros((actual_num_stations, actual_num_stations), dtype=float)
        print(f"Initialized new CIO matrix with shape {cio_matrix.shape} ({actual_num_stations}x{actual_num_stations})")
    
    metrics = []

    for iteration in range(args.iterations):
        print(f"\n=== Iteration {iteration + 1}/{args.iterations} ===")
        iteration_dir = os.path.join(args.data_dir, f"iteration_{iteration}")
        os.makedirs(iteration_dir, exist_ok=True)

        node_df, edge_df, stats, cio_matrix = simulate_dataset(
            iteration_dir,
            cio_matrix=cio_matrix,
            cio_scale=args.cio_scale,
            seed_base=args.seed + iteration * 100 if args.seed is not None else 42,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            env_name=args.env_name,
            online_cio=args.online_cio,
            online_cio_interval=args.online_cio_interval,
            online_cio_lr=args.online_cio_lr,
            fte_threshold=args.fte_threshold,
            ftl_threshold=args.ftl_threshold,
            ping_pong_window=args.ping_pong_window,
            ensure_bs_coverage=args.ensure_bs_coverage,
            synthetic_neighbor_handovers=args.synthetic_neighbor_handovers,
            cio_clip=args.cio_clip,
        )
        print(f"Simulation failure counts: {stats}")

        graphs = load_graph_snapshots(
            os.path.join(iteration_dir, "gnn_node_features.csv"),
            os.path.join(iteration_dir, "gnn_edge_features.csv"),
            root_dir=iteration_dir,
            target=args.target,
        )
        train_graphs, val_graphs, test_graphs = split_graphs(graphs, seed=args.seed)
        train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)

        sample = graphs[0]
        model = build_model(
            num_node_features=sample.x.shape[1],
            edge_attr_dim=0 if args.no_edge_attr else sample.edge_attr.shape[1],
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            conv_type=args.conv_type,
            alpha=args.alpha,
            device=device,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        classification = args.target in {"any_failure", "ping_pong"}
        if classification:
            train_labels = torch.cat([graph.y for graph in train_graphs])
            positives = train_labels.sum()
            negatives = train_labels.numel() - positives
            pos_weight = (negatives / positives.clamp_min(1.0)).to(device)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = torch.nn.MSELoss()

        for epoch in range(1, args.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device, clip_norm=args.clip_norm, use_edge_attr=not args.no_edge_attr)
            val_metrics = evaluate(model, val_loader, criterion, device, classification=classification, use_edge_attr=not args.no_edge_attr)
            if epoch % max(1, args.epochs // 5) == 0 or epoch == args.epochs:
                print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f}" + (f" val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} val_pos={val_metrics['positive_rate']:.4f} pred_pos={val_metrics['predicted_positive_rate']:.4f}" if classification else f" val_mae={val_metrics['mae']:.4f}"))

        preds = predict_edge_risks(model, graphs, device, classification=classification, use_edge_attr=not args.no_edge_attr)
        cio_matrix = update_cio_from_predictions(edge_df.sort_values(["episode", "step", "src_bs", "dst_bs"]).reset_index(drop=True), preds, cio_matrix, learning_rate=args.cio_lr, clip=(-args.cio_clip, args.cio_clip))
        
        # Save CIO matrix at each iteration
        save_cio_matrix(cio_matrix, args.data_dir, iteration=iteration)

        metrics.append({
            "iteration": iteration,
            "failure_count": stats["failure_count"],
            "early_failures": stats["early_failures"],
            "late_failures": stats["late_failures"],
            "ping_pongs": stats["ping_pongs"],
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics.get("mae", None),
            "val_accuracy": val_metrics.get("accuracy", None),
            "val_precision": val_metrics.get("precision", None),
            "val_recall": val_metrics.get("recall", None),
            "val_f1": val_metrics.get("f1", None),
            "val_positive_rate": val_metrics.get("positive_rate", None),
            "val_predicted_positive_rate": val_metrics.get("predicted_positive_rate", None),
            "val_tp": val_metrics.get("tp", None),
            "val_tn": val_metrics.get("tn", None),
            "val_fp": val_metrics.get("fp", None),
            "val_fn": val_metrics.get("fn", None),
        })

    results_df = pd.DataFrame(metrics)
    results_path = os.path.join(args.data_dir, "cio_training_progress.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Saved iterative training log to {results_path}")

    # Save final CIO matrix
    save_cio_matrix(cio_matrix, args.data_dir, iteration=None)
    
    sns.set_theme(style="whitegrid", palette="deep")
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.28)

    # Plot 1: Failure counts evolution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(results_df["iteration"], results_df["failure_count"], marker="o", linewidth=2.8, markersize=7, label="Total", color="#d62728")
    ax1.plot(results_df["iteration"], results_df["early_failures"], marker="X", linewidth=2.0, label="Early", color="#ff7f0e")
    ax1.plot(results_df["iteration"], results_df["late_failures"], marker="s", linewidth=2.0, label="Late", color="#9467bd")
    ax1.plot(results_df["iteration"], results_df["ping_pongs"], marker="^", linewidth=2.0, label="Ping-pong", color="#8c564b")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Event count")
    ax1.set_title("Failure evolution by CIO update", fontweight="bold")
    ax1.legend(frameon=True)

    # Plot 2: Training loss over iterations
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(results_df["iteration"], results_df["train_loss"], marker="o", linewidth=2.4, markersize=7, label="Train", color="#1f77b4")
    ax2.plot(results_df["iteration"], results_df["val_loss"], marker="s", linewidth=2.4, markersize=7, label="Validation", color="#2ca02c")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training and validation loss", fontweight="bold")
    ax2.legend(frameon=True)
    ax2.set_yscale("log")

    # Plot 3: Validation accuracy (if available)
    if "val_accuracy" in results_df.columns and results_df["val_accuracy"].notna().any():
        ax3 = fig.add_subplot(gs[1, 0])
        valid_acc = results_df[results_df["val_accuracy"].notna()]
        ax3.plot(valid_acc["iteration"], valid_acc["val_accuracy"], marker="o", linewidth=2.2, markersize=7, label="Accuracy", color="#2ca02c")
        if "val_f1" in valid_acc.columns and valid_acc["val_f1"].notna().any():
            ax3.plot(valid_acc["iteration"], valid_acc["val_f1"], marker="D", linewidth=2.2, markersize=6, label="F1", color="#17becf")
        if "val_precision" in valid_acc.columns and valid_acc["val_precision"].notna().any():
            ax3.plot(valid_acc["iteration"], valid_acc["val_precision"], marker="^", linewidth=1.8, markersize=6, label="Precision", color="#bcbd22")
        if "val_recall" in valid_acc.columns and valid_acc["val_recall"].notna().any():
            ax3.plot(valid_acc["iteration"], valid_acc["val_recall"], marker="v", linewidth=1.8, markersize=6, label="Recall", color="#e377c2")
        ax3.set_xlabel("Iteration")
        ax3.set_ylabel("Accuracy")
        ax3.set_title("Classification metrics", fontweight="bold")
        ax3.set_ylim([0, 1.05])
        ax3.legend(frameon=True)
    
    # Plot 4: Validation MAE (if available)
    if "val_mae" in results_df.columns and results_df["val_mae"].notna().any():
        ax4 = fig.add_subplot(gs[1, 1])
        valid_mae = results_df[results_df["val_mae"].notna()]
        ax4.plot(valid_mae["iteration"], valid_mae["val_mae"], marker="s", linewidth=2.4, markersize=7, color="#9467bd")
        ax4.set_xlabel("Iteration")
        ax4.set_ylabel("MAE")
        ax4.set_title("Validation MAE", fontweight="bold")
    
    # Plot 5: CIO matrix heatmap (final)
    ax5 = fig.add_subplot(gs[2, :])
    im = ax5.imshow(cio_matrix, cmap="RdBu_r", aspect="auto", vmin=-1.0, vmax=1.0)
    ax5.set_xlabel("Destination BS")
    ax5.set_ylabel("Source BS")
    ax5.set_title("Final directed CIO matrix", fontweight="bold")
    cbar = plt.colorbar(im, ax=ax5)
    cbar.set_label("CIO value")
    
    # Add grid
    ax5.set_xticks(np.arange(cio_matrix.shape[1]))
    ax5.set_yticks(np.arange(cio_matrix.shape[0]))
    for i in range(cio_matrix.shape[0]):
        for j in range(cio_matrix.shape[1]):
            text = ax5.text(j, i, f'{cio_matrix[i, j]:.2f}',
                          ha="center", va="center", color="black", fontsize=7)

    fig_path = os.path.join(args.data_dir, "cio_training_comprehensive.png")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    print(f"Saved comprehensive plots to {fig_path}")

    fig_dist, axes = plt.subplots(1, 2, figsize=(16, 6))
    failure_totals = results_df[["early_failures", "late_failures", "ping_pongs"]].sum()
    sns.barplot(x=failure_totals.index, y=failure_totals.values, ax=axes[0], palette=["#ff7f0e", "#9467bd", "#8c564b"])
    axes[0].set_title("Total failures by type", fontweight="bold")
    axes[0].set_ylabel("Count")
    axes[0].set_xlabel("Failure type")
    axes[0].tick_params(axis="x", rotation=20)
    sns.heatmap(results_df[["early_failures", "late_failures", "ping_pongs"]].T, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label": "Count"}, ax=axes[1])
    axes[1].set_title("Failure type per iteration", fontweight="bold")
    axes[1].set_xlabel("Iteration index")
    axes[1].set_ylabel("Failure type")
    fig_dist.tight_layout()
    dist_path = os.path.join(args.data_dir, "failure_distribution.png")
    fig_dist.savefig(dist_path, dpi=150, bbox_inches="tight")
    print(f"Saved failure distribution plot to {dist_path}")
    
    # Create a separate heatmap figure for CIO matrix
    fig_hm, ax_hm = plt.subplots(figsize=(10, 8))
    sns.heatmap(cio_matrix, annot=True, fmt=".3f", cmap="RdBu_r", center=0.0,
                cbar_kws={"label": "CIO value"}, ax=ax_hm, vmin=-1.0, vmax=1.0)
    ax_hm.set_xlabel("Destination BS")
    ax_hm.set_ylabel("Source BS")
    ax_hm.set_title("Matrice CIO finale (Heatmap détaillée)")
    fig_hm.tight_layout()
    hm_path = os.path.join(args.data_dir, "cio_matrix_heatmap.png")
    fig_hm.savefig(hm_path, dpi=150, bbox_inches="tight")
    print(f"Saved CIO heatmap to {hm_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative CIO training with risk prediction and policy updates")
    parser.add_argument("--data-dir", type=str, default="mobile_env", help="Directory to save intermediate datasets and results")
    parser.add_argument("--iterations", type=int, default=10, help="Number of iterative CIO training cycles")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes per simulation")
    parser.add_argument("--max-steps", type=int, default=60, help="Max steps per episode")
    parser.add_argument("--target", type=str, default="any_failure", choices=["any_failure", "failure_count", "ping_pong", "early_failures", "late_failures", "ping_pongs", "early_severity", "late_severity", "ping_pong_severity", "failure_severity"], help="Prediction target used for CIO update")
    parser.add_argument("--conv-type", type=str, default="dir-gcn", choices=["gcn", "sage", "gat", "dir-gcn", "dir-sage", "dir-gat"], help="Conv type for the GNN")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--cio-lr", type=float, default=0.12, help="Learning rate for CIO update from predicted risk")
    parser.add_argument("--cio-scale", type=float, default=0.8, help="Scale factor for CIO bias in action selection")
    parser.add_argument("--cio-clip", type=float, default=1.0, help="Symmetric absolute clipping bound for CIO values")
    parser.add_argument("--online-cio", action="store_true", help="Update CIO dynamically during each simulation")
    parser.add_argument("--online-cio-interval", type=int, default=10, help="Simulation steps between online CIO updates")
    parser.add_argument("--online-cio-lr", type=float, default=0.02, help="Learning rate for online CIO updates inside simulation")
    parser.add_argument("--fte-threshold", type=float, default=FTE_THRESHOLD, help="SNR drop threshold for too-early handover detection")
    parser.add_argument("--ftl-threshold", type=float, default=FTL_THRESHOLD, help="Serving SNR threshold for too-late handover detection")
    parser.add_argument("--ping-pong-window", type=int, default=PING_PONG_WINDOW, help="Step window for ping-pong handover detection")
    parser.add_argument("--ensure-bs-coverage", action="store_true", help="Add synthetic normal nearest-neighbor handovers so every BS has incoming and outgoing coverage")
    parser.add_argument("--synthetic-neighbor-handovers", type=int, default=0, help="Add synthetic bidirectional normal handovers between each BS and its k nearest neighboring BSs")
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--no-edge-attr", action="store_true", help="Do not use edge attributes in the model")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-stations", type=int, default=None, help="Number of stations for the CIO matrix (auto-detected from env if not specified)")
    parser.add_argument("--load-cio", type=str, default=None, help="Path to load existing CIO matrix (NPZ or CSV)")
    parser.add_argument("--env-name", type=str, default="mobile-medium-central-v0", 
                       choices=["mobile-small-central-v0", "mobile-small-ma-v0", 
                               "mobile-medium-central-v0", "mobile-medium-ma-v0",
                               "mobile-large-central-v0", "mobile-large-ma-v0"],
                       help="Environment name/scenario to use")
    args = parser.parse_args()
        # Auto-detect number of stations if not explicitly specified
    if args.num_stations is None:
        args.num_stations = detect_num_stations(args.env_name, ENV_NUM_STATIONS.get(args.env_name, 9))
        # Load CIO matrix if provided
    if args.load_cio is not None:
        args.cio_matrix = load_cio_matrix(args.load_cio, args.num_stations)
    else:
        args.cio_matrix = None
    
    print(f"Using environment: {args.env_name}")
    run_iterative_training(args)
