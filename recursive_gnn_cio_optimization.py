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
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from mobile_env_local_2.env_loader import load_installed_mobile_env_package
from mobile_env_local_2.dirgnn_cio import DirGNNCIOModel
from mobile_env_local_2.iterative_cio_training import detect_num_stations, sanitize_snr

load_installed_mobile_env_package()


EDGE_FEATURE_COLUMNS = [
    "handover_count",
    "transition_prob",
    "normal_handovers",
    "cio",
    "early_severity",
    "late_severity",
    "ping_pong_severity",
    "failure_severity",
]

NODE_FEATURE_COLUMNS = [
    "load",
    "avg_velocity",
    "ues_in_range",
    "avg_snr",
    "min_snr",
]


def get_serving_bs(core, ue, stations_sorted):
    best_snr = -float("inf")
    serving_bs = None
    for bs in stations_sorted:
        if ue in core.connections.get(bs, set()):
            try:
                snr = sanitize_snr(core.channel.snr(bs, ue))
            except Exception:
                continue
            if snr > best_snr:
                best_snr = snr
                serving_bs = bs.bs_id
    return serving_bs


def compute_node_rows(core, stations_sorted, episode, step):
    rows = []
    for bs in stations_sorted:
        connected_ues = core.connections.get(bs, set())
        velocities = [ue.velocity for ue in connected_ues]
        snr_values = []
        for ue in connected_ues:
            try:
                snr_values.append(sanitize_snr(core.channel.snr(bs, ue)))
            except Exception:
                pass
        rows.append({
            "episode": episode,
            "step": step,
            "bs_id": bs.bs_id,
            "bs_x": bs.x,
            "bs_y": bs.y,
            "load": len(connected_ues),
            "avg_velocity": float(np.mean(velocities)) if velocities else 0.0,
            "ues_in_range": sum(1 for ue in core.active if core.check_connectivity(bs, ue)),
            "avg_snr": float(np.mean(snr_values)) if snr_values else 0.0,
            "min_snr": float(np.min(snr_values)) if snr_values else 0.0,
        })
    return rows


def observed_risk_from_edge_df(edge_df, ping_pong_weight):
    active = edge_df["handover_count"].astype(float).clip(lower=1.0)
    raw = edge_df["failure_count"].astype(float)
    risk = (raw / active).clip(lower=0.0, upper=1.0)
    return risk.values.astype(np.float32)


def edge_loss_weight_from_edge_df(edge_df, inactive_weight):
    weights = np.where(edge_df["handover_count"].astype(float).values > 0, 1.0, inactive_weight)
    return weights.astype(np.float32)


def summarize_events(events):
    return {
        "handover_count": len(events),
        "failure_count": sum(1 for e in events if e["is_failure"]),
        "early_failures": sum(1 for e in events if e["failure_class"] == "too_early_handover"),
        "late_failures": sum(1 for e in events if e["failure_class"] == "too_late_handover"),
        "ping_pongs": sum(1 for e in events if e["is_ping_pong"]),
    }


def make_recursive_edge_rows(episode, step, num_stations, step_transitions, cumulative_transitions, step_events, cio_matrix):
    rows = []
    for i in range(num_stations):
        total_from_i = int(cumulative_transitions[i].sum())
        for j in range(num_stations):
            if i == j:
                continue
            edge_events = [e for e in step_events if e["from_bs"] == i and e["to_bs"] == j]
            rows.append({
                "episode": episode,
                "step": step,
                "src_bs": i,
                "dst_bs": j,
                "handover_count": int(step_transitions[i, j]),
                "transition_prob": float(cumulative_transitions[i, j] / total_from_i) if total_from_i > 0 else 0.0,
                "early_failures": sum(1 for e in edge_events if e["failure_class"] == "too_early_handover"),
                "late_failures": sum(1 for e in edge_events if e["failure_class"] == "too_late_handover"),
                "ping_pongs": sum(1 for e in edge_events if e["is_ping_pong"]),
                "failure_count": sum(1 for e in edge_events if e["is_failure"]),
                "early_severity": sum(float(e["early_severity"]) for e in edge_events),
                "late_severity": sum(float(e["late_severity"]) for e in edge_events),
                "ping_pong_severity": sum(float(e["ping_pong_severity"]) for e in edge_events),
                "failure_severity": sum(float(e["failure_severity"]) for e in edge_events),
                "normal_handovers": sum(1 for e in edge_events if not e["is_failure"]),
                "cio": float(cio_matrix[i, j]),
            })
    return rows


def safe_snr(core, bs, ue):
    try:
        value = float(core.channel.snr(bs, ue))
    except Exception:
        return 0.0
    return float(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))


def median_active_snr(core, stations_sorted):
    values = []
    for ue in core.active:
        for bs in stations_sorted:
            values.append(safe_snr(core, bs, ue))
    return float(np.median(values)) if values else 0.0


def detect_scaled_step_events(core, stations_sorted, previous_bs, current_bs, ue_handover_times, episode, step, args):
    events = []
    reference_snr = max(median_active_snr(core, stations_sorted), 1e-12)
    ftl_threshold = args.ftl_threshold if args.absolute_failure_thresholds else args.ftl_relative_threshold * reference_snr
    for ue_id, curr_bs_id in current_bs.items():
        if ue_id not in previous_bs:
            previous_bs[ue_id] = curr_bs_id
            continue
        prev_bs_id = previous_bs[ue_id]
        if prev_bs_id == curr_bs_id:
            previous_bs[ue_id] = curr_bs_id
            continue
        ue = core.users[ue_id]
        prev_bs_obj = stations_sorted[prev_bs_id]
        curr_bs_obj = stations_sorted[curr_bs_id]
        snr_old = safe_snr(core, prev_bs_obj, ue)
        snr_new = safe_snr(core, curr_bs_obj, ue)
        snr_drop = max(0.0, snr_old - snr_new)
        snr_gain_ratio = snr_new / max(snr_old, 1e-12)
        snr_similarity_ratio = min(snr_new, snr_old) / max(max(snr_new, snr_old), 1e-12)
        if args.absolute_failure_thresholds:
            is_too_early = snr_new < snr_old - args.fte_threshold
            early_severity = max(0.0, snr_drop - args.fte_threshold)
        else:
            is_too_early = snr_gain_ratio < args.too_early_ratio
            early_severity = max(0.0, args.too_early_ratio - snr_gain_ratio)
        late_by_low_serving = snr_old < ftl_threshold
        late_by_large_gain = snr_gain_ratio > args.too_late_ratio
        is_too_late = late_by_low_serving or late_by_large_gain
        late_severity = max(0.0, (ftl_threshold - snr_old) / reference_snr, snr_gain_ratio - args.too_late_ratio)
        early_severity = min(1.0, early_severity)
        late_severity = min(1.0, late_severity)
        is_ping_pong = False
        ping_pong_severity = 0.0
        time_since_reverse_handover = None
        last_ping_pong_step = None
        for previous_event in reversed(ue_handover_times[ue_id]):
            if len(previous_event) >= 4 and previous_event[3]:
                last_ping_pong_step = previous_event[0]
                break
        for previous_event in reversed(ue_handover_times[ue_id]):
            prev_ho_step, prev_from, prev_to = previous_event[:3]
            stay_time = step - prev_ho_step
            if prev_from == curr_bs_id and prev_to == prev_bs_id:
                time_since_reverse_handover = stay_time
            if stay_time <= args.min_stay_steps and prev_from == curr_bs_id and prev_to == prev_bs_id:
                if last_ping_pong_step is None or step - last_ping_pong_step > args.ping_pong_window:
                    is_ping_pong = True
                    ping_pong_severity = 1.0 - stay_time / max(args.min_stay_steps + 1, 1)
                break
            if stay_time > args.min_stay_steps:
                break
        is_valid_window = args.valid_snr_gain_min <= snr_gain_ratio <= args.valid_snr_gain_max
        if is_valid_window:
            is_too_early = False
            is_too_late = False
            early_severity = 0.0
            late_severity = 0.0
        labels = []
        if is_too_early:
            labels.append("too_early_handover")
        if is_too_late:
            labels.append("too_late_handover")
        if is_ping_pong:
            labels.append("ping_pong")
        failure_scores = {
            "too_early_handover": early_severity if is_too_early else 0.0,
            "too_late_handover": late_severity if is_too_late else 0.0,
        }
        failure_class = max(failure_scores, key=failure_scores.get)
        is_failure = failure_scores[failure_class] >= args.min_failure_severity
        if not is_failure:
            failure_class = "normal"
        ue_handover_times[ue_id].append((step, prev_bs_id, curr_bs_id, is_ping_pong))
        events.append({
            "episode": episode,
            "step": step,
            "ue_id": ue_id,
            "from_bs": prev_bs_id,
            "to_bs": curr_bs_id,
            "snr_old_bs": snr_old,
            "snr_new_bs": snr_new,
            "snr_change": snr_new - snr_old,
            "snr_drop": snr_drop,
            "snr_gain_ratio": snr_gain_ratio,
            "snr_similarity_ratio": snr_similarity_ratio,
            "reference_snr": reference_snr,
            "valid_snr_similarity_min": args.valid_snr_similarity_min,
            "valid_snr_gain_min": args.valid_snr_gain_min,
            "valid_snr_gain_max": args.valid_snr_gain_max,
            "too_early_ratio": args.too_early_ratio,
            "too_late_ratio": args.too_late_ratio,
            "is_valid_window": is_valid_window,
            "time_since_reverse_handover": time_since_reverse_handover,
            "ho_type": "+".join(labels) if labels else "normal",
            "failure_class": failure_class,
            "is_failure": is_failure,
            "is_too_early": is_too_early,
            "is_too_late": is_too_late,
            "is_ping_pong": is_ping_pong,
            "early_severity": early_severity,
            "late_severity": late_severity,
            "ping_pong_severity": ping_pong_severity,
            "failure_severity": max(failure_scores.values()),
            "ue_x": ue.x,
            "ue_y": ue.y,
        })
        previous_bs[ue_id] = curr_bs_id
    return events


def build_a3_action_for_ue(ue, stations_sorted, core, cio_matrix, cio_scale, off_a3, hys_a3):
    serving_bs = None
    serving_snr = -float("inf")
    for bs in stations_sorted:
        if ue in core.connections.get(bs, set()):
            try:
                snr = float(core.channel.snr(bs, ue))
            except Exception:
                continue
            if snr > serving_snr:
                serving_snr = snr
                serving_bs = bs
    if serving_bs is None:
        return 1
    candidates = []
    for target_bs in stations_sorted:
        if target_bs.bs_id == serving_bs.bs_id:
            continue
        try:
            target_snr = float(core.channel.snr(target_bs, ue))
        except Exception:
            continue
        cio_bias = 0.0
        if serving_bs.bs_id < cio_matrix.shape[0] and target_bs.bs_id < cio_matrix.shape[1]:
            cio_bias = float(cio_matrix[serving_bs.bs_id, target_bs.bs_id])
        snr_gain = target_snr - serving_snr
        a3_margin = snr_gain - off_a3 - hys_a3 - cio_scale * cio_bias
        if a3_margin > 0.0:
            candidates.append((a3_margin, target_bs.bs_id))
    if not candidates:
        return serving_bs.bs_id + 1
    return max(candidates, key=lambda item: item[0])[1] + 1


def build_risk_graph(node_rows, edge_rows, ping_pong_weight, inactive_weight):
    node_df = pd.DataFrame(node_rows).sort_values("bs_id")
    edge_df = pd.DataFrame(edge_rows).sort_values(["src_bs", "dst_bs"])
    x_values = np.nan_to_num(node_df[NODE_FEATURE_COLUMNS].values, nan=0.0, posinf=100.0, neginf=-100.0)
    edge_values = np.nan_to_num(edge_df[EDGE_FEATURE_COLUMNS].values, nan=0.0, posinf=100.0, neginf=-100.0)
    x = torch.tensor(np.clip(x_values, -100.0, 100.0), dtype=torch.float)
    edge_index = torch.tensor(edge_df[["src_bs", "dst_bs"]].values.T, dtype=torch.long)
    edge_attr = torch.tensor(np.clip(edge_values, -100.0, 100.0), dtype=torch.float)
    y = torch.tensor(observed_risk_from_edge_df(edge_df, ping_pong_weight), dtype=torch.float)
    edge_weight = torch.tensor(edge_loss_weight_from_edge_df(edge_df, inactive_weight), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.edge_loss_weight = edge_weight
    return data


def build_model(sample_graph, args, device):
    model = DirGNNCIOModel(
        num_node_features=sample_graph.x.shape[1],
        edge_attr_dim=sample_graph.edge_attr.shape[1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        conv_type=args.conv_type,
        alpha=args.alpha,
        normalize=False,
        learn_alpha=False,
    )
    return model.to(device)


def weighted_mse_loss(pred, target, weight):
    risk = torch.sigmoid(pred)
    return (weight * (risk - target).pow(2)).sum() / weight.sum().clamp_min(1e-8)


def train_risk_model(model, graphs, args, device):
    loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    last_loss = None
    for _ in range(args.epochs_per_iteration):
        model.train()
        total = 0.0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = weighted_mse_loss(pred, batch.y, batch.edge_loss_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()
            total += loss.item() * batch.num_graphs
        last_loss = total / max(len(loader.dataset), 1)
    return float(last_loss) if last_loss is not None else None


def predict_edge_risk(model, graph, num_stations, device):
    model.eval()
    with torch.no_grad():
        graph = graph.to(device)
        pred = torch.sigmoid(model(graph.x, graph.edge_index, graph.edge_attr)).detach().cpu().numpy()
        edge_index = graph.edge_index.detach().cpu().numpy()
    risk_matrix = np.zeros((num_stations, num_stations), dtype=float)
    for value, src, dst in zip(pred, edge_index[0], edge_index[1]):
        risk_matrix[int(src), int(dst)] = float(np.clip(value, 0.0, 1.0))
    return risk_matrix


def update_cio_from_risk(cio_matrix, risk_matrix, activity_matrix, args):
    new_cio = cio_matrix.copy()
    mask = ~np.eye(cio_matrix.shape[0], dtype=bool)
    active_mask = (activity_matrix > 0) & mask
    if active_mask.any():
        for i in range(cio_matrix.shape[0]):
            row_mask = mask[i]
            if args.update_active_edges_only:
                row_mask = row_mask & (activity_matrix[i] > 0)
            if not row_mask.any():
                continue
            row_risk = risk_matrix[i, row_mask]
            row_center = float(np.mean(row_risk))
            row_delta = np.zeros(cio_matrix.shape[1], dtype=float)
            for j in range(cio_matrix.shape[1]):
                if i == j:
                    continue
                if args.update_active_edges_only and activity_matrix[i, j] <= 0:
                    continue
                centered = risk_matrix[i, j] - row_center
                activity_scale = min(1.0, activity_matrix[i, j] / max(args.activity_normalizer, 1.0))
                row_delta[j] = args.cio_lr * centered * max(activity_scale, args.inactive_update_scale)
            if row_mask.any():
                row_delta[row_mask] -= float(np.mean(row_delta[row_mask]))
            row_delta = np.clip(row_delta, -args.max_cio_step, args.max_cio_step)
            for j in range(cio_matrix.shape[1]):
                if i == j:
                    continue
                if args.update_active_edges_only and activity_matrix[i, j] <= 0:
                    continue
                new_cio[i, j] = float(np.clip(new_cio[i, j] + row_delta[j], -args.cio_clip, args.cio_clip))
            new_cio[i, mask[i]] -= float(np.mean(new_cio[i, mask[i]]))
            new_cio[i, mask[i]] = np.clip(new_cio[i, mask[i]], -args.cio_clip, args.cio_clip)
    np.fill_diagonal(new_cio, 0.0)
    return new_cio


def relax_cio_toward_previous(current_cio, previous_cio, factor):
    relaxed = previous_cio + factor * (current_cio - previous_cio)
    np.fill_diagonal(relaxed, 0.0)
    return relaxed


def simulate_with_cio(cio_matrix, args, iteration):
    num_stations = cio_matrix.shape[0]
    all_node_rows = []
    all_edge_rows = []
    all_events = []
    graphs = []
    activity_matrix = np.zeros_like(cio_matrix, dtype=float)
    cumulative_transitions = np.zeros_like(cio_matrix, dtype=int)
    for episode in range(args.num_episodes):
        base_seed = args.seed if args.seed is not None else 42
        seed = base_seed + episode * 7 if args.fixed_scenario else base_seed + iteration * 1000 + episode * 7
        env = gym.make(args.env_name, config={"seed": seed, "reset_rng_episode": True})
        env.reset(seed=seed)
        core = env.unwrapped
        stations_sorted = sorted(core.stations.values(), key=lambda bs: bs.bs_id)
        previous_bs = {}
        ue_handover_times = defaultdict(list)
        done = False
        step = 0
        while not done and step < args.max_steps:
            action = np.zeros(core.NUM_USERS, dtype=int)
            for ue_idx, ue_id in enumerate(sorted(core.users.keys())):
                ue = core.users[ue_id]
                if ue not in core.active:
                    action[ue_idx] = 0
                    continue
                action[ue_idx] = build_a3_action_for_ue(ue, stations_sorted, core, cio_matrix, args.cio_scale, args.off_a3, args.hys_a3)
            _, _, terminated, truncated, _ = env.step(action)
            current_bs = {}
            for ue in core.active:
                serving_bs = get_serving_bs(core, ue, stations_sorted)
                current_bs[ue.ue_id] = serving_bs if serving_bs is not None else 0
            step_events = detect_scaled_step_events(core, stations_sorted, previous_bs, current_bs, ue_handover_times, episode, step, args)
            step_transitions = np.zeros_like(cio_matrix, dtype=int)
            for event in step_events:
                i = int(event["from_bs"])
                j = int(event["to_bs"])
                step_transitions[i, j] += 1
                cumulative_transitions[i, j] += 1
                activity_matrix[i, j] += 1
            node_rows = compute_node_rows(core, stations_sorted, episode, step)
            edge_rows = make_recursive_edge_rows(episode, step, num_stations, step_transitions, cumulative_transitions, step_events, cio_matrix)
            graph = build_risk_graph(node_rows, edge_rows, args.ping_pong_weight, args.inactive_edge_weight)
            graph.iteration = int(iteration)
            graph.episode = int(episode)
            graph.step = int(step)
            graphs.append(graph.cpu())
            all_node_rows.extend(node_rows)
            all_edge_rows.extend(edge_rows)
            all_events.extend(step_events)
            done = terminated or truncated
            step += 1
        env.close()
    stats = summarize_events(all_events)
    stats["steps"] = len(graphs)
    return graphs, all_node_rows, all_edge_rows, all_events, activity_matrix, stats


def candidate_preserves_handovers(candidate_cio, args, baseline_handovers):
    _, _, _, _, _, stats = simulate_with_cio(candidate_cio, args, 0)
    handovers = int(stats["handover_count"])
    return handovers >= baseline_handovers * args.min_handover_ratio, handovers


def choose_safe_cio_update(current_cio, candidate_cio, fallback_cio, args, baseline_handovers):
    last_handovers = None
    for scale in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0):
        trial_cio = current_cio + scale * (candidate_cio - current_cio)
        preserves, handovers = candidate_preserves_handovers(trial_cio, args, baseline_handovers)
        last_handovers = handovers
        if preserves:
            return trial_cio, handovers, scale < 1.0
    return fallback_cio.copy(), last_handovers, True


def plot_matrix(matrix, path, title, label, clip, annotate_threshold=80):
    sns.set_theme(style="white")
    annot = matrix.shape[0] <= annotate_threshold
    fmt = ".5f" if label == "CIO" else ".3f"
    figsize = (max(8, matrix.shape[1] * 0.7), max(7, matrix.shape[0] * 0.55))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(matrix, cmap="RdBu_r", center=0, vmin=-clip, vmax=clip, annot=annot, fmt=fmt, annot_kws={"fontsize": 7}, linewidths=0.25, linecolor="white", square=True, ax=ax, cbar_kws={"label": label, "shrink": 0.8})
    ax.set_title(title)
    ax.set_xlabel("Destination BS")
    ax.set_ylabel("Source BS")
    ax.tick_params(axis="x", labelrotation=0, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metrics(metrics_df, output_dir):
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].plot(metrics_df["iteration"], metrics_df["failure_count"], marker="o", label="Failures")
    axes[0, 0].plot(metrics_df["iteration"], metrics_df["handover_count"], marker="o", label="Handovers")
    axes[0, 0].set_title("Simulation outcome per recursive iteration")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].legend()
    axes[0, 1].plot(metrics_df["iteration"], metrics_df["train_loss"], marker="o", color="#2ca02c")
    axes[0, 1].set_title("Weighted MSE risk-prediction loss")
    axes[0, 1].set_xlabel("Iteration")
    axes[1, 0].plot(metrics_df["iteration"], metrics_df["mean_predicted_risk"], marker="o", color="#9467bd", label="Predicted risk")
    if "handover_ratio" in metrics_df.columns:
        axes[1, 0].plot(metrics_df["iteration"], metrics_df["handover_ratio"], marker="o", color="#ff7f0e", label="Handover ratio")
    axes[1, 0].set_title("Predicted risk and handover preservation")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].legend()
    axes[1, 1].plot(metrics_df["iteration"], metrics_df["cio_min"], marker="o", label="min")
    axes[1, 1].plot(metrics_df["iteration"], metrics_df["cio_max"], marker="o", label="max")
    axes[1, 1].plot(metrics_df["iteration"], metrics_df["cio_std"], marker="o", label="std")
    axes[1, 1].set_title("CIO matrix statistics")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "recursive_optimization_metrics.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_failure_distribution(events_df, args, output_dir):
    if events_df.empty:
        return
    sns.set_theme(style="whitegrid")
    class_order = ["normal", "too_early_handover", "too_late_handover"]
    per_iteration = []
    for iteration, group in events_df.groupby("iteration"):
        total = max(len(group), 1)
        row = {
            "iteration": iteration,
            "valid_handover_rate": float((group["failure_class"] == "normal").mean()),
            "ping_pong_overlay_rate": float(group["is_ping_pong"].mean()),
        }
        counts = group["failure_class"].value_counts()
        for label in class_order:
            row[label] = float(counts.get(label, 0) / total)
        per_iteration.append(row)
    dist_df = pd.DataFrame(per_iteration).sort_values("iteration")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={"width_ratios": [2.1, 1.0]})
    bottom = np.zeros(len(dist_df))
    colors = {
        "normal": "#2ca02c",
        "too_early_handover": "#1f77b4",
        "too_late_handover": "#d62728",
    }
    for label in class_order:
        axes[0].bar(dist_df["iteration"], dist_df[label], bottom=bottom, label=label.replace("_", " "), color=colors[label], alpha=0.85)
        bottom += dist_df[label].values
    axes[0].plot(dist_df["iteration"], dist_df["valid_handover_rate"], color="black", marker="o", linewidth=2.2, label="valid handover rate")
    axes[0].plot(dist_df["iteration"], dist_df["ping_pong_overlay_rate"], color="#ff7f0e", marker="s", linestyle="--", linewidth=2.2, label="ping-pong overlay rate")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Fraction of handovers")
    axes[0].set_title("Failure type distribution vs valid handover frequency")
    axes[0].legend(loc="upper right", fontsize=8)
    params = [
        ("valid_snr_similarity_min", args.valid_snr_similarity_min),
        ("valid_snr_gain_min", args.valid_snr_gain_min),
        ("valid_snr_gain_max", args.valid_snr_gain_max),
        ("too_early_ratio", args.too_early_ratio),
        ("too_late_ratio", args.too_late_ratio),
        ("min_stay_steps", args.min_stay_steps),
        ("min_failure_severity", args.min_failure_severity),
        ("ping_pong_weight", args.ping_pong_weight),
        ("ping_pong_valid_tolerance", args.ping_pong_valid_tolerance),
        ("ftl_relative_threshold", args.ftl_relative_threshold),
        ("off_a3", args.off_a3),
        ("hys_a3", args.hys_a3),
    ]
    axes[1].axis("off")
    axes[1].set_title("Threshold parameters", loc="left")
    text = "\n".join(f"{name}: {value}" for name, value in params)
    axes[1].text(0.0, 0.95, text, va="top", ha="left", fontsize=11, family="monospace", bbox={"boxstyle": "round", "facecolor": "#f5f5f5", "edgecolor": "#cccccc"})
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "failure_type_distribution.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize_iteration_events(events_df, label, iteration):
    group = events_df[events_df["iteration"] == iteration]
    total = max(len(group), 1)
    return {
        "method": label,
        "iteration": iteration,
        "handover_count": int(len(group)),
        "normal_count": int((group["failure_class"] == "normal").sum()),
        "too_early_count": int((group["failure_class"] == "too_early_handover").sum()),
        "too_late_count": int((group["failure_class"] == "too_late_handover").sum()),
        "ping_pong_overlay_count": int(group["is_ping_pong"].sum()),
        "normal_rate": float((group["failure_class"] == "normal").sum() / total),
        "failure_rate": float((group["is_failure"]).sum() / total),
        "too_early_rate": float((group["failure_class"] == "too_early_handover").sum() / total),
        "too_late_rate": float((group["failure_class"] == "too_late_handover").sum() / total),
        "ping_pong_overlay_rate": float(group["is_ping_pong"].sum() / total),
    }


def save_improvement_comparison(metrics_df, events_df, cio_matrix, risk_matrix, output_dir):
    if metrics_df.empty or events_df.empty:
        return
    baseline_iteration = int(metrics_df["iteration"].min())
    final_iteration = int(metrics_df["iteration"].max())
    rows = [
        summarize_iteration_events(events_df, "baseline_static_a3_initial_cio", baseline_iteration),
        summarize_iteration_events(events_df, "gnn_recursive_final_cio", final_iteration),
    ]
    metric_lookup = metrics_df.set_index("iteration")
    num_stations = cio_matrix.shape[0]
    off_diag_mask = ~np.eye(num_stations, dtype=bool)
    asymmetry_values = []
    for i in range(num_stations):
        for j in range(i + 1, num_stations):
            asymmetry_values.append(abs(float(cio_matrix[i, j] - cio_matrix[j, i])))
    final_cio_asymmetry_mean = float(np.mean(asymmetry_values)) if asymmetry_values else 0.0
    final_cio_asymmetry_max = float(np.max(asymmetry_values)) if asymmetry_values else 0.0
    for row in rows:
        iteration = row["iteration"]
        if iteration in metric_lookup.index:
            metric_row = metric_lookup.loc[iteration]
            row["baseline_handovers"] = int(metric_row["baseline_handovers"])
            row["handover_ratio"] = float(metric_row["handover_ratio"])
            row["train_loss"] = float(metric_row["train_loss"]) if pd.notna(metric_row["train_loss"]) else np.nan
            row["mean_predicted_risk"] = float(metric_row["mean_predicted_risk"])
            row["starvation_corrected"] = bool(metric_row["starvation_corrected"])
        if row["method"] == "gnn_recursive_final_cio":
            row["final_cio_min"] = float(cio_matrix[off_diag_mask].min())
            row["final_cio_max"] = float(cio_matrix[off_diag_mask].max())
            row["final_cio_std"] = float(cio_matrix[off_diag_mask].std())
            row["final_cio_asymmetry_mean"] = final_cio_asymmetry_mean
            row["final_cio_asymmetry_max"] = final_cio_asymmetry_max
            row["final_risk_mean"] = float(risk_matrix[off_diag_mask].mean())
            row["final_risk_std"] = float(risk_matrix[off_diag_mask].std())
    comparison_df = pd.DataFrame(rows)
    comparison_df.to_csv(os.path.join(output_dir, "gnn_vs_baseline_comparison.csv"), index=False)


def run_recursive_optimization(args):
    os.makedirs(args.output_dir, exist_ok=True)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    num_stations = detect_num_stations(args.env_name, args.num_stations)
    cio_matrix = np.zeros((num_stations, num_stations), dtype=float)
    model = None
    replay_buffer = []
    metrics = []
    all_node_rows = []
    all_edge_rows = []
    all_events = []
    final_risk_matrix = np.zeros_like(cio_matrix)
    baseline_handovers = None
    last_safe_cio_matrix = cio_matrix.copy()

    for iteration in range(args.iterations):
        previous_cio_matrix = cio_matrix.copy()
        graphs, node_rows, edge_rows, events, activity_matrix, stats = simulate_with_cio(cio_matrix, args, iteration)
        if baseline_handovers is None:
            baseline_handovers = int(stats["handover_count"])
            if baseline_handovers < args.min_baseline_handovers:
                raise RuntimeError(
                    f"Baseline A3 generated only {baseline_handovers} handovers. "
                    f"Off_A3={args.off_a3} and Hys_A3={args.hys_a3} are too strict for this environment/run. "
                    "Lower --off-a3, lower --hys-a3, increase --max-steps, or use a larger/mobile environment."
                )
        all_node_rows.extend({**row, "iteration": iteration} for row in node_rows)
        all_edge_rows.extend({**row, "iteration": iteration} for row in edge_rows)
        all_events.extend({**event, "iteration": iteration} for event in events)
        replay_buffer.extend(graphs)
        if args.replay_window > 0 and len(replay_buffer) > args.replay_window:
            replay_buffer = replay_buffer[-args.replay_window:]
        train_loss = None
        mean_predicted_risk = 0.0
        handover_ratio = float(stats["handover_count"] / max(baseline_handovers, 1))
        starvation_corrected = False
        candidate_handovers = None
        if handover_ratio >= args.min_handover_ratio:
            last_safe_cio_matrix = previous_cio_matrix.copy()
        if len(replay_buffer) >= args.min_train_graphs:
            if model is None:
                model = build_model(replay_buffer[0], args, device)
            train_loss = train_risk_model(model, replay_buffer, args, device)
            final_risk_matrix = predict_edge_risk(model, graphs[-1], num_stations, device)
            mean_predicted_risk = float(final_risk_matrix[~np.eye(num_stations, dtype=bool)].mean())
            candidate_cio = update_cio_from_risk(cio_matrix, final_risk_matrix, activity_matrix, args)
            cio_matrix, candidate_handovers, starvation_corrected = choose_safe_cio_update(cio_matrix, candidate_cio, last_safe_cio_matrix, args, baseline_handovers)
            if iteration > 0 and handover_ratio < args.min_handover_ratio:
                cio_matrix = relax_cio_toward_previous(cio_matrix, last_safe_cio_matrix, args.starvation_relaxation)
                starvation_corrected = True
        off_diag = cio_matrix[~np.eye(num_stations, dtype=bool)]
        metrics.append({
            "iteration": iteration,
            "steps": stats["steps"],
            "handover_count": stats["handover_count"],
            "baseline_handovers": baseline_handovers,
            "handover_ratio": handover_ratio,
            "candidate_handovers": candidate_handovers,
            "starvation_corrected": starvation_corrected,
            "failure_count": stats["failure_count"],
            "early_failures": stats["early_failures"],
            "late_failures": stats["late_failures"],
            "ping_pongs": stats["ping_pongs"],
            "train_graphs": len(replay_buffer),
            "train_loss": train_loss,
            "mean_predicted_risk": mean_predicted_risk,
            "cio_min": float(off_diag.min()),
            "cio_max": float(off_diag.max()),
            "cio_std": float(off_diag.std()),
        })
        pd.DataFrame(cio_matrix).to_csv(os.path.join(args.output_dir, f"cio_matrix_iteration_{iteration:03d}.csv"), index=False, header=False)
        pd.DataFrame(final_risk_matrix).to_csv(os.path.join(args.output_dir, f"predicted_risk_matrix_iteration_{iteration:03d}.csv"), index=False, header=False)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(args.output_dir, "recursive_iteration_metrics.csv"), index=False)
    events_df = pd.DataFrame(all_events)
    pd.DataFrame(all_node_rows).to_csv(os.path.join(args.output_dir, "recursive_node_features.csv"), index=False)
    pd.DataFrame(all_edge_rows).to_csv(os.path.join(args.output_dir, "recursive_edge_features.csv"), index=False)
    events_df.to_csv(os.path.join(args.output_dir, "recursive_handover_events.csv"), index=False)
    pd.DataFrame(cio_matrix).to_csv(os.path.join(args.output_dir, "final_cio_matrix.csv"), index=False, header=False)
    pd.DataFrame(final_risk_matrix).to_csv(os.path.join(args.output_dir, "final_predicted_risk_matrix.csv"), index=False, header=False)
    if model is not None:
        torch.save(model.state_dict(), os.path.join(args.output_dir, "recursive_risk_gnn_model.pt"))
    plot_metrics(metrics_df, args.output_dir)
    plot_failure_distribution(events_df, args, args.output_dir)
    save_improvement_comparison(metrics_df, events_df, cio_matrix, final_risk_matrix, args.output_dir)
    plot_matrix(cio_matrix, os.path.join(args.output_dir, "final_cio_matrix.png"), "Final recursive CIO matrix", "CIO", args.cio_clip)
    plot_matrix(final_risk_matrix, os.path.join(args.output_dir, "final_predicted_risk_matrix.png"), "Final predicted directed-edge failure risk", "Predicted risk", 1.0)


def apply_parameter_preset(args):
    preset = args.parameter_preset
    if preset == "auto":
        preset = "large-balanced" if "large" in args.env_name else "small-balanced"
    if preset == "small-balanced":
        args.off_a3 = 0.0
        args.hys_a3 = 0.0
        args.cio_scale = 0.000001
        args.cio_lr = 2000.0
        args.max_cio_step = 2.0
        args.cio_clip = 5.0
        args.min_handover_ratio = 0.9
        args.starvation_relaxation = 0.0
        args.inactive_edge_weight = 0.0
        args.valid_snr_gain_min = 0.05
        args.valid_snr_gain_max = 10.0
        args.too_early_ratio = 0.70
        args.too_late_ratio = 4.5
        args.min_failure_severity = 0.65
    elif preset == "large-balanced":
        args.off_a3 = 0.0
        args.hys_a3 = 0.0
        args.cio_scale = 0.000001
        args.cio_lr = 2000.0
        args.max_cio_step = 2.0
        args.cio_clip = 5.0
        args.min_handover_ratio = 0.9
        args.starvation_relaxation = 0.0
        args.inactive_edge_weight = 0.0
        args.valid_snr_gain_min = 0.02
        args.valid_snr_gain_max = 25.0
        args.too_early_ratio = 0.60
        args.too_late_ratio = 12.0
        args.min_failure_severity = 0.70
    return args


def parse_args():
    parser = argparse.ArgumentParser(description="Recursive simulator-in-the-loop GNN CIO optimization")
    parser.add_argument("--output-dir", type=str, default="results_recursive/gnn_cio_optimization")
    parser.add_argument("--env-name", type=str, default="mobile-small-central-v0")
    parser.add_argument("--parameter-preset", type=str, default="auto", choices=["auto", "none", "small-balanced", "large-balanced"])
    parser.add_argument("--num-stations", type=int, default=3)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fixed-scenario", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-train-graphs", type=int, default=5)
    parser.add_argument("--replay-window", type=int, default=0)
    parser.add_argument("--epochs-per-iteration", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--conv-type", type=str, default="dir-gcn", choices=["gcn", "sage", "gat", "dir-gcn", "dir-sage", "dir-gat"])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--cio-scale", type=float, default=0.8)
    parser.add_argument("--off-a3", type=float, default=3.0)
    parser.add_argument("--hys-a3", type=float, default=0.0)
    parser.add_argument("--cio-clip", type=float, default=0.5)
    parser.add_argument("--cio-lr", type=float, default=0.08)
    parser.add_argument("--max-cio-step", type=float, default=0.01)
    parser.add_argument("--min-handover-ratio", type=float, default=0.35)
    parser.add_argument("--min-baseline-handovers", type=int, default=5)
    parser.add_argument("--starvation-relaxation", type=float, default=0.25)
    parser.add_argument("--ping-pong-weight", type=float, default=0.9)
    parser.add_argument("--inactive-edge-weight", type=float, default=0.1)
    parser.add_argument("--inactive-update-scale", type=float, default=0.0)
    parser.add_argument("--activity-normalizer", type=float, default=10.0)
    parser.add_argument("--update-active-edges-only", action="store_true")
    parser.add_argument("--fte-threshold", type=float, default=1.0)
    parser.add_argument("--ftl-threshold", type=float, default=0.0)
    parser.add_argument("--fte-relative-drop", type=float, default=0.25)
    parser.add_argument("--ftl-relative-threshold", type=float, default=0.75)
    parser.add_argument("--ftl-relative-gain", type=float, default=0.5)
    parser.add_argument("--valid-snr-similarity-min", type=float, default=0.30)
    parser.add_argument("--valid-snr-gain-min", type=float, default=0.08)
    parser.add_argument("--valid-snr-gain-max", type=float, default=8.0)
    parser.add_argument("--too-early-ratio", type=float, default=0.75)
    parser.add_argument("--too-late-ratio", type=float, default=3.5)
    parser.add_argument("--min-stay-steps", type=int, default=3)
    parser.add_argument("--min-failure-severity", type=float, default=0.65)
    parser.add_argument("--ping-pong-valid-tolerance", type=float, default=0.8)
    parser.add_argument("--absolute-failure-thresholds", action="store_true")
    parser.add_argument("--ping-pong-window", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    return apply_parameter_preset(parser.parse_args())


if __name__ == "__main__":
    run_recursive_optimization(parse_args())
