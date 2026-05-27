"""
DIRECTED GNN MODELS - Edge Risk Prediction for CIO Optimization

Purpose:
  Implements directed graph neural networks (GNNs) to predict handover failures.
  Separates in-degree and out-degree neighbors for proper directed graphs.
  
Models:
  - DirGCNConv: Directed Graph Convolution
  - DirSageConv: Directed GraphSAGE
  - DirGATConv: Directed Graph Attention
  
Architecture:
  DirGNNEncoder → (node features) → edge embeddings
  DirGNNEdgePredictor → (edge features) → risk scores
  
Key Functions:
  load_graph_snapshots() - Convert CSV data to PyG graph format
  train_epoch() - Single training epoch
  evaluate() - Validation/test evaluation
  
Targets:
  - "any_failure": Binary classification (fail/no-fail)
  - "failure_count": Regression (# failures per edge)
  - "ping_pong": Focus on oscillatory failures
"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
from mobile_env_local.data_utils import EDGE_FEATURE_COLUMNS, FAILURE_TARGETS, add_edge_cio, build_failure_target, ensure_edge_feature_columns


def directed_norm(adj: torch.Tensor) -> torch.Tensor:
    in_deg = adj.sum(dim=0)
    in_deg_inv_sqrt = in_deg.pow(-0.5)
    in_deg_inv_sqrt[torch.isinf(in_deg_inv_sqrt)] = 0.0

    out_deg = adj.sum(dim=1)
    out_deg_inv_sqrt = out_deg.pow(-0.5)
    out_deg_inv_sqrt[torch.isinf(out_deg_inv_sqrt)] = 0.0

    return out_deg_inv_sqrt.view(-1, 1) * adj * in_deg_inv_sqrt.view(1, -1)


def row_norm(adj: torch.Tensor) -> torch.Tensor:
    row_sum = adj.sum(dim=1)
    inv = torch.zeros_like(row_sum)
    inv[row_sum != 0.0] = 1.0 / row_sum[row_sum != 0.0]
    return inv.view(-1, 1) * adj


def sym_norm(adj: torch.Tensor) -> torch.Tensor:
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
    return deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)


def get_norm_adj(adj: torch.Tensor, norm: str):
    if norm == "sym":
        return sym_norm(adj)
    elif norm == "row":
        return row_norm(adj)
    elif norm == "dir":
        return directed_norm(adj)
    else:
        raise ValueError(f"Unsupported normalization: {norm}")


class DirGCNConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirGCNConv, self).__init__()
        self.lin_src_to_dst = nn.Linear(input_dim, output_dim)
        self.lin_dst_to_src = nn.Linear(input_dim, output_dim)
        self.alpha = alpha
        self.adj_norm = None
        self.adj_t_norm = None

    def forward(self, x, edge_index):
        row, col = edge_index
        num_nodes = x.shape[0]
        if self.adj_norm is None or self.adj_norm.shape[0] != num_nodes:
            adj = torch.zeros((num_nodes, num_nodes), dtype=x.dtype, device=x.device)
            adj[row, col] = 1.0
            self.adj_norm = get_norm_adj(adj, norm="dir")
            adj_t = torch.zeros((num_nodes, num_nodes), dtype=x.dtype, device=x.device)
            adj_t[col, row] = 1.0
            self.adj_t_norm = get_norm_adj(adj_t, norm="dir")

        return (
            self.alpha * self.lin_src_to_dst(self.adj_norm @ x)
            + (1.0 - self.alpha) * self.lin_dst_to_src(self.adj_t_norm @ x)
        )


class DirSageConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, alpha):
        super(DirSageConv, self).__init__()
        self.conv_src_to_dst = SAGEConv(input_dim, output_dim, flow="source_to_target", root_weight=False)
        self.conv_dst_to_src = SAGEConv(input_dim, output_dim, flow="target_to_source", root_weight=False)
        self.lin_self = nn.Linear(input_dim, output_dim)
        self.alpha = alpha

    def forward(self, x, edge_index):
        return (
            self.lin_self(x)
            + (1.0 - self.alpha) * self.conv_src_to_dst(x, edge_index)
            + self.alpha * self.conv_dst_to_src(x, edge_index)
        )


class DirGATConv(torch.nn.Module):
    def __init__(self, input_dim, output_dim, heads, alpha):
        super(DirGATConv, self).__init__()
        self.conv_src_to_dst = GATConv(input_dim, output_dim, heads=heads)
        self.conv_dst_to_src = GATConv(input_dim, output_dim, heads=heads)
        self.alpha = alpha

    def forward(self, x, edge_index):
        edge_index_t = torch.stack([edge_index[1], edge_index[0]], dim=0)
        return (
            (1.0 - self.alpha) * self.conv_src_to_dst(x, edge_index)
            + self.alpha * self.conv_dst_to_src(x, edge_index_t)
        )


def get_conv(conv_type, input_dim, output_dim, alpha):
    if conv_type == "gcn":
        return GCNConv(input_dim, output_dim, add_self_loops=False)
    elif conv_type == "sage":
        return SAGEConv(input_dim, output_dim)
    elif conv_type == "gat":
        return GATConv(input_dim, output_dim, heads=1)
    elif conv_type == "dir-gcn":
        return DirGCNConv(input_dim, output_dim, alpha)
    elif conv_type == "dir-sage":
        return DirSageConv(input_dim, output_dim, alpha)
    elif conv_type == "dir-gat":
        return DirGATConv(input_dim, output_dim, heads=1, alpha=alpha)
    else:
        raise ValueError(f"Unknown conv_type: {conv_type}")


class DirGNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, conv_type, alpha, normalize, learn_alpha):
        super(DirGNNEncoder, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=learn_alpha)
        self.normalize = normalize
        self.dropout = dropout
        layers = []
        if num_layers == 1:
            layers.append(get_conv(conv_type, input_dim, hidden_dim, self.alpha))
        else:
            layers.append(get_conv(conv_type, input_dim, hidden_dim, self.alpha))
            for _ in range(num_layers - 2):
                layers.append(get_conv(conv_type, hidden_dim, hidden_dim, self.alpha))
            layers.append(get_conv(conv_type, hidden_dim, hidden_dim, self.alpha))
        self.convs = nn.ModuleList(layers)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x


class DirGNNEdgePredictor(nn.Module):
    def __init__(self, node_dim, edge_attr_dim, hidden_dim, dropout):
        super(DirGNNEdgePredictor, self).__init__()
        self.edge_attr_dim = edge_attr_dim
        input_dim = node_dim * 2 + edge_attr_dim
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_embeddings, edge_index, edge_attr=None):
        src = node_embeddings[edge_index[0]]
        dst = node_embeddings[edge_index[1]]
        if edge_attr is None:
            edge_input = torch.cat([src, dst], dim=-1)
        else:
            edge_input = torch.cat([src, dst, edge_attr], dim=-1)
        return self.edge_mlp(edge_input).squeeze(-1)


class DirGNNCIOModel(nn.Module):
    def __init__(self, num_node_features, edge_attr_dim, hidden_dim, num_layers, dropout, conv_type, alpha, normalize, learn_alpha):
        super(DirGNNCIOModel, self).__init__()
        self.encoder = DirGNNEncoder(
            input_dim=num_node_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            conv_type=conv_type,
            alpha=alpha,
            normalize=normalize,
            learn_alpha=learn_alpha,
        )
        self.predictor = DirGNNEdgePredictor(hidden_dim, edge_attr_dim, hidden_dim, dropout)

    def forward(self, x, edge_index, edge_attr=None):
        node_embeddings = self.encoder(x, edge_index)
        return self.predictor(node_embeddings, edge_index, edge_attr=edge_attr)


def load_graph_snapshots(node_csv, edge_csv, root_dir=None, target="ping_pong"):
    node_df = pd.read_csv(node_csv)
    edge_df = pd.read_csv(edge_csv)
    edge_df = add_edge_cio(edge_df)
    edge_df = ensure_edge_feature_columns(edge_df)
    node_df = node_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    edge_df = edge_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    node_FEATURE_COLUMNS = [
        "load",
        "avg_velocity",
        "ues_in_range",
        "avg_snr",
        "min_snr",
    ]
    traj_cols = [c for c in node_df.columns if c.startswith("traj_prob_to_")]
    node_feature_cols = node_FEATURE_COLUMNS + traj_cols

    edge_feature_cols = EDGE_FEATURE_COLUMNS

    valid_targets = {"ping_pong", "transition_prob", "handover_count"} | set(FAILURE_TARGETS)
    if target not in valid_targets:
        raise ValueError(f"target must be one of {sorted(valid_targets)}")

    graphs = []
    grouped = node_df.groupby(["episode", "step"])
    for (episode, step), group in grouped:
        group = group.sort_values("bs_id")
        x_values = np.nan_to_num(group[node_feature_cols].values, nan=0.0, posinf=100.0, neginf=-100.0)
        x_values = np.clip(x_values, -100.0, 100.0)
        x = torch.tensor(x_values, dtype=torch.float)

        edge_group = edge_df[(edge_df["episode"] == episode) & (edge_df["step"] == step)]
        edge_group = edge_group.sort_values(["src_bs", "dst_bs"])
        edge_index = torch.tensor(edge_group[["src_bs", "dst_bs"]].values.T, dtype=torch.long)
        edge_attr_values = np.nan_to_num(edge_group[edge_feature_cols].values, nan=0.0, posinf=100.0, neginf=-100.0)
        edge_attr_values = np.clip(edge_attr_values, -100.0, 100.0)
        edge_attr = torch.tensor(edge_attr_values, dtype=torch.float)

        if target == "ping_pong":
            y = torch.tensor((edge_group["ping_pongs"] > 0).astype(float).values, dtype=torch.float)
            is_classification = True
        elif target == "any_failure":
            y = torch.tensor((edge_group[["early_failures", "late_failures", "ping_pongs"]].sum(axis=1) > 0).astype(float).values, dtype=torch.float)
            is_classification = True
        elif target in {"transition_prob", "handover_count"} | set(FAILURE_TARGETS):
            y_values = np.nan_to_num(build_failure_target(edge_group, target).values, nan=0.0, posinf=100.0, neginf=0.0)
            y_values = np.clip(y_values, 0.0, 100.0)
            y = torch.tensor(y_values, dtype=torch.float)
            is_classification = False
        else:
            raise ValueError(f"Unsupported target: {target}")

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        data.episode = int(episode)
        data.step = int(step)
        data.is_classification = is_classification
        graphs.append(data)

    return graphs


def split_graphs(graphs, train_ratio=0.7, val_ratio=0.15, seed=42):
    random.Random(seed).shuffle(graphs)
    n = len(graphs)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return graphs[:train_end], graphs[train_end:val_end], graphs[val_end:]


def train_epoch(model, loader, optimizer, criterion, device, clip_norm=1.0, use_edge_attr=True):
    model.train()
    total_loss = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        edge_attr = batch.edge_attr if use_edge_attr else None
        out = model(batch.x, batch.edge_index, edge_attr)
        loss = criterion(out, batch.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device, classification=True, use_edge_attr=True):
    model.eval()
    total_loss = 0.0
    preds = []
    labels = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            edge_attr = batch.edge_attr if use_edge_attr else None
            out = model(batch.x, batch.edge_index, edge_attr)
            loss = criterion(out, batch.y)
            total_loss += loss.item() * batch.num_graphs
            preds.append(out.cpu())
            labels.append(batch.y.cpu())
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)

    metrics = {"loss": total_loss / len(loader.dataset)}
    if classification:
        prob = torch.sigmoid(preds)
        pred_label = (prob >= 0.5).float()
        accuracy = (pred_label == labels).float().mean().item()
        tp = ((pred_label == 1) & (labels == 1)).sum().item()
        tn = ((pred_label == 0) & (labels == 0)).sum().item()
        fp = ((pred_label == 1) & (labels == 0)).sum().item()
        fn = ((pred_label == 0) & (labels == 1)).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        metrics.update({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "positive_rate": labels.mean().item(),
            "predicted_positive_rate": pred_label.mean().item(),
            "tp": int(tp),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
        })
    else:
        mse = F.mse_loss(preds, labels).item()
        mae = F.l1_loss(preds, labels).item()
        metrics.update({"mse": mse, "mae": mae})
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train a Directed GNN for CIO edge prediction")
    parser.add_argument("--data-dir", type=str, default="mobile_env", help="Directory containing gnn_node_features.csv and gnn_edge_features.csv")
    parser.add_argument(
        "--target",
        type=str,
        default="ping_pong",
        choices=["ping_pong", "transition_prob", "handover_count", "early_failures", "late_failures", "ping_pongs", "failure_count", "any_failure"],
        help="Edge target to predict",
    )
    parser.add_argument("--conv-type", type=str, default="dir-gcn", choices=["gcn", "sage", "gat", "dir-gcn", "dir-sage", "dir-gat"], help="Type of graph convolution")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.5, help="Alpha blending for directed convs")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-edge-attr", action="store_true", help="Do not use edge attribute features in the edge predictor")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    node_csv = os.path.join(args.data_dir, "gnn_node_features.csv")
    edge_csv = os.path.join(args.data_dir, "gnn_edge_features.csv")
    graphs = load_graph_snapshots(node_csv, edge_csv, root_dir=args.data_dir, target=args.target)
    train_graphs, val_graphs, test_graphs = split_graphs(graphs, seed=args.seed)

    train_loader = DataLoader(train_graphs, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size, shuffle=False)

    sample = graphs[0]
    num_node_features = sample.x.shape[1]
    edge_attr_dim = 0 if args.no_edge_attr else sample.edge_attr.shape[1]
    model = DirGNNCIOModel(
        num_node_features=num_node_features,
        edge_attr_dim=edge_attr_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        conv_type=args.conv_type,
        alpha=args.alpha,
        normalize=False,
        learn_alpha=False,
    )

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = model.to(device)

    if args.target in {"ping_pong", "any_failure"}:
        train_labels = torch.cat([graph.y for graph in train_graphs])
        positives = train_labels.sum()
        negatives = train_labels.numel() - positives
        pos_weight = (negatives / positives.clamp_min(1.0)).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        classification = True
    else:
        criterion = nn.MSELoss()
        classification = False

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    best_model = None
    use_edge_attr = not args.no_edge_attr
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, clip_norm=1.0, use_edge_attr=use_edge_attr)
        val_metrics = evaluate(model, val_loader, criterion, device, classification=classification, use_edge_attr=use_edge_attr)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_model = model.state_dict()
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f}" + (f" val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['f1']:.4f} val_pos={val_metrics['positive_rate']:.4f} pred_pos={val_metrics['predicted_positive_rate']:.4f}" if classification else f" val_mse={val_metrics['mse']:.4f}"))

    if best_model is not None:
        model.load_state_dict(best_model)

    test_metrics = evaluate(model, test_loader, criterion, device, classification=classification)
    print("\nFinal test metrics:")
    print(test_metrics)

    model_path = os.path.join(args.data_dir, f"dirgnn_cio_{args.target}_{args.conv_type}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model checkpoint to {model_path}")


if __name__ == "__main__":
    main()
