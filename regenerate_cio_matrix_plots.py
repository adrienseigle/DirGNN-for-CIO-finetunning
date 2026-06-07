import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_matrix(matrix, output_path, title, label, cio_clip):
    sns.set_theme(style="white")
    max_abs = max(float(cio_clip), 1e-6)
    annot = matrix.shape[0] <= 6
    figsize = (max(8, matrix.shape[1] * 0.7), max(7, matrix.shape[0] * 0.55))
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        cmap="RdBu_r",
        center=0,
        vmin=-max_abs,
        vmax=max_abs,
        annot=annot,
        fmt=".3f",
        linewidths=0.25,
        linecolor="white",
        square=True,
        ax=ax,
        cbar_kws={"label": label, "shrink": 0.8},
    )
    ax.set_title(title)
    ax.set_xlabel("Destination BS")
    ax.set_ylabel("Source BS")
    ax.tick_params(axis="x", labelrotation=0, labelsize=8)
    ax.tick_params(axis="y", labelrotation=0, labelsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def regenerate_run(run_dir, cio_clip):
    run_dir = Path(run_dir)
    pred_path = run_dir / "predicted_cio_matrix_target_step.csv"
    rule_path = run_dir / "rule_based_cio_matrix_target_step.csv"
    target_step = None
    target_path = run_dir / "predicted_cio_target_step.txt"
    if target_path.exists():
        target_step = target_path.read_text(encoding="utf-8").strip().replace("global_step=", "")
    if pred_path.exists():
        pred = pd.read_csv(pred_path, header=None).values
        title = "Predicted CIO matrix"
        if target_step:
            title += f" for global step {target_step}"
        plot_matrix(pred, run_dir / "predicted_cio_matrix_target_step.png", title, "Predicted CIO", cio_clip)
    if rule_path.exists():
        rule = pd.read_csv(rule_path, header=None).values
        title = "Rule-based CIO matrix"
        if target_step:
            title += f" for global step {target_step}"
        plot_matrix(rule, run_dir / "rule_based_cio_matrix_target_step.png", title, "Rule-based CIO", cio_clip)


def main():
    parser = argparse.ArgumentParser(description="Regenerate CIO matrix plots from existing CSV files")
    parser.add_argument("path", type=str)
    parser.add_argument("--cio-clip", type=float, default=0.5)
    args = parser.parse_args()
    root = Path(args.path)
    if (root / "predicted_cio_matrix_target_step.csv").exists() or (root / "rule_based_cio_matrix_target_step.csv").exists():
        regenerate_run(root, args.cio_clip)
    else:
        for run_dir in root.glob("**"):
            if run_dir.is_dir() and ((run_dir / "predicted_cio_matrix_target_step.csv").exists() or (run_dir / "rule_based_cio_matrix_target_step.csv").exists()):
                regenerate_run(run_dir, args.cio_clip)


if __name__ == "__main__":
    main()
