"""
DATA UTILITIES - Processing and Feature Engineering

Purpose:
  Utilities for manipulating handover data and failure metrics.
  
Functions:
  add_edge_cio() - Attach CIO values to edge dataframe
  compute_failure_counts_by_category() - Aggregate failure types
  build_failure_target() - Format targets for model training
"""

import pandas as pd

EDGE_FEATURE_COLUMNS = ["handover_count", "transition_prob", "normal_handovers", "cio"]
FAILURE_TARGETS = ["early_failures", "late_failures", "ping_pongs", "failure_count", "any_failure"]


def add_edge_cio(edge_df: pd.DataFrame, mode: str = "transition_prob") -> pd.DataFrame:
    """Add an initial CIO edge feature for model training.

    If CIO is not available from the source data, use a heuristic based on
    transition probability as an initial edge weight.
    """
    if "cio" in edge_df.columns:
        return edge_df

    if mode == "transition_prob":
        edge_df["cio"] = edge_df["transition_prob"].astype(float)
    elif mode == "uniform":
        edge_df["cio"] = 0.0
    else:
        raise ValueError(f"Unsupported CIO init mode: {mode}")
    return edge_df


def compute_failure_counts_by_category(edge_df: pd.DataFrame) -> dict:
    """Return the total number of failures of each type in the edge dataset."""
    counts = {
        "early_failures": int(edge_df["early_failures"].sum()) if "early_failures" in edge_df.columns else 0,
        "late_failures": int(edge_df["late_failures"].sum()) if "late_failures" in edge_df.columns else 0,
        "ping_pongs": int(edge_df["ping_pongs"].sum()) if "ping_pongs" in edge_df.columns else 0,
    }
    counts["failure_count"] = counts["early_failures"] + counts["late_failures"] + counts["ping_pongs"]
    return counts


def build_failure_target(edge_df: pd.DataFrame, target: str) -> pd.Series:
    """Return a pandas Series containing the requested failure target values."""
    if target == "failure_count":
        return edge_df[["early_failures", "late_failures", "ping_pongs"]].sum(axis=1).astype(float)
    if target == "any_failure":
        return (edge_df[["early_failures", "late_failures", "ping_pongs"]].sum(axis=1) > 0).astype(float)
    if target in edge_df.columns:
        return edge_df[target].astype(float)
    raise ValueError(f"Unknown failure target: {target}")
