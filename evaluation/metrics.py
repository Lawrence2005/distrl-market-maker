"""
evaluation/metrics.py

Core evaluation metrics for distrl-market-maker.

All analysis notebooks and evaluation scripts call this module.
Never compute metrics directly in notebook cells.

Metrics computed
----------------
Sharpe          : mean(step_pnl) / std(step_pnl) * sqrt(T)
MAP             : mean(|inventory|) — Mean Absolute Position
CVaR_alpha      : mean of worst-alpha fraction of episode PnLs
MDD             : max drawdown on cumulative PnL
PnL             : final cumulative PnL
WinRate         : fraction of steps with positive step PnL
FillRate        : fraction of steps where at least one side filled
AdverseSelCost  : mean step PnL on steps where inventory moved adverse direction
QuantileCrossing: fraction of quantile pairs that cross (QR-DQN/IQN only)
InventoryAtBounds: fraction of steps where |inventory| == Q_max

All functions accept either:
  - Lists/arrays of per-step values (for single-episode computation)
  - Lists of per-episode dicts (for multi-episode aggregation)

Returns are plain Python floats or pandas DataFrames — never torch tensors.

Week 8 deliverable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# Per-episode metrics (single episode)
# ══════════════════════════════════════════════════════════════════════════════

def sharpe(step_pnls: np.ndarray) -> float:
    """
    Intraday Sharpe ratio.

    Sharpe = mean(step_pnl) / std(step_pnl) * sqrt(T)

    Parameters
    ----------
    step_pnls : np.ndarray shape (T,) — per-step PnL values

    Returns
    -------
    float
    """
    r = np.asarray(step_pnls, dtype=np.float64)
    if len(r) < 2:
        return 0.0
    mu  = np.mean(r)
    sig = np.std(r) + 1e-10
    return float(mu / sig * np.sqrt(len(r)))


def map_score(inventories: np.ndarray) -> float:
    """
    Mean Absolute Position.

    MAP = mean(|q_t|)

    Lower is better — measures how well the agent manages inventory.

    Parameters
    ----------
    inventories : np.ndarray shape (T,)

    Returns
    -------
    float
    """
    return float(np.mean(np.abs(np.asarray(inventories, dtype=np.float64))))


def mdd(cum_pnls: np.ndarray) -> float:
    """
    Maximum Drawdown on cumulative PnL series.

    MDD = min_t (cum_pnl_t - max_{s<=t} cum_pnl_s)

    Always <= 0. Closer to 0 = smaller worst-case loss.

    Parameters
    ----------
    cum_pnls : np.ndarray shape (T,)

    Returns
    -------
    float
    """
    c = np.asarray(cum_pnls, dtype=np.float64)
    if len(c) == 0:
        return 0.0
    rolling_max = np.maximum.accumulate(c)
    return float(np.min(c - rolling_max))


def cvar(values: np.ndarray, alpha: float = 0.10) -> float:
    """
    Conditional Value-at-Risk at level alpha.

    CVaR_alpha = mean of worst alpha fraction of values.

    Used for:
      - Episode-level CVaR: values = per-episode final PnLs
      - Step-level CVaR:    values = per-step PnLs

    Parameters
    ----------
    values : np.ndarray — array of values (PnL, returns, etc.)
    alpha  : float      — tail fraction in (0, 1]

    Returns
    -------
    float — CVaR (always <= mean(values))
    """
    v      = np.asarray(values, dtype=np.float64)
    n_tail = max(1, int(alpha * len(v)))
    return float(np.sort(v)[:n_tail].mean())


def win_rate(step_pnls: np.ndarray) -> float:
    """Fraction of steps with positive step PnL."""
    r = np.asarray(step_pnls, dtype=np.float64)
    return float(np.mean(r > 0))


def fill_rate(bid_fills: np.ndarray, ask_fills: np.ndarray) -> float:
    """
    Fraction of steps where at least one side filled.

    Parameters
    ----------
    bid_fills : np.ndarray shape (T,) — bid fill quantity per step
    ask_fills : np.ndarray shape (T,) — ask fill quantity per step

    Returns
    -------
    float
    """
    b = np.asarray(bid_fills, dtype=np.float64)
    a = np.asarray(ask_fills, dtype=np.float64)
    return float(np.mean((b > 0) | (a > 0)))


def adverse_selection_cost(
    step_pnls:   np.ndarray,
    inventories: np.ndarray,
) -> float:
    """
    Mean step PnL on steps where inventory moved in an adverse direction.

    Adverse = inventory increased when agent was already long,
              or decreased when agent was already short.
    A negative value indicates the agent is losing money on
    adverse-selection fills.

    Parameters
    ----------
    step_pnls   : np.ndarray shape (T,)
    inventories : np.ndarray shape (T,)

    Returns
    -------
    float
    """
    r   = np.asarray(step_pnls,   dtype=np.float64)
    inv = np.asarray(inventories, dtype=np.float64)

    if len(inv) < 2:
        return 0.0

    delta_inv = np.diff(inv, prepend=inv[0])
    # Adverse: inventory moving further from zero
    adverse_mask = (inv > 0) & (delta_inv > 0) | (inv < 0) & (delta_inv < 0)

    if adverse_mask.sum() == 0:
        return 0.0
    return float(r[adverse_mask].mean())


def inventory_at_bounds(
    inventories: np.ndarray,
    q_max:       int,
) -> float:
    """
    Fraction of steps where |inventory| == Q_max.

    Target < 5% per the Week 6 convergence check.

    Parameters
    ----------
    inventories : np.ndarray shape (T,)
    q_max       : int — inventory constraint

    Returns
    -------
    float in [0, 1]
    """
    inv = np.asarray(inventories, dtype=np.float64)
    return float(np.mean(np.abs(inv) >= q_max))


def quantile_crossing_rate(Z: np.ndarray) -> float:
    """
    Fraction of (action, quantile-pair) combinations where quantiles cross.

    A quantile crossing means Z[a, i] > Z[a, i+1] for some i — the
    distribution is not monotone. Target < 5% for a well-trained QR-DQN.

    Parameters
    ----------
    Z : np.ndarray shape (n_actions, n_quantiles) — sorted quantile values

    Returns
    -------
    float in [0, 1]
    """
    diffs    = np.diff(Z, axis=-1)          # (n_actions, n_quantiles-1)
    crossings = (diffs < 0).mean()
    return float(crossings)


# ══════════════════════════════════════════════════════════════════════════════
# Multi-episode aggregation
# ══════════════════════════════════════════════════════════════════════════════

def episode_metrics(
    step_pnls:   np.ndarray,
    inventories: np.ndarray,
    cum_pnls:    np.ndarray,
    q_max:       int = 10,
    bid_fills:   Optional[np.ndarray] = None,
    ask_fills:   Optional[np.ndarray] = None,
) -> dict:
    """
    Compute all per-episode metrics from raw step data.

    Parameters
    ----------
    step_pnls   : np.ndarray shape (T,)
    inventories : np.ndarray shape (T,)
    cum_pnls    : np.ndarray shape (T,)
    q_max       : int — inventory constraint
    bid_fills   : np.ndarray shape (T,) or None
    ask_fills   : np.ndarray shape (T,) or None

    Returns
    -------
    dict with keys: sharpe, map, mdd, cvar_10, final_pnl, win_rate,
                    fill_rate, adverse_sel_cost, inventory_at_bounds, n_steps
    """
    metrics = {
        "sharpe":             sharpe(step_pnls),
        "map":                map_score(inventories),
        "mdd":                mdd(cum_pnls),
        "cvar_10":            cvar(step_pnls, alpha=0.10),
        "cvar_25":            cvar(step_pnls, alpha=0.25),
        "final_pnl":          float(cum_pnls[-1]) if len(cum_pnls) > 0 else 0.0,
        "win_rate":           win_rate(step_pnls),
        "adverse_sel_cost":   adverse_selection_cost(step_pnls, inventories),
        "inventory_at_bounds": inventory_at_bounds(inventories, q_max),
        "n_steps":            len(step_pnls),
    }

    if bid_fills is not None and ask_fills is not None:
        metrics["fill_rate"] = fill_rate(bid_fills, ask_fills)

    return metrics


def aggregate_episodes(episode_metrics_list: list[dict]) -> dict:
    """
    Aggregate a list of per-episode metric dicts into mean ± std.

    Parameters
    ----------
    episode_metrics_list : list of dicts from episode_metrics()

    Returns
    -------
    dict with keys: {metric}_mean, {metric}_std for each metric
    """
    if not episode_metrics_list:
        return {}

    keys   = [k for k in episode_metrics_list[0] if k != "n_steps"]
    result = {}

    for k in keys:
        vals = np.array([ep[k] for ep in episode_metrics_list
                         if k in ep], dtype=np.float64)
        result[f"{k}_mean"] = float(np.mean(vals))
        result[f"{k}_std"]  = float(np.std(vals))

    result["n_episodes"] = len(episode_metrics_list)
    result["n_steps"]    = int(
        np.mean([ep.get("n_steps", 0) for ep in episode_metrics_list])
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Load metrics from saved experiment directories
# ══════════════════════════════════════════════════════════════════════════════

def load_train_history(run_dir: str | Path) -> pd.DataFrame:
    """
    Load training history JSON from an experiment run directory.

    Parameters
    ----------
    run_dir : str | Path — path to logs/{run_tag}/ directory

    Returns
    -------
    pd.DataFrame — one row per episode, columns = metric names
    """
    path = Path(run_dir) / "train_history.json"
    if not path.exists():
        raise FileNotFoundError(f"No train_history.json in {run_dir}")
    with open(path) as f:
        history = json.load(f)
    return pd.DataFrame(history)


def load_eval_history(run_dir: str | Path) -> pd.DataFrame:
    """
    Load evaluation history JSON from an experiment run directory.

    Parameters
    ----------
    run_dir : str | Path — path to logs/{run_tag}/ directory

    Returns
    -------
    pd.DataFrame — one row per eval checkpoint, columns = metric names
    """
    path = Path(run_dir) / "eval_history.json"
    if not path.exists():
        raise FileNotFoundError(f"No eval_history.json in {run_dir}")
    with open(path) as f:
        history = json.load(f)
    return pd.DataFrame(history)


def load_all_runs(
    log_root:   str | Path,
    pattern:    str = "*",
) -> pd.DataFrame:
    """
    Load training histories from all matching run directories.

    Parses the run_tag naming convention to extract agent, encoder,
    reward, regime, and seed as separate columns.

    Parameters
    ----------
    log_root : str | Path — path to logs/ directory
    pattern  : str        — glob pattern for run subdirectories

    Returns
    -------
    pd.DataFrame — all runs combined, with parsed metadata columns
    """
    log_root = Path(log_root)
    rows     = []

    for run_dir in sorted(log_root.glob(pattern)):
        if not run_dir.is_dir():
            continue
        try:
            df = load_train_history(run_dir)
        except FileNotFoundError:
            continue

        # Parse run_tag: {agent}_{encoder}_{reward}_{regime}_seed{seed}
        tag   = run_dir.name
        parts = tag.split("_")
        meta  = {
            "run_tag":  tag,
            "agent":    parts[0] if len(parts) > 0 else "unknown",
            "encoder":  parts[1] if len(parts) > 1 else "unknown",
            "reward":   parts[2] if len(parts) > 2 else "unknown",
            "regime":   parts[3] if len(parts) > 3 else "unknown",
            "recurrent": "recurrent" in tag,
            "seed":     int(tag.split("seed")[-1]) if "seed" in tag else 0,
        }
        for k, v in meta.items():
            df[k] = v

        rows.append(df)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summary_table(
    log_root: str | Path,
    metric:   str = "sharpe",
    episodes_window: int = 100,
) -> pd.DataFrame:
    """
    Build a summary table of final performance across all runs.

    Parameters
    ----------
    log_root        : str | Path — path to logs/ directory
    metric          : str        — metric to summarise (default 'sharpe')
    episodes_window : int        — average over last N episodes

    Returns
    -------
    pd.DataFrame — one row per run, sorted by metric descending
    """
    all_runs = load_all_runs(log_root)
    if all_runs.empty:
        return pd.DataFrame()

    meta_cols = ["run_tag", "agent", "encoder", "reward", "regime",
                 "recurrent", "seed"]
    rows = []

    for run_tag, group in all_runs.groupby("run_tag"):
        group = group.sort_values("episode")
        final = group.tail(episodes_window)

        meta = {c: group[c].iloc[0] for c in meta_cols if c in group.columns}
        meta[f"{metric}_final_mean"] = (
            final[metric].mean() if metric in final.columns else np.nan
        )
        meta[f"{metric}_final_std"] = (
            final[metric].std() if metric in final.columns else np.nan
        )
        rows.append(meta)

    return (
        pd.DataFrame(rows)
        .sort_values(f"{metric}_final_mean", ascending=False)
        .reset_index(drop=True)
    )