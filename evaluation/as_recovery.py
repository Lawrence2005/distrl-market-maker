"""
evaluation/as_recovery.py

Formal AS/GLFT theory-recovery analysis.

Tests the core research thesis:
    "Does a CVaR distributional RL agent rediscover the GLFT closed-form
     policy in low-volatility regimes?"

Method
------
1. Load a trained agent checkpoint
2. Run N evaluation episodes in the low-vol Poisson regime
3. At each step, record (inventory q, bid_offset δ_b, ask_offset δ_a)
4. Compute mean bid_offset per inventory level → empirical skew curve
5. Compute the GLFT theoretical skew curve at the same parameters
6. Regress empirical on theoretical → report R²

R² interpretation:
    R² > 0.6  : meaningful recovery — agent has rediscovered GLFT structure
    R² > 0.5  : moderate recovery
    R² < 0.3  : no recovery — agent policy is unrelated to GLFT

Also fits against AS (upper bound) since AS has a simpler analytical form.

Usage
-----
    from evaluation.as_recovery import run_recovery_analysis

    result = run_recovery_analysis(
        agent=agent,
        env=env,
        enc_type="handcrafted",
        n_episodes=20,
        gamma=1.0, kappa=19.5, sigma=0.245,
        tick_size=0.01, Q_max=10, T=390,
    )
    print(result)
    # {"r2_glft": 0.71, "r2_as": 0.65, "slope_glft": 0.88, ...}

Week 8 deliverable.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import numpy as np
from scipy import stats


# ══════════════════════════════════════════════════════════════════════════════
# GLFT and AS theoretical skew curves
# ══════════════════════════════════════════════════════════════════════════════

def glft_skew_curve(
    inventory_levels: np.ndarray,
    gamma:            float,
    kappa:            float,
    sigma:            float,
    xi:               float,
    A:                float,
    Q_max:            int,
    T:                float,
    tick_size:        float,
    tau_hat:          float = 0.5,
) -> np.ndarray:
    """
    Compute the GLFT theoretical bid offset for each inventory level.

    Uses the ODE solution from baselines/glft.py to compute
    delta_bid*(tau_hat, q) in ticks for each q in inventory_levels.

    Parameters
    ----------
    inventory_levels : np.ndarray — inventory values to evaluate at
    gamma            : float — risk-aversion coefficient
    kappa            : float — fill-rate intensity
    sigma            : float — volatility (log-return units)
    xi               : float — market-impact parameter
    A                : float — Poisson arrival rate scale
    Q_max            : int   — inventory constraint
    T                : float — episode length in steps
    tick_size        : float — dollar value of one tick
    tau_hat          : float — normalised time-to-go in [0,1] (default 0.5)

    Returns
    -------
    np.ndarray — bid offset in ticks for each inventory level
    """
    import sys, os
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

    from baselines.glft import (
        build_ode_matrix, terminal_condition, solve_v, delta_bid
    )

    M   = build_ode_matrix(gamma, kappa, sigma, xi, A, Q_max)
    v_T = terminal_condition(kappa, xi, Q_max)
    taus, V = solve_v(M, v_T, float(T), int(T))

    # Look up v at tau_hat
    tau_clipped = float(np.clip(tau_hat, 0.0, 1.0))
    idx = int(np.searchsorted(taus, tau_clipped))
    idx = int(np.clip(idx, 0, len(taus) - 1))
    v   = V[idx]

    offsets = []
    for q in inventory_levels:
        q = int(np.clip(q, -Q_max, Q_max))
        db = delta_bid(v, q, Q_max, gamma, kappa, xi)
        if np.isfinite(db):
            offsets.append(db / tick_size)   # convert dollars → ticks
        else:
            offsets.append(np.nan)

    return np.array(offsets)


def as_skew_curve(
    inventory_levels: np.ndarray,
    gamma:            float,
    kappa:            float,
    sigma:            float,
    tick_size:        float,
    tau_hat:          float = 0.5,
) -> np.ndarray:
    """
    Compute the AS theoretical bid offset for each inventory level.

    From AS (2008) Prop 3.1:
        bid*(q, tau_hat) = mid - r(q, tau_hat) + delta*(tau_hat)/2

    The bid offset from mid in ticks is:
        delta_b*(q, tau_hat) = delta*(tau_hat)/2 - q * gamma * sigma^2 * tau_hat

    Note: when q > 0 (long), bid offset increases (bid moves away from mid).

    Parameters
    ----------
    inventory_levels : np.ndarray
    gamma            : float
    kappa            : float
    sigma            : float — log-return units
    tick_size        : float
    tau_hat          : float — normalised time-to-go

    Returns
    -------
    np.ndarray — bid offset in ticks
    """
    base_spread = (2.0 / gamma) * np.log(1.0 + gamma / kappa)
    inv_risk    = gamma * sigma ** 2 * tau_hat

    offsets = []
    for q in inventory_levels:
        # Half-spread + inventory skew term
        delta_b = base_spread / 2.0 + q * inv_risk
        offsets.append(delta_b / tick_size)

    return np.array(offsets)


# ══════════════════════════════════════════════════════════════════════════════
# Empirical skew curve from rollout
# ══════════════════════════════════════════════════════════════════════════════

def collect_skew_data(
    agent:      Any,
    env:        Any,
    enc_type:   str,
    n_episodes: int = 20,
    seed:       int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run evaluation episodes and collect (inventory, bid_offset) pairs.

    Parameters
    ----------
    agent      : trained agent with act() interface
    env        : LOBMarketMakingEnv
    enc_type   : str — encoder type for input extraction
    n_episodes : int — number of evaluation episodes
    seed       : int — base seed

    Returns
    -------
    (inventories, bid_offsets) : np.ndarray each shape (N_total_steps,)
    """
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[1]))

    from envs.lob_env import TICK_OFFSETS
    from training.train import get_encoder_input, decode_action

    all_inventories = []
    all_bid_offsets = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        agent.reset_hidden(batch_size=1)

        terminated = truncated = False
        while not (terminated or truncated):
            enc_input  = get_encoder_input(obs, info, enc_type)
            flat_action = agent.act(enc_input, greedy=True)
            action      = decode_action(flat_action)

            obs, reward, terminated, truncated, info = env.step(action)

            inv        = int(info["inventory"])
            bid_offset = int(TICK_OFFSETS[action[0]])

            all_inventories.append(inv)
            all_bid_offsets.append(bid_offset)

    return np.array(all_inventories), np.array(all_bid_offsets)


def empirical_skew_curve(
    inventories:     np.ndarray,
    bid_offsets:     np.ndarray,
    min_visits:      int = 5,
    boundary_buffer: int = 2,
    q_max:           int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean bid offset per inventory level from rollout data.

    Parameters
    ----------
    inventories     : np.ndarray shape (N,)
    bid_offsets     : np.ndarray shape (N,)
    min_visits      : minimum observations per level
    boundary_buffer : exclude levels within N of ±Q_max
    q_max           : inventory constraint

    Returns
    -------
    (inv_levels, mean_offsets) : np.ndarray each shape (K,)
    """
    groups = defaultdict(list)
    for inv, off in zip(inventories, bid_offsets):
        inv = int(inv)
        if abs(inv) <= q_max - boundary_buffer:
            groups[inv].append(float(off))

    inv_levels   = sorted(k for k in groups if len(groups[k]) >= min_visits)
    mean_offsets = np.array([np.mean(groups[k]) for k in inv_levels])

    return np.array(inv_levels), mean_offsets


# ══════════════════════════════════════════════════════════════════════════════
# R² regression
# ══════════════════════════════════════════════════════════════════════════════

def fit_r2(
    empirical: np.ndarray,
    theoretical: np.ndarray,
) -> dict:
    """
    Regress empirical bid offsets on theoretical predictions.

    Fits: empirical = slope * theoretical + intercept

    Parameters
    ----------
    empirical    : np.ndarray — agent's mean bid offset per inventory level
    theoretical  : np.ndarray — GLFT or AS theoretical prediction

    Returns
    -------
    dict with keys: r2, slope, intercept, p_value, std_err
    """
    # Drop NaN pairs
    mask = np.isfinite(empirical) & np.isfinite(theoretical)
    if mask.sum() < 3:
        return {"r2": np.nan, "slope": np.nan, "intercept": np.nan,
                "p_value": np.nan, "std_err": np.nan, "n": int(mask.sum())}

    x = theoretical[mask]
    y = empirical[mask]

    slope, intercept, r, p_value, std_err = stats.linregress(x, y)

    return {
        "r2":        float(r ** 2),
        "slope":     float(slope),
        "intercept": float(intercept),
        "p_value":   float(p_value),
        "std_err":   float(std_err),
        "n":         int(mask.sum()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Full recovery analysis
# ══════════════════════════════════════════════════════════════════════════════

def run_recovery_analysis(
    agent:      Any,
    env:        Any,
    enc_type:   str,
    n_episodes: int   = 20,
    gamma:      float = 1.0,
    kappa:      float = 19.5,
    sigma:      float = 0.245,
    xi:         float = 0.0,
    A:          float = 1.0,
    Q_max:      int   = 10,
    T:          float = 390.0,
    tick_size:  float = 0.01,
    tau_hat:    float = 0.5,
    seed:       int   = 0,
) -> dict:
    """
    Full AS/GLFT theory-recovery analysis for one agent.

    Runs evaluation episodes, computes empirical skew curve, fits
    against GLFT and AS theoretical curves, returns R² and regression stats.

    Parameters
    ----------
    agent       : trained agent
    env         : LOBMarketMakingEnv
    enc_type    : str — encoder type
    n_episodes  : int — number of eval episodes
    gamma       : float — risk-aversion (should match training config)
    kappa       : float — fill-rate intensity
    sigma       : float — volatility (log-return units)
    xi          : float — GLFT market-impact (0 = no impact)
    A           : float — Poisson arrival rate
    Q_max       : int   — inventory constraint
    T           : float — episode length
    tick_size   : float — dollar per tick
    tau_hat     : float — time-to-go fraction for theoretical curves
    seed        : int   — base eval seed

    Returns
    -------
    dict with keys:
        inv_levels    : inventory levels evaluated
        empirical     : empirical mean bid offsets
        glft_theory   : GLFT theoretical bid offsets
        as_theory     : AS theoretical bid offsets
        glft_fit      : regression dict (r2, slope, intercept, ...)
        as_fit        : regression dict
        summary       : human-readable summary string
    """
    print(f"Collecting rollout data ({n_episodes} episodes)...", flush=True)
    inventories, bid_offsets = collect_skew_data(
        agent, env, enc_type, n_episodes=n_episodes, seed=seed
    )
    print(f"  Total steps collected: {len(inventories)}", flush=True)

    # Empirical curve
    inv_levels, emp_offsets = empirical_skew_curve(
        inventories, bid_offsets, q_max=Q_max
    )
    print(f"  Inventory levels with data: {inv_levels}", flush=True)

    if len(inv_levels) < 3:
        print("  WARNING: fewer than 3 inventory levels — "
              "agent may not be exploring enough inventory states.")
        return {"inv_levels": inv_levels, "empirical": emp_offsets,
                "glft_theory": None, "as_theory": None,
                "glft_fit": None, "as_fit": None,
                "summary": "Insufficient data"}

    # Theoretical curves at the same inventory levels
    glft_theory = glft_skew_curve(
        inv_levels, gamma, kappa, sigma, xi, A, Q_max, T, tick_size, tau_hat
    )
    as_theory = as_skew_curve(
        inv_levels, gamma, kappa, sigma, tick_size, tau_hat
    )

    # R² regression
    glft_fit = fit_r2(emp_offsets, glft_theory)
    as_fit   = fit_r2(emp_offsets, as_theory)

    # Summary
    r2_g = glft_fit["r2"]
    r2_a = as_fit["r2"]

    if np.isnan(r2_g):
        recovery_verdict = "INSUFFICIENT DATA"
    elif r2_g > 0.6:
        recovery_verdict = "STRONG RECOVERY (R² > 0.6)"
    elif r2_g > 0.5:
        recovery_verdict = "MODERATE RECOVERY (R² > 0.5)"
    elif r2_g > 0.3:
        recovery_verdict = "WEAK RECOVERY (R² > 0.3)"
    else:
        recovery_verdict = "NO RECOVERY (R² < 0.3)"

    summary = (
        f"GLFT R²={r2_g:.3f}  AS R²={r2_a:.3f}  "
        f"GLFT slope={glft_fit['slope']:.3f}  "
        f"→ {recovery_verdict}"
    )
    print(f"  {summary}", flush=True)

    return {
        "inv_levels":  inv_levels,
        "empirical":   emp_offsets,
        "glft_theory": glft_theory,
        "as_theory":   as_theory,
        "glft_fit":    glft_fit,
        "as_fit":      as_fit,
        "summary":     summary,
    }


def run_recovery_all_agents(
    agents:     dict[str, Any],
    env:        Any,
    enc_type:   str,
    n_episodes: int   = 20,
    **kwargs,
) -> dict[str, dict]:
    """
    Run recovery analysis for multiple agents and return all results.

    Parameters
    ----------
    agents     : dict mapping agent_name → trained agent
    env        : LOBMarketMakingEnv
    enc_type   : str
    n_episodes : int
    **kwargs   : passed to run_recovery_analysis

    Returns
    -------
    dict mapping agent_name → recovery result dict
    """
    results = {}
    for name, agent in agents.items():
        print(f"\n{'─'*50}", flush=True)
        print(f"Recovery analysis: {name}", flush=True)
        results[name] = run_recovery_analysis(
            agent, env, enc_type, n_episodes=n_episodes, **kwargs
        )
    return results


def recovery_summary_df(results: dict[str, dict]) -> "pd.DataFrame":
    """
    Build a summary DataFrame from run_recovery_all_agents output.

    Parameters
    ----------
    results : dict from run_recovery_all_agents

    Returns
    -------
    pd.DataFrame — one row per agent
    """
    import pandas as pd

    rows = []
    for agent_name, r in results.items():
        if r.get("glft_fit") is None:
            continue
        rows.append({
            "agent":           agent_name,
            "r2_glft":         r["glft_fit"]["r2"],
            "r2_as":           r["as_fit"]["r2"],
            "slope_glft":      r["glft_fit"]["slope"],
            "intercept_glft":  r["glft_fit"]["intercept"],
            "p_value_glft":    r["glft_fit"]["p_value"],
            "n_levels":        len(r["inv_levels"]),
            "summary":         r["summary"],
        })

    return (
        pd.DataFrame(rows)
        .sort_values("r2_glft", ascending=False)
        .reset_index(drop=True)
    )