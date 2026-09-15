"""
scripts/run_baseline.py

Runs one non-RL analytical baseline (Fixed Spread, Avellaneda-Stoikov, or
GLFT) through the same LOBMarketMakingEnv used for RL training, logging
the same logs/<run_tag>/train_history.json shape training/train.py
produces so evaluation/metrics.py's load_all_runs()/summary_table() can
compare RL agents and baselines in one table with no separate code path.

There's no learned state to checkpoint — these are closed-form quote
rules — so "episode" here means an independent simulation replicate for a
stable mean, not a training iteration.

run_tag follows the same {agent}_{encoder}_{reward}_{regime}_seed{N}
convention load_all_runs() parses by position, using "analytical" in the
encoder slot and a single-token baseline name (fixedspread/as/glft) in the
agent slot.

Usage:
    python scripts/run_baseline.py --baseline fixedspread --regime low_vol --n_episodes 25 --seed 42
    python scripts/run_baseline.py --baseline as           --regime low_vol --n_episodes 25 --seed 42
    python scripts/run_baseline.py --baseline glft         --regime low_vol --n_episodes 25 --seed 42
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.factory import build_env
from training.rollout import _NumpyEncoder, compute_episode_metrics

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "training" / "configs"

_BASELINES = {
    "fixedspread": ("baselines.fixed_spread",       "FixedSpreadBaseline"),
    "as":          ("baselines.avellaneda_stoikov", "AvellanedaStoikovBaseline"),
    "glft":        ("baselines.glft",                "GLFTBaseline"),
}


# kappa/A/sigma: empirically calibrated PER REGIME against this ABIDES
# instance via scripts/calibrate_fill_intensity.py --regime <name> (quoted
# FixedSpreadBaseline at a grid of tick offsets, measured realized fill
# rate, fit lambda(delta) = A*exp(-kappa*delta) in dollars by log-linear
# regression). Regimes differ in abides_overrides.fund_vol (see
# training/configs/env/regime/*.yaml), which materially changes fill
# dynamics — a single global kappa doesn't transfer across regimes, so this
# is keyed by regime rather than one constant.
#
# Superseded two earlier, both-wrong single-value guesses: the original
# uncited kappa=100.0 implied a ~$0.02 (2-tick) spread — implausibly tight —
# and AS(2008)'s own paper-illustrative kappa=1.5 (still used by
# tests/test_baselines.py for formula-arithmetic tests, not market
# validity) implied a ~$0.65 (65-tick) spread that produced zero fills in
# 6/6 verification episodes.
#
# Fitted implied spreads (gamma=0.1) increase monotonically with regime
# volatility, as expected: low_vol 12.6 ticks (R^2=0.94) < normal 26.1
# ticks (R^2=0.93) < high_vol 52.9 ticks (R^2=0.95). sigma follows the same
# ordering within this class's own documented ABIDES-typical range
# (0.0003-0.001) — adapt_sigma (on by default) self-corrects it from
# realized volatility within ~20-30 steps regardless, so this mainly
# matters for the first fraction of each episode.
#
# flash_crash/trending regimes were removed from the project (see git
# history) — only low_vol/normal/high_vol remain.
#
# Re-run calibrate_fill_intensity.py per regime and update this dict if
# the env's fill dynamics change materially (order_size, tick range,
# fund_vol values, etc).
_CALIBRATED_GAMMA = 0.1
_REGIME_CALIBRATION = {
    "low_vol":  dict(kappa=15.8798, A=0.017989, sigma=0.0003),
    "normal":   dict(kappa=7.6261,  A=0.006294, sigma=0.0007),
    "high_vol": dict(kappa=3.7333,  A=0.011559, sigma=0.0010),
}
_DEFAULT_CALIBRATION = _REGIME_CALIBRATION["normal"]


def build_baseline(name: str, episode_len: int, tick_size: float, Q_max: int, regime: str):
    module_name, class_name = _BASELINES[name]
    cls = getattr(importlib.import_module(module_name), class_name)

    if name == "fixedspread":
        # 30 ticks = $0.30 — confirmed by scripts/calibrate_fill_intensity.py
        # (low_vol regime) to actually get filled (~0.27 mfills/sec) unlike
        # 70-90 ticks (zero fills in the calibration episodes); the old
        # default (2 ticks = $0.02) was far too tight.
        return cls(half_spread_ticks=30)

    calib = _REGIME_CALIBRATION.get(regime, _DEFAULT_CALIBRATION)
    kwargs = dict(T=episode_len, tick_size=tick_size, gamma=_CALIBRATED_GAMMA,
                  kappa=calib["kappa"], sigma=calib["sigma"])
    if name == "glft":
        kwargs["Q_max"] = Q_max
        kwargs["A"]     = calib["A"]
    return cls(**kwargs)


def run_episode(env, baseline, seed: int) -> dict:
    obs, info = env.reset(seed=seed)
    baseline.reset()

    step_pnls, inventories, cum_pnls = [], [], []
    cum_pnl  = 0.0
    prev_mid = info["mid_price"]
    prev_inv = 0
    terminated = truncated = False

    while not (terminated or truncated):
        action = baseline.act(obs, info)
        next_obs, reward, terminated, truncated, next_info = env.step(action)

        inv = next_info["inventory"]
        mid = next_info["mid_price"]
        step_pnl = next_info.get("spread_pnl", 0.0) + prev_inv * (mid - prev_mid)
        cum_pnl += step_pnl

        step_pnls.append(step_pnl)
        inventories.append(inv)
        cum_pnls.append(cum_pnl)

        obs, info, prev_mid, prev_inv = next_obs, next_info, mid, inv

    metrics = compute_episode_metrics(step_pnls, inventories, cum_pnls)
    metrics["mean_loss"] = 0.0
    metrics["steps"]     = len(step_pnls)
    metrics["epsilon"]   = 0.0
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, choices=list(_BASELINES))
    p.add_argument("--regime", default="low_vol")
    p.add_argument("--reward", default="asymmetric")
    p.add_argument("--n_episodes", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_abides", type=lambda s: s.lower() != "false", default=True)
    args = p.parse_args()

    env_cfg = OmegaConf.load(_CONFIG_DIR / "env" / "base.yaml")
    OmegaConf.set_struct(env_cfg, False)
    env_cfg.regime     = args.regime
    env_cfg.use_abides = args.use_abides
    reward_cfg = OmegaConf.load(_CONFIG_DIR / "reward" / f"{args.reward}.yaml")

    env = build_env(env_cfg, reward_cfg, args.seed)
    baseline = build_baseline(
        args.baseline,
        env_cfg.get("episode_len", 390),
        env_cfg.get("tick_size", 0.01),
        10,  # GLFT's own ODE-grid Q_max, in lots — NOT env_cfg.Q_max (real
             # shares, ~1000x too large for a tractable matrix exponential;
             # GLFTBaseline converts the env's real inventory into lots
             # itself via its lot_size param, default 100)
        args.regime,
    )

    run_tag = f"{args.baseline}_analytical_{args.reward}_{args.regime}_seed{args.seed}"
    project_root = Path(__file__).resolve().parents[1]
    log_dir = project_root / "logs" / run_tag
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run tag: {run_tag}", flush=True)
    print(f"Logs:    {log_dir}", flush=True)

    history: list[dict] = []
    t_start = time.time()

    for ep in range(1, args.n_episodes + 1):
        metrics = run_episode(env, baseline, seed=args.seed + ep)
        metrics["episode"] = ep
        metrics["elapsed"] = time.time() - t_start
        history.append(metrics)

        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"ep {ep:3d}/{args.n_episodes} | sharpe {metrics['sharpe']:+.3f} | "
            f"map {metrics['map']:.2f} | pnl {metrics['final_pnl']:+.2f} | "
            f"elapsed {metrics['elapsed']:.0f}s",
            flush=True,
        )

        status = {
            "run_tag":      run_tag,
            "state":        "complete" if ep == args.n_episodes else "running",
            "episode":      ep,
            "n_episodes":   args.n_episodes,
            "pct_complete": round(100.0 * ep / args.n_episodes, 1),
            "elapsed_sec":  round(metrics["elapsed"], 1),
            "updated_at":   datetime.now().isoformat(timespec="seconds"),
            "sharpe":       metrics["sharpe"],
            "final_pnl":    metrics["final_pnl"],
        }
        with open(log_dir / "status.json", "w") as f:
            json.dump(status, f, indent=2)

    with open(log_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2, cls=_NumpyEncoder)

    mean_sharpe = float(np.mean([m["sharpe"] for m in history]))
    print(f"\nMean Sharpe over {args.n_episodes} episodes: {mean_sharpe:+.4f}")
    print("Baseline run complete.", flush=True)
    env.close()


if __name__ == "__main__":
    main()
