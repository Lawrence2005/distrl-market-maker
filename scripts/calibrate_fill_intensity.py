"""
scripts/calibrate_fill_intensity.py

Empirically calibrates AS/GLFT's fill-intensity model lambda(delta) =
A * exp(-kappa*delta) against this specific ABIDES instance, replacing the
uncited-to-ABIDES kappa=100 default and the AS(2008)-paper illustrative
kappa=1.5 fallback (both since found to be wrong for this market: kappa=100
implied a ~$0.02 spread that's meaninglessly tight, kappa=1.5 implied a
~$0.65 spread wide enough to get zero fills in 6/6 test episodes).

Method: quote a fixed symmetric spread (FixedSpreadBaseline) at each of a
grid of tick offsets, run several episodes at each, measure the realized
fill rate (fill events per second, pooling bid+ask by symmetry), then fit
ln(lambda) = ln(A) - kappa*delta by least squares (delta in dollars, per
avellaneda_stoikov.py's own documented convention).

Usage:
    python scripts/calibrate_fill_intensity.py --regime low_vol --n_episodes 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.factory import build_env
from baselines.fixed_spread import FixedSpreadBaseline

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "training" / "configs"

TIMESTEP_SECONDS = 60.0  # matches AbidesMarketMakingEnv's timestep_duration


def measure_fill_rate(env, half_spread_ticks: int, n_episodes: int, seed: int) -> float:
    """Run n_episodes at a fixed quote distance, return fill events/sec (bid+ask pooled)."""
    baseline = FixedSpreadBaseline(half_spread_ticks=half_spread_ticks)
    fill_events = 0
    total_seconds = 0.0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        baseline.reset()
        terminated = truncated = False
        while not (terminated or truncated):
            action = baseline.act(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            fill_events += int(info.get("bid_filled", 0) > 0)
            fill_events += int(info.get("ask_filled", 0) > 0)
            total_seconds += TIMESTEP_SECONDS

    return fill_events / total_seconds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--regime", default="low_vol")
    p.add_argument("--reward", default="asymmetric")
    p.add_argument("--n_episodes", type=int, default=4)
    p.add_argument("--seed", type=int, default=1000)
    args = p.parse_args()

    env_cfg = OmegaConf.load(_CONFIG_DIR / "env" / "base.yaml")
    OmegaConf.set_struct(env_cfg, False)
    env_cfg.regime     = args.regime
    env_cfg.use_abides = True
    reward_cfg = OmegaConf.load(_CONFIG_DIR / "reward" / f"{args.reward}.yaml")
    env = build_env(env_cfg, reward_cfg, args.seed)
    tick_size = env.tick_size

    grid_ticks = [10, 20, 30, 40, 50, 70, 90]
    deltas_dollars = []
    lambdas = []

    print(f"{'ticks':>6} {'$offset':>8} {'fills/sec':>12}", flush=True)
    for ticks in grid_ticks:
        rate = measure_fill_rate(env, ticks, args.n_episodes, args.seed)
        delta_dollars = ticks * tick_size
        print(f"{ticks:>6} {delta_dollars:>8.2f} {rate:>12.6f}", flush=True)
        if rate > 0:
            deltas_dollars.append(delta_dollars)
            lambdas.append(rate)

    env.close()

    deltas_dollars = np.array(deltas_dollars)
    log_lambdas    = np.log(np.array(lambdas))

    # ln(lambda) = ln(A) - kappa*delta  ->  linear fit, slope = -kappa
    slope, intercept = np.polyfit(deltas_dollars, log_lambdas, 1)
    kappa = -slope
    A     = np.exp(intercept)

    pred        = intercept + slope * deltas_dollars
    ss_res      = np.sum((log_lambdas - pred) ** 2)
    ss_tot      = np.sum((log_lambdas - log_lambdas.mean()) ** 2)
    r2          = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    print(f"\nFitted: kappa={kappa:.4f} (1/dollar), A={A:.6f} (events/sec), R^2={r2:.4f}")
    print(f"Implied AS spread at gamma=0.1, tau_hat=1.0: "
          f"{(2.0/0.1)*np.log(1.0 + 0.1/kappa):.4f} dollars "
          f"({(2.0/0.1)*np.log(1.0 + 0.1/kappa)/tick_size:.1f} ticks)")


if __name__ == "__main__":
    main()
