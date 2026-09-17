"""
scripts/run_holdout_eval.py

Held-out test-set evaluation for Round 3: every RL agent's best checkpoint
and every baseline, run on the SAME fixed block of episode seeds that
neither training, checkpoint selection, nor any prior evaluation has ever
touched.

Why this exists (see the Round 3 dashboard's methodology discussion):

1. RL eval rollouts always used the same 3 fixed seeds
   (`seed+10000+{0,1,2}`) at every checkpoint throughout training — so
   "best checkpoint" selection was implicitly chosen against a fixed,
   tiny validation set reused for every comparison, a classic
   selection-bias setup (only 3 unique market draws ever evaluated, no
   matter how many checkpoints).
2. Baselines' 50 "replicate" episodes used seeds `seed+1..seed+50`, which
   is the exact same range RL agents' first 50 *training* episodes used —
   not a fairness problem for the baseline's own number (it doesn't
   learn, so any seed is unbiased), but it means baseline and RL numbers
   were never measured on comparable, let alone identical, market
   conditions.

This script fixes both: one holdout seed block, far outside every range
used above (training: seed+1..seed+n_episodes; RL eval: seed+10000+0..2;
AS-recovery: base seed 500), applied identically to all 8 models per
regime — a paired comparison on genuinely unseen data.

Usage:
    python scripts/run_holdout_eval.py --n_episodes 15
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from training.evaluate import load_agent
from training.factory import build_env
from training.rollout import run_episode as rl_run_episode
from scripts.run_analysis import AGENT_CKPT_KWARGS, best_checkpoints
from scripts.run_baseline import build_baseline, run_episode as baseline_run_episode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR   = PROJECT_ROOT / "training" / "configs"
OUT_DIR      = PROJECT_ROOT / "evaluation" / "results_scaled_down_sweep"

BASE_SEED       = 42
# Far outside training (seed+1..seed+500 = 43..542), RL eval
# (seed+10000+{0,1,2} = 10042..10044), and AS-recovery (base seed 500,
# 500..514) — never touched by anything upstream of this script.
HOLDOUT_OFFSET  = 80000
BASELINES       = ["fixedspread", "as", "glft"]
RL_AGENTS       = list(AGENT_CKPT_KWARGS)


def _load_env_cfg(regime: str):
    env_cfg = OmegaConf.load(CONFIG_DIR / "env" / "base.yaml")
    OmegaConf.set_struct(env_cfg, False)
    env_cfg.regime     = regime
    env_cfg.use_abides = True
    return env_cfg


def build_regime_env(regime: str, seed: int):
    env_cfg = _load_env_cfg(regime)
    reward_cfg = OmegaConf.load(CONFIG_DIR / "reward" / "asymmetric.yaml")
    return build_env(env_cfg, reward_cfg, seed)


def run_regime(regime: str, n_episodes: int, holdout_seeds: list[int]) -> list[dict]:
    rows = []
    env_cfg = _load_env_cfg(regime)

    # ── Baselines ────────────────────────────────────────────────────
    env = build_regime_env(regime, seed=BASE_SEED)
    for name in BASELINES:
        # Q_max=10 is GLFT's own ODE-grid size in lots, not env_cfg.Q_max
        # (real shares, ~1000x too large for a tractable matrix
        # exponential) — matches scripts/run_baseline.py exactly.
        baseline = build_baseline(name, env_cfg.get("episode_len", 390),
                                   env_cfg.get("tick_size", 0.01), 10, regime)
        for i, seed in enumerate(holdout_seeds):
            t0 = time.time()
            m = baseline_run_episode(env, baseline, seed=seed)
            rows.append({"model": name, "type": "baseline", "regime": regime,
                         "seed": seed, "sharpe": m["sharpe"], "map": m["map"],
                         "mdd": m["mdd"], "final_pnl": m["final_pnl"]})
            print(f"  [{regime}] {name:12s} ep {i+1:2d}/{n_episodes} "
                  f"sharpe {m['sharpe']:+.3f}  ({time.time()-t0:.0f}s)", flush=True)
    env.close()

    # ── RL agents (best checkpoint each) ────────────────────────────
    ckpts = best_checkpoints(regime)
    for agent_type in RL_AGENTS:
        ckpt = ckpts.get(agent_type)
        if ckpt is None:
            print(f"  [{regime}] {agent_type}: no checkpoint found — skipping")
            continue
        agent, _ = load_agent(ckpt, agent_type=agent_type, encoder_type="handcrafted",
                               **AGENT_CKPT_KWARGS[agent_type])
        env = build_regime_env(regime, seed=BASE_SEED)
        for i, seed in enumerate(holdout_seeds):
            t0 = time.time()
            m = rl_run_episode(env, agent, "handcrafted", training=False, seed=seed)
            rows.append({"model": agent_type, "type": "rl", "regime": regime,
                         "seed": seed, "sharpe": m["sharpe"], "map": m["map"],
                         "mdd": m["mdd"], "final_pnl": m["final_pnl"]})
            print(f"  [{regime}] {agent_type:12s} ep {i+1:2d}/{n_episodes} "
                  f"sharpe {m['sharpe']:+.3f}  ({time.time()-t0:.0f}s)  ckpt={Path(ckpt).name}",
                  flush=True)
        env.close()

    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=15)
    p.add_argument("--regimes", nargs="+", default=["low_vol", "normal", "high_vol"])
    args = p.parse_args()

    holdout_seeds = [BASE_SEED + HOLDOUT_OFFSET + i for i in range(args.n_episodes)]
    print(f"Holdout seed block: {holdout_seeds[0]}..{holdout_seeds[-1]} "
          f"({args.n_episodes} episodes, identical across every model)")

    all_rows = []
    for regime in args.regimes:
        print(f"\n{'='*78}\nHOLDOUT EVAL — {regime}\n{'='*78}")
        all_rows.extend(run_regime(regime, args.n_episodes, holdout_seeds))

    df = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "holdout_eval.csv", index=False)

    summary = (
        df.groupby(["model", "type", "regime"])["sharpe"]
        .agg(sharpe_mean="mean", sharpe_std="std", n="count")
        .reset_index()
        .sort_values(["regime", "sharpe_mean"], ascending=[True, False])
    )
    summary.to_csv(OUT_DIR / "holdout_summary.csv", index=False)

    print(f"\n{'='*78}\nHOLDOUT SUMMARY\n{'='*78}")
    print(summary.to_string(index=False))
    print(f"\nSaved → {OUT_DIR / 'holdout_eval.csv'}, {OUT_DIR / 'holdout_summary.csv'}")


if __name__ == "__main__":
    main()
