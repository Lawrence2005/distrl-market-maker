"""
scripts/run_cvar_sweep_eval.py

CVaR alpha sweep evaluation. QR-DQN and IQN were
each trained at alpha in {0.05, 0.10, 0.25(already trained separately), 0.50, 1.0},
normal regime — this script evaluates all 5 alpha values per agent on the
SAME held-out seed block scripts/run_holdout_eval.py uses (seed+80000+i),
runs AS-recovery for each, and builds the CVaR efficient frontier (mean
held-out P&L vs. CVaR_0.10 of held-out per-episode PnLs) directly from
that held-out data — not from in-sample train/eval history, unlike
evaluation/efficient_frontier.py's EfficientFrontier class (which also
guesses a run_tag format that doesn't match this project's actual
f"_alpha{alpha:.2f}" convention from training/train.py, e.g. alpha=1.0 →
"alpha1.00" not "alpha1").

alpha=0.25 is NOT re-run here — its 15 held-out episodes already exist in
holdout_eval.csv from scripts/run_holdout_eval.py, using the identical
seed block, so it's pulled in from there instead of duplicating rollouts.

Usage:
    python scripts/run_cvar_sweep_eval.py --n_episodes 15
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
from training.rollout import run_episode as rl_run_episode, find_best_checkpoint, find_latest_checkpoint
from evaluation.as_recovery import run_recovery_all_agents, recovery_summary_df
from evaluation.ablation import _run_tag as _ablation_run_tag
from scripts.run_analysis import AGENT_CKPT_KWARGS, _MIN_EPISODES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR   = PROJECT_ROOT / "training" / "configs"
OUT_DIR      = PROJECT_ROOT / "evaluation" / "results_scaled_down_sweep"

BASE_SEED      = 42
HOLDOUT_OFFSET = 80000  # matches scripts/run_holdout_eval.py — same seed block, comparable data
# Was 500 — with n_episodes=500 training runs use seed+1..seed+500 (43-542),
# so seeds 500-514 replayed each agent's own training episodes 458-472
# instead of unseen data (same bug found and fixed the same way in
# scripts/run_encoder_ablation_eval.py). 95000 is disjoint from training,
# RL-eval-during-training (10042-10044), and the holdout block (80042+).
AS_RECOVERY_SEED = 95000

AGENTS       = ["qrdqn", "iqn"]
NEW_ALPHAS   = [0.05, 0.10, 0.50, 1.0]   # 0.25 already exists in holdout_eval.csv
ALL_ALPHAS   = [0.05, 0.10, 0.25, 0.50, 1.0]
REGIME       = "normal"


def _run_tag(agent_type: str, alpha: float) -> str:
    # Delegates to evaluation.ablation's run_tag builder (single source of
    # truth matching training/train.py's convention) — this used to
    # hand-build the tag missing sampling_tag, silently breaking every
    # lookup here the moment that tag was introduced. sampling="per"
    # explicitly: qrdqn/iqn.yaml set prioritized_replay=true for every alpha
    # in this sweep, so the "_uniform" tag _ablation_run_tag defaults to
    # points at a checkpoint/log dir that doesn't exist (same bug just found
    # and fixed in scripts/run_analysis.py and scripts/build_dashboard_data.py).
    return _ablation_run_tag(agent_type, "handcrafted", "asymmetric", REGIME, BASE_SEED,
                              alpha=alpha, sampling="per")


def _resolve_best_checkpoint(run_tag: str, ext: str) -> Path | None:
    ckpt_dir = PROJECT_ROOT / "checkpoints" / run_tag
    best = find_best_checkpoint(ckpt_dir)
    if best is not None:
        return best

    import json
    eval_hist_path = PROJECT_ROOT / "logs" / run_tag / "eval_history.json"
    if eval_hist_path.exists():
        with open(eval_hist_path) as f:
            eval_history = json.load(f)
        eligible = [e for e in eval_history if e["episode"] >= _MIN_EPISODES]
        if eligible:
            best_ep = max(eligible, key=lambda e: e["sharpe"])["episode"]
            candidate = ckpt_dir / f"ep{best_ep:05d}.{ext}"
            if candidate.exists():
                return candidate

    return find_latest_checkpoint(ckpt_dir)


def build_regime_env(seed: int):
    env_cfg = OmegaConf.load(CONFIG_DIR / "env" / "base.yaml")
    OmegaConf.set_struct(env_cfg, False)
    env_cfg.regime     = REGIME
    env_cfg.use_abides = True
    reward_cfg = OmegaConf.load(CONFIG_DIR / "reward" / "asymmetric.yaml")
    return build_env(env_cfg, reward_cfg, seed)


def run_holdout_for_alpha(agent_type: str, alpha: float, holdout_seeds: list[int]) -> list[dict]:
    run_tag = _run_tag(agent_type, alpha)
    ckpt = _resolve_best_checkpoint(run_tag, ext="pt")
    if ckpt is None:
        print(f"  [skip] {run_tag}: no checkpoint found")
        return []

    agent, _ = load_agent(str(ckpt), agent_type=agent_type, encoder_type="handcrafted",
                           **AGENT_CKPT_KWARGS[agent_type])
    env = build_regime_env(seed=BASE_SEED)
    rows = []
    for i, seed in enumerate(holdout_seeds):
        t0 = time.time()
        m = rl_run_episode(env, agent, "handcrafted", training=False, seed=seed)
        rows.append({"agent": agent_type, "alpha": alpha, "seed": seed,
                     "sharpe": m["sharpe"], "map": m["map"], "mdd": m["mdd"],
                     "final_pnl": m["final_pnl"]})
        print(f"  {agent_type} alpha={alpha:.2f} ep {i+1:2d}/{len(holdout_seeds)} "
              f"sharpe {m['sharpe']:+.3f}  ({time.time()-t0:.0f}s)  ckpt={ckpt.name}", flush=True)
    env.close()
    return rows


def run_as_recovery_for_alpha(agent_type: str, alpha: float) -> dict | None:
    run_tag = _run_tag(agent_type, alpha)
    ckpt = _resolve_best_checkpoint(run_tag, ext="pt")
    if ckpt is None:
        return None

    agent, _ = load_agent(str(ckpt), agent_type=agent_type, encoder_type="handcrafted",
                           **AGENT_CKPT_KWARGS[agent_type])
    env = build_regime_env(seed=AS_RECOVERY_SEED)
    results = run_recovery_all_agents({agent_type: agent}, env, enc_type="handcrafted",
                                       n_episodes=15, seed=AS_RECOVERY_SEED)
    env.close()
    df = recovery_summary_df(results)
    df["alpha"] = alpha
    return df


def load_existing_alpha025(regime: str = REGIME) -> pd.DataFrame:
    """Pull the already-computed alpha=0.25 holdout rows out of holdout_eval.csv."""
    holdout_path = OUT_DIR / "holdout_eval.csv"
    df = pd.read_csv(holdout_path)
    sub = df[(df["type"] == "rl") & (df["regime"] == regime) & (df["model"].isin(AGENTS))].copy()
    sub = sub.rename(columns={"model": "agent"})
    sub["alpha"] = 0.25
    return sub[["agent", "alpha", "seed", "sharpe", "map", "mdd", "final_pnl"]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=15)
    args = p.parse_args()

    holdout_seeds = [BASE_SEED + HOLDOUT_OFFSET + i for i in range(args.n_episodes)]
    print(f"Holdout seed block: {holdout_seeds[0]}..{holdout_seeds[-1]} "
          f"({args.n_episodes} episodes)")

    all_rows = list(load_existing_alpha025().to_dict("records"))
    as_recovery_frames = []

    for agent_type in AGENTS:
        for alpha in NEW_ALPHAS:
            print(f"\n{'='*78}\nHOLDOUT EVAL — {agent_type} alpha={alpha:.2f}\n{'='*78}")
            all_rows.extend(run_holdout_for_alpha(agent_type, alpha, holdout_seeds))

    for agent_type in AGENTS:
        for alpha in ALL_ALPHAS:
            print(f"\n{'='*78}\nAS-RECOVERY — {agent_type} alpha={alpha:.2f}\n{'='*78}")
            df = run_as_recovery_for_alpha(agent_type, alpha)
            if df is not None:
                as_recovery_frames.append(df)
                print(df.to_string(index=False))

    holdout_df = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    holdout_df.to_csv(OUT_DIR / "cvar_sweep_holdout.csv", index=False)

    if as_recovery_frames:
        as_recovery_df = pd.concat(as_recovery_frames, ignore_index=True)
        as_recovery_df.to_csv(OUT_DIR / "cvar_sweep_as_recovery.csv", index=False)

    # ── Efficient frontier: mean held-out P&L vs. CVaR_0.10 of held-out PnLs ──
    frontier_rows = []
    for agent_type in AGENTS:
        for alpha in ALL_ALPHAS:
            sub = holdout_df[(holdout_df["agent"] == agent_type) & (holdout_df["alpha"] == alpha)]
            if sub.empty:
                continue
            pnls = np.sort(sub["final_pnl"].to_numpy())
            n_tail = max(1, int(0.10 * len(pnls)))
            frontier_rows.append({
                "agent": agent_type,
                "alpha": alpha,
                "mean_pnl": float(pnls.mean()),
                "cvar_10": float(pnls[:n_tail].mean()),
                "pnl_std": float(pnls.std()),
                "mean_sharpe": float(sub["sharpe"].mean()),
                "n_episodes": len(pnls),
            })
    frontier_df = pd.DataFrame(frontier_rows).sort_values(["agent", "alpha"])
    frontier_df.to_csv(OUT_DIR / "cvar_efficient_frontier.csv", index=False)

    print(f"\n{'='*78}\nCVAR EFFICIENT FRONTIER\n{'='*78}")
    print(frontier_df.to_string(index=False))
    print(f"\nSaved → {OUT_DIR / 'cvar_sweep_holdout.csv'}, "
          f"{OUT_DIR / 'cvar_sweep_as_recovery.csv'}, "
          f"{OUT_DIR / 'cvar_efficient_frontier.csv'}")


if __name__ == "__main__":
    main()
