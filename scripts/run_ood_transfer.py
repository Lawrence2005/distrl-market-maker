"""
scripts/run_ood_transfer.py

Round 4 OOD-transfer workstream: evaluate each agent's already-trained
`low_vol` checkpoint (no new training) inside the `high_vol` environment,
to measure out-of-distribution performance degradation under a harder
regime the agent never saw during training.

Uses a fresh, disjoint holdout seed block (`seed+90000+i`) — every other
range is already spoken for: training (seed+1..seed+n_episodes), RL eval
(seed+10000+{0,1,2}), AS-recovery (base seed 500), same-regime holdout
(seed+80000+i).

Reports, per agent:
    in_dist  — held-out low_vol Sharpe (from evaluation/results_scaled_down_sweep/holdout_eval.csv)
    ood      — held-out Sharpe of the SAME low_vol checkpoint, rolled out in high_vol
    degradation_pct — (in_dist - ood) / (|in_dist| + eps), matching
                       evaluation/visualize.py's plot_ood_transfer convention.

Usage:
    python scripts/run_ood_transfer.py --n_episodes 15
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from training.evaluate import load_agent
from training.rollout import run_episode as rl_run_episode
from scripts.run_analysis import AGENT_CKPT_KWARGS, best_checkpoints, build_regime_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR      = PROJECT_ROOT / "evaluation" / "results_scaled_down_sweep"

BASE_SEED      = 42
OOD_OFFSET     = 90000  # disjoint from training/eval/AS-recovery/holdout (80000) blocks


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=15)
    args = p.parse_args()

    ood_seeds = [BASE_SEED + OOD_OFFSET + i for i in range(args.n_episodes)]
    print(f"OOD holdout seed block: {ood_seeds[0]}..{ood_seeds[-1]} "
          f"({args.n_episodes} episodes, disjoint from every other seed range)")

    # Source: each agent's already-trained BEST low_vol checkpoint.
    low_vol_ckpts = best_checkpoints("low_vol")

    rows = []
    env = build_regime_env("high_vol", seed=BASE_SEED)
    for agent_type, ckpt in low_vol_ckpts.items():
        agent, _ = load_agent(ckpt, agent_type=agent_type, encoder_type="handcrafted",
                               **AGENT_CKPT_KWARGS[agent_type])
        for i, seed in enumerate(ood_seeds):
            t0 = time.time()
            m = rl_run_episode(env, agent, "handcrafted", training=False, seed=seed)
            rows.append({"model": agent_type, "seed": seed, "sharpe": m["sharpe"],
                         "map": m["map"], "mdd": m["mdd"], "final_pnl": m["final_pnl"]})
            print(f"  {agent_type:8s} ep {i+1:2d}/{args.n_episodes} "
                  f"sharpe {m['sharpe']:+.3f}  ({time.time()-t0:.0f}s)  "
                  f"ckpt={Path(ckpt).name}", flush=True)
    env.close()

    ood_df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ood_df.to_csv(OUT_DIR / "ood_transfer_episodes.csv", index=False)

    ood_summary = ood_df.groupby("model")["sharpe"].agg(ood_sharpe="mean", ood_std="std").reset_index()

    # In-distribution reference: the SAME low_vol checkpoints' already-computed
    # held-out low_vol Sharpe from holdout_summary.csv (rl rows, low_vol regime,
    # "low" as parsed by evaluation.metrics — but holdout_summary.csv stores the
    # regime string as passed to run_holdout_eval.py, i.e. "low_vol" verbatim).
    holdout_summary_path = OUT_DIR / "holdout_summary.csv"
    if not holdout_summary_path.exists():
        raise FileNotFoundError(
            f"{holdout_summary_path} not found — run scripts/run_holdout_eval.py first "
            "(in_dist reference numbers come from its low_vol rl rows)."
        )
    holdout_summary = pd.read_csv(holdout_summary_path)
    in_dist = holdout_summary[
        (holdout_summary["type"] == "rl") & (holdout_summary["regime"] == "low_vol")
    ][["model", "sharpe_mean"]].rename(columns={"sharpe_mean": "in_dist_sharpe"})

    transfer_df = in_dist.merge(ood_summary, on="model", how="inner")
    transfer_df["degradation_pct"] = (
        (transfer_df["in_dist_sharpe"] - transfer_df["ood_sharpe"])
        / (transfer_df["in_dist_sharpe"].abs() + 1e-10) * 100.0
    )
    transfer_df = transfer_df.sort_values("degradation_pct")

    transfer_df.to_csv(OUT_DIR / "ood_transfer_summary.csv", index=False)

    print(f"\n{'='*78}\nOOD TRANSFER SUMMARY — low_vol-trained agents evaluated in high_vol\n{'='*78}")
    print(transfer_df.to_string(index=False))
    print(f"\nSaved → {OUT_DIR / 'ood_transfer_episodes.csv'}, {OUT_DIR / 'ood_transfer_summary.csv'}")


if __name__ == "__main__":
    main()
