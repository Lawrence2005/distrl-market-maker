"""
scripts/run_ood_transfer.py

OOD-transfer evaluation: evaluate each agent's already-trained
checkpoint (no new training) inside a DIFFERENT regime's environment than it
was trained in, to measure out-of-distribution performance degradation under
a regime shift the agent never saw during training.

All 6 directed regime pairs (every ordered pair among low_vol/normal/high_vol):
    low_vol -> high_vol   (original 3)
    normal  -> low_vol
    normal  -> high_vol
    low_vol -> normal     (added later — completes the 3x2 directed-pair grid)
    high_vol -> normal
    high_vol -> low_vol

Only ADDED_SCENARIOS actually get rolled out when this runs now — the
original 3 already have saved, correct (seed=42, deterministic) results in
ood_transfer_episodes.csv, so re-running them would just burn real-ABIDES
wall time to reproduce identical numbers. main() merges the new rollouts
with the existing ones rather than recomputing from scratch.

Uses a fresh, disjoint holdout seed block (`seed+90000+i`) — every other
range is already spoken for: training (seed+1..seed+n_episodes), RL eval
(seed+10000+{0,1,2}), AS-recovery (base seed 500), same-regime holdout
(seed+80000+i). The same seed block is reused across scenarios that share a
test_regime (e.g. both high_vol scenarios use seeds 90042..90056 against the
high_vol env) — that's intentional, not a collision: different scenarios
evaluate different checkpoints, so identical held-out episodes make the
comparison apples-to-apples rather than needing disjoint seeds per scenario.

Reports, per (agent, scenario):
    in_dist  — held-out Sharpe in the TRAINING regime (from
               results/holdout_summary.csv)
    ood      — held-out Sharpe of the SAME checkpoint, rolled out in the
               TEST regime
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
OUT_DIR      = PROJECT_ROOT / "results"

BASE_SEED  = 42
OOD_OFFSET = 90000  # disjoint from training/eval/AS-recovery/holdout (80000) blocks

ORIGINAL_SCENARIOS = [
    ("low_vol", "high_vol"),
    ("normal", "low_vol"),
    ("normal", "high_vol"),
]
ADDED_SCENARIOS = [
    ("low_vol", "normal"),
    ("high_vol", "normal"),
    ("high_vol", "low_vol"),
]
TRANSFER_SCENARIOS = ORIGINAL_SCENARIOS + ADDED_SCENARIOS


def run_scenario(train_regime: str, test_regime: str, ood_seeds: list[int]) -> pd.DataFrame:
    print(f"\n--- Scenario: {train_regime} -> {test_regime} ---")

    # Source: each agent's already-trained BEST checkpoint from train_regime.
    ckpts = best_checkpoints(train_regime)

    rows = []
    env = build_regime_env(test_regime, seed=BASE_SEED)
    for agent_type, ckpt in ckpts.items():
        agent, _ = load_agent(ckpt, agent_type=agent_type, encoder_type="handcrafted",
                               **AGENT_CKPT_KWARGS[agent_type])
        for i, seed in enumerate(ood_seeds):
            t0 = time.time()
            m = rl_run_episode(env, agent, "handcrafted", training=False, seed=seed)
            rows.append({"train_regime": train_regime, "test_regime": test_regime,
                         "model": agent_type, "seed": seed, "sharpe": m["sharpe"],
                         "map": m["map"], "mdd": m["mdd"], "final_pnl": m["final_pnl"]})
            print(f"  {agent_type:8s} ep {i+1:2d}/{len(ood_seeds)} "
                  f"sharpe {m['sharpe']:+.3f}  ({time.time()-t0:.0f}s)  "
                  f"ckpt={Path(ckpt).name}", flush=True)
    env.close()

    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=15)
    args = p.parse_args()

    ood_seeds = [BASE_SEED + OOD_OFFSET + i for i in range(args.n_episodes)]
    print(f"OOD holdout seed block: {ood_seeds[0]}..{ood_seeds[-1]} "
          f"({args.n_episodes} episodes, disjoint from every other seed range)")

    # Per-scenario checkpointing: each finished scenario's rollout is saved to
    # its own partial file immediately, and a scenario whose partial file
    # already exists is loaded from disk instead of re-run. Without this, an
    # interruption anywhere in this ~2h run (this project has already lost a
    # whole training job to what looked like a laptop sleep but was actually
    # a full WSL2 VM restart — see research_gap_closing_plan.md) would lose
    # ALL scenarios completed so far, not just the one in flight, since the
    # original version only wrote output once at the very end.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    partial_dir = OUT_DIR / "_ood_transfer_partial"
    partial_dir.mkdir(exist_ok=True)

    new_frames = []
    for train_regime, test_regime in ADDED_SCENARIOS:
        partial_path = partial_dir / f"{train_regime}_to_{test_regime}.csv"
        if partial_path.exists():
            print(f"\n--- Scenario: {train_regime} -> {test_regime} --- "
                  f"[already done, loading from {partial_path.name}]")
            new_frames.append(pd.read_csv(partial_path))
            continue
        scenario_df = run_scenario(train_regime, test_regime, ood_seeds)
        scenario_df.to_csv(partial_path, index=False)
        new_frames.append(scenario_df)
    new_df = pd.concat(new_frames, ignore_index=True)

    existing_path = OUT_DIR / "ood_transfer_episodes.csv"
    if existing_path.exists():
        existing_df = pd.read_csv(existing_path)
        ood_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        ood_df = new_df

    ood_df.to_csv(existing_path, index=False)
    # All 3 added scenarios are now folded into the canonical file — the
    # per-scenario partials have served their purpose (resume safety net)
    # and would otherwise sit around as stale duplicate data.
    for f in partial_dir.glob("*.csv"):
        f.unlink()
    partial_dir.rmdir()

    ood_summary = (
        ood_df.groupby(["train_regime", "test_regime", "model"])["sharpe"]
        .agg(ood_sharpe="mean", ood_std="std")
        .reset_index()
    )

    # In-distribution reference: each scenario's SAME checkpoints' already-
    # computed held-out Sharpe IN THEIR OWN TRAINING REGIME, from
    # holdout_summary.csv (rl rows). Joined per-scenario on train_regime so
    # e.g. the normal->low_vol and normal->high_vol scenarios both reference
    # the normal-regime in-distribution number, not low_vol's.
    holdout_summary_path = OUT_DIR / "holdout_summary.csv"
    if not holdout_summary_path.exists():
        raise FileNotFoundError(
            f"{holdout_summary_path} not found — run scripts/run_holdout_eval.py first "
            "(in_dist reference numbers come from its per-regime rl rows)."
        )
    holdout_summary = pd.read_csv(holdout_summary_path)
    in_dist = holdout_summary[holdout_summary["type"] == "rl"][
        ["model", "regime", "sharpe_mean"]
    ].rename(columns={"regime": "train_regime", "sharpe_mean": "in_dist_sharpe"})

    transfer_df = in_dist.merge(ood_summary, on=["train_regime", "model"], how="inner")
    transfer_df["degradation_pct"] = (
        (transfer_df["in_dist_sharpe"] - transfer_df["ood_sharpe"])
        / (transfer_df["in_dist_sharpe"].abs() + 1e-10) * 100.0
    )
    transfer_df = transfer_df.sort_values(["train_regime", "test_regime", "degradation_pct"])

    transfer_df.to_csv(OUT_DIR / "ood_transfer_summary.csv", index=False)

    print(f"\n{'='*78}\nOOD TRANSFER SUMMARY\n{'='*78}")
    print(transfer_df.to_string(index=False))
    print(f"\nSaved → {OUT_DIR / 'ood_transfer_episodes.csv'}, {OUT_DIR / 'ood_transfer_summary.csv'}")


if __name__ == "__main__":
    main()
