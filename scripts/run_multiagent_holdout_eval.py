"""
scripts/run_multiagent_holdout_eval.py

Held-out evaluation pass for the N=2 QR-DQN multi-agent tournament
(scripts/run_multiagent_tournament.py) — this never got the same rigor as
every single-agent run: the tournament trained to completion (500 episodes,
both agents' checkpoints saved) but was never evaluated on unseen episodes,
only ever measured via its own training-time (exploration-contaminated)
history.

Loads both agents' FINAL checkpoints (ep00500 — there is no "best" checkpoint
concept here, since the tournament script never tracked eval_history.json the
way single-agent training.train.py does) and rolls out 15 held-out episodes,
both agents greedy, no further training — reusing
scripts.run_multiagent_tournament.run_episode(training=False) directly rather
than reimplementing episode mechanics.

Seed block: BASE_SEED+80000+i, the same holdout offset scripts/run_holdout_eval.py
uses — disjoint from training's BASE_SEED+1..+500 (43-542) range. Requires the
envs/multi_agent_env.py ABIDES-seeding fix (MultiAgentMarketEnv.reset() now
forwards seed into self._abides.np_random) to be reproducible/meaningful at
all — without it, this environment has the identical wall-clock-seeding bug
envs/lob_env.py had before this session's fix.

IMPORTANT CAVEAT (see scripts/run_multiagent_tournament.py's own comment):
MultiAgentMarketEnv has no `regime` parameter — it always uses rmsc04's
hardcoded defaults, not calibrated to match the single-agent `normal` regime's
fund_vol overlay. So the single-agent qrdqn/normal holdout Sharpe (+4.19) is
NOT a clean apples-to-apples comparison for this script's numbers — reported
alongside for context, not as a fair baseline.

Usage:
    python scripts/run_multiagent_holdout_eval.py --n_episodes 15
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from envs.multi_agent_env import MultiAgentMarketEnv
from scripts.run_multiagent_tournament import (
    N_AGENTS, REGIME, BASE_SEED, CKPT_DIR, build_agent, run_episode,
)

PROJECT_ROOT   = Path(__file__).resolve().parents[1]
OUT_DIR        = PROJECT_ROOT / "evaluation" / "results_scaled_down_sweep"
HOLDOUT_OFFSET = 80000  # matches scripts/run_holdout_eval.py's convention


def load_final_agents() -> list:
    agents = [build_agent(seed_offset=i) for i in range(N_AGENTS)]
    eps = []
    for i in range(N_AGENTS):
        ckpts = sorted(CKPT_DIR.glob(f"agent{i}_ep*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No checkpoints found for agent{i} in {CKPT_DIR}")
        last = ckpts[-1]
        eps.append(int(last.stem.split("_ep")[1]))
        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        agents[i].load_state_dict(ckpt["agent_state"])
        print(f"  agent{i}: loaded {last.name}")
    assert len(set(eps)) == 1, f"Agents' final checkpoints are at different episodes: {eps}"
    return agents


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=15)
    args = p.parse_args()

    holdout_seeds = [BASE_SEED + HOLDOUT_OFFSET + i for i in range(args.n_episodes)]
    print(f"Holdout seed block: {holdout_seeds[0]}..{holdout_seeds[-1]} "
          f"({args.n_episodes} episodes, disjoint from training's 43-542 range)")

    agents = load_final_agents()
    env = MultiAgentMarketEnv(n_agents=N_AGENTS, episode_len=390, order_size=100,
                               reward_type="asymmetric", eta=0.5,
                               use_abides=True, seed=BASE_SEED)

    rows, market_rows = [], []
    for i, seed in enumerate(holdout_seeds):
        per_agent_metrics, market = run_episode(env, agents, seed=seed, training=False)
        for a in range(N_AGENTS):
            m = dict(per_agent_metrics[a])
            m["agent"] = a
            m["seed"] = seed
            rows.append(m)
        market_row = dict(market)
        market_row["seed"] = seed
        market_rows.append(market_row)
        sh = [f"{per_agent_metrics[a]['sharpe']:+.3f}" for a in range(N_AGENTS)]
        print(f"  ep {i+1:2d}/{args.n_episodes}  sharpe(agents)={sh}  "
              f"mean_spread={market['mean_spread']:.4f}", flush=True)
    env.close()

    df = pd.DataFrame(rows)
    market_df = pd.DataFrame(market_rows)

    summary = (
        df.groupby("agent")[["sharpe", "map", "mdd", "final_pnl", "fill_rate"]]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "multiagent_holdout_episodes.csv", index=False)
    summary.to_csv(OUT_DIR / "multiagent_holdout_summary.csv", index=False)
    market_df.to_csv(OUT_DIR / "multiagent_holdout_market.csv", index=False)

    print(f"\n{'='*78}\nMULTI-AGENT HOLDOUT SUMMARY (N={N_AGENTS}, regime='{REGIME}' — "
          f"see module docstring's calibration caveat)\n{'='*78}")
    print(summary.to_string(index=False))
    print(f"\nMarket quality — mean_spread: {market_df['mean_spread'].mean():.4f} "
          f"(std across episodes {market_df['mean_spread'].std():.4f}), "
          f"spread_std: {market_df['spread_std'].mean():.4f}")
    print(f"\nSaved → {OUT_DIR / 'multiagent_holdout_episodes.csv'}, "
          f"{OUT_DIR / 'multiagent_holdout_summary.csv'}, "
          f"{OUT_DIR / 'multiagent_holdout_market.csv'}")


if __name__ == "__main__":
    main()
