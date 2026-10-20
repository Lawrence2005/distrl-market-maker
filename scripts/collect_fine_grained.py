"""
scripts/collect_fine_grained.py

Collects sub-decision-interval market data from the real ABIDES simulator
into per-second (or any-grid) feature tables for exploratory analysis.

The agent still decides once per decision interval (60 s); between
decisions ABIDES publishes order-book snapshots and fills, which
envs/fine_grained.py's FineGrainedRecorder keeps losslessly and resamples
onto the grid chosen here. The simulation and the policy's observations are
unchanged by recording (see tests/test_fine_grained.py), so every granularity
below comes from the same episodes — only the post-hoc resampling differs.

Policies are the same non-RL quote rules scripts/run_baseline.py runs, plus
a uniformly random policy (the convention scripts/collect_abides_snapshots.py
uses for simulator-characterisation rollouts), plus the trained agents
(dqn, qrdqn, iqn, ppo, sarsa: the regime's best checkpoint, handcrafted
encoder, run greedily as in evaluation).

Writes, under --out_dir/<regime>_<policy>_seed<seed>/:
    grid_<g>s.parquet  one per --grid_s value: one row per grid point with
                   book/fill state + the 18 handcrafted features (same
                   names/definitions as the env's observation), quote and
                   inventory and quote-in-force state, reward on decision rows.
                   The default writes a 5 s grid and a 60 s
                   grid; at 60 s (= the decision interval) every row is a
                   decision row and equals the agent's observation exactly.
    steps.parquet  one row per decision (action, reward, inventory, ...)
    book.parquet   one row per L2 snapshot (event-level, un-resampled), with the
                   trailing-60 s market volume pushed alongside it
                   (mkt_bid_volume, mkt_ask_volume)
    fills.parquet  one row per own fill (event-level)
    meta.json      the arguments and settings that produced them

Seeds default to a block disjoint from the training and held-out evaluation
seeds; the held-out block (80042-80056) is refused unless --allow_held_out,
so exploration there can't leak into a method that is later scored on it.

Usage:
    python scripts/collect_fine_grained.py --regime normal --policy glft --n_episodes 20
    python scripts/collect_fine_grained.py --regime high_vol --policy random --grid_s 1
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from omegaconf import OmegaConf

from envs.fine_grained import FineGrainedRecorder, merge_book_volume
from envs.lob_env import N_OFFSET_LEVELS
from training.evaluate import load_agent
from training.factory import build_env
from training.rollout import decode_action
from scripts.run_analysis import AGENT_CKPT_KWARGS, best_checkpoints
from scripts.run_baseline import build_baseline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR   = PROJECT_ROOT / "training" / "configs"

RULE_POLICIES = ("random", "fixedspread", "as", "glft")
# Trained agents: the regime's best-eval checkpoint (handcrafted encoder, the
# alpha=0.25 run for qrdqn/iqn), the same one scripts/run_holdout_eval.py scores
RL_POLICIES   = tuple(AGENT_CKPT_KWARGS)
POLICIES      = RULE_POLICIES + RL_POLICIES

# Held-out evaluation episodes (scripts/run_holdout_eval.py: BASE_SEED+80000+i)
HELD_OUT_SEEDS = range(80042, 80057)


def build_recording_env(regime: str, args) -> FineGrainedRecorder:
    """Real-ABIDES env for `regime` with recording on, wrapped in a recorder."""
    env_cfg = OmegaConf.load(CONFIG_DIR / "env" / "base.yaml")
    OmegaConf.set_struct(env_cfg, False)
    env_cfg.regime     = regime
    env_cfg.use_abides = True
    env_cfg.market_data.record            = True
    env_cfg.market_data.subscribe_freq_ns = args.subscribe_freq_ns
    env_cfg.market_data.timestep_duration = args.timestep_duration
    if args.episode_len is not None:
        env_cfg.episode_len = args.episode_len
    reward_cfg = OmegaConf.load(CONFIG_DIR / "reward" / "asymmetric.yaml")
    return FineGrainedRecorder(build_env(env_cfg, reward_cfg, args.seed))


def build_policy(name: str, env, regime: str, rng: np.random.Generator):
    """Returns act(obs, info) -> action, and a reset() to call per episode."""
    if name == "random":
        n_actions = N_OFFSET_LEVELS ** 2
        return (lambda obs, info: decode_action(int(rng.integers(n_actions)))), (lambda: None)
    # GLFT's own ODE-grid Q_max is in lots, not the env's share-valued Q_max
    # (same convention as scripts/run_baseline.py)
    if name in RL_POLICIES:
        ckpt = best_checkpoints(regime).get(name)
        if ckpt is None:
            sys.exit(f"No checkpoint found for {name} in {regime}")
        agent, _ = load_agent(ckpt, agent_type=name, encoder_type="handcrafted",
                              **AGENT_CKPT_KWARGS[name])
        print(f"  {name}: checkpoint {ckpt}", flush=True)
        # Same action rule as training/rollout.py::run_episode with
        # training=False: PPO samples its policy, the others act greedily
        if name == "ppo":
            act = lambda obs, info: decode_action(agent.act_with_value(obs)[0])
        else:
            act = lambda obs, info: decode_action(agent.act(obs, greedy=True))
        return act, (lambda: agent.reset_hidden(batch_size=1))
    baseline = build_baseline(name, env.episode_len, env.tick_size, 10, regime)
    return baseline.act, baseline.reset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--regime", default="normal", choices=["low_vol", "normal", "high_vol"])
    p.add_argument("--policy", default="glft", choices=POLICIES)
    p.add_argument("--n_episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=90000,
                   help="Episode i is seeded --seed + i (default 90000: clear of "
                        "training and held-out blocks)")
    p.add_argument("--grid_s", type=str, default="5,60",
                   help="Comma-separated grid spacings, seconds (default 5,60). "
                        "rmsc04 publishes a book snapshot every ~5 s on average "
                        "(median 4 s), so finer grids only forward-fill.")
    p.add_argument("--vol_window_s", type=float, default=1200.0,
                   help="Realized-vol window, seconds (default 1200 = the agent's "
                        "20 decision steps)")
    p.add_argument("--rsi_window_s", type=float, default=840.0,
                   help="RSI window, seconds (default 840 = the agent's 14 steps)")
    p.add_argument("--subscribe_freq_ns", type=int, default=100_000_000,
                   help="Min gap between exchange L2 pushes (default 1e8 = 0.1 s). "
                        "Changes the simulator's message traffic, so also its "
                        "random stream — leave at the default to keep paths "
                        "comparable with existing runs.")
    p.add_argument("--timestep_duration", default=None,
                   help="Decision cadence, e.g. '60s' (default: ABIDES's 60s). "
                        "Changes the MDP; rescale --episode_len to match.")
    p.add_argument("--episode_len", type=int, default=None,
                   help="Decision steps per episode (default: env config, 390)")
    p.add_argument("--out_dir", default="data/processed/fine_grained")
    p.add_argument("--allow_held_out", action="store_true",
                   help="Permit seeds in the held-out evaluation block 80042-80056")
    args = p.parse_args()

    grid_values = [float(x) for x in args.grid_s.split(",")]
    seeds = [args.seed + i for i in range(args.n_episodes)]
    overlap = [s for s in seeds if s in HELD_OUT_SEEDS]
    if overlap and not args.allow_held_out:
        sys.exit(f"Seeds {overlap[0]}..{overlap[-1]} fall in the held-out block "
                 f"{HELD_OUT_SEEDS.start}-{HELD_OUT_SEEDS.stop - 1}; pick another "
                 f"--seed or pass --allow_held_out.")

    rec = build_recording_env(args.regime, args)
    act, reset_policy = build_policy(
        args.policy, rec, args.regime, np.random.default_rng(args.seed),
    )

    t_start = time.time()
    for i, seed in enumerate(seeds):
        t0 = time.time()
        obs, info = rec.reset(seed=seed)
        reset_policy()
        terminated = truncated = False
        n_steps = 0
        while not (terminated or truncated):
            obs, reward, terminated, truncated, info = rec.step(act(obs, info))
            n_steps += 1
        print(f"  ep {i + 1:3d}/{args.n_episodes}  seed={seed}  steps={n_steps}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    # Windows count grid points, so convert from seconds: the features then
    # span the same time as the agent's at any grid spacing.
    grids = {}
    for g in grid_values:
        grids[g] = rec.frames(
            grid_s=g,
            vol_window=max(2, round(args.vol_window_s / g)),
            rsi_window=max(2, round(args.rsi_window_s / g)),
        )
    frames = grids[grid_values[0]]          # event-level tables are grid-independent
    rec.close()

    out = PROJECT_ROOT / args.out_dir / f"{args.regime}_{args.policy}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    for g, fr in grids.items():
        fr["grid"].assign(regime=args.regime, policy=args.policy).to_parquet(
            out / f"grid_{g:g}s.parquet", index=False,
        )
    tables = {"steps": frames["steps"], "fills": frames["fills"],
              "book": merge_book_volume(frames["book"], frames["volume"])}
    for name, table in tables.items():
        table.assign(regime=args.regime, policy=args.policy).to_parquet(
            out / f"{name}.parquet", index=False,
        )
    with open(out / "meta.json", "w") as f:
        json.dump({**vars(args), "seeds": seeds,
                   "checkpoint": best_checkpoints(args.regime).get(args.policy),
                   "elapsed_sec": round(time.time() - t_start, 1)}, f, indent=2)

    print(f"\nSaved to {out}")
    for g, fr in grids.items():
        gr = fr["grid"]
        print(f"  grid_{g:g}s {len(gr):>8d} rows x {gr.shape[1]} cols")
    for name, table in tables.items():
        print(f"  {name:<6s} {len(table):>8d} rows")


if __name__ == "__main__":
    main()
