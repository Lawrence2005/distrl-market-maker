"""
scripts/collect_abides_snapshots.py

Collects LOB depth snapshots from the actual ABIDES rmsc04 simulator for
AE pretraining, replacing the original real-market (LOBSTER/crypto) data
source used by data/process_lobster.py.

The AE encoder is pretrained offline, then frozen and used at RL
training time to encode snapshots produced by the ABIDES simulator.

Uses a uniformly random policy for exploration diversity — no dependency
on a trained agent's policy, matching envs/stylized_facts.py's own
random-policy convention for simulator-characterization rollouts. Reuses
training/rollout.py's get_encoder_input() so the collected snapshots are
byte-for-byte the same per-side-normalized, thin-book-zero-padded
[ask_sizes(10), bid_sizes(10)] vectors that get_encoder_input() will hand
the frozen AE at actual RL-training time. get_encoder_input() divides
safely (see that function's docstring), so a transient moment where a
whole side of the book is empty just yields zeros for that side rather
than needing to be dropped as un-normalizable.

Usage:
    python scripts/collect_abides_snapshots.py --n_episodes 200 --regime normal
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from omegaconf import OmegaConf

from training.factory import build_env
from training.rollout import get_encoder_input, decode_action
from envs.lob_env import N_OFFSET_LEVELS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR   = PROJECT_ROOT / "training" / "configs"


def build_regime_env(regime: str, seed: int):
    env_cfg = OmegaConf.load(CONFIG_DIR / "env" / "base.yaml")
    OmegaConf.set_struct(env_cfg, False)
    env_cfg.regime     = regime
    env_cfg.use_abides = True
    reward_cfg = OmegaConf.load(CONFIG_DIR / "reward" / "asymmetric.yaml")
    return build_env(env_cfg, reward_cfg, seed)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=200)
    p.add_argument("--regime", type=str, default="normal",
                   help="Matches the regime the AE-encoder RL run actually "
                        "trains in (see scripts/jobs/qrdqn_autoencoder_normal_seed42.args)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="data/processed/lob_snapshots_abides.npy")
    args = p.parse_args()

    n_actions = N_OFFSET_LEVELS ** 2
    rng = np.random.default_rng(args.seed)

    env = build_regime_env(args.regime, args.seed)
    all_snapshots: list[np.ndarray] = []

    for ep in range(args.n_episodes):
        t0 = time.time()
        obs, info = env.reset(seed=args.seed + ep)
        terminated = truncated = False
        ep_count = 0
        while not (terminated or truncated):
            action = int(rng.integers(n_actions))
            next_obs, reward, terminated, truncated, next_info = env.step(decode_action(action))
            snap = get_encoder_input(next_obs, next_info, "autoencoder")
            all_snapshots.append(snap)
            ep_count += 1
            obs, info = next_obs, next_info
        print(f"  ep {ep + 1:3d}/{args.n_episodes}  steps={ep_count:4d}  "
              f"total_snapshots={len(all_snapshots):6d}  ({time.time() - t0:.0f}s)",
              flush=True)

    env.close()

    arr = np.stack(all_snapshots).astype(np.float32)

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr)
    print(f"\nSaved {len(arr)} ABIDES-collected LOB snapshots -> {out_path}")
    print(f"  Snapshot shape: {arr.shape}  (each row = {arr.shape[1]}-dim depth profile, "
          f"[ask_sizes(10), bid_sizes(10)], per-side normalized proportions)")


if __name__ == "__main__":
    main()
