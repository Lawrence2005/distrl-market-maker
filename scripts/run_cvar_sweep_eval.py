"""
scripts/run_cvar_sweep_eval.py

CVaR alpha sweep evaluation, multi-seed. QR-DQN and IQN were each trained
at alpha in {0.05, 0.10, 0.25, 0.50, 1.0} x seed in {42, 1042, 2042, 3042}
(the last three added specifically to give the efficient frontier genuine
seed variance instead of resting each alpha's point on a single training
run) — this script evaluates all 40 (agent, alpha, seed) checkpoints on
the SAME held-out seed block scripts/run_holdout_eval.py uses
(seed+80000+i), runs AS-recovery for each, and builds the CVaR efficient
frontier (mean held-out P&L vs. CVaR_0.10 of held-out per-episode PnLs)
directly from that held-out data — not from in-sample train/eval history,
unlike evaluation/efficient_frontier.py's EfficientFrontier class (which
also guesses a run_tag format that doesn't match this project's actual
f"_alpha{alpha:.2f}" convention from training/train.py, e.g. alpha=1.0 ->
"alpha1.00" not "alpha1").

This supersedes the single-seed version of this script (which only ever
evaluated seed=42 and pulled alpha=0.25 from holdout_eval.csv instead of
re-running it) — every (agent, alpha, seed) combination is now evaluated
fresh here for a consistent, self-contained set of output CSVs, and the
frontier reports both per-seed points (for error bars / spread) and the
across-seed mean +/- std per alpha (the actual "genuine variance" figure).

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
OUT_DIR      = PROJECT_ROOT / "results"

# Far outside training (seed+1..seed+500), RL eval-during-training
# (seed+10000+{0,1,2}), and AS-recovery's own seed block below — matches
# scripts/run_holdout_eval.py's holdout block exactly so RL and baseline
# numbers stay comparable across every script in this evaluation suite.
HOLDOUT_OFFSET   = 80000
# Was 500 — with n_episodes=500 training runs use seed+1..seed+500
# (43-542), so seeds 500-514 replayed each agent's own training episodes
# 458-472 instead of unseen data (same bug found and fixed the same way
# in scripts/run_encoder_ablation_eval.py). 95000 is disjoint from
# training, RL-eval-during-training, and the holdout block above.
AS_RECOVERY_SEED = 95000

AGENTS      = ["qrdqn", "iqn"]
ALL_ALPHAS  = [0.05, 0.10, 0.25, 0.50, 1.0]
SEEDS       = [42, 1042, 2042, 3042]
REGIME      = "normal"


def _run_tag(agent_type: str, alpha: float, seed: int) -> str:
    # Delegates to evaluation.ablation's run_tag builder (single source of
    # truth matching training/train.py's convention) — this used to
    # hand-build the tag missing sampling_tag, silently breaking every
    # lookup here the moment that tag was introduced. sampling="per"
    # explicitly: qrdqn/iqn.yaml set prioritized_replay=true for every
    # alpha/seed in this sweep, so the "_uniform" tag _ablation_run_tag
    # defaults to points at a checkpoint/log dir that doesn't exist (same
    # bug found and fixed in scripts/run_analysis.py and
    # scripts/build_dashboard_data.py).
    return _ablation_run_tag(agent_type, "handcrafted", "asymmetric", REGIME, seed,
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


def run_holdout_for_combo(agent_type: str, alpha: float, train_seed: int,
                           holdout_seeds: list[int]) -> list[dict]:
    run_tag = _run_tag(agent_type, alpha, train_seed)
    ckpt = _resolve_best_checkpoint(run_tag, ext="pt")
    if ckpt is None:
        print(f"  [skip] {run_tag}: no checkpoint found")
        return []

    agent, _ = load_agent(str(ckpt), agent_type=agent_type, encoder_type="handcrafted",
                           **AGENT_CKPT_KWARGS[agent_type])
    env = build_regime_env(seed=train_seed)
    rows = []
    for i, seed in enumerate(holdout_seeds):
        t0 = time.time()
        m = rl_run_episode(env, agent, "handcrafted", training=False, seed=seed)
        rows.append({"agent": agent_type, "alpha": alpha, "train_seed": train_seed,
                     "holdout_seed": seed, "sharpe": m["sharpe"], "map": m["map"],
                     "mdd": m["mdd"], "final_pnl": m["final_pnl"]})
        print(f"  {agent_type} alpha={alpha:.2f} seed={train_seed} "
              f"ep {i+1:2d}/{len(holdout_seeds)} sharpe {m['sharpe']:+.3f}  "
              f"({time.time()-t0:.0f}s)  ckpt={ckpt.name}", flush=True)
    env.close()
    return rows


def run_as_recovery_for_combo(agent_type: str, alpha: float, train_seed: int) -> pd.DataFrame | None:
    run_tag = _run_tag(agent_type, alpha, train_seed)
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
    df["train_seed"] = train_seed
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=15)
    p.add_argument("--agents", type=str, default=None,
                    help="comma-separated subset of {qrdqn,iqn} to run "
                         "(default: both) — lets two invocations run in "
                         "parallel, one per agent, to halve wall time; "
                         "output CSVs get an --out_suffix to avoid clobbering "
                         "each other and are merged afterward.")
    p.add_argument("--out_suffix", type=str, default="",
                    help="appended to output CSV filenames, e.g. '_qrdqn'.")
    args = p.parse_args()

    agents = args.agents.split(",") if args.agents else AGENTS
    for a in agents:
        assert a in AGENTS, f"unknown agent {a!r}, expected one of {AGENTS}"

    # BASE_SEED is fixed at 42 here (not each combo's train_seed) because
    # every (agent, alpha, train_seed) checkpoint is meant to be evaluated on
    # the exact same held-out episode set as every other holdout eval in this
    # project (scripts/run_holdout_eval.py, scripts/run_recurrent_variant_eval.py)
    # — this was dropped in an earlier rewrite, producing seeds 80000-80014
    # instead of 80042-80056 and silently breaking cross-file comparability.
    BASE_SEED = 42
    holdout_seeds = [BASE_SEED + HOLDOUT_OFFSET + i for i in range(args.n_episodes)]
    print(f"Holdout seed block: {holdout_seeds[0]}..{holdout_seeds[-1]} "
          f"({args.n_episodes} episodes) x {len(SEEDS)} train seeds x "
          f"{len(ALL_ALPHAS)} alphas x {len(agents)} agent(s) = "
          f"{args.n_episodes * len(SEEDS) * len(ALL_ALPHAS) * len(agents)} holdout rollouts")

    all_rows = []
    as_recovery_frames = []

    for agent_type in agents:
        for alpha in ALL_ALPHAS:
            for train_seed in SEEDS:
                print(f"\n{'='*78}\nHOLDOUT EVAL — {agent_type} alpha={alpha:.2f} "
                      f"seed={train_seed}\n{'='*78}")
                all_rows.extend(run_holdout_for_combo(agent_type, alpha, train_seed, holdout_seeds))

    for agent_type in agents:
        for alpha in ALL_ALPHAS:
            for train_seed in SEEDS:
                print(f"\n{'='*78}\nAS-RECOVERY — {agent_type} alpha={alpha:.2f} "
                      f"seed={train_seed}\n{'='*78}")
                df = run_as_recovery_for_combo(agent_type, alpha, train_seed)
                if df is not None:
                    as_recovery_frames.append(df)
                    print(df.to_string(index=False))

    holdout_df = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    holdout_df.to_csv(OUT_DIR / f"cvar_sweep_holdout{args.out_suffix}.csv", index=False)

    if as_recovery_frames:
        as_recovery_df = pd.concat(as_recovery_frames, ignore_index=True)
        as_recovery_df.to_csv(OUT_DIR / f"cvar_sweep_as_recovery{args.out_suffix}.csv", index=False)

    if args.out_suffix:
        print(f"\nRan with --out_suffix={args.out_suffix!r} (partial agent subset "
              f"{agents}) — skipping frontier computation here; merge the "
              f"per-agent CSVs and build the frontier from the combined file.")
        return

    # ── Efficient frontier ──────────────────────────────────────────────
    # Per-seed points (for a scatter / error-bar view) ...
    per_seed_rows = []
    for agent_type in agents:
        for alpha in ALL_ALPHAS:
            for train_seed in SEEDS:
                sub = holdout_df[(holdout_df["agent"] == agent_type) &
                                  (holdout_df["alpha"] == alpha) &
                                  (holdout_df["train_seed"] == train_seed)]
                if sub.empty:
                    continue
                pnls = np.sort(sub["final_pnl"].to_numpy())
                n_tail = max(1, int(0.10 * len(pnls)))
                per_seed_rows.append({
                    "agent": agent_type, "alpha": alpha, "train_seed": train_seed,
                    "mean_pnl": float(pnls.mean()), "cvar_10": float(pnls[:n_tail].mean()),
                    "pnl_std": float(pnls.std()), "mean_sharpe": float(sub["sharpe"].mean()),
                    "n_episodes": len(pnls),
                })
    per_seed_df = pd.DataFrame(per_seed_rows).sort_values(["agent", "alpha", "train_seed"])
    per_seed_df.to_csv(OUT_DIR / "cvar_efficient_frontier_per_seed.csv", index=False)

    # ... and the across-seed aggregate (the actual "genuine 4-seed
    # variance" frontier point per agent/alpha: mean +/- std of each
    # per-seed statistic, n=4 seeds).
    frontier_rows = []
    for agent_type in AGENTS:
        for alpha in ALL_ALPHAS:
            sub = per_seed_df[(per_seed_df["agent"] == agent_type) & (per_seed_df["alpha"] == alpha)]
            if sub.empty:
                continue
            frontier_rows.append({
                "agent": agent_type, "alpha": alpha,
                "mean_pnl": float(sub["mean_pnl"].mean()),
                "mean_pnl_std": float(sub["mean_pnl"].std()),
                "cvar_10": float(sub["cvar_10"].mean()),
                "cvar_10_std": float(sub["cvar_10"].std()),
                "mean_sharpe": float(sub["mean_sharpe"].mean()),
                "mean_sharpe_std": float(sub["mean_sharpe"].std()),
                "n_seeds": len(sub),
            })
    frontier_df = pd.DataFrame(frontier_rows).sort_values(["agent", "alpha"])
    frontier_df.to_csv(OUT_DIR / "cvar_efficient_frontier.csv", index=False)

    print(f"\n{'='*78}\nCVAR EFFICIENT FRONTIER (mean +/- std across {len(SEEDS)} seeds)\n{'='*78}")
    print(frontier_df.to_string(index=False))
    print(f"\nSaved -> {OUT_DIR / 'cvar_sweep_holdout.csv'}, "
          f"{OUT_DIR / 'cvar_sweep_as_recovery.csv'}, "
          f"{OUT_DIR / 'cvar_efficient_frontier_per_seed.csv'}, "
          f"{OUT_DIR / 'cvar_efficient_frontier.csv'}")


if __name__ == "__main__":
    main()
