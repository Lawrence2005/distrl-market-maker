"""
scripts/run_reward_comparison_eval.py

Holdout comparison of two asymmetric reward functions, trained and evaluated
with identical settings:

  ours     : 'asymmetric'          r = PnL - eta * max(0, -inv_pnl)
             (this project's reward; envs/lob_env.py)
  spooner  : 'asymmetric_spooner'  r = PnL - eta * max(0,  inv_pnl)
             (the functional form of Spooner et al. 2018, eq. 6, implemented in
             our environment; a comparator, not a replication of their setup)

Each (agent, regime, run seed) is evaluated on the SAME 15 held-out episode
seeds used by scripts/run_holdout_eval.py (BASE_SEED + HOLDOUT_OFFSET + i),
on its best eval-Sharpe checkpoint, and compared to GLFT on those seeds.
Holdout Sharpe is computed from step P&L, so the policies are scored on the
same metric regardless of the training reward.

Outputs (new files; canonical CSVs are untouched):
  results/reward_comparison_holdout.csv   one row per episode
  results/reward_comparison_summary.csv   one row per (agent, regime, run seed, variant)
                                          with paired tests vs the other reward and vs GLFT

Usage:
    python scripts/run_reward_comparison_eval.py
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from scipy import stats

from evaluation.ablation import _run_tag
from scripts.run_analysis import AGENT_CKPT_KWARGS
from scripts.run_baseline import build_baseline, run_episode as baseline_run_episode
from scripts.run_holdout_eval import BASE_SEED, HOLDOUT_OFFSET, _load_env_cfg
from training.evaluate import load_agent
from training.factory import build_env
from training.rollout import run_episode as rl_run_episode

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR   = PROJECT_ROOT / "training" / "configs"
CKPT_DIR     = PROJECT_ROOT / "checkpoints"
LOGS_DIR     = PROJECT_ROOT / "logs"
OUT_EPISODES = PROJECT_ROOT / "results" / "reward_comparison_holdout.csv"
OUT_SUMMARY  = PROJECT_ROOT / "results" / "reward_comparison_summary.csv"

REGIMES  = ["low_vol", "normal", "high_vol"]
AGENTS   = ["sarsa", "dqn", "ppo", "qrdqn", "iqn"]
VARIANTS = {
    "ours":    {"reward_type": "asymmetric",          "reward_yaml": "asymmetric.yaml"},
    "spooner": {"reward_type": "asymmetric_spooner",  "reward_yaml": "asymmetric_spooner.yaml"},
}

# Run seeds: the main campaign (seed 42) for every agent and regime, plus
# three extra seeds for QR-DQN and IQN in the normal regime (alpha=0.25).
MAIN_SEED = 42
EXTRA_SEEDS = {("qrdqn", "normal"): [1042, 2042, 3042],
               ("iqn", "normal"):   [1042, 2042, 3042]}


def run_tag(agent: str, regime: str, reward_type: str, seed: int) -> str:
    alpha = 0.25 if agent in ("qrdqn", "iqn") else None
    sampling = "per" if agent in ("dqn", "qrdqn", "iqn") else None
    return _run_tag(agent, "handcrafted", reward_type, regime, seed,
                    alpha=alpha, sampling=sampling)


def best_checkpoint(tag: str) -> Path:
    """Best eval-Sharpe checkpoint, same rule as scripts/run_analysis.py."""
    ckpt_dir = CKPT_DIR / tag
    for name in ("best.pt", "best.npz"):
        if (ckpt_dir / name).exists():
            return ckpt_dir / name
    hist = json.loads((LOGS_DIR / tag / "eval_history.json").read_text())
    ep = max(hist, key=lambda e: e["sharpe"])["episode"]
    matches = sorted(glob.glob(str(ckpt_dir / f"ep{ep:05d}.*")))
    if not matches:
        raise FileNotFoundError(f"no checkpoint for episode {ep} in {ckpt_dir}")
    return Path(matches[0])


def build_eval_env(regime: str, reward_yaml: str):
    env_cfg = _load_env_cfg(regime)
    reward_cfg = OmegaConf.load(CONFIG_DIR / "reward" / reward_yaml)
    return build_env(env_cfg, reward_cfg, BASE_SEED)


def run_configs() -> list[tuple[str, str, int]]:
    out = []
    for agent in AGENTS:
        for regime in REGIMES:
            out.append((agent, regime, MAIN_SEED))
            for s in EXTRA_SEEDS.get((agent, regime), []):
                out.append((agent, regime, s))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=15)
    args = p.parse_args()
    seeds = [BASE_SEED + HOLDOUT_OFFSET + i for i in range(args.n_episodes)]

    rows = []

    # GLFT reference on the same holdout seeds, per regime
    for regime in REGIMES:
        env = build_eval_env(regime, "asymmetric.yaml")
        glft = build_baseline("glft", 390, 0.01, 10, regime)
        for seed in seeds:
            m = baseline_run_episode(env, glft, seed=seed)
            rows.append({"agent": "glft", "regime": regime, "run_seed": MAIN_SEED,
                         "variant": "glft", "holdout_seed": seed,
                         "sharpe": m["sharpe"], "final_pnl": m["final_pnl"],
                         "mdd": m["mdd"], "map": m.get("map", np.nan), "checkpoint": ""})
        env.close()

    for agent, regime, run_seed in run_configs():
        for variant, cfg in VARIANTS.items():
            tag = run_tag(agent, regime, cfg["reward_type"], run_seed)
            ckpt = best_checkpoint(tag)
            print(f"[{agent} {regime} seed{run_seed} {variant}] {ckpt.relative_to(PROJECT_ROOT)}",
                  flush=True)
            kwargs = AGENT_CKPT_KWARGS[agent]
            agent_obj, _ = load_agent(str(ckpt), agent_type=agent,
                                      encoder_type="handcrafted", **kwargs)
            env = build_eval_env(regime, cfg["reward_yaml"])
            for seed in seeds:
                t0 = time.time()
                m = rl_run_episode(env, agent_obj, "handcrafted", training=False, seed=seed)
                rows.append({"agent": agent, "regime": regime, "run_seed": run_seed,
                             "variant": variant, "holdout_seed": seed,
                             "sharpe": m["sharpe"], "final_pnl": m["final_pnl"],
                             "mdd": m["mdd"], "map": m.get("map", np.nan), "checkpoint": ckpt.name})
                print(f"    seed {seed}: sharpe {m['sharpe']:+.3f} ({time.time()-t0:.0f}s)",
                      flush=True)
            env.close()
            # write incrementally so a crash doesn't lose finished combinations
            pd.DataFrame(rows).to_csv(OUT_EPISODES, index=False)

    df = pd.DataFrame(rows)
    df.to_csv(OUT_EPISODES, index=False)

    summary = []
    for (agent, regime, run_seed), g in df[df.variant != "glft"].groupby(["agent", "regime", "run_seed"]):
        piv = g.pivot(index="holdout_seed", columns="variant", values="sharpe")
        glft_s = df[(df.agent == "glft") & (df.regime == regime)].set_index("holdout_seed")["sharpe"]
        for variant in VARIANTS:
            v = piv[variant]
            summary.append({
                "agent": agent, "regime": regime, "run_seed": run_seed, "variant": variant,
                "mean_sharpe": v.mean(), "std_sharpe": v.std(),
                "mean_final_pnl": g[g.variant == variant]["final_pnl"].mean(),
                "diff_vs_glft": (v - glft_s.loc[v.index]).mean(),
                "p_wilcoxon_vs_glft": stats.wilcoxon(v, glft_s.loc[v.index]).pvalue,
                "p_wilcoxon_pair": np.nan,
            })
        other = {"ours": "spooner", "spooner": "ours"}
        for variant in VARIANTS:
            d = piv[variant] - piv[other[variant]]
            summary.append({
                "agent": agent, "regime": regime, "run_seed": run_seed,
                "variant": f"{variant}_minus_{other[variant]}",
                "mean_sharpe": d.mean(), "std_sharpe": d.std(),
                "mean_final_pnl": np.nan, "diff_vs_glft": np.nan,
                "p_wilcoxon_vs_glft": np.nan,
                "p_wilcoxon_pair": stats.wilcoxon(piv[variant], piv[other[variant]]).pvalue,
            })
    pd.DataFrame(summary).to_csv(OUT_SUMMARY, index=False)
    print(f"Wrote {OUT_EPISODES.relative_to(PROJECT_ROOT)} and {OUT_SUMMARY.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
