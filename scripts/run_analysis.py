"""
scripts/run_analysis.py

Phase 3 analysis for the scaled-down local sweep (see the approved plan
"Scaled-down local training sweep"): AS-recovery R^2 for the low_vol
agents and a unified RL-vs-baseline comparison table.

(flash_crash and trending regimes, and the flash-crash QR-DQN(alpha=0.05)
vs DQN drawdown comparison this file used to run, were removed from the
project — see git history if reviving either.)

Reuses existing evaluation utilities rather than recomputing metrics:
training.evaluate.load_agent, evaluation.as_recovery.run_recovery_all_agents
/ recovery_summary_df, evaluation.metrics.summary_table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from omegaconf import OmegaConf

from training.evaluate import load_agent
from training.factory import build_env
from training.rollout import find_best_checkpoint, find_latest_checkpoint
from evaluation.as_recovery import run_recovery_all_agents, recovery_summary_df
from evaluation.metrics import summary_table
from evaluation.ablation import _run_tag as _ablation_run_tag

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR   = PROJECT_ROOT / "training" / "configs"

# Reconstruction kwargs matching each agent's local-scale config (see
# training/configs/agent/*.yaml) — must match what each checkpoint was
# actually trained with, not load_agent's HPC-scale defaults.
AGENT_CKPT_KWARGS = {
    "sarsa": dict(n_tilings=16, n_tiles=8, memory_size=16384),
    "dqn":   dict(hidden_dim=256),
    "ppo":   dict(hidden_dim=128),
    "qrdqn": dict(hidden_dim=256, n_quantiles=32),
    "iqn":   dict(hidden_dim=256, n_quantile_samples=16, embedding_dim=32),
}

# Matches training.configs.config.yaml's early_stopping_min_episodes default
# — the epsilon-annealing burn-in floor, reused here so "best checkpoint"
# selection can't pick a lucky pre-convergence eval draw (see
# _resolve_best_checkpoint).
_MIN_EPISODES = 150


def _run_tag(agent_type: str, regime: str) -> str:
    # Delegates to evaluation.ablation's run_tag builder (single source of
    # truth matching training/train.py's convention: alpha=0.25 for
    # qrdqn/iqn in this sweep). sampling="per" explicitly — this campaign's
    # dqn/qrdqn/iqn.yaml all set prioritized_replay=true and no uniform
    # variant was trained (that stale "uniform" default here — inherited
    # from _ablation_run_tag's own default — silently pointed every
    # dqn/qrdqn/iqn lookup at a checkpoint dir that doesn't exist,
    # returning None from _resolve_best_checkpoint for all three agents
    # across every consumer: run_analysis.py's AS-recovery/model-comparison
    # and run_ood_transfer.py's best_checkpoints()).
    return _ablation_run_tag(agent_type, "handcrafted", "asymmetric", regime, 42,
                              sampling="per")


def _checkpoint_dir(agent_type: str, regime: str) -> Path:
    return PROJECT_ROOT / "checkpoints" / _run_tag(agent_type, regime)


def _resolve_best_checkpoint(agent_type: str, regime: str) -> Path | None:
    """
    Prefer the agent's best-ever eval checkpoint over its latest one:
    patience-based early stopping lets training run `patience` checkpoints
    past a peak before stopping, so the latest/final checkpoint can be a
    materially degraded policy relative to what the agent actually reached
    (see the "peak vs. final" drift analysis on the dashboard —
    e.g. qrdqn/normal peaked at eval Sharpe +4.05 and ended its last 3
    checkpoints averaging -0.18).

    Resolution order:
    1. best.pt/best.npz in the checkpoint dir (runs trained after train.py
       started tracking this explicitly).
    2. For runs trained before that: read eval_history.json,
       find the episode with the highest eval Sharpe, and use the periodic
       ep-numbered checkpoint already saved at that episode (ckpt_every ==
       eval_every == 25 for this sweep, so one exists for every eval point).
    3. Fall back to the latest checkpoint, with a warning — better than
       crashing if a run has no eval history at all (e.g. it never reached
       one eval checkpoint).
    """
    ckpt_dir = _checkpoint_dir(agent_type, regime)

    best = find_best_checkpoint(ckpt_dir)
    if best is not None:
        return best

    eval_hist_path = PROJECT_ROOT / "logs" / _run_tag(agent_type, regime) / "eval_history.json"
    if eval_hist_path.exists():
        with open(eval_hist_path) as f:
            eval_history = json.load(f)
        # Same burn-in floor train.py's best-checkpoint tracking applies —
        # epsilon hasn't finished annealing before ~episode 128 for the DQN
        # family (epsilon_decay_steps=50000), so an eval checkpoint before
        # _MIN_EPISODES risks being a lucky high-variance draw under a
        # still-mostly-random policy, not genuine convergence.
        eligible = [e for e in eval_history if e["episode"] >= _MIN_EPISODES]
        if eligible:
            best_ep = max(eligible, key=lambda e: e["sharpe"])["episode"]
            ext = "npz" if agent_type == "sarsa" else "pt"
            candidate = ckpt_dir / f"ep{best_ep:05d}.{ext}"
            if candidate.exists():
                return candidate
            print(f"  [warn] {agent_type}/{regime}: best eval episode {best_ep} has no "
                  f"matching checkpoint file ({candidate.name}) — falling back to latest")

    latest = find_latest_checkpoint(ckpt_dir)
    if latest is not None:
        print(f"  [warn] {agent_type}/{regime}: using latest checkpoint, not best "
              f"(no eval_history.json / best.pt found)")
    return latest


def best_checkpoints(regime: str) -> dict[str, str]:
    """
    Resolve each agent's best-eval checkpoint for `regime` dynamically —
    see _resolve_best_checkpoint for why "best" rather than "latest".
    """
    ckpts = {}
    for agent_type in AGENT_CKPT_KWARGS:
        resolved = _resolve_best_checkpoint(agent_type, regime)
        if resolved is not None:
            ckpts[agent_type] = str(resolved)
    return ckpts


def build_regime_env(regime: str, seed: int):
    env_cfg = OmegaConf.load(CONFIG_DIR / "env" / "base.yaml")
    OmegaConf.set_struct(env_cfg, False)
    env_cfg.regime     = regime
    env_cfg.use_abides = True
    reward_cfg = OmegaConf.load(CONFIG_DIR / "reward" / "asymmetric.yaml")
    return build_env(env_cfg, reward_cfg, seed)


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ══════════════════════════════════════════════════════════════════════════
# 1. AS-recovery test
# ══════════════════════════════════════════════════════════════════════════

def run_as_recovery(regime: str) -> pd.DataFrame:
    section(f"AS-RECOVERY TEST — {regime} regime")

    ckpts = best_checkpoints(regime)
    agents = {}
    for agent_type, ckpt in ckpts.items():
        agent, _ = load_agent(
            ckpt, agent_type=agent_type, encoder_type="handcrafted",
            **AGENT_CKPT_KWARGS[agent_type],
        )
        agents[agent_type] = agent
        print(f"  {agent_type}: {ckpt}")

    # AS_RECOVERY_SEED=95000: was 500, which for any n_episodes=500 training
    # run overlaps training seeds 43-542 (base 500 + 15 episodes = 500-514,
    # replaying training episodes 458-472 instead of unseen data). 95000 is
    # disjoint from training, RL-eval-during-training (10042-10044), and the
    # holdout block (80042+) — matches the fix applied to
    # scripts/run_encoder_ablation_eval.py and run_cvar_sweep_eval.py.
    AS_RECOVERY_SEED = 95000
    env = build_regime_env(regime, seed=AS_RECOVERY_SEED)
    results = run_recovery_all_agents(agents, env, enc_type="handcrafted",
                                       n_episodes=15, seed=AS_RECOVERY_SEED)
    env.close()

    df = recovery_summary_df(results)
    df["regime"] = regime
    print(df.to_string(index=False))
    return df


# ══════════════════════════════════════════════════════════════════════════
# 2. Model comparison table (RL agents + baselines, both regimes)
# ══════════════════════════════════════════════════════════════════════════

def run_model_comparison() -> pd.DataFrame:
    section("MODEL COMPARISON TABLE — RL agents vs. baselines")

    df = summary_table(PROJECT_ROOT / "logs", metric="sharpe", episodes_window=1000000)
    # Keep this sweep's runs only (5 RL agents + 3 baselines, low_vol + normal).
    # Excludes a stale pre-existing synthetic-smoke-test run_tag that collides
    # with our naming pattern minus the alpha tag (2-episode dummy data, not
    # part of this sweep): qrdqn_handcrafted_asymmetric_flash_crash_seed42.
    keep_agents = {"sarsa", "dqn", "ppo", "qrdqn", "iqn", "fixedspread", "as", "glft"}
    df = df[df["agent"].isin(keep_agents) & df["regime"].isin(["low", "normal", "high"])]
    df = df[df["run_tag"] != "qrdqn_handcrafted_asymmetric_flash_crash_seed42"]
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    as_recovery_low_vol_df  = run_as_recovery("low_vol")
    as_recovery_normal_df   = run_as_recovery("normal")
    as_recovery_high_vol_df = run_as_recovery("high_vol")
    as_recovery_df = pd.concat(
        [as_recovery_low_vol_df, as_recovery_normal_df, as_recovery_high_vol_df],
        ignore_index=True,
    )
    comparison_df  = run_model_comparison()

    out_dir = PROJECT_ROOT / "evaluation" / "results_scaled_down_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    as_recovery_df.to_csv(out_dir / "as_recovery.csv", index=False)
    comparison_df.to_csv(out_dir / "model_comparison.csv", index=False)
    print(f"\nSaved CSVs to {out_dir}")
