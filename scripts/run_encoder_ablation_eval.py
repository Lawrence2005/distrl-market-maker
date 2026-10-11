"""
scripts/run_encoder_ablation_eval.py

Encoder ablation evaluation for the full grid: DQN, PPO, QR-DQN, and IQN
were each trained with 2 additional snapshot encoders (cnn, autoencoder),
alongside their existing handcrafted-snapshot result already in
holdout_eval.csv. This single script evaluates all 8 (agent, encoder)
combinations in one process — every checkpoint is loaded and rolled out
exactly once here, rather than split across a QR-DQN-only script and a
second script for the other three agents.

For each combo, this runs the same rigor pass as every other result in
this project: holdout eval on the standard 15-episode seed block
(seed+80000+i), AS-recovery on the standard AS_RECOVERY_SEED, and a
paired significance test against that SAME agent's own existing
handcrafted-normal held-out episodes (already in holdout_eval.csv) — not
against GLFT, since the question here is "does a different state
representation change this agent's policy," not "does it beat the
analytical baseline."

Usage:
    python scripts/run_encoder_ablation_eval.py --n_episodes 15
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from omegaconf import OmegaConf
from scipy.stats import ttest_rel, wilcoxon

from training.evaluate import load_agent
from training.factory import build_env
from training.rollout import run_episode as rl_run_episode, find_best_checkpoint, find_latest_checkpoint
from evaluation.as_recovery import run_recovery_all_agents, recovery_summary_df
from evaluation.ablation import _run_tag as _ablation_run_tag
from scripts.run_analysis import AGENT_CKPT_KWARGS, _MIN_EPISODES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR   = PROJECT_ROOT / "training" / "configs"
OUT_DIR      = PROJECT_ROOT / "results"

BASE_SEED        = 42
HOLDOUT_OFFSET   = 80000  # matches scripts/run_holdout_eval.py
# Was 500 — with n_episodes=500 training runs use seed+1..seed+500 (43-542),
# so seeds 500-514 (base 500 + 15 AS-recovery episodes) exactly replayed
# each agent's own training episodes 458-472, rather than unseen market
# realizations. 95000 is disjoint from training, RL eval-during-training
# (10042-10044), and the holdout block (80042+).
AS_RECOVERY_SEED = 95000
REGIME           = "normal"

AGENTS   = ["dqn", "ppo", "qrdqn", "iqn"]
ENCODERS = ["cnn", "autoencoder"]

# Capacity-matched encoder extras: latent_dim=8 for both, so the cnn-vs-AE
# comparison isn't confounded by one encoder having more representational
# capacity than the other (see encoders/cnn.yaml and encoders/autoencoder.yaml
# for the numeric evidence behind this choice).
_CNN_EXTRAS = dict(n_levels=10, latent_dim=8)
_AE_EXTRAS  = dict(latent_dim=8, ae_checkpoint=str(PROJECT_ROOT / "checkpoints" / "ae_encoder_8.pt"))
_ENCODER_EXTRAS = {"cnn": _CNN_EXTRAS, "autoencoder": _AE_EXTRAS}


def _combo_spec(agent: str, encoder: str) -> dict:
    # alpha/sampling convention matches every other script this campaign
    # uses to build run tags (run_reward_comparison_eval.py, run_analysis.py):
    # alpha applies only to the two CVaR agents, PER sampling to the three
    # replay-buffer agents (ppo has neither).
    alpha    = 0.25 if agent in ("qrdqn", "iqn") else None
    sampling = "per" if agent in ("dqn", "qrdqn", "iqn") else None
    return dict(
        agent=agent, encoder=encoder,
        run_tag=_ablation_run_tag(agent, encoder, "asymmetric", REGIME, BASE_SEED,
                                   alpha=alpha, sampling=sampling),
        load_kwargs=dict(**AGENT_CKPT_KWARGS[agent], **_ENCODER_EXTRAS[encoder]),
    )


COMBOS = {f"{agent}_{encoder}": _combo_spec(agent, encoder)
          for agent in AGENTS for encoder in ENCODERS}


def _resolve_best_checkpoint(run_tag: str, ext: str = "pt") -> Path | None:
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


def run_holdout_for_combo(name: str, spec: dict, holdout_seeds: list[int]) -> list[dict]:
    ckpt = _resolve_best_checkpoint(spec["run_tag"])
    if ckpt is None:
        print(f"  [skip] {spec['run_tag']}: no checkpoint found")
        return []

    agent, enc_type = load_agent(str(ckpt), agent_type=spec["agent"], encoder_type=spec["encoder"],
                                  **spec["load_kwargs"])
    env = build_regime_env(seed=BASE_SEED)
    rows = []
    for i, seed in enumerate(holdout_seeds):
        t0 = time.time()
        m = rl_run_episode(env, agent, enc_type, training=False, seed=seed)
        rows.append({"variant": name, "agent": spec["agent"], "encoder": spec["encoder"], "seed": seed,
                     "sharpe": m["sharpe"], "map": m["map"], "mdd": m["mdd"], "final_pnl": m["final_pnl"]})
        print(f"  {name:16s} ep {i+1:2d}/{len(holdout_seeds)} "
              f"sharpe {m['sharpe']:+.3f}  ({time.time()-t0:.0f}s)  ckpt={ckpt.name}", flush=True)
    env.close()
    return rows


def run_as_recovery_for_combo(name: str, spec: dict) -> pd.DataFrame | None:
    ckpt = _resolve_best_checkpoint(spec["run_tag"])
    if ckpt is None:
        return None
    agent, enc_type = load_agent(str(ckpt), agent_type=spec["agent"], encoder_type=spec["encoder"],
                                  **spec["load_kwargs"])
    env = build_regime_env(seed=AS_RECOVERY_SEED)
    results = run_recovery_all_agents({name: agent}, env, enc_type=enc_type,
                                       n_episodes=15, seed=AS_RECOVERY_SEED)
    env.close()
    df = recovery_summary_df(results)
    return df


def load_baseline_holdout(agent_type: str) -> pd.DataFrame:
    """This agent's existing handcrafted-normal held-out episodes, same 15 seeds."""
    holdout_path = OUT_DIR / "holdout_eval.csv"
    df = pd.read_csv(holdout_path)
    sub = df[(df["type"] == "rl") & (df["regime"] == REGIME) & (df["model"] == agent_type)].copy()
    return sub[["seed", "sharpe", "map", "mdd", "final_pnl"]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=15)
    p.add_argument("--skip", nargs="+", default=[],
                   help="Combo names to skip (e.g. --skip ppo_autoencoder), "
                        "for when a checkpoint isn't ready yet — its rows "
                        "just won't appear in the output, no placeholder.")
    args = p.parse_args()

    combos_to_run = {k: v for k, v in COMBOS.items() if k not in args.skip}
    if args.skip:
        print(f"Skipping: {', '.join(args.skip)}")

    holdout_seeds = [BASE_SEED + HOLDOUT_OFFSET + i for i in range(args.n_episodes)]
    print(f"Holdout seed block: {holdout_seeds[0]}..{holdout_seeds[-1]} "
          f"({args.n_episodes} episodes)")
    print(f"Combos ({len(combos_to_run)}): {', '.join(combos_to_run)}")

    all_rows = []
    as_recovery_frames = []

    for name, spec in combos_to_run.items():
        print(f"\n{'='*78}\nHOLDOUT EVAL — {name}\n{'='*78}")
        all_rows.extend(run_holdout_for_combo(name, spec, holdout_seeds))

    for name, spec in combos_to_run.items():
        print(f"\n{'='*78}\nAS-RECOVERY — {name}\n{'='*78}")
        df = run_as_recovery_for_combo(name, spec)
        if df is not None:
            df["variant"] = name
            df["agent"] = spec["agent"]
            df["encoder"] = spec["encoder"]
            as_recovery_frames.append(df)
            print(df.to_string(index=False))

    holdout_df = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    holdout_df.to_csv(OUT_DIR / "encoder_ablation_holdout.csv", index=False)

    if as_recovery_frames:
        pd.concat(as_recovery_frames, ignore_index=True).to_csv(
            OUT_DIR / "encoder_ablation_as_recovery.csv", index=False)

    # ── Significance test: each combo vs. its own agent's handcrafted baseline ──
    sig_rows = []
    for name, spec in combos_to_run.items():
        combo_rows = holdout_df[holdout_df["variant"] == name].set_index("seed")["sharpe"]
        if combo_rows.empty:
            continue
        baseline_df = load_baseline_holdout(spec["agent"])
        baseline_by_seed = baseline_df.set_index("seed")["sharpe"]
        shared_seeds = sorted(set(combo_rows.index) & set(baseline_by_seed.index))
        if not shared_seeds:
            print(f"  [warn] {name}: no shared seeds with handcrafted baseline — skipping sig test")
            continue
        a = combo_rows.loc[shared_seeds].to_numpy()
        b = baseline_by_seed.loc[shared_seeds].to_numpy()
        diff = a - b
        t_p = ttest_rel(a, b).pvalue
        try:
            w_p = wilcoxon(a, b).pvalue
        except ValueError:
            w_p = float("nan")
        sig_rows.append({
            "variant": name, "agent": spec["agent"], "encoder": spec["encoder"],
            "encoder_sharpe_mean": a.mean(), "handcrafted_sharpe_mean": b.mean(),
            "mean_diff_vs_handcrafted": diff.mean(),
            "t_p": t_p, "wilcoxon_p": w_p,
            "significantly_different_p05": bool(w_p < 0.05),
        })
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(OUT_DIR / "encoder_ablation_significance.csv", index=False)

    summary = (
        holdout_df.groupby(["agent", "encoder", "variant"])["sharpe"]
        .agg(sharpe_mean="mean", sharpe_std="std", n="count")
        .reset_index()
        .sort_values(["agent", "sharpe_mean"], ascending=[True, False])
    )
    summary.to_csv(OUT_DIR / "encoder_ablation_summary.csv", index=False)

    print(f"\n{'='*78}\nENCODER ABLATION SUMMARY (all 8 combos)\n{'='*78}")
    print(summary.to_string(index=False))
    print(f"\n{'='*78}\nSIGNIFICANCE VS. EACH AGENT'S OWN HANDCRAFTED BASELINE\n{'='*78}")
    print(sig_df.to_string(index=False))
    print(f"\nSaved -> {OUT_DIR / 'encoder_ablation_holdout.csv'}, "
          f"{OUT_DIR / 'encoder_ablation_as_recovery.csv'}, "
          f"{OUT_DIR / 'encoder_ablation_significance.csv'}, "
          f"{OUT_DIR / 'encoder_ablation_summary.csv'}")


if __name__ == "__main__":
    main()
