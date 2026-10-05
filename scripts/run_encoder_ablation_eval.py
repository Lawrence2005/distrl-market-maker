"""
scripts/run_encoder_ablation_eval.py

Encoder ablation evaluation. QR-DQN was trained with 2 additional SNAPSHOT
encoders (cnn, autoencoder) at alpha=0.25, normal regime, alongside the
existing handcrafted-snapshot result already on the dashboard. This script
evaluates both on the SAME held-out seed block scripts/run_holdout_eval.py
uses (seed+80000+i), runs AS-recovery for each, and runs a paired
significance test against the EXISTING handcrafted-QRDQN-normal-alpha0.25
held-out episodes (already in holdout_eval.csv, same 15 seeds) — not against
GLFT, since the research question here is "does a different state
representation change QR-DQN's policy," not "does it beat the analytical
baseline."

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
from scripts.run_analysis import _MIN_EPISODES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR   = PROJECT_ROOT / "training" / "configs"
OUT_DIR      = PROJECT_ROOT / "evaluation" / "results_scaled_down_sweep"

BASE_SEED         = 42
HOLDOUT_OFFSET    = 80000  # matches scripts/run_holdout_eval.py
# Was 500 — with n_episodes=500 training runs use seed+1..seed+500 (43-542),
# so seeds 500-514 (base 500 + 15 AS-recovery episodes) exactly replayed
# each agent's own training episodes 458-472, rather than unseen market
# realizations. 95000 is disjoint from training (43 up to 2042 even at
# config.yaml's top-level default n_episodes=2000), RL eval-during-training
# (10042-10044), and the holdout block (80042+).
AS_RECOVERY_SEED  = 95000
REGIME            = "normal"
BASELINE_ENCODER  = "handcrafted"  # the comparison point, already in holdout_eval.csv

# (label, run_tag, load_agent kwargs beyond agent_type/n_actions)
ENCODERS = {
    "cnn": dict(
        # _per, not _uniform: this is the fresh checkpoint from the current
        # full-corpus retrain campaign (qrdqn defaults to real PER now).
        run_tag="qrdqn_cnn_asymmetric_normal_alpha0.25_per_seed42",
        encoder_type="cnn",
        # latent_dim=8, matching encoders/cnn.yaml — capacity-matched
        # against the AE encoder below (see that file's comment for why).
        load_kwargs=dict(hidden_dim=256, n_quantiles=32, n_levels=10, latent_dim=8),
    ),
    "autoencoder": dict(
        run_tag="qrdqn_autoencoder_asymmetric_normal_alpha0.25_per_seed42",
        encoder_type="autoencoder",
        # latent_dim=8: input_dim is 20, so the old latent_dim=32 was an
        # over-complete, non-bottlenecked autoencoder (see
        # encoders/autoencoder.yaml's comment for the numeric evidence).
        load_kwargs=dict(hidden_dim=256, n_quantiles=32, latent_dim=8,
                          ae_checkpoint=str(PROJECT_ROOT / "checkpoints" / "ae_encoder_8.pt")),
    ),
    # "recurrent" used to live here — moved to scripts/run_recurrent_variant_eval.py
    # (see this file's module docstring for why).
}


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


def run_holdout_for_encoder(name: str, spec: dict, holdout_seeds: list[int]) -> list[dict]:
    ckpt = _resolve_best_checkpoint(spec["run_tag"])
    if ckpt is None:
        print(f"  [skip] {spec['run_tag']}: no checkpoint found")
        return []

    agent, enc_type = load_agent(str(ckpt), agent_type="qrdqn", encoder_type=spec["encoder_type"],
                                  **spec["load_kwargs"])
    env = build_regime_env(seed=BASE_SEED)
    rows = []
    for i, seed in enumerate(holdout_seeds):
        t0 = time.time()
        m = rl_run_episode(env, agent, enc_type, training=False, seed=seed)
        rows.append({"encoder": name, "seed": seed, "sharpe": m["sharpe"],
                     "map": m["map"], "mdd": m["mdd"], "final_pnl": m["final_pnl"]})
        print(f"  {name:12s} ep {i+1:2d}/{len(holdout_seeds)} "
              f"sharpe {m['sharpe']:+.3f}  ({time.time()-t0:.0f}s)  ckpt={ckpt.name}", flush=True)
    env.close()
    return rows


def run_as_recovery_for_encoder(name: str, spec: dict) -> pd.DataFrame | None:
    ckpt = _resolve_best_checkpoint(spec["run_tag"])
    if ckpt is None:
        return None
    agent, enc_type = load_agent(str(ckpt), agent_type="qrdqn", encoder_type=spec["encoder_type"],
                                  **spec["load_kwargs"])
    env = build_regime_env(seed=AS_RECOVERY_SEED)
    results = run_recovery_all_agents({name: agent}, env, enc_type=enc_type,
                                       n_episodes=15, seed=AS_RECOVERY_SEED)
    env.close()
    df = recovery_summary_df(results)
    return df


def load_baseline_holdout() -> pd.DataFrame:
    """Existing handcrafted-QRDQN-normal-alpha0.25 held-out episodes, same 15 seeds."""
    holdout_path = OUT_DIR / "holdout_eval.csv"
    df = pd.read_csv(holdout_path)
    sub = df[(df["type"] == "rl") & (df["regime"] == REGIME) & (df["model"] == "qrdqn")].copy()
    sub["encoder"] = BASELINE_ENCODER
    return sub[["encoder", "seed", "sharpe", "map", "mdd", "final_pnl"]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=15)
    p.add_argument("--skip", nargs="+", default=[],
                   help="Encoder names to skip (e.g. --skip autoencoder), "
                        "for when a checkpoint isn't ready yet — its rows "
                        "just won't appear in the output, no placeholder.")
    args = p.parse_args()

    encoders_to_run = {k: v for k, v in ENCODERS.items() if k not in args.skip}
    if args.skip:
        print(f"Skipping: {', '.join(args.skip)}")

    holdout_seeds = [BASE_SEED + HOLDOUT_OFFSET + i for i in range(args.n_episodes)]
    print(f"Holdout seed block: {holdout_seeds[0]}..{holdout_seeds[-1]} "
          f"({args.n_episodes} episodes)")

    baseline_df = load_baseline_holdout()
    all_rows = list(baseline_df.to_dict("records"))
    as_recovery_frames = []

    for name, spec in encoders_to_run.items():
        print(f"\n{'='*78}\nHOLDOUT EVAL — QR-DQN + {name} encoder\n{'='*78}")
        all_rows.extend(run_holdout_for_encoder(name, spec, holdout_seeds))

    for name, spec in encoders_to_run.items():
        print(f"\n{'='*78}\nAS-RECOVERY — QR-DQN + {name} encoder\n{'='*78}")
        df = run_as_recovery_for_encoder(name, spec)
        if df is not None:
            df["encoder"] = name
            as_recovery_frames.append(df)
            print(df.to_string(index=False))

    holdout_df = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    holdout_df.to_csv(OUT_DIR / "encoder_ablation_holdout.csv", index=False)

    if as_recovery_frames:
        pd.concat(as_recovery_frames, ignore_index=True).to_csv(
            OUT_DIR / "encoder_ablation_as_recovery.csv", index=False)

    # ── Significance test: each new encoder vs. handcrafted, paired on seed ──
    sig_rows = []
    baseline_by_seed = baseline_df.set_index("seed")["sharpe"]
    for name in encoders_to_run:
        enc_rows = holdout_df[holdout_df["encoder"] == name].set_index("seed")["sharpe"]
        if enc_rows.empty:
            continue
        shared_seeds = sorted(set(enc_rows.index) & set(baseline_by_seed.index))
        a = enc_rows.loc[shared_seeds].to_numpy()
        b = baseline_by_seed.loc[shared_seeds].to_numpy()
        diff = a - b
        t_p = ttest_rel(a, b).pvalue
        try:
            w_p = wilcoxon(a, b).pvalue
        except ValueError:
            w_p = float("nan")
        sig_rows.append({
            "encoder": name,
            "mean_diff_vs_handcrafted": diff.mean(),
            "t_p": t_p,
            "wilcoxon_p": w_p,
            "significantly_different_p05": bool(w_p < 0.05),
        })
    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(OUT_DIR / "encoder_ablation_significance.csv", index=False)

    summary = (
        holdout_df.groupby("encoder")["sharpe"]
        .agg(sharpe_mean="mean", sharpe_std="std", n="count")
        .reset_index()
        .sort_values("sharpe_mean", ascending=False)
    )
    summary.to_csv(OUT_DIR / "encoder_ablation_summary.csv", index=False)

    print(f"\n{'='*78}\nENCODER ABLATION SUMMARY\n{'='*78}")
    print(summary.to_string(index=False))
    print(f"\n{'='*78}\nSIGNIFICANCE VS. HANDCRAFTED\n{'='*78}")
    print(sig_df.to_string(index=False))
    print(f"\nSaved → {OUT_DIR / 'encoder_ablation_holdout.csv'}, "
          f"{OUT_DIR / 'encoder_ablation_as_recovery.csv'}, "
          f"{OUT_DIR / 'encoder_ablation_significance.csv'}, "
          f"{OUT_DIR / 'encoder_ablation_summary.csv'}")


if __name__ == "__main__":
    main()
