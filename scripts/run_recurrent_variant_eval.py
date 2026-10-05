"""
scripts/run_recurrent_variant_eval.py

Rigor-pass evaluation for every recurrent-variant checkpoint this project has
trained, across every agent/regime combination — the single source of truth
for "does the recurrent (LSTM-backbone) variant beat this agent's own
non-recurrent snapshot baseline, in this regime," used to close
research-question gaps Q4 ("does recurrent integration help consistently
across agents") and Q6 ("does temporal memory help most in high-vol
regimes").

qrdqn_normal is the original recurrent checkpoint (predates this project's
item-2/3 gap-closing work); the other 5 were trained specifically to extend
recurrent coverage beyond QR-DQN/normal. All 6 are evaluated identically
here rather than splitting qrdqn_normal's own comparison across a
differently-shaped script (scripts/run_encoder_ablation_eval.py used to
carry it as a 3rd "encoder" entry, which was never really accurate — the
recurrent variant is a different backbone architecture, not a swappable
encoder — see that file's own docstring for the same note) — a single
codebase-wide invariant: any recurrent-vs-snapshot comparison for any
agent/regime combination lives here, run once, reproducible in one pass.

Generalizes across the (agent_type, regime) combinations below, each
compared against the SAME agent's already-evaluated non-recurrent snapshot
result in holdout_eval.csv, in the SAME regime — the right comparison for
"does recurrent help this agent in this regime," as opposed to a cross-agent
or cross-regime comparison.

Usage:
    python scripts/run_recurrent_variant_eval.py --n_episodes 15
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
from scripts.run_analysis import _MIN_EPISODES, AGENT_CKPT_KWARGS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR   = PROJECT_ROOT / "training" / "configs"
OUT_DIR      = PROJECT_ROOT / "results"

BASE_SEED        = 42
HOLDOUT_OFFSET   = 80000  # matches scripts/run_holdout_eval.py — same block,
                          # so results are directly comparable to the
                          # existing non-recurrent holdout_eval.csv rows.
AS_RECOVERY_SEED = 95000  # matches scripts/run_encoder_ablation_eval.py

# (label, agent_type, regime, run_tag, load_agent kwargs beyond encoder_type)
# load_kwargs mirrors AGENT_CKPT_KWARGS per agent_type plus use_lstm=True —
# the recurrent variant's backbone, not a different encoder.
RECURRENT_VARIANTS = {
    "qrdqn_normal": dict(
        agent_type="qrdqn", regime="normal",
        run_tag="qrdqn_handcrafted_asymmetric_normal_recurrent_alpha0.25_per_seed42",
        load_kwargs=dict(**AGENT_CKPT_KWARGS["qrdqn"], use_lstm=True),
    ),
    "qrdqn_low_vol": dict(
        agent_type="qrdqn", regime="low_vol",
        run_tag="qrdqn_handcrafted_asymmetric_low_vol_recurrent_alpha0.25_per_seed42",
        load_kwargs=dict(**AGENT_CKPT_KWARGS["qrdqn"], use_lstm=True),
    ),
    "qrdqn_high_vol": dict(
        agent_type="qrdqn", regime="high_vol",
        run_tag="qrdqn_handcrafted_asymmetric_high_vol_recurrent_alpha0.25_per_seed42",
        load_kwargs=dict(**AGENT_CKPT_KWARGS["qrdqn"], use_lstm=True),
    ),
    "dqn_normal": dict(
        agent_type="dqn", regime="normal",
        run_tag="dqn_handcrafted_asymmetric_normal_recurrent_per_seed42",
        load_kwargs=dict(**AGENT_CKPT_KWARGS["dqn"], use_lstm=True),
    ),
    "iqn_normal": dict(
        agent_type="iqn", regime="normal",
        run_tag="iqn_handcrafted_asymmetric_normal_recurrent_alpha0.25_per_seed42",
        load_kwargs=dict(**AGENT_CKPT_KWARGS["iqn"], use_lstm=True),
    ),
    "ppo_normal": dict(
        agent_type="ppo", regime="normal",
        run_tag="ppo_handcrafted_asymmetric_normal_recurrent_seed42",
        load_kwargs=dict(**AGENT_CKPT_KWARGS["ppo"], use_lstm=True),
    ),
    # Recurrent QR-DQN x CVaR-alpha interaction (Q4's "does recurrent compound
    # with CVaR" sub-clause) — alpha=0.25 already exists above as qrdqn_normal;
    # these 4 fill in the rest of the alpha sweep for the recurrent variant.
    # cvar_alpha marks these for comparison against cvar_sweep_holdout.csv
    # (the alpha-matched non-recurrent baseline, seed=42) rather than
    # holdout_eval.csv's single alpha=0.25 snapshot row — see
    # load_cvar_baseline_holdout below.
    "qrdqn_normal_alpha0.05": dict(
        agent_type="qrdqn", regime="normal", cvar_alpha=0.05,
        run_tag="qrdqn_handcrafted_asymmetric_normal_recurrent_alpha0.05_per_seed42",
        load_kwargs=dict(**AGENT_CKPT_KWARGS["qrdqn"], use_lstm=True),
    ),
    "qrdqn_normal_alpha0.10": dict(
        agent_type="qrdqn", regime="normal", cvar_alpha=0.10,
        run_tag="qrdqn_handcrafted_asymmetric_normal_recurrent_alpha0.10_per_seed42",
        load_kwargs=dict(**AGENT_CKPT_KWARGS["qrdqn"], use_lstm=True),
    ),
    "qrdqn_normal_alpha0.50": dict(
        agent_type="qrdqn", regime="normal", cvar_alpha=0.50,
        run_tag="qrdqn_handcrafted_asymmetric_normal_recurrent_alpha0.50_per_seed42",
        load_kwargs=dict(**AGENT_CKPT_KWARGS["qrdqn"], use_lstm=True),
    ),
    "qrdqn_normal_alpha1.00": dict(
        agent_type="qrdqn", regime="normal", cvar_alpha=1.0,
        run_tag="qrdqn_handcrafted_asymmetric_normal_recurrent_alpha1.00_per_seed42",
        load_kwargs=dict(**AGENT_CKPT_KWARGS["qrdqn"], use_lstm=True),
    ),
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


def build_regime_env(regime: str, seed: int):
    env_cfg = OmegaConf.load(CONFIG_DIR / "env" / "base.yaml")
    OmegaConf.set_struct(env_cfg, False)
    env_cfg.regime     = regime
    env_cfg.use_abides = True
    reward_cfg = OmegaConf.load(CONFIG_DIR / "reward" / "asymmetric.yaml")
    return build_env(env_cfg, reward_cfg, seed)


def run_holdout_for_variant(name: str, spec: dict, holdout_seeds: list[int]) -> list[dict]:
    ckpt = _resolve_best_checkpoint(spec["run_tag"])
    if ckpt is None:
        print(f"  [skip] {spec['run_tag']}: no checkpoint found")
        return []

    agent, enc_type = load_agent(str(ckpt), agent_type=spec["agent_type"],
                                  encoder_type="handcrafted", **spec["load_kwargs"])
    env = build_regime_env(spec["regime"], seed=BASE_SEED)
    rows = []
    for i, seed in enumerate(holdout_seeds):
        t0 = time.time()
        m = rl_run_episode(env, agent, enc_type, training=False, seed=seed)
        rows.append({"variant": name, "agent": spec["agent_type"], "regime": spec["regime"],
                     "seed": seed, "sharpe": m["sharpe"], "map": m["map"],
                     "mdd": m["mdd"], "final_pnl": m["final_pnl"]})
        print(f"  {name:16s} ep {i+1:2d}/{len(holdout_seeds)} "
              f"sharpe {m['sharpe']:+.3f}  ({time.time()-t0:.0f}s)  ckpt={ckpt.name}", flush=True)
    env.close()
    return rows


def run_as_recovery_for_variant(name: str, spec: dict) -> pd.DataFrame | None:
    ckpt = _resolve_best_checkpoint(spec["run_tag"])
    if ckpt is None:
        return None
    agent, enc_type = load_agent(str(ckpt), agent_type=spec["agent_type"],
                                  encoder_type="handcrafted", **spec["load_kwargs"])
    env = build_regime_env(spec["regime"], seed=AS_RECOVERY_SEED)
    results = run_recovery_all_agents({name: agent}, env, enc_type=enc_type,
                                       n_episodes=15, seed=AS_RECOVERY_SEED)
    env.close()
    return recovery_summary_df(results)


def load_baseline_holdout(agent_type: str, regime: str) -> pd.DataFrame:
    """Existing non-recurrent snapshot held-out episodes for this agent/regime, same 15 seeds."""
    df = pd.read_csv(OUT_DIR / "holdout_eval.csv")
    sub = df[(df["type"] == "rl") & (df["regime"] == regime) & (df["model"] == agent_type)].copy()
    return sub[["seed", "sharpe", "map", "mdd", "final_pnl"]]


def load_cvar_baseline_holdout(agent_type: str, alpha: float, train_seed: int = BASE_SEED) -> pd.DataFrame:
    """Non-recurrent snapshot held-out episodes for this agent at a specific CVaR
    alpha and training seed — the correct baseline for a recurrent variant
    trained at a non-default alpha, where holdout_eval.csv only has the main
    campaign's single alpha=0.25 snapshot row.

    Reads from cvar_sweep_holdout.csv (seeds BASE_SEED+80000+i = 80042-80056).
    An earlier version of run_cvar_sweep_eval.py dropped the BASE_SEED term,
    producing seeds 80000-80014 instead, which silently broke every cross-file
    paired comparison (no shared seeds) — fixed, and the full 40-combo sweep
    was re-run on the corrected block, so this reads the canonical file again.
    (cvar_sweep_holdout_qrdqn_alpha_baseline_corrected.csv was a small targeted
    stopgap computed before that full re-run finished; kept for the historical
    record, superseded by this file.)"""
    df = pd.read_csv(OUT_DIR / "cvar_sweep_holdout.csv")
    sub = df[(df["agent"] == agent_type) & (df["alpha"] == alpha) & (df["train_seed"] == train_seed)].copy()
    return sub.rename(columns={"holdout_seed": "seed"})[["seed", "sharpe", "map", "mdd", "final_pnl"]]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=15)
    p.add_argument("--skip", nargs="+", default=[],
                   help="Variant names to skip (e.g. --skip ppo_normal).")
    args = p.parse_args()

    variants_to_run = {k: v for k, v in RECURRENT_VARIANTS.items() if k not in args.skip}
    if args.skip:
        print(f"Skipping: {', '.join(args.skip)}")

    holdout_seeds = [BASE_SEED + HOLDOUT_OFFSET + i for i in range(args.n_episodes)]
    print(f"Holdout seed block: {holdout_seeds[0]}..{holdout_seeds[-1]} "
          f"({args.n_episodes} episodes, same block as scripts/run_holdout_eval.py)")

    all_rows = []
    as_recovery_frames = []
    sig_rows = []

    for name, spec in variants_to_run.items():
        print(f"\n{'='*78}\nHOLDOUT EVAL — {name} (recurrent)\n{'='*78}")
        all_rows.extend(run_holdout_for_variant(name, spec, holdout_seeds))

    for name, spec in variants_to_run.items():
        print(f"\n{'='*78}\nAS-RECOVERY — {name} (recurrent)\n{'='*78}")
        df = run_as_recovery_for_variant(name, spec)
        if df is not None:
            df["variant"] = name
            as_recovery_frames.append(df)
            print(df.to_string(index=False))

    holdout_df = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    holdout_df.to_csv(OUT_DIR / "recurrent_variant_holdout.csv", index=False)

    if as_recovery_frames:
        pd.concat(as_recovery_frames, ignore_index=True).to_csv(
            OUT_DIR / "recurrent_variant_as_recovery.csv", index=False)

    # ── Significance test: each recurrent variant vs. its own non-recurrent
    # snapshot baseline (same agent, same regime), paired on shared seeds ──
    for name, spec in variants_to_run.items():
        variant_rows = holdout_df[holdout_df["variant"] == name].set_index("seed")["sharpe"]
        if variant_rows.empty:
            continue
        if "cvar_alpha" in spec:
            baseline_df = load_cvar_baseline_holdout(spec["agent_type"], spec["cvar_alpha"])
        else:
            baseline_df = load_baseline_holdout(spec["agent_type"], spec["regime"])
        baseline_by_seed = baseline_df.set_index("seed")["sharpe"]
        shared_seeds = sorted(set(variant_rows.index) & set(baseline_by_seed.index))
        if not shared_seeds:
            print(f"  [warn] {name}: no shared seeds with non-recurrent baseline — skipping sig test")
            continue
        a = variant_rows.loc[shared_seeds].to_numpy()
        b = baseline_by_seed.loc[shared_seeds].to_numpy()
        diff = a - b
        t_p = ttest_rel(a, b).pvalue
        try:
            w_p = wilcoxon(a, b).pvalue
        except ValueError:
            w_p = float("nan")
        sig_rows.append({
            "variant": name, "agent": spec["agent_type"], "regime": spec["regime"],
            "alpha": spec.get("cvar_alpha"),
            "recurrent_sharpe_mean": a.mean(), "non_recurrent_sharpe_mean": b.mean(),
            "mean_diff": diff.mean(), "t_p": t_p, "wilcoxon_p": w_p,
            "significantly_different_p05": bool(w_p < 0.05),
        })

    sig_df = pd.DataFrame(sig_rows)
    sig_df.to_csv(OUT_DIR / "recurrent_variant_significance.csv", index=False)

    print(f"\n{'='*78}\nRECURRENT VARIANT HOLDOUT SUMMARY (vs. own non-recurrent baseline)\n{'='*78}")
    print(sig_df.to_string(index=False))
    print(f"\nSaved → {OUT_DIR / 'recurrent_variant_holdout.csv'}, "
          f"{OUT_DIR / 'recurrent_variant_as_recovery.csv'}, "
          f"{OUT_DIR / 'recurrent_variant_significance.csv'}")


if __name__ == "__main__":
    main()
