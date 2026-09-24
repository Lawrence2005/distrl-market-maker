"""
scripts/build_dashboard_data.py

Assembles the dashboard's viz_data.json from logs/ and the CSV/JSON outputs
of scripts/run_analysis.py, run_holdout_eval.py, run_significance_test.py,
run_stylized_facts.py, and run_ood_transfer.py. Regenerate this any time one
of those upstream artifacts changes, then rebuild dashboard.html from
dashboard.template.html + this file (see the dashboard build step in
evaluation/results_scaled_down_sweep/README.md).

Usage:
    python scripts/build_dashboard_data.py [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from evaluation.ablation import _run_tag as _ablation_run_tag

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR     = PROJECT_ROOT / "logs"
RESULTS_DIR  = PROJECT_ROOT / "evaluation" / "results_scaled_down_sweep"

REGIMES     = ["low_vol", "normal", "high_vol"]
RL_AGENTS   = ["sarsa", "dqn", "ppo", "qrdqn", "iqn"]
BASELINES   = ["fixedspread", "as", "glft"]


def _run_tag(agent: str, regime: str) -> str:
    # Delegates to evaluation.ablation's run_tag builder (the single source
    # of truth matching training/train.py's convention) rather than
    # reimplementing it — this file used to hand-build the tag without the
    # alpha_tag or sampling_tag, which silently dropped every dqn/qrdqn/iqn
    # run from the dashboard the moment either tag was introduced.
    # sampling="per" explicitly: this campaign trained dqn/qrdqn/iqn with
    # prioritized_replay=true only, so the "_uniform" tag _ablation_run_tag
    # defaults to points at checkpoint/log dirs that don't exist (same bug
    # just found and fixed in scripts/run_analysis.py::_run_tag).
    return _ablation_run_tag(agent, "handcrafted", "asymmetric", regime, 42,
                              sampling="per")


def _baseline_tag(agent: str, regime: str) -> str:
    return f"{agent}_analytical_asymmetric_{regime}_seed42"


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def build_rl_runs() -> list[dict]:
    runs = []
    for agent in RL_AGENTS:
        for regime in REGIMES:
            tag = _run_tag(agent, regime)
            log_dir = LOGS_DIR / tag
            status_path = log_dir / "status.json"
            if not status_path.exists():
                continue
            status = load_json(status_path)
            train_hist = load_json(log_dir / "train_history.json")
            eval_hist_path = log_dir / "eval_history.json"
            eval_hist = load_json(eval_hist_path) if eval_hist_path.exists() else []

            n_budget = status["n_episodes"]
            final_ep = status["episode"]
            state = status["state"]

            train = [{"ep": t["episode"], "sharpe": round(t["sharpe"], 4)} for t in train_hist]
            loss = [round(t.get("mean_loss", 0.0), 6) for t in train_hist]
            ev = [{"ep": e["episode"], "sharpe": round(e["sharpe"], 4),
                   "sharpe_std": round(e.get("sharpe_std", 0.0), 4),
                   "mdd": round(e.get("mdd", 0.0), 2), "map": round(e.get("map", 0.0), 2),
                   "pnl": round(e.get("final_pnl", 0.0), 2)} for e in eval_hist]

            best = max(eval_hist, key=lambda e: e["sharpe"], default=None)
            last3 = eval_hist[-3:] if len(eval_hist) >= 3 else eval_hist
            final_mean = (sum(e["sharpe"] for e in last3) / len(last3)) if last3 else float("nan")
            final_std = (pd.Series([e["sharpe"] for e in last3]).std() if len(last3) > 1 else 0.0)

            runs.append({
                "agent": agent, "regime": regime,
                "n_episodes_budget": n_budget, "final_episode": final_ep, "state": state,
                "train": train, "eval": ev,
                "best_eval_sharpe": round(best["sharpe"], 4) if best else float("nan"),
                "best_eval_episode": best["episode"] if best else None,
                "final_eval_sharpe_mean": round(final_mean, 4),
                "final_eval_sharpe_std": round(float(final_std), 4),
                "final_eval_window_n": len(last3),
                "loss": loss,
            })
    return runs


def build_baseline_runs() -> list[dict]:
    runs = []
    for agent in BASELINES:
        for regime in REGIMES:
            tag = _baseline_tag(agent, regime)
            log_dir = LOGS_DIR / tag
            status_path = log_dir / "status.json"
            if not status_path.exists():
                continue
            status = load_json(status_path)
            train_hist = load_json(log_dir / "train_history.json")
            train = [{"ep": t["episode"], "sharpe": round(t["sharpe"], 4)} for t in train_hist]
            runs.append({
                "agent": agent, "regime": regime,
                "n_episodes": status["n_episodes"],
                "train": train,
            })
    return runs


def csv_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(pd.read_csv(path).to_json(orient="records"))


def build() -> dict:
    data = {
        "rl_runs": build_rl_runs(),
        "baseline_runs": build_baseline_runs(),
        "as_recovery": csv_records(RESULTS_DIR / "as_recovery.csv"),
        "model_comparison": csv_records(RESULTS_DIR / "model_comparison.csv"),
        "holdout_eval": csv_records(RESULTS_DIR / "holdout_eval.csv"),
        "holdout_summary": csv_records(RESULTS_DIR / "holdout_summary.csv"),
        "significance_vs_glft": csv_records(RESULTS_DIR / "significance_vs_glft.csv"),
        "ood_transfer_episodes": csv_records(RESULTS_DIR / "ood_transfer_episodes.csv"),
        "ood_transfer_summary": csv_records(RESULTS_DIR / "ood_transfer_summary.csv"),
        "cvar_efficient_frontier": csv_records(RESULTS_DIR / "cvar_efficient_frontier.csv"),
        "encoder_ablation_summary": csv_records(RESULTS_DIR / "encoder_ablation_summary.csv"),
        "encoder_ablation_significance": csv_records(RESULTS_DIR / "encoder_ablation_significance.csv"),
    }
    stylized_path = RESULTS_DIR / "stylized_facts.json"
    data["stylized_facts"] = load_json(stylized_path) if stylized_path.exists() else {}
    return data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=RESULTS_DIR / "viz_data.json")
    args = p.parse_args()

    data = build()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(data, f)
    print(f"Saved → {args.out}  "
          f"({len(data['rl_runs'])} rl_runs, {len(data['baseline_runs'])} baseline_runs, "
          f"{len(data['as_recovery'])} as_recovery rows, {len(data['significance_vs_glft'])} significance rows)")


if __name__ == "__main__":
    main()
