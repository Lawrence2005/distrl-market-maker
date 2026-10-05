"""
scripts/run_significance_test.py

Paired significance test of each RL agent vs. GLFT, per regime, on the
held-out episode data from scripts/run_holdout_eval.py — every model in a
regime sees the SAME fixed seed block, so per-episode differences are a
valid paired sample (paired t-test + Wilcoxon signed-rank, both computed;
Wilcoxon as the primary call since Sharpe differences aren't guaranteed
normal, t-test reported alongside for reference).

Reads:  results/holdout_eval.csv
Writes: results/significance_vs_glft.csv

Usage:
    python scripts/run_significance_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR      = PROJECT_ROOT / "results"

RL_AGENTS = ["sarsa", "dqn", "ppo", "qrdqn", "iqn"]


def main() -> None:
    holdout = pd.read_csv(OUT_DIR / "holdout_eval.csv")
    rows = []
    for regime, group in holdout.groupby("regime"):
        glft = group[group["model"] == "glft"].set_index("seed")["sharpe"]
        if glft.empty:
            print(f"  [warn] {regime}: no glft rows — skipping")
            continue
        for agent in RL_AGENTS:
            agent_rows = group[group["model"] == agent].set_index("seed")["sharpe"]
            if agent_rows.empty:
                continue
            shared_seeds = sorted(set(agent_rows.index) & set(glft.index))
            a = agent_rows.loc[shared_seeds].to_numpy()
            g = glft.loc[shared_seeds].to_numpy()
            diff = a - g
            t_p = ttest_rel(a, g).pvalue
            try:
                w_p = wilcoxon(a, g).pvalue
            except ValueError:
                w_p = float("nan")  # all-zero-difference edge case
            rows.append({
                "regime": regime,
                "agent": agent,
                "mean_diff_vs_glft": diff.mean(),
                "t_p": t_p,
                "wilcoxon_p": w_p,
                "beats_glft_p05": bool(diff.mean() > 0 and w_p < 0.05),
            })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(OUT_DIR / "significance_vs_glft.csv", index=False)
    print(f"\nSaved → {OUT_DIR / 'significance_vs_glft.csv'}")


if __name__ == "__main__":
    main()
