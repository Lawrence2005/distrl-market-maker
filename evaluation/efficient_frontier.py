"""
evaluation/efficient_frontier.py

CVaR efficient frontier analysis for distrl-market-maker.

Reads results from the CVaR α-sweep (5 runs at α=0.05,0.10,0.25,0.50,1.0)
and builds the mean PnL vs CVaR_0.10 efficient frontier — the key
distributional result of the project.

Research question answered
--------------------------
Does the CVaR objective trace a non-dominated efficient frontier?
    - Lower α → more risk-averse → lower mean PnL but better tail protection
    - α=1.0   → risk-neutral    → should behave like DQN (mean maximiser)
    - α→0     → extreme caution → very low CVaR, potentially low mean PnL

Expected shape:
    - Non-dominated: no α gives better BOTH mean AND CVaR than another
    - If some α dominates → CVaR policy found genuinely better strategy
      (acceptable result — report honestly per plan)

Also answers:
    - At α=1.0, does the policy match DQN? (sanity check for CVaR wrapper)
    - As α decreases, do spreads widen and MAP decrease? (risk-aversion check)

Usage
-----
    from evaluation.efficient_frontier import EfficientFrontier

    ef = EfficientFrontier(
        log_root="logs/",
        agent="qrdqn",
        encoder="autoencoder",
        regime="high_vol",
        seed=42,
    )
    df     = ef.build_frontier()
    checks = ef.sanity_checks()
    print(ef.summary())

Week 9 deliverable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from evaluation.metrics import load_eval_history, load_train_history


# ── Default alpha sweep values ────────────────────────────────────────────────

DEFAULT_ALPHAS = [0.05, 0.10, 0.25, 0.50, 1.0]


def _run_tag(
    agent:   str,
    encoder: str,
    reward:  str,
    regime:  str,
    alpha:   float,
    seed:    int,
) -> str:
    """
    Build run tag for a CVaR α-sweep run.

    Convention: {agent}_{encoder}_{reward}_{regime}_alpha{alpha}_seed{seed}
    Matches the naming produced by:
        python training/train.py agent=qrdqn ... alpha=0.05
    """
    alpha_str = f"{alpha:.2f}".rstrip("0").rstrip(".")
    return f"{agent}_{encoder}_{reward}_{regime}_alpha{alpha_str}_seed{seed}"


class EfficientFrontier:
    """
    CVaR efficient frontier analysis.

    Reads α-sweep run directories and computes:
        - mean episode PnL per α
        - CVaR_0.10 of episode PnLs per α
        - MAP (inventory management) per α
        - MDD per α
        - Spread width per α (proxy for risk-aversion behaviour)

    Parameters
    ----------
    log_root : str | Path — path to logs/ directory
    agent    : str        — agent name (default 'qrdqn')
    encoder  : str        — encoder name (default 'autoencoder')
    reward   : str        — reward type (default 'asymmetric')
    regime   : str        — regime (default 'high_vol')
    seed     : int        — seed (default 42)
    alphas   : list[float]— CVaR alpha values (default [0.05,0.10,0.25,0.50,1.0])
    cvar_tail: float      — tail fraction for episode-level CVaR (default 0.10)
    """

    def __init__(
        self,
        log_root: str | Path = "logs/",
        agent:    str   = "qrdqn",
        encoder:  str   = "autoencoder",
        reward:   str   = "asymmetric",
        regime:   str   = "high_vol",
        seed:     int   = 42,
        alphas:   list[float] = None,
        cvar_tail: float = 0.10,
    ):
        self.log_root  = Path(log_root)
        self.agent     = agent
        self.encoder   = encoder
        self.reward    = reward
        self.regime    = regime
        self.seed      = seed
        self.alphas    = alphas or DEFAULT_ALPHAS
        self.cvar_tail = cvar_tail

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load(self, alpha: float) -> Optional[pd.DataFrame]:
        """
        Load eval history for one alpha value.

        Tries the alpha-tagged run first, then falls back to the
        untagged run (for α=1.0 which is the baseline QR-DQN run).

        Returns None if neither found.
        """
        tag     = _run_tag(self.agent, self.encoder, self.reward,
                           self.regime, alpha, self.seed)
        run_dir = self.log_root / tag

        if not run_dir.exists():
            # Fallback: no alpha tag (base run)
            base_tag = (f"{self.agent}_{self.encoder}_{self.reward}"
                        f"_{self.regime}_seed{self.seed}")
            run_dir = self.log_root / base_tag

        if not run_dir.exists():
            return None

        try:
            return load_eval_history(run_dir)
        except FileNotFoundError:
            return None

    def _load_train(self, alpha: float) -> Optional[pd.DataFrame]:
        """Load train history for one alpha value."""
        tag     = _run_tag(self.agent, self.encoder, self.reward,
                           self.regime, alpha, self.seed)
        run_dir = self.log_root / tag
        if not run_dir.exists():
            return None
        try:
            return load_train_history(run_dir)
        except FileNotFoundError:
            return None

    # ------------------------------------------------------------------
    # Frontier computation
    # ------------------------------------------------------------------

    def build_frontier(self, episodes_window: int = 50) -> pd.DataFrame:
        """
        Build the efficient frontier DataFrame.

        For each α, computes from the last episodes_window eval episodes:
            mean_pnl   : mean final PnL
            cvar_10    : CVaR_0.10 of final PnLs (worst 10% episodes)
            cvar_25    : CVaR_0.25 of final PnLs
            map        : mean absolute position
            mdd        : mean max drawdown
            sharpe     : mean Sharpe ratio
            pnl_std    : std of final PnLs (spread of outcomes)

        Parameters
        ----------
        episodes_window : int — use last N eval episodes (default 50)

        Returns
        -------
        pd.DataFrame — one row per α, sorted by α ascending
        """
        rows = []

        for alpha in self.alphas:
            df = self._load(alpha)
            if df is None:
                print(f"  [skip] α={alpha} — run not found")
                continue

            tail = df.tail(episodes_window)
            if tail.empty:
                continue

            pnls = tail["final_pnl"].values if "final_pnl" in tail.columns else np.array([])

            if len(pnls) == 0:
                continue

            n_tail = max(1, int(self.cvar_tail * len(pnls)))

            row = {
                "alpha":     alpha,
                "mean_pnl":  float(pnls.mean()),
                "cvar_10":   float(np.sort(pnls)[:n_tail].mean()),
                "pnl_std":   float(pnls.std()),
                "n_episodes": len(pnls),
            }

            # Add other metrics if available
            for col in ["sharpe", "map", "mdd"]:
                if col in tail.columns:
                    row[col] = float(tail[col].mean())

            rows.append(row)

        if not rows:
            print("No frontier data found. Run CVaR α-sweep first.")
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Pareto dominance check
    # ------------------------------------------------------------------

    def pareto_analysis(self, frontier_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Check which α values are Pareto-dominated.

        A point (mean_pnl, cvar_10) is dominated if another point has
        both higher mean_pnl AND higher cvar_10.

        Parameters
        ----------
        frontier_df : pd.DataFrame from build_frontier() or None (builds it)

        Returns
        -------
        pd.DataFrame — frontier_df with added 'dominated' column
        """
        if frontier_df is None:
            frontier_df = self.build_frontier()

        if frontier_df.empty:
            return frontier_df

        df = frontier_df.copy()
        df["dominated"] = False

        for i, row in df.iterrows():
            for j, other in df.iterrows():
                if i == j:
                    continue
                # other dominates row if it's better on BOTH axes
                if (other["mean_pnl"] > row["mean_pnl"] and
                        other["cvar_10"] > row["cvar_10"]):
                    df.at[i, "dominated"] = True
                    break

        n_dominated = df["dominated"].sum()
        n_total     = len(df)
        print(f"Pareto analysis: {n_total - n_dominated}/{n_total} points "
              f"on the non-dominated frontier")

        return df

    # ------------------------------------------------------------------
    # Sanity checks (per plan)
    # ------------------------------------------------------------------

    def sanity_checks(self, frontier_df: pd.DataFrame = None) -> dict:
        """
        Run the three sanity checks from the Week 7 plan.

        Check 1: At α=1.0, does the policy behave like DQN (risk-neutral)?
                 Verify by comparing mean_pnl with a DQN baseline run.

        Check 2: As α decreases, does MAP decrease?
                 (risk-averse agent reduces inventory faster)

        Check 3: As α decreases, does CVaR improve relative to mean_pnl?
                 (risk-averse agent sacrifices mean for better tail)

        Parameters
        ----------
        frontier_df : pd.DataFrame or None

        Returns
        -------
        dict with keys: check1_pass, check2_pass, check3_pass, details
        """
        if frontier_df is None:
            frontier_df = self.build_frontier()

        if frontier_df.empty or len(frontier_df) < 2:
            return {"check1_pass": None, "check2_pass": None,
                    "check3_pass": None,
                    "details": "Insufficient data"}

        df = frontier_df.sort_values("alpha")

        details = {}

        # Check 1: α=1.0 should have highest mean_pnl (risk-neutral = mean maximiser)
        check1_pass = None
        if 1.0 in df["alpha"].values:
            alpha_1_pnl = float(df[df["alpha"] == 1.0]["mean_pnl"].iloc[0])
            max_pnl     = float(df["mean_pnl"].max())
            check1_pass = abs(alpha_1_pnl - max_pnl) < 0.1 * abs(max_pnl + 1e-10)
            details["check1"] = (
                f"α=1.0 mean_pnl={alpha_1_pnl:.4f}, "
                f"max mean_pnl={max_pnl:.4f} "
                f"→ {'PASS' if check1_pass else 'FAIL'}"
            )
        else:
            details["check1"] = "α=1.0 run not found"

        # Check 2: MAP should decrease as α decreases (more risk-averse = less inventory)
        check2_pass = None
        if "map" in df.columns and df["map"].notna().sum() >= 2:
            corr = df["alpha"].corr(df["map"])
            # Positive correlation means higher α → higher MAP
            # i.e. lower α → lower MAP (more conservative)
            check2_pass = corr > 0.3
            details["check2"] = (
                f"Pearson corr(α, MAP)={corr:.3f} "
                f"→ {'PASS (MAP increases with α)' if check2_pass else 'FAIL'}"
            )
        else:
            details["check2"] = "MAP data not available"

        # Check 3: CVaR should improve (be less negative) as α decreases
        # relative to mean_pnl — risk-averse agent protects the tail
        check3_pass = None
        if "cvar_10" in df.columns and df["cvar_10"].notna().sum() >= 2:
            # cvar_tail_ratio = cvar_10 / mean_pnl — should increase as α decreases
            df_clean = df[df["mean_pnl"].abs() > 1e-6].copy()
            if len(df_clean) >= 2:
                df_clean["tail_ratio"] = df_clean["cvar_10"] / df_clean["mean_pnl"]
                corr = df_clean["alpha"].corr(df_clean["tail_ratio"])
                # Negative corr: lower α → higher tail ratio (better tail relative to mean)
                check3_pass = corr < -0.2
                details["check3"] = (
                    f"Pearson corr(α, CVaR/mean)={corr:.3f} "
                    f"→ {'PASS (tail improves with lower α)' if check3_pass else 'FAIL'}"
                )
            else:
                details["check3"] = "Insufficient non-zero mean_pnl runs"
        else:
            details["check3"] = "CVaR data not available"

        return {
            "check1_pass": check1_pass,
            "check2_pass": check2_pass,
            "check3_pass": check3_pass,
            "details":     details,
        }

    # ------------------------------------------------------------------
    # Convergence speed across alphas
    # ------------------------------------------------------------------

    def convergence_by_alpha(
        self,
        metric:    str   = "sharpe",
        threshold: float = None,
    ) -> pd.DataFrame:
        """
        Compare convergence speed across α values.

        Parameters
        ----------
        metric    : str   — training metric to track
        threshold : float — convergence threshold (default: 80% of α=1.0 final)

        Returns
        -------
        pd.DataFrame — one row per α with convergence episode
        """
        # Determine threshold from α=1.0 run
        if threshold is None:
            df_base = self._load_train(1.0)
            if df_base is not None and metric in df_base.columns:
                final_base = float(df_base[metric].iloc[-50:].mean())
                threshold  = 0.8 * final_base
            else:
                threshold = 0.0

        rows = []
        for alpha in self.alphas:
            df = self._load_train(alpha)
            if df is None or metric not in df.columns:
                continue

            vals    = df[metric].values
            conv_ep = None
            for i, v in enumerate(vals):
                if v >= threshold:
                    conv_ep = int(df["episode"].iloc[i]) if "episode" in df.columns else i
                    break

            rows.append({
                "alpha":      alpha,
                "conv_episode": conv_ep,
                "final_metric": float(vals[-50:].mean()) if len(vals) >= 50 else float(vals.mean()),
            })

        return pd.DataFrame(rows).sort_values("alpha").reset_index(drop=True)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """
        Print a concise summary of the efficient frontier.

        Returns
        -------
        str — formatted summary
        """
        df     = self.build_frontier()
        checks = self.sanity_checks(df)

        lines = [
            f"{'='*60}",
            f"CVaR Efficient Frontier — {self.agent} + {self.encoder}",
            f"{self.regime} regime · seed={self.seed}",
            f"{'='*60}",
        ]

        if df.empty:
            lines.append("No data found. Run CVaR α-sweep first.")
            return "\n".join(lines)

        # Frontier table
        cols = ["alpha", "mean_pnl", "cvar_10", "pnl_std"]
        if "sharpe" in df.columns:
            cols.append("sharpe")
        if "map" in df.columns:
            cols.append("map")

        lines.append("")
        lines.append(df[cols].to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        ))

        # Best α
        best_sharpe_idx = df["sharpe"].idxmax() if "sharpe" in df.columns else None
        best_cvar_idx   = df["cvar_10"].idxmax()
        best_mean_idx   = df["mean_pnl"].idxmax()

        lines.append("")
        lines.append(f"Best mean PnL:  α={df.loc[best_mean_idx, 'alpha']:.2f}")
        lines.append(f"Best CVaR_0.10: α={df.loc[best_cvar_idx, 'alpha']:.2f}")
        if best_sharpe_idx is not None:
            lines.append(f"Best Sharpe:    α={df.loc[best_sharpe_idx, 'alpha']:.2f}")

        # Sanity checks
        lines.append("")
        lines.append("Sanity checks:")
        for k, v in checks["details"].items():
            lines.append(f"  {k}: {v}")

        return "\n".join(lines)