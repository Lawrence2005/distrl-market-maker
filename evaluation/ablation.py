"""
evaluation/ablation.py

Ablation matrix analysis for distrl-market-maker.

Reads all 17 experiment run directories and builds the agent×encoder
performance matrix. Called once; output feeds Figure 2 (ablation heatmap)
and answers the four research questions from Week 8.

17 variants:
    1  SARSA + handcrafted
    4  agents (DQN, PPO, QR-DQN, IQN) × 3 snapshot encoders = 12 runs
    4  agents × 1 recurrent variant = 4 runs

Research questions answered
----------------------------
Q1: Does recurrent integration (LSTM) consistently outperform snapshot
    encoders across DQN, QR-DQN, and IQN — or only for some agents?

Q2: Is the recurrent advantage largest in high-vol / trending regimes
    (where temporal order flow clustering is most predictive)?

Q3: Does the recurrent advantage compound with the CVaR distributional
    objective (QR-DQN/IQN > DQN)?

Q4: Does AE pre-training accelerate convergence vs CNN trained from scratch?

Usage
-----
    from evaluation.ablation import AblationAnalysis

    ab = AblationAnalysis(log_root="logs/", regime="high_vol", seed=42)
    matrix_df  = ab.build_matrix(metric="sharpe")
    conv_df    = ab.convergence_speed(target_sharpe=0.1)
    recurrent_df = ab.recurrent_advantage()
    ae_vs_cnn  = ab.ae_vs_cnn_convergence()
    print(ab.summary())

Week 8 deliverable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from evaluation.metrics import (
    load_train_history,
    load_eval_history,
    summary_table,
)


# ── Constants ──────────────────────────────────────────────────────────────────

AGENTS   = ["sarsa", "dqn", "ppo", "qrdqn", "iqn"]
ENCODERS = ["handcrafted", "cnn", "autoencoder"]
VARIANTS = ["snapshot", "recurrent"]

# SARSA only supports handcrafted encoder
_SARSA_ONLY = {"handcrafted"}
# Which agents support recurrent variant
_RECURRENT_AGENTS = {"dqn", "ppo", "qrdqn", "iqn"}


def _run_tag(
    agent:     str,
    encoder:   str,
    reward:    str,
    regime:    str,
    seed:      int,
    recurrent: bool = False,
) -> str:
    """Build the run directory name matching train.py's run_tag convention."""
    variant = "_recurrent" if recurrent else ""
    return f"{agent}_{encoder}_{reward}_{regime}{variant}_seed{seed}"


class AblationAnalysis:
    """
    Reads all ablation run directories and answers the four research questions.

    Parameters
    ----------
    log_root : str | Path — path to logs/ directory
    regime   : str        — primary regime for the ablation (default 'high_vol')
    reward   : str        — reward type (default 'asymmetric')
    seed     : int        — seed used for all runs (default 42)
    episodes_window : int — average over last N episodes for final metrics
    """

    def __init__(
        self,
        log_root:        str | Path = "logs/",
        regime:          str = "high_vol",
        reward:          str = "asymmetric",
        seed:            int = 42,
        episodes_window: int = 100,
    ):
        self.log_root        = Path(log_root)
        self.regime          = regime
        self.reward          = reward
        self.seed            = seed
        self.episodes_window = episodes_window

        # Cache of loaded histories keyed by run_tag
        self._cache: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load(self, run_tag: str, history: str = "eval") -> Optional[pd.DataFrame]:
        """
        Load eval or train history for a run, with caching.

        Parameters
        ----------
        run_tag : str    — run directory name
        history : str    — 'eval' or 'train'

        Returns
        -------
        pd.DataFrame | None — None if run not found
        """
        cache_key = f"{run_tag}_{history}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        run_dir = self.log_root / run_tag
        if not run_dir.exists():
            return None

        try:
            if history == "eval":
                df = load_eval_history(run_dir)
            else:
                df = load_train_history(run_dir)
            self._cache[cache_key] = df
            return df
        except FileNotFoundError:
            return None

    def _final_metric(
        self,
        run_tag: str,
        metric:  str,
        history: str = "eval",
    ) -> float:
        """
        Get the final (mean of last episodes_window) value of a metric.

        Returns np.nan if run not found or metric missing.
        """
        df = self._load(run_tag, history)
        if df is None or metric not in df.columns:
            return np.nan
        return float(df[metric].iloc[-self.episodes_window:].mean())

    def _available_runs(self) -> list[str]:
        """List all run directories that exist under log_root."""
        if not self.log_root.exists():
            return []
        return [d.name for d in self.log_root.iterdir() if d.is_dir()]

    # ------------------------------------------------------------------
    # Q1: Main ablation matrix
    # ------------------------------------------------------------------

    def build_matrix(
        self,
        metric:    str = "sharpe",
        history:   str = "eval",
    ) -> pd.DataFrame:
        """
        Build the agent × encoder performance matrix.

        Rows: agents (sarsa, dqn, ppo, qrdqn, iqn)
        Cols: encoders + recurrent variant
              (handcrafted, cnn, autoencoder, recurrent)

        Parameters
        ----------
        metric  : str — metric to populate (default 'sharpe')
        history : str — 'eval' or 'train'

        Returns
        -------
        pd.DataFrame — shape (5, 4), NaN where combination not valid/run
        """
        cols = ENCODERS + ["recurrent"]
        rows = []

        for agent in AGENTS:
            row = {"agent": agent}

            for enc in ENCODERS:
                if agent == "sarsa" and enc != "handcrafted":
                    row[enc] = np.nan
                    continue

                tag = _run_tag(agent, enc, self.reward, self.regime, self.seed)
                row[enc] = self._final_metric(tag, metric, history)

            # Recurrent variant
            if agent in _RECURRENT_AGENTS:
                tag = _run_tag(agent, "handcrafted", self.reward,
                               self.regime, self.seed, recurrent=True)
                row["recurrent"] = self._final_metric(tag, metric, history)
            else:
                row["recurrent"] = np.nan   # SARSA has no recurrent variant

            rows.append(row)

        df = pd.DataFrame(rows).set_index("agent")
        df.index.name = "agent"
        return df

    def build_matrix_all_metrics(self) -> dict[str, pd.DataFrame]:
        """
        Build ablation matrices for all key metrics.

        Returns
        -------
        dict mapping metric_name → pd.DataFrame
        """
        metrics = ["sharpe", "map", "mdd", "final_pnl"]
        return {m: self.build_matrix(metric=m) for m in metrics}

    # ------------------------------------------------------------------
    # Q1: Is recurrent consistently better?
    # ------------------------------------------------------------------

    def recurrent_advantage(
        self,
        metric: str = "sharpe",
    ) -> pd.DataFrame:
        """
        Compute recurrent vs snapshot advantage for each agent.

        advantage = recurrent_metric - best_snapshot_metric

        Positive = recurrent is better.
        Negative = snapshot is better.

        Parameters
        ----------
        metric : str — metric to compare (default 'sharpe')

        Returns
        -------
        pd.DataFrame — one row per agent with columns:
            best_snapshot, recurrent, advantage, advantage_pct, consistent
        """
        matrix = self.build_matrix(metric=metric)
        rows   = []

        for agent in AGENTS:
            if agent not in _RECURRENT_AGENTS:
                continue

            snapshot_vals = matrix.loc[agent, ENCODERS].dropna()
            if snapshot_vals.empty:
                continue

            best_snap  = float(snapshot_vals.max())
            recur_val  = float(matrix.loc[agent, "recurrent"])

            if np.isnan(recur_val):
                continue

            advantage     = recur_val - best_snap
            advantage_pct = advantage / (abs(best_snap) + 1e-10) * 100
            best_enc      = snapshot_vals.idxmax()

            rows.append({
                "agent":          agent,
                "best_snapshot":  best_snap,
                "best_encoder":   best_enc,
                "recurrent":      recur_val,
                "advantage":      advantage,
                "advantage_pct":  advantage_pct,
                "recurrent_wins": advantage > 0,
            })

        return pd.DataFrame(rows).sort_values("advantage", ascending=False)

    # ------------------------------------------------------------------
    # Q2: Recurrent advantage by regime
    # ------------------------------------------------------------------

    def recurrent_advantage_by_regime(
        self,
        regimes: list[str] = None,
        agent:   str = "qrdqn",
        metric:  str = "sharpe",
    ) -> pd.DataFrame:
        """
        Compare recurrent vs snapshot advantage across regimes.

        Tests whether recurrent advantage is largest in high-vol/trending
        regimes where temporal order flow clustering is most predictive.

        Parameters
        ----------
        regimes : list of regime names to compare
        agent   : agent to use for comparison (default 'qrdqn')
        metric  : metric to compare

        Returns
        -------
        pd.DataFrame — one row per regime
        """
        if regimes is None:
            regimes = ["low_vol", "high_vol", "trending", "normal"]

        rows = []
        for regime in regimes:
            snap_tag = _run_tag(agent, "handcrafted", self.reward,
                                regime, self.seed, recurrent=False)
            recur_tag = _run_tag(agent, "handcrafted", self.reward,
                                 regime, self.seed, recurrent=True)

            snap_val  = self._final_metric(snap_tag,  metric)
            recur_val = self._final_metric(recur_tag, metric)

            rows.append({
                "regime":         regime,
                "snapshot":       snap_val,
                "recurrent":      recur_val,
                "advantage":      recur_val - snap_val,
                "recurrent_wins": recur_val > snap_val,
            })

        return (
            pd.DataFrame(rows)
            .sort_values("advantage", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Q3: CVaR advantage — does distributional + recurrent compound?
    # ------------------------------------------------------------------

    def distributional_advantage(
        self,
        metric:    str = "sharpe",
        encoder:   str = "handcrafted",
        recurrent: bool = False,
    ) -> pd.DataFrame:
        """
        Compare distributional (QR-DQN, IQN) vs non-distributional (DQN, PPO).

        Tests whether the CVaR distributional objective adds value
        independently of the encoder / recurrent choice.

        Parameters
        ----------
        metric    : str  — metric to compare
        encoder   : str  — encoder to compare on
        recurrent : bool — use recurrent variant

        Returns
        -------
        pd.DataFrame — one row per agent with final metric value
        """
        rows = []
        for agent in ["dqn", "ppo", "qrdqn", "iqn"]:
            if agent == "sarsa":
                continue

            tag = _run_tag(agent, encoder, self.reward,
                           self.regime, self.seed, recurrent)
            val = self._final_metric(tag, metric)

            rows.append({
                "agent":           agent,
                "distributional":  agent in ("qrdqn", "iqn"),
                "encoder":         encoder,
                "recurrent":       recurrent,
                metric:            val,
            })

        df = pd.DataFrame(rows).sort_values(metric, ascending=False)

        # Add group means
        dist_mean  = df[df["distributional"]][metric].mean()
        nodist_mean = df[~df["distributional"]][metric].mean()
        print(f"Distributional mean {metric}: {dist_mean:.4f}")
        print(f"Non-distributional mean {metric}: {nodist_mean:.4f}")
        print(f"Advantage: {dist_mean - nodist_mean:+.4f}")

        return df

    # ------------------------------------------------------------------
    # Q4: AE pre-training vs CNN from scratch
    # ------------------------------------------------------------------

    def ae_vs_cnn_convergence(
        self,
        agent:  str = "qrdqn",
        metric: str = "sharpe",
    ) -> pd.DataFrame:
        """
        Compare AE pre-training vs CNN (trained from scratch) convergence speed.

        Tests whether frozen AE features accelerate convergence by providing
        a better starting representation.

        Parameters
        ----------
        agent  : str — agent to compare
        metric : str — metric to track

        Returns
        -------
        pd.DataFrame — episode-level comparison with encoder column
        """
        dfs = []

        for enc in ["autoencoder", "cnn"]:
            tag = _run_tag(agent, enc, self.reward, self.regime, self.seed)
            df  = self._load(tag, history="train")
            if df is None or metric not in df.columns:
                print(f"  [skip] {tag} not found")
                continue
            df = df[["episode", metric]].copy()
            df["encoder"] = enc
            dfs.append(df)

        if not dfs:
            return pd.DataFrame()

        combined = pd.concat(dfs, ignore_index=True)

        # Find convergence episode (first episode where metric > threshold)
        # Use 80% of final value as convergence threshold
        for enc in combined["encoder"].unique():
            enc_df   = combined[combined["encoder"] == enc]
            final    = enc_df[metric].iloc[-50:].mean()
            threshold = 0.8 * final
            conv_ep   = enc_df[enc_df[metric] >= threshold]["episode"].min()
            print(f"  {enc}: final {metric}={final:.4f}, "
                  f"80% convergence at episode {conv_ep}")

        return combined

    # ------------------------------------------------------------------
    # Missing runs report
    # ------------------------------------------------------------------

    def missing_runs(self) -> list[str]:
        """
        List all expected run tags that don't have log directories.

        Useful for checking which runs still need to be submitted.

        Returns
        -------
        list[str] — missing run tags
        """
        missing = []

        for agent in AGENTS:
            for enc in ENCODERS:
                if agent == "sarsa" and enc != "handcrafted":
                    continue
                tag = _run_tag(agent, enc, self.reward, self.regime, self.seed)
                if not (self.log_root / tag).exists():
                    missing.append(tag)

            # Recurrent variant
            if agent in _RECURRENT_AGENTS:
                tag = _run_tag(agent, "handcrafted", self.reward,
                               self.regime, self.seed, recurrent=True)
                if not (self.log_root / tag).exists():
                    missing.append(tag)

        return missing

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, metric: str = "sharpe") -> str:
        """
        Print a concise summary of all ablation results.

        Parameters
        ----------
        metric : str — primary metric for summary (default 'sharpe')

        Returns
        -------
        str — formatted summary
        """
        matrix = self.build_matrix(metric=metric)
        missing = self.missing_runs()

        lines = [
            f"{'='*60}",
            f"Ablation Summary — {metric} — {self.regime} regime",
            f"{'='*60}",
            "",
            "Agent × Encoder Matrix:",
            matrix.to_string(float_format=lambda x: f"{x:.4f}"),
            "",
        ]

        # Best overall
        best_val  = np.nanmax(matrix.values)
        best_idx  = np.unravel_index(
            np.nanargmax(matrix.values), matrix.shape
        )
        best_agent = matrix.index[best_idx[0]]
        best_enc   = matrix.columns[best_idx[1]]
        lines.append(f"Best: {best_agent} + {best_enc} = {best_val:.4f}")

        # Recurrent advantage
        rec_df = self.recurrent_advantage(metric=metric)
        if not rec_df.empty:
            lines.append("")
            lines.append("Recurrent advantage:")
            for _, row in rec_df.iterrows():
                sign = "+" if row["recurrent_wins"] else "-"
                lines.append(
                    f"  {row['agent']:8s}: {sign}{abs(row['advantage']):.4f} "
                    f"({'recurrent wins' if row['recurrent_wins'] else 'snapshot wins'})"
                )

        if missing:
            lines.append("")
            lines.append(f"Missing runs ({len(missing)}):")
            for tag in missing:
                lines.append(f"  {tag}")

        return "\n".join(lines)