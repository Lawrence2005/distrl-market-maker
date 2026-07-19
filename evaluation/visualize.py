"""
evaluation/visualize.py

All plot generation for distrl-market-maker.

Called from notebooks — never generate plots directly in cells.
Any figure regenerates in one line:

    from evaluation.visualize import Visualizer
    viz = Visualizer(log_root="logs/", ckpt_root="checkpoints/")
    viz.plot_convergence(agents=["qrdqn", "dqn", "sarsa"])
    viz.plot_quote_skew(run_tags=["qrdqn_handcrafted_asymmetric_low_vol_seed42"])
    viz.plot_ablation_matrix()
    viz.plot_efficient_frontier()
    viz.plot_flash_crash()

Dark theme matches Week 3 baseline figures so overlays are consistent.

Figures saved to experiments/<notebook>/ by default.

Week 8 deliverable.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from evaluation.metrics import load_train_history, load_eval_history, load_all_runs


# ── Theme ─────────────────────────────────────────────────────────────────────

THEME = {
    "bg":      "#0d1117",
    "panel":   "#161b22",
    "grid":    "#21262d",
    "text":    "#e6edf3",
    "subtext": "#8b949e",
    "border":  "#30363d",
}

AGENT_COLORS = {
    "sarsa":  "#a78bfa",   # purple
    "dqn":    "#34d399",   # green
    "ppo":    "#f87171",   # red
    "qrdqn":  "#f59e0b",   # amber
    "iqn":    "#06b6d4",   # cyan
}

ENCODER_MARKERS = {
    "handcrafted": "o",
    "cnn":         "s",
    "autoencoder": "^",
}

REGIME_LINESTYLES = {
    "low_vol":    "-",
    "high_vol":   "--",
    "trending":   "-.",
    "normal":     ":",
    "flash_crash": (0, (3, 1, 1, 1)),
}


def _dark_fig(figsize=(10, 5.5), nrows=1, ncols=1):
    """Create a dark-themed figure matching Week 3 baseline style."""
    fig, axes = plt.subplots(
        nrows, ncols, figsize=figsize,
        facecolor=THEME["bg"],
        squeeze=False,
    )
    for ax in axes.flat:
        ax.set_facecolor(THEME["panel"])
        ax.tick_params(colors=THEME["subtext"], labelsize=9)
        ax.grid(color=THEME["grid"], lw=0.5, zorder=0)
        for spine in ax.spines.values():
            spine.set_edgecolor(THEME["border"])
    if nrows == 1 and ncols == 1:
        return fig, axes[0, 0]
    return fig, axes


def _legend(ax, **kwargs):
    leg = ax.legend(
        framealpha=0.15, facecolor=THEME["bg"],
        edgecolor=THEME["border"], labelcolor=THEME["text"],
        fontsize=9, **kwargs
    )
    return leg


def _label(ax, xlabel=None, ylabel=None, title=None):
    if xlabel:
        ax.set_xlabel(xlabel, color=THEME["text"], fontsize=11, labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, color=THEME["text"], fontsize=11, labelpad=6)
    if title:
        ax.set_title(title, color=THEME["text"], fontsize=12, pad=10, loc="left")


# ══════════════════════════════════════════════════════════════════════════════
# Visualizer class
# ══════════════════════════════════════════════════════════════════════════════

class Visualizer:
    """
    All plot generation for distrl-market-maker evaluation.

    Parameters
    ----------
    log_root  : str | Path — path to logs/ directory
    ckpt_root : str | Path — path to checkpoints/ directory
    out_root  : str | Path — path to save figures (default experiments/)
    """

    def __init__(
        self,
        log_root:  str | Path = "logs/",
        ckpt_root: str | Path = "checkpoints/",
        out_root:  str | Path = "experiments/",
    ):
        self.log_root  = Path(log_root)
        self.ckpt_root = Path(ckpt_root)
        self.out_root  = Path(out_root)

    def _save(self, fig, name: str, subdir: str = "") -> Path:
        out = self.out_root / subdir if subdir else self.out_root
        out.mkdir(parents=True, exist_ok=True)
        path = out / name
        fig.savefig(path, dpi=150, bbox_inches="tight",
                    facecolor=THEME["bg"])
        plt.close(fig)
        print(f"Saved → {path}")
        return path

    # ------------------------------------------------------------------
    # Figure 1 reference (load Week 3 baselines for overlay)
    # ------------------------------------------------------------------

    def _load_baseline_skew(self) -> dict:
        """
        Load Week 3 baseline quote skew data for overlay on Figure 2.
        Returns dict with keys 'AS' and 'GLFT', each a (inv_levels, mean_offsets) tuple.
        Falls back to empty dict if baseline data not found.
        """
        baseline_dir = self.out_root / "w03_baselines"
        result = {}

        for name in ["AS", "GLFT"]:
            path = baseline_dir / f"baseline_skew_{name}.npy"
            if path.exists():
                data = np.load(path)
                result[name] = (data[0], data[1])
        return result

    # ------------------------------------------------------------------
    # Convergence curves (Week 6 notebook)
    # ------------------------------------------------------------------

    def plot_convergence(
        self,
        agents:   list[str] = None,
        metric:   str = "sharpe",
        smooth:   int = 20,
        regime:   str = "low_vol",
        encoder:  str = "handcrafted",
        reward:   str = "asymmetric",
        seed:     int = 42,
        save:     bool = True,
    ) -> plt.Figure:
        """
        Plot training convergence curves for multiple agents.

        Parameters
        ----------
        agents  : list of agent names to plot (default: all found)
        metric  : metric to plot (default 'sharpe')
        smooth  : rolling window for smoothing (default 20)
        regime  : regime filter
        encoder : encoder filter
        reward  : reward filter
        seed    : seed filter
        save    : save to experiments/w06_convergence/

        Returns
        -------
        matplotlib Figure
        """
        fig, ax = _dark_fig(figsize=(11, 5))

        if agents is None:
            agents = list(AGENT_COLORS.keys())

        for agent_name in agents:
            run_tag = f"{agent_name}_{encoder}_{reward}_{regime}_seed{seed}"
            run_dir = self.log_root / run_tag

            if not run_dir.exists():
                print(f"  [skip] {run_tag} not found")
                continue

            try:
                df = load_train_history(run_dir)
            except FileNotFoundError:
                continue

            if metric not in df.columns:
                print(f"  [skip] metric '{metric}' not in {run_tag}")
                continue

            vals     = df[metric].values.astype(float)
            smoothed = np.convolve(vals, np.ones(smooth) / smooth, mode="valid")
            x        = np.arange(len(smoothed)) + smooth

            color = AGENT_COLORS.get(agent_name, "#ffffff")
            ax.plot(x, smoothed, color=color, lw=1.8, label=agent_name)

            # ±1σ band (rolling)
            if len(vals) >= smooth:
                stds = np.array([
                    vals[max(0, i-smooth):i].std()
                    for i in range(smooth, len(vals)+1)
                ])
                ax.fill_between(x, smoothed - stds, smoothed + stds,
                                color=color, alpha=0.08)

        ax.axhline(0, color=THEME["subtext"], lw=0.6, ls=":", alpha=0.5)
        _label(ax,
               xlabel="Episode",
               ylabel=metric.capitalize(),
               title=f"Training Convergence — {metric} (smoothed {smooth}-ep)\n"
                     f"{regime} regime · {encoder} encoder")
        _legend(ax)

        if save:
            return self._save(fig, f"convergence_{metric}.png", "w06_convergence")
        return fig

    # ------------------------------------------------------------------
    # Quote skew overlay on Figure 1 (Week 6 + Week 8)
    # ------------------------------------------------------------------

    def plot_quote_skew(
        self,
        agent_data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
        title:      str = "Figure 2 — Agent Quote Skew vs GLFT Reference",
        overlay_baselines: bool = True,
        min_visits: int = 5,
        boundary_buffer: int = 2,
        save:       bool = True,
        save_name:  str = "figure2_quote_skew.png",
        save_subdir: str = "w06_convergence",
    ) -> plt.Figure:
        """
        Plot bid_offset vs inventory for trained agents, overlaid on GLFT.

        Parameters
        ----------
        agent_data : dict mapping agent_name →
                     (inv_levels, mean_offsets, std_offsets)
                     as returned by quote_skew_curve()
        title      : figure title
        overlay_baselines : if True, load and overlay Week 3 AS/GLFT curves
        min_visits : minimum visits per inventory level to include
        boundary_buffer : exclude levels within N of ±Q_max
        save       : save figure to disk

        Returns
        -------
        matplotlib Figure
        """
        fig, ax = _dark_fig(figsize=(10, 5.5))

        # Week 3 baseline overlay
        if overlay_baselines:
            baselines = self._load_baseline_skew()
            baseline_colors = {"AS": "#f59e0b", "GLFT": "#06b6d4"}
            for name, (inv_lvls, mean_off) in baselines.items():
                ax.plot(inv_lvls, mean_off,
                        color=baseline_colors.get(name, "#ffffff"),
                        lw=1.5, ls="--", alpha=0.5, label=f"{name} (baseline)")

        # FixedSpread reference line
        ax.axhline(2, color=THEME["subtext"], lw=1.0, ls="--",
                   alpha=0.5, label="FixedSpread (floor)")

        # Agent curves
        for agent_name, (inv_levels, mean_offsets, std_offsets) in agent_data.items():
            color = AGENT_COLORS.get(agent_name, "#ffffff")
            ax.plot(inv_levels, mean_offsets, color=color,
                    lw=2.0, marker="o", ms=4, label=agent_name, zorder=4)
            ax.fill_between(inv_levels,
                            mean_offsets - std_offsets,
                            mean_offsets + std_offsets,
                            color=color, alpha=0.10, zorder=3)

        ax.axvline(0, color=THEME["subtext"], lw=0.7, ls=":", alpha=0.5)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        _label(ax,
               xlabel="Inventory  q",
               ylabel="Bid offset  (ticks below mid)",
               title=title)
        _legend(ax, loc="upper left")

        if save:
            return self._save(fig, save_name, save_subdir)
        return fig

    # ------------------------------------------------------------------
    # Ablation matrix heatmap (Week 8)
    # ------------------------------------------------------------------

    def plot_ablation_matrix(
        self,
        metric:  str = "sharpe",
        regime:  str = "high_vol",
        seed:    int = 42,
        save:    bool = True,
    ) -> plt.Figure:
        """
        Plot 4×4 agent×encoder ablation matrix as a heatmap.

        Rows: agents (dqn, ppo, qrdqn, iqn)
        Cols: encoders (handcrafted, cnn, autoencoder, recurrent)

        Parameters
        ----------
        metric : metric to display (default 'sharpe')
        regime : regime filter
        seed   : seed filter
        save   : save figure to disk

        Returns
        -------
        matplotlib Figure
        """
        agents   = ["sarsa", "dqn", "ppo", "qrdqn", "iqn"]
        encoders = ["handcrafted", "cnn", "autoencoder", "recurrent"]

        matrix = np.full((len(agents), len(encoders)), np.nan)

        for i, agent_name in enumerate(agents):
            for j, enc_name in enumerate(encoders):
                if agent_name == "sarsa" and enc_name != "handcrafted":
                    continue   # SARSA only supports handcrafted

                if enc_name == "recurrent":
                    run_tag = (
                        f"{agent_name}_handcrafted_asymmetric"
                        f"_{regime}_recurrent_seed{seed}"
                    )
                else:
                    run_tag = (
                        f"{agent_name}_{enc_name}_asymmetric"
                        f"_{regime}_seed{seed}"
                    )

                run_dir = self.log_root / run_tag
                if not run_dir.exists():
                    continue

                try:
                    df  = load_eval_history(run_dir)
                    val = df[metric].iloc[-10:].mean() if metric in df.columns else np.nan
                    matrix[i, j] = val
                except Exception:
                    continue

        fig, ax = _dark_fig(figsize=(9, 5))

        # Mask NaN
        masked = np.ma.masked_invalid(matrix)
        vmin   = np.nanmin(matrix) if not np.all(np.isnan(matrix)) else -1
        vmax   = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else  1

        im = ax.imshow(masked, cmap="RdYlGn", vmin=vmin, vmax=vmax,
                       aspect="auto")

        # Annotate cells
        for i in range(len(agents)):
            for j in range(len(encoders)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                            color="black" if abs(val) < 0.7 * max(abs(vmin), abs(vmax)) else "white",
                            fontsize=9, fontweight="bold")
                else:
                    ax.text(j, i, "n/a", ha="center", va="center",
                            color=THEME["subtext"], fontsize=8)

        ax.set_xticks(range(len(encoders)))
        ax.set_yticks(range(len(agents)))
        ax.set_xticklabels(encoders, color=THEME["text"], fontsize=10)
        ax.set_yticklabels(agents,   color=THEME["text"], fontsize=10)

        cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cb.ax.tick_params(colors=THEME["subtext"])
        cb.set_label(metric.capitalize(), color=THEME["text"])

        _label(ax,
               xlabel="Encoder",
               ylabel="Agent",
               title=f"Ablation Matrix — {metric} (final 10 eval episodes)\n"
                     f"{regime} regime")

        if save:
            return self._save(fig, f"ablation_{metric}_{regime}.png",
                              "w07_ablation")
        return fig

    # ------------------------------------------------------------------
    # CVaR efficient frontier (Week 9)
    # ------------------------------------------------------------------

    def plot_efficient_frontier(
        self,
        alphas:  list[float] = None,
        agent:   str = "qrdqn",
        encoder: str = "autoencoder",
        regime:  str = "high_vol",
        seed:    int = 42,
        save:    bool = True,
    ) -> plt.Figure:
        """
        Plot mean PnL vs CVaR_0.10 efficient frontier across α values.

        Parameters
        ----------
        alphas  : list of CVaR alpha values to plot
        agent   : agent name
        encoder : encoder name
        regime  : regime
        seed    : seed
        save    : save figure

        Returns
        -------
        matplotlib Figure
        """
        if alphas is None:
            alphas = [0.05, 0.10, 0.25, 0.50, 1.0]

        fig, ax = _dark_fig(figsize=(8, 5.5))

        mean_pnls = []
        cvar10s   = []
        labels    = []

        for alpha in alphas:
            run_tag = (
                f"{agent}_{encoder}_asymmetric_{regime}"
                f"_alpha{alpha}_seed{seed}"
            )
            run_dir = self.log_root / run_tag
            if not run_dir.exists():
                # Try without alpha in tag (older naming)
                run_tag = f"{agent}_{encoder}_asymmetric_{regime}_seed{seed}"
                run_dir = self.log_root / run_tag

            if not run_dir.exists():
                print(f"  [skip] {run_tag}")
                continue

            try:
                df = load_eval_history(run_dir)
            except FileNotFoundError:
                continue

            if "final_pnl" not in df.columns:
                continue

            # CVaR_0.10 across eval episodes: worst 10% of episode PnLs
            pnls       = df["final_pnl"].values
            n_tail     = max(1, int(0.10 * len(pnls)))
            cvar_val   = float(np.sort(pnls)[:n_tail].mean())
            mean_pnl   = float(pnls.mean())

            mean_pnls.append(mean_pnl)
            cvar10s.append(cvar_val)
            labels.append(f"α={alpha}")

        if not mean_pnls:
            print("No frontier data found. Run CVaR α-sweep first.")
            plt.close(fig)
            return fig

        mean_pnls = np.array(mean_pnls)
        cvar10s   = np.array(cvar10s)

        # Scatter with labels
        scatter = ax.scatter(cvar10s, mean_pnls,
                             c=np.arange(len(alphas[:len(mean_pnls)])),
                             cmap="plasma", s=120, zorder=5)

        for i, label in enumerate(labels):
            ax.annotate(label, (cvar10s[i], mean_pnls[i]),
                        textcoords="offset points", xytext=(8, 4),
                        color=THEME["text"], fontsize=9)

        # Connect points to show frontier
        order = np.argsort(cvar10s)
        ax.plot(cvar10s[order], mean_pnls[order],
                color=THEME["subtext"], lw=1.0, ls="--", alpha=0.6, zorder=3)

        ax.axhline(0, color=THEME["subtext"], lw=0.6, ls=":", alpha=0.4)

        _label(ax,
               xlabel="CVaR₀.₁₀  (worst 10% episode PnL)",
               ylabel="Mean Episode PnL",
               title=f"CVaR Efficient Frontier — {agent.upper()} + {encoder}\n"
                     f"{regime} regime")

        if save:
            return self._save(fig, f"efficient_frontier_{agent}_{regime}.png",
                              "w09_frontier")
        return fig

    # ------------------------------------------------------------------
    # Flash crash stress test (Week 9)
    # ------------------------------------------------------------------

    def plot_flash_crash(
        self,
        agent_pnl_data: dict[str, np.ndarray],
        crash_step:     int = 150,
        save:           bool = True,
    ) -> plt.Figure:
        """
        Plot cumulative PnL through flash crash for multiple agents.

        Parameters
        ----------
        agent_pnl_data : dict mapping agent_name → cum_pnl array shape (T,)
        crash_step     : step at which crash begins (vertical line)
        save           : save figure

        Returns
        -------
        matplotlib Figure
        """
        fig, ax = _dark_fig(figsize=(11, 5))

        for agent_name, cum_pnl in agent_pnl_data.items():
            color = AGENT_COLORS.get(agent_name, "#ffffff")
            ax.plot(cum_pnl, color=color, lw=2.0, label=agent_name)

        # Crash annotation
        ax.axvline(crash_step, color="#f87171", lw=1.5, ls="--", alpha=0.8)
        ax.annotate("Flash crash", xy=(crash_step, ax.get_ylim()[0]),
                    xytext=(crash_step + 5, ax.get_ylim()[0] * 0.8),
                    color="#f87171", fontsize=9,
                    arrowprops=dict(arrowstyle="->", color="#f87171", lw=0.8))

        ax.axhline(0, color=THEME["subtext"], lw=0.6, ls=":", alpha=0.4)

        _label(ax,
               xlabel="Step",
               ylabel="Cumulative PnL  ($)",
               title="Flash Crash Stress Test — DQN vs QR-DQN(α=0.05)\n"
                     "Agents trained on low_vol, tested on flash_crash (OOD)")
        _legend(ax)

        if save:
            return self._save(fig, "flash_crash_stress.png", "w09_frontier")
        return fig

    # ------------------------------------------------------------------
    # PCA of AE latent space (Week 8)
    # ------------------------------------------------------------------

    def plot_latent_space_pca(
        self,
        latent_vectors: np.ndarray,
        labels:         np.ndarray,
        label_names:    list[str] = None,
        save:           bool = True,
    ) -> plt.Figure:
        """
        PCA scatter of AE encoder latent vectors, coloured by regime/label.

        Parameters
        ----------
        latent_vectors : np.ndarray shape (N, latent_dim)
        labels         : np.ndarray shape (N,) — integer class labels
        label_names    : list of label strings (e.g. regime names)
        save           : save figure

        Returns
        -------
        matplotlib Figure
        """
        from sklearn.decomposition import PCA

        pca   = PCA(n_components=2)
        z_2d  = pca.fit_transform(latent_vectors)
        ev    = pca.explained_variance_ratio_

        fig, ax = _dark_fig(figsize=(8, 6))

        unique_labels = np.unique(labels)
        palette       = ["#f59e0b", "#06b6d4", "#34d399", "#f87171", "#a78bfa"]

        for i, lab in enumerate(unique_labels):
            mask  = labels == lab
            name  = label_names[lab] if label_names and lab < len(label_names) else str(lab)
            color = palette[i % len(palette)]
            ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                       c=color, label=name, alpha=0.5, s=15, edgecolors="none")

        _label(ax,
               xlabel=f"PC1 ({ev[0]:.1%} var)",
               ylabel=f"PC2 ({ev[1]:.1%} var)",
               title="AE Encoder Latent Space (PCA)\n"
                     "Clusters should correspond to market regimes")
        _legend(ax)

        if save:
            return self._save(fig, "latent_space_pca.png", "w07_ablation")
        return fig

    # ------------------------------------------------------------------
    # OOD transfer degradation bar chart (Week 9)
    # ------------------------------------------------------------------

    def plot_ood_transfer(
        self,
        transfer_data: dict[str, dict[str, float]],
        metric:        str = "sharpe",
        save:          bool = True,
    ) -> plt.Figure:
        """
        Bar chart of in-distribution vs OOD performance for each agent.

        Parameters
        ----------
        transfer_data : dict mapping agent_name →
                        {"in_dist": float, "ood": float}
        metric        : metric name for axis label
        save          : save figure

        Returns
        -------
        matplotlib Figure
        """
        fig, ax = _dark_fig(figsize=(9, 5))

        agents     = list(transfer_data.keys())
        in_vals    = [transfer_data[a]["in_dist"] for a in agents]
        ood_vals   = [transfer_data[a]["ood"] for a in agents]
        degrade    = [(i - o) / (abs(i) + 1e-10) for i, o in zip(in_vals, ood_vals)]

        x = np.arange(len(agents))
        w = 0.35

        for i, agent_name in enumerate(agents):
            color = AGENT_COLORS.get(agent_name, "#ffffff")
            ax.bar(x[i] - w/2, in_vals[i], w, color=color, alpha=0.9, label=agent_name if i == 0 else "")
            ax.bar(x[i] + w/2, ood_vals[i], w, color=color, alpha=0.4)

        # Degradation labels
        for i, (xi, deg) in enumerate(zip(x, degrade)):
            ax.text(xi, max(in_vals[i], ood_vals[i]) + 0.01,
                    f"Δ{deg:.0%}", ha="center", va="bottom",
                    color=THEME["subtext"], fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(agents, color=THEME["text"], fontsize=10)
        ax.axhline(0, color=THEME["subtext"], lw=0.5, alpha=0.4)

        from matplotlib.patches import Patch
        legend_els = [
            Patch(facecolor="#888888", alpha=0.9, label="In-distribution"),
            Patch(facecolor="#888888", alpha=0.4, label="OOD"),
        ]
        ax.legend(handles=legend_els, framealpha=0.15,
                  facecolor=THEME["bg"], edgecolor=THEME["border"],
                  labelcolor=THEME["text"], fontsize=9)

        _label(ax,
               xlabel="Agent",
               ylabel=metric.capitalize(),
               title=f"OOD Transfer Degradation — trained low_vol, tested high_vol\n"
                     f"Δ = (in_dist − OOD) / |in_dist|")

        if save:
            return self._save(fig, f"ood_transfer_{metric}.png", "w09_frontier")
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# Standalone helpers (called directly from notebooks without Visualizer)
# ══════════════════════════════════════════════════════════════════════════════

def quote_skew_curve(
    inventories:     np.ndarray,
    bid_offsets:     np.ndarray,
    min_visits:      int = 5,
    boundary_buffer: int = 2,
    q_max:           int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute mean ± std bid offset per inventory level.

    Filters out inventory levels near ±Q_max (boundary fallback artifacts)
    and levels with fewer than min_visits observations.

    Parameters
    ----------
    inventories     : np.ndarray shape (N,) — inventory at each step
    bid_offsets     : np.ndarray shape (N,) — bid tick offset at each step
    min_visits      : int — minimum observations per level
    boundary_buffer : int — exclude levels within N of ±Q_max
    q_max           : int — inventory constraint

    Returns
    -------
    (inv_levels, mean_offsets, std_offsets) : each shape (K,)
    """
    groups = defaultdict(list)
    for inv, off in zip(inventories, bid_offsets):
        inv = int(inv)
        if abs(inv) <= q_max - boundary_buffer:
            groups[inv].append(float(off))

    inv_levels   = sorted(k for k in groups if len(groups[k]) >= min_visits)
    mean_offsets = np.array([np.mean(groups[k]) for k in inv_levels])
    std_offsets  = np.array([np.std(groups[k])  for k in inv_levels])

    return np.array(inv_levels), mean_offsets, std_offsets


def smoothed(values: np.ndarray, window: int = 20) -> np.ndarray:
    """Uniform moving average — for quick notebook use."""
    return np.convolve(values, np.ones(window) / window, mode="valid")