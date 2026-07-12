"""
agents/cvar_policy.py

CVaR risk measure wrapper for distributional RL agents (QR-DQN, IQN).

Applies Conditional Value-at-Risk (CVaR) at the POLICY SELECTION level only.
Does NOT modify the Bellman update or the quantile regression loss.

Design
------
Wraps any distributional agent (QRDQNAgent or IQNAgent) and overrides
action selection to use CVaR instead of mean. The wrapped agent handles:
    - Training (Bellman update, replay buffer, quantile loss) — unchanged
    - Representation (encoder, LSTM, quantile head) — unchanged

The wrapper handles:
    - Policy selection: argmax_a CVaR_α(Z(s,a))
    - Risk measure computation from the quantile distribution

This separation enables the ablation:
    "Does CVaR at policy selection add value over mean,
     independent of the distributional Bellman update?"

Run as:
    policy=mean  → standard argmax over mean quantile
    policy=cvar  → argmax over CVaR_α (this wrapper)

CVaR definition
---------------
CVaR_α(Z) = (1/α) · E[Z · 1{Z ≤ VaR_α(Z)}]

For a discrete quantile distribution with N atoms:
    CVaR_α(Z(s,a)) = (1/⌊αN⌋) · Σ_{i=1}^{⌊αN⌋} Z_{(i)}(s,a)

where Z_{(i)} are the sorted (ascending) quantile values.
alpha=0.25 → average over the worst 25% of outcomes.

Relationship to existing agents
--------------------------------
QRDQNAgent._cvar_action() and IQNAgent._cvar_action() implement CVaR
internally with a fixed alpha. This wrapper:
    1. Extracts that logic into a standalone, swappable class
    2. Makes alpha configurable at the wrapper level independently of training
    3. Adds mean and VaR policy alternatives for ablation comparison
    4. Exposes a clean `policy_name` attribute for logging

Usage
-----
    from agents.qrdqn import QRDQNAgent
    from agents.cvar_policy import CVaRPolicy

    base_agent  = QRDQNAgent(encoder=enc, n_actions=121, ...)
    cvar_agent  = CVaRPolicy(agent=base_agent, alpha=0.25)

    # act() uses CVaR selection
    action = cvar_agent.act(obs)

    # Training is unchanged — delegated to base agent
    cvar_agent.observe(obs, action, reward, next_obs, done)
    loss = cvar_agent.train_step()

    # All other agent attributes pass through transparently
    cvar_agent.reset_hidden()
    cvar_agent.state_dict()

Supported risk measures
-----------------------
    'cvar'  — CVaR_α: mean of bottom α quantiles  (default)
    'mean'  — E[Z]:   mean over all quantiles      (risk-neutral baseline)
    'var'   — VaR_α:  the α-quantile value         (Value-at-Risk)

Reference
---------
Rockafellar, R. & Uryasev, S. (2000).
    Optimization of Conditional Value-at-Risk. J. of Risk.

Dabney et al. (2018). Implicit Quantile Networks for Distributional RL.
    ICML 2018 — Section 4.2 (risk-sensitive policy evaluation).

Week 5 deliverable.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from agents.qrdqn import QRDQNAgent
from agents.iqn import IQNAgent


# ── Risk measure functions ────────────────────────────────────────────────────

def cvar_action(Z: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Select action maximising CVaR_α over the quantile distribution.

    CVaR_α(Z(s,a)) = mean of bottom α fraction of quantile values.

    Parameters
    ----------
    Z     : Tensor shape (B, n_quantiles, n_actions) for QR-DQN
              OR     shape (B, K, n_actions)          for IQN
    alpha : float — tail fraction ∈ (0, 1]

    Returns
    -------
    Tensor shape (B,) — action indices
    """
    n_q    = Z.shape[1]
    n_tail = max(1, int(alpha * n_q))

    Z_sorted = Z.sort(dim=1).values           # (B, n_q, n_actions) sorted ascending
    cvar     = Z_sorted[:, :n_tail, :].mean(dim=1)  # (B, n_actions)
    return cvar.argmax(dim=-1)                 # (B,)


def mean_action(Z: torch.Tensor) -> torch.Tensor:
    """
    Select action maximising E[Z] — risk-neutral baseline.

    Parameters
    ----------
    Z : Tensor shape (B, n_quantiles, n_actions)

    Returns
    -------
    Tensor shape (B,) — action indices
    """
    return Z.mean(dim=1).argmax(dim=-1)        # (B,)


def var_action(Z: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Select action maximising VaR_α (the α-quantile value).

    VaR_α(Z(s,a)) = Z_{(⌊αN⌋)}(s,a) — the α-th order statistic.

    Parameters
    ----------
    Z     : Tensor shape (B, n_quantiles, n_actions)
    alpha : float — quantile level ∈ (0, 1]

    Returns
    -------
    Tensor shape (B,) — action indices
    """
    n_q   = Z.shape[1]
    idx   = max(0, int(alpha * n_q) - 1)
    Z_sorted = Z.sort(dim=1).values           # ascending
    var   = Z_sorted[:, idx, :]               # (B, n_actions)
    return var.argmax(dim=-1)


_RISK_MEASURES = {
    "cvar": cvar_action,
    "mean": mean_action,
    "var":  var_action,
}


# ── CVaRPolicy wrapper ────────────────────────────────────────────────────────

class CVaRPolicy:
    """
    Risk-sensitive policy wrapper for distributional RL agents.

    Wraps QRDQNAgent or IQNAgent and overrides action selection to use
    a configurable risk measure. All training operations are delegated
    to the base agent unchanged.

    Parameters
    ----------
    agent  : QRDQNAgent | IQNAgent — base distributional agent
    alpha  : float — risk level ∈ (0, 1] (default 0.25)
    measure: str   — risk measure: 'cvar' | 'mean' | 'var' (default 'cvar')
    """

    def __init__(
        self,
        agent:   QRDQNAgent | IQNAgent,
        alpha:   float = 0.25,
        measure: str   = "cvar",
    ):
        assert isinstance(agent, (QRDQNAgent, IQNAgent)), (
            f"CVaRPolicy requires QRDQNAgent or IQNAgent, got {type(agent).__name__}. "
            f"Only distributional agents have quantile distributions to apply CVaR to."
        )
        assert 0.0 < alpha <= 1.0, f"alpha must be in (0, 1], got {alpha}"
        assert measure in _RISK_MEASURES, (
            f"Unknown risk measure '{measure}'. "
            f"Choose from: {list(_RISK_MEASURES.keys())}"
        )

        self._agent   = agent
        self.alpha    = alpha
        self.measure  = measure
        self._risk_fn = _RISK_MEASURES[measure]

        self._is_qrdqn = isinstance(agent, QRDQNAgent)
        self._is_iqn   = isinstance(agent, IQNAgent)

    # ------------------------------------------------------------------
    # Policy selection (overridden)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        obs:    np.ndarray,
        greedy: bool = False,
    ) -> int:
        """
        Select action using the configured risk measure.

        Overrides the base agent's act() to use CVaR/mean/VaR
        instead of the internal default (which varies by agent).

        Parameters
        ----------
        obs    : np.ndarray shape (obs_dim,) or (snapshot_dim,)
        greedy : bool — if True, skip ε-greedy (always use risk measure)

        Returns
        -------
        int — flat action index
        """
        # ε-greedy exploration (delegated to base agent's epsilon schedule)
        if not greedy and np.random.rand() < self._agent.epsilon:
            return np.random.randint(self._agent.n_actions)

        device = self._agent.device
        x      = torch.from_numpy(obs).float().unsqueeze(0).to(device)

        # Get quantile distribution Z(s, ·) from base agent
        Z = self._get_z(x)   # (1, n_quantiles, n_actions)

        # Apply risk measure
        if self.measure == "cvar":
            action = cvar_action(Z, self.alpha)
        elif self.measure == "mean":
            action = mean_action(Z)
        else:
            action = var_action(Z, self.alpha)

        return int(action.item())

    def _get_z(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get quantile distribution Z(s, ·) from base agent.

        Returns Tensor shape (1, n_quantiles_or_K, n_actions).

        Handles both QR-DQN (fixed quantiles) and IQN (sampled quantiles).
        """
        if self._is_qrdqn:
            # QR-DQN: fixed N quantile levels
            lstm_out, _ = self._agent.online_base.forward(x.unsqueeze(1))
            h_last      = lstm_out[:, -1, :]                      # (1, hidden_dim)
            Z           = self._agent.online_head(h_last)          # (1, n_actions, N)
            # Reorder to (B, N, n_actions) for consistent interface
            return Z.permute(0, 2, 1)                             # (1, N, n_actions)

        if self._is_iqn:
            # IQN: sample K quantile levels from lower tail for CVaR
            K   = self._agent.n_quantile_samples
            lstm_out, _ = self._agent.online_base.forward(x.unsqueeze(1))
            h_last      = lstm_out[:, -1, :]                      # (1, hidden_dim)
            tau         = self._agent._sample_tau(1, K)           # (1, K)
            Z           = self._agent.online_head(h_last, tau)    # (1, K, n_actions)
            return Z                                               # (1, K, n_actions)

        raise RuntimeError("Base agent is neither QRDQNAgent nor IQNAgent")

    # ------------------------------------------------------------------
    # Training delegation (unchanged)
    # ------------------------------------------------------------------

    def observe(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        """Delegate to base agent's replay buffer."""
        self._agent.observe(obs, action, reward, next_obs, done)

    def train_step(self) -> Optional[float]:
        """Delegate to base agent's training step (Bellman update unchanged)."""
        return self._agent.train_step()

    def reset_hidden(self, batch_size: int = 1) -> None:
        """Delegate LSTM hidden state reset to base agent."""
        self._agent.reset_hidden(batch_size=batch_size)

    def state_dict(self) -> dict:
        """Include wrapper metadata alongside base agent state."""
        return {
            "base_agent":   self._agent.state_dict(),
            "alpha":        self.alpha,
            "measure":      self.measure,
            "agent_type":   type(self._agent).__name__,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore base agent state and wrapper config."""
        self._agent.load_state_dict(state["base_agent"])
        self.alpha   = state["alpha"]
        self.measure = state["measure"]

    # ------------------------------------------------------------------
    # Transparent attribute access
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        """
        Pass through all other attribute access to the base agent.

        This means cvar_agent.epsilon, cvar_agent.n_actions,
        cvar_agent.buffer, etc. all work transparently.
        """
        return getattr(self._agent, name)

    # ------------------------------------------------------------------
    # Identity / logging
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return f"{self._agent.name}+CVaR({self.measure},α={self.alpha})"

    @property
    def policy_name(self) -> str:
        """Short name for logging and plot labels."""
        if self.measure == "cvar":
            return f"CVaR(α={self.alpha})"
        if self.measure == "mean":
            return "Mean"
        return f"VaR(α={self.alpha})"

    def __repr__(self) -> str:
        return (
            f"CVaRPolicy("
            f"agent={self._agent.name}, "
            f"measure={self.measure}, "
            f"alpha={self.alpha})"
        )