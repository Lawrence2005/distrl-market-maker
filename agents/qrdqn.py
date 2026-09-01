"""
agents/qrdqn.py

Quantile Regression DQN (QR-DQN) agent with CVaR action selection.

Extends the DRQN-LSTM backbone (RecurrentBase) with a quantile projection
head that outputs a full return distribution Z(s,a) rather than a scalar
Q(s,a). The distributional output enables CVaR-based action selection,
which is the core research contribution of this project.

Architecture
------------
    obs → Encoder → latent_z → LSTM → h_t → QuantileHead → Z(s,a)

    QuantileHead (replaces DuelingHead from DQN):
        Linear(hidden_dim → hidden_dim//2) → ReLU → Linear(n_actions * n_quantiles)
        Reshape → (B, n_actions, n_quantiles)

    With dueling decomposition (Wang et al. 2016):
        Value stream:     → (B, 1,         n_quantiles)
        Advantage stream: → (B, n_actions, n_quantiles)
        Z(s,a) = V + A - mean_a(A)   per quantile

QR-DQN Loss (Dabney et al. 2017)
----------------------------------
Quantile regression loss with asymmetric Huber kernel:

    L = (1/N) Σ_i Σ_j ρ_{τ_i}^κ (r + γ·Z_j(s',a*) - Z_i(s,a))

where:
    τ_i = (2i-1)/(2N)  for i=1,...,N   (fixed quantile levels)
    ρ_τ^κ(u) = |τ - 1{u<0}| · L_κ(u)  (asymmetric Huber)
    L_κ(u)   = u²/2 if |u|≤κ else κ(|u|-κ/2)
    a*        = argmax_a Σ_j Z_j(s',a)  (greedy next action from online net)

CVaR Action Selection
----------------------
Instead of argmax over mean(Z(s,a)), select action that maximises
Conditional Value-at-Risk at level alpha:

    CVaR_α(Z(s,a)) = (1/⌊αN⌋) Σ_{i=1}^{⌊αN⌋} Z_{(i)}(s,a)

where Z_{(i)} are the sorted (ascending) quantiles.
alpha=0.25 → average over the worst 25% of outcomes.

This makes the agent risk-averse: it prefers actions with better
worst-case returns even at the cost of lower expected return.

Parameters (from configs/agent/qrdqn.yaml)
-------------------------------------------
n_quantiles       : int   — number of quantile atoms N (default 200)
hidden_dim        : int   — LSTM hidden size (default 256)
dueling           : bool  — use dueling decomposition (default True)
lr                : float — Adam learning rate (default 1e-4)
batch_size        : int   — sequences per gradient step (default 256)
gamma             : float — discount factor (default 0.99)
target_update_freq: int   — hard target update interval (default 1000)
cvar_alpha        : float — CVaR tail fraction for action selection (default 0.25)
kappa             : float — Huber loss threshold (default 1.0)
prioritized_replay: bool  — use PER (default True)

Reference
---------
Dabney et al. (2017) — Distributional Reinforcement Learning with
    Quantile Regression. AAAI 2018.
Wang et al. (2016)   — Dueling Network Architectures for Deep RL.
Rockafellar & Uryasev (2000) — CVaR optimisation.

Week 5 deliverable.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base import AgentBase
from agents.recurrent_base import RecurrentBase
from training.replay_buffer import ReplayBuffer


# ── Quantile head ─────────────────────────────────────────────────────────────

class QuantileHead(nn.Module):
    """
    Dueling quantile projection head for QR-DQN.

    Outputs a full return distribution Z(s,a) with shape
    (B, n_actions, n_quantiles) using the dueling decomposition
    applied independently per quantile level.

    Parameters
    ----------
    input_dim   : int — LSTM hidden state dimension
    n_actions   : int — flat action space size
    n_quantiles : int — number of quantile atoms
    dueling     : bool — use value/advantage decomposition (default True)
    """

    def __init__(
        self,
        input_dim:   int,
        n_actions:   int,
        n_quantiles: int,
        dueling:     bool = True,
    ):
        super().__init__()

        self.n_actions   = n_actions
        self.n_quantiles = n_quantiles
        self.dueling     = dueling
        mid = max(input_dim // 2, n_actions)

        if dueling:
            # Value stream: scalar per quantile
            self.value_stream = nn.Sequential(
                nn.Linear(input_dim, mid),
                nn.ReLU(),
                nn.Linear(mid, n_quantiles),         # (B, n_quantiles)
            )
            # Advantage stream: one value per action per quantile
            self.advantage_stream = nn.Sequential(
                nn.Linear(input_dim, mid),
                nn.ReLU(),
                nn.Linear(mid, n_actions * n_quantiles),  # (B, n_actions*n_quantiles)
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(input_dim, mid),
                nn.ReLU(),
                nn.Linear(mid, n_actions * n_quantiles),
            )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h : Tensor shape (B, input_dim)

        Returns
        -------
        Tensor shape (B, n_actions, n_quantiles) — return distribution Z(s,a)
        """
        B = h.shape[0]

        if self.dueling:
            V = self.value_stream(h)                          # (B, n_quantiles)
            A = self.advantage_stream(h)                      # (B, n_actions*n_quantiles)
            A = A.view(B, self.n_actions, self.n_quantiles)   # (B, n_actions, n_quantiles)
            V = V.unsqueeze(1)                                 # (B, 1, n_quantiles)
            # Dueling: Z = V + A - mean_a(A), applied per quantile
            Z = V + A - A.mean(dim=1, keepdim=True)
        else:
            Z = self.proj(h).view(B, self.n_actions, self.n_quantiles)

        return Z


# ── QR-DQN agent ─────────────────────────────────────────────────────────────

class QRDQNAgent(AgentBase):
    """
    QR-DQN agent with DRQN-LSTM backbone and CVaR action selection.

    Parameters
    ----------
    encoder            : nn.Module — any encoder with .latent_dim property
    n_actions          : int       — flat action space size
    n_quantiles        : int       — quantile atoms N (default 200)
    hidden_dim         : int       — LSTM hidden size (default 128)
    dueling            : bool      — dueling decomposition (default True)
    lr                 : float     — Adam learning rate (default 1e-4)
    gamma              : float     — discount factor (default 0.99)
    batch_size         : int       — sequences per update (default 256)
    seq_len            : int       — DRQN sequence length (default 30)
    target_update_freq : int       — target net update interval (default 1000)
    cvar_alpha         : float     — CVaR tail fraction (default 0.25)
    kappa              : float     — Huber loss threshold (default 1.0)
    epsilon_start      : float     — initial ε (default 1.0)
    epsilon_end        : float     — minimum ε (default 0.05)
    epsilon_decay_steps: int       — ε decay steps (default 50_000)
    buffer_capacity    : int       — replay buffer size (default 100_000)
    prioritized        : bool      — use PER (default True)
    device             : str       — 'cpu' or 'cuda'
    """

    name = "QR-DQN"

    def __init__(
        self,
        encoder:             nn.Module,
        n_actions:           int,
        n_quantiles:         int   = 200,
        hidden_dim:          int   = 128,
        dueling:             bool  = True,
        lr:                  float = 1e-4,
        gamma:               float = 0.99,
        batch_size:          int   = 256,
        seq_len:             int   = 30,
        target_update_freq:  int   = 1000,
        cvar_alpha:          float = 0.25,
        kappa:               float = 1.0,
        epsilon_start:       float = 1.0,
        epsilon_end:         float = 0.05,
        epsilon_decay_steps: int   = 50_000,
        buffer_capacity:     int   = 100_000,
        prioritized:         bool  = True,
        use_lstm:            bool  = True,
        device:              str   = "cpu",
    ):
        self.n_actions           = n_actions
        self.n_quantiles         = n_quantiles
        self.gamma               = gamma
        self.batch_size          = batch_size
        self.seq_len             = seq_len
        self.target_update_freq  = target_update_freq
        self.cvar_alpha          = cvar_alpha
        self.kappa               = kappa
        self.epsilon_start       = epsilon_start
        self.epsilon_end         = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.device              = torch.device(device)

        self.is_online = False

        # Fixed quantile levels τ_i = (2i-1)/(2N), i=1,...,N
        self.taus = torch.FloatTensor(
            [(2 * i - 1) / (2 * n_quantiles) for i in range(1, n_quantiles + 1)]
        ).to(self.device)   # shape (N,)

        # Number of tail quantiles for CVaR
        self._n_tail = max(1, int(cvar_alpha * n_quantiles))

        # ── Networks ──────────────────────────────────────────────────
        # RecurrentBase with dueling=False — we add our own quantile head
        self.online_base = RecurrentBase(
            encoder    = encoder,
            n_actions  = n_actions,
            hidden_dim = hidden_dim,
            dueling    = False,   # head added separately below
            use_lstm   = use_lstm,
        ).to(self.device)

        self.online_head = QuantileHead(
            input_dim   = hidden_dim,
            n_actions   = n_actions,
            n_quantiles = n_quantiles,
            dueling     = dueling,
        ).to(self.device)

        self.target_base = RecurrentBase(
            encoder    = copy.deepcopy(encoder),
            n_actions  = n_actions,
            hidden_dim = hidden_dim,
            dueling    = False,
            use_lstm   = use_lstm,
        ).to(self.device)
        self.target_base.load_state_dict(self.online_base.state_dict())

        self.target_head = QuantileHead(
            input_dim   = hidden_dim,
            n_actions   = n_actions,
            n_quantiles = n_quantiles,
            dueling     = dueling,
        ).to(self.device)
        self.target_head.load_state_dict(self.online_head.state_dict())

        self.target_base.eval()
        self.target_head.eval()
        for p in list(self.target_base.parameters()) + list(self.target_head.parameters()):
            p.requires_grad = False

        # ── Optimiser — joint online_base + online_head ───────────────
        self.optimiser = torch.optim.Adam(
            list(self.online_base.parameters()) +
            list(self.online_head.parameters()),
            lr = lr,
        )

        # ── Replay buffer ─────────────────────────────────────────────
        self.buffer = ReplayBuffer(
            capacity    = buffer_capacity,
            prioritized = prioritized,
            seq_len     = seq_len,
        )

        # ── Counters ──────────────────────────────────────────────────
        self._steps   = 0
        self._updates = 0

    # ------------------------------------------------------------------
    # Exploration schedule
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        frac = min(1.0, self._steps / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _online_forward(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full online network forward pass.

        Parameters
        ----------
        obs : Tensor shape (B, T, obs_dim)

        Returns
        -------
        Z : Tensor shape (B, T, n_actions, n_quantiles)
        """
        lstm_out, _ = self.online_base.forward(obs)   # (B, T, hidden_dim)
        B, T, H = lstm_out.shape
        Z = self.online_head(lstm_out.reshape(B * T, H))  # (B*T, n_actions, N)
        return Z.reshape(B, T, self.n_actions, self.n_quantiles)

    def _target_forward(
        self,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """Target network forward — no gradient."""
        lstm_out, _ = self.target_base.forward(obs)
        B, T, H = lstm_out.shape
        Z = self.target_head(lstm_out.reshape(B * T, H))
        return Z.reshape(B, T, self.n_actions, self.n_quantiles)

    # ------------------------------------------------------------------
    # CVaR action selection
    # ------------------------------------------------------------------

    def _cvar_action(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Select action maximising CVaR_alpha of the return distribution.

        CVaR_α(Z(s,a)) = mean of bottom α fraction of quantiles.
        With alpha=0.25 and N=200: average over the 50 lowest quantiles.

        Parameters
        ----------
        Z : Tensor shape (B, n_actions, n_quantiles) — sorted ascending

        Returns
        -------
        Tensor shape (B,) — action indices
        """
        Z_sorted = Z.sort(dim=-1).values                     # (B, n_actions, N)
        cvar     = Z_sorted[:, :, :self._n_tail].mean(dim=-1) # (B, n_actions)
        return cvar.argmax(dim=-1)                            # (B,)

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(
        self,
        obs:    np.ndarray,
        greedy: bool = False,
    ) -> int:
        """
        Select action using ε-greedy with CVaR policy.

        During exploration (ε-greedy): random action.
        During exploitation: argmax CVaR_α(Z(s,a)).

        Parameters
        ----------
        obs    : np.ndarray shape (obs_dim,)
        greedy : bool — skip ε-greedy, always use CVaR policy

        Returns
        -------
        int — flat action index
        """
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        x = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)  # (1, obs_dim)

        # Single-step inference via LSTM step()
        lstm_h, _ = self.online_base.forward(x.unsqueeze(1))  # (1, 1, hidden_dim)
        Z = self.online_head(lstm_h[:, -1, :])                 # (1, n_actions, N)

        return int(self._cvar_action(Z).item())

    def reset_hidden(self, batch_size: int = 1) -> None:
        """Reset LSTM hidden state. Call at episode start."""
        self.online_base.reset_hidden(batch_size=batch_size, device=self.device)
        self.target_base.reset_hidden(batch_size=batch_size, device=self.device)

    # ------------------------------------------------------------------
    # Experience storage
    # ------------------------------------------------------------------

    def observe(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        """Store transition and increment step counter."""
        self.buffer.push(obs, action, reward, next_obs, done)
        self._steps += 1

    # ------------------------------------------------------------------
    # QR-DQN loss
    # ------------------------------------------------------------------

    @staticmethod
    def _quantile_huber_loss(
        pred:  torch.Tensor,
        target: torch.Tensor,
        taus:  torch.Tensor,
        kappa: float = 1.0,
    ) -> torch.Tensor:
        """
        Quantile regression loss with asymmetric Huber kernel.

        L = mean_i mean_j |τ_i - 1{δ_ij < 0}| · L_κ(δ_ij)

        where δ_ij = target_j - pred_i  (TD error)
              L_κ  = Huber loss with threshold κ

        Parameters
        ----------
        pred   : Tensor shape (B, N)  — predicted quantiles for taken action
        target : Tensor shape (B, N)  — target quantile distribution
        taus   : Tensor shape (N,)    — quantile levels τ_i
        kappa  : float                — Huber threshold

        Returns
        -------
        Tensor scalar — mean quantile regression loss
        """
        B, N  = pred.shape
        _, Nt = target.shape

        # TD errors: (B, N, Nt)
        pred_tile   = pred.unsqueeze(2)    # (B, N, 1)
        target_tile = target.unsqueeze(1)  # (B, 1, Nt)
        td          = target_tile - pred_tile  # (B, N, Nt)

        # Huber loss element-wise
        huber = torch.where(
            td.abs() <= kappa,
            0.5 * td ** 2,
            kappa * (td.abs() - 0.5 * kappa),
        )

        # Asymmetric weighting
        taus_tile = taus.view(1, N, 1)             # (1, N, 1)
        indicator = (td.detach() < 0).float()      # (B, N, Nt)
        weights   = (taus_tile - indicator).abs()  # (B, N, Nt)

        # Mean over target quantiles, sum over pred quantiles, mean over batch
        loss = (weights * huber).mean(dim=2).sum(dim=1).mean()
        return loss

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self) -> Optional[float]:
        """
        Sample a sequence batch and perform one QR-DQN gradient update.

        Returns
        -------
        float | None — loss if update performed, None if buffer not ready
        """
        if not self.buffer.is_ready(self.batch_size):
            return None

        batch, indices, weights = self.buffer.sample_sequences(self.batch_size)

        obs      = torch.from_numpy(batch["obs"]).float().to(self.device)
        actions  = torch.from_numpy(batch["action"]).long().to(self.device)
        rewards  = torch.from_numpy(batch["reward"]).float().to(self.device)
        next_obs = torch.from_numpy(batch["next_obs"]).float().to(self.device)
        dones    = torch.from_numpy(batch["done"]).float().to(self.device)
        is_w     = torch.from_numpy(weights).float().to(self.device)

        B, T, _ = obs.shape

        # ── Online Z(s,a) at last step ────────────────────────────────
        self.online_base.reset_hidden(batch_size=B, device=self.device)
        Z_all  = self._online_forward(obs)             # (B, T, n_actions, N)
        Z_last = Z_all[:, -1, :, :]                    # (B, n_actions, N)

        # Gather quantiles for taken action
        a_last = actions[:, -1]                        # (B,)
        a_idx  = a_last.view(B, 1, 1).expand(B, 1, self.n_quantiles)
        Z_sa   = Z_last.gather(1, a_idx).squeeze(1)   # (B, N)

        # ── Target Z(s', a*) at last step ─────────────────────────────
        with torch.no_grad():
            self.target_base.reset_hidden(batch_size=B, device=self.device)
            Z_next_all  = self._target_forward(next_obs)
            Z_next_last = Z_next_all[:, -1, :, :]

            # Double DQN: greedy next action from online net
            self.online_base.reset_hidden(batch_size=B, device=self.device)
            Z_online_next = self._online_forward(next_obs)[:, -1, :, :]
            a_next = Z_online_next.mean(dim=-1).argmax(dim=-1)

            a_next_idx = a_next.view(B, 1, 1).expand(B, 1, self.n_quantiles)
            Z_next_sa  = Z_next_last.gather(1, a_next_idx).squeeze(1)

            r     = rewards[:, -1].unsqueeze(1)
            d     = dones[:, -1].unsqueeze(1)
            Z_tgt = r + self.gamma * Z_next_sa * (1.0 - d)

        # ── Quantile regression loss ───────────────────────────────────
        element_loss = self._quantile_huber_loss(
            Z_sa, Z_tgt, self.taus, self.kappa
        )
        loss = (is_w * element_loss).mean() if is_w.shape == (B,) else element_loss

        # ── Gradient update ───────────────────────────────────────────
        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.online_base.parameters()) +
            list(self.online_head.parameters()),
            max_norm=10.0,
        )
        self.optimiser.step()

        # ── PER priority update ───────────────────────────────────────
        td_errors = (Z_sa - Z_tgt).abs().mean(dim=-1).detach().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors + 1e-6)

        # ── Target network hard update ────────────────────────────────
        self._updates += 1
        if self._updates % self.target_update_freq == 0:
            self.target_base.load_state_dict(self.online_base.state_dict())
            self.target_head.load_state_dict(self.online_head.state_dict())

        # ── Restore hidden state for single-step inference ────────────
        self.online_base.reset_hidden(batch_size=1, device=self.device)
        self.target_base.reset_hidden(batch_size=1, device=self.device)

        return float(loss.item())

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "online_base":  self.online_base.state_dict(),
            "online_head":  self.online_head.state_dict(),
            "target_base":  self.target_base.state_dict(),
            "target_head":  self.target_head.state_dict(),
            "optimiser":    self.optimiser.state_dict(),
            "steps":        self._steps,
            "updates":      self._updates,
            "beta":         self.buffer.beta if self.buffer.prioritized else None,
        }

    def load_state_dict(self, state: dict) -> None:
        self.online_base.load_state_dict(state["online_base"])
        self.online_head.load_state_dict(state["online_head"])
        self.target_base.load_state_dict(state["target_base"])
        self.target_head.load_state_dict(state["target_head"])
        self.optimiser.load_state_dict(state["optimiser"])
        self._steps   = state["steps"]
        self._updates = state["updates"]
        if self.buffer.prioritized and state["beta"] is not None:
            self.buffer.beta = state["beta"]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"QRDQNAgent("
            f"n_actions={self.n_actions}, "
            f"n_quantiles={self.n_quantiles}, "
            f"cvar_alpha={self.cvar_alpha}, "
            f"epsilon={self.epsilon:.3f}, "
            f"steps={self._steps})"
        )