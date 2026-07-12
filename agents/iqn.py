"""
agents/iqn.py

Implicit Quantile Network (IQN) agent with CVaR action selection.

Extends the DRQN-LSTM backbone with implicit quantile sampling. Unlike
QR-DQN which uses N fixed quantile levels as output heads, IQN embeds
sampled quantile levels τ ~ Uniform(0,1) as inputs to the network via
cosine embedding, learning a continuous mapping (s, a, τ) → Z.

Architecture
------------
    obs → Encoder → latent_z → LSTM → h_t ──────────────────┐
                                                              ⊙  → Linear → Z(s,a,τ)
    τ ~ Uniform(0,1)^K → CosineEmbedding → φ(τ) → Linear ──┘

    CosineEmbedding:
        φ(τ)_i = ReLU( Σ_{j=0}^{n-1} cos(π·j·τ) · w_j + b )
        input:  (B·K,)        — K sampled quantile levels per batch element
        output: (B·K, hidden_dim) — embedded and projected to match h_t

    Element-wise product h_t ⊙ φ(τ) then projects to n_actions.

    With dueling decomposition (applied per τ sample):
        Value stream:     (B·K, 1)
        Advantage stream: (B·K, n_actions)
        Z(s,a,τ) = V + A - mean_a(A)

IQN Loss
--------
Same quantile regression loss as QR-DQN but with sampled τ:

    L = (1/K) Σ_i Σ_j ρ_{τ_i}^κ (r + γ·Z(s',a*,τ_j') - Z(s,a,τ_i))

where τ_i ~ Uniform(0,1) (prediction quantiles)
      τ_j' ~ Uniform(0,1) (target quantiles, independently sampled)

CVaR Action Selection
---------------------
Sample K quantile levels, sort ascending, take bottom α fraction:

    CVaR_α(Z(s,a)) = (1/⌊αK⌋) Σ_{i=1}^{⌊αK⌋} Z(s,a,τ_{(i)})

where τ_{(i)} are order statistics of K uniform samples.
With α=0.25 and K=64: average over ~16 lowest sampled returns.

Key difference from QR-DQN
---------------------------
QR-DQN: fixed τ_i = (2i-1)/(2N), N separate output heads
IQN:    τ ~ Uniform(0,1) sampled per forward pass, single network
         → learns a continuous return distribution function
         → more expressive: can represent any distribution shape
         → more sample-efficient: each forward pass explores the full [0,1] range

Parameters (from configs/agent/iqn.yaml)
-----------------------------------------
n_quantile_samples : int   — K quantile levels sampled per forward pass (default 64)
embedding_dim      : int   — cosine embedding dimension n (default 64)
hidden_dim         : int   — LSTM hidden size (default 256)
dueling            : bool  — dueling decomposition per τ (default True)
lr                 : float — Adam learning rate (default 1e-4)
batch_size         : int   — sequences per gradient step (default 256)
gamma              : float — discount factor (default 0.99)
target_update_freq : int   — hard target update interval (default 1000)
cvar_alpha         : float — CVaR tail fraction (default 0.25)
kappa              : float — Huber loss threshold (default 1.0)

Reference
---------
Dabney et al. (2018) — Implicit Quantile Networks for Distributional RL.
    ICML 2018.
Dabney et al. (2017) — Distributional RL with Quantile Regression (QR-DQN).

Week 5 deliverable.
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.recurrent_base import RecurrentBase
from agents.qrdqn import QRDQNAgent  # reuse quantile_huber_loss
from training.replay_buffer import ReplayBuffer


# ── Cosine embedding ──────────────────────────────────────────────────────────

class CosineEmbedding(nn.Module):
    """
    Cosine embedding for quantile levels τ ∈ [0, 1].

    φ(τ) = ReLU( Σ_{j=0}^{n-1} cos(π·j·τ) · w_j + b )

    Maps a scalar τ to a vector of dimension output_dim that can be
    multiplied element-wise with the LSTM hidden state h_t.

    Parameters
    ----------
    embedding_dim : int — number of cosine basis functions n (default 64)
    output_dim    : int — must equal LSTM hidden_dim so ⊙ is well-defined
    """

    def __init__(self, embedding_dim: int = 64, output_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.output_dim    = output_dim

        # Basis indices j = 0, 1, ..., n-1  (fixed, not learned)
        self.register_buffer(
            "basis",
            torch.arange(1, embedding_dim + 1).float() * math.pi,
        )   # shape (n,)

        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Embed quantile levels τ into output_dim-dimensional vectors.

        Parameters
        ----------
        tau : Tensor shape (M,) — M quantile levels ∈ [0, 1]

        Returns
        -------
        Tensor shape (M, output_dim)
        """
        # cos(π·j·τ): outer product → (M, n)
        cos_features = torch.cos(tau.unsqueeze(1) * self.basis.unsqueeze(0))
        return self.proj(cos_features)   # (M, output_dim)


# ── IQN head ──────────────────────────────────────────────────────────────────

class IQNHead(nn.Module):
    """
    IQN quantile head with cosine embedding and dueling decomposition.

    Takes LSTM hidden state h_t and K sampled τ values, returns
    Z(s,a,τ) for all actions and all τ samples.

    Parameters
    ----------
    hidden_dim    : int  — LSTM hidden state dimension
    n_actions     : int  — flat action space size
    embedding_dim : int  — cosine embedding dimension (default 64)
    dueling       : bool — dueling decomposition per τ (default True)
    """

    def __init__(
        self,
        hidden_dim:    int,
        n_actions:     int,
        embedding_dim: int  = 64,
        dueling:       bool = True,
    ):
        super().__init__()
        self.hidden_dim    = hidden_dim
        self.n_actions     = n_actions
        self.dueling       = dueling

        self.tau_embed = CosineEmbedding(
            embedding_dim = embedding_dim,
            output_dim    = hidden_dim,
        )

        mid = max(hidden_dim // 2, n_actions)

        if dueling:
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, mid), nn.ReLU(), nn.Linear(mid, 1)
            )
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, mid), nn.ReLU(), nn.Linear(mid, n_actions)
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim, mid), nn.ReLU(), nn.Linear(mid, n_actions)
            )

    def forward(
        self,
        h:   torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Z(s,a,τ) for all actions and all τ samples.

        Parameters
        ----------
        h   : Tensor shape (B, hidden_dim) — LSTM hidden state
        tau : Tensor shape (B, K)          — K sampled quantile levels per element

        Returns
        -------
        Tensor shape (B, K, n_actions) — return distribution samples
        """
        B, K    = tau.shape
        tau_flat = tau.reshape(B * K)                       # (B*K,)
        phi      = self.tau_embed(tau_flat)                 # (B*K, hidden_dim)

        # Broadcast h: (B, hidden_dim) → (B*K, hidden_dim)
        h_expanded = h.unsqueeze(1).expand(B, K, -1).reshape(B * K, -1)

        # Element-wise product: modulate hidden state by quantile embedding
        combined = h_expanded * phi                         # (B*K, hidden_dim)

        if self.dueling:
            V = self.value_stream(combined)                 # (B*K, 1)
            A = self.advantage_stream(combined)             # (B*K, n_actions)
            Z = V + A - A.mean(dim=-1, keepdim=True)        # (B*K, n_actions)
        else:
            Z = self.proj(combined)                         # (B*K, n_actions)

        return Z.reshape(B, K, self.n_actions)              # (B, K, n_actions)


# ── IQN agent ─────────────────────────────────────────────────────────────────

class IQNAgent:
    """
    IQN agent with DRQN-LSTM backbone and CVaR action selection.

    Parameters
    ----------
    encoder              : nn.Module — any encoder with .latent_dim property
    n_actions            : int       — flat action space size
    n_quantile_samples   : int       — K τ samples per forward pass (default 64)
    embedding_dim        : int       — cosine embedding dim (default 64)
    hidden_dim           : int       — LSTM hidden size (default 128)
    dueling              : bool      — dueling decomposition (default True)
    lr                   : float     — Adam learning rate (default 1e-4)
    gamma                : float     — discount factor (default 0.99)
    batch_size           : int       — sequences per update (default 256)
    seq_len              : int       — DRQN sequence length (default 30)
    target_update_freq   : int       — target net update interval (default 1000)
    cvar_alpha           : float     — CVaR tail fraction (default 0.25)
    kappa                : float     — Huber loss threshold (default 1.0)
    epsilon_start        : float     — initial ε (default 1.0)
    epsilon_end          : float     — minimum ε (default 0.05)
    epsilon_decay_steps  : int       — ε decay steps (default 50_000)
    buffer_capacity      : int       — replay buffer size (default 100_000)
    prioritized          : bool      — use PER (default True)
    device               : str       — 'cpu' or 'cuda'
    """

    name = "IQN"

    def __init__(
        self,
        encoder:             nn.Module,
        n_actions:           int,
        n_quantile_samples:  int   = 64,
        embedding_dim:       int   = 64,
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
        device:              str   = "cpu",
    ):
        self.n_actions           = n_actions
        self.n_quantile_samples  = n_quantile_samples
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

        # Number of tail samples for CVaR
        self._n_tail = max(1, int(cvar_alpha * n_quantile_samples))

        # ── Networks ──────────────────────────────────────────────────
        self.online_base = RecurrentBase(
            encoder    = encoder,
            n_actions  = n_actions,
            hidden_dim = hidden_dim,
            dueling    = False,
        ).to(self.device)

        self.online_head = IQNHead(
            hidden_dim    = hidden_dim,
            n_actions     = n_actions,
            embedding_dim = embedding_dim,
            dueling       = dueling,
        ).to(self.device)

        self.target_base = copy.deepcopy(self.online_base).to(self.device)
        self.target_head = copy.deepcopy(self.online_head).to(self.device)

        self.target_base.eval()
        self.target_head.eval()
        for p in list(self.target_base.parameters()) + \
                 list(self.target_head.parameters()):
            p.requires_grad = False

        # ── Optimiser ─────────────────────────────────────────────────
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
    # τ sampling helpers
    # ------------------------------------------------------------------

    def _sample_tau(self, batch_size: int, K: int) -> torch.Tensor:
        """Sample K quantile levels uniformly ∈ (0,1) per batch element."""
        return torch.rand(batch_size, K, device=self.device)

    def _sample_tau_cvar(self, batch_size: int, K: int) -> torch.Tensor:
        """
        Sample K quantile levels restricted to [0, alpha] for CVaR.

        Sampling from the lower tail directly gives a more accurate
        CVaR estimate at action selection time.
        """
        return torch.rand(batch_size, K, device=self.device) * self.cvar_alpha

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _online_forward(
        self,
        obs: torch.Tensor,
        K:   int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Online network forward: obs sequence → Z(s,a,τ) at last step.

        Parameters
        ----------
        obs : Tensor shape (B, T, obs_dim)
        K   : int — number of quantile samples

        Returns
        -------
        Z   : Tensor shape (B, K, n_actions)
        tau : Tensor shape (B, K) — sampled quantile levels
        """
        lstm_out, _ = self.online_base.forward(obs)     # (B, T, hidden_dim)
        h_last      = lstm_out[:, -1, :]                # (B, hidden_dim)
        B           = h_last.shape[0]
        tau         = self._sample_tau(B, K)
        Z           = self.online_head(h_last, tau)     # (B, K, n_actions)
        return Z, tau

    def _target_forward(
        self,
        obs: torch.Tensor,
        K:   int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Target network forward — no gradient."""
        lstm_out, _ = self.target_base.forward(obs)
        h_last      = lstm_out[:, -1, :]
        B           = h_last.shape[0]
        tau         = self._sample_tau(B, K)
        Z           = self.target_head(h_last, tau)
        return Z, tau

    # ------------------------------------------------------------------
    # CVaR action selection
    # ------------------------------------------------------------------

    def _cvar_action(self, Z: torch.Tensor) -> torch.Tensor:
        """
        Select action maximising CVaR_α over K quantile samples.

        Z is already sampled from lower tail when using _sample_tau_cvar,
        so CVaR = mean over all K samples in that case.
        For full-range sampling, take bottom α fraction.

        Parameters
        ----------
        Z : Tensor shape (B, K, n_actions)

        Returns
        -------
        Tensor shape (B,) — action indices
        """
        # Z: (B, K, n_actions) → sort along K dim
        Z_sorted = Z.sort(dim=1).values                       # (B, K, n_actions)
        cvar     = Z_sorted[:, :self._n_tail, :].mean(dim=1)  # (B, n_actions)
        return cvar.argmax(dim=-1)                             # (B,)

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

        Parameters
        ----------
        obs    : np.ndarray shape (obs_dim,)
        greedy : bool — skip ε-greedy, always use CVaR

        Returns
        -------
        int — flat action index
        """
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        x = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)  # (1, obs_dim)

        lstm_out, _ = self.online_base.forward(x.unsqueeze(1))  # (1, 1, hidden_dim)
        h_last      = lstm_out[:, -1, :]                         # (1, hidden_dim)

        # Sample τ from lower tail for CVaR action selection
        tau = self._sample_tau_cvar(1, self.n_quantile_samples)
        Z   = self.online_head(h_last, tau)                      # (1, K, n_actions)

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
    # Training step
    # ------------------------------------------------------------------

    def train_step(self) -> Optional[float]:
        """
        Sample a sequence batch and perform one IQN gradient update.

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
        K       = self.n_quantile_samples

        # ── Online Z(s,a,τ) at last step ─────────────────────────────
        self.online_base.reset_hidden(batch_size=B, device=self.device)
        Z_all, tau = self._online_forward(obs, K)         # (B, K, n_actions), (B, K)

        # Gather Z for taken action: (B, K)
        a_last = actions[:, -1]                           # (B,)
        a_idx  = a_last.view(B, 1, 1).expand(B, K, 1)
        Z_sa   = Z_all.gather(2, a_idx).squeeze(2)        # (B, K)

        # ── Target Z(s', a*) at last step ────────────────────────────
        with torch.no_grad():
            self.target_base.reset_hidden(batch_size=B, device=self.device)
            Z_next, tau_prime = self._target_forward(next_obs, K)  # (B, K, n_actions)

            # Double DQN: greedy action from online net
            self.online_base.reset_hidden(batch_size=B, device=self.device)
            Z_online_next, _ = self._online_forward(next_obs, K)
            a_next = Z_online_next.mean(dim=1).argmax(dim=-1)      # (B,)

            a_next_idx = a_next.view(B, 1, 1).expand(B, K, 1)
            Z_next_sa  = Z_next.gather(2, a_next_idx).squeeze(2)   # (B, K)

            r     = rewards[:, -1].unsqueeze(1)   # (B, 1)
            d     = dones[:, -1].unsqueeze(1)     # (B, 1)
            Z_tgt = r + self.gamma * Z_next_sa * (1.0 - d)         # (B, K)

        # ── IQN quantile regression loss ──────────────────────────────
        # Reuse QR-DQN loss with sampled taus
        # tau shape (B, K) → need (K,) for the loss function
        # Use mean tau across batch as the quantile levels
        taus_mean = tau.mean(dim=0)   # (K,) — average quantile levels

        element_loss = QRDQNAgent._quantile_huber_loss(
            Z_sa, Z_tgt, taus_mean, self.kappa
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

        return float(loss.item())

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "online_base": self.online_base.state_dict(),
            "online_head": self.online_head.state_dict(),
            "target_base": self.target_base.state_dict(),
            "target_head": self.target_head.state_dict(),
            "optimiser":   self.optimiser.state_dict(),
            "steps":       self._steps,
            "updates":     self._updates,
            "beta":        self.buffer.beta if self.buffer.prioritized else None,
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
            f"IQNAgent("
            f"n_actions={self.n_actions}, "
            f"K={self.n_quantile_samples}, "
            f"cvar_alpha={self.cvar_alpha}, "
            f"epsilon={self.epsilon:.3f}, "
            f"steps={self._steps})"
        )