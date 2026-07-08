"""
agents/dqn.py

Vanilla DQN agent with DRQN-LSTM architecture.

Used as the non-distributional ablation baseline. Identical architecture
to QR-DQN and IQN (Encoder → LSTM → DuelingHead) but outputs a single
Q-value per action rather than a distribution of quantiles.

Loss: standard Huber (smooth-L1) TD loss
    L = HuberLoss(Q(s,a), r + γ · max_a' Q_target(s',a'))

Action selection: ε-greedy over Q(s,a)

This agent is used to answer the ablation question:
    "Does distributional RL add value over vanilla DQN for market making?"

Parameters (from configs/agent/dqn.yaml)
-----------------------------------------
hidden_dim        : int   — LSTM hidden state size (default 256)
lr                : float — Adam learning rate (default 1e-4)
batch_size        : int   — transitions per gradient step (default 256)
gamma             : float — discount factor (default 0.99)
target_update_freq: int   — hard target network update interval in steps
replay_buffer_size: int   — maximum buffer capacity (default 100_000)
epsilon_start     : float — initial exploration rate (default 1.0)
epsilon_end       : float — minimum exploration rate (default 0.05)
epsilon_decay_steps: int  — steps to decay epsilon (default 50_000)
lstm_hidden       : int   — LSTM hidden size (default 128, from dqn.yaml)
lstm_window       : int   — sequence length for DRQN (default 30)

Reference
---------
Mnih et al. (2015) — Human-level control through deep reinforcement learning.
Hausknecht & Stone (2015) — Deep Recurrent Q-Networks.
Wang et al. (2016) — Dueling Network Architectures.

Week 5 deliverable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.recurrent_base import RecurrentBase
from training.replay_buffer import ReplayBuffer


class DQNAgent:
    """
    Vanilla DQN agent with DRQN-LSTM backbone and dueling head.

    Parameters
    ----------
    encoder        : nn.Module — any encoder with .latent_dim property
    n_actions      : int       — flat action space size
    hidden_dim     : int       — LSTM hidden state size (default 128)
    lr             : float     — Adam learning rate (default 1e-4)
    gamma          : float     — discount factor (default 0.99)
    batch_size     : int       — training batch size (default 256)
    seq_len        : int       — DRQN sequence length (default 30)
    target_update_freq : int   — target network update interval (default 1000)
    epsilon_start  : float     — initial ε for ε-greedy (default 1.0)
    epsilon_end    : float     — minimum ε (default 0.05)
    epsilon_decay_steps : int  — steps over which ε decays (default 50_000)
    buffer_capacity: int       — replay buffer size (default 100_000)
    prioritized    : bool      — use PER (default False for DQN ablation)
    device         : str       — 'cpu' or 'cuda'
    """

    name = "DQN"

    def __init__(
        self,
        encoder:             nn.Module,
        n_actions:           int,
        hidden_dim:          int   = 128,
        lr:                  float = 1e-4,
        gamma:               float = 0.99,
        batch_size:          int   = 256,
        seq_len:             int   = 30,
        target_update_freq:  int   = 1000,
        epsilon_start:       float = 1.0,
        epsilon_end:         float = 0.05,
        epsilon_decay_steps: int   = 50_000,
        buffer_capacity:     int   = 100_000,
        prioritized:         bool  = False,
        device:              str   = "cpu",
    ):
        self.n_actions          = n_actions
        self.gamma              = gamma
        self.batch_size         = batch_size
        self.seq_len            = seq_len
        self.target_update_freq = target_update_freq
        self.epsilon_start      = epsilon_start
        self.epsilon_end        = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.device             = torch.device(device)

        # ── Networks ──────────────────────────────────────────────────
        self.online = RecurrentBase(
            encoder    = encoder,
            n_actions  = n_actions,
            hidden_dim = hidden_dim,
            dueling    = True,
        ).to(self.device)

        # Target network: same architecture, weights updated periodically
        import copy
        self.target = copy.deepcopy(self.online).to(self.device)
        self.target.eval()
        for p in self.target.parameters():
            p.requires_grad = False

        # ── Optimiser ─────────────────────────────────────────────────
        self.optimiser = torch.optim.Adam(
            self.online.parameters(), lr=lr
        )

        # ── Replay buffer ─────────────────────────────────────────────
        self.buffer = ReplayBuffer(
            capacity    = buffer_capacity,
            prioritized = prioritized,
            seq_len     = seq_len,
        )

        # ── Counters ──────────────────────────────────────────────────
        self._steps      = 0   # total env steps seen
        self._updates    = 0   # total gradient updates

    # ------------------------------------------------------------------
    # Exploration schedule
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        """Current ε: linearly decayed from start to end over decay_steps."""
        frac = min(1.0, self._steps / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

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
        Select an action using ε-greedy policy.

        Parameters
        ----------
        obs    : np.ndarray shape (obs_dim,) — single observation
        greedy : bool — if True, always select argmax (evaluation mode)

        Returns
        -------
        int — flat action index ∈ [0, n_actions)
        """
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)

        x = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        q = self.online.step(x)   # (1, n_actions)
        return int(q.argmax(dim=-1).item())

    def reset_hidden(self, batch_size: int = 1) -> None:
        """Reset LSTM hidden state. Call at episode start."""
        self.online.reset_hidden(batch_size=batch_size, device=self.device)
        self.target.reset_hidden(batch_size=batch_size, device=self.device)

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
        """
        Store a transition in the replay buffer and increment step counter.

        Parameters
        ----------
        obs      : np.ndarray — current observation
        action   : int        — action taken
        reward   : float      — reward received
        next_obs : np.ndarray — next observation
        done     : bool       — episode ended
        """
        self.buffer.push(obs, action, reward, next_obs, done)
        self._steps += 1

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(self) -> Optional[float]:
        """
        Sample a batch and perform one gradient update.

        Returns
        -------
        float | None — loss value if update was performed, None if buffer
                       does not yet have enough transitions.
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

        # obs shape: (B, seq_len, obs_dim)
        B, T, _ = obs.shape

        # ── Online Q-values ───────────────────────────────────────────
        self.online.reset_hidden(batch_size=B, device=self.device)
        q_all, _ = self.online.forward(obs)           # (B, T, n_actions)

        # Take Q-value of the action taken at the last step of each sequence
        q_sa = q_all[:, -1, :].gather(
            1, actions[:, -1].unsqueeze(1)
        ).squeeze(1)                                   # (B,)

        # ── Target Q-values ───────────────────────────────────────────
        with torch.no_grad():
            self.target.reset_hidden(batch_size=B, device=self.device)
            q_next_all, _ = self.target.forward(next_obs)  # (B, T, n_actions)
            q_next = q_next_all[:, -1, :].max(dim=-1).values  # (B,)

            # Bellman target: r + γ · max Q_target(s', a') · (1 - done)
            target = rewards[:, -1] + self.gamma * q_next * (1.0 - dones[:, -1])

        # ── Huber loss (IS-weighted for PER) ──────────────────────────
        td_errors   = (q_sa - target).abs().detach().cpu().numpy()
        element_loss = F.smooth_l1_loss(q_sa, target, reduction="none")
        loss         = (is_w * element_loss).mean()

        # ── Gradient update ───────────────────────────────────────────
        self.optimiser.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimiser.step()

        # ── PER priority update ───────────────────────────────────────
        self.buffer.update_priorities(indices, td_errors + 1e-6)

        # ── Target network hard update ────────────────────────────────
        self._updates += 1
        if self._updates % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())

        return float(loss.item())

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return full agent state for checkpointing."""
        return {
            "online":    self.online.state_dict(),
            "target":    self.target.state_dict(),
            "optimiser": self.optimiser.state_dict(),
            "steps":     self._steps,
            "updates":   self._updates,
            "beta":      self.buffer.beta if self.buffer.prioritized else None,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore agent state from checkpoint."""
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
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
            f"DQNAgent("
            f"n_actions={self.n_actions}, "
            f"epsilon={self.epsilon:.3f}, "
            f"steps={self._steps}, "
            f"updates={self._updates})"
        )