"""
agents/ppo.py

Proximal Policy Optimisation (PPO) with KL-penalty adaptive coefficient.

Implements the KL-penalty formulation of PPO (Schulman et al. 2017)
as described in the project spec, using a shared Actor-Critic backbone
with DRQN-LSTM recurrence.

Architecture (shared backbone)
-------------------------------
    obs → Encoder → latent_z → LSTM → h_t ─┬→ Policy head → π(a|s)  [Actor]
                                             └→ Value head  → V(s)    [Critic]

Policy head:  Linear(hidden_dim → hidden_dim//2) → ReLU → Linear → n_actions
              + Softmax → categorical distribution π(a|s)

Value head:   Linear(hidden_dim → hidden_dim//2) → ReLU → Linear → scalar V(s)

Sharing the LSTM backbone means actor and critic see the same temporal
representation. This is parameter-efficient and standard for PPO in
environments with recurrent structure.

PPO KL-penalty objective (eq. 2.9 from spec)
---------------------------------------------
    L(θ) = E_t [ π_θ(a_t|s_t) / π_θ_old(a_t|s_t) · A_t
                 - β · KL[π_θ_old(·|s_t) || π_θ(·|s_t)] ]

Adaptive β:
    if KL > 1.5 · δ_target  →  β ← β · 2      (KL too large, tighten)
    if KL < δ_target / 1.5  →  β ← β / 2      (KL too small, relax)
    else                     →  β unchanged

Advantage estimation: Generalised Advantage Estimation (GAE)
-------------------------------------------------------------
    δ_t   = r_t + γ · V(s_{t+1}) · (1 - done_t) - V(s_t)
    A_t   = Σ_{k=0}^{T-t} (γ · λ_gae)^k · δ_{t+k}
    returns_t = A_t + V(s_t)   (used as value target)

PPO training
------------
Each episode collects a full trajectory (on-policy). After the episode:
    1. Compute GAE advantages and returns
    2. Run K epochs of minibatch updates over the trajectory
    3. Update β based on mean KL divergence
    4. Clear trajectory buffer

On-policy: no replay buffer. Each transition is used once (or K times
within one update cycle) then discarded.

Parameters
----------
gamma       : float — discount factor (default 0.99)
gae_lambda  : float — GAE λ parameter (default 0.95)
epsilon_clip: float — PPO clip ratio (default 0.2, used as alternative
                      to KL penalty — set to None to use KL only)
beta_init   : float — initial KL penalty coefficient (default 0.5)
delta_target: float — KL divergence target (default 0.01)
k_epochs    : int   — update epochs per episode (default 4)
minibatch_size: int — minibatch size within epoch (default 64)
entropy_coef: float — entropy bonus coefficient (default 0.01)
value_coef  : float — value loss coefficient (default 0.5)
lr_actor    : float — actor learning rate (default 3e-4)
lr_critic   : float — critic learning rate (default 1e-3)
max_grad_norm: float — gradient clipping (default 0.5)

Reference
---------
Schulman, J., Wolski, F., Dhariwal, P., Radford, A. & Klimov, O. (2017).
Proximal Policy Optimization Algorithms. arXiv:1707.06347.

Schulman, J., Moritz, P., Levine, S., Jordan, M. & Abbeel, P. (2016).
High-Dimensional Continuous Control Using Generalized Advantage Estimation.
ICLR 2016.

Week 5 deliverable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.base import AgentBase

# ══════════════════════════════════════════════════════════════════════════════
# Shared Actor-Critic network
# ══════════════════════════════════════════════════════════════════════════════

class ActorCriticNet(nn.Module):
    """
    Shared backbone Actor-Critic network with DRQN-LSTM recurrence.

    Encoder → LSTM → shared hidden state h_t
        → Policy head: π(a|s)   [Actor]
        → Value head:  V(s)     [Critic]

    Parameters
    ----------
    encoder    : nn.Module — any encoder with .latent_dim property
    n_actions  : int       — flat action space size
    hidden_dim : int       — LSTM hidden state dimension (default 128)
    num_layers : int       — LSTM depth (default 1)
    """

    def __init__(
        self,
        encoder:    nn.Module,
        n_actions:  int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        use_lstm:   bool = True,
    ):
        super().__init__()

        self.encoder    = encoder
        self.n_actions  = n_actions
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_lstm   = use_lstm

        if use_lstm:
            # Shared LSTM backbone
            self.lstm = nn.LSTM(
                input_size  = encoder.latent_dim,
                hidden_size = hidden_dim,
                num_layers  = num_layers,
                batch_first = True,
            )
            self.proj = None
        else:
            # Snapshot mode: no temporal memory
            self.lstm = None
            self.proj = nn.Linear(encoder.latent_dim, hidden_dim)

        mid = max(hidden_dim // 2, n_actions)

        # Policy head (actor): h_t → π(a|s)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, n_actions),
        )

        # Value head (critic): h_t → V(s)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, 1),
        )

        # Hidden state: initialised on reset
        self._hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None

    def reset_hidden(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None,
    ) -> None:
        """Reset LSTM hidden state to zeros. Call at episode start."""
        if not self.use_lstm:
            self._hidden = None
            return
        if device is None:
            device = next(self.lstm.parameters()).device
        zeros = torch.zeros(self.num_layers, batch_size, self.hidden_dim,
                            device=device)
        self._hidden = (zeros, zeros.clone())

    def detach_hidden(self) -> None:
        """Detach hidden state from computation graph."""
        if self._hidden is not None:
            h, c = self._hidden
            self._hidden = (h.detach(), c.detach())

    def forward(
        self,
        x:      torch.Tensor,
        hidden: Optional[tuple] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """
        Full sequence forward pass.

        Parameters
        ----------
        x      : Tensor shape (B, T, input_dim)
        hidden : (h, c) or None

        Returns
        -------
        logits : Tensor shape (B, T, n_actions) — raw policy logits
        values : Tensor shape (B, T)            — state values V(s)
        hidden : (h, c) tuple                   — updated hidden state
        """
        # Encode: (B, T, input_dim) → (B, T, latent_dim)
        z = self.encoder(x)

        if self.use_lstm:
            if hidden is None:
                hidden = self._hidden
            if hidden is None:
                self.reset_hidden(batch_size=x.size(0), device=x.device)
                hidden = self._hidden

            # LSTM: (B, T, latent_dim) → (B, T, hidden_dim)
            lstm_out, new_hidden = self.lstm(z, hidden)
            self._hidden = new_hidden
        else:
            # Snapshot mode: per-timestep projection, no temporal memory
            lstm_out   = self.proj(z)
            new_hidden = None
            self._hidden = None

        B, T, H = lstm_out.shape
        h_flat = lstm_out.reshape(B * T, H)

        logits = self.policy_head(h_flat).reshape(B, T, self.n_actions)
        values = self.value_head(h_flat).reshape(B, T)

        return logits, values, new_hidden

    def step(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single-step inference: (B, input_dim) → (logits, value).

        Parameters
        ----------
        x : Tensor shape (B, input_dim)

        Returns
        -------
        logits : Tensor shape (B, n_actions)
        value  : Tensor shape (B,)
        """
        logits, values, _ = self.forward(x.unsqueeze(1))
        return logits.squeeze(1), values.squeeze(1)


# ══════════════════════════════════════════════════════════════════════════════
# Trajectory buffer (on-policy, no replay)
# ══════════════════════════════════════════════════════════════════════════════

class TrajectoryBuffer:
    """
    On-policy trajectory buffer for PPO.

    Stores one episode's worth of transitions, then cleared after
    each PPO update. No experience replay — each transition is used
    at most K times within one update cycle.
    """

    def __init__(self):
        self.obs:      list[np.ndarray] = []
        self.actions:  list[int]        = []
        self.rewards:  list[float]      = []
        self.values:   list[float]      = []
        self.log_probs:list[float]      = []
        self.dones:    list[bool]       = []

    def push(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        value:    float,
        log_prob: float,
        done:     bool,
    ) -> None:
        self.obs.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)

    def clear(self) -> None:
        self.__init__()

    def __len__(self) -> int:
        return len(self.rewards)


# ══════════════════════════════════════════════════════════════════════════════
# PPO agent
# ══════════════════════════════════════════════════════════════════════════════

class PPOAgent(AgentBase):
    """
    PPO agent with KL-penalty, shared Actor-Critic backbone, GAE.

    Parameters
    ----------
    encoder        : nn.Module — encoder with .latent_dim property
    n_actions      : int       — flat action space size
    hidden_dim     : int       — LSTM hidden dimension (default 128)
    gamma          : float     — discount factor (default 0.99)
    gae_lambda     : float     — GAE λ (default 0.95)
    beta_init      : float     — initial KL penalty coefficient (default 0.5)
    delta_target   : float     — KL divergence target (default 0.01)
    k_epochs       : int       — update epochs per episode (default 4)
    minibatch_size : int       — minibatch size (default 64)
    entropy_coef   : float     — entropy bonus (default 0.01)
    value_coef     : float     — value loss weight (default 0.5)
    lr             : float     — learning rate (default 3e-4)
    max_grad_norm  : float     — gradient clip norm (default 0.5)
    device         : str       — 'cpu' or 'cuda'
    """

    name = "PPO"
    is_online = False   # uses trajectory buffer, not step-by-step like SARSA

    def __init__(
        self,
        encoder:        nn.Module,
        n_actions:      int,
        hidden_dim:     int   = 128,
        gamma:          float = 0.99,
        gae_lambda:     float = 0.95,
        beta_init:      float = 0.5,
        delta_target:   float = 0.01,
        k_epochs:       int   = 4,
        minibatch_size: int   = 64,
        entropy_coef:   float = 0.01,
        value_coef:     float = 0.5,
        lr:             float = 3e-4,
        max_grad_norm:  float = 0.5,
        use_lstm:       bool  = True,
        device:         str   = "cpu",
    ):
        self.n_actions      = n_actions
        self.gamma          = gamma
        self.gae_lambda     = gae_lambda
        self.beta           = beta_init
        self.delta_target   = delta_target
        self.k_epochs       = k_epochs
        self.minibatch_size = minibatch_size
        self.entropy_coef   = entropy_coef
        self.value_coef     = value_coef
        self.max_grad_norm  = max_grad_norm
        self.device         = torch.device(device)

        # Shared actor-critic network
        self.ac = ActorCriticNet(
            encoder    = encoder,
            n_actions  = n_actions,
            hidden_dim = hidden_dim,
            use_lstm   = use_lstm,
        ).to(self.device)

        self.optimiser = torch.optim.Adam(
            self.ac.parameters(), lr=lr
        )

        self.buffer   = TrajectoryBuffer()
        self._steps   = 0
        self._updates = 0


    # ------------------------------------------------------------------
    # Interface compatibility
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        """PPO uses stochastic policy — no epsilon-greedy exploration."""
        return 0.0

    def reset_hidden(self, batch_size: int = 1) -> None:
        """Reset LSTM hidden state. Call at episode start."""
        self.ac.reset_hidden(batch_size=batch_size, device=self.device)

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
        Sample action from policy π(a|s).

        Parameters
        ----------
        obs    : np.ndarray shape (obs_dim,)
        greedy : bool — if True, take argmax (evaluation mode)

        Returns
        -------
        int — flat action index
        """
        x = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        logits, value = self.ac.step(x)   # (1, n_actions), (1,)

        if greedy:
            action = int(logits.argmax(dim=-1).item())
        else:
            dist   = Categorical(logits=logits)
            action = int(dist.sample().item())

        return action

    @torch.no_grad()
    def act_with_value(
        self,
        obs: np.ndarray,
    ) -> tuple[int, float, float]:
        """
        Sample action and return value + log_prob for buffer storage.

        Parameters
        ----------
        obs : np.ndarray shape (obs_dim,)

        Returns
        -------
        (action, value, log_prob) : tuple
        """
        x = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        logits, value = self.ac.step(x)
        dist     = Categorical(logits=logits)
        action   = dist.sample()
        log_prob = dist.log_prob(action)

        return (
            int(action.item()),
            float(value.item()),
            float(log_prob.item()),
        )

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
        value:    float = 0.0,
        log_prob: float = 0.0,
    ) -> None:
        """
        Store transition in trajectory buffer.

        For PPO, call act_with_value() to get value and log_prob,
        then pass them here. The training loop handles this.
        """
        self.buffer.push(obs, action, reward, value, log_prob, done)
        self._steps += 1

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def _compute_gae(
        self,
        rewards:   np.ndarray,
        values:    np.ndarray,
        dones:     np.ndarray,
        last_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalised Advantage Estimates and value targets.

        A_t = Σ_{k=0}^{T-t} (γ·λ)^k · δ_{t+k}
        δ_t = r_t + γ·V(s_{t+1})·(1-done_t) - V(s_t)

        Parameters
        ----------
        rewards    : np.ndarray shape (T,)
        values     : np.ndarray shape (T,)  — V(s_t) from critic
        dones      : np.ndarray shape (T,)  — 1.0 if episode ended
        last_value : float                  — V(s_{T+1}), 0 if done

        Returns
        -------
        advantages : np.ndarray shape (T,)
        returns    : np.ndarray shape (T,) — advantages + values (value targets)
        """
        T          = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        gae        = 0.0

        # Bootstrap from last value
        next_value = last_value

        for t in reversed(range(T)):
            next_non_terminal = 1.0 - dones[t]
            delta  = (rewards[t]
                      + self.gamma * next_value * next_non_terminal
                      - values[t])
            gae    = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
            next_value    = values[t]

        returns = advantages + values
        return advantages, returns

    # ------------------------------------------------------------------
    # PPO update
    # ------------------------------------------------------------------

    def train_step(self) -> Optional[float]:
        """
        Run PPO update over stored trajectory.

        Called once per episode (after episode completion).
        Runs K epochs of minibatch updates, then clears the buffer.

        Returns
        -------
        float | None — mean total loss across epochs, or None if buffer empty
        """
        if len(self.buffer) == 0:
            return None

        # ── Prepare trajectory data ────────────────────────────────────
        obs_arr      = np.array(self.buffer.obs,       dtype=np.float32)
        actions_arr  = np.array(self.buffer.actions,   dtype=np.int64)
        rewards_arr  = np.array(self.buffer.rewards,   dtype=np.float32)
        values_arr   = np.array(self.buffer.values,    dtype=np.float32)
        log_probs_arr= np.array(self.buffer.log_probs, dtype=np.float32)
        dones_arr    = np.array(self.buffer.dones,     dtype=np.float32)

        # Bootstrap last value
        last_value = 0.0   # episode ended, V(terminal) = 0

        # ── GAE ────────────────────────────────────────────────────────
        advantages, returns = self._compute_gae(
            rewards_arr, values_arr, dones_arr, last_value
        )

        # Normalise advantages (reduces variance)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ── Convert to tensors ─────────────────────────────────────────
        obs_t      = torch.from_numpy(obs_arr).to(self.device)
        actions_t  = torch.from_numpy(actions_arr).to(self.device)
        old_lp_t   = torch.from_numpy(log_probs_arr).to(self.device)
        adv_t      = torch.from_numpy(advantages).to(self.device)
        returns_t  = torch.from_numpy(returns).to(self.device)

        T          = len(rewards_arr)
        total_loss = 0.0
        n_updates  = 0

        # ── K epochs of minibatch updates ─────────────────────────────
        for _ in range(self.k_epochs):
            # Shuffle indices for minibatches
            indices = np.random.permutation(T)

            for start in range(0, T, self.minibatch_size):
                idx = indices[start : start + self.minibatch_size]
                if len(idx) < 2:
                    continue

                mb_obs     = obs_t[idx]
                mb_actions = actions_t[idx]
                mb_old_lp  = old_lp_t[idx]
                mb_adv     = adv_t[idx]
                mb_returns = returns_t[idx]

                # Forward pass: treat each minibatch element as T=1 sequence
                # Reset hidden for each minibatch (minibatches are shuffled,
                # not contiguous sequences — hidden state is irrelevant here)
                self.ac.reset_hidden(
                    batch_size=len(idx), device=self.device
                )
                logits, values, _ = self.ac.forward(
                    mb_obs.unsqueeze(1)   # (B, 1, obs_dim)
                )
                logits = logits.squeeze(1)   # (B, n_actions)
                values = values.squeeze(1)   # (B,)

                dist     = Categorical(logits=logits)
                new_lp   = dist.log_prob(mb_actions)
                entropy  = dist.entropy()

                # ── Policy loss (KL-penalty PPO objective) ─────────────
                # ratio = π_new(a|s) / π_old(a|s)
                ratio       = torch.exp(new_lp - mb_old_lp)
                policy_loss = -(ratio * mb_adv).mean()

                # KL divergence: KL[π_old || π_new]
                # For categorical: KL = Σ_a π_old(a) · (log π_old(a) - log π_new(a))
                # Approximated as: KL ≈ mean((old_lp - new_lp))
                kl_div = (mb_old_lp - new_lp).mean()

                # ── Value loss ─────────────────────────────────────────
                value_loss = F.mse_loss(values, mb_returns)

                # ── Entropy bonus (encourages exploration) ─────────────
                entropy_loss = -entropy.mean()

                # ── Total loss ─────────────────────────────────────────
                # L = policy_loss + β·KL - entropy_coef·H + value_coef·V_loss
                loss = (policy_loss
                        + self.beta * kl_div
                        + self.entropy_coef * entropy_loss
                        + self.value_coef * value_loss)

                self.optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.ac.parameters(), self.max_grad_norm
                )
                self.optimiser.step()

                total_loss += loss.item()
                n_updates  += 1

        # ── Adaptive β update ──────────────────────────────────────────
        # Compute mean KL over full trajectory with current policy
        with torch.no_grad():
            self.ac.reset_hidden(batch_size=1, device=self.device)
            # Process full trajectory as one sequence for KL estimate
            logits_full, _, _ = self.ac.forward(obs_t.unsqueeze(0))
            logits_full = logits_full.squeeze(0)   # (T, n_actions)
            dist_new    = Categorical(logits=logits_full)
            new_lp_full = dist_new.log_prob(actions_t)
            mean_kl     = (old_lp_t - new_lp_full).mean().item()

        if mean_kl > 1.5 * self.delta_target:
            self.beta *= 2.0      # KL too large: tighten penalty
        elif mean_kl < self.delta_target / 1.5:
            self.beta /= 2.0      # KL too small: relax penalty
        # else: β unchanged (KL within target range)

        # ── Clear trajectory buffer ────────────────────────────────────
        self.buffer.clear()
        self._updates += 1

        # Restore hidden state for next episode
        self.ac.reset_hidden(batch_size=1, device=self.device)

        mean_loss = total_loss / max(n_updates, 1)
        return float(mean_loss)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        return {
            "ac":        self.ac.state_dict(),
            "optimiser": self.optimiser.state_dict(),
            "beta":      self.beta,
            "steps":     self._steps,
            "updates":   self._updates,
        }

    def load_state_dict(self, state: dict) -> None:
        self.ac.load_state_dict(state["ac"])
        self.optimiser.load_state_dict(state["optimiser"])
        self.beta      = state["beta"]
        self._steps    = state["steps"]
        self._updates  = state["updates"]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PPOAgent("
            f"n_actions={self.n_actions}, "
            f"beta={self.beta:.4f}, "
            f"steps={self._steps}, "
            f"updates={self._updates})"
        )