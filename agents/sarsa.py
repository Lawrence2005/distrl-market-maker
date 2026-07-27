"""
agents/sarsa.py

SARSA(λ) agent with tile coding and linear value function approximation.

Implements the linear tile-coding market-making agent described in:
    Spooner et al. (2018) — Market Making via Reinforcement Learning.

Architecture
------------
Three independent tile codings over different state subsets:

    TC0 (agent-state):   [inventory, time_remaining]         λ=0.6
    TC1 (market-state):  [spread, imbalance, vol, rsi]       λ=0.1
    TC2 (full-state):    all obs_dim features                λ=0.3

Value function (per action a):
    q̂(x, a) = Σ_{i=0}^{2} λ_i · Σ_{j=0}^{n_i-1} b_ij(x) · w_ij(a)

where b_ij(x) ∈ {0,1} is the binary tile activation and w_ij(a) is the
weight for tile j in coding i for action a.

Three independent weight matrices W_i ∈ R^{n_i × n_actions}, one per
tile coding. All three are updated with the same TD error.

SARSA(λ) update (eligibility traces):
    δ   = r + γ · q̂(s', a') − q̂(s, a)          TD error
    e_i ← γ · λ_trace · e_i + ∇q̂_i(s, a)       trace update
    W_i ← W_i + α · λ_i · δ · e_i               weight update

Tile coding implementation
--------------------------
Uses index hashing (modulo hash) to map continuous features into tile
indices without building an explicit grid. This is the standard approach
for high-dimensional state spaces (Sutton & Barto 2018, Chapter 9).

For each tiling t = 0..M-1:
    offset(t, dim) = t * iht_offset[dim]          (tiling offset)
    idx(t, dim)    = floor((x[dim] - lo[dim]) / width[dim] + offset(t, dim))
    tile_index     = hash(tuple(idx)) % n_tiles_per_tiling

Eligibility traces
------------------
Replacing traces (standard for tile coding — prevents trace explosion):
    e_ij ← γ·λ·e_ij   if tile j was NOT active this step
    e_ij ← 1           if tile j WAS active this step (replacing)

Parameters (from Spooner et al. 2018 Table)
-------------------------------------------
n_tilings    : int   — number of tilings M (default 32)
n_tiles      : int   — tiles per tiling per dimension (default 8)
alpha        : float — learning rate (default 0.001)
gamma        : float — discount factor (default 0.97)
lambda_trace : float — eligibility trace decay (default 0.96)
lambda_i     : tuple — tile coding weights (0.6, 0.1, 0.3)
epsilon      : float — initial exploration rate (default 0.7)
epsilon_floor: float — minimum exploration rate (default 0.0001)
epsilon_T    : int   — steps to decay epsilon (default 1000)

Reference
---------
Spooner, T., Fearnley, J., Savani, R. & Koukorinis, A. (2018).
Market Making via Reinforcement Learning. AAMAS 2018.

Sutton, R. & Barto, A. (2018). Reinforcement Learning: An Introduction.
Chapter 9 (tile coding), Chapter 12 (eligibility traces).

Week 5 deliverable.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from agents.base import AgentBase

# ══════════════════════════════════════════════════════════════════════════════
# Feature extraction helpers
# ══════════════════════════════════════════════════════════════════════════════

# Observation vector layout (LOBMarketMakingEnv, n_lob_levels=3, obs_dim=18):
#   0  bid_ask_spread       1  midprice_log_return   2  queue_imbalance
#   3  signed_volume        4  realized_vol          5  rsi
#   6  lob_bid_L1           7  lob_bid_L2            8  lob_bid_L3
#   9  lob_ask_L1          10  lob_ask_L2           11  lob_ask_L3
#  12  inventory            13 bid_distance          14 ask_distance
#  15  outstanding_bid      16 outstanding_ask       17 time_remaining

_IDX_INVENTORY     = 12
_IDX_TIME          = 17
_IDX_SPREAD        = 0
_IDX_IMBALANCE     = 2
_IDX_VOL           = 4
_IDX_RSI           = 5

# Feature indices for each tile coding
_TC0_FEATURES = [_IDX_INVENTORY, _IDX_TIME]              # agent-state
_TC1_FEATURES = [_IDX_SPREAD, _IDX_IMBALANCE, _IDX_VOL, _IDX_RSI]  # market-state
# TC2 uses all features (indices 0..obs_dim-1)

# Feature bounds for normalisation (all features already normalised in lob_env)
# Tile coding maps normalised features to [0, 1] range for hashing
_TC0_LO  = np.array([-1.0,  0.0])   # inventory ∈ [-1,1], time ∈ [0,1]
_TC0_HI  = np.array([ 1.0,  1.0])
_TC1_LO  = np.array([ 0.0, -1.0,  0.0, -1.0])
_TC1_HI  = np.array([10.0,  1.0, 50.0,  1.0])


# ══════════════════════════════════════════════════════════════════════════════
# Tile coding
# ══════════════════════════════════════════════════════════════════════════════

class TileCoding:
    """
    Hash-based tile coding for continuous state features.

    Maps a feature vector x ∈ R^d to a set of M active tile indices,
    one per tiling. Each tiling partitions the feature space into a grid
    of n_tiles^d tiles, offset slightly from the others.

    Uses modulo hashing to keep memory bounded:
        tile_index = hash(tiling_id, grid_coords) % memory_size

    Parameters
    ----------
    n_features  : int   — feature vector dimension d
    n_tilings   : int   — number of tilings M (default 32)
    n_tiles     : int   — tiles per dimension per tiling (default 8)
    memory_size : int   — hash table size (default 2^17 = 131072)
    lo          : np.ndarray shape (d,) — feature lower bounds
    hi          : np.ndarray shape (d,) — feature upper bounds
    seed        : int   — random seed for tiling offsets
    """

    def __init__(
        self,
        n_features:  int,
        n_tilings:   int        = 32,
        n_tiles:     int        = 8,
        memory_size: int        = 2 ** 17,
        lo:          Optional[np.ndarray] = None,
        hi:          Optional[np.ndarray] = None,
        seed:        int        = 42,
    ):
        self.n_features  = n_features
        self.n_tilings   = n_tilings
        self.n_tiles     = n_tiles
        self.memory_size = memory_size

        self.lo = lo if lo is not None else np.zeros(n_features)
        self.hi = hi if hi is not None else np.ones(n_features)

        # Width of each tile per dimension
        self.width = (self.hi - self.lo) / n_tiles

        # Random offsets per tiling per dimension for staggering
        rng = np.random.default_rng(seed)
        self.offsets = rng.uniform(0, 1, size=(n_tilings, n_features))

        self.is_online = True

    def active_tiles(self, x: np.ndarray) -> np.ndarray:
        """
        Return indices of M active tiles (one per tiling) for feature vector x.

        Parameters
        ----------
        x : np.ndarray shape (n_features,)

        Returns
        -------
        np.ndarray shape (n_tilings,) dtype int — one tile index per tiling
        """
        # Clip to bounds
        x_clipped = np.clip(x, self.lo + 1e-8, self.hi - 1e-8)

        # Normalise to [0, n_tiles) range
        x_norm = (x_clipped - self.lo) / self.width   # (d,)

        indices = np.zeros(self.n_tilings, dtype=np.int64)
        for t in range(self.n_tilings):
            # Offset this tiling
            coords = np.floor(x_norm + self.offsets[t]).astype(np.int64)
            # Hash: combine tiling id and grid coords into single index
            h = hash((t,) + tuple(coords)) % self.memory_size
            indices[t] = h

        return indices

    @property
    def n_weights(self) -> int:
        """Number of weight entries = memory_size (hashed)."""
        return self.memory_size


# ══════════════════════════════════════════════════════════════════════════════
# SARSA(λ) agent
# ══════════════════════════════════════════════════════════════════════════════

class SARSAAgent(AgentBase):
    """
    SARSA(λ) agent with tile coding and linear value function approximation.

    Three independent tile codings (TC0, TC1, TC2) over agent-state,
    market-state, and full-state respectively. Each maintained as a
    separate weight matrix W_i ∈ R^{memory_size × n_actions}.

    Action-value function:
        q̂(x, a) = Σ_i λ_i · (1/M) · Σ_{t active tile in TC_i} w_it(a)

    The (1/M) normalisation by number of tilings keeps the scale consistent
    regardless of M.

    Parameters
    ----------
    obs_dim      : int   — full observation dimension (default 18)
    n_actions    : int   — flat action space size (default 121)
    n_tilings    : int   — tilings per tile coding M (default 32)
    n_tiles      : int   — tiles per dimension (default 8)
    memory_size  : int   — hash table size per coding (default 2^17)
    alpha        : float — learning rate (default 0.001)
    gamma        : float — discount factor (default 0.97)
    lambda_trace : float — eligibility trace parameter (default 0.96)
    lambda_i     : tuple — tile coding weights (default (0.6, 0.1, 0.3))
    epsilon      : float — initial exploration rate (default 0.7)
    epsilon_floor: float — minimum ε (default 0.0001)
    epsilon_T    : int   — steps over which ε decays (default 1000)
    seed         : int   — random seed (default 42)
    """

    name = "SARSA"

    def __init__(
        self,
        obs_dim:      int   = 18,
        n_actions:    int   = 121,
        n_tilings:    int   = 32,
        n_tiles:      int   = 8,
        memory_size:  int   = 2 ** 17,
        alpha:        float = 0.001,
        gamma:        float = 0.97,
        lambda_trace: float = 0.96,
        lambda_i:     tuple = (0.6, 0.1, 0.3),
        epsilon:      float = 0.7,
        epsilon_floor:float = 0.0001,
        epsilon_T:    int   = 1000,
        seed:         int   = 42,
    ):
        assert abs(sum(lambda_i) - 1.0) < 1e-6, \
            f"lambda_i must sum to 1.0, got {sum(lambda_i)}"
        assert len(lambda_i) == 3, "Exactly 3 tile codings required"

        self.obs_dim       = obs_dim
        self.n_actions     = n_actions
        self.n_tilings     = n_tilings
        self.alpha         = alpha
        self.gamma         = gamma
        self.lambda_trace  = lambda_trace
        self.lambda_i      = np.array(lambda_i, dtype=np.float32)
        self.epsilon_start = epsilon
        self.epsilon_floor = epsilon_floor
        self.epsilon_T     = epsilon_T

        self._rng    = np.random.default_rng(seed)
        self._steps  = 0
        self._updates = 0

        # ── Tile codings ──────────────────────────────────────────────
        tc0_lo = _TC0_LO
        tc0_hi = _TC0_HI

        tc1_lo = _TC1_LO
        tc1_hi = _TC1_HI

        # TC2: full state — use per-feature bounds from lob_env normalisation
        # All features are approximately in [-1, 1] or [0, 1] after normalisation
        # Use [-1, 1] as a safe universal bound; clipping handles edge cases
        tc2_lo = -np.ones(obs_dim)
        tc2_hi =  np.ones(obs_dim)
        # Override known ranges for features that go outside [-1, 1]
        tc2_hi[_IDX_SPREAD] = 10.0
        tc2_hi[_IDX_VOL]    = 50.0

        self.tc0 = TileCoding(
            n_features=len(_TC0_FEATURES), n_tilings=n_tilings,
            n_tiles=n_tiles, memory_size=memory_size,
            lo=tc0_lo, hi=tc0_hi, seed=seed,
        )
        self.tc1 = TileCoding(
            n_features=len(_TC1_FEATURES), n_tilings=n_tilings,
            n_tiles=n_tiles, memory_size=memory_size,
            lo=tc1_lo, hi=tc1_hi, seed=seed + 1,
        )
        self.tc2 = TileCoding(
            n_features=obs_dim, n_tilings=n_tilings,
            n_tiles=n_tiles, memory_size=memory_size,
            lo=tc2_lo, hi=tc2_hi, seed=seed + 2,
        )

        self.tile_codings = [self.tc0, self.tc1, self.tc2]

        # ── Weight matrices ───────────────────────────────────────────
        # W_i shape: (memory_size, n_actions)
        # Initialised to zero (standard for tile coding)
        self.W = [
            np.zeros((tc.n_weights, n_actions), dtype=np.float32)
            for tc in self.tile_codings
        ]

        # ── Eligibility traces ────────────────────────────────────────
        # e_i shape: (memory_size, n_actions) — matching W_i
        self.E = [
            np.zeros((tc.n_weights, n_actions), dtype=np.float32)
            for tc in self.tile_codings
        ]

        # ── Episode state ─────────────────────────────────────────────
        self._prev_obs:    Optional[np.ndarray] = None
        self._prev_action: Optional[int]        = None
        self._prev_tiles:  Optional[list]       = None

    # ------------------------------------------------------------------
    # Epsilon schedule
    # ------------------------------------------------------------------

    @property
    def epsilon(self) -> float:
        """Linear decay from epsilon_start to epsilon_floor over epsilon_T steps."""
        frac = min(1.0, self._steps / max(self.epsilon_T, 1))
        return self.epsilon_start + frac * (self.epsilon_floor - self.epsilon_start)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract feature subsets for each tile coding from obs vector.

        Returns
        -------
        (x0, x1, x2) : feature vectors for TC0, TC1, TC2
        """
        x0 = obs[_TC0_FEATURES]
        x1 = obs[_TC1_FEATURES]
        x2 = obs   # full state
        return x0, x1, x2

    def _get_active_tiles(self, obs: np.ndarray) -> list[np.ndarray]:
        """
        Get active tile indices for all three tile codings.

        Returns
        -------
        list of 3 arrays, each shape (n_tilings,)
        """
        x0, x1, x2 = self._extract_features(obs)
        return [
            self.tc0.active_tiles(x0),
            self.tc1.active_tiles(x1),
            self.tc2.active_tiles(x2),
        ]

    # ------------------------------------------------------------------
    # Value function
    # ------------------------------------------------------------------

    def _q_values(self, tiles: list[np.ndarray]) -> np.ndarray:
        """
        Compute q̂(x, ·) for all actions given active tiles.

        q̂(x, a) = Σ_i λ_i · (1/M) · Σ_{j in tiles_i} W_i[j, a]

        Parameters
        ----------
        tiles : list of 3 arrays shape (n_tilings,) — active tile indices

        Returns
        -------
        np.ndarray shape (n_actions,) — Q-values for all actions
        """
        q = np.zeros(self.n_actions, dtype=np.float64)
        for i, (tc_tiles, lam) in enumerate(zip(tiles, self.lambda_i)):
            # Sum weights at active tiles, normalised by n_tilings
            q += lam * self.W[i][tc_tiles].sum(axis=0) / self.n_tilings
        return q

    def _q_value(self, tiles: list[np.ndarray], action: int) -> float:
        """Q-value for a single (state, action) pair."""
        return self._q_values(tiles)[action]

    # ------------------------------------------------------------------
    # Eligibility trace update
    # ------------------------------------------------------------------

    def _update_traces(
        self,
        tiles:  list[np.ndarray],
        action: int,
    ) -> None:
        """
        Update eligibility traces using replacing traces.

        Replacing traces:
            e_i[j, a] ← γ·λ·e_i[j, a]   for all (j, a)
            e_i[j, action] ← 1            for j in active tiles_i

        Replacing (vs accumulating) prevents trace explosion and is
        standard for tile coding (Sutton & Barto 2018, p. 305).
        """
        gl = self.gamma * self.lambda_trace

        for i, tc_tiles in enumerate(tiles):
            # Decay all traces
            self.E[i] *= gl
            # Replace active tile traces for the taken action
            self.E[i][tc_tiles, action] = 1.0

    # ------------------------------------------------------------------
    # Weight update
    # ------------------------------------------------------------------

    def _update_weights(self, td_error: float) -> None:
        """
        Update all weight matrices using TD error and eligibility traces.

        W_i ← W_i + α · λ_i · δ · E_i

        The λ_i weighting means each tile coding contributes proportionally
        to its influence on the value estimate.

        Parameters
        ----------
        td_error : float — δ = r + γ·q̂(s',a') - q̂(s,a)
        """
        for i in range(3):
            self.W[i] += self.alpha * self.lambda_i[i] * td_error * self.E[i]

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def act(
        self,
        obs:    np.ndarray,
        greedy: bool = False,
    ) -> int:
        """
        Select action using ε-greedy policy over tile-coded Q-values.

        Parameters
        ----------
        obs    : np.ndarray shape (obs_dim,) — observation from env
        greedy : bool — if True, always select argmax (evaluation mode)

        Returns
        -------
        int — flat action index ∈ [0, n_actions)
        """
        if not greedy and self._rng.random() < self.epsilon:
            return int(self._rng.integers(self.n_actions))

        tiles = self._get_active_tiles(obs)
        q     = self._q_values(tiles)
        return int(np.argmax(q))

    # ------------------------------------------------------------------
    # SARSA(λ) step
    # ------------------------------------------------------------------

    def reset_hidden(self, batch_size: int = 1) -> None:
        """
        Reset episode state — clear traces and prev obs/action.

        Called at episode start. Matches the interface of neural agents
        so the training loop can treat all agents identically.
        """
        for i in range(3):
            self.E[i].fill(0.0)
        self._prev_obs    = None
        self._prev_action = None
        self._prev_tiles  = None

    def observe(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        """
        Store transition — increments step counter for epsilon decay.

        SARSA update is performed in train_step() to match the interface
        of neural agents.
        """
        self._steps += 1

    def train_step(
        self,
        obs:      Optional[np.ndarray] = None,
        action:   Optional[int]        = None,
        reward:   Optional[float]      = None,
        next_obs: Optional[np.ndarray] = None,
        done:     Optional[bool]       = None,
    ) -> Optional[float]:
        """
        Perform one SARSA(λ) update step.

        Unlike neural agents, SARSA needs the full (s, a, r, s', a')
        tuple simultaneously — it cannot sample from a replay buffer.
        Call this immediately after each env step with the current
        transition data.

        Parameters
        ----------
        obs      : np.ndarray — current state s
        action   : int        — action taken a
        reward   : float      — reward received r
        next_obs : np.ndarray — next state s'
        done     : bool       — episode ended

        Returns
        -------
        float — |TD error| for logging, or None if called without args
                (compatibility with neural agent interface)
        """
        if obs is None:
            return None   # called without args — no-op

        # Active tiles for current state
        tiles = self._get_active_tiles(obs)

        # Update eligibility traces
        self._update_traces(tiles, action)

        # Select next action (on-policy: SARSA uses π for next action)
        if done:
            q_next = 0.0
        else:
            next_tiles = self._get_active_tiles(next_obs)
            next_action = self.act(next_obs)   # on-policy next action
            q_next = self._q_value(next_tiles, next_action)

        # TD error: δ = r + γ·q̂(s', a') - q̂(s, a)
        q_curr   = self._q_value(tiles, action)
        td_error = reward + self.gamma * q_next - q_curr

        # Update weights
        self._update_weights(td_error)

        # Reset traces at episode end
        if done:
            for i in range(3):
                self.E[i].fill(0.0)

        self._updates += 1
        return float(abs(td_error))

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def state_dict(self) -> dict:
        """Return full agent state for checkpointing."""
        return {
            "W":       self.W,
            "E":       self.E,
            "steps":   self._steps,
            "updates": self._updates,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore agent state from checkpoint."""
        self.W        = [np.array(w) for w in state["W"]]
        self.E        = [np.array(e) for e in state["E"]]
        self._steps   = state["steps"]
        self._updates = state["updates"]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def weight_norm(self) -> list[float]:
        """L2 norm of each weight matrix — useful for monitoring divergence."""
        return [float(np.linalg.norm(w)) for w in self.W]

    def __repr__(self) -> str:
        return (
            f"SARSAAgent("
            f"n_tilings={self.n_tilings}, "
            f"lambda_i={tuple(self.lambda_i)}, "
            f"alpha={self.alpha}, "
            f"epsilon={self.epsilon:.4f}, "
            f"steps={self._steps})"
        )