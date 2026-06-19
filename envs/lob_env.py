"""
Base ABIDES-Gym market-making environment wrapper.

Wraps the ABIDES-Gym MarketMakingEnvironment with:
- Configurable observation space (handcrafted features)
- Configurable action space (discrete bid/ask offsets)
- Three reward formulations (asymmetric-eta / quadratic / sparse)

Week 2 deliverable.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any

# Action space: bid/ask offsets in ticks
# δ ∈ {−4, −3, −2, −1, 0, +1, +2, +3, +4} → 9 levels per side → 81 total
TICK_OFFSETS = np.arange(-4, 5)   # shape (9,)
N_OFFSET_LEVELS = len(TICK_OFFSETS)

class LOBMarketMakingEnv(gym.Env):
    """
    Limit Order Book market-making environment.

    Observation space: handcrafted feature vector (~17 dims, see §2 MDP doc)
    Action space:      Discrete(81) — 9×9 bid/ask offset combinations
    Reward:            one of asymmetric / quadratic / sparse (see §4 MDP doc)

    Parameters
    ----------
    reward_type : str   — 'asymmetric' | 'quadratic' | 'sparse'
    eta         : float — dampening parameter for asymmetric reward (default 0.5)
    lam         : float — inventory penalty for quadratic reward (default 0.1)
    Q_max       : int   — hard inventory constraint |q| ≤ Q_max (default 10)
    tick_size   : float — price tick size in dollars (default 0.01)
    episode_len : int   — steps per episode (default 3900 = one trading day)
    kappa       : float — terminal inventory penalty coefficient (default 1.0)
    seed        : int

    Usage:
        env = LOBMarketMakingEnv(reward_type='asymmetric')
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        reward_type: str   = "asymmetric",
        eta:         float = 0.5,
        lam:         float = 0.1,
        Q_max:       int   = 10,
        tick_size:   float = 0.01,
        episode_len: int   = 3900,
        kappa:       float = 1.0,
        n_lob_levels: int  = 3,
        seed:        int   = 42,
    ):
        super().__init__()

        assert reward_type in ("asymmetric", "quadratic", "sparse"), \
            f"reward_type must be 'asymmetric', 'quadratic', or 'sparse', got {reward_type}"

        self.reward_type  = reward_type
        self.eta          = eta
        self.lam          = lam
        self.Q_max        = Q_max
        self.tick_size    = tick_size
        self.episode_len  = episode_len
        self.kappa        = kappa
        self.n_lob_levels = n_lob_levels
        self.seed_val     = seed

        # ── Action space: two 9-dim heads ─────
        self.action_space = spaces.MultiDiscrete([N_OFFSET_LEVELS, N_OFFSET_LEVELS])

        # ── Observation space: handcrafted feature vector ─────────────
        # Features (docs/mdp_formulation.md §2.1 + §2.2):
        #   Market (7): spread, mid_move, imbalance, signed_vol, vol, rsi,
        #               lob_depth (n_lob_levels × 2 sides)
        #   Private (6+K): inventory, bid_dist, ask_dist, outstanding_bid,
        #                  outstanding_ask, time_remaining
        # Total: ~17 dims (exact count depends on n_lob_levels)
        obs_dim = self._compute_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(obs_dim,), dtype=np.float32,
        )

        # ── Internal state ────────────────────────────────────────────
        self._step       = 0
        self._inventory  = 0
        self._cash       = 0.0
        self._mid_price  = 0.0
        self._prev_mid   = 0.0
        self._bid_dist   = 0.0
        self._ask_dist   = 0.0
        self._rng        = np.random.default_rng(seed)

        # Price and LOB history for feature computation
        self._price_history: list = []
        self._lob_history:   list = []

    def _compute_obs_dim(self) -> int:
        """Compute observation dimension from feature spec."""
        # Market features
        n_market  = 6 + 2 * self.n_lob_levels   # spread, move, imbal, svol, vol, rsi, depth
        # Private features
        n_private = 6                           # q, δ_b, δ_a, p_o^b, p_o^a, τ
        return n_market + n_private

    def _get_obs(self) -> np.ndarray:
        """
        Compute the handcrafted feature vector from current LOB state.

        Features in order (see docs/mdp_formulation.md §2.1–2.2):
            [bid_ask_spread, mid_price_move, queue_imbalance, signed_volume,
             realized_vol, rsi, lob_depth_bid_L1..LK, lob_depth_ask_L1..LK,
             inventory, bid_distance, ask_distance,
             outstanding_bid, outstanding_ask, time_remaining]

        All features normalised before returning.
        """
        # TODO: implement — extract features from ABIDES-Gym observation
        # and compute each feature listed above.
        raise NotImplementedError

    def _compute_reward(
        self,
        matched_bid: float,
        matched_ask: float,
        bid_price: float,
        ask_price: float,
        inventory: float,
        filled_both: bool = False,
        cross_spread_fill: bool = False,
    ) -> float:
        """
        Compute step reward from one of three formulations.

        Returns
        -------
        float reward
        """
        mid = self._mid_price
        prev_mid = self._prev_mid

        psi_a = matched_ask * (ask_price - mid)
        psi_b = matched_bid * (mid - bid_price)

        delta_m = mid - prev_mid

        inventory_pnl = inventory * delta_m

        pnl = (
            psi_a
            + psi_b
            + inventory_pnl
        )

        if self.reward_type == "asymmetric":
            return (
                pnl
                - max(
                    0.0,
                    self.eta * inventory_pnl
                )
            )

        elif self.reward_type == "sparse":
            if filled_both:
                return 1.0

            if cross_spread_fill:
                return -0.5

            return 0.0

        raise RuntimeError("Unknown reward type")

    def _terminal_reward(self) -> float:
        """
        Terminal inventory penalty applied at episode end.

        r_terminal = −κ · |q_T| · σ_T
        (docs/mdp_formulation.md §4 Terminal)
        """
        if len(self._price_history) < 2:
            return 0
    
        prices = np.asarray(self._price_history)

        returns = np.diff(np.log(prices))
        sigma_T = float(np.std(returns))
        return -self.kappa * abs(self._inventory) * sigma_T

    def reset(
        self,
        seed:    Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset environment to start of a new episode.

        Returns
        -------
        obs  : np.ndarray — initial observation
        info : dict       — metadata (mid_price, inventory, step)
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Reset internal state
        self._step       = 0
        self._inventory  = 0
        self._cash       = 0.0
        self._prev_mid   = 0.0
        self._bid_dist   = 0.0
        self._ask_dist   = 0.0
        self._price_history.clear()
        self._lob_history.clear()

        # TODO: reset ABIDES-Gym environment and get initial LOB state
        # obs_abides, info = self._abides_env.reset()
        # self._mid_price = self._extract_mid_price(obs_abides)
        # self._price_history.append(self._mid_price)

        obs  = self._get_obs()
        info = {
            "step":      self._step,
            "inventory": self._inventory,
            "mid_price": self._mid_price,
            "cash":      self._cash,
        }
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in the environment.

        Parameters
        ----------
        action : int — flat action index ∈ [0, 80]

        Returns
        -------
        obs        : np.ndarray — next observation
        reward     : float
        terminated : bool — True if episode ended naturally
        truncated  : bool — True if episode hit max length
        info       : dict — metadata for logging
        """
        # ── Decode action ──────────────────────────────────────────────
        bid_idx, ask_idx = action
        bid_offset = TICK_OFFSETS[bid_idx]   # ticks from mid
        ask_offset = TICK_OFFSETS[ask_idx]   # ticks from mid

        # ── Compute quote prices ───────────────────────────────────────
        bid_price = self._mid_price - bid_offset * self.tick_size
        ask_price = self._mid_price + ask_offset * self.tick_size

        # ── Submit quotes to ABIDES and advance one step ───────────────
        # TODO: submit bid_price/ask_price to ABIDES-Gym
        # obs_abides, _, _, _, abides_info = self._abides_env.step(...)
        # fills = abides_info.get('fills', {})

        # ── Extract fills and update inventory ─────────────────────────
        # TODO: extract from ABIDES fills
        bid_filled   = False   # TODO
        ask_filled   = False   # TODO
        filled_both = bid_filled and ask_filled

        cross_spread_fill = False

        spread_pnl   = 0.0    # TODO: cash received from fills
        prev_inv     = self._inventory
        # self._inventory += bid_filled * size - ask_filled * size

        # ── Enforce hard inventory constraint ──────────────────────────
        self._inventory = np.clip(self._inventory, -self.Q_max, self.Q_max)

        delta_q = abs(self._inventory - prev_inv)

        # ── Compute reward ─────────────────────────────────────────────
        reward = self._compute_reward(
            matched_bid=float(bid_filled),
            matched_ask=float(ask_filled),
            bid_price=bid_price,
            ask_price=ask_price,
            inventory=self._inventory,
            filled_both=filled_both,
            cross_spread_fill=cross_spread_fill,
        )

        # ── Advance step counter ───────────────────────────────────────
        self._step  += 1
        terminated   = False
        truncated    = self._step >= self.episode_len

        # ── Terminal reward ────────────────────────────────────────────
        if truncated or terminated:
            reward += self._terminal_reward()

        # ── Next observation ───────────────────────────────────────────
        obs  = self._get_obs()
        info = {
            "step":        self._step,
            "inventory":   self._inventory,
            "mid_price":   self._mid_price,
            "cash":        self._cash,
            "spread_pnl":  spread_pnl,
            "bid_filled":  bid_filled,
            "ask_filled":  ask_filled,
            "bid_price":   bid_price,
            "ask_price":   ask_price,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        """Optional: print current state for debugging."""
        print(
            f"Step {self._step:4d} | "
            f"Mid: {self._mid_price:.4f} | "
            f"Inventory: {self._inventory:+3d} | "
            f"Cash: {self._cash:.2f}"
        )