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
from collections import deque
from typing import Optional, Tuple, Dict, Any
import yaml

from abides_gym.envs.markets_execution_environment_v0 import SubGymMarketsExecutionEnv_v0

# Action space: bid/ask offsets in ticks
# δ ∈ {−4, −3, −2, −1, 0, +1, +2, +3, +4} → 9 levels per side → 81 total
TICK_OFFSETS = np.arange(-4, 5)   # shape (9,)
N_OFFSET_LEVELS = len(TICK_OFFSETS)

# Rolling-history window caps (avoid unbounded memory growth)
_PRICE_HISTORY_MAXLEN  = 500
_VOLUME_HISTORY_MAXLEN = 500
_LOB_HISTORY_MAXLEN    = 100

class LOBMarketMakingEnv(gym.Env):
    """
    Limit Order Book market-making environment.

    Observation space: handcrafted feature vector (~17 dims, see §2 MDP doc)
    Action space:      MultiDiscrete([9, 9]) — 9 bid-offset × 9 ask-offset
                       combinations (equivalent to 81 joint actions).
                       Bid index selects δ_b ∈ {−4,…,+4} ticks below mid.
                       Ask index selects δ_a ∈ {−4,…,+4} ticks above mid.
    Reward:            one of asymmetric / quadratic / sparse (see §4 MDP doc)

    Parameters
    ----------
    reward_type : str   — 'asymmetric' | 'quadratic' | 'sparse'
    eta         : float — inventory-PnL dampening for asymmetric reward (default 0.5)
    lam         : float — quadratic inventory penalty coefficient (default 0.1)
    Q_max       : int   — hard inventory constraint |q| ≤ Q_max (default 10)
    tick_size   : float — price tick size in dollars (default 0.01)
    episode_len : int   — steps per episode (default 3900 = one trading day)
    kappa       : float — terminal inventory penalty coefficient (default 1.0)
    n_lob_levels: int   — number of LOB depth levels to include (default 3)
    seed        : int

    Usage
    -----
        env = LOBMarketMakingEnv(reward_type='asymmetric')
        obs, info = env.reset()
        obs, reward, terminated, truncated, info = env.step(action)

    ABIDES-Gym wiring
    -----------------
    Three abstract hooks must be implemented when attaching the real simulator:

        _extract_mid_price(abides_obs)         → float
        _encode_abides_action(bid, ask)        → abides action object
        _parse_abides_step(abides_info)        → dict with keys:
            bid_qty, ask_qty, signed_volume, lob_snapshot

    Until those are wired, the environment runs a synthetic GBM mid-price
    and returns zero fills so the feature pipeline and reward maths can be
    exercised in isolation.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        config: str | None = None,
        reward_type: str = "asymmetric",
        eta: float = 0.5,
        lam: float = 0.1,
        Q_max: int = 10,
        tick_size: float = 0.01,
        episode_len: int = 3900,
        kappa: float = 1.0,
        n_lob_levels: int = 3,
        seed: int = 42,
    ):
        super().__init__()

        self._abides_env = AbidesMarketMakingEnv(background_config="rmsc04")

        cfg = {}
        if config is not None:
            with open(config, "r") as f:
                cfg = yaml.safe_load(f)

        self.reward_type = cfg.get("reward_type", reward_type)
        self.eta = cfg.get("eta", eta)
        self.lam = cfg.get("lam", lam)
        self.Q_max = cfg.get("Q_max", Q_max)
        self.tick_size = cfg.get("tick_size", tick_size)
        self.episode_len = cfg.get("episode_len", episode_len)
        self.kappa = cfg.get("kappa", kappa)
        self.n_lob_levels = cfg.get("n_lob_levels", n_lob_levels)
        self.seed_val = cfg.get("seed", seed)
                        
        assert self.reward_type in ("asymmetric", "quadratic", "sparse"), (f"reward_type must be 'asymmetric', 'quadratic', or 'sparse', "f"got '{self.reward_type}'")

        # ── Action space: two 9-dim heads (MultiDiscrete) ─────────────
        self.action_space = spaces.MultiDiscrete([N_OFFSET_LEVELS, N_OFFSET_LEVELS])

        # ── Observation space ─────────────────────────────────────────
        # Features (docs/mdp_formulation.md §2.1 + §2.2):
        #   Market  (6 + 2·K): spread, mid_move, imbalance, signed_vol,
        #                       vol, rsi, bid_depths[K], ask_depths[K]
        #   Private (6):        inventory, bid_dist, ask_dist,
        #                       outstanding_bid, outstanding_ask, time_remaining
        self._obs_dim = self._compute_obs_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._obs_dim,), dtype=np.float32,
        )

        # ── Internal state ────────────────────────────────────────────
        self._rng = np.random.default_rng(seed)
        self._reset_state()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_obs_dim(self) -> int:
        """Observation dimension from feature spec (cached in __init__)."""
        n_market  = 6 + 2 * self.n_lob_levels
        n_private = 6
        return n_market + n_private

    def _reset_state(self) -> None:
        """Zero / clear all mutable episode state."""
        self._step            = 0
        self._inventory       = 0
        self._cash            = 0.0
        self._mid_price       = 0.0
        self._prev_mid        = 0.0
        self._bid_dist        = 0.0
        self._ask_dist        = 0.0
        self._outstanding_bid = 0.0
        self._outstanding_ask = 0.0

        # Capped rolling histories (avoid unbounded memory growth)
        self._price_history:  deque = deque(maxlen=_PRICE_HISTORY_MAXLEN)
        self._volume_history: deque = deque(maxlen=_VOLUME_HISTORY_MAXLEN)
        self._lob_history:    deque = deque(maxlen=_LOB_HISTORY_MAXLEN)

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------

    def _compute_rsi(self, window: int = 14) -> float:
        """
        Relative Strength Index from price history.

        RSI = 100 − 100 / (1 + avg_gain / avg_loss)

        Returns 50.0 (neutral) when history is insufficient.
        """
        if len(self._price_history) < window + 1:
            return 50.0

        prices   = np.asarray(list(self._price_history)[-(window + 1):])
        deltas   = np.diff(prices)
        gains    = np.where(deltas > 0, deltas, 0.0)
        losses   = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = gains.mean()
        avg_loss = losses.mean()

        if avg_loss < 1e-10:
            return 100.0
        rs = avg_gain / avg_loss
        return float(100.0 - 100.0 / (1.0 + rs))

    def _compute_realized_vol(self, window: int = 20) -> float:
        """
        Realized volatility: std of log-returns over the last `window` steps.

        Returns 0.0 when history is insufficient.
        """
        prices = np.asarray(list(self._price_history))
        if len(prices) < 2:
            return 0.0

        prices  = prices[-min(window + 1, len(prices)):]
        log_ret = np.diff(np.log(np.maximum(prices, 1e-10)))
        return float(np.std(log_ret))

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """
        Build the handcrafted feature vector from current LOB state.

        Index layout (docs/mdp_formulation.md §2.1–2.2):
        ┌──────────────────┬────────────────────────────────────────────────┐
        │ idx              │ feature                                        │
        ├──────────────────┼────────────────────────────────────────────────┤
        │  0               │ bid-ask spread / tick, clipped [0, 10]         │
        │  1               │ mid log-return this step / tick, clipped ±10   │
        │  2               │ queue imbalance I=(V_b−V_a)/(V_b+V_a) ∈ [−1,1] │
        │  3               │ signed volume, normalised, clipped ±1          │
        │  4               │ realized vol (20-step), scaled by tick         │
        │  5               │ RSI (14-step), normalised to [−1, +1]          │
        │  6 .. 6+K        │ LOB bid depth levels L1..LK, per-side norm     │
        │  6+K .. 6+2K     │ LOB ask depth levels L1..LK, per-side norm     │
        │  base+0          │ inventory q / Q_max ∈ [−1, +1]                 │
        │  base+1          │ active bid distance δ_b / 4 ∈ [−1, +1]         │
        │  base+2          │ active ask distance δ_a / 4 ∈ [−1, +1]         │
        │  base+3          │ outstanding bid offset from mid, normalised    │
        │  base+4          │ outstanding ask offset from mid, normalised    │
        │  base+5          │ time remaining τ = (T−t)/T ∈ [0, 1]            │
        └──────────────────┴────────────────────────────────────────────────┘

        LOB depth and queue imbalance are zero until ABIDES is wired in;
        the rest are live from internal state.
        """
        obs = np.zeros(self._obs_dim, dtype=np.float32)

        # ── Market features ───────────────────────────────────────────

        # [0] Bid-ask spread normalised by tick_size, clipped [0, 10]
        #     Uses outstanding quotes as best-bid/ask proxy.
        raw_spread = 0.0
        if self._lob_history:
            snap = self._lob_history[-1]

            bid_prices, ask_prices = snap["bid_prices"], snap["ask_prices"]

            if len(bid_prices) > 0 and len(ask_prices) > 0:
                best_bid, best_ask = bid_prices[0], ask_prices[0]

                raw_spread = (best_ask - best_bid) / self.tick_size
        obs[0] = float(np.clip(raw_spread / 10.0, 0.0, 10.0))

        # [1] Mid-price log-return this step, scaled by tick_size, clipped ±10
        log_ret = 0.0
        if self._prev_mid > 0 and self._mid_price > 0:
            log_ret = np.log(self._mid_price / self._prev_mid) / self.tick_size
        obs[1] = float(np.clip(log_ret, -10.0, 10.0))

        # [2] Queue imbalance I = (V_b − V_a) / (V_b + V_a)
        #     Populated from _lob_history once ABIDES is wired in.
        imbalance = 0.0
        if self._lob_history:
            snap      = self._lob_history[-1]
            bid_vol   = float(np.sum(snap.get("bid_sizes", [0])))
            ask_vol   = float(np.sum(snap.get("ask_sizes", [0])))
            total_vol = bid_vol + ask_vol
            imbalance = (bid_vol - ask_vol) / total_vol if total_vol > 0 else 0.0
        obs[2] = float(np.clip(imbalance, -1.0, 1.0))

        # [3] Signed volume normalised by rolling max, clipped ±1
        obs[3] = 0.0
        if self._volume_history:
            signed_vol = float(self._volume_history[-1])
            vol_scale  = max(abs(v) for v in self._volume_history) + 1e-8
            obs[3]     = float(np.clip(signed_vol / vol_scale, -1.0, 1.0))

        # [4] Realized volatility (20-step window), scaled by tick_size, clipped [0, 50]
        obs[4] = float(np.clip(
            self._compute_realized_vol() / max(self.tick_size, 1e-8),
            0.0, 50.0,
        ))

        # [5] RSI normalised from [0, 100] → [−1, +1]
        obs[5] = float((self._compute_rsi() - 50.0) / 50.0)

        # [6 : 6+K]   LOB bid-side depth at levels L1..LK
        # [6+K : 6+2K] LOB ask-side depth at levels L1..LK
        # Normalised per-side so proportions across levels sum to 1.
        # (Per-side normalisation is standard in LOB ML literature;
        #  cross-side normalisation conflates bid and ask liquidity.)
        bid_depths = np.zeros(self.n_lob_levels, dtype=np.float32)
        ask_depths = np.zeros(self.n_lob_levels, dtype=np.float32)
        if self._lob_history:
            snap       = self._lob_history[-1]
            bid_depths = np.asarray(
                snap.get("bid_sizes", [0] * self.n_lob_levels), dtype=np.float32
            )[:self.n_lob_levels]
            ask_depths = np.asarray(
                snap.get("ask_sizes", [0] * self.n_lob_levels), dtype=np.float32
            )[:self.n_lob_levels]

            # Pad to n_lob_levels if snapshot has fewer levels
            if len(bid_depths) < self.n_lob_levels:
                bid_depths = np.pad(bid_depths, (0, self.n_lob_levels - len(bid_depths)))
            if len(ask_depths) < self.n_lob_levels:
                ask_depths = np.pad(ask_depths, (0, self.n_lob_levels - len(ask_depths)))

            # Per-side normalisation: proportions within each side sum to 1
            bid_total  = bid_depths.sum() + 1e-8
            ask_total  = ask_depths.sum() + 1e-8
            bid_depths = bid_depths / bid_total
            ask_depths = ask_depths / ask_total

        obs[6 : 6 + self.n_lob_levels]                          = bid_depths
        obs[6 + self.n_lob_levels : 6 + 2 * self.n_lob_levels] = ask_depths

        # ── Private features ──────────────────────────────────────────
        base = 6 + 2 * self.n_lob_levels

        # [base+0] Inventory normalised to [−1, +1]
        obs[base + 0] = float(np.clip(self._inventory / self.Q_max, -1.0, 1.0))

        # [base+1] Active bid-offset from mid (ticks), normalised by max offset
        obs[base + 1] = float(np.clip(self._bid_dist / 4.0, -1.0, 1.0))

        # [base+2] Active ask-offset from mid (ticks), normalised by max offset
        obs[base + 2] = float(np.clip(self._ask_dist / 4.0, -1.0, 1.0))

        # [base+3] Outstanding bid price offset from mid, normalised by Q_max·tick
        #          (Sun et al. 2022 private-state feature)
        if self._mid_price > 0 and self._outstanding_bid > 0:
            bid_offset_norm = (self._outstanding_bid - self._mid_price) / \
                              (self.Q_max * self.tick_size)
            obs[base + 3] = float(np.clip(bid_offset_norm, -1.0, 1.0))

        # [base+4] Outstanding ask price offset from mid, normalised by Q_max·tick
        if self._mid_price > 0 and self._outstanding_ask > 0:
            ask_offset_norm = (self._outstanding_ask - self._mid_price) / \
                              (self.Q_max * self.tick_size)
            obs[base + 4] = float(np.clip(ask_offset_norm, -1.0, 1.0))

        # [base+5] Time remaining τ = (T − t) / T ∈ [0, 1]
        obs[base + 5] = float(1.0 - self._step / self.episode_len)

        return obs

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        matched_bid:      float,
        matched_ask:      float,
        bid_price:        float,
        ask_price:        float,
        inventory:        float,
        filled_both:      bool = False,
        cross_spread_fill: bool = False,
    ) -> float:
        """
        Compute step reward from one of three formulations.

        Notation follows docs/mdp_formulation.md §4:

            ψ_a = matched_ask  · (ask_price  − mid)   ← ask half-spread capture
            ψ_b = matched_bid  · (mid        − bid_price) ← bid half-spread capture
            ΔM  = mid − prev_mid                       ← mid-price move
            inv_pnl = q · ΔM                           ← mark-to-market inventory PnL

        Asymmetric (Avellaneda–Stoikov flavour)
        ----------------------------------------
            r = PnL − η · max(0, inv_pnl)
            Penalises only adverse inventory PnL; lets favourable moves pass through.
            Encourages the agent to reduce inventory *before* adverse moves.

        Quadratic (Cartea–Jaimungal flavour)
        -------------------------------------
            r = PnL − λ · q²
            Symmetric quadratic penalty on inventory level at every step.
            λ controls the risk-aversion coefficient; larger λ → tighter quotes
            and faster inventory mean-reversion.

        Sparse
        ------
            r = +1.0 if both sides filled in the same step (round-trip)
            r = −0.5 if a fill crosses the spread (adverse selection signal)
            r =  0.0 otherwise
            Useful as a diagnostic / sanity-check formulation.

        Parameters
        ----------
        matched_bid      : quantity filled on the bid side this step
        matched_ask      : quantity filled on the ask side this step
        bid_price        : price of the submitted bid quote
        ask_price        : price of the submitted ask quote
        inventory        : current inventory *after* fills (= self._inventory)
        filled_both      : True if both sides filled > 0 in this step
        cross_spread_fill: True if a fill occurred at an adverse price

        Returns
        -------
        float reward for this step
        """
        mid      = self._mid_price
        prev_mid = self._prev_mid

        # Half-spread captures from fills
        psi_a = matched_ask * (ask_price - mid)
        psi_b = matched_bid * (mid - bid_price)

        # Inventory mark-to-market PnL
        delta_m       = mid - prev_mid
        inventory_pnl = inventory * delta_m

        pnl = psi_a + psi_b + inventory_pnl

        if self.reward_type == "asymmetric":
            # Penalise adverse (negative) inventory PnL only.
            # When eta=0 → pure PnL; when eta=1 → fully hedge inventory risk.
            return pnl - max(0.0, self.eta * inventory_pnl)

        if self.reward_type == "quadratic":
            # Symmetric quadratic inventory penalty at every step.
            # q² is bounded in [0, Q_max²]; lambda scales the risk aversion.
            return pnl - self.lam * (inventory ** 2)

        if self.reward_type == "sparse":
            if filled_both:
                return 1.0          # captured the full spread round-trip
            if cross_spread_fill:
                return -0.5         # adverse selection: filled on wrong side of mid
            return 0.0

        raise RuntimeError(f"Unknown reward_type: '{self.reward_type}'")

    def _terminal_reward(self) -> float:
        """
        Terminal inventory penalty applied at episode end.

        r_terminal = −κ · |q_T| · σ_T

        where σ_T is the realized volatility over the full episode.
        This penalises residual inventory proportional to its liquidation risk,
        incentivising the agent to reach q=0 before the close.

        (docs/mdp_formulation.md §4 Terminal)
        """
        if len(self._price_history) < 2:
            return 0.0

        prices  = np.asarray(list(self._price_history))
        returns = np.diff(np.log(np.maximum(prices, 1e-10)))
        sigma_T = float(np.std(returns))
        return -self.kappa * abs(self._inventory) * sigma_T

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed:    Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        Reset environment to start of a new episode.

        Returns
        -------
        obs  : np.ndarray — initial observation (all market features zero at t=0)
        info : dict       — metadata (mid_price, inventory, step, cash)
        """
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._reset_state()

        _ = self._abides_env.reset()  # returns processed obs array — ignore it

        # Read raw_state directly from the gym agent after reset
        # raw_state is a deque; [-1] is the most recent snapshot
        raw_state = self._abides_env.gym_agent.raw_state[-1]

        self._mid_price = self._extract_mid_price(raw_state)
        self._prev_mid  = self._mid_price
        self._price_history.append(self._mid_price)

        parsed = self._parse_abides_step(raw_state)
        if parsed["lob_snapshot"]["bid_sizes"]:
            self._lob_history.append(parsed["lob_snapshot"])

        obs  = self._get_obs()
        info = {
            "step":      self._step,
            "inventory": self._inventory,
            "mid_price": self._mid_price,
            "cash":      parsed["cash"],
        }
        return obs, info

    def step(self, action):
        bid_idx, ask_idx = int(action[0]), int(action[1])
        bid_offset = int(TICK_OFFSETS[bid_idx])
        ask_offset = int(TICK_OFFSETS[ask_idx])
        bid_price  = self._mid_price - bid_offset * self.tick_size
        ask_price  = self._mid_price + ask_offset * self.tick_size
        self._bid_dist = float(bid_offset)
        self._ask_dist = float(ask_offset)

        abides_action       = self._encode_abides_action(bid_price, ask_price)
        _, _, done, _       = self._abides_env.step(abides_action)  # 4-tuple old gym API

        # Read raw_state from agent after step (same pattern as reset)
        raw_state = self._abides_env.gym_agent.raw_state[-1]

        parsed    = self._parse_abides_step(raw_state)
        new_mid   = self._extract_mid_price(raw_state)

        bid_filled = parsed["bid_qty"]
        ask_filled = parsed["ask_qty"]
        signed_vol = parsed["signed_volume"]
        lob_snap   = parsed["lob_snapshot"]

        # ── Update internal state ──────────────────────────────────────
        self._prev_mid  = self._mid_price
        self._mid_price = max(new_mid, self.tick_size)   # price must stay positive
        self._price_history.append(self._mid_price)
        self._volume_history.append(signed_vol)
        if lob_snap:
            self._lob_history.append(lob_snap)

        self._outstanding_bid = bid_price
        self._outstanding_ask = ask_price

        # Inventory update (clamp to [−Q_max, +Q_max])
        prev_inventory  = self._inventory
        self._inventory = int(np.clip(
            self._inventory + bid_filled - ask_filled,
            -self.Q_max, self.Q_max,
        ))
        delta_q = abs(self._inventory - prev_inventory)   # shares actually transacted

        # Cash update: receive ask_price for sells, pay bid_price for buys
        self._cash += ask_filled * ask_price - bid_filled * bid_price

        # Spread PnL: half-spread capture from fills
        spread_pnl = (
            ask_filled * (ask_price - self._mid_price) +
            bid_filled * (self._mid_price - bid_price)
        )

        # ── Compute reward ─────────────────────────────────────────────
        filled_both = bid_filled > 0 and ask_filled > 0
        cross_spread_fill = (
            (bid_filled > 0 and bid_price > self._mid_price) or
            (ask_filled > 0 and ask_price < self._mid_price)
        )

        reward = self._compute_reward(
            matched_bid=float(bid_filled),
            matched_ask=float(ask_filled),
            bid_price=bid_price,
            ask_price=ask_price,
            inventory=float(self._inventory),
            filled_both=filled_both,
            cross_spread_fill=cross_spread_fill,
        )

        # ── Termination / truncation ───────────────────────────────────
        self._step += 1
        terminated  = False          # reserved for ABIDES market-close signal
        truncated   = self._step >= self.episode_len

        if terminated or truncated:
            reward += self._terminal_reward()

        # ── Build next observation ─────────────────────────────────────
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
            "bid_offset":  bid_offset,
            "ask_offset":  ask_offset,
            "delta_q":     delta_q,
            "reward":      reward,
        }
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Gymnasium bookkeeping
    # ------------------------------------------------------------------

    def render(self, mode: str = "human") -> None:
        """Print a concise state summary for debugging."""
        vol = self._compute_realized_vol()
        rsi = self._compute_rsi()
        print(
            f"[{self._step:>4d}/{self.episode_len}] "
            f"Mid: {self._mid_price:>8.4f}  "
            f"Bid: {self._outstanding_bid:>8.4f} (+{self._bid_dist:+.0f}t)  "
            f"Ask: {self._outstanding_ask:>8.4f} (+{self._ask_dist:+.0f}t)  "
            f"Inv: {self._inventory:>+3d}/{self.Q_max}  "
            f"Cash: {self._cash:>+10.2f}  "
            f"σ: {vol:.5f}  RSI: {rsi:.1f}"
        )

    def close(self) -> None:
        """
        Clean up resources.
        """
        if hasattr(self, "_abides_env"):
            self._abides_env.close()

    # ------------------------------------------------------------------
    # ABIDES-Gym wiring hooks  (implement when attaching real simulator)
    # ------------------------------------------------------------------
    def _extract_mid_price(self, raw_state: dict) -> float:
        """
        Extract mid-price from a single ABIDES raw_state dict.

        raw_state["parsed_mkt_data"]["bids"] is a list of (price, volume) tuples,
        best-first. "last_transaction" is the scalar fallback when book is empty.
        """
        mkt      = raw_state["parsed_mkt_data"][-1]   # ← deque, take latest
        bids             = mkt["bids"]
        asks             = mkt["asks"]
        last_transaction = mkt["last_transaction"]

        best_bid = bids[0][0] if len(bids) > 0 else last_transaction
        best_ask = asks[0][0] if len(asks) > 0 else last_transaction

        return 0.5 * (best_bid + best_ask) / 100.0

    def _encode_abides_action(self, bid_price: float, ask_price: float) -> int:
        """
        Store bid/ask prices (dollars) on the sub-env for pickup by
        _map_action_space_to_ABIDES_SIMULATOR_SPACE, then return the
        integer token that ABIDES action_space.contains() expects.
        """
        self._abides_env._pending_bid_price = int(round(bid_price * 100))
        self._abides_env._pending_ask_price = int(round(ask_price * 100))
        return 0  # dummy token, always 0

    def _parse_abides_step(self, raw_state: dict) -> dict:
        """
        Extract fill and market data from a single ABIDES raw_state dict.

        raw_state["internal_data"]["inter_wakeup_executed_orders"] is a list
        of Order objects with attributes:
            .fill_price  (int, in ABIDES cent units)
            .quantity    (int, always positive)
            .side        "BID" | "ASK"  (or check direction from order_status)

        bids/asks are lists of (price, volume) tuples, best-first, up to
        subscribe_num_levels deep (default 10).
        """
        mkt      = raw_state["parsed_mkt_data"][-1]   # ← deque, take latest
        internal = raw_state["internal_data"]

        bids             = mkt["bids"]
        asks             = mkt["asks"]

        fills     = internal.get("inter_wakeup_executed_orders", [])
        bid_qty   = sum(o.quantity for o in fills if o.side.value == "BID")
        ask_qty   = sum(o.quantity for o in fills if o.side.value == "ASK")
        signed_volume = bid_qty - ask_qty

        K = self.n_lob_levels
        lob_snapshot = {
            "bid_prices": [b[0] / 100.0 for b in bids[:K]],
            "bid_sizes":  [b[1]         for b in bids[:K]],
            "ask_prices": [a[0] / 100.0 for a in asks[:K]],
            "ask_sizes":  [a[1]         for a in asks[:K]],
        }

        return {
            "bid_qty":       bid_qty,
            "ask_qty":       ask_qty,
            "signed_volume": signed_volume,
            "lob_snapshot":  lob_snapshot,
            "cash":          internal.get("cash", 0.0),
            "holdings":      internal.get("holdings", 0),
        }

class AbidesMarketMakingEnv(SubGymMarketsExecutionEnv_v0):
    """Thin subclass that overrides action mapping for two-sided LMT quoting."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Single dummy action token — we only ever pass 0
        self.action_space = gym.spaces.Discrete(1)
        self._pending_bid_price: int = 0  # cents
        self._pending_ask_price: int = 0  # cents

        # ABIDES declares tight bounds on its own obs space but background agents regularly push features (holdings_pct, time_pct, etc.) outside them. Widen to float32 max to suppress the internal contains() assert.
        n = self.observation_space.shape[0]
        self.observation_space = gym.spaces.Box(
            low  = -np.finfo(np.float32).max,
            high =  np.finfo(np.float32).max,
            shape = self.observation_space.shape,
            dtype = np.float32,
        )

    def _map_action_space_to_ABIDES_SIMULATOR_SPACE(self, action: int):
        return [
            {"type": "CCL_ALL"},
            {"type": "LMT", "direction": "BUY",  "size": 1,
             "limit_price": self._pending_bid_price},
            {"type": "LMT", "direction": "SELL", "size": 1,
             "limit_price": self._pending_ask_price},
        ]