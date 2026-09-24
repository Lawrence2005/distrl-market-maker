"""
envs/multi_agent_env.py

Multi-agent LOB environment for simultaneous market maker competition.

Runs N RL agents quoting simultaneously in one ABIDES simulation.
At each step all agents act, all quotes are submitted, simulation
advances once, each agent receives its own obs/reward/info.

Usage
-----
    from envs.multi_agent_env import MultiAgentMarketEnv

    env    = MultiAgentMarketEnv(n_agents=2, episode_len=390)
    obs_n  = env.reset()                        # list of N obs arrays
    while True:
        actions_n  = [agent_i.act(obs_n[i]) for i in range(env.n_agents)]
        obs_n, rewards_n, dones_n, infos_n = env.step(actions_n)
        if all(dones_n):
            break
    env.close()

Research questions (Week 6 pilot, Week 8 full tournament)
----------------------------------------------------------
    - Does competition tighten spreads? (market_spread should decrease with N)
    - Does it destabilise the LOB? (price_vol should increase with N)
    - Do agents find a stable equilibrium or keep undercutting each other?

Week 6 deliverable.
"""

from __future__ import annotations

from collections import deque
from typing import Optional

import gymnasium as gym
import numpy as np

from envs.lob_env import (
    LOBMarketMakingEnv,
    AbidesMarketMakingEnv,
    TICK_OFFSETS,
    N_OFFSET_LEVELS,
    parse_abides_raw_state,
    RAW_SNAPSHOT_LEVELS,
    compute_rsi,
    compute_realized_vol,
    _PRICE_HISTORY_MAXLEN,
    _VOLUME_HISTORY_MAXLEN,
    _LOB_HISTORY_MAXLEN,
)


def _argmin_random_tiebreak(dists: np.ndarray, rng: np.random.Generator) -> int:
    """np.argmin, but ties broken uniformly at random instead of always
    toward the lowest index — see parse_fills() for why this matters."""
    candidates = np.flatnonzero(np.isclose(dists, dists.min()))
    if len(candidates) == 1:
        return int(candidates[0])
    return int(rng.choice(candidates))


class MultiAgentMarketEnv:
    """
    N simultaneous market makers in one ABIDES simulation.

    Each agent has its own inventory, cash, and reward stream.
    All agents share the same LOB — their quotes compete for fills.

    Parameters
    ----------
    n_agents    : int   — number of simultaneous market makers (default 2)
    episode_len : int   — steps per episode (default 390)
    Q_max       : int   — per-agent inventory constraint, in shares (default 1000)
    order_size  : int   — shares per submitted order (default 100)
    tick_size   : float — dollar value of one tick (default 0.01)
    reward_type : str   — reward formulation (default 'asymmetric')
    eta         : float — asymmetric reward dampening (default 0.5)
    use_abides  : bool  — use ABIDES or synthetic GBM (default True)
    seed        : int   — random seed (default 42)
    """

    def __init__(
        self,
        n_agents:    int   = 2,
        episode_len: int   = 390,
        Q_max:       int   = 1000,
        order_size:  int   = 100,
        tick_size:   float = 0.01,
        reward_type: str   = "asymmetric",
        eta:         float = 0.5,
        use_abides:  bool  = True,
        seed:        int   = 42,
    ):
        self.n_agents    = n_agents
        self.episode_len = episode_len
        self.Q_max       = Q_max
        self.order_size  = order_size
        self.tick_size   = tick_size
        self.reward_type = reward_type
        self.eta         = eta
        self.use_abides  = use_abides
        self.seed        = seed

        # One shared ABIDES env (single simulation kernel)
        # N agents submit quotes via N pending bid/ask price slots
        if use_abides:
            self._abides = _MultiAgentAbidesEnv(
                n_agents    = n_agents,
                order_size  = order_size,
                background_config = "rmsc04",
            )
        else:
            self._abides = None

        # Per-agent state
        self._inventories  = np.zeros(n_agents, dtype=np.int32)
        self._cash         = np.zeros(n_agents, dtype=np.float64)
        self._mid_price    = 0.0
        self._prev_mid     = 0.0   # shared previous mid, for obs[1] log-return
        self._prev_mids    = np.zeros(n_agents, dtype=np.float64)
        self._step         = 0
        self._rng          = np.random.default_rng(seed)

        # Per-agent quoting state (mirrors LOBMarketMakingEnv._bid_dist etc.)
        self._bid_dist         = np.zeros(n_agents, dtype=np.float64)
        self._ask_dist         = np.zeros(n_agents, dtype=np.float64)
        self._outstanding_bid  = np.zeros(n_agents, dtype=np.float64)
        self._outstanding_ask  = np.zeros(n_agents, dtype=np.float64)

        # Shared market state — one LOB, so these are common to all agents
        # (mirrors LOBMarketMakingEnv._get_obs()'s feature layout, envs/lob_env.py)
        self.n_lob_levels    = 3   # matches obs_dim=18 = 6 + 2*3 + 6
        self._price_history:  deque = deque(maxlen=_PRICE_HISTORY_MAXLEN)
        self._volume_history: deque = deque(maxlen=_VOLUME_HISTORY_MAXLEN)
        self._lob_history:    deque = deque(maxlen=_LOB_HISTORY_MAXLEN)

        # GBM params for synthetic fallback
        self._gbm_price    = 1000.0
        self._gbm_sigma    = 0.001

        # Observation space per agent (same as single-agent handcrafted)
        obs_dim = 18
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.MultiDiscrete(
            [N_OFFSET_LEVELS, N_OFFSET_LEVELS]
        )

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
    ) -> list[np.ndarray]:
        """
        Reset environment. Returns list of N initial observations.

        Parameters
        ----------
        seed : int | None

        Returns
        -------
        list[np.ndarray] — one obs per agent, shape (obs_dim,)
        """
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._inventories = np.zeros(self.n_agents, dtype=np.int32)
        self._cash        = np.zeros(self.n_agents, dtype=np.float64)
        self._step        = 0
        self._gbm_price   = 1000.0

        self._bid_dist        = np.zeros(self.n_agents, dtype=np.float64)
        self._ask_dist        = np.zeros(self.n_agents, dtype=np.float64)
        self._outstanding_bid = np.zeros(self.n_agents, dtype=np.float64)
        self._outstanding_ask = np.zeros(self.n_agents, dtype=np.float64)

        self._price_history  = deque(maxlen=_PRICE_HISTORY_MAXLEN)
        self._volume_history = deque(maxlen=_VOLUME_HISTORY_MAXLEN)
        self._lob_history    = deque(maxlen=_LOB_HISTORY_MAXLEN)

        if self._abides is not None:
            # .reset() returns the gym obs array, not the raw_state dict —
            # pull the actual raw_state off gym_agent, same as
            # LOBMarketMakingEnv.reset() (envs/lob_env.py) does.
            _ = self._abides.reset()
            raw_state = self._abides.gym_agent.raw_state[-1]
            self._mid_price = self._extract_mid(raw_state)
            parsed = parse_abides_raw_state(raw_state, RAW_SNAPSHOT_LEVELS)
            if parsed["lob_snapshot"]["bid_sizes"]:
                self._lob_history.append(parsed["lob_snapshot"])
        else:
            self._mid_price = self._gbm_price

        self._prev_mid  = self._mid_price
        self._price_history.append(self._mid_price)
        self._prev_mids = np.full(self.n_agents, self._mid_price)

        return [self._get_obs(i) for i in range(self.n_agents)]

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        actions: list[np.ndarray],
    ) -> tuple[list, list, list, list]:
        """
        All N agents act simultaneously.

        Parameters
        ----------
        actions : list of N arrays, each shape (2,) = [bid_idx, ask_idx]

        Returns
        -------
        obs_n     : list[np.ndarray] — one obs per agent
        rewards_n : list[float]      — one reward per agent
        dones_n   : list[bool]       — True for all if episode ended
        infos_n   : list[dict]       — per-agent info dict
        """
        assert len(actions) == self.n_agents

        prev_mid_shared = self._mid_price

        # Convert indices → prices for each agent
        bid_prices = []
        ask_prices = []
        for i, action in enumerate(actions):
            bid_idx = int(action[0])
            ask_idx = int(action[1])
            bid_offset = int(TICK_OFFSETS[bid_idx])
            ask_offset = int(TICK_OFFSETS[ask_idx])
            bid_prices.append(self._mid_price - bid_offset * self.tick_size)
            ask_prices.append(self._mid_price + ask_offset * self.tick_size)
            self._bid_dist[i]        = float(bid_offset)
            self._ask_dist[i]        = float(ask_offset)
            self._outstanding_bid[i] = bid_prices[i]
            self._outstanding_ask[i] = ask_prices[i]

        # Submit all quotes and advance simulation one step
        if self._abides is not None:
            raw_state  = self._abides.step_multi(bid_prices, ask_prices)
            new_mid    = self._extract_mid(raw_state)
            fills      = self._abides.parse_fills(raw_state, self.n_agents, rng=self._rng)
            parsed     = parse_abides_raw_state(raw_state, RAW_SNAPSHOT_LEVELS)
            signed_vol = parsed["signed_volume"]
            lob_snap   = parsed["lob_snapshot"]
        else:
            new_mid    = self._gbm_step()
            fills      = self._synthetic_fills(bid_prices, ask_prices)
            signed_vol = 0
            lob_snap   = {}

        self._prev_mid = prev_mid_shared
        self._price_history.append(new_mid)
        self._volume_history.append(signed_vol)
        if lob_snap:
            self._lob_history.append(lob_snap)

        self._step += 1
        done = self._step >= self.episode_len

        obs_n     = []
        rewards_n = []
        infos_n   = []

        for i in range(self.n_agents):
            bid_filled = fills[i]["bid_qty"]
            ask_filled = fills[i]["ask_qty"]

            prev_inv = self._inventories[i]
            self._inventories[i] = int(np.clip(
                self._inventories[i] + bid_filled - ask_filled,
                -self.Q_max, self.Q_max,
            ))
            self._cash[i] += (
                ask_filled * ask_prices[i] - bid_filled * bid_prices[i]
            )

            # Step PnL
            spread_pnl = (
                ask_filled * (ask_prices[i] - new_mid) +
                bid_filled * (new_mid - bid_prices[i])
            )
            # Mark-to-market against PRE-fill inventory: shares that just
            # transacted this step are already compensated via spread_pnl at
            # their actual fill price, so marking them against this step's
            # mid-move too would double-count (bid_filled-ask_filled)*ΔM —
            # see envs/lob_env.py's _compute_reward for the same fix.
            inv_pnl = prev_inv * (new_mid - self._prev_mids[i])
            pnl     = spread_pnl + inv_pnl

            reward  = self._compute_reward(pnl, inv_pnl, prev_inv)

            self._prev_mids[i] = new_mid

            infos_n.append({
                "inventory":   int(self._inventories[i]),
                "mid_price":   new_mid,
                "cash":        float(self._cash[i]),
                "spread_pnl":  spread_pnl,
                "bid_filled":  bid_filled,
                "ask_filled":  ask_filled,
                "bid_price":   bid_prices[i],
                "ask_price":   ask_prices[i],
                "agent_id":    i,
            })
            rewards_n.append(float(reward))

        self._mid_price = new_mid
        obs_n   = [self._get_obs(i) for i in range(self.n_agents)]
        dones_n = [done] * self.n_agents

        return obs_n, rewards_n, dones_n, infos_n

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(
        self,
        pnl:       float,
        inv_pnl:   float,
        inventory: int,
    ) -> float:
        if self.reward_type == "asymmetric":
            # Penalise adverse (negative) inventory PnL only — matches
            # envs/lob_env.py's asymmetric reward (that one is correct;
            # this reimplementation had the sign backwards).
            return pnl - self.eta * max(0.0, -inv_pnl)
        if self.reward_type == "quadratic":
            return pnl - 0.1 * inventory ** 2
        return pnl

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self, agent_id: int) -> np.ndarray:
        """
        Build 18-dim handcrafted obs for agent i.

        Market features (idx 0-5, 6..6+2K) are identical across agents —
        all agents share one LOB. Private features (base+0..+5) are
        per-agent. Mirrors LOBMarketMakingEnv._get_obs() (envs/lob_env.py)
        feature-for-feature; keep the two in sync if either changes.
        """
        K   = self.n_lob_levels
        obs = np.zeros(18, dtype=np.float32)

        # [0] Bid-ask spread / tick, clipped [0, 10]
        raw_spread = 0.0
        if self._lob_history:
            snap = self._lob_history[-1]
            bid_prices, ask_prices = snap["bid_prices"], snap["ask_prices"]
            if len(bid_prices) > 0 and len(ask_prices) > 0:
                raw_spread = (ask_prices[0] - bid_prices[0]) / self.tick_size
        obs[0] = float(np.clip(raw_spread / 10.0, 0.0, 10.0))

        # [1] Mid log-return this step / tick, clipped ±10 (shared mid, so
        # same prev/current pair for every agent)
        log_ret = 0.0
        if self._prev_mid > 0 and self._mid_price > 0:
            log_ret = np.log(self._mid_price / self._prev_mid) / self.tick_size
        obs[1] = float(np.clip(log_ret, -10.0, 10.0))

        # [2] Queue imbalance I = (V_b − V_a) / (V_b + V_a)
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

        # [4] Realized volatility (20-step window), scaled by tick_size
        obs[4] = float(np.clip(
            compute_realized_vol(self._price_history) / max(self.tick_size, 1e-8),
            0.0, 50.0,
        ))

        # [5] RSI normalised from [0, 100] → [−1, +1]
        obs[5] = float((compute_rsi(self._price_history) - 50.0) / 50.0)

        # [6:6+K] bid depth, [6+K:6+2K] ask depth, per-side normalised
        bid_depths = np.zeros(K, dtype=np.float32)
        ask_depths = np.zeros(K, dtype=np.float32)
        if self._lob_history:
            snap = self._lob_history[-1]
            bid_depths = np.asarray(snap.get("bid_sizes", [0] * K), dtype=np.float32)[:K]
            ask_depths = np.asarray(snap.get("ask_sizes", [0] * K), dtype=np.float32)[:K]

            if len(bid_depths) < K:
                bid_depths = np.pad(bid_depths, (0, K - len(bid_depths)))
            if len(ask_depths) < K:
                ask_depths = np.pad(ask_depths, (0, K - len(ask_depths)))

            bid_total  = bid_depths.sum() + 1e-8
            ask_total  = ask_depths.sum() + 1e-8
            bid_depths = bid_depths / bid_total
            ask_depths = ask_depths / ask_total

        obs[6 : 6 + K]         = bid_depths
        obs[6 + K : 6 + 2 * K] = ask_depths

        # ── Private features (per-agent) ────────────────────────────────
        base = 6 + 2 * K

        # [base+0] Inventory normalised to [−1, +1]
        obs[base + 0] = float(np.clip(
            self._inventories[agent_id] / self.Q_max, -1.0, 1.0
        ))

        max_offset = float(TICK_OFFSETS[-1])

        # [base+1] Active bid-offset from mid (ticks), normalised
        obs[base + 1] = float(np.clip(self._bid_dist[agent_id] / max_offset, 0.0, 1.0))

        # [base+2] Active ask-offset from mid (ticks), normalised
        obs[base + 2] = float(np.clip(self._ask_dist[agent_id] / max_offset, 0.0, 1.0))

        # [base+3] Outstanding bid price offset from mid, normalised
        if self._mid_price > 0 and self._outstanding_bid[agent_id] > 0:
            bid_offset_norm = (self._outstanding_bid[agent_id] - self._mid_price) / \
                              (max_offset * self.tick_size)
            obs[base + 3] = float(np.clip(bid_offset_norm, -1.0, 1.0))

        # [base+4] Outstanding ask price offset from mid, normalised
        if self._mid_price > 0 and self._outstanding_ask[agent_id] > 0:
            ask_offset_norm = (self._outstanding_ask[agent_id] - self._mid_price) / \
                              (max_offset * self.tick_size)
            obs[base + 4] = float(np.clip(ask_offset_norm, -1.0, 1.0))

        # [base+5] Time remaining τ = (T − t) / T ∈ [0, 1]
        obs[base + 5] = float(1.0 - self._step / self.episode_len)

        return obs

    # ------------------------------------------------------------------
    # GBM fallback (no ABIDES)
    # ------------------------------------------------------------------

    def _gbm_step(self) -> float:
        """Advance synthetic GBM price one step."""
        shock = self._rng.standard_normal() * self._gbm_sigma
        self._gbm_price *= np.exp(shock)
        return float(self._gbm_price)

    def _synthetic_fills(
        self,
        bid_prices: list[float],
        ask_prices: list[float],
    ) -> list[dict]:
        """
        Synthetic fill model for N competing agents.

        Fill probability decreases with distance from mid.
        When multiple agents quote at similar prices, fills are
        distributed proportionally (competitive fill sharing).
        """
        fills = [{"bid_qty": 0, "ask_qty": 0} for _ in range(self.n_agents)]

        # Base fill probability per side
        base_p = 0.3

        for i in range(self.n_agents):
            bid_dist = (self._mid_price - bid_prices[i]) / self.tick_size
            ask_dist = (ask_prices[i] - self._mid_price) / self.tick_size
            p_bid    = base_p * np.exp(-0.1 * max(bid_dist, 0))
            p_ask    = base_p * np.exp(-0.1 * max(ask_dist, 0))

            if self._rng.random() < p_bid:
                fills[i]["bid_qty"] = 1
            if self._rng.random() < p_ask:
                fills[i]["ask_qty"] = 1

        return fills

    # ------------------------------------------------------------------
    # Market metrics (for research questions)
    # ------------------------------------------------------------------

    def market_metrics(self, infos_n: list[dict]) -> dict:
        """
        Compute market-level metrics from a step's info dicts.

        Used to track: spread tightening, LOB stability.

        Parameters
        ----------
        infos_n : list of per-agent info dicts from step()

        Returns
        -------
        dict with keys: market_spread, price, n_fills, fill_rate
        """
        if not infos_n:
            return {}

        # Best bid = max of all agents' bids, best ask = min of all agents' asks
        best_bid = max(info["bid_price"] for info in infos_n)
        best_ask = min(info["ask_price"] for info in infos_n)
        spread   = best_ask - best_bid

        total_fills = sum(
            info["bid_filled"] + info["ask_filled"] for info in infos_n
        )

        return {
            "market_spread": float(spread),
            "best_bid":      float(best_bid),
            "best_ask":      float(best_ask),
            "price":         float(infos_n[0]["mid_price"]),
            "total_fills":   int(total_fills),
        }

    # ------------------------------------------------------------------

    def close(self) -> None:
        if self._abides is not None:
            self._abides.close()

    # ------------------------------------------------------------------
    # Mid price extraction
    # ------------------------------------------------------------------

    def _extract_mid(self, raw_state) -> float:
        """Extract mid price from ABIDES raw state."""
        try:
            mkt  = raw_state["parsed_mkt_data"][-1]
            bids = mkt["bids"]
            asks = mkt["asks"]
            last = mkt["last_transaction"]
            best_bid = bids[0][0] if bids else last
            best_ask = asks[0][0] if asks else last
            return 0.5 * (best_bid + best_ask) / 100.0
        except Exception:
            return self._mid_price


# ══════════════════════════════════════════════════════════════════════════════
# ABIDES multi-agent wrapper
# ══════════════════════════════════════════════════════════════════════════════

class _MultiAgentAbidesEnv(AbidesMarketMakingEnv):
    """
    Thin ABIDES subclass that accepts N agents' quotes per step.

    Stores N pending bid/ask price pairs and submits them all as
    separate LMT orders in one _map_action_space_to_ABIDES call.
    """

    def __init__(self, n_agents: int = 2, order_size: int = 100, **kwargs):
        super().__init__(**kwargs)
        self.n_agents           = n_agents
        self._pending_bids: list[int] = [0] * n_agents
        self._pending_asks: list[int] = [0] * n_agents
        self._pending_order_size      = order_size

    def step_multi(
        self,
        bid_prices: list[float],
        ask_prices: list[float],
    ):
        """Submit all agents' quotes and advance simulation one step."""
        self._pending_bids = [int(round(p * 100)) for p in bid_prices]
        self._pending_asks = [int(round(p * 100)) for p in ask_prices]
        _, _, done, _      = self.step(0)
        return self.gym_agent.raw_state[-1]

    def _map_action_space_to_ABIDES_SIMULATOR_SPACE(self, action: int):
        """Submit CCL_ALL + one LMT pair per agent."""
        orders = [{"type": "CCL_ALL"}]
        for i in range(self.n_agents):
            orders.append({
                "type": "LMT", "direction": "BUY",
                "size": self._pending_order_size, "limit_price": self._pending_bids[i],
            })
            orders.append({
                "type": "LMT", "direction": "SELL",
                "size": self._pending_order_size, "limit_price": self._pending_asks[i],
            })
        return orders

    def parse_fills(
        self,
        raw_state: dict,
        n_agents:  int,
        rng:       Optional[np.random.Generator] = None,
    ) -> list[dict]:
        """
        Parse fills and assign to agents by fill price proximity.

        Since ABIDES returns fills without agent tags, we attribute
        each fill to the agent whose quote price is closest to the
        fill price. On an exact tie (plausible and likely common: agents
        share one discrete tick-offset grid, and independent learners
        studying this exact "do quotes converge/undercut" dynamic
        frequently land on identical quotes), break randomly rather than
        always toward the lowest agent index — `np.argmin` alone would
        systematically favor agent 0 on every tie, contaminating the very
        convergence dynamic this environment exists to study. Pass `rng`
        (the caller's seeded generator) for reproducible tie-breaking.
        """
        internal = raw_state["internal_data"]
        fills    = internal.get("inter_wakeup_executed_orders", [])
        result   = [{"bid_qty": 0, "ask_qty": 0} for _ in range(n_agents)]
        rng      = rng if rng is not None else np.random.default_rng()

        for order in fills:
            fp    = order.fill_price / 100.0
            side  = order.side.value

            # Find closest agent quote (random tie-break, not lowest-index)
            if side == "BID":
                prices = self._pending_bids
                dists  = np.array([abs(fp - p / 100.0) for p in prices])
                agent_id = _argmin_random_tiebreak(dists, rng)
                result[agent_id]["bid_qty"] += order.quantity
            else:
                prices = self._pending_asks
                dists  = np.array([abs(fp - p / 100.0) for p in prices])
                agent_id = _argmin_random_tiebreak(dists, rng)
                result[agent_id]["ask_qty"] += order.quantity

        return result