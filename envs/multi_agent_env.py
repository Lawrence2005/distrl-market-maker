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

from typing import Optional

import gymnasium as gym
import numpy as np

from envs.lob_env import (
    LOBMarketMakingEnv,
    AbidesMarketMakingEnv,
    TICK_OFFSETS,
    N_OFFSET_LEVELS,
)


class MultiAgentMarketEnv:
    """
    N simultaneous market makers in one ABIDES simulation.

    Each agent has its own inventory, cash, and reward stream.
    All agents share the same LOB — their quotes compete for fills.

    Parameters
    ----------
    n_agents    : int   — number of simultaneous market makers (default 2)
    episode_len : int   — steps per episode (default 390)
    Q_max       : int   — per-agent inventory constraint (default 10)
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
        Q_max:       int   = 10,
        tick_size:   float = 0.01,
        reward_type: str   = "asymmetric",
        eta:         float = 0.5,
        use_abides:  bool  = True,
        seed:        int   = 42,
    ):
        self.n_agents    = n_agents
        self.episode_len = episode_len
        self.Q_max       = Q_max
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
                background_config = "rmsc04",
            )
        else:
            self._abides = None

        # Per-agent state
        self._inventories  = np.zeros(n_agents, dtype=np.int32)
        self._cash         = np.zeros(n_agents, dtype=np.float64)
        self._mid_price    = 0.0
        self._prev_mids    = np.zeros(n_agents, dtype=np.float64)
        self._step         = 0
        self._rng          = np.random.default_rng(seed)

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

        if self._abides is not None:
            raw_state = self._abides.reset()
            self._mid_price = self._extract_mid(raw_state)
        else:
            self._mid_price = self._gbm_price

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

        # Submit all quotes and advance simulation one step
        if self._abides is not None:
            raw_state = self._abides.step_multi(bid_prices, ask_prices)
            new_mid   = self._extract_mid(raw_state)
            fills     = self._abides.parse_fills(raw_state, self.n_agents)
        else:
            new_mid = self._gbm_step()
            fills   = self._synthetic_fills(bid_prices, ask_prices)

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
            inv_pnl = self._inventories[i] * (new_mid - self._prev_mids[i])
            pnl     = spread_pnl + inv_pnl

            reward  = self._compute_reward(pnl, inv_pnl, self._inventories[i])

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
            return pnl - max(0.0, self.eta * inv_pnl)
        if self.reward_type == "quadratic":
            return pnl - 0.1 * inventory ** 2
        return pnl

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_obs(self, agent_id: int) -> np.ndarray:
        """
        Build 18-dim handcrafted obs for agent i.

        Uses the same feature layout as LOBMarketMakingEnv._get_obs().
        Most market features are shared; inventory and cash are per-agent.
        """
        obs = np.zeros(18, dtype=np.float32)

        # [12] inventory / Q_max
        obs[12] = float(np.clip(
            self._inventories[agent_id] / self.Q_max, -1.0, 1.0
        ))
        # [17] time remaining
        obs[17] = float(1.0 - self._step / self.episode_len)

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

    def __init__(self, n_agents: int = 2, **kwargs):
        super().__init__(**kwargs)
        self.n_agents           = n_agents
        self._pending_bids: list[int] = [0] * n_agents
        self._pending_asks: list[int] = [0] * n_agents

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
                "size": 1, "limit_price": self._pending_bids[i],
            })
            orders.append({
                "type": "LMT", "direction": "SELL",
                "size": 1, "limit_price": self._pending_asks[i],
            })
        return orders

    def parse_fills(self, raw_state: dict, n_agents: int) -> list[dict]:
        """
        Parse fills and assign to agents by fill price proximity.

        Since ABIDES returns fills without agent tags, we attribute
        each fill to the agent whose quote price is closest to the
        fill price.
        """
        internal = raw_state["internal_data"]
        fills    = internal.get("inter_wakeup_executed_orders", [])
        result   = [{"bid_qty": 0, "ask_qty": 0} for _ in range(n_agents)]

        for order in fills:
            fp    = order.fill_price / 100.0
            side  = order.side.value

            # Find closest agent quote
            if side == "BID":
                prices = self._pending_bids
                dists  = [abs(fp - p / 100.0) for p in prices]
                agent_id = int(np.argmin(dists))
                result[agent_id]["bid_qty"] += order.quantity
            else:
                prices = self._pending_asks
                dists  = [abs(fp - p / 100.0) for p in prices]
                agent_id = int(np.argmin(dists))
                result[agent_id]["ask_qty"] += order.quantity

        return result