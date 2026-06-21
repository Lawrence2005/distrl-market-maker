"""
envs/background_agents.py

NOTE (post-Week-2 revision): This module is NOT integrated with lob_env.py.
Background agent population is provided by rmsc04 (ABIDES built-in config),
which supplies equivalent noise/momentum/informed traders wired to the
ABIDES exchange. This file documents the intended agent design and calibration
logic; it would be the starting point for a custom ABIDES config if rmsc04
is replaced in a future experiment.

Week 2 deliverable.
"""

import numpy as np
import json
from typing import Optional


def load_agent_params(params_path: str = "data/calibration/agent_params.json") -> dict:
    """
    Load calibrated background agent parameters from JSON.

    Keys used here:
        arrival_rate_per_sec  — mean events/sec (sets noise trader Poisson rate)
        mean_order_size       — mean shares per order
        std_order_size        — std of order sizes
        cancellation_rate     — fraction of events that are cancellations
        buy_sell_ratio        — fraction of events on the buy side
    """
    with open(params_path) as f:
        p = json.load(f)
    print(f"Loaded agent params: arrival_rate={p['arrival_rate_per_sec']:.3f}/sec, "f"mean_size={p['mean_order_size']:.1f} shares")
    return p


class NoiseAgent:
    """
    Random order placement agent — provides baseline liquidity.

    Places limit orders randomly around the current mid-price.
    Arrival times follow a Poisson process calibrated to LOBSTER data.
    Order sizes drawn from a log-normal distribution fitted to data.

    In ABIDES terms this is the market's background noise — it creates
    the basic order flow that gives the RL agent something to trade against.

    Parameters
    ----------
    arrival_rate : float — mean orders per second (from agent_params.json)
    mean_size    : float — mean order size in shares
    std_size     : float — std of order sizes
    cancel_prob  : float — probability of cancelling an existing order
    seed         : int
    """

    def __init__(
        self,
        arrival_rate: float,
        mean_size:    float,
        std_size:     float,
        cancel_prob:  float = 0.2,
        seed:         int   = 42,
    ):
        self.arrival_rate = arrival_rate
        self.mean_size    = mean_size
        self.std_size     = std_size
        self.cancel_prob  = cancel_prob
        self.rng          = np.random.default_rng(seed)

    def next_arrival_time(self, current_time: float) -> float:
        """
        Sample next order arrival time from Poisson process.
        Returns absolute time of next event.
        """
        dt = self.rng.exponential(1.0 / self.arrival_rate)
        return current_time + dt

    def sample_order(self, mid_price: float) -> dict:
        """
        Sample a random limit order around the current mid-price.

        Returns dict with keys:
            side     : 'bid' or 'ask'
            price    : limit price in dollars
            size     : number of shares
            type     : 'limit' or 'cancel'
        """
        # Cancel an existing order with probability cancel_prob
        order_type = 'cancel' if self.rng.uniform() < self.cancel_prob else 'limit'

        side = 'bid' if self.rng.uniform() < 0.5 else 'ask'

        # Parameterise log-normal so that mean and std match calibrated values
        # If X ~ LogNormal(mu_ln, sigma_ln), then:
        #   E[X] = exp(mu_ln + sigma_ln²/2) = mean_size
        #   Var[X] = (exp(sigma_ln²) - 1) * exp(2*mu_ln + sigma_ln²)
        sigma_ln = np.sqrt(np.log(1 + (self.std_size / self.mean_size) ** 2))
        mu_ln    = np.log(self.mean_size) - 0.5 * sigma_ln ** 2
        size     = max(1, int(self.rng.lognormal(mu_ln, sigma_ln)))

        tick_size  = 0.01
        n_ticks    = int(self.rng.integers(0, 5))   # 0 to 4 ticks
        offset     = n_ticks * tick_size

        price = mid_price - offset if side == 'bid' else mid_price + offset
        price = round(price, 2)

        return {
            "side":  side,
            "price": price,
            "size":  size,
            "type":  order_type,
        }

    @classmethod
    def from_params(cls, params: dict, seed: int = 42) -> "NoiseAgent":
        """Construct from agent_params.json dict."""
        return cls(
            arrival_rate=params["arrival_rate_per_sec"],
            mean_size=params["mean_order_size"],
            std_size=params["std_order_size"],
            cancel_prob=params["cancellation_rate"],
            seed=seed,
        )


class MomentumAgent:
    """
    Trend-following agent — buys when price rising, sells when falling.

    Tracks a short-term moving average of mid-price changes and places
    market orders in the direction of the trend. Creates the autocorrelated
    order flow that makes markets trend in the short term.

    Parameters
    ----------
    arrival_rate  : float — mean orders per second
    mean_size     : float — mean order size
    lookback      : int   — number of steps to compute momentum signal
    threshold     : float — minimum price move to trigger an order
    seed          : int
    """

    def __init__(
        self,
        arrival_rate: float,
        mean_size:    float,
        lookback:     int   = 10,
        threshold:    float = 0.01,
        seed:         int   = 42,
    ):
        self.arrival_rate = arrival_rate
        self.mean_size    = mean_size
        self.lookback     = lookback
        self.threshold    = threshold
        self.rng          = np.random.default_rng(seed)
        self.price_history: list = []

    def update_price(self, mid_price: float) -> None:
        """Record latest mid-price. Call at each timestep."""
        self.price_history.append(mid_price)
        if len(self.price_history) > self.lookback:
            self.price_history.pop(0)

    def momentum_signal(self) -> float:
        """
        Compute momentum signal from price history.
        Returns positive value for uptrend, negative for downtrend, 0 if flat.
        """
        if len(self.price_history) < 2:
            return 0.0
        return self.price_history[-1] - self.price_history[0]

    def sample_order(self, mid_price: float) -> Optional[dict]:
        """
        Sample a momentum-driven order if signal exceeds threshold.
        Returns None if no order to place this step.
        """
        signal = self.momentum_signal()
        if abs(signal) < self.threshold:
            return None

        side = "bid" if signal > 0 else "ask"

        # Size scales with signal strength — stronger trend → larger order
        # Clamp between 1 share and 5× mean size
        signal_scale = min(abs(signal) / self.threshold, 5.0)
        size = max(1, int(self.mean_size * signal_scale))

        return {
            "type": "market",
            "side": side,
            "size": size
        }

    @classmethod
    def from_params(cls, params: dict, seed: int = 42) -> "MomentumAgent":
        """Construct from agent_params.json dict."""
        return cls(
            arrival_rate=params["arrival_rate_per_sec"] * 0.2,  # 20% of noise rate
            mean_size=params["mean_order_size"],
            seed=seed,
        )


class InformedAgent:
    """
    Adverse selection source — has private signal about future price.

    Places limit orders in the direction of a private price signal,
    causing adverse selection for the market maker. The RL agent's
    CVaR objective should learn to protect against this agent.

    The private signal is a noisy version of the future mid-price
    (simulated by looking ahead in the price path, or driven by
    a separate signal process in ABIDES).

    Parameters
    ----------
    arrival_rate    : float — mean orders per second (lower than noise agent)
    mean_size       : float — mean order size (typically larger than noise)
    signal_horizon  : int   — steps ahead the signal looks
    signal_noise    : float — noise added to private signal (0 = perfect info)
    informed_frac   : float — fraction of time agent has signal (vs. random)
    seed            : int
    """

    def __init__(
        self,
        arrival_rate:   float,
        mean_size:      float,
        signal_horizon: int   = 5,
        signal_noise:   float = 0.005,
        informed_frac:  float = 1.0,
        seed:           int   = 42,
    ):
        self.arrival_rate   = arrival_rate
        self.mean_size      = mean_size
        self.signal_horizon = signal_horizon
        self.signal_noise   = signal_noise
        self.informed_frac  = informed_frac
        self.rng            = np.random.default_rng(seed)

    def get_signal(self, current_price: float, future_price: float) -> float:
        """
        Generate private signal about future price direction.

        In real markets, informed traders have private information.
        Here we simulate this with noisy look-ahead.

        Returns signed signal: positive = expect price rise, negative = fall.
        """
        true_direction = future_price - current_price
        noise = self.rng.normal(0, self.signal_noise)
        return true_direction + noise

    def sample_order(
        self,
        mid_price:    float,
        signal:       float,
    ) -> Optional[dict]:
        """
        Place an order in direction of private signal.
        Returns None if agent is dormant this step.
        """
        if self.rng.uniform() > self.informed_frac:
            return None

        if abs(signal) < self.signal_noise:
            return None

        min_signal = 0.001   # $0.001 minimum price move to act on
        if abs(signal) < min_signal:
            return None

        side = "bid" if signal > 0 else "ask"

        # Size scales mildly with signal confidence
        signal_confidence = min(abs(signal) / min_signal, 3.0)
        size = max(1, int(self.mean_size * signal_confidence))

        return {
            "type": "market",
            "side": side,
            "size": size,
        }

    @classmethod
    def from_params(cls, params: dict, seed: int = 42) -> "InformedAgent":
        """Construct from agent_params.json dict."""
        return cls(
            arrival_rate=params["arrival_rate_per_sec"] * 0.05,  # 5% of noise rate
            mean_size=params["mean_order_size"] * 2.0,            # larger orders
            seed=seed,
        )


class BackgroundAgentPopulation:
    """
    Container for all background agents.

    Manages the full population of noise, momentum, and informed traders.
    Called by lob_env.py at each timestep to get background order flow.

    Parameters
    ----------
    n_noise    : int — number of noise traders (default 50)
    n_momentum : int — number of momentum traders (default 10)
    n_informed : int — number of informed traders (default 5)
    params     : dict — from agent_params.json
    seed       : int
    """

    def __init__(
        self,
        params:     dict,
        n_noise:    int = 50,
        n_momentum: int = 10,
        n_informed: int = 5,
        seed:       int = 42,
    ):
        self.noise_agents = [
            NoiseAgent.from_params(params, seed=seed + i)
            for i in range(n_noise)
        ]
        self.momentum_agents = [
            MomentumAgent.from_params(params, seed=seed + 1000 + i)
            for i in range(n_momentum)
        ]
        self.informed_agents = [
            InformedAgent.from_params(params, seed=seed + 2000 + i)
            for i in range(n_informed)
        ]

    def step(
        self,
        mid_price:    float,
        future_price: Optional[float] = None,
    ) -> list:
        """
        Generate all background orders for this timestep.

        Returns list of order dicts, each with keys:
            side, price, size, type, agent_type
        """
        orders = []
        for agent in self.noise_agents:
            order = agent.sample_order(mid_price)
            if order: orders.append({**order, 'agent_type': 'noise'})

        for agent in self.momentum_agents:
            agent.update_price(mid_price)
            order = agent.sample_order(mid_price)
            if order: orders.append({**order, 'agent_type': 'momentum'})
        
        if future_price is not None:
          for agent in self.informed_agents:
              signal = agent.get_signal(mid_price, future_price)
              order = agent.sample_order(mid_price, signal)
              if order: orders.append({**order, 'agent_type': 'informed'})
              
        return orders

    @classmethod
    def from_json(
        cls,
        params_path: str = "data/calibration/agent_params.json",
        **kwargs,
    ) -> "BackgroundAgentPopulation":
        """Load from calibration JSON file."""
        params = load_agent_params(params_path)
        return cls(params=params, **kwargs)