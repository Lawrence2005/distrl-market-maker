"""
baselines/avellaneda_stoikov.py

Avellaneda-Stoikov (2008) closed-form market-making baseline.

Derives optimal bid/ask quotes from the solution to a stochastic control
problem under exponential utility. The key insight is that the optimal
policy consists of:

  1. A reservation price that skews quotes away from inventory:
         r(s, q, t) = s - q * gamma * sigma^2 * tau_hat

  2. An optimal spread that widens with volatility and time remaining:
         delta*(t) = gamma * sigma^2 * tau_hat + (2/gamma) * ln(1 + gamma/kappa)

  3. Symmetric quotes around the reservation price:
         bid* = r - delta*/2
         ask* = r + delta*/2

where tau_hat = (T - t) / T in [0, 1] is the normalised time-to-go.

The inventory skew in step 1 is the core mechanism: when long (q > 0),
the reservation price falls below mid, making the ask more competitive
and the bid less competitive, encouraging inventory reduction.

Units note
----------
sigma is kept in log-return units per step (dimensionless), matching the
theoretical derivation in AS (2008) where sigma is the volatility of the
log mid-price process. The spread delta* is therefore also in log-return
units. At mid ~$1000 and sigma ~0.0003 per step, the inventory risk term
gamma * sigma^2 * tau_hat ~= 0.1 * 9e-8 * 1 ~= 9e-9 -- negligible
relative to the base spread (2/gamma)*ln(1+gamma/kappa) which dominates.

Time normalisation
------------------
tau_hat = (T - t) / T is used instead of raw steps so that
gamma * sigma^2 * tau_hat stays bounded in [0, gamma * sigma^2] regardless
of episode length. Without normalisation a 390-step episode produces an
inventory risk term 390x larger than intended for a unit horizon.

Parameters
----------
gamma : float
    Risk-aversion coefficient. Higher gamma -> wider spread, stronger
    inventory skew. Typical range [0.01, 0.5].
kappa : float
    Order-arrival intensity parameter. Higher kappa -> tighter spread.
    Calibrate so that (2/gamma)*ln(1 + gamma/kappa) fits within half the
    action space. With gamma=0.1 and tick_size=0.01, kappa=100 gives
    ~2 tick base spread.
sigma : float
    Initial log-return volatility estimate per step (dimensionless).
    Updated each step via EMA of realised log-return std when
    adapt_sigma=True. Typical ABIDES value: 0.0003-0.001.
T : int
    Episode length in steps. Used only for tau_hat normalisation.

Reference
---------
Avellaneda, M. & Stoikov, S. (2008).
"High-frequency trading in a limit order book."
Quantitative Finance, 8(3), 217-224.

Week 3 deliverable.
"""

import numpy as np
from typing import Dict, Any
from baselines.glft import _MAX_OFFSET
from envs.lob_env import TICK_OFFSETS, N_OFFSET_LEVELS


def _price_to_action_idx(
    quote_price: float,
    mid_price:   float,
    tick_size:   float,
) -> int:
    """
    Convert an absolute quote price to a TICK_OFFSETS index.

    With TICK_OFFSETS = np.arange(0, N), index == tick count from mid directly.

    Parameters
    ----------
    quote_price : float -- desired absolute price
    mid_price   : float -- current mid price
    tick_size   : float -- dollar value of one tick

    Returns
    -------
    int -- index into TICK_OFFSETS, clamped to [0, N_OFFSET_LEVELS-1]
    """
    offset_dollars = abs(quote_price - mid_price)
    offset_ticks   = int(round(offset_dollars / tick_size))
    return int(np.clip(offset_ticks, 0, N_OFFSET_LEVELS - 1))


class AvellanedaStoikovBaseline:
    """
    Closed-form Avellaneda-Stoikov (2008) market maker.

    At each step:
        1. Compute normalised time-to-go: tau_hat = (T - t) / T in [0, 1]
        2. Compute reservation price:     r = s - q*gamma*sigma^2*tau_hat
        3. Compute optimal full spread:   delta* = gamma*sigma^2*tau_hat
                                                 + (2/gamma)*ln(1+gamma/kappa)
        4. Quote bid = r - delta*/2, ask = r + delta*/2
        5. Convert dollar offsets to tick-offset action indices

    The formula follows AS (2008) Proposition 3.1, derived under symmetric
    exponential fill rates lambda(delta) = A*exp(-kappa*delta). The
    branching parameter A cancels in the spread; only kappa (the fill-rate
    decay rate) determines quote width.

    Parameters
    ----------
    gamma       : float -- risk-aversion coefficient (default 0.1)
    kappa       : float -- fill-rate intensity (default 100.0)
    sigma       : float -- initial log-return vol per step (default 0.01)
    T           : int   -- episode length in steps (default 390)
    tick_size   : float -- dollar value of one tick (default 0.01)
    adapt_sigma : bool  -- update sigma from live price history via EMA
    """

    name = "AvellanedaStoikov"

    def __init__(
        self,
        gamma:       float = 0.1,
        kappa:       float = 100.0,
        sigma:       float = 0.01,
        T:           int   = 390,
        tick_size:   float = 0.01,
        adapt_sigma: bool  = True,
    ):
        assert gamma > 0,     f"gamma must be positive, got {gamma}"
        assert kappa > 0,     f"kappa must be positive, got {kappa}"
        assert sigma > 0,     f"sigma must be positive, got {sigma}"
        assert T > 0,         f"T must be positive, got {T}"
        assert tick_size > 0, f"tick_size must be positive, got {tick_size}"

        self.gamma       = gamma
        self.kappa       = kappa
        self.sigma       = sigma
        self.sigma_init  = sigma
        self.T           = T
        self.tick_size   = tick_size
        self.adapt_sigma = adapt_sigma

        self._price_history: list = []
        self._t: int = 0

    # ------------------------------------------------------------------
    # Core AS equations  (AS 2008, Proposition 3.1)
    # ------------------------------------------------------------------

    def reservation_price(
        self,
        mid:       float,
        inventory: float,
        tau_hat:   float,
    ) -> float:
        """
        AS reservation price (AS 2008, eq. 3.4-3.5 combined).

        r(s, q, tau_hat) = s - q * gamma * sigma^2 * tau_hat

        The skew term q*gamma*sigma^2*tau_hat shifts both quotes toward
        inventory reduction: when long (q > 0), r < s so the ask is
        closer to mid (more competitive) and the bid is further away.

        Parameters
        ----------
        mid       : float -- current mid-price s
        inventory : float -- current inventory q (signed)
        tau_hat   : float -- normalised time-to-go in [0, 1]

        Returns
        -------
        float -- reservation price r
        """
        return mid - inventory * self.gamma * (self.sigma ** 2) * tau_hat

    def optimal_spread(self, tau_hat: float) -> float:
        """
        AS optimal full spread (AS 2008, Proposition 3.1).

        delta*(tau_hat) = gamma * sigma^2 * tau_hat
                        + (2/gamma) * ln(1 + gamma/kappa)

        Two additive components:
          - gamma*sigma^2*tau_hat   : inventory risk term, widens with
                                      time remaining; negligible in practice
                                      with log-return sigma and tau_hat in [0,1]
          - (2/gamma)*ln(1+gamma/kappa) : base spread from fill-rate economics,
                                          time-independent, dominates quote width

        With gamma=0.1, kappa=100, tick_size=0.01:
          base = 20*ln(1.001) ~= 0.02 dollars = 2 ticks.

        Parameters
        ----------
        tau_hat : float -- normalised time-to-go in [0, 1]

        Returns
        -------
        float -- optimal full spread delta* in dollars
        """
        inventory_risk = self.gamma * (self.sigma ** 2) * tau_hat
        base_spread    = (2.0 / self.gamma) * np.log(1.0 + self.gamma / self.kappa)
        return inventory_risk + base_spread

    def compute_quotes(
        self,
        mid:       float,
        inventory: float,
        tau_hat:   float,
    ) -> tuple[float, float]:
        """
        Compute optimal bid and ask prices.

            bid* = r - delta*/2
            ask* = r + delta*/2

        Parameters
        ----------
        mid       : float -- current mid-price
        inventory : float -- current inventory (signed)
        tau_hat   : float -- normalised time-to-go in [0, 1]

        Returns
        -------
        (bid_price, ask_price) : tuple[float, float]
        """
        r      = self.reservation_price(mid, inventory, tau_hat)
        spread = self.optimal_spread(tau_hat)
        return r - spread / 2.0, r + spread / 2.0

    # ------------------------------------------------------------------
    # Sigma adaptation
    # ------------------------------------------------------------------

    def update_sigma(self, mid_price: float, window: int = 20) -> None:
        """
        Update sigma estimate from recent price history via EMA.

        Computes realised log-return std over the last `window` steps,
        then blends it into the current sigma estimate with EMA weight 0.1.
        sigma stays in log-return units (dimensionless) throughout --
        do NOT multiply by mid_price.

        Parameters
        ----------
        mid_price : float -- current mid-price to append to history
        window    : int   -- rolling window length in steps (default 20)
        """
        self._price_history.append(mid_price)
        if len(self._price_history) < 3:
            return

        prices  = np.array(
            self._price_history[-min(window + 1, len(self._price_history)):]
        )
        log_ret = np.diff(np.log(np.maximum(prices, 1e-10)))
        vol     = float(np.std(log_ret))

        if vol > 1e-10:
            alpha      = 0.1
            self.sigma = (1 - alpha) * self.sigma + alpha * vol

    # ------------------------------------------------------------------
    # Gymnasium-compatible interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset episode state. Call at the start of each episode."""
        self._price_history.clear()
        self._t    = 0
        self.sigma = self.sigma_init

    def act(
        self,
        obs:  np.ndarray,
        info: Dict[str, Any],
    ) -> np.ndarray:
        """
        Compute AS optimal action from current market state.

        Parameters
        ----------
        obs  : np.ndarray -- current observation (unused; AS only needs s, q, tau)
        info : dict       -- must contain 'mid_price' and 'inventory'

        Returns
        -------
        np.ndarray shape (2,) -- [bid_idx, ask_idx] into TICK_OFFSETS
        """
        mid       = float(info["mid_price"])
        inventory = float(info["inventory"])
        tau_hat   = max(float(self.T - self._t), 0.0) / self.T

        if self.adapt_sigma:
            self.update_sigma(mid)

        bid_price, ask_price = self.compute_quotes(mid, inventory, tau_hat)

        bid_idx = _price_to_action_idx(bid_price, mid, self.tick_size)
        ask_idx = _price_to_action_idx(ask_price, mid, self.tick_size)

        self._t += 1

        return np.array([bid_idx, ask_idx], dtype=np.int64)

    def __repr__(self) -> str:
        return (
            f"AvellanedaStoikovBaseline("
            f"gamma={self.gamma}, kappa={self.kappa}, "
            f"sigma={self.sigma_init}, T={self.T})"
        )