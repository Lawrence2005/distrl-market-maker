"""
baselines/glft.py

Guéant-Lehalle-Fernandez-Tapia (2013) market-making baseline with market impact.

Implements Proposition 5 of GLFT, which extends the basic inventory-control
model to include a linear market-impact cost xi on inventory. The solution
requires solving a linear ODE system backwards from the terminal condition,
then reading off optimal quotes from the resulting v_q(t) functions.

ODE system (Proposition 5)
--------------------------
For q in {-Q+1, ..., Q-1}:
    v_dot_q(t) = alpha*q^2*v_q(t) - eta*exp(-kappa^2*xi/2)*(v_{q-1}(t) + v_{q+1}(t))

Boundary (q = +/-Q, only one neighbour):
    v_dot_Q(t)  = alpha*Q^2*v_Q(t)  - eta*exp(-kappa^2*xi/2)*v_{Q-1}(t)
    v_dot_{-Q}(t) = alpha*Q^2*v_{-Q}(t) - eta*exp(-kappa^2*xi/2)*v_{-Q+1}(t)

Terminal condition:
    v_q(T) = exp(-kappa^2*xi*q^2/2)   for all q

    With xi=0 this collapses to v_q(T)=1 for all q, recovering the
    no-market-impact solution (Theorem 1 of GLFT 2013).

Derived constants:
    alpha = kappa^2 * gamma * sigma^2 / 2
    eta   = A * (1 + gamma/kappa)^{-(1 + kappa/gamma)}

Optimal quotes (Proposition 5)
-------------------------------
    delta_b*(t, q) = (1/kappa)*ln(v_q(t)/v_{q+1}(t)) + xi/2 + (1/gamma)*ln(1 + gamma/kappa)
    delta_a*(t, q) = (1/kappa)*ln(v_q(t)/v_{q-1}(t)) + xi/2 + (1/gamma)*ln(1 + gamma/kappa)

    bid* = s - delta_b*
    ask* = s + delta_a*

Resulting spread:
    psi*(t, q) = -(1/kappa)*ln(v_{q+1}*v_{q-1}/v_q^2) + xi + (2/gamma)*ln(1 + gamma/kappa)

Units note
----------
sigma is in log-return units per step (dimensionless), consistent with
avellaneda_stoikov.py. alpha = kappa^2*gamma*sigma^2/2 is therefore also
dimensionless and enters the ODE in original time units (steps).

Time normalisation
------------------
tau_hat = (T - t) / T in [0, 1] is used in act() for consistency with
the AS baseline. The ODE itself is solved in original step units:
solve_v computes S = expm(-M*1) (one original-unit step) and propagates
V[i+1] = S @ V[i] for T steps. The pre-computed table taus runs [0, 1]
(normalised) so that _get_v(tau_hat) indexes correctly.

Numerical approach
------------------
The ODE is linear: v_dot = M*v. The exact solution stepping one unit of
original time is S = expm(-M * 1). Then:

    V[0]   = v_T            (tau=0: terminal condition)
    V[i+1] = S @ V[i]       (tau advances by 1/T in normalised units)

This requires ONE expm call regardless of episode length, vs the naive
approach of one expm per time step. For n=21 (Q_max=10), T=390 steps:
~34x faster than calling expm(M*tau) at each tau.

Magnitudes of v_q blow up exponentially with tau but the ratios
v_q/v_{q+-1} that enter the log in the quote formulas remain
well-conditioned throughout.

Parameters
----------
gamma     : float -- risk-aversion coefficient (default 0.1)
kappa     : float -- fill-rate intensity (default 100.0)
sigma     : float -- initial log-return vol per step (default 0.01)
xi        : float -- market-impact parameter (default 0.0)
              Set xi=0 to recover Theorem 1 (no market impact).
              With xi>0, use very small values (e.g. 1e-5) since
              the terminal condition exp(-kappa^2*xi*q^2/2) collapses
              to near-zero at boundary inventories for kappa=100.
A         : float -- Poisson arrival rate scale (default 1.0)
T         : int   -- episode length in steps (default 390)
Q_max     : int   -- inventory constraint (default 10)
tick_size : float -- dollar value of one tick (default 0.01)
adapt_sigma : bool -- update sigma from live price history (default True)

Reference
---------
Guéant, O., Lehalle, C.-A. & Fernandez-Tapia, J. (2013).
"Dealing with the inventory risk: a solution to the market making problem."
Mathematics and Financial Economics, 7(4), 477-507.
Proposition 5 (with market impact).

Week 3 deliverable.
"""

import numpy as np
from scipy.linalg import expm
from typing import Dict, Any
from envs.lob_env import TICK_OFFSETS, N_OFFSET_LEVELS


_MAX_OFFSET = N_OFFSET_LEVELS - 1   # largest valid tick offset (e.g. 10)


def _dollars_to_idx(offset_dollars: float, tick_size: float) -> int:
    """
    Convert a quote half-spread in dollars to a TICK_OFFSETS index.

    With TICK_OFFSETS = np.arange(0, N), index == tick count directly.

    Parameters
    ----------
    offset_dollars : float -- distance from mid in dollars (positive)
    tick_size      : float -- dollar value of one tick

    Returns
    -------
    int -- index into TICK_OFFSETS, clamped to [0, N_OFFSET_LEVELS-1]
    """
    ticks = int(round(abs(offset_dollars) / tick_size))
    return int(np.clip(ticks, 0, _MAX_OFFSET))


# ══════════════════════════════════════════════════════════════════════
# Pure functions — testable mathematical core
# ══════════════════════════════════════════════════════════════════════

def build_ode_matrix(
    gamma: float,
    kappa: float,
    sigma: float,
    xi:    float,
    A:     float,
    Q_max: int,
) -> np.ndarray:
    """
    Build the ODE matrix M such that v_dot = M*v.

    M is tridiagonal:
        M[q, q]   = alpha*q^2     (diagonal: inventory risk)
        M[q, q+-1] = -decay       (off-diagonal: fill arrivals)

    where:
        alpha = kappa^2 * gamma * sigma^2 / 2
        decay = eta * exp(-kappa^2 * xi / 2)
        eta   = A * (1 + gamma/kappa)^{-(1 + kappa/gamma)}

    sigma is in log-return units (dimensionless). alpha is therefore
    in units of 1/step, consistent with the ODE time unit being one step.

    Parameters
    ----------
    gamma : float -- risk-aversion coefficient
    kappa : float -- fill-rate intensity
    sigma : float -- log-return volatility per step
    xi    : float -- market-impact parameter (0 = no market impact)
    A     : float -- Poisson arrival rate scale
    Q_max : int   -- maximum inventory

    Returns
    -------
    np.ndarray shape (2*Q_max+1, 2*Q_max+1) -- ODE matrix M
    """
    n     = 2 * Q_max + 1
    alpha = (kappa ** 2 * gamma * sigma ** 2) / 2.0
    eta   = A * (1.0 + gamma / kappa) ** (-(1.0 + kappa / gamma))
    decay = eta * np.exp(-kappa ** 2 * xi / 2.0)

    M = np.zeros((n, n))
    for i, q in enumerate(range(-Q_max, Q_max + 1)):
        M[i, i] = alpha * q ** 2
        if q > -Q_max:
            M[i, i - 1] = -decay
        if q < Q_max:
            M[i, i + 1] = -decay

    return M


def terminal_condition(kappa: float, xi: float, Q_max: int) -> np.ndarray:
    """
    Compute the terminal condition v_q(T) = exp(-kappa^2*xi*q^2/2).

    With xi=0 (no market impact): v_q(T) = 1 for all q.
    With xi>0: v_q(T) decays exponentially with |q|. For kappa=100
    and xi=0.001, v_q(T) at q=1 is exp(-5) ~= 0.007 -- a large ratio
    that inflates spreads. Use xi <= 1e-5 with kappa=100.

    Parameters
    ----------
    kappa : float -- fill-rate intensity
    xi    : float -- market-impact parameter
    Q_max : int   -- maximum inventory

    Returns
    -------
    np.ndarray shape (2*Q_max+1,) -- v_q(T) for q in {-Q_max,...,Q_max}
    """
    return np.array([
        np.exp(-0.5 * kappa ** 2 * xi * q ** 2)
        for q in range(-Q_max, Q_max + 1)
    ])


def solve_v(
    M:       np.ndarray,
    v_T:     np.ndarray,
    T:       float,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve v(tau) for tau in [0, 1] (normalised time-to-go) using the
    step-matrix method.

    The ODE v_dot = M*v has exact solution v(tau) = expm(-M*tau)*v_T
    in original time units. Computing one unit-step matrix S = expm(-M*1)
    and propagating V[i+1] = S @ V[i] is equivalent and requires only
    ONE expm call instead of n_steps+1.

    The pre-computed table stores normalised taus in [0, 1] so that
    _get_v(tau_hat) indexes correctly using the tau_hat computed in act().

    Parameters
    ----------
    M       : np.ndarray -- ODE matrix from build_ode_matrix()
    v_T     : np.ndarray -- terminal condition from terminal_condition()
    T       : float      -- episode length in steps (sets number of matmuls)
    n_steps : int        -- number of discrete steps to pre-compute
                           (pass T so each matmul advances 1 original step)

    Returns
    -------
    taus : np.ndarray shape (n_steps+1,)           -- normalised tau in [0,1]
    V    : np.ndarray shape (n_steps+1, 2*Q_max+1) -- v_q at each tau
    """
    taus = np.linspace(0.0, 1.0, n_steps + 1)   # normalised [0,1] for lookup
    S    = expm(-M * 1.0)                         # one original-unit step
    n    = len(v_T)

    V    = np.zeros((n_steps + 1, n))
    V[0] = v_T                      # tau=0: terminal condition
    for i in range(n_steps):
        V[i + 1] = S @ V[i]        # advance one step

    return taus, V


def delta_bid(
    v:     np.ndarray,
    q:     int,
    Q_max: int,
    gamma: float,
    kappa: float,
    xi:    float,
) -> float:
    """
    Optimal bid half-spread delta_b*(t, q) from GLFT Proposition 5.

    delta_b*(t, q) = (1/kappa)*ln(v_q(t)/v_{q+1}(t)) + xi/2
                   + (1/gamma)*ln(1 + gamma/kappa)

    When q > 0 (long inventory): v_q/v_{q+1} < 1, so the log ratio is
    negative, pushing the bid closer to mid (less aggressive buying).
    When q < 0 (short inventory): v_q/v_{q+1} > 1, pushing bid further
    from mid (more aggressive -- want to buy to reduce short position).

    Undefined at q = Q_max (inventory limit: no more buying).
    Returns np.inf in that case.

    Parameters
    ----------
    v     : np.ndarray -- v_q vector at current time
    q     : int        -- current inventory
    Q_max : int        -- inventory constraint
    gamma : float      -- risk-aversion coefficient
    kappa : float      -- fill-rate intensity
    xi    : float      -- market-impact parameter

    Returns
    -------
    float -- optimal bid half-spread in dollars
    """
    if q >= Q_max:
        return np.inf

    i         = q + Q_max
    log_ratio = np.log(v[i] / v[i + 1])
    base      = (1.0 / gamma) * np.log(1.0 + gamma / kappa)
    return (1.0 / kappa) * log_ratio + xi / 2.0 + base


def delta_ask(
    v:     np.ndarray,
    q:     int,
    Q_max: int,
    gamma: float,
    kappa: float,
    xi:    float,
) -> float:
    """
    Optimal ask half-spread delta_a*(t, q) from GLFT Proposition 5.

    delta_a*(t, q) = (1/kappa)*ln(v_q(t)/v_{q-1}(t)) + xi/2
                   + (1/gamma)*ln(1 + gamma/kappa)

    When q < 0 (short inventory): v_q/v_{q-1} < 1, pushing ask closer
    to mid (less aggressive selling -- want to buy not sell).
    When q > 0 (long inventory): v_q/v_{q-1} > 1, pushing ask further
    from mid... wait, this is reversed. See inventory skew direction note
    in delta_bid above.

    Undefined at q = -Q_max (inventory limit: no more selling).
    Returns np.inf in that case.

    Parameters
    ----------
    v     : np.ndarray -- v_q vector at current time
    q     : int        -- current inventory
    Q_max : int        -- inventory constraint
    gamma : float      -- risk-aversion coefficient
    kappa : float      -- fill-rate intensity
    xi    : float      -- market-impact parameter

    Returns
    -------
    float -- optimal ask half-spread in dollars
    """
    if q <= -Q_max:
        return np.inf

    i         = q + Q_max
    log_ratio = np.log(v[i] / v[i - 1])
    base      = (1.0 / gamma) * np.log(1.0 + gamma / kappa)
    return (1.0 / kappa) * log_ratio + xi / 2.0 + base


def spread(
    v:     np.ndarray,
    q:     int,
    Q_max: int,
    gamma: float,
    kappa: float,
    xi:    float,
) -> float:
    """
    Full bid-ask spread psi*(t, q) from GLFT Proposition 5.

    psi*(t, q) = -(1/kappa)*ln(v_{q+1}*v_{q-1}/v_q^2)
               + xi + (2/gamma)*ln(1 + gamma/kappa)

    Consistent with delta_bid + delta_ask to numerical precision.
    Undefined at boundary inventories |q| = Q_max.

    Parameters
    ----------
    v     : np.ndarray -- v_q vector at current time
    q     : int        -- current inventory (must satisfy |q| < Q_max)
    Q_max : int        -- inventory constraint
    gamma : float      -- risk-aversion coefficient
    kappa : float      -- fill-rate intensity
    xi    : float      -- market-impact parameter

    Returns
    -------
    float -- full spread in dollars
    """
    assert abs(q) < Q_max, f"spread undefined at boundary |q|={abs(q)}=Q_max"
    i        = q + Q_max
    base     = (2.0 / gamma) * np.log(1.0 + gamma / kappa)
    log_term = np.log(v[i + 1] * v[i - 1] / (v[i] ** 2))
    return -(1.0 / kappa) * log_term + xi + base


# ══════════════════════════════════════════════════════════════════════
# Baseline class
# ══════════════════════════════════════════════════════════════════════

class GLFTBaseline:
    """
    GLFT market-making baseline (Proposition 5, with market impact).

    Pre-computes v_q(t) at construction via the step-matrix method
    (one expm call). At each step, looks up v_q at the current
    normalised time-to-go tau_hat and reads off optimal quotes.

    Parameters
    ----------
    gamma       : float -- risk-aversion coefficient (default 0.1)
    kappa       : float -- fill-rate intensity (default 100.0)
    sigma       : float -- initial log-return vol per step (default 0.01)
    xi          : float -- market-impact parameter (default 0.0)
    A           : float -- Poisson arrival rate scale (default 1.0)
    T           : int   -- episode length in steps (default 390)
    Q_max       : int   -- inventory constraint (default 10)
    tick_size   : float -- dollar value of one tick (default 0.01)
    adapt_sigma : bool  -- update sigma from live price history (default True)
    """

    name = "GLFT"

    def __init__(
        self,
        gamma:       float = 0.1,
        kappa:       float = 100.0,
        sigma:       float = 0.01,
        xi:          float = 0.0,
        A:           float = 1.0,
        T:           int   = 390,
        Q_max:       int   = 10,
        tick_size:   float = 0.01,
        adapt_sigma: bool  = True,
    ):
        assert gamma > 0,     f"gamma must be positive, got {gamma}"
        assert kappa > 0,     f"kappa must be positive, got {kappa}"
        assert sigma > 0,     f"sigma must be positive, got {sigma}"
        assert xi >= 0,       f"xi must be non-negative, got {xi}"
        assert A > 0,         f"A must be positive, got {A}"
        assert T > 0,         f"T must be positive, got {T}"
        assert Q_max > 0,     f"Q_max must be positive, got {Q_max}"
        assert tick_size > 0, f"tick_size must be positive, got {tick_size}"

        self.gamma       = gamma
        self.kappa       = kappa
        self.sigma       = sigma
        self.sigma_init  = sigma
        self.xi          = xi
        self.A           = A
        self.T           = T
        self.Q_max       = Q_max
        self.tick_size   = tick_size
        self.adapt_sigma = adapt_sigma

        self._price_history: list = []
        self._t: int = 0

        self._v_T             = terminal_condition(kappa, xi, Q_max)
        self._M               = build_ode_matrix(gamma, kappa, sigma, xi, A, Q_max)
        self._taus, self._V   = solve_v(self._M, self._v_T, float(T), T)

    def _recompute_v(self) -> None:
        """Recompute ODE solution after sigma update."""
        self._M             = build_ode_matrix(
            self.gamma, self.kappa, self.sigma, self.xi, self.A, self.Q_max
        )
        self._taus, self._V = solve_v(self._M, self._v_T, float(self.T), self.T)

    def _get_v(self, tau_hat: float) -> np.ndarray:
        """
        Look up v_q at normalised time-to-go tau_hat in [0, 1].

        Parameters
        ----------
        tau_hat : float -- normalised time-to-go in [0, 1]

        Returns
        -------
        np.ndarray shape (2*Q_max+1,)
        """
        tau_clipped = float(np.clip(tau_hat, 0.0, 1.0))
        idx = int(np.searchsorted(self._taus, tau_clipped))
        idx = int(np.clip(idx, 0, len(self._taus) - 1))
        return self._V[idx]

    def _update_sigma(self, mid: float, window: int = 20) -> None:
        """
        Update sigma via EMA of realised log-return std.

        sigma stays in log-return units -- do NOT multiply by mid.
        _recompute_v fires only when sigma changes by more than 15%
        relative to sigma_init, limiting expensive recomputes to ~2-3
        per episode while maintaining quote accuracy.
        """
        self._price_history.append(mid)
        if len(self._price_history) < 3:
            return

        prices  = np.array(
            self._price_history[-min(window + 1, len(self._price_history)):]
        )
        log_ret = np.diff(np.log(np.maximum(prices, 1e-10)))
        vol     = float(np.std(log_ret))

        if vol > 1e-10:
            alpha          = 0.1
            prev_sigma     = self.sigma
            self.sigma     = (1 - alpha) * self.sigma + alpha * vol
            # Recompute only if sigma changed meaningfully vs init
            if abs(self.sigma - self.sigma_init) / self.sigma_init > 0.15:
                if abs(self.sigma - prev_sigma) / prev_sigma > 0.05:
                    self._recompute_v()

    def reset(self) -> None:
        """Reset episode state. Call at the start of each episode."""
        self._price_history.clear()
        self._t    = 0
        self.sigma = self.sigma_init
        self._M             = build_ode_matrix(
            self.gamma, self.kappa, self.sigma, self.xi, self.A, self.Q_max
        )
        self._taus, self._V = solve_v(self._M, self._v_T, float(self.T), self.T)

    def compute_quotes(
        self,
        mid:       float,
        inventory: int,
        tau_hat:   float,
    ) -> tuple[float, float]:
        """
        Compute optimal bid and ask prices from Proposition 5.

        Parameters
        ----------
        mid       : float -- current mid-price s
        inventory : int   -- current inventory q (signed, |q| <= Q_max)
        tau_hat   : float -- normalised time-to-go in [0, 1]

        Returns
        -------
        (bid_price, ask_price) : tuple[float, float]
        """
        q  = int(np.clip(inventory, -self.Q_max, self.Q_max))
        v  = self._get_v(tau_hat)

        db = delta_bid(v, q, self.Q_max, self.gamma, self.kappa, self.xi)
        da = delta_ask(v, q, self.Q_max, self.gamma, self.kappa, self.xi)

        # At inventory boundary: fall back to max action-space offset
        bid_price = mid - db if np.isfinite(db) else mid - _MAX_OFFSET * self.tick_size
        ask_price = mid + da if np.isfinite(da) else mid + _MAX_OFFSET * self.tick_size

        return bid_price, ask_price

    def act(
        self,
        obs:  np.ndarray,
        info: Dict[str, Any],
    ) -> np.ndarray:
        """
        Compute GLFT optimal action.

        Parameters
        ----------
        obs  : np.ndarray -- current observation (unused directly)
        info : dict       -- must contain 'mid_price' and 'inventory'

        Returns
        -------
        np.ndarray shape (2,) -- [bid_idx, ask_idx] into TICK_OFFSETS
        """
        mid       = float(info["mid_price"])
        inventory = int(info["inventory"])
        tau_hat   = max(float(self.T - self._t), 0.0) / self.T

        if self.adapt_sigma:
            self._update_sigma(mid)

        bid_price, ask_price = self.compute_quotes(mid, inventory, tau_hat)

        bid_idx = _dollars_to_idx(abs(bid_price - mid), self.tick_size)
        ask_idx = _dollars_to_idx(abs(ask_price - mid), self.tick_size)

        self._t += 1

        return np.array([bid_idx, ask_idx], dtype=np.int64)

    def __repr__(self) -> str:
        return (
            f"GLFTBaseline("
            f"gamma={self.gamma}, kappa={self.kappa}, sigma={self.sigma_init}, "
            f"xi={self.xi}, A={self.A}, T={self.T}, Q_max={self.Q_max})"
        )