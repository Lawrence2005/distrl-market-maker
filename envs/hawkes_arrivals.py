"""
Hawkes process order arrival model.

Replaces ABIDES's default Poisson arrivals with a self-exciting Hawkes
process, which produces the bursty, clustered order flow observed in
real limit order books.

A Hawkes process has intensity:
    lambda(t) = mu + sum_j phi(t - t_j)
where phi is the excitation kernel. We use an exponential kernel:
    phi(t) = alpha * exp(-beta * t)
with branching ratio rho = alpha / beta < 1 for stationarity.

Parameters (mu, alpha, beta) are calibrated via MLE on LOBSTER tick data.
See data/calibration/hawkes_params.json for fitted values.

Why Hawkes over Poisson:
  - Poisson: constant arrival rate — unrealistic; misses clustering
  - Queue-reactive (Huang et al. 2015): intensity depends on queue size —
    empirically validated but harder to calibrate
  - Hawkes (this module): self-exciting; each arrival temporarily raises
    future intensity — captures the same clustering empirically documented
    by Huang et al. with a tractable mathematical framework

References:
  PRIMARY:    Bacry, Mastromatteo & Muzy (2015) 'Hawkes Processes in Finance'
              Canonical reference for self-exciting order flow in finance;
              derives multivariate Hawkes for bid/ask flow; calibration via MLE.
  MOTIVATION: Huang, Lehalle & Rosenbaum (2015) — empirically documents
              arrival clustering in real LOBs via queue-reactive model;
              motivates Hawkes over Poisson even though they use a different
              (queue-reactive) formulation.

Week 2 deliverable.
"""
import numpy as np
from typing import Tuple, Optional


class HawkesProcess:
    """
    Univariate Hawkes process with exponential kernel.

    Parameters
    ----------
    mu    : float — baseline intensity (events/sec)
    alpha : float — excitation magnitude
    beta  : float — decay rate of excitation kernel
    """

    def __init__(self, mu: float, alpha: float, beta: float):
        assert alpha / beta < 1, "Branching ratio rho = alpha/beta must be < 1 for stationarity"
        self.mu    = mu
        self.alpha = alpha
        self.beta  = beta

    def intensity(self, t: float, history: np.ndarray) -> float:
        """
        Compute current intensity lambda(t) given event history.

        lambda(t) = mu + sum_{t_j < t} alpha * exp(-beta * (t - t_j))
        """
        if len(history) == 0:
            return self.mu
        past = history[history < t]
        excitation = self.alpha * np.sum(np.exp(-self.beta * (t - past)))
        return self.mu + excitation

    def simulate(self, T: float, seed: Optional[int] = None) -> np.ndarray:
        """
        Simulate Hawkes process over [0, T] via Ogata's thinning algorithm.

        Returns array of event times.
        """
        # TODO: implement Ogata thinning
        raise NotImplementedError

    @classmethod
    def from_lobster(cls, params_path: str) -> "HawkesProcess":
        """Load calibrated parameters from data/calibration/hawkes_params.json."""
        import json
        with open(params_path) as f:
            p = json.load(f)
        return cls(mu=p["mu"], alpha=p["alpha"], beta=p["beta"])


class MultivariateHawkes:
    """
    Bivariate Hawkes process for bid-side and ask-side arrivals.

    Each side can excite itself (self-excitation) and the other side
    (cross-excitation) — models the empirical finding that a large
    bid-side order burst often triggers ask-side responses.

    Reference: Bacry et al. (2015) Section 3 — multivariate extension.
    """
    # TODO: implement
    pass
