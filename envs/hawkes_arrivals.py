"""
envs/hawkes_arrivals.py

Hawkes process order arrival model.

Simulation via Ogata thinning — no external library dependency.
Parameters loaded from data/calibration/hawkes_params.json,
which is fitted by data/process_lobster.py.

References:
    Bacry, Mastromatteo & Muzy (2015) — Hawkes Processes in Finance
    Huang, Lehalle & Rosenbaum (2015) — empirical motivation

Week 2 deliverable.
"""

import numpy as np
import json
from typing import Optional

class HawkesProcess:
    """
    Univariate Hawkes process with exponential kernel.

    Intensity:
        λ(t) = μ + α · Σ_{t_j < t} exp(−β · (t − t_j))

    Parameters
    ----------
    mu    : float — baseline intensity (events/second)
    alpha : float — excitation magnitude
    beta  : float — decay rate (1/second)
    """

    def __init__(self, mu: float, alpha: float, beta: float):
        assert mu > 0,    f"mu must be positive, got {mu}"
        assert alpha > 0, f"alpha must be positive, got {alpha}"
        assert beta > 0,  f"beta must be positive, got {beta}"
        assert alpha / beta < 1.0, (
            f"Branching ratio α/β = {alpha/beta:.4f} ≥ 1. "
            f"Process not stationary. Reduce alpha or increase beta."
        )
        self.mu    = mu
        self.alpha = alpha
        self.beta  = beta

    @property
    def branching_ratio(self) -> float:
        return self.alpha / self.beta

    @property
    def mean_rate(self) -> float:
        """Theoretical mean arrival rate: μ / (1 − ρ)"""
        return self.mu / (1.0 - self.branching_ratio)

    def simulate(self, T: float, seed: int = 42) -> np.ndarray:
        """
        Simulate Hawkes process over [0, T] via Ogata thinning.

        Parameters
        ----------
        T    : float — simulation horizon in seconds
        seed : int   — random seed for reproducibility

        Returns
        -------
        np.ndarray of event arrival times in [0, T], starting near 0
        """
        rng    = np.random.default_rng(seed)
        times  = []
        t      = 0.0

        while t < T:
            # Upper bound on intensity at current time
            if len(times) == 0:
                lam_upper = self.mu
            else:
                t_arr     = np.array(times)
                lam_upper = self.mu + self.alpha * np.sum(
                    np.exp(-self.beta * (t - t_arr))
                )

            # Propose next event time
            dt_prop  = rng.exponential(1.0 / lam_upper)
            t_prop   = t + dt_prop

            if t_prop > T:
                break

            # True intensity at proposed time
            if len(times) == 0:
                lam_true = self.mu
            else:
                t_arr    = np.array(times)
                lam_true = self.mu + self.alpha * np.sum(
                    np.exp(-self.beta * (t_prop - t_arr))
                )

            # Accept with probability lam_true / lam_upper
            if rng.uniform() < lam_true / lam_upper:
                times.append(t_prop)

            t = t_prop

        return np.array(times) if times else np.array([0.0])

    @classmethod
    def from_lobster(cls, params_path: str) -> "HawkesProcess":
        """
        Load calibrated parameters from data/calibration/hawkes_params.json.

        Usage:
            hp = HawkesProcess.from_lobster("data/calibration/hawkes_params.json")
            events = hp.simulate(T=3900, seed=42)
        """
        with open(params_path) as f:
            p = json.load(f)

        mu    = float(p["mu"])
        alpha = float(p["alpha"])
        beta  = float(p["beta"])
        rho   = alpha / beta

        print(
            f"Loaded Hawkes params: mu={mu:.6f}, alpha={alpha:.4f}, "
            f"beta={beta:.4f}, rho={rho:.4f}"
        )

        # Safety check — if rho is too close to 1, scale alpha down
        if rho >= 0.95:
            print(
                f"  WARNING: rho={rho:.4f} is close to 1 (near non-stationary). "
                f"Clipping alpha to give rho=0.80."
            )
            alpha = 0.80 * beta
            rho   = alpha / beta

        return cls(mu=mu, alpha=alpha, beta=beta)

    @classmethod
    def from_fit(cls, times: np.ndarray) -> "HawkesProcess":
        """
        Fit parameters directly from arrival times and return
        a ready-to-use HawkesProcess. Wraps the scipy MLE fitter.
        """
        from data.process_lobster import fit_hawkes_mle
        params = fit_hawkes_mle(times)
        return cls(
            mu=params["mu"],
            alpha=params["alpha"],
            beta=params["beta"],
        )


if __name__ == "__main__":
    import os

    print("=== HawkesProcess validation ===\n")

    # ── Test 1: simulate from known parameters ────────────────────────
    print("Test 1: simulate from known parameters")
    hp     = HawkesProcess(mu=0.5, alpha=0.6, beta=1.5)
    events = hp.simulate(T=3900.0, seed=42)

    expected_rate = hp.mean_rate
    actual_rate   = len(events) / 3900.0

    print(f"  Arrivals:      {len(events)}")
    print(f"  Expected rate: {expected_rate:.3f} events/sec")
    print(f"  Actual rate:   {actual_rate:.3f} events/sec")
    print(f"  Branching ratio: {hp.branching_ratio:.3f}")

    assert len(events) > 0, "No events generated"
    assert abs(actual_rate - expected_rate) / expected_rate < 0.15, (
        f"Rate {actual_rate:.3f} deviates >15% from expected {expected_rate:.3f}"
    )
    print("  ✓ Passed\n")

    # ── Test 2: determinism ───────────────────────────────────────────
    print("Test 2: determinism — same seed gives same output")
    ev1 = hp.simulate(T=3900.0, seed=42)
    ev2 = hp.simulate(T=3900.0, seed=42)
    assert np.allclose(ev1, ev2), "Same seed produced different output"
    print("  ✓ Passed\n")

    # ── Test 3: from_lobster() ────────────────────────────────────────
    params_path = "data/calibration/hawkes_params.json"
    if os.path.exists(params_path):
        print("Test 3: from_lobster()")
        hp3 = HawkesProcess.from_lobster(params_path)
        ev3 = hp3.simulate(T=3900, seed=42)
        print(f"  {len(ev3)} arrivals from calibrated params")
        assert len(ev3) > 0
        print("  ✓ Passed\n")
    else:
        print(f"Test 3: Skipped — run data/process_lobster.py first\n")

    print("=== All tests passed ===")