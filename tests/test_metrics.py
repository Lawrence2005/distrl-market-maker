"""
tests/test_metrics.py

Unit tests for evaluation/metrics.py::cvar() — the episode-level risk
metric reported in results (CVaR_0.10 of per-episode or per-step P&Ls).

This is a different function from the action-SELECTION CVaR the agent
uses to pick quotes (agents/cvar_policy.py — see tests/test_cvar.py).
This one only ever summarizes a finished array of P&Ls for reporting;
it never influences training or action choice.

NOTE: this file's docstring also names Sharpe, MAP, MDD, PnLMAP and
win_rate — those remain untested; this pass covers cvar() only, which
had zero coverage anywhere in the suite. Extend this file for the rest
rather than assuming they're covered because this file exists.

Run with:
    pytest tests/test_metrics.py -v
"""

import numpy as np
import pytest

from evaluation.metrics import cvar


class TestCVaRAnalytical:
    @staticmethod
    def _uniform_midpoints(n: int) -> np.ndarray:
        """Same grid convention as QR-DQN's quantile midpoints: these ARE
        the quantile function of U[0,1] evaluated at tau_i=(2i-1)/(2n)."""
        return (np.arange(1, n + 1) - 0.5) / n

    def test_uniform_cvar_0_10_is_0_05(self):
        """Matches this file's own long-standing docstring claim."""
        values = self._uniform_midpoints(1000)
        assert cvar(values, alpha=0.10) == pytest.approx(0.05, abs=1e-6)

    def test_uniform_cvar_0_25_is_0_125(self):
        values = self._uniform_midpoints(1000)
        assert cvar(values, alpha=0.25) == pytest.approx(0.125, abs=1e-6)

    def test_alpha_1_equals_mean(self):
        """CVaR_1.0 = E[Z] — the risk-neutral limit."""
        rng = np.random.default_rng(0)
        values = rng.normal(size=500)
        assert cvar(values, alpha=1.0) == pytest.approx(values.mean(), abs=1e-9)

    @pytest.mark.parametrize("alpha", [0.05, 0.10, 0.25, 0.5, 0.9])
    def test_cvar_never_exceeds_mean(self, alpha):
        """CVaR_alpha is an average of the WORST outcomes, so it can never
        beat (exceed) the overall mean."""
        rng = np.random.default_rng(int(alpha * 1000))
        values = rng.normal(size=200)
        assert cvar(values, alpha=alpha) <= values.mean() + 1e-9

    def test_single_worst_value_at_tiny_alpha(self):
        """alpha small enough that n_tail == 1 — CVaR is exactly the worst
        single value."""
        values = np.array([5.0, -3.0, 1.0, 10.0, -100.0])
        assert cvar(values, alpha=0.01) == pytest.approx(-100.0)

    def test_constant_values_cvar_equals_the_constant(self):
        values = np.full(50, 7.0)
        assert cvar(values, alpha=0.1) == pytest.approx(7.0)

    def test_single_value(self):
        assert cvar(np.array([42.0]), alpha=0.10) == pytest.approx(42.0)

    def test_empty_array_returns_zero(self):
        """Matches mdd()'s own empty-input guard in the same module — cvar()
        was missing this (would raise a 'mean of empty slice' warning and
        return nan); fixed alongside this test."""
        assert cvar(np.array([]), alpha=0.10) == 0.0

    def test_order_independent(self):
        """CVaR sorts internally — input order must not matter."""
        values = np.array([3.0, -1.0, 7.0, -5.0, 2.0])
        shuffled = values[::-1].copy()
        assert cvar(values, alpha=0.4) == cvar(shuffled, alpha=0.4)

    @pytest.mark.parametrize("alpha", [0.1, 0.25, 0.5])
    def test_matches_sorted_slice_definition_directly(self, alpha):
        """Cross-check against the DEFINITION, not the implementation —
        otherwise this would just restate the source line for line."""
        rng = np.random.default_rng(1)
        values = rng.normal(size=137)
        n_tail = max(1, int(alpha * len(values)))
        expected = np.sort(values)[:n_tail].mean()
        assert cvar(values, alpha=alpha) == pytest.approx(expected)


class TestCVaRRealisticScale:
    def test_episode_pnl_scale_tail_is_the_worst_episode(self):
        """Mirrors actual project usage: CVaR_0.10 over a 15-episode
        holdout block of per-episode P&Ls — n_tail = max(1, int(0.10*15))
        = 1, so it's exactly the single worst episode."""
        pnls = np.array([120.0, -340.0, 80.0, 15.0, -50.0, 200.0, -900.0,
                          30.0, -20.0, 60.0, 10.0, -5.0, 400.0, -70.0, 90.0])
        assert cvar(pnls, alpha=0.10) == pytest.approx(-900.0)

    def test_step_pnl_scale_averages_several_worst_steps(self):
        """At step scale (390 steps/episode), alpha=0.10 averages ~39 of
        the worst steps, not just one — sanity-check the scale difference
        against the episode-scale test above."""
        rng = np.random.default_rng(2)
        step_pnls = rng.normal(loc=0.0, scale=10.0, size=390)
        n_tail = max(1, int(0.10 * 390))
        assert n_tail == 39
        result = cvar(step_pnls, alpha=0.10)
        assert result < step_pnls.mean()   # the tail must be worse than the average
