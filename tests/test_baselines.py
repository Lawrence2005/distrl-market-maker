"""
tests/test_baselines.py
-----------------------
Unit tests for baselines/fixed_spread.py, baselines/avellaneda_stoikov.py,
and baselines/glft.py.

Testing philosophy:
  - fixed_spread.py  : interface contract only (trivial maths)
  - avellaneda_stoikov.py : analytical values from the AS (2008) paper
  - glft.py          : analytical values from GLFT Proposition 5,
                       plus ODE system properties and numerical stability

Run with:
    python -m pytest tests/test_baselines.py -v
"""

import numpy as np
import pytest

from baselines.fixed_spread import FixedSpreadBaseline, _offset_to_idx
from baselines.avellaneda_stoikov import AvellanedaStoikovBaseline
from baselines.glft import (
    GLFTBaseline,
    build_ode_matrix,
    terminal_condition,
    solve_v,
    delta_bid,
    delta_ask,
    spread,
)
from envs.lob_env import TICK_OFFSETS, N_OFFSET_LEVELS


# ── Shared helpers ────────────────────────────────────────────────────────────

DUMMY_OBS  = np.zeros(17, dtype=np.float32)
DUMMY_INFO = {"mid_price": 1000.0, "inventory": 0}


def _info(mid: float = 1000.0, inventory: int = 0) -> dict:
    return {"mid_price": mid, "inventory": inventory}


# ═══════════════════════════════════════════════════════════════════════════════
# FixedSpreadBaseline
# ═══════════════════════════════════════════════════════════════════════════════

class TestFixedSpread:

    # ── Construction ──────────────────────────────────────────────────────────

    def test_default_construction(self):
        b = FixedSpreadBaseline()
        assert b.half_spread_ticks == 2

    def test_custom_half_spread(self):
        b = FixedSpreadBaseline(half_spread_ticks=3)
        assert b.half_spread_ticks == 3

    @pytest.mark.parametrize("ticks", [0, 5, -1, 10])
    def test_invalid_half_spread_raises(self, ticks):
        with pytest.raises(AssertionError):
            FixedSpreadBaseline(half_spread_ticks=ticks)

    @pytest.mark.parametrize("ticks", [1, 2, 3, 4])
    def test_valid_half_spreads(self, ticks):
        b = FixedSpreadBaseline(half_spread_ticks=ticks)
        assert b.half_spread_ticks == ticks

    # ── Action contract ───────────────────────────────────────────────────────

    def test_act_returns_ndarray(self):
        b = FixedSpreadBaseline()
        action = b.act(DUMMY_OBS, DUMMY_INFO)
        assert isinstance(action, np.ndarray)

    def test_act_shape(self):
        b = FixedSpreadBaseline()
        action = b.act(DUMMY_OBS, DUMMY_INFO)
        assert action.shape == (2,)

    def test_act_dtype(self):
        b = FixedSpreadBaseline()
        action = b.act(DUMMY_OBS, DUMMY_INFO)
        assert np.issubdtype(action.dtype, np.integer)

    def test_act_indices_in_range(self):
        for ticks in range(1, 5):
            b = FixedSpreadBaseline(half_spread_ticks=ticks)
            action = b.act(DUMMY_OBS, DUMMY_INFO)
            assert 0 <= action[0] < N_OFFSET_LEVELS
            assert 0 <= action[1] < N_OFFSET_LEVELS

    def test_action_is_constant(self):
        """Fixed spread must return the same action regardless of obs/info."""
        b = FixedSpreadBaseline(half_spread_ticks=2)
        a1 = b.act(DUMMY_OBS, _info(mid=100.0,  inventory=+5))
        a2 = b.act(DUMMY_OBS, _info(mid=2000.0, inventory=-5))
        a3 = b.act(np.ones(17, dtype=np.float32), DUMMY_INFO)
        np.testing.assert_array_equal(a1, a2)
        np.testing.assert_array_equal(a1, a3)

    def test_symmetric_quotes(self):
        """bid_idx == ask_idx — always symmetric."""
        b = FixedSpreadBaseline(half_spread_ticks=3)
        action = b.act(DUMMY_OBS, DUMMY_INFO)
        assert action[0] == action[1]

    def test_wider_ticks_gives_larger_index(self):
        """Larger half_spread_ticks → deeper offset → larger index."""
        b1 = FixedSpreadBaseline(half_spread_ticks=1)
        b2 = FixedSpreadBaseline(half_spread_ticks=3)
        a1 = b1.act(DUMMY_OBS, DUMMY_INFO)
        a2 = b2.act(DUMMY_OBS, DUMMY_INFO)
        assert a2[0] > a1[0]

    # ── Reset ─────────────────────────────────────────────────────────────────

    def test_reset_does_not_crash(self):
        b = FixedSpreadBaseline()
        b.reset()   # stateless — should be a no-op

    def test_action_unchanged_after_reset(self):
        b = FixedSpreadBaseline(half_spread_ticks=2)
        a_before = b.act(DUMMY_OBS, DUMMY_INFO).copy()
        b.reset()
        a_after  = b.act(DUMMY_OBS, DUMMY_INFO)
        np.testing.assert_array_equal(a_before, a_after)

    # ── Helper ────────────────────────────────────────────────────────────────

    def test_offset_to_idx_centre(self):
        assert TICK_OFFSETS[_offset_to_idx(1)] == 1
        assert TICK_OFFSETS[_offset_to_idx(10)] == 10

    def test_offset_to_idx_clamping(self):
        assert _offset_to_idx(-10) == 0
        assert _offset_to_idx(+10) == N_OFFSET_LEVELS - 1

    # ── repr ──────────────────────────────────────────────────────────────────

    def test_repr(self):
        b = FixedSpreadBaseline(half_spread_ticks=2)
        assert "FixedSpread" in repr(b)
        assert "2" in repr(b)


# ═══════════════════════════════════════════════════════════════════════════════
# AvellanedaStoikovBaseline — analytical tests against AS (2008) paper
# ═══════════════════════════════════════════════════════════════════════════════

class TestAvellanedaStoikov:

    @pytest.fixture
    def as_baseline(self):
        return AvellanedaStoikovBaseline(
            gamma=0.1, kappa=1.5, sigma=0.01, T=390,
            tick_size=0.01, adapt_sigma=False,
        )

    # ── reservation_price ─────────────────────────────────────────────────────

    def test_reservation_price_at_zero_inventory(self, as_baseline):
        """r(s, 0, t) = s exactly — no skew when flat."""
        r = as_baseline.reservation_price(mid=1000.0, inventory=0.0, tau=100.0)
        assert r == pytest.approx(1000.0)

    def test_reservation_price_long_inventory_below_mid(self, as_baseline):
        """Long inventory (q>0) → r < s (quotes skew toward selling)."""
        r = as_baseline.reservation_price(mid=1000.0, inventory=5.0, tau=100.0)
        assert r < 1000.0

    def test_reservation_price_short_inventory_above_mid(self, as_baseline):
        """Short inventory (q<0) → r > s (quotes skew toward buying)."""
        r = as_baseline.reservation_price(mid=1000.0, inventory=-5.0, tau=100.0)
        assert r > 1000.0

    def test_reservation_price_linear_in_inventory(self, as_baseline):
        """r decreases linearly with inventory — equal steps."""
        r0 = as_baseline.reservation_price(1000.0, 0.0, 100.0)
        r1 = as_baseline.reservation_price(1000.0, 1.0, 100.0)
        r2 = as_baseline.reservation_price(1000.0, 2.0, 100.0)
        assert (r0 - r1) == pytest.approx(r1 - r2, rel=1e-6)

    def test_reservation_price_zero_at_terminal(self, as_baseline):
        """At tau=0 (terminal), skew vanishes: r = s regardless of q."""
        r = as_baseline.reservation_price(mid=1000.0, inventory=10.0, tau=0.0)
        assert r == pytest.approx(1000.0)

    def test_reservation_price_analytical_value(self, as_baseline):
        """r = s - q·γ·σ²·τ."""
        mid, q, tau = 1000.0, 3.0, 100.0
        expected = mid - q * 0.1 * (0.01 ** 2) * tau
        r = as_baseline.reservation_price(mid, q, tau)
        assert r == pytest.approx(expected, rel=1e-8)

    # ── optimal_spread ────────────────────────────────────────────────────────

    def test_spread_at_terminal_equals_base_term(self, as_baseline):
        """At tau=0: δ* = (2/γ)·ln(1 + γ/κ) — inventory risk term vanishes."""
        gamma, kappa = 0.1, 1.5
        base = (2.0 / gamma) * np.log(1.0 + gamma / kappa)
        s = as_baseline.optimal_spread(tau=0.0)
        assert s == pytest.approx(base, rel=1e-8)

    def test_spread_increases_with_tau(self, as_baseline):
        """Wider spread earlier in episode (more inventory risk remaining)."""
        s1 = as_baseline.optimal_spread(tau=1.0)
        s2 = as_baseline.optimal_spread(tau=100.0)
        assert s2 > s1

    def test_spread_increases_with_sigma(self):
        """Higher volatility → wider spread."""
        b1 = AvellanedaStoikovBaseline(gamma=0.1, kappa=1.5, sigma=0.01,
                                        T=390, adapt_sigma=False)
        b2 = AvellanedaStoikovBaseline(gamma=0.1, kappa=1.5, sigma=0.02,
                                        T=390, adapt_sigma=False)
        s1 = b1.optimal_spread(tau=100.0)
        s2 = b2.optimal_spread(tau=100.0)
        assert s2 > s1

    def test_spread_increases_with_kappa_decreasing(self):
        """Lower κ (fewer fills) → wider spread."""
        b1 = AvellanedaStoikovBaseline(gamma=0.1, kappa=3.0, sigma=0.01,
                                        T=390, adapt_sigma=False)
        b2 = AvellanedaStoikovBaseline(gamma=0.1, kappa=1.0, sigma=0.01,
                                        T=390, adapt_sigma=False)
        s1 = b1.optimal_spread(tau=100.0)
        s2 = b2.optimal_spread(tau=100.0)
        assert s2 > s1

    def test_spread_positive(self, as_baseline):
        """Spread must always be positive."""
        for tau in [0.0, 1.0, 50.0, 390.0]:
            assert as_baseline.optimal_spread(tau) > 0.0

    def test_spread_analytical_value(self, as_baseline):
        """δ* = γ·σ²·τ + (2/γ)·ln(1 + γ/κ)."""
        gamma, kappa, sigma, tau = 0.1, 1.5, 0.01, 100.0
        expected = gamma * sigma**2 * tau + (2/gamma) * np.log(1 + gamma/kappa)
        assert as_baseline.optimal_spread(tau) == pytest.approx(expected, rel=1e-8)

    # ── compute_quotes ────────────────────────────────────────────────────────

    def test_bid_below_mid(self, as_baseline):
        bid, ask = as_baseline.compute_quotes(1000.0, 0.0, 100.0)
        assert bid < 1000.0

    def test_ask_above_mid(self, as_baseline):
        bid, ask = as_baseline.compute_quotes(1000.0, 0.0, 100.0)
        assert ask > 1000.0

    def test_bid_below_ask(self, as_baseline):
        bid, ask = as_baseline.compute_quotes(1000.0, 0.0, 100.0)
        assert bid < ask

    def test_symmetric_at_zero_inventory(self, as_baseline):
        """At q=0: bid and ask equidistant from mid."""
        bid, ask = as_baseline.compute_quotes(1000.0, 0.0, 100.0)
        assert (1000.0 - bid) == pytest.approx(ask - 1000.0, rel=1e-6)

    def test_long_inventory_skews_ask_closer(self, as_baseline):
        """Long (q>0): ask closer to mid than bid (want to sell)."""
        bid, ask = as_baseline.compute_quotes(1000.0, 5.0, 100.0)
        assert (ask - 1000.0) < (1000.0 - bid)

    def test_short_inventory_skews_bid_closer(self, as_baseline):
        """Short (q<0): bid closer to mid than ask (want to buy)."""
        bid, ask = as_baseline.compute_quotes(1000.0, -5.0, 100.0)
        assert (1000.0 - bid) < (ask - 1000.0)

    def test_quotes_scale_with_mid(self, as_baseline):
        """Quotes should track mid-price."""
        bid1, ask1 = as_baseline.compute_quotes(1000.0, 0.0, 100.0)
        bid2, ask2 = as_baseline.compute_quotes(2000.0, 0.0, 100.0)
        assert bid2 > bid1
        assert ask2 > ask1

    # ── act() interface ───────────────────────────────────────────────────────

    def test_act_shape(self, as_baseline):
        as_baseline.reset()
        action = as_baseline.act(DUMMY_OBS, DUMMY_INFO)
        assert action.shape == (2,)

    def test_act_dtype(self, as_baseline):
        as_baseline.reset()
        action = as_baseline.act(DUMMY_OBS, DUMMY_INFO)
        assert np.issubdtype(action.dtype, np.integer)

    def test_act_indices_in_range(self, as_baseline):
        as_baseline.reset()
        action = as_baseline.act(DUMMY_OBS, DUMMY_INFO)
        assert 0 <= action[0] < N_OFFSET_LEVELS
        assert 0 <= action[1] < N_OFFSET_LEVELS

    def test_act_increments_step_counter(self, as_baseline):
        as_baseline.reset()
        assert as_baseline._t == 0
        as_baseline.act(DUMMY_OBS, DUMMY_INFO)
        assert as_baseline._t == 1

    def test_reset_clears_step_counter(self, as_baseline):
        as_baseline.reset()
        for _ in range(10):
            as_baseline.act(DUMMY_OBS, DUMMY_INFO)
        as_baseline.reset()
        assert as_baseline._t == 0

    def test_reset_clears_price_history(self):
        b = AvellanedaStoikovBaseline(adapt_sigma=True)
        b.reset()
        for i in range(10):
            b.act(DUMMY_OBS, _info(mid=1000.0 + i))
        b.reset()
        assert len(b._price_history) == 0

    def test_long_inventory_gives_smaller_ask_idx(self, as_baseline):
        """Long position: ask quoted closer to mid → smaller ask index."""
        as_baseline.reset()
        a_long = as_baseline.act(DUMMY_OBS, _info(inventory=+5))
        as_baseline._t -= 1
        a_flat = as_baseline.act(DUMMY_OBS, _info(inventory=0))
        # ask_idx smaller means ask is closer to mid
        assert a_long[1] <= a_flat[1]

    def test_short_inventory_gives_smaller_bid_idx(self, as_baseline):
        """Short position: bid quoted closer to mid → smaller bid index."""
        as_baseline.reset()
        a_short = as_baseline.act(DUMMY_OBS, _info(inventory=-5))
        as_baseline._t -= 1
        a_flat  = as_baseline.act(DUMMY_OBS, _info(inventory=0))
        assert a_short[0] <= a_flat[0]

    # ── update_sigma ──────────────────────────────────────────────────────────

    def test_sigma_updates_from_price_history(self):
        b = AvellanedaStoikovBaseline(sigma=0.01, adapt_sigma=True)
        b.reset()
        sigma_before = b.sigma
        # Feed volatile prices
        for p in [1000, 1010, 990, 1020, 980, 1015, 985]:
            b.update_sigma(float(p))
        # sigma should have changed
        assert b.sigma != sigma_before

    def test_constant_prices_sigma_unchanged(self):
        b = AvellanedaStoikovBaseline(sigma=0.01, adapt_sigma=True)
        b.reset()
        for _ in range(30):
            b.update_sigma(1000.0)
        # Constant prices → zero vol → sigma stays at init (guard in update_sigma)
        assert b.sigma == pytest.approx(0.01)

    # ── Construction guards ───────────────────────────────────────────────────

    @pytest.mark.parametrize("param,val", [
        ("gamma", 0.0), ("gamma", -0.1),
        ("kappa", 0.0), ("kappa", -1.0),
        ("sigma", 0.0), ("sigma", -0.01),
        ("T",     0),   ("T",     -10),
        ("tick_size", 0.0), ("tick_size", -0.01),
    ])
    def test_invalid_params_raise(self, param, val):
        kwargs = dict(gamma=0.1, kappa=1.5, sigma=0.01, T=390, tick_size=0.01)
        kwargs[param] = val
        with pytest.raises(AssertionError):
            AvellanedaStoikovBaseline(**kwargs)

    # ── repr ──────────────────────────────────────────────────────────────────

    def test_repr(self, as_baseline):
        r = repr(as_baseline)
        assert "AvellanedaStoikov" in r
        assert "gamma" in r


# ═══════════════════════════════════════════════════════════════════════════════
# GLFTBaseline — Proposition 5 analytical tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGLFTPureFunctions:
    """Tests for the module-level pure functions — the testable mathematical core."""

    GAMMA = 0.1
    KAPPA = 1.5
    SIGMA = 0.01
    XI    = 0.01
    A     = 1.0
    Q_MAX = 3

    @pytest.fixture
    def M(self):
        return build_ode_matrix(
            self.GAMMA, self.KAPPA, self.SIGMA, self.XI, self.A, self.Q_MAX
        )

    @pytest.fixture
    def v_T(self):
        return terminal_condition(self.KAPPA, self.XI, self.Q_MAX)

    @pytest.fixture
    def V(self, M, v_T):
        _, V = solve_v(M, v_T, T=390.0, n_steps=390)
        return V

    # ── build_ode_matrix ──────────────────────────────────────────────────────

    def test_matrix_shape(self, M):
        n = 2 * self.Q_MAX + 1
        assert M.shape == (n, n)

    def test_matrix_tridiagonal(self, M):
        """M should be tridiagonal — no elements beyond first off-diagonal."""
        n = M.shape[0]
        for i in range(n):
            for j in range(n):
                if abs(i - j) > 1:
                    assert M[i, j] == 0.0, f"Non-zero at ({i},{j})"

    def test_diagonal_zero_at_q0(self, M):
        """α·q² = 0 at q=0."""
        assert M[self.Q_MAX, self.Q_MAX] == 0.0

    def test_diagonal_positive_at_nonzero_q(self, M):
        """α·q² > 0 for q ≠ 0."""
        for i in range(M.shape[0]):
            if i != self.Q_MAX:
                assert M[i, i] > 0.0

    def test_off_diagonal_negative(self, M):
        """Fill-arrival terms are negative (decay)."""
        n = M.shape[0]
        for i in range(n - 1):
            assert M[i, i + 1] < 0.0
            assert M[i + 1, i] < 0.0

    def test_diagonal_scales_with_q_squared(self, M):
        """M[q, q] = α·q² → ratio of diagonals = ratio of q²."""
        alpha = M[self.Q_MAX + 1, self.Q_MAX + 1]   # q=1
        alpha4 = M[self.Q_MAX + 2, self.Q_MAX + 2]  # q=2
        assert alpha4 == pytest.approx(4 * alpha, rel=1e-8)

    def test_off_diagonal_constant(self, M):
        """All off-diagonal elements equal (same decay rate)."""
        n = M.shape[0]
        off_vals = [M[i, i+1] for i in range(n-1)]
        assert all(abs(v - off_vals[0]) < 1e-12 for v in off_vals)

    def test_alpha_formula(self):
        """α = κ²·γ·σ²/2."""
        gamma, kappa, sigma = 0.1, 1.5, 0.01
        expected_alpha = (kappa**2 * gamma * sigma**2) / 2.0
        M = build_ode_matrix(gamma, kappa, sigma, 0.01, 1.0, 2)
        # diagonal at q=1: α*1^2 = α
        assert M[3, 3] == pytest.approx(expected_alpha, rel=1e-8)

    def test_eta_formula(self):
        """η = A·(1 + γ/κ)^{-(1 + κ/γ)}."""
        gamma, kappa, A = 0.1, 1.5, 1.0
        expected_eta = A * (1.0 + gamma/kappa) ** (-(1.0 + kappa/gamma))
        M = build_ode_matrix(gamma, kappa, 0.01, 0.0, A, 2)
        # off-diagonal = -decay = -η·exp(-κ²ξ/2) with ξ=0 → -η
        assert M[2, 3] == pytest.approx(-expected_eta, rel=1e-6)

    # ── terminal_condition ────────────────────────────────────────────────────

    def test_terminal_length(self, v_T):
        assert len(v_T) == 2 * self.Q_MAX + 1

    def test_terminal_at_q0_is_one(self, v_T):
        """v_0(T) = exp(0) = 1."""
        assert v_T[self.Q_MAX] == pytest.approx(1.0)

    def test_terminal_symmetric(self, v_T):
        """v_q(T) = v_{-q}(T) — symmetric around q=0."""
        for q in range(1, self.Q_MAX + 1):
            assert v_T[self.Q_MAX + q] == pytest.approx(v_T[self.Q_MAX - q], rel=1e-10)

    def test_terminal_all_positive(self, v_T):
        assert np.all(v_T > 0)

    def test_terminal_leq_one(self, v_T):
        """exp(-κ²ξq²/2) ≤ 1 for all q (exponent non-positive)."""
        assert np.all(v_T <= 1.0 + 1e-12)

    def test_terminal_decreasing_from_centre(self, v_T):
        """v_q(T) decreases as |q| increases."""
        for q in range(self.Q_MAX):
            assert v_T[self.Q_MAX + q] >= v_T[self.Q_MAX + q + 1]

    def test_terminal_xi_zero_all_ones(self):
        """With ξ=0: v_q(T) = exp(0) = 1 for all q."""
        v = terminal_condition(kappa=1.5, xi=0.0, Q_max=3)
        np.testing.assert_allclose(v, 1.0)

    def test_terminal_analytical_value(self):
        """v_q(T) = exp(-κ²ξq²/2) for q=2."""
        kappa, xi, q = 1.5, 0.01, 2
        expected = np.exp(-0.5 * kappa**2 * xi * q**2)
        v = terminal_condition(kappa, xi, Q_max=3)
        assert v[3 + q] == pytest.approx(expected, rel=1e-10)

    # ── solve_v ───────────────────────────────────────────────────────────────

    def test_solve_v_shape(self, M, v_T):
        taus, V = solve_v(M, v_T, T=390.0, n_steps=390)
        assert V.shape == (391, 2 * self.Q_MAX + 1)
        assert len(taus) == 391

    def test_solve_v_terminal_matches(self, M, v_T):
        """V at tau=0 should exactly equal v_T."""
        _, V = solve_v(M, v_T, T=390.0, n_steps=390)
        np.testing.assert_allclose(V[0], v_T, atol=1e-10)

    def test_solve_v_all_positive(self, M, v_T):
        """v_q(t) must remain positive throughout — it's an exponential."""
        _, V = solve_v(M, v_T, T=390.0, n_steps=390)
        assert np.all(V > 0)

    def test_solve_v_symmetry(self, M, v_T):
        """v_q(t) = v_{-q}(t) at all times (by ODE symmetry)."""
        _, V = solve_v(M, v_T, T=390.0, n_steps=10)
        for i in range(V.shape[0]):
            for q in range(1, self.Q_MAX + 1):
                rel_err = abs(V[i, self.Q_MAX + q] - V[i, self.Q_MAX - q]) / (V[i, self.Q_MAX + q] + 1e-300)
                assert rel_err < 1e-6, f"Symmetry broken at tau_idx={i}, q={q}"

    # ── delta_bid ─────────────────────────────────────────────────────────────

    def test_delta_bid_at_boundary_is_inf(self, V):
        """At q=Q_max: no bid — δ_b = inf."""
        v = V[200]
        assert np.isinf(delta_bid(v, self.Q_MAX, self.Q_MAX,
                                   self.GAMMA, self.KAPPA, self.XI))

    def test_delta_bid_positive_interior(self, V):
        """δ_b > 0 for all interior q."""
        v = V[200]
        for q in range(-self.Q_MAX, self.Q_MAX):
            db = delta_bid(v, q, self.Q_MAX, self.GAMMA, self.KAPPA, self.XI)
            assert db > 0.0, f"Negative delta_bid at q={q}"

    def test_delta_bid_analytical_formula(self, V):
        """δ_b = (1/κ)·ln(v_q/v_{q+1}) + ξ/2 + (1/γ)·ln(1+γ/κ)."""
        v = V[200]
        q = 0
        i = q + self.Q_MAX
        expected = ((1/self.KAPPA) * np.log(v[i]/v[i+1])
                    + self.XI/2
                    + (1/self.GAMMA) * np.log(1 + self.GAMMA/self.KAPPA))
        result = delta_bid(v, q, self.Q_MAX, self.GAMMA, self.KAPPA, self.XI)
        assert result == pytest.approx(expected, rel=1e-8)

    # ── delta_ask ─────────────────────────────────────────────────────────────

    def test_delta_ask_at_boundary_is_inf(self, V):
        """At q=-Q_max: no ask — δ_a = inf."""
        v = V[200]
        assert np.isinf(delta_ask(v, -self.Q_MAX, self.Q_MAX,
                                   self.GAMMA, self.KAPPA, self.XI))

    def test_delta_ask_positive_interior(self, V):
        v = V[200]
        for q in range(-self.Q_MAX + 1, self.Q_MAX + 1):
            da = delta_ask(v, q, self.Q_MAX, self.GAMMA, self.KAPPA, self.XI)
            assert da > 0.0, f"Negative delta_ask at q={q}"

    def test_delta_ask_analytical_formula(self, V):
        v = V[200]
        q = 0
        i = q + self.Q_MAX
        expected = ((1/self.KAPPA) * np.log(v[i]/v[i-1])
                    + self.XI/2
                    + (1/self.GAMMA) * np.log(1 + self.GAMMA/self.KAPPA))
        result = delta_ask(v, q, self.Q_MAX, self.GAMMA, self.KAPPA, self.XI)
        assert result == pytest.approx(expected, rel=1e-8)

    # ── Symmetry at q=0 ───────────────────────────────────────────────────────

    def test_symmetric_quotes_at_zero_inventory(self, V):
        """At q=0: δ_b = δ_a (by v symmetry)."""
        v = V[200]
        db = delta_bid(v, 0, self.Q_MAX, self.GAMMA, self.KAPPA, self.XI)
        da = delta_ask(v, 0, self.Q_MAX, self.GAMMA, self.KAPPA, self.XI)
        assert db == pytest.approx(da, rel=1e-6)

    # ── Inventory skew direction ───────────────────────────────────────────────

    def test_long_inventory_ask_tighter_than_bid(self, V):
        """q>0: δ_a < δ_b (ask closer to mid — want to sell)."""
        v = V[200]
        for q in range(1, self.Q_MAX):
            db = delta_bid(v, q, self.Q_MAX, self.GAMMA, self.KAPPA, self.XI)
            da = delta_ask(v, q, self.Q_MAX, self.GAMMA, self.KAPPA, self.XI)
            assert da < db, f"q={q}: ask should be tighter (da={da:.4f}, db={db:.4f})"

    def test_short_inventory_bid_tighter_than_ask(self, V):
        """q<0: δ_b < δ_a (bid closer to mid — want to buy)."""
        v = V[200]
        for q in range(-self.Q_MAX + 1, 0):
            db = delta_bid(v, q, self.Q_MAX, self.GAMMA, self.KAPPA, self.XI)
            da = delta_ask(v, q, self.Q_MAX, self.GAMMA, self.KAPPA, self.XI)
            assert db < da, f"q={q}: bid should be tighter (db={db:.4f}, da={da:.4f})"

    # ── spread ────────────────────────────────────────────────────────────────

    def test_spread_consistency_with_deltas(self, V):
        """ψ*(t,q) = δ_b + δ_a at all interior q."""
        v = V[200]
        for q in range(-self.Q_MAX + 1, self.Q_MAX):
            db = delta_bid(v, q, self.Q_MAX, self.GAMMA, self.KAPPA, self.XI)
            da = delta_ask(v, q, self.Q_MAX, self.GAMMA, self.KAPPA, self.XI)
            sp = spread(v, q, self.Q_MAX, self.GAMMA, self.KAPPA, self.XI)
            assert sp == pytest.approx(db + da, rel=1e-6), f"q={q}"

    def test_spread_positive(self, V):
        """Spread must be positive at all interior (q, t)."""
        for i in range(0, V.shape[0], 50):
            v = V[i]
            for q in range(-self.Q_MAX + 1, self.Q_MAX):
                sp = spread(v, q, self.Q_MAX, self.GAMMA, self.KAPPA, self.XI)
                assert sp > 0.0, f"Negative spread at q={q}, tau_idx={i}"

    def test_spread_boundary_raises(self, V):
        """spread() at |q|=Q_max should raise AssertionError."""
        v = V[200]
        with pytest.raises(AssertionError):
            spread(v, self.Q_MAX, self.Q_MAX, self.GAMMA, self.KAPPA, self.XI)
        with pytest.raises(AssertionError):
            spread(v, -self.Q_MAX, self.Q_MAX, self.GAMMA, self.KAPPA, self.XI)

    def test_spread_analytical_formula_at_q0(self, V):
        """ψ* = -(1/κ)·ln(v_{+1}·v_{-1}/v_0²) + ξ + (2/γ)·ln(1 + γ/κ)."""
        v = V[200]
        i = self.Q_MAX
        expected = (-(1/self.KAPPA) * np.log(v[i+1]*v[i-1]/v[i]**2)
                    + self.XI
                    + (2/self.GAMMA) * np.log(1 + self.GAMMA/self.KAPPA))
        result = spread(v, 0, self.Q_MAX, self.GAMMA, self.KAPPA, self.XI)
        assert result == pytest.approx(expected, rel=1e-8)


class TestGLFTBaseline:
    """Tests for the GLFTBaseline class interface and act() behaviour."""

    @pytest.fixture
    def glft(self):
        return GLFTBaseline(
            gamma=0.1, kappa=1.5, sigma=0.01, xi=0.01, A=1.0,
            T=390, Q_max=10, tick_size=0.01, adapt_sigma=False,
        )

    # ── Construction ──────────────────────────────────────────────────────────

    @pytest.mark.parametrize("param,val", [
        ("gamma", 0.0), ("gamma", -0.1),
        ("kappa", 0.0), ("kappa", -1.0),
        ("sigma", 0.0), ("sigma", -0.01),
        ("xi",   -0.01),
        ("A",     0.0), ("A",    -1.0),
        ("T",     0),
        ("Q_max", 0),   ("Q_max", -1),
        ("tick_size", 0.0),
    ])
    def test_invalid_params_raise(self, param, val):
        kwargs = dict(gamma=0.1, kappa=1.5, sigma=0.01, xi=0.01,
                      A=1.0, T=390, Q_max=10, tick_size=0.01)
        kwargs[param] = val
        with pytest.raises(AssertionError):
            GLFTBaseline(**kwargs)

    def test_xi_zero_valid(self):
        """ξ=0 (no market impact) is a valid special case."""
        b = GLFTBaseline(xi=0.0)
        assert b.xi == 0.0

    # ── act() interface ───────────────────────────────────────────────────────

    def test_act_shape(self, glft):
        glft.reset()
        action = glft.act(DUMMY_OBS, DUMMY_INFO)
        assert action.shape == (2,)

    def test_act_dtype(self, glft):
        glft.reset()
        action = glft.act(DUMMY_OBS, DUMMY_INFO)
        assert np.issubdtype(action.dtype, np.integer)

    def test_act_indices_in_range(self, glft):
        glft.reset()
        action = glft.act(DUMMY_OBS, DUMMY_INFO)
        assert 0 <= action[0] < N_OFFSET_LEVELS
        assert 0 <= action[1] < N_OFFSET_LEVELS

    def test_act_symmetric_at_zero_inventory(self, glft):
        """q=0: bid_idx == ask_idx (symmetric quotes)."""
        glft.reset()
        action = glft.act(DUMMY_OBS, _info(inventory=0))
        assert action[0] == action[1]

    def test_act_long_inventory_ask_closer(self, glft):
        """q>0: ask_idx < bid_idx (ask tighter)."""
        glft.reset()
        bid, ask = glft.compute_quotes(mid=1000.0, inventory=+5, tau=10.0)
        assert (ask - 1000.0) < (1000.0 - bid), \
            f"Long: ask_dist={ask-1000:.4f} should be < bid_dist={1000-bid:.4f}"

    def test_act_short_inventory_bid_closer(self, glft):
        """q<0: bid_idx < ask_idx (bid tighter)."""
        glft.reset()
        bid, ask = glft.compute_quotes(mid=1000.0, inventory=-5, tau=10.0)
        assert (1000.0 - bid) < (ask - 1000.0), \
            f"Short: bid_dist={1000-bid:.4f} should be < ask_dist={ask-1000:.4f}"
        
    def test_act_increments_step(self, glft):
        glft.reset()
        assert glft._t == 0
        glft.act(DUMMY_OBS, DUMMY_INFO)
        assert glft._t == 1

    def test_reset_clears_step(self, glft):
        glft.reset()
        for _ in range(20):
            glft.act(DUMMY_OBS, DUMMY_INFO)
        glft.reset()
        assert glft._t == 0

    def test_reset_restores_sigma(self):
        b = GLFTBaseline(sigma=0.01, adapt_sigma=True)
        b.reset()
        b.sigma = 0.99   # manually corrupt
        b.reset()
        assert b.sigma == pytest.approx(0.01)

    # ── compute_quotes ────────────────────────────────────────────────────────

    def test_compute_quotes_bid_below_mid(self, glft):
        bid, ask = glft.compute_quotes(1000.0, 0, 100.0)
        assert bid < 1000.0

    def test_compute_quotes_ask_above_mid(self, glft):
        bid, ask = glft.compute_quotes(1000.0, 0, 100.0)
        assert ask > 1000.0

    def test_compute_quotes_bid_below_ask(self, glft):
        bid, ask = glft.compute_quotes(1000.0, 0, 100.0)
        assert bid < ask

    def test_compute_quotes_at_max_inventory_no_bid(self, glft):
        """At q=Q_max: bid falls back to max offset (no more buying)."""
        bid, ask = glft.compute_quotes(1000.0, glft.Q_max, 100.0)
        # bid should be at max offset (4 ticks below mid)
        assert bid == pytest.approx(1000.0 - 4 * glft.tick_size)

    def test_compute_quotes_at_min_inventory_no_ask(self, glft):
        """At q=-Q_max: ask falls back to max offset (no more selling)."""
        bid, ask = glft.compute_quotes(1000.0, -glft.Q_max, 100.0)
        assert ask == pytest.approx(1000.0 + 4 * glft.tick_size)

    # ── repr ──────────────────────────────────────────────────────────────────

    def test_repr(self, glft):
        r = repr(glft)
        assert "GLFT" in r
        assert "gamma" in r
        assert "xi" in r


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-baseline interface consistency
# ═══════════════════════════════════════════════════════════════════════════════

class TestBaselineInterface:
    """All three baselines must share the same interface for eval loop compatibility."""

    @pytest.fixture(params=["fixed", "as", "glft"])
    def baseline(self, request):
        if request.param == "fixed":
            return FixedSpreadBaseline()
        elif request.param == "as":
            return AvellanedaStoikovBaseline(adapt_sigma=False)
        else:
            return GLFTBaseline(adapt_sigma=False)

    def test_has_reset(self, baseline):
        assert hasattr(baseline, "reset") and callable(baseline.reset)

    def test_has_act(self, baseline):
        assert hasattr(baseline, "act") and callable(baseline.act)

    def test_has_name(self, baseline):
        assert hasattr(baseline, "name")
        assert isinstance(baseline.name, str)
        assert len(baseline.name) > 0

    def test_reset_then_act_no_crash(self, baseline):
        baseline.reset()
        action = baseline.act(DUMMY_OBS, DUMMY_INFO)
        assert action is not None

    def test_act_output_shape(self, baseline):
        baseline.reset()
        action = baseline.act(DUMMY_OBS, DUMMY_INFO)
        assert action.shape == (2,)

    def test_act_output_in_action_space(self, baseline):
        baseline.reset()
        action = baseline.act(DUMMY_OBS, DUMMY_INFO)
        assert 0 <= action[0] < N_OFFSET_LEVELS
        assert 0 <= action[1] < N_OFFSET_LEVELS

    def test_multiple_resets(self, baseline):
        """Multiple resets should not crash or corrupt state."""
        for _ in range(3):
            baseline.reset()
            for _ in range(5):
                baseline.act(DUMMY_OBS, DUMMY_INFO)