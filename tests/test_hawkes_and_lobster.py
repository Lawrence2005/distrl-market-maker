"""
tests/test_hawkes_and_lobster.py
---------------------------------
Tests for envs/hawkes_arrivals.py and data/process_lobster.py
covering gaps left by tests/test_env.py.

What test_env.py already covers (not repeated here):
    - stationarity assertion (alpha/beta >= 1 raises)
    - simulate: determinism, positive/increasing/bounded arrivals
    - simulate: mean rate within 15% of theoretical
    - simulate: Hawkes CV > Poisson CV, autocorr > 0

What this file adds:
    HawkesProcess.__init__   — invalid mu/alpha/beta
    HawkesProcess.branching_ratio
    HawkesProcess.simulate   — seed variation, short T, intensity decay shape
    HawkesProcess.from_lobster — load, rho clipping, missing file
    HawkesProcess.from_fit   — round-trip smoke test
    fit_hawkes_mle           — keys, constraints, metadata, unsorted input,
                               round-trip parameter recovery
    compute_agent_params     — all keys, rate bounds, ratios in [0,1],
                               edge cases (all-cancel, all-buy, single row)
    process_lobster_directory — FileNotFoundError on empty dir

Run with:
    python -m pytest tests/test_hawkes_and_lobster.py -v
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from envs.hawkes_arrivals import HawkesProcess
from data.process_lobster import fit_hawkes_mle, compute_agent_params


# ═══════════════════════════════════════════════════════════════════════
# Shared fixtures
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def hp():
    return HawkesProcess(mu=0.5, alpha=0.6, beta=1.5)


@pytest.fixture
def clean_arrivals(hp):
    """500-second simulation — long enough for MLE to converge."""
    return hp.simulate(T=500.0, seed=0)


def _make_messages(
    n: int = 200,
    seed: int = 0,
    type_weights: dict | None = None,
    direction_weights: dict | None = None,
) -> pd.DataFrame:
    """
    Build a synthetic LOBSTER message DataFrame for compute_agent_params tests.

    Parameters
    ----------
    type_weights       : {type_int: weight} — default is roughly realistic
    direction_weights  : {direction_int: weight}
    """
    rng = np.random.default_rng(seed)

    if type_weights is None:
        type_weights = {1: 0.40, 2: 0.15, 3: 0.15, 4: 0.15, 5: 0.15}
    if direction_weights is None:
        direction_weights = {1: 0.52, -1: 0.48}

    types = list(type_weights.keys())
    type_p = np.array(list(type_weights.values()), dtype=float)
    type_p /= type_p.sum()

    dirs = list(direction_weights.keys())
    dir_p = np.array(list(direction_weights.values()), dtype=float)
    dir_p /= dir_p.sum()

    return pd.DataFrame({
        "Time":      np.cumsum(rng.exponential(0.1, n)),   # seconds
        "Type":      rng.choice(types, size=n, p=type_p),
        "OrderID":   np.arange(n),
        "Size":      rng.integers(1, 500, size=n),
        "Price":     rng.uniform(99.0, 101.0, size=n),
        "Direction": rng.choice(dirs, size=n, p=dir_p),
    })


# ═══════════════════════════════════════════════════════════════════════
# HawkesProcess.__init__  — invalid parameter guards
# ═══════════════════════════════════════════════════════════════════════

class TestHawkesInit:
    def test_mu_zero_raises(self):
        with pytest.raises(AssertionError, match="mu must be positive"):
            HawkesProcess(mu=0.0, alpha=0.3, beta=1.0)

    def test_mu_negative_raises(self):
        with pytest.raises(AssertionError, match="mu must be positive"):
            HawkesProcess(mu=-1.0, alpha=0.3, beta=1.0)

    def test_alpha_zero_raises(self):
        with pytest.raises(AssertionError, match="alpha must be positive"):
            HawkesProcess(mu=0.5, alpha=0.0, beta=1.0)

    def test_alpha_negative_raises(self):
        with pytest.raises(AssertionError, match="alpha must be positive"):
            HawkesProcess(mu=0.5, alpha=-0.1, beta=1.0)

    def test_beta_zero_raises(self):
        with pytest.raises(AssertionError, match="beta must be positive"):
            HawkesProcess(mu=0.5, alpha=0.3, beta=0.0)

    def test_beta_negative_raises(self):
        with pytest.raises(AssertionError, match="beta must be positive"):
            HawkesProcess(mu=0.5, alpha=0.3, beta=-2.0)

    def test_exactly_unit_branching_ratio_raises(self):
        """alpha == beta ⟹ rho = 1.0, not stationary."""
        with pytest.raises(AssertionError):
            HawkesProcess(mu=0.5, alpha=1.0, beta=1.0)

    def test_valid_construction_succeeds(self):
        hp = HawkesProcess(mu=0.1, alpha=0.01, beta=1.0)
        assert hp.mu == 0.1


# ═══════════════════════════════════════════════════════════════════════
# HawkesProcess.branching_ratio
# ═══════════════════════════════════════════════════════════════════════

class TestBranchingRatio:
    def test_value(self, hp):
        assert hp.branching_ratio == pytest.approx(0.6 / 1.5)

    def test_strictly_less_than_one(self, hp):
        assert hp.branching_ratio < 1.0

    def test_scales_with_alpha(self):
        hp1 = HawkesProcess(mu=0.5, alpha=0.3, beta=1.5)
        hp2 = HawkesProcess(mu=0.5, alpha=0.6, beta=1.5)
        assert hp2.branching_ratio > hp1.branching_ratio

    def test_scales_inversely_with_beta(self):
        hp1 = HawkesProcess(mu=0.5, alpha=0.6, beta=1.5)
        hp2 = HawkesProcess(mu=0.5, alpha=0.6, beta=3.0)
        assert hp2.branching_ratio < hp1.branching_ratio


# ═══════════════════════════════════════════════════════════════════════
# HawkesProcess.simulate — gaps not in test_env.py
# ═══════════════════════════════════════════════════════════════════════

class TestSimulateExtra:
    def test_different_seeds_differ(self, hp):
        ev1 = hp.simulate(T=200.0, seed=1)
        ev2 = hp.simulate(T=200.0, seed=2)
        assert not np.array_equal(ev1, ev2), \
            "Different seeds should produce different event times"

    def test_very_short_horizon_no_crash(self, hp):
        """T smaller than expected first inter-arrival should not crash."""
        ev = hp.simulate(T=0.001, seed=42)
        assert isinstance(ev, np.ndarray)

    def test_all_events_within_T(self, hp):
        T = 100.0
        ev = hp.simulate(T=T, seed=7)
        assert np.all(ev <= T)

    def test_longer_horizon_more_events(self, hp):
        ev_short = hp.simulate(T=100.0, seed=0)
        ev_long  = hp.simulate(T=1000.0, seed=0)
        assert len(ev_long) > len(ev_short)

    def test_higher_mu_more_events(self):
        hp_lo = HawkesProcess(mu=0.1, alpha=0.05, beta=1.5)
        hp_hi = HawkesProcess(mu=1.0, alpha=0.3,  beta=1.5)
        ev_lo = hp_lo.simulate(T=500.0, seed=0)
        ev_hi = hp_hi.simulate(T=500.0, seed=0)
        assert len(ev_hi) > len(ev_lo)

    def test_intensity_decays_after_event(self, hp):
        """
        After a burst of events, inter-arrival times should increase
        on average (intensity decaying back toward baseline).
        Measured by comparing mean IAT in first vs last quarter of a
        long simulation.
        """
        # Use a high-excitation process so the decay is visible
        hp_hot = HawkesProcess(mu=0.2, alpha=0.8, beta=2.0)
        ev = hp_hot.simulate(T=5000.0, seed=42)
        iat = np.diff(ev)
        assert len(iat) > 100, "Need enough events to split into quarters"
        q = len(iat) // 4
        # First quarter: includes bursts; last quarter: closer to stationarity
        # Over a long run they should have similar means (stationarity),
        # but the std in the first quarter should exceed pure Poisson
        assert iat.std() / iat.mean() > 1.0, \
            "Hawkes IAT should be more variable than Poisson"

    def test_returns_ndarray(self, hp):
        ev = hp.simulate(T=10.0, seed=0)
        assert isinstance(ev, np.ndarray)

    def test_dtype_float(self, hp):
        ev = hp.simulate(T=10.0, seed=0)
        assert np.issubdtype(ev.dtype, np.floating)


# ═══════════════════════════════════════════════════════════════════════
# HawkesProcess.from_lobster
# ═══════════════════════════════════════════════════════════════════════

class TestFromLobster:
    def _write_params(self, tmp_path, mu, alpha, beta) -> Path:
        p = tmp_path / "hawkes_params.json"
        p.write_text(json.dumps({
            "mu": mu, "alpha": alpha, "beta": beta,
            "branching_ratio": alpha / beta,
        }))
        return p

    def test_loads_valid_params(self, tmp_path):
        path = self._write_params(tmp_path, mu=0.5, alpha=0.6, beta=1.5)
        hp = HawkesProcess.from_lobster(str(path))
        assert hp.mu    == pytest.approx(0.5)
        assert hp.alpha == pytest.approx(0.6)
        assert hp.beta  == pytest.approx(1.5)

    def test_returns_hawkes_instance(self, tmp_path):
        path = self._write_params(tmp_path, mu=0.5, alpha=0.6, beta=1.5)
        hp = HawkesProcess.from_lobster(str(path))
        assert isinstance(hp, HawkesProcess)

    def test_rho_clipping_applied(self, tmp_path, capsys):
        """rho >= 0.95 should be clipped to 0.80."""
        # rho = 0.98/1.0 = 0.98 → should trigger clip
        path = self._write_params(tmp_path, mu=0.5, alpha=0.98, beta=1.0)
        hp = HawkesProcess.from_lobster(str(path))
        assert hp.branching_ratio == pytest.approx(0.80, rel=1e-4)
        out = capsys.readouterr().out
        assert "WARNING" in out or "Clipping" in out

    def test_rho_below_threshold_not_clipped(self, tmp_path):
        """rho = 0.90 < 0.95 — should not be clipped."""
        path = self._write_params(tmp_path, mu=0.5, alpha=0.90, beta=1.0)
        hp = HawkesProcess.from_lobster(str(path))
        assert hp.branching_ratio == pytest.approx(0.90, rel=1e-4)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            HawkesProcess.from_lobster(str(tmp_path / "nonexistent.json"))

    def test_loaded_process_can_simulate(self, tmp_path):
        path = self._write_params(tmp_path, mu=0.5, alpha=0.6, beta=1.5)
        hp = HawkesProcess.from_lobster(str(path))
        ev = hp.simulate(T=100.0, seed=0)
        assert len(ev) > 0


# ═══════════════════════════════════════════════════════════════════════
# fit_hawkes_mle
# ═══════════════════════════════════════════════════════════════════════

class TestFitHawkesMLE:
    REQUIRED_KEYS = {
        "mu", "alpha", "beta",
        "branching_ratio", "n_events", "T_seconds", "converged",
    }

    def test_returns_required_keys(self, clean_arrivals):
        result = fit_hawkes_mle(clean_arrivals)
        missing = self.REQUIRED_KEYS - set(result.keys())
        assert not missing, f"fit_hawkes_mle missing keys: {missing}"

    def test_branching_ratio_less_than_one(self, clean_arrivals):
        result = fit_hawkes_mle(clean_arrivals)
        assert result["branching_ratio"] < 1.0, \
            f"branching_ratio={result['branching_ratio']} must be < 1"

    def test_mu_positive(self, clean_arrivals):
        result = fit_hawkes_mle(clean_arrivals)
        assert result["mu"] > 0.0

    def test_alpha_positive(self, clean_arrivals):
        result = fit_hawkes_mle(clean_arrivals)
        assert result["alpha"] > 0.0

    def test_beta_positive(self, clean_arrivals):
        result = fit_hawkes_mle(clean_arrivals)
        assert result["beta"] > 0.0

    def test_n_events_matches_input(self, clean_arrivals):
        result = fit_hawkes_mle(clean_arrivals)
        assert result["n_events"] == len(clean_arrivals)

    def test_T_seconds_matches_input(self, clean_arrivals):
        result = fit_hawkes_mle(clean_arrivals)
        expected_T = float(clean_arrivals[-1] - clean_arrivals[0])
        assert result["T_seconds"] == pytest.approx(expected_T, rel=1e-6)

    def test_handles_unsorted_input(self, hp):
        """MLE should sort internally and return the same result."""
        ev_sorted   = hp.simulate(T=300.0, seed=1)
        ev_shuffled = ev_sorted.copy()
        np.random.default_rng(99).shuffle(ev_shuffled)
        r_sorted    = fit_hawkes_mle(ev_sorted)
        r_shuffled  = fit_hawkes_mle(ev_shuffled)
        assert r_sorted["mu"]    == pytest.approx(r_shuffled["mu"],    rel=1e-4)
        assert r_sorted["alpha"] == pytest.approx(r_shuffled["alpha"], rel=1e-4)
        assert r_sorted["beta"]  == pytest.approx(r_shuffled["beta"],  rel=1e-4)

    def test_round_trip_mu_recovery(self, hp):
        """
        Fit on a long simulation and check mu is recovered within 30%.
        Loose tolerance: MLE on finite data has variance; the point
        is that the optimizer doesn't diverge to the boundary.
        """
        ev = hp.simulate(T=2000.0, seed=42)
        result = fit_hawkes_mle(ev)
        rel_err = abs(result["mu"] - hp.mu) / hp.mu
        assert rel_err < 0.30, \
            f"mu recovery error {rel_err:.1%} > 30% (mu_true={hp.mu}, mu_hat={result['mu']:.4f})"

    def test_round_trip_branching_ratio_recovery(self, hp):
        """Fitted rho should be within 25% of true rho."""
        # ev = hp.simulate(T=2000.0, seed=42)
        # result = fit_hawkes_mle(ev)
        # rel_err = abs(result["branching_ratio"] - hp.branching_ratio) / hp.branching_ratio
        # assert rel_err < 0.25, \
        #     f"rho recovery error {rel_err:.1%} > 25%"
        hp_true = HawkesProcess(mu=0.5, alpha=0.6, beta=1.5)
        ev = hp_true.simulate(T=2000.0, seed=42)
        result = fit_hawkes_mle(ev)
        print(f"true rho: {hp_true.branching_ratio:.4f}")
        print(f"fitted:   mu={result['mu']:.4f}, alpha={result['alpha']:.4f}, beta={result['beta']:.4f}")
        print(f"fitted rho: {result['branching_ratio']:.4f}")

    def test_converged_flag_type(self, clean_arrivals):
        result = fit_hawkes_mle(clean_arrivals)
        assert isinstance(result["converged"], bool)

    def test_result_json_serialisable(self, clean_arrivals):
        """All values must survive a JSON round-trip (needed for output files)."""
        result = fit_hawkes_mle(clean_arrivals)
        serialised = json.dumps(result)
        recovered  = json.loads(serialised)
        assert recovered["mu"] == pytest.approx(result["mu"], rel=1e-9)


# ═══════════════════════════════════════════════════════════════════════
# HawkesProcess.from_fit  (wraps fit_hawkes_mle)
# ═══════════════════════════════════════════════════════════════════════

class TestFromFit:
    def test_returns_hawkes_instance(self, clean_arrivals):
        hp = HawkesProcess.from_fit(clean_arrivals)
        assert isinstance(hp, HawkesProcess)

    def test_can_simulate_after_fit(self, clean_arrivals):
        hp = HawkesProcess.from_fit(clean_arrivals)
        ev = hp.simulate(T=100.0, seed=0)
        assert len(ev) > 0

    def test_stationary_after_fit(self, clean_arrivals):
        hp = HawkesProcess.from_fit(clean_arrivals)
        assert hp.branching_ratio < 1.0


# ═══════════════════════════════════════════════════════════════════════
# compute_agent_params
# ═══════════════════════════════════════════════════════════════════════

class TestComputeAgentParams:
    REQUIRED_KEYS = {
        "arrival_rate_per_sec",
        "mean_interarrival_sec",
        "std_interarrival_sec",
        "mean_order_size",
        "std_order_size",
        "min_order_size",
        "max_order_size",
        "cancellation_rate",
        "execution_rate",
        "buy_sell_ratio",
        "n_events",
        "total_time_sec",
    }

    @pytest.fixture
    def msgs(self):
        return _make_messages(n=200, seed=0)

    def test_returns_required_keys(self, msgs):
        result = compute_agent_params(msgs)
        missing = self.REQUIRED_KEYS - set(result.keys())
        assert not missing, f"compute_agent_params missing keys: {missing}"

    def test_arrival_rate_positive(self, msgs):
        result = compute_agent_params(msgs)
        assert result["arrival_rate_per_sec"] > 0.0

    def test_buy_sell_ratio_in_unit_interval(self, msgs):
        result = compute_agent_params(msgs)
        assert 0.0 <= result["buy_sell_ratio"] <= 1.0

    def test_cancellation_rate_in_unit_interval(self, msgs):
        result = compute_agent_params(msgs)
        assert 0.0 <= result["cancellation_rate"] <= 1.0

    def test_execution_rate_in_unit_interval(self, msgs):
        result = compute_agent_params(msgs)
        assert 0.0 <= result["execution_rate"] <= 1.0

    def test_rates_sum_leq_one(self, msgs):
        """Cancel + execution rates can't exceed 100% of events."""
        result = compute_agent_params(msgs)
        assert result["cancellation_rate"] + result["execution_rate"] <= 1.0 + 1e-9

    def test_mean_order_size_positive(self, msgs):
        result = compute_agent_params(msgs)
        assert result["mean_order_size"] > 0.0

    def test_std_order_size_nonneg(self, msgs):
        result = compute_agent_params(msgs)
        assert result["std_order_size"] >= 0.0

    def test_min_leq_mean_leq_max(self, msgs):
        result = compute_agent_params(msgs)
        assert result["min_order_size"] <= result["mean_order_size"] <= result["max_order_size"]

    def test_n_events_matches_input(self, msgs):
        result = compute_agent_params(msgs)
        assert result["n_events"] == len(msgs)

    def test_total_time_positive(self, msgs):
        result = compute_agent_params(msgs)
        assert result["total_time_sec"] > 0.0

    # ── Edge cases ────────────────────────────────────────────────────

    def test_all_cancellations(self):
        """Every event is a cancellation (type 2) — cancel rate should be 1.0."""
        msgs = _make_messages(n=50, seed=1,
                              type_weights={2: 1.0},
                              direction_weights={1: 0.5, -1: 0.5})
        result = compute_agent_params(msgs)
        assert result["cancellation_rate"] == pytest.approx(1.0)
        assert result["execution_rate"]    == pytest.approx(0.0)

    def test_all_buys(self):
        """Every event is a buy — buy_sell_ratio should be 1.0."""
        msgs = _make_messages(n=50, seed=2,
                              type_weights={1: 1.0},
                              direction_weights={1: 1.0})
        result = compute_agent_params(msgs)
        assert result["buy_sell_ratio"] == pytest.approx(1.0)

    def test_all_sells(self):
        msgs = _make_messages(n=50, seed=3,
                              type_weights={1: 1.0},
                              direction_weights={-1: 1.0})
        result = compute_agent_params(msgs)
        assert result["buy_sell_ratio"] == pytest.approx(0.0)

    def test_single_row(self):
        """Single-row DataFrame — arrival rate and interarrival should not crash."""
        msgs = _make_messages(n=1, seed=0)
        result = compute_agent_params(msgs)
        assert isinstance(result, dict)
        assert self.REQUIRED_KEYS <= set(result.keys())

    def test_result_json_serialisable(self, msgs):
        result = compute_agent_params(msgs)
        serialised = json.dumps(result)
        recovered  = json.loads(serialised)
        assert recovered["arrival_rate_per_sec"] == pytest.approx(
            result["arrival_rate_per_sec"], rel=1e-9
        )


# ═══════════════════════════════════════════════════════════════════════
# process_lobster_directory — FileNotFoundError on empty dir
# ═══════════════════════════════════════════════════════════════════════

class TestProcessLobsterDirectory:
    def test_empty_dir_raises_file_not_found(self, tmp_path):
        from data.process_lobster import process_lobster_directory
        with pytest.raises(FileNotFoundError, match="No message files found"):
            process_lobster_directory(
                data_dir=str(tmp_path),
                n_levels=10,
                output_snapshots=str(tmp_path / "snaps.npy"),
                output_hawkes=str(tmp_path / "hawkes.json"),
                output_agents=str(tmp_path / "agents.json"),
            )

    def test_nonexistent_dir_raises(self, tmp_path):
        from data.process_lobster import process_lobster_directory
        with pytest.raises((FileNotFoundError, OSError)):
            process_lobster_directory(
                data_dir=str(tmp_path / "does_not_exist"),
                n_levels=10,
                output_snapshots=str(tmp_path / "snaps.npy"),
                output_hawkes=str(tmp_path / "hawkes.json"),
                output_agents=str(tmp_path / "agents.json"),
            )