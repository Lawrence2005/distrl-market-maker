"""
archive/lobster_calibration/test_compute_agent_params.py

Archived alongside compute_agent_params.py — see that file's module
docstring for why this pipeline was never wired into the live simulator.

Run with:
    python -m pytest archive/lobster_calibration/test_compute_agent_params.py -v
"""

import json

import numpy as np
import pandas as pd
import pytest

from archive.lobster_calibration.compute_agent_params import compute_agent_params


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
