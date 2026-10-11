"""
tests/test_rollout.py

Unit tests for training/rollout.py::get_encoder_input().

This function had no direct test coverage before the private-feature bug
was found: the CNN and autoencoder encoders were silently given only the
raw LOB depth snapshot, never the agent's inventory, quote distances, or
time remaining (obs[12:18]). These tests pin the fixed behaviour so that
regression can't happen again.

Run with:
    pytest tests/test_rollout.py -v
"""

import numpy as np
import pytest

from encoders.base import PRIVATE_FEATURE_DIM
from training.rollout import get_encoder_input

N_LEVELS = 10
PRIVATE_SLICE = slice(12, 18)   # envs/lob_env.py::_get_obs()'s private features


def _make_obs(private_values=None):
    """An 18-dim handcrafted-style obs vector with distinguishable values,
    so misplacement (wrong slice, wrong order) would be caught."""
    obs = np.arange(18, dtype=np.float32)
    if private_values is not None:
        obs[PRIVATE_SLICE] = private_values
    return obs


def _make_info(n_bid=N_LEVELS, n_ask=N_LEVELS, bid_val=1.0, ask_val=2.0):
    return {"lob_snapshot": {
        "bid_sizes": [bid_val] * n_bid,
        "ask_sizes": [ask_val] * n_ask,
    }}


class TestHandcraftedPassthrough:
    def test_returns_obs_unchanged(self):
        obs = _make_obs()
        out = get_encoder_input(obs, _make_info(), "handcrafted")
        assert np.array_equal(out, obs)
        assert out.shape == (18,)


class TestCNNEncoderInput:
    def test_shape_is_market_plus_private(self):
        obs = _make_obs()
        out = get_encoder_input(obs, _make_info(), "cnn")
        assert out.shape == (2 * N_LEVELS + PRIVATE_FEATURE_DIM,)

    def test_private_tail_matches_obs_slice(self):
        """Regression: the agent must receive its own inventory, quote
        distances, and time remaining — not just the raw order book."""
        private_vals = [0.5, 0.1, 0.2, 0.3, 0.4, 0.9]
        obs = _make_obs(private_vals)
        out = get_encoder_input(obs, _make_info(), "cnn")
        np.testing.assert_allclose(out[-PRIVATE_FEATURE_DIM:], private_vals)

    def test_market_part_is_ask_then_bid_raw(self):
        obs = _make_obs()
        out = get_encoder_input(obs, _make_info(bid_val=3.0, ask_val=7.0), "cnn")
        assert np.allclose(out[:N_LEVELS], 7.0)             # ask first, raw (not normalised)
        assert np.allclose(out[N_LEVELS:2 * N_LEVELS], 3.0)  # then bid, raw

    def test_pads_thin_book(self):
        """A book thinner than N_LEVELS on one side is zero-padded, not truncated away."""
        obs = _make_obs()
        out = get_encoder_input(obs, _make_info(n_bid=3, n_ask=3), "cnn")
        assert out.shape == (2 * N_LEVELS + PRIVATE_FEATURE_DIM,)
        assert np.allclose(out[3:N_LEVELS], 0.0)              # ask padding
        assert np.allclose(out[N_LEVELS + 3:2 * N_LEVELS], 0.0)  # bid padding


class TestAEEncoderInput:
    def test_shape_is_market_plus_private(self):
        obs = _make_obs()
        out = get_encoder_input(obs, _make_info(), "autoencoder")
        assert out.shape == (2 * N_LEVELS + PRIVATE_FEATURE_DIM,)

    def test_private_tail_matches_obs_slice_and_is_not_renormalised(self):
        """The AE branch normalises the market side sums to 1.0 — it must
        NOT also rescale the private tail (inventory etc. are not a
        probability distribution)."""
        private_vals = [-1.0, 0.25, 0.75, -0.5, 0.5, 1.0]
        obs = _make_obs(private_vals)
        out = get_encoder_input(obs, _make_info(), "autoencoder")
        np.testing.assert_allclose(out[-PRIVATE_FEATURE_DIM:], private_vals)

    def test_market_side_sums_to_one(self):
        obs = _make_obs()
        out = get_encoder_input(obs, _make_info(bid_val=3.0, ask_val=7.0), "autoencoder")
        ask, bid = out[:N_LEVELS], out[N_LEVELS:2 * N_LEVELS]
        assert ask.sum() == pytest.approx(1.0, rel=1e-5)
        assert bid.sum() == pytest.approx(1.0, rel=1e-5)

    def test_empty_side_yields_zeros_not_nan(self):
        obs = _make_obs()
        out = get_encoder_input(obs, _make_info(n_bid=0, n_ask=0, bid_val=0.0, ask_val=0.0),
                                 "autoencoder")
        assert not np.any(np.isnan(out))
        assert np.allclose(out[: 2 * N_LEVELS], 0.0)
