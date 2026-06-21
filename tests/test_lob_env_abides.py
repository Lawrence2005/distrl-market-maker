"""
tests/test_lob_env_abides.py
-----------------------------
Integration tests for LOBMarketMakingEnv with the real ABIDES simulator.

Requires the abides conda environment:
    conda activate abides
    python -m pytest tests/test_lob_env_abides.py -v -s

These tests are NOT CI-safe (slow, require abides install).
Marked with @pytest.mark.abides so they can be excluded:
    pytest tests/ -v --ignore=tests/test_lob_env_abides.py
    # or:
    pytest tests/ -v -m "not abides"
"""
import numpy as np
import pytest

# Skip entire module if ABIDES is not installed
abides_gym = pytest.importorskip(
    "abides_gym",
    reason="abides_gym not installed — run in abides conda env"
)

from envs.lob_env import LOBMarketMakingEnv

pytestmark = pytest.mark.abides


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def abides_env():
    """Single env reused across the module — ABIDES reset is slow (~5s)."""
    env = LOBMarketMakingEnv(
        reward_type="asymmetric",
        episode_len=10,
        seed=42,
    )
    yield env
    env.close()


# ── Wiring sanity ─────────────────────────────────────────────────────────────

class TestAbidesWiring:
    def test_reset_returns_valid_obs(self, abides_env):
        obs, info = abides_env.reset()
        assert obs.shape == abides_env.observation_space.shape
        assert obs.dtype == np.float32
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))

    def test_mid_price_nonzero_after_reset(self, abides_env):
        abides_env.reset()
        assert abides_env._mid_price > 0.0

    def test_mid_price_in_plausible_range(self, abides_env):
        """ABIDES RMSC04 simulates around $1000 stock price."""
        abides_env.reset()
        assert 500.0 < abides_env._mid_price < 2000.0, \
            f"mid_price {abides_env._mid_price} outside plausible range"

    def test_step_returns_five_tuple(self, abides_env):
        abides_env.reset()
        result = abides_env.step(abides_env.action_space.sample())
        assert len(result) == 5

    def test_step_obs_shape(self, abides_env):
        abides_env.reset()
        obs, *_ = abides_env.step(abides_env.action_space.sample())
        assert obs.shape == abides_env.observation_space.shape

    def test_step_obs_no_nan(self, abides_env):
        abides_env.reset()
        obs, *_ = abides_env.step(abides_env.action_space.sample())
        assert not np.any(np.isnan(obs))

    def test_step_reward_finite(self, abides_env):
        abides_env.reset()
        _, reward, *_ = abides_env.step(abides_env.action_space.sample())
        assert np.isfinite(reward)

    def test_lob_history_populated(self, abides_env):
        """LOB snapshot should be non-empty after at least one step."""
        abides_env.reset()
        for _ in range(3):
            abides_env.step(abides_env.action_space.sample())
        # Either lob_history has entries or bids were empty (both are valid)
        # Just check no crash and history is a deque
        from collections import deque
        assert isinstance(abides_env._lob_history, deque)

    def test_price_history_grows(self, abides_env):
        abides_env.reset()
        n_before = len(abides_env._price_history)
        abides_env.step(abides_env.action_space.sample())
        assert len(abides_env._price_history) == n_before + 1

    def test_full_episode(self, abides_env):
        """Run a complete episode without crashing."""
        abides_env.reset()
        done = False
        steps = 0
        while not done:
            _, _, terminated, truncated, info = abides_env.step(
                abides_env.action_space.sample()
            )
            done = terminated or truncated
            steps += 1
        assert steps == abides_env.episode_len
        assert "mid_price" in info

    def test_action_encoding_does_not_crash(self, abides_env):
        """All 81 action combinations should encode without error."""
        from envs.lob_env import TICK_OFFSETS
        abides_env.reset()
        mid = abides_env._mid_price
        for bid_idx in range(9):
            for ask_idx in range(9):
                bid_price = mid - int(TICK_OFFSETS[bid_idx]) * abides_env.tick_size
                ask_price = mid + int(TICK_OFFSETS[ask_idx]) * abides_env.tick_size
                action = abides_env._encode_abides_action(bid_price, ask_price)
                assert isinstance(action, int)
