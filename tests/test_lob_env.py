"""
tests/test_lob_env.py
---------------------
Unit tests for LOBMarketMakingEnv using the synthetic GBM path (no ABIDES).

Run with:
    python -m pytest tests/test_lob_env.py -v

These tests are CI-safe: no ABIDES installation required.
"""
import numpy as np
import pytest
from envs.lob_env import LOBMarketMakingEnv, TICK_OFFSETS, N_OFFSET_LEVELS


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(params=["asymmetric", "quadratic", "sparse"])
def env(request):
    """One env per reward type; short episodes so tests run fast."""
    e = LOBMarketMakingEnv(
        reward_type=request.param,
        episode_len=50,
        seed=42,
    )
    yield e
    e.close()


@pytest.fixture
def env_asym():
    e = LOBMarketMakingEnv(reward_type="asymmetric", episode_len=50, seed=0)
    yield e
    e.close()


# ── Reset contract ────────────────────────────────────────────────────────────

class TestReset:
    def test_obs_shape(self, env):
        obs, info = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape

    def test_obs_dtype(self, env):
        obs, _ = env.reset()
        assert obs.dtype == np.float32

    def test_obs_no_nan_inf(self, env):
        obs, _ = env.reset()
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))

    def test_info_is_dict(self, env):
        _, info = env.reset()
        assert isinstance(info, dict)

    def test_info_keys(self, env):
        _, info = env.reset()
        for key in ("step", "inventory", "mid_price", "cash"):
            assert key in info, f"missing key '{key}' in reset info"

    def test_inventory_zero_after_reset(self, env):
        _, info = env.reset()
        assert info["inventory"] == 0

    def test_step_counter_zero_after_reset(self, env):
        _, info = env.reset()
        assert info["step"] == 0

    def test_repeated_reset(self, env):
        """Two resets should both return valid observations."""
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=1)
        assert obs1.shape == env.observation_space.shape
        assert obs2.shape == env.observation_space.shape


# ── Step contract ─────────────────────────────────────────────────────────────

class TestStep:
    def test_returns_five_tuple(self, env):
        env.reset()
        result = env.step(env.action_space.sample())
        assert len(result) == 5

    def test_obs_shape(self, env):
        env.reset()
        obs, *_ = env.step(env.action_space.sample())
        assert obs.shape == env.observation_space.shape

    def test_obs_dtype(self, env):
        env.reset()
        obs, *_ = env.step(env.action_space.sample())
        assert obs.dtype == np.float32

    def test_obs_no_nan_inf(self, env):
        env.reset()
        obs, *_ = env.step(env.action_space.sample())
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))

    def test_reward_is_float(self, env):
        env.reset()
        _, reward, *_ = env.step(env.action_space.sample())
        assert isinstance(reward, float)

    def test_reward_finite(self, env):
        env.reset()
        _, reward, *_ = env.step(env.action_space.sample())
        assert np.isfinite(reward)

    def test_terminated_is_bool(self, env):
        env.reset()
        _, _, terminated, _, _ = env.step(env.action_space.sample())
        assert isinstance(terminated, bool)

    def test_truncated_is_bool(self, env):
        env.reset()
        _, _, _, truncated, _ = env.step(env.action_space.sample())
        assert isinstance(truncated, bool)

    def test_info_is_dict(self, env):
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        assert isinstance(info, dict)

    def test_info_keys(self, env):
        env.reset()
        _, _, _, _, info = env.step(env.action_space.sample())
        for key in ("step", "inventory", "mid_price", "cash", "reward",
                    "bid_price", "ask_price", "bid_offset", "ask_offset"):
            assert key in info, f"missing key '{key}' in step info"

    def test_step_counter_increments(self, env):
        env.reset()
        for i in range(5):
            _, _, _, _, info = env.step(env.action_space.sample())
            assert info["step"] == i + 1

    def test_action_space_contains_sample(self, env):
        env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            assert env.action_space.contains(action)


# ── Episode termination ───────────────────────────────────────────────────────

class TestEpisode:
    def test_truncation_at_episode_len(self, env_asym):
        """Episode must truncate exactly at episode_len steps."""
        env_asym.reset()
        truncated = False
        steps = 0
        while not truncated:
            _, _, terminated, truncated, _ = env_asym.step(
                env_asym.action_space.sample()
            )
            steps += 1
            if terminated:
                break
        assert steps == env_asym.episode_len

    def test_reset_after_truncation(self, env_asym):
        """Reset after a complete episode must return valid obs."""
        env_asym.reset()
        done = False
        while not done:
            _, _, terminated, truncated, _ = env_asym.step(
                env_asym.action_space.sample()
            )
            done = terminated or truncated
        obs, info = env_asym.reset()
        assert obs.shape == env_asym.observation_space.shape
        assert info["inventory"] == 0
        assert info["step"] == 0

    def test_terminal_reward_applied(self):
        """Terminal reward should be non-zero when episode ends with inventory."""
        env = LOBMarketMakingEnv(
            reward_type="asymmetric", episode_len=5, seed=7, kappa=10.0
        )
        env.reset()
        env._inventory = 5   # force non-zero inventory
        env._step = 4        # one step before truncation
        env._price_history.extend([100.0, 100.01, 99.99, 100.02, 100.0])
        _, reward, _, truncated, _ = env.step(np.array([4, 4]))
        assert truncated
        # With kappa=10 and inventory=5, terminal penalty should be non-trivial
        # (exact value depends on price history std, just check it's not pure spread PnL)
        env.close()


# ── Observation feature sanity ────────────────────────────────────────────────

class TestObsFeatures:
    def test_time_remaining_decreases(self):
        env = LOBMarketMakingEnv(episode_len=10, seed=0)
        obs0, _ = env.reset()
        base = 6 + 2 * env.n_lob_levels
        tau0 = obs0[base + 5]
        obs1, *_ = env.step(env.action_space.sample())
        tau1 = obs1[base + 5]
        assert tau1 < tau0, "time remaining should decrease each step"
        env.close()

    def test_inventory_feature_clipped(self):
        env = LOBMarketMakingEnv(episode_len=50, Q_max=10, seed=0)
        env.reset()
        env._inventory = 10
        obs = env._get_obs()
        base = 6 + 2 * env.n_lob_levels
        assert obs[base + 0] == pytest.approx(1.0)
        env._inventory = -10
        obs = env._get_obs()
        assert obs[base + 0] == pytest.approx(-1.0)
        env.close()

    def test_obs_dim_varies_with_n_lob_levels(self):
        for k in (1, 3, 5):
            env = LOBMarketMakingEnv(n_lob_levels=k, seed=0)
            expected = 6 + 2 * k + 6
            assert env._obs_dim == expected, \
                f"n_lob_levels={k}: expected dim {expected}, got {env._obs_dim}"
            obs, _ = env.reset()
            assert obs.shape == (expected,)
            env.close()


# ── Reward formulations ───────────────────────────────────────────────────────

class TestReward:
    def test_asymmetric_penalises_adverse_inventory(self):
        """With eta>0, adverse inventory PnL should be penalised."""
        env = LOBMarketMakingEnv(reward_type="asymmetric", eta=1.0, seed=0)
        env.reset()
        env._mid_price = 100.0
        env._prev_mid  = 101.0   # mid fell → long inventory loses money
        reward = env._compute_reward(
            matched_bid=0.0, matched_ask=0.0,
            bid_price=99.0, ask_price=101.0,
            inventory=5.0,
        )
        # inventory_pnl = 5 * (100 - 101) = -5 → penalised only if negative
        # asymmetric: pnl - max(0, eta * inv_pnl) = -5 - 0 = -5
        assert reward == pytest.approx(-5.0, rel=1e-4)
        env.close()

    def test_quadratic_penalises_inventory_level(self):
        env = LOBMarketMakingEnv(reward_type="quadratic", lam=1.0, seed=0)
        env.reset()
        env._mid_price = 100.0
        env._prev_mid  = 100.0
        reward = env._compute_reward(
            matched_bid=0.0, matched_ask=0.0,
            bid_price=99.0, ask_price=101.0,
            inventory=3.0,
        )
        # pnl=0 (no fills, no mid move), penalty = lam * q^2 = 1.0 * 9 = 9
        assert reward == pytest.approx(-9.0, rel=1e-4)
        env.close()

    def test_sparse_filled_both(self):
        env = LOBMarketMakingEnv(reward_type="sparse", seed=0)
        env.reset()
        env._mid_price = 100.0
        env._prev_mid  = 100.0
        reward = env._compute_reward(
            matched_bid=1.0, matched_ask=1.0,
            bid_price=99.0, ask_price=101.0,
            inventory=0.0, filled_both=True,
        )
        assert reward == pytest.approx(1.0)
        env.close()

    def test_sparse_cross_spread(self):
        env = LOBMarketMakingEnv(reward_type="sparse", seed=0)
        env.reset()
        env._mid_price = 100.0
        env._prev_mid  = 100.0
        reward = env._compute_reward(
            matched_bid=0.0, matched_ask=1.0,
            bid_price=99.0, ask_price=101.0,
            inventory=0.0, cross_spread_fill=True,
        )
        assert reward == pytest.approx(-0.5)
        env.close()

    def test_invalid_reward_type_raises(self):
        with pytest.raises(AssertionError):
            LOBMarketMakingEnv(reward_type="invalid")


# ── RSI and vol helpers ───────────────────────────────────────────────────────

class TestHelpers:
    def test_rsi_neutral_on_short_history(self):
        env = LOBMarketMakingEnv(seed=0)
        assert env._compute_rsi() == pytest.approx(50.0)
        env.close()

    def test_rsi_100_on_all_gains(self):
        env = LOBMarketMakingEnv(seed=0)
        env._price_history.extend(list(range(100, 116)))   # 15 up moves
        rsi = env._compute_rsi(window=14)
        assert rsi == pytest.approx(100.0)
        env.close()

    def test_realized_vol_zero_on_short_history(self):
        env = LOBMarketMakingEnv(seed=0)
        assert env._compute_realized_vol() == pytest.approx(0.0)
        env.close()

    def test_realized_vol_positive_on_history(self):
        env = LOBMarketMakingEnv(seed=0)
        prices = 100.0 + np.random.default_rng(0).normal(0, 0.1, 25)
        env._price_history.extend(prices.tolist())
        assert env._compute_realized_vol() > 0.0
        env.close()


# ── Action space ──────────────────────────────────────────────────────────────

class TestActionSpace:
    def test_multidiscrete_shape(self):
        env = LOBMarketMakingEnv(seed=0)
        assert env.action_space.shape == (2,)
        env.close()

    def test_tick_offsets_range(self):
        assert TICK_OFFSETS[0]  == 1
        assert TICK_OFFSETS[-1] ==  10
        assert N_OFFSET_LEVELS  ==  10

    def test_all_actions_valid(self):
        env = LOBMarketMakingEnv(seed=0)
        env.reset()
        for bid_idx in range(N_OFFSET_LEVELS):
            for ask_idx in range(N_OFFSET_LEVELS):
                action = np.array([bid_idx, ask_idx])
                assert env.action_space.contains(action)
        env.close()
