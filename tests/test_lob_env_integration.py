# tests/test_env.py
"""
tests/test_env.py

Unit tests for the ABIDES-Gym LOB environment wrapper.

Three tests (specs from docs/mdp_formulation.md and Week 2 research plan):
    1. test_step_output_shape    — obs, reward, done, info correct shapes/types
    2. test_reward_finite        — no NaN or Inf in reward across 100 random steps
    3. test_inventory_constraint — |q| never exceeds Q_max=10

Run with:
    python -m pytest tests/test_lob_env_integration.py -v

Week 2 deliverable.
"""

import pytest
import numpy as np


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def env():
    """
    Return a fresh LOBMarketMakingEnv instance for each test.
    Import is inside fixture so tests can be collected even before
    lob_env.py is fully implemented.
    """
    from envs.lob_env import LOBMarketMakingEnv
    return LOBMarketMakingEnv(reward_type="asymmetric", seed=42)


# ── Test 1: step output shape ─────────────────────────────────────────────

def test_step_output_shape(env):
    """
    env.step() must return (obs, reward, terminated, truncated, info)
    with correct types and shapes.
    """
    obs, info = env.reset(seed=0)

    # Observation shape must match observation_space
    assert isinstance(obs, np.ndarray), \
        f"obs must be np.ndarray, got {type(obs)}"
    assert obs.shape == env.observation_space.shape, \
        f"obs shape {obs.shape} != observation_space shape {env.observation_space.shape}"
    assert obs.dtype == np.float32, \
        f"obs dtype must be float32, got {obs.dtype}"

    # Take one random step
    action = env.action_space.sample()
    obs2, reward, terminated, truncated, info = env.step(action)

    # Output types
    assert isinstance(obs2,       np.ndarray), "obs must be np.ndarray"
    assert isinstance(reward,     float),      "reward must be float"
    assert isinstance(terminated, bool),       "terminated must be bool"
    assert isinstance(truncated,  bool),       "truncated must be bool"
    assert isinstance(info,       dict),       "info must be dict"

    # Observation shape consistency
    assert obs2.shape == env.observation_space.shape, \
        "obs shape changed between steps"

    # Required info keys
    required_keys = {"step", "inventory", "mid_price", "cash"}
    missing = required_keys - set(info.keys())
    assert not missing, f"info dict missing keys: {missing}"


# ── Test 2: reward is finite ──────────────────────────────────────────────

@pytest.mark.parametrize("reward_type", ["asymmetric", "quadratic", "sparse"])
def test_reward_finite(reward_type):
    """
    Reward must never be NaN or Inf across 100 random steps,
    for all three reward formulations.
    """
    from envs.lob_env import LOBMarketMakingEnv
    env = LOBMarketMakingEnv(reward_type=reward_type, seed=42)

    env.reset(seed=42)
    for step in range(100):
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)

        assert np.isfinite(reward), \
            f"reward={reward} is not finite at step {step} with reward_type={reward_type}"

        if terminated or truncated:
            env.reset()


# ── Test 3: inventory constraint ─────────────────────────────────────────

def test_inventory_constraint(env):
    """
    Inventory |q| must never exceed Q_max=10 at any step.
    Run for 500 steps to stress-test the constraint.
    """
    Q_max = env.Q_max
    env.reset(seed=0)

    for step in range(500):
        action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)

        inventory = info.get("inventory", 0)
        assert abs(inventory) <= Q_max, (
            f"Inventory constraint violated at step {step}: "
            f"|q|={abs(inventory)} > Q_max={Q_max}"
        )

        if terminated or truncated:
            env.reset()


# ── Additional sanity tests ───────────────────────────────────────────────

def test_episode_terminates(env):
    """
    Episode should terminate or truncate within episode_len steps.
    """
    env.reset(seed=0)
    max_steps = env.episode_len + 10   # small buffer

    for step in range(max_steps):
        action = env.action_space.sample()
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            return   # passed

    pytest.fail(
        f"Episode did not terminate within {max_steps} steps "
        f"(episode_len={env.episode_len})"
    )


def test_reset_clears_state(env):
    """
    After reset(), inventory should be 0 and step counter should be 0.
    """
    # Run half an episode
    env.reset(seed=0)
    for _ in range(50):
        env.step(env.action_space.sample())

    # Reset and check
    obs, info = env.reset(seed=1)
    assert info["inventory"] == 0, \
        f"Inventory not cleared on reset: {info['inventory']}"
    assert info["step"] == 0, \
        f"Step counter not cleared on reset: {info['step']}"
    assert np.all(np.isfinite(obs)), "Non-finite obs after reset"
