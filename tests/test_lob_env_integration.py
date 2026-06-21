# tests/test_env.py
"""
tests/test_env.py

Unit tests for the ABIDES-Gym LOB environment wrapper.

Four tests (specs from docs/mdp_formulation.md and Week 2 research plan):
    1. test_step_output_shape    — obs, reward, done, info correct shapes/types
    2. test_reward_finite        — no NaN or Inf in reward across 100 random steps
    3. test_inventory_constraint — |q| never exceeds Q_max=10
    4. test_hawkes_more_clustered_than_poisson — Hawkes arrival CV > Poisson CV

Run with:
    python -m pytest tests/test_lob_env_integration.py -v

Week 2 deliverable.
"""

import pytest
import numpy as np
from envs.hawkes_arrivals import HawkesProcess


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


@pytest.fixture
def hawkes_process():
    """Return a calibrated HawkesProcess with known clustering parameters."""
    return HawkesProcess(mu=0.5, alpha=0.6, beta=1.5)


@pytest.fixture
def poisson_process():
    """
    Return a degenerate HawkesProcess that behaves like Poisson
    (alpha very close to 0, so no self-excitation).
    """
    return HawkesProcess(mu=0.714, alpha=1e-6, beta=1.5)


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


# ── Test 4: Hawkes more clustered than Poisson ────────────────────────────

def test_hawkes_more_clustered_than_poisson(hawkes_process, poisson_process):
    """
    Hawkes process should produce more clustered arrivals than Poisson.

    We measure clustering via coefficient of variation (CV = std/mean)
    of interarrival times:
        - Poisson: CV ≈ 1.0 (exponential interarrivals)
        - Hawkes:  CV > 1.0 (clustered, bursty arrivals)

    Also check interarrival autocorrelation:
        - Hawkes should have positive autocorrelation (bursts follow bursts)
    """
    T    = 10000.0   # long horizon for statistical power
    seed = 42

    # Simulate both processes
    hawkes_times  = hawkes_process.simulate(T=T, seed=seed)
    poisson_times = poisson_process.simulate(T=T, seed=seed + 1)

    assert len(hawkes_times)  > 10, "Hawkes process generated too few events"
    assert len(poisson_times) > 10, "Poisson process generated too few events"

    # Compute interarrival times
    hawkes_iat  = np.diff(hawkes_times)
    poisson_iat = np.diff(poisson_times)

    # Coefficient of variation: Hawkes CV should exceed Poisson CV
    hawkes_cv  = hawkes_iat.std()  / hawkes_iat.mean()
    poisson_cv = poisson_iat.std() / poisson_iat.mean()

    assert hawkes_cv > poisson_cv, (
        f"Hawkes CV={hawkes_cv:.4f} should exceed Poisson CV={poisson_cv:.4f}"
    )
    assert hawkes_cv > 1.0, (
        f"Hawkes CV={hawkes_cv:.4f} should be > 1.0 (more clustered than Poisson)"
    )

    # Interarrival autocorrelation at lag 1
    hawkes_autocorr = np.corrcoef(hawkes_iat[:-1], hawkes_iat[1:])[0, 1]

    assert hawkes_autocorr > 0, (
        f"Hawkes interarrival autocorrelation={hawkes_autocorr:.4f} should be positive "
        f"(clustered arrivals: short gaps tend to follow short gaps)"
    )

    print(f"\n  Hawkes  CV={hawkes_cv:.4f}, autocorr={hawkes_autocorr:.4f}")
    print(f"  Poisson CV={poisson_cv:.4f}")


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


# ── Hawkes-only tests (no env dependency) ─────────────────────────────────

def test_hawkes_deterministic():
    """Same seed must produce identical arrival times."""
    hp  = HawkesProcess(mu=0.5, alpha=0.6, beta=1.5)
    ev1 = hp.simulate(T=1000.0, seed=42)
    ev2 = hp.simulate(T=1000.0, seed=42)
    assert np.allclose(ev1, ev2), "Hawkes simulation not deterministic"


def test_hawkes_positive_arrivals():
    """All arrival times must be positive and strictly increasing."""
    hp     = HawkesProcess(mu=0.5, alpha=0.6, beta=1.5)
    events = hp.simulate(T=3900.0, seed=42)
    assert len(events) > 0,                    "No events generated"
    assert np.all(events >= 0),                "Negative arrival times"
    assert np.all(np.diff(events) > 0),        "Arrival times not strictly increasing"
    assert np.all(events <= 3900.0),           "Events outside simulation window"


def test_hawkes_stationarity_check():
    """HawkesProcess should reject non-stationary parameters."""
    with pytest.raises(AssertionError):
        HawkesProcess(mu=0.5, alpha=2.0, beta=1.0)   # rho = 2.0 ≥ 1


def test_hawkes_mean_rate():
    """
    Empirical mean arrival rate should be close to theoretical rate
    μ / (1 − ρ) within 15%.
    """
    hp             = HawkesProcess(mu=0.5, alpha=0.6, beta=1.5)
    T              = 50000.0
    events         = hp.simulate(T=T, seed=42)
    empirical_rate = len(events) / T
    theoretical    = hp.mean_rate

    rel_error = abs(empirical_rate - theoretical) / theoretical
    assert rel_error < 0.15, (
        f"Empirical rate {empirical_rate:.4f} deviates "
        f"{rel_error:.1%} from theoretical {theoretical:.4f}"
    )