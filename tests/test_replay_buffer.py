"""
tests/test_replay_buffer.py

Unit tests for training/replay_buffer.py.

Covers:
    SumTree — structure, total, update, overflow, sampling
    ReplayBuffer (PER) — fill, sample shapes, weights, priority update,
                         beta annealing, max_priority tracking
    ReplayBuffer (uniform) — weights=1, no priority update
    Sequence sampling — shape, done flags preserved
    Edge cases — is_ready, capacity overflow, single-element buffer

Run with:
    pytest tests/test_replay_buffer.py -v
"""

import numpy as np
import pytest
from training.replay_buffer import ReplayBuffer, SumTree, Transition


# ── Shared fixtures ───────────────────────────────────────────────────────────

OBS_DIM   = 18
N_ACTIONS = 121
SEQ_LEN   = 5
CAPACITY  = 100


def _obs():
    return np.random.randn(OBS_DIM).astype(np.float32)


def _fill(buf: ReplayBuffer, n: int, done_every: int = 0) -> None:
    for i in range(n):
        done = (done_every > 0) and (i % done_every == done_every - 1)
        buf.push(_obs(), i % N_ACTIONS, float(i) * 0.1, _obs(), done)


@pytest.fixture
def per_buf():
    return ReplayBuffer(capacity=CAPACITY, prioritized=True,
                        seq_len=SEQ_LEN, seed=42)


@pytest.fixture
def uni_buf():
    return ReplayBuffer(capacity=CAPACITY, prioritized=False,
                        seq_len=SEQ_LEN, seed=42)


@pytest.fixture
def filled_per(per_buf):
    _fill(per_buf, 50, done_every=10)
    return per_buf


@pytest.fixture
def filled_uni(uni_buf):
    _fill(uni_buf, 50, done_every=10)
    return uni_buf


# ═══════════════════════════════════════════════════════════════════════════════
# SumTree
# ═══════════════════════════════════════════════════════════════════════════════

class TestSumTree:

    def test_total_correct_after_adds(self):
        tree = SumTree(8)
        priorities = [3.0, 1.0, 2.0, 4.0, 5.0, 2.0, 1.0, 3.0]
        for p in priorities:
            tree.add(p, "x")
        assert abs(tree.total - sum(priorities)) < 1e-6

    def test_n_entries_increments(self):
        tree = SumTree(8)
        for i in range(5):
            tree.add(1.0, i)
        assert tree.n_entries == 5

    def test_n_entries_capped_at_capacity(self):
        tree = SumTree(4)
        for i in range(10):
            tree.add(1.0, i)
        assert tree.n_entries == 4

    def test_overflow_wraps_correctly(self):
        """After overflow, oldest entries are overwritten and total is updated."""
        tree = SumTree(4)
        for _ in range(4):
            tree.add(1.0, "old")
        assert abs(tree.total - 4.0) < 1e-6
        # Overwrite with higher priority
        tree.add(10.0, "new")
        # One old entry replaced — total should be 3*1 + 10 = 13
        assert abs(tree.total - 13.0) < 1e-6

    def test_update_changes_total(self):
        tree = SumTree(8)
        tree.add(1.0, "x")
        tree.update(tree.capacity, 5.0)   # first leaf
        assert abs(tree.total - 5.0) < 1e-6

    def test_get_returns_valid_data(self):
        tree = SumTree(8)
        for i in range(8):
            tree.add(float(i + 1), f"item_{i}")
        idx, priority, data = tree.get(tree.total / 2)
        assert data is not None
        assert priority > 0
        assert isinstance(idx, (int, np.integer))

    def test_get_priority_matches_tree(self):
        tree = SumTree(4)
        for p in [1.0, 2.0, 3.0, 4.0]:
            tree.add(p, "x")
        idx, priority, _ = tree.get(tree.total / 2)
        assert abs(tree.tree[idx] - priority) < 1e-6

    def test_high_priority_sampled_more_often(self):
        """Item with 10x priority should be sampled ~10x more often."""
        tree = SumTree(2)
        tree.add(1.0, "low")
        tree.add(10.0, "high")
        counts = {"low": 0, "high": 0}
        rng = np.random.default_rng(0)
        for _ in range(1000):
            s = rng.uniform(0, tree.total)
            _, _, data = tree.get(s)
            if data is not None:
                counts[data] += 1
        # High-priority item should be sampled at least 5x more
        assert counts["high"] > counts["low"] * 5


# ═══════════════════════════════════════════════════════════════════════════════
# ReplayBuffer — construction and basic properties
# ═══════════════════════════════════════════════════════════════════════════════

class TestReplayBufferConstruction:
    def test_per_initial_length_zero(self, per_buf):
        assert len(per_buf) == 0

    def test_uniform_initial_length_zero(self, uni_buf):
        assert len(uni_buf) == 0

    def test_is_ready_false_when_empty(self, per_buf):
        assert not per_buf.is_ready(10)

    def test_repr_per(self, per_buf):
        r = repr(per_buf)
        assert "PER" in r
        assert str(CAPACITY) in r

    def test_repr_uniform(self, uni_buf):
        r = repr(uni_buf)
        assert "Uniform" in r


# ═══════════════════════════════════════════════════════════════════════════════
# ReplayBuffer — push
# ═══════════════════════════════════════════════════════════════════════════════

class TestReplayBufferPush:
    def test_length_increments(self, per_buf):
        _fill(per_buf, 10)
        assert len(per_buf) == 10

    def test_length_capped_at_capacity(self, per_buf):
        _fill(per_buf, CAPACITY + 50)
        assert len(per_buf) == CAPACITY

    def test_is_ready_after_sufficient_pushes(self, per_buf):
        _fill(per_buf, 10)
        assert per_buf.is_ready(10)
        assert not per_buf.is_ready(11)

    def test_push_with_done_true(self, per_buf):
        per_buf.push(_obs(), 0, 1.0, _obs(), True)
        assert len(per_buf) == 1

    def test_push_stores_float32_obs(self, per_buf):
        obs = np.ones(OBS_DIM, dtype=np.float64)
        per_buf.push(obs, 0, 0.0, obs, False)
        # Verify via sequence sample after filling enough
        _fill(per_buf, SEQ_LEN)
        batch, _, _ = per_buf.sample_sequences(1)
        assert batch["obs"].dtype == np.float32


# ═══════════════════════════════════════════════════════════════════════════════
# ReplayBuffer — sample (i.i.d.)
# ═══════════════════════════════════════════════════════════════════════════════

class TestReplayBufferSample:

    def test_per_sample_shapes(self, filled_per):
        batch, indices, weights = filled_per.sample(10)
        assert batch["obs"].shape      == (10, OBS_DIM)
        assert batch["action"].shape   == (10,)
        assert batch["reward"].shape   == (10,)
        assert batch["next_obs"].shape == (10, OBS_DIM)
        assert batch["done"].shape     == (10,)
        assert indices.shape           == (10,)
        assert weights.shape           == (10,)

    def test_per_weights_in_unit_interval(self, filled_per):
        _, _, weights = filled_per.sample(20)
        assert np.all(weights >= 0.0)
        assert np.all(weights <= 1.0 + 1e-6)

    def test_per_weights_max_is_one(self, filled_per):
        _, _, weights = filled_per.sample(20)
        assert abs(weights.max() - 1.0) < 1e-5

    def test_uniform_weights_all_ones(self, filled_uni):
        _, _, weights = filled_uni.sample(20)
        np.testing.assert_array_equal(weights, np.ones(20, dtype=np.float32))

    def test_uniform_indices_all_zeros(self, filled_uni):
        """Uniform buffer returns dummy zero indices."""
        _, indices, _ = filled_uni.sample(10)
        np.testing.assert_array_equal(indices, np.zeros(10, dtype=np.int32))

    def test_sample_raises_if_not_ready(self, per_buf):
        _fill(per_buf, 5)
        with pytest.raises(AssertionError):
            per_buf.sample(10)

    def test_per_action_dtype(self, filled_per):
        batch, _, _ = filled_per.sample(10)
        assert batch["action"].dtype == np.int64

    def test_per_done_dtype_float(self, filled_per):
        batch, _, _ = filled_per.sample(10)
        assert batch["done"].dtype == np.float32


# ═══════════════════════════════════════════════════════════════════════════════
# ReplayBuffer — priority updates and beta annealing
# ═══════════════════════════════════════════════════════════════════════════════

class TestPriorityUpdates:
    def test_update_priorities_noop_for_uniform(self, filled_uni):
        """update_priorities should silently do nothing for uniform buffer."""
        _, indices, _ = filled_uni.sample(10)
        filled_uni.update_priorities(indices, np.ones(10))   # no error

    def test_update_priorities_changes_sampling(self, per_buf):
        """After boosting one transition's priority, it should be sampled more."""
        _fill(per_buf, 50)
        batch, indices, _ = per_buf.sample(10)
        # Set all sampled priorities very high
        per_buf.update_priorities(indices, np.ones(10) * 1000.0)
        # Re-sample — should get overlapping indices
        batch2, indices2, _ = per_buf.sample(10)
        overlap = len(set(indices.tolist()) & set(indices2.tolist()))
        assert overlap > 0, "High priority indices should be re-sampled"

    def test_beta_anneals_toward_one(self):
        buf = ReplayBuffer(capacity=100, prioritized=True,
                           beta=0.4, beta_increment=0.1, seq_len=SEQ_LEN)
        _fill(buf, 50)
        for _ in range(6):
            buf.sample(10)
        assert buf.beta == pytest.approx(1.0)

    def test_beta_does_not_exceed_one(self):
        buf = ReplayBuffer(capacity=100, prioritized=True,
                           beta=0.9, beta_increment=0.5, seq_len=SEQ_LEN)
        _fill(buf, 50)
        buf.sample(10)
        assert buf.beta <= 1.0

    def test_max_priority_updated_after_push(self):
        buf = ReplayBuffer(capacity=100, prioritized=True, seq_len=SEQ_LEN)
        _fill(buf, 10)
        _, indices, _ = buf.sample(5)
        buf.update_priorities(indices, np.array([100.0] * 5))
        assert buf._max_priority >= 100.0


# ═══════════════════════════════════════════════════════════════════════════════
# ReplayBuffer — sequence sampling
# ═══════════════════════════════════════════════════════════════════════════════

class TestSequenceSampling:

    def test_sequence_output_shapes(self, filled_uni):
        batch, indices, weights = filled_uni.sample_sequences(8)
        assert batch["obs"].shape      == (8, SEQ_LEN, OBS_DIM)
        assert batch["action"].shape   == (8, SEQ_LEN)
        assert batch["reward"].shape   == (8, SEQ_LEN)
        assert batch["next_obs"].shape == (8, SEQ_LEN, OBS_DIM)
        assert batch["done"].shape     == (8, SEQ_LEN)
        assert indices.shape           == (8,)
        assert weights.shape           == (8,)

    def test_sequence_weights_all_ones(self, filled_uni):
        """Sequence sampling always returns uniform weights."""
        _, _, weights = filled_uni.sample_sequences(8)
        np.testing.assert_array_equal(weights, np.ones(8, dtype=np.float32))

    def test_sequence_preserves_done_flags(self):
        """done=True transitions should appear in sampled sequences."""
        buf = ReplayBuffer(capacity=200, prioritized=False,
                           seq_len=SEQ_LEN, seed=0)
        for i in range(100):
            done = (i % 20 == 19)   # done every 20 steps
            buf.push(_obs(), 0, 0.0, _obs(), done)
        # Sample many sequences — at least one should contain a done=True
        batch, _, _ = buf.sample_sequences(32)
        # done array shape: (32, SEQ_LEN), values are 0.0 or 1.0
        assert batch["done"].max() == 1.0, \
            "At least one done=True should appear across 32 sampled sequences"

    def test_sequence_raises_if_not_enough_data(self):
        buf = ReplayBuffer(capacity=100, prioritized=False,
                           seq_len=10, seed=0)
        _fill(buf, 5)   # less than seq_len
        with pytest.raises(AssertionError):
            buf.sample_sequences(4)

    def test_sequence_per_buffer_also_works(self, filled_per):
        """Sequence sampling works regardless of prioritized flag."""
        batch, _, weights = filled_per.sample_sequences(4)
        assert batch["obs"].shape == (4, SEQ_LEN, OBS_DIM)
        np.testing.assert_array_equal(weights, np.ones(4, dtype=np.float32))

    def test_sequence_obs_dtype_float32(self, filled_uni):
        batch, _, _ = filled_uni.sample_sequences(4)
        assert batch["obs"].dtype == np.float32

    def test_sequence_action_dtype_int64(self, filled_uni):
        batch, _, _ = filled_uni.sample_sequences(4)
        assert batch["action"].dtype == np.int64


# ═══════════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases:
    def test_capacity_exactly_filled(self):
        buf = ReplayBuffer(capacity=10, prioritized=False,
                           seq_len=SEQ_LEN, seed=0)
        _fill(buf, 10)
        assert len(buf) == 10
        assert buf.is_ready(10)

    def test_large_batch_size_equals_buffer(self, filled_uni):
        """Sampling batch_size == len(buffer) should not crash."""
        batch, _, _ = filled_uni.sample(50)
        assert batch["obs"].shape == (50, OBS_DIM)

    def test_deterministic_with_seed(self):
        """
        Sampling is reproducible: resetting both random and numpy seeds
        before sample produces identical results.
        Uniform buffer uses random.sample() internally.
        """
        import random as _random
        buf = ReplayBuffer(capacity=100, prioritized=False,
                           seq_len=SEQ_LEN, seed=7)
        for i in range(30):
            obs = np.full(OBS_DIM, float(i), dtype=np.float32)
            buf.push(obs, i % N_ACTIONS, float(i), obs, False)

        _random.seed(7); np.random.seed(7)
        b1, _, _ = buf.sample(10)
        _random.seed(7); np.random.seed(7)
        b2, _, _ = buf.sample(10)
        np.testing.assert_array_equal(b1["reward"], b2["reward"])

    def test_push_hidden_none_stored(self):
        """hidden=None should be stored without error."""
        buf = ReplayBuffer(capacity=10, prioritized=False,
                           seq_len=SEQ_LEN, seed=0)
        buf.push(_obs(), 0, 0.0, _obs(), False, hidden=None)
        assert len(buf) == 1

    def test_per_new_entries_get_max_priority(self):
        """
        New transitions get max_priority so they are sampled at least once.
        After adding a high-priority transition, new transitions should
        have priority >= all other transitions.
        """
        buf = ReplayBuffer(capacity=100, prioritized=True,
                           seq_len=SEQ_LEN, seed=0)
        _fill(buf, 20)
        batch, indices, _ = buf.sample(5)
        buf.update_priorities(indices, np.array([500.0] * 5))
        # New transition should get max_priority >= 500
        buf.push(_obs(), 0, 0.0, _obs(), False)
        assert buf._max_priority >= 500.0