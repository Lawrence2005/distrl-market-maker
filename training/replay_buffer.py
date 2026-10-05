"""
training/replay_buffer.py

Experience replay buffer with optional Prioritized Experience Replay (PER).

Two modes controlled by the `prioritized` flag:
    prioritized=True  — Prioritized Experience Replay (Schaul et al. 2016)
                        Samples transitions by TD-error priority via sum-tree.
                        Importance-sampling weights correct for the bias.
    prioritized=False — Uniform random sampling (standard DQN replay)
                        Used for the non-prioritized DQN ablation variant.

Transition format
-----------------
Each transition stored as a named tuple:
    obs        : np.ndarray  — encoder input at time t
                               shape depends on encoder type:
                               - handcrafted: (obs_dim,)       = (18,)
                               - AE / CNN:    (snapshot_dim,)  = (20,)
    action     : int         — flat action index ∈ [0, N_OFFSET_LEVELS²)
    reward     : float       — step reward from LOBMarketMakingEnv
    next_obs   : np.ndarray  — encoder input at time t+1
    done       : bool        — True if episode ended (terminated or truncated)
    hidden     : tuple | None — LSTM (h, c) at time t, for DRQN sequence replay
                               None if not using recurrent agent

DRQN sequence sampling
-----------------------
For recurrent agents, experiences must be sampled as contiguous sequences
of length `seq_len` rather than individual transitions, so the LSTM can
learn temporal dependencies. Use `sample_sequences()` for this. When
prioritized=True, the sequence's LAST transition (the one every agent's
train_step actually bootstraps from, via `[:, -1]`) is sampled through the
same sum-tree stratified scheme as `sample()`/`_sample_per()`, with
matching importance-sampling weights — the priority attaches to the
transition the TD update actually uses.

A sampled window is REJECTED and resampled if any `done=True` falls at any
position except the last (i.e. if it would splice the tail of one episode
onto the head of a different, unrelated one under a single continuous LSTM
hidden state). No agent's train_step() resets hidden state mid-window — they
all do a single reset_hidden() + forward() over the whole seq_len window
(confirmed by reading dqn.py/qrdqn.py/iqn.py's train_step, which only ever
reads dones[:, -1] for the Bellman terminal mask, never the intermediate
positions) — so a cross-episode window would silently feed the network
nonsense "history" blending two independently-seeded, unrelated market
realizations. This buffer used to allow such windows and document that "the
training loop" handles the reset — it never did; found via user-prompted
investigation into why recurrent variants underperform (see
results/README.md). Matches the "Random
Updates" DRQN training regime from Hausknecht & Stone (2015, cited below) —
one of the two training regimes their paper studies, neither of which lets
a window span two independent episodes.

For non-recurrent ablations, use `sample()` for i.i.d. transition sampling.

PER parameters
--------------
alpha : float — priority exponent (0 = uniform, 1 = fully prioritised)
                typical value: 0.6
beta  : float — importance-sampling exponent, annealed 0.4 → 1.0 over training
                controls bias correction strength
beta_increment : float — per-sample increment to beta (anneals toward 1.0)
epsilon : float — small constant added to priorities to ensure non-zero
                  probability for all transitions (default 1e-6)

Reference
---------
Schaul et al. (2015) — Prioritized Experience Replay. ICLR 2016.
Hausknecht & Stone (2015) — Deep Recurrent Q-Networks.

Week 5 deliverable.
"""

from __future__ import annotations

import random
from collections import namedtuple
from typing import Optional

import numpy as np


# ── Transition ────────────────────────────────────────────────────────────────

Transition = namedtuple(
    "Transition",
    ["obs", "action", "reward", "next_obs", "done", "hidden"],
)


# ── Sum-tree (PER backbone) ───────────────────────────────────────────────────

class SumTree:
    """
    Binary sum-tree for O(log n) priority sampling.

    Leaf nodes store individual transition priorities. Internal nodes
    store the sum of their children, so the root always holds the total
    priority sum. Sampling a value s ∈ [0, total] in O(log n) by
    traversing from root to leaf.

    Parameters
    ----------
    capacity : int — maximum number of transitions (must be power of 2
                     for clean indexing, but any positive int works)
    """

    def __init__(self, capacity: int):
        self.capacity  = capacity
        self.tree      = np.zeros(2 * capacity, dtype=np.float64)
        self.data      = [None] * capacity
        self._write    = 0       # next write position (circular)
        self.n_entries = 0       # number of valid entries

    # ── Internal helpers ──────────────────────────────────────────────

    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up to root."""
        parent = idx // 2
        self.tree[parent] += change
        if parent != 1:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Find leaf index for cumulative priority value s."""
        left  = 2 * idx
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    # ── Public API ────────────────────────────────────────────────────

    @property
    def total(self) -> float:
        """Total sum of all priorities (stored at root = index 1)."""
        return float(self.tree[1])

    def add(self, priority: float, data) -> None:
        """
        Add a new transition with given priority.

        Overwrites the oldest entry when buffer is full (circular).
        """
        idx = self._write + self.capacity
        self.data[self._write] = data
        self.update(idx, priority)
        self._write    = (self._write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        """Update priority of leaf at tree index idx."""
        change        = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> tuple[int, float, object]:
        """
        Sample leaf for cumulative priority value s.

        Parameters
        ----------
        s : float — value in [0, total]

        Returns
        -------
        (tree_idx, priority, data) : tuple
            tree_idx : index in self.tree (used for priority updates)
            priority : priority of sampled leaf
            data     : stored Transition
        """
        idx      = self._retrieve(1, s)
        data_idx = idx - self.capacity
        return idx, float(self.tree[idx]), self.data[data_idx]


# ── Replay buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Experience replay buffer with optional Prioritized Experience Replay.

    Parameters
    ----------
    capacity       : int   — maximum number of transitions to store
    prioritized    : bool  — use PER (True) or uniform sampling (False)
    alpha          : float — PER priority exponent (default 0.6)
    beta           : float — PER IS weight exponent, annealed to 1.0 (default 0.4)
    beta_increment : float — per-sample increment to beta (default 1e-4)
    epsilon        : float — minimum priority floor (default 1e-6)
    seq_len        : int   — sequence length for DRQN sampling (default 30)
    seed           : int   — random seed (default 42)
    """

    def __init__(
        self,
        capacity:       int   = 100_000,
        prioritized:    bool  = True,
        alpha:          float = 0.6,
        beta:           float = 0.4,
        beta_increment: float = 1e-4,
        epsilon:        float = 1e-6,
        seq_len:        int   = 30,
        seed:           int   = 42,
    ):
        self.capacity       = capacity
        self.prioritized    = prioritized
        self.alpha          = alpha
        self.beta           = beta
        self.beta_increment = beta_increment
        self.epsilon        = epsilon
        self.seq_len        = seq_len

        random.seed(seed)
        np.random.seed(seed)

        if prioritized:
            self._tree = SumTree(capacity)
            # Max priority seen so far — new transitions get max priority
            # so they are sampled at least once before being updated
            self._max_priority = 1.0
        else:
            self._buffer: list[Transition] = []
            self._write  = 0

    # ── Adding transitions ────────────────────────────────────────────

    def push(
        self,
        obs:      np.ndarray,
        action:   int,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
        hidden:   Optional[tuple] = None,
    ) -> None:
        """
        Store a single transition.

        New transitions get maximum priority so they are guaranteed to
        be sampled at least once (PER) before their priority is updated
        by the agent after computing the TD error.

        Parameters
        ----------
        obs      : np.ndarray — encoder input at time t
        action   : int        — flat action index
        reward   : float      — step reward
        next_obs : np.ndarray — encoder input at time t+1
        done     : bool       — episode ended flag
        hidden   : tuple|None — LSTM (h, c) state at time t (DRQN only)
        """
        transition = Transition(
            obs      = np.array(obs,      dtype=np.float32),
            action   = int(action),
            reward   = float(reward),
            next_obs = np.array(next_obs, dtype=np.float32),
            done     = bool(done),
            hidden   = hidden,
        )

        if self.prioritized:
            priority = self._max_priority ** self.alpha
            self._tree.add(priority, transition)
        else:
            if len(self._buffer) < self.capacity:
                self._buffer.append(transition)
            else:
                self._buffer[self._write] = transition
            self._write = (self._write + 1) % self.capacity

    # ── Sampling ──────────────────────────────────────────────────────

    def sample(
        self,
        batch_size: int,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """
        Sample a batch of individual transitions.

        Used for non-recurrent agents (DQN ablation) or when sequence
        structure is not needed.

        Parameters
        ----------
        batch_size : int — number of transitions to sample

        Returns
        -------
        batch   : dict — keys: obs, action, reward, next_obs, done
                         each value is a np.ndarray of shape (B, ...)
        indices : np.ndarray shape (B,) — tree indices for PER updates
                  (all zeros for uniform sampling — not used)
        weights : np.ndarray shape (B,) — IS weights (all ones for uniform)
        """
        assert len(self) >= batch_size, (
            f"Buffer has {len(self)} transitions, need {batch_size}"
        )

        if self.prioritized:
            return self._sample_per(batch_size)
        return self._sample_uniform(batch_size)

    def sample_sequences(
        self,
        batch_size: int,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """
        Sample a batch of contiguous sequences for DRQN training.

        Each sequence has length seq_len and is entirely from ONE episode —
        a candidate window is rejected and resampled if any `done=True`
        falls before its last position (see module docstring for why: no
        agent's train_step actually resets hidden state mid-window, so a
        cross-episode window would corrupt the LSTM's hidden state with two
        unrelated market realizations spliced together).

        Parameters
        ----------
        batch_size : int — number of sequences to sample

        Returns
        -------
        batch   : dict — keys: obs, action, reward, next_obs, done
                         each value shape (B, seq_len, ...)
        indices : np.ndarray shape (B,) — for prioritized=True, real
                  sum-tree leaf indices (always >= capacity) for the
                  sequence's LAST transition, usable directly with
                  update_priorities(); for prioritized=False, start
                  indices (< capacity, silently skipped by
                  update_priorities()).
        weights : np.ndarray shape (B,) — IS weights (all ones for uniform)
        """
        assert len(self) >= self.seq_len, (
            f"Buffer has {len(self)} transitions, need at least seq_len={self.seq_len}"
        )

        if self.prioritized:
            return self._sample_sequences_per(batch_size)
        return self._sample_sequences_uniform(batch_size)

    def _sample_sequences_uniform(
        self,
        batch_size: int,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Uniform start-index sequence sampling, rejecting cross-episode windows."""
        n       = len(self)
        starts  = np.zeros(batch_size, dtype=np.int64)
        seqs    = []

        for i in range(batch_size):
            while True:
                start = int(np.random.randint(0, n - self.seq_len + 1))
                seq   = [self._get(start + t) for t in range(self.seq_len)]
                if not self._crosses_episode_boundary(seq):
                    break
            starts[i] = start
            seqs.append(seq)

        indices = starts
        batch   = self._collate_sequences(seqs)
        weights = np.ones(batch_size, dtype=np.float32)
        return batch, indices, weights

    @staticmethod
    def _crosses_episode_boundary(seq: list[Transition]) -> bool:
        """True if any transition before the last one ends an episode."""
        return any(t.done for t in seq[:-1])

    def _sample_sequences_per(
        self,
        batch_size: int,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """
        PER-weighted sequence sampling.

        Samples each sequence's END position (the transition every agent's
        train_step actually indexes with `[:, -1]` for the TD update) via
        the same stratified sum-tree scheme as `_sample_per()`, then walks
        backward `seq_len` steps to build the contiguous window. Returned
        indices are real tree leaf indices, so update_priorities() applies
        to them unmodified.
        """
        segment = self._tree.total / batch_size

        seqs       = []
        indices    = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float64)

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s  = np.random.uniform(lo, hi)
            idx, priority, transition = self._tree.get(s)
            end = idx - self._tree.capacity
            # Retry (full-range resample, same fallback as _sample_per) on an
            # unwritten slot, a position with no full seq_len window of
            # history behind it, or a window that crosses an episode
            # boundary (see module docstring — the priority still attaches
            # to a genuinely different transition each retry, so this
            # doesn't bias which END position ultimately gets used, only
            # rules out ends whose full window isn't a single episode).
            while True:
                while transition is None or end < self.seq_len - 1:
                    s = np.random.uniform(0, self._tree.total)
                    idx, priority, transition = self._tree.get(s)
                    end = idx - self._tree.capacity
                start = end - self.seq_len + 1
                seq   = [self._get(start + t) for t in range(self.seq_len)]
                if not self._crosses_episode_boundary(seq):
                    break
                s = np.random.uniform(0, self._tree.total)
                idx, priority, transition = self._tree.get(s)
                end = idx - self._tree.capacity

            seqs.append(seq)
            indices[i]    = idx
            priorities[i] = priority

        # Importance-sampling weights — identical formula to _sample_per()
        n       = self._tree.n_entries
        probs   = priorities / self._tree.total
        weights = (n * probs) ** (-self.beta)
        weights = (weights / weights.max()).astype(np.float32)

        # Anneal beta toward 1.0
        self.beta = min(1.0, self.beta + self.beta_increment)

        batch = self._collate_sequences(seqs)
        return batch, indices, weights

    # ── Priority updates (PER only) ───────────────────────────────────

    def update_priorities(
        self,
        indices:    np.ndarray,
        priorities: np.ndarray,
    ) -> None:
        """
        Update transition priorities after TD-error computation.

        Called by the agent after each training step with the new
        absolute TD errors for the sampled batch.

        Valid for indices returned by sample() or, when prioritized=True,
        sample_sequences() — both hand back real tree leaf indices
        (always >= capacity). If sample_sequences() was called on a
        non-prioritized buffer, its start-position indices (< capacity)
        are silently skipped here (no-op, since there is nothing to
        update on a uniform buffer).

        Parameters
        ----------
        indices    : np.ndarray shape (B,) — tree indices from sample()
                     or sample_sequences()
        priorities : np.ndarray shape (B,) — |TD error| + epsilon
        """
        if not self.prioritized:
            return   # no-op for uniform buffer

        for idx, priority in zip(indices, priorities):
            idx = int(idx)
            # Leaf nodes start at capacity; skip dummy/sequence indices
            if idx < self._tree.capacity:
                continue
            p = float(priority) + self.epsilon
            self._tree.update(idx, p ** self.alpha)
            self._max_priority = max(self._max_priority, p)

    # ── Private sampling helpers ──────────────────────────────────────

    def _sample_per(
        self,
        batch_size: int,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """PER sampling via sum-tree stratified sampling."""
        transitions = []
        indices     = np.zeros(batch_size, dtype=np.int32)
        priorities  = np.zeros(batch_size, dtype=np.float64)

        segment = self._tree.total / batch_size

        for i in range(batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s  = np.random.uniform(lo, hi)
            idx, priority, transition = self._tree.get(s)
            # Guard against None data (tree slots not yet written)
            while transition is None:
                s   = np.random.uniform(0, self._tree.total)
                idx, priority, transition = self._tree.get(s)
            transitions.append(transition)
            indices[i]    = idx
            priorities[i] = priority

        # Importance-sampling weights
        n           = self._tree.n_entries
        probs       = priorities / self._tree.total
        weights     = (n * probs) ** (-self.beta)
        weights     = (weights / weights.max()).astype(np.float32)

        # Anneal beta toward 1.0
        self.beta   = min(1.0, self.beta + self.beta_increment)

        batch = self._collate(transitions)
        return batch, indices, weights

    def _sample_uniform(
        self,
        batch_size: int,
    ) -> tuple[dict, np.ndarray, np.ndarray]:
        """Uniform random sampling."""
        transitions = random.sample(self._buffer[:len(self)], batch_size)
        indices     = np.zeros(batch_size, dtype=np.int32)   # unused
        weights     = np.ones(batch_size, dtype=np.float32)  # unweighted
        batch       = self._collate(transitions)
        return batch, indices, weights

    def _get(self, idx: int) -> Transition:
        """Get transition at circular buffer position idx."""
        if self.prioritized:
            return self._tree.data[idx % self.capacity]
        return self._buffer[idx % len(self._buffer)]

    @staticmethod
    def _collate(transitions: list[Transition]) -> dict:
        """Stack a list of Transitions into batched numpy arrays."""
        return {
            "obs":      np.stack([t.obs      for t in transitions]),
            "action":   np.array([t.action   for t in transitions], dtype=np.int64),
            "reward":   np.array([t.reward   for t in transitions], dtype=np.float32),
            "next_obs": np.stack([t.next_obs for t in transitions]),
            "done":     np.array([t.done     for t in transitions], dtype=np.float32),
        }

    @staticmethod
    def _collate_sequences(seqs: list[list[Transition]]) -> dict:
        """
        Stack a list of sequences into batched arrays.

        Output shapes: (B, seq_len, ...) for obs/next_obs,
                       (B, seq_len) for action/reward/done.
        """
        return {
            "obs":      np.stack([[t.obs      for t in seq] for seq in seqs]),
            "action":   np.array([[t.action   for t in seq] for seq in seqs],
                                  dtype=np.int64),
            "reward":   np.array([[t.reward   for t in seq] for seq in seqs],
                                  dtype=np.float32),
            "next_obs": np.stack([[t.next_obs for t in seq] for seq in seqs]),
            "done":     np.array([[t.done     for t in seq] for seq in seqs],
                                  dtype=np.float32),
        }

    # ── Properties ────────────────────────────────────────────────────

    def __len__(self) -> int:
        if self.prioritized:
            return self._tree.n_entries
        return len(self._buffer)

    def is_ready(self, batch_size: int) -> bool:
        """True when buffer has enough transitions to sample a batch."""
        return len(self) >= batch_size

    # ── Serialization ─────────────────────────────────────────────────
    # `run_agent_supervised.sh` always passes training.resume=true, even
    # on the very first attempt, so it takes effect on every OOM/crash/
    # host-restart mid-run. Without this, a resumed run restores weights,
    # optimizer state, and the epsilon/PER-beta schedule position (already
    # far along), but refills an EMPTY buffer under a low-epsilon,
    # near-converged policy — losing the transition diversity those
    # schedules assume is already present. Included in the agent's own
    # state_dict() so it round-trips through the existing checkpoint
    # format with no separate file; old checkpoints saved before this
    # existed simply have no "buffer" key, and load_state_dict() must
    # leave the buffer empty in that case (today's existing behavior),
    # not raise.

    def get_state(self) -> dict:
        """Full buffer contents + PER bookkeeping, for checkpointing."""
        if self.prioritized:
            return {
                "prioritized":   True,
                "tree_array":    self._tree.tree.copy(),
                "tree_data":     list(self._tree.data),
                "tree_write":    self._tree._write,
                "tree_entries":  self._tree.n_entries,
                "max_priority":  self._max_priority,
                "beta":          self.beta,
            }
        return {
            "prioritized": False,
            "buffer":      list(self._buffer),
            "write":       self._write,
        }

    def set_state(self, state: dict) -> None:
        """Restore buffer contents saved by get_state(). No-op on None."""
        if state is None:
            return
        if state["prioritized"] and self.prioritized:
            self._tree.tree      = state["tree_array"].copy()
            self._tree.data      = list(state["tree_data"])
            self._tree._write    = state["tree_write"]
            self._tree.n_entries = state["tree_entries"]
            self._max_priority   = state["max_priority"]
            self.beta            = state["beta"]
        elif not state["prioritized"] and not self.prioritized:
            self._buffer = list(state["buffer"])
            self._write  = state["write"]
        # else: prioritized-mode mismatch between the saved state and this
        # buffer (e.g. prioritized_replay config changed between the run
        # that saved this checkpoint and this resume) — leave the buffer
        # empty rather than guess; the warmup_steps gate re-applies from
        # the restored _steps count regardless.

    def __repr__(self) -> str:
        mode = "PER" if self.prioritized else "Uniform"
        return (
            f"ReplayBuffer("
            f"capacity={self.capacity}, "
            f"mode={mode}, "
            f"size={len(self)}, "
            f"seq_len={self.seq_len})"
        )