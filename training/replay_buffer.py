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
learn temporal dependencies. Use `sample_sequences()` for this.

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

        Each sequence has length seq_len. Sequences that cross episode
        boundaries are valid — the LSTM hidden state is reset at episode
        boundaries by the training loop using the `done` flags.

        Parameters
        ----------
        batch_size : int — number of sequences to sample

        Returns
        -------
        batch   : dict — keys: obs, action, reward, next_obs, done
                         each value shape (B, seq_len, ...)
        indices : np.ndarray shape (B,) — start indices of each sequence
        weights : np.ndarray shape (B,) — IS weights (all ones for uniform)
        """
        assert len(self) >= self.seq_len, (
            f"Buffer has {len(self)} transitions, need at least seq_len={self.seq_len}"
        )

        # For sequence sampling, always use uniform start-index sampling
        # (PER sequence sampling is complex and offers marginal benefit
        #  over PER transition sampling; use uniform here and PER for
        #  the TD-error update step)
        n       = len(self)
        starts  = np.random.randint(0, n - self.seq_len + 1, size=batch_size)
        indices = starts

        seqs = []
        for start in starts:
            seq = [self._get(start + t) for t in range(self.seq_len)]
            seqs.append(seq)

        batch   = self._collate_sequences(seqs)
        weights = np.ones(batch_size, dtype=np.float32)
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

        Only valid for indices returned by sample() (tree leaf indices,
        always >= capacity). Indices from sample_sequences() are start
        positions (< capacity) and are silently skipped — sequence PER
        is non-standard and not implemented.

        Parameters
        ----------
        indices    : np.ndarray shape (B,) — tree indices from sample()
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

    def __repr__(self) -> str:
        mode = "PER" if self.prioritized else "Uniform"
        return (
            f"ReplayBuffer("
            f"capacity={self.capacity}, "
            f"mode={mode}, "
            f"size={len(self)}, "
            f"seq_len={self.seq_len})"
        )