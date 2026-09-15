"""
baselines/fixed_spread.py

Fixed-spread baseline market maker.

The simplest possible market-making policy: always quote a symmetric
spread of fixed width around the current mid-price, ignoring inventory,
volatility, and time-to-close entirely.

Used as the performance floor in the evaluation suite. Any RL agent or
closed-form baseline that cannot beat this policy is not worth deploying.

Interface
---------
All three baselines share the same interface so they can be swapped
into the evaluation loop without changes:

    baseline = FixedSpreadBaseline(half_spread_ticks=2)
    action   = baseline.act(obs, info)      # → np.ndarray shape (2,)
    baseline.reset()                         # call at episode start

The returned action is a MultiDiscrete index pair (bid_idx, ask_idx)
matching LOBMarketMakingEnv's action space.

Reference
---------
Trivial benchmark — no citation needed.
Used in e.g. Spooner et al. (2018) as the na¨ıve floor.

Week 3 deliverable.
"""

import numpy as np
from typing import Dict, Any
from envs.lob_env import N_OFFSET_LEVELS, TICK_OFFSETS

def _offset_to_idx(offset: int) -> int:
    """
    Convert a desired tick offset to the nearest index in TICK_OFFSETS.

    Looks up the nearest available level rather than assuming
    TICK_OFFSETS[i] == i — that held when TICK_OFFSETS was contiguous
    (0..10) but not once its spacing changed (0,10,...,100).

    Parameters
    ----------
    offset : int — desired tick offset from mid

    Returns
    -------
    int — index into TICK_OFFSETS of the closest available level
    """
    return int(np.abs(TICK_OFFSETS - offset).argmin())


class FixedSpreadBaseline:
    """
    Fixed-spread market maker — quotes a constant symmetric spread.

    At every step, submits:
        bid at mid - half_spread_ticks * tick_size
        ask at mid + half_spread_ticks * tick_size

    No inventory adjustment, no volatility scaling, no time decay.

    Parameters
    ----------
    half_spread_ticks : int
        Desired half-spread in ticks — snapped to the nearest available
        TICK_OFFSETS level by _offset_to_idx, not required to match one
        exactly. Must be within [TICK_OFFSETS[0], TICK_OFFSETS[-1]].
    """

    name = "FixedSpread"

    def __init__(self, half_spread_ticks: int = 2

                 ):
        assert TICK_OFFSETS[0] <= half_spread_ticks <= TICK_OFFSETS[-1], (
            f"half_spread_ticks must be in [{TICK_OFFSETS[0]}, {TICK_OFFSETS[-1]}], "
            f"got {half_spread_ticks}"
        )

        self.half_spread_ticks = half_spread_ticks
        self._bid_idx = _offset_to_idx(half_spread_ticks)
        self._ask_idx = _offset_to_idx(half_spread_ticks)
        self._action  = np.array([self._bid_idx, self._ask_idx], dtype=np.int64)

    def reset(self) -> None:
        """Called at the start of each episode. No state to reset."""
        pass

    def act(
        self,
        obs:  np.ndarray,
        info: Dict[str, Any],
    ) -> np.ndarray:
        """
        Return fixed symmetric quotes regardless of market state.

        Parameters
        ----------
        obs  : np.ndarray — current observation (ignored)
        info : dict       — step info dict (ignored)

        Returns
        -------
        np.ndarray shape (2,) — [bid_idx, ask_idx]
        """
        return self._action.copy()

    def __repr__(self) -> str:
        return (
            f"FixedSpreadBaseline("
            f"half_spread_ticks={self.half_spread_ticks})"
        )