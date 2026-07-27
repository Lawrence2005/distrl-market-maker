from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

class AgentBase(ABC):
    """
    Abstract base class for all agents in the environment.
    """

    name: str
    is_online: bool

    @property
    @abstractmethod
    def epsilon(self) -> float:
        """Exploration level used by training loop for logging/control"""

    @abstractmethod
    def act(self, obs: np.ndarray, greedy: bool = False) -> int:
        """Select a flat action index for one observation."""

    @abstractmethod
    def reset_hidden(self, batch_size: int = 1) -> None:
        """Reset any recurrent state at epsilon boundaries."""

    @abstractmethod
    def observe(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Consume one transition from the environment"""

    @abstractmethod
    def train_step(self) -> Optional[float]:
        """Perform a single training step and return the loss (if any)"""

    @abstractmethod
    def state_dict(self) -> dict:
        """Return a dictionary of the agent's state for checkpointing."""

    @abstractmethod
    def load_state_dict(self, state_dict: dict) -> None:
        """Load the agent's state from a checkpoint dictionary."""