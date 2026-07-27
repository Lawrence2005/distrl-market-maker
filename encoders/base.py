from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

class BaseEncoder(nn.Module, ABC):
    """Shared encoder interface plus generic 2D/3D batch handling"""

    @property
    @abstractmethod
    def latent_dim(self) -> int:
        """Output dimension fed to the LSTM"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode either a 2D batch (B, input_dim) or a 3D sequence batch (B, seq_len, input_dim)"""
        x = x.float()
        if x.dim() == 3:
            batch_size, seq_len, input_dim = x.shape
            z = self._encode_batch(x.reshape(batch_size * seq_len, input_dim))
            return z.reshape(batch_size, seq_len, self.latent_dim)
        return self._encode_batch(x)

    @abstractmethod
    def _encode_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Encode a 2D batch shaped (B, input_dim)"""