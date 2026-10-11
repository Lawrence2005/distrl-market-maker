from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

# Private features appended (raw, already normalised) after the market
# encoding for the CNN and autoencoder encoders: inventory, active bid
# distance, active ask distance, outstanding bid offset, outstanding ask
# offset, time remaining. These are envs/lob_env.py::_get_obs()'s obs[12:18]
# — the same six features the handcrafted encoder already carries through
# unchanged. Added so CNN/AE agents are not blind to their own inventory,
# quote positions, and time remaining (see training/rollout.py's
# get_encoder_input() for where this slice is assembled).
PRIVATE_FEATURE_DIM = 6


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