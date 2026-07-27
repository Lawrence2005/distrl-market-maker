"""
encoders/cnn.py

Convolutional encoder for LOB depth snapshots — trained end-to-end with RL.

Unlike the AE encoder (which is pre-trained and frozen), the CNN encoder
is trained jointly with the RL agent. Its weights are updated by gradients
flowing back from the agent's loss function.

Architecture (from configs/encoder/cnn.yaml)
--------------------------------------------
Input:  (B, 2*n_levels) = (B, 20) LOB depth snapshot
        ask_sizes[L1..LK] + bid_sizes[L1..LK], per-side normalised

    Reshape    → (B, 1, 2*n_levels)
    Conv1d(1 → conv1_channels, kernel_size, padding=kernel_size//2)
    ReLU
    Conv1d(conv1_channels → conv2_channels, kernel_size, padding=kernel_size//2)
    ReLU
    Flatten    → (B, conv2_channels * 2*n_levels)
    Linear(conv2_channels * 2*n_levels → latent_dim)

Output: (B, latent_dim)

Same-padding (padding = kernel_size // 2) preserves spatial length so
the flat dimension is always conv2_channels * 2*n_levels regardless of
kernel size. This makes latent_dim independent of kernel_size.

CNN vs AE encoder
-----------------
AE encoder:  pre-trained on LOB snapshots offline, frozen during RL.
             Learns a compression optimised for reconstruction.
CNN encoder: trained end-to-end with RL agent.
             Learns a compression optimised for reward maximisation.
             No pre-training step required.

Both take the same input (raw LOB depth snapshot) and expose the same
latent_dim property, so RecurrentBase treats them identically.

Ablation
--------
The CNN encoder appears in your ablation matrix as the
"snapshot encoder" alternative to the AE encoder.
Controlled via configs/encoder/cnn.yaml.

Reference
---------
Motivated by Gašperov & Kostanjčar (2021) whose parallel SGU + LOB
architecture shows that learned feature extraction improves performance.

Architecture follows Shi et al. (2019) and related LOB CNN work.
n_levels=10, conv1_channels=32, conv2_channels=16, latent_dim=32
from configs/encoder/cnn.yaml.

Week 5 deliverable.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from encoders.base import BaseEncoder


class CNNEncoder(BaseEncoder):
    """
    Convolutional LOB encoder trained end-to-end with the RL agent.

    Parameters
    ----------
    n_levels       : int — LOB depth levels per side (default 10)
    conv1_channels : int — output channels of first conv layer (default 32)
    conv2_channels : int — output channels of second conv layer (default 16)
    latent_dim     : int — output latent dimension (default 32)
    kernel_size    : int — conv kernel size (default 3, must be odd for
                           same-padding to work correctly)

    Usage
    -----
        enc = CNNEncoder()
        z   = enc(lob_snapshot)   # shape (B, 32)

        # Or from config dict matching configs/encoder/cnn.yaml:
        enc = CNNEncoder.from_config(cfg)
    """

    def __init__(
        self,
        n_levels:       int = 10,
        conv1_channels: int = 32,
        conv2_channels: int = 16,
        latent_dim:     int = 32,
        kernel_size:    int = 3,
    ):
        super().__init__()

        assert kernel_size % 2 == 1, (
            f"kernel_size must be odd for same-padding, got {kernel_size}"
        )

        self.n_levels       = n_levels
        self._latent_dim    = latent_dim
        self.input_dim      = 2 * n_levels   # ask_sizes + bid_sizes

        padding = kernel_size // 2           # same-padding: output length = input length

        self.conv = nn.Sequential(
            # (B, 1, 2*n_levels) → (B, conv1_channels, 2*n_levels)
            nn.Conv1d(1, conv1_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            # (B, conv1_channels, 2*n_levels) → (B, conv2_channels, 2*n_levels)
            nn.Conv1d(conv1_channels, conv2_channels,
                      kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
        )

        # Flat dimension after conv: conv2_channels * spatial_length
        self._flat_dim = conv2_channels * self.input_dim

        self.proj = nn.Linear(self._flat_dim, latent_dim)

    @property
    def latent_dim(self) -> int:
        """
        Output dimension fed to the LSTM.
        RecurrentBase reads this to set lstm.input_size.
        """
        return self._latent_dim

    def _encode_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Single-step encoding: (B, input_dim) → (B, latent_dim)."""
        h = x.unsqueeze(1)           # (B, 1, input_dim)
        h = self.conv(h)             # (B, conv2_channels, input_dim)
        h = h.flatten(start_dim=1)  # (B, flat_dim)
        return self.proj(h)          # (B, latent_dim)

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict) -> "CNNEncoder":
        """
        Instantiate from a config dict matching configs/encoder/cnn.yaml.

        Parameters
        ----------
        cfg : dict — keys: n_levels, conv1_channels, conv2_channels,
                     latent_dim, kernel_size

        Example
        -------
            import yaml
            with open("configs/encoder/cnn.yaml") as f:
                cfg = yaml.safe_load(f)
            enc = CNNEncoder.from_config(cfg)
        """
        return cls(
            n_levels       = cfg.get("n_levels",       10),
            conv1_channels = cfg.get("conv1_channels", 32),
            conv2_channels = cfg.get("conv2_channels", 16),
            latent_dim     = cfg.get("latent_dim",     32),
            kernel_size    = cfg.get("kernel_size",     3),
        )

    def __repr__(self) -> str:
        return (
            f"CNNEncoder("
            f"n_levels={self.n_levels}, "
            f"input_dim={self.input_dim}, "
            f"latent_dim={self.latent_dim}, "
            f"flat_dim={self._flat_dim})"
        )
