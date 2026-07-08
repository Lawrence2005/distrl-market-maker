"""
encoders/autoencoder.py

Frozen pre-trained autoencoder encoder for use as RL feature extractor.

Wraps the LOBEncoder from training/pretrain_ae.py, loads weights from a
checkpoint produced by training/pretrain_ae.py, and freezes all parameters.
The decoder is discarded — only the encoder forward pass is used at
agent inference time.

Pre-training
------------
Run training/pretrain_ae.py before using this encoder:

    python training/pretrain_ae.py --latent_dim 16
    # Saves: checkpoints/ae_encoder_16.pt

Then instantiate this encoder:

    enc = AEEncoder.from_checkpoint("checkpoints/ae_encoder_16.pt")
    z   = enc(lob_snapshot)   # shape (B, 16)

Input
-----
The AE was trained on LOB depth vectors produced by process_lobster.py:
    shape (B, 2*n_levels) = (B, 20) for n_levels=10

These are NOT the same as the handcrafted obs vector (18-dim). The AE
encoder expects raw LOB depth snapshots — ask_sizes[L1..LK] + bid_sizes[L1..LK]
— as produced by _parse_abides_step()["lob_snapshot"], not _get_obs().

In the RL training loop, the LOB snapshot must be extracted separately
from the info dict and passed to this encoder. The handcrafted features
from _get_obs() are NOT fed to the AE encoder.

latent_dim property returns the bottleneck dimension (8, 16, or 32)
so that RecurrentBase can set lstm.input_size dynamically.

Ablation
--------
Three variants, each with its own checkpoint:
    checkpoints/ae_encoder_8.pt   → latent_dim=8
    checkpoints/ae_encoder_16.pt  → latent_dim=16  (default)
    checkpoints/ae_encoder_32.pt  → latent_dim=32

Reference
---------
Architecture: Conv1d(1→32) → ReLU → Conv1d(32→16) → ReLU → Linear(latent_dim)
See training/pretrain_ae.py for full specification.

Week 5 deliverable.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

# Import LOBEncoder from pretrain_ae so the architecture definition
# lives in exactly one place — no duplication.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.pretrain_ae import LOBEncoder


class AEEncoder(nn.Module):
    """
    Frozen pre-trained LOB autoencoder encoder.

    Wraps LOBEncoder with:
      - Weight loading from checkpoint
      - Parameter freezing (no gradients flow through encoder)
      - Consistent latent_dim property for RecurrentBase

    Parameters
    ----------
    input_dim  : int — 2 * n_levels (default 20)
    latent_dim : int — bottleneck dimension (default 16)

    Do not instantiate directly — use AEEncoder.from_checkpoint().
    """

    def __init__(self, input_dim: int = 20, latent_dim: int = 16):
        super().__init__()
        self._encoder   = LOBEncoder(input_dim=input_dim, latent_dim=latent_dim)
        self._freeze()

    def _freeze(self) -> None:
        """Freeze all encoder parameters — no gradient updates during RL."""
        for param in self._encoder.parameters():
            param.requires_grad = False
        self._encoder.eval()

    @property
    def latent_dim(self) -> int:
        """
        Output dimension fed to the LSTM.
        RecurrentBase reads this to set lstm.input_size.
        """
        return self._encoder.latent_dim

    @property
    def input_dim(self) -> int:
        """Expected input dimension = 2 * n_levels."""
        return self._encoder.input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a LOB depth snapshot to a latent vector.

        Parameters
        ----------
        x : Tensor shape (B, input_dim) or (B, T, input_dim)
            Raw LOB depth snapshot — ask_sizes[L1..LK] + bid_sizes[L1..LK].
            NOT the handcrafted obs vector from _get_obs().

        Returns
        -------
        Tensor shape (B, latent_dim) or (B, T, latent_dim)
        """
        # Handle sequence input (B, T, input_dim) — encode each step
        if x.dim() == 3:
            B, T, D = x.shape
            z = self._encoder(x.reshape(B * T, D))   # (B*T, latent_dim)
            return z.reshape(B, T, self.latent_dim)
        return self._encoder(x.float())

    def train(self, mode: bool = True) -> "AEEncoder":
        """
        Override train() to keep encoder frozen even if agent is in train mode.
        The encoder is always in eval mode to disable dropout/batchnorm updates.
        """
        super().train(mode)
        self._encoder.eval()    # always eval — frozen
        return self

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "AEEncoder":
        """
        Load encoder weights from a checkpoint produced by pretrain_ae.py.

        Parameters
        ----------
        path : str | Path — path to ae_encoder_<latent_dim>.pt

        Returns
        -------
        AEEncoder — frozen, ready for inference

        Example
        -------
            enc = AEEncoder.from_checkpoint("checkpoints/ae_encoder_16.pt")
            z   = enc(lob_snapshot)
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"AE checkpoint not found: {path}\n"
                f"Run: python training/pretrain_ae.py --latent_dim "
                f"{_infer_latent_dim(path)}"
            )

        ckpt = torch.load(path, map_location="cpu")

        # Validate checkpoint has expected keys
        required = {"state_dict", "input_dim", "latent_dim"}
        missing  = required - set(ckpt.keys())
        if missing:
            raise ValueError(
                f"Checkpoint {path} is missing keys: {missing}. "
                f"Re-run training/pretrain_ae.py to regenerate."
            )

        enc = cls(
            input_dim  = ckpt["input_dim"],
            latent_dim = ckpt["latent_dim"],
        )
        enc._encoder.load_state_dict(ckpt["state_dict"])
        enc._freeze()

        metrics = ckpt.get("metrics", {})
        print(
            f"Loaded AE encoder: input_dim={ckpt['input_dim']}, "
            f"latent_dim={ckpt['latent_dim']}, "
            f"val_loss={metrics.get('best_val_loss', 'n/a'):.6f}"
            if metrics else
            f"Loaded AE encoder: input_dim={ckpt['input_dim']}, "
            f"latent_dim={ckpt['latent_dim']}"
        )
        return enc

    @classmethod
    def from_config(cls, cfg: dict) -> "AEEncoder":
        """
        Instantiate from a config dict matching configs/encoder/autoencoder.yaml.

        Parameters
        ----------
        cfg : dict — must contain 'pretrained_weights' and 'latent_dim'

        Example
        -------
            import yaml
            with open("configs/encoder/autoencoder.yaml") as f:
                cfg = yaml.safe_load(f)
            enc = AEEncoder.from_config(cfg)
        """
        return cls.from_checkpoint(cfg["pretrained_weights"])

    def __repr__(self) -> str:
        return (
            f"AEEncoder("
            f"input_dim={self.input_dim}, "
            f"latent_dim={self.latent_dim}, "
            f"frozen=True)"
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _infer_latent_dim(path: Path) -> int:
    """
    Try to infer latent_dim from checkpoint filename.
    ae_encoder_16.pt → 16. Falls back to 16 if pattern not matched.
    """
    stem = path.stem   # e.g. "ae_encoder_16"
    parts = stem.split("_")
    try:
        return int(parts[-1])
    except (ValueError, IndexError):
        return 16