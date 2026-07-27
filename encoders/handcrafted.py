"""
encoders/handcrafted.py

Handcrafted feature encoder — identity wrapper.

The handcrafted observation vector produced by LOBMarketMakingEnv._get_obs()
is already a normalised, ready-to-use feature vector. This encoder does
no learned compression — it passes the obs through as-is, so that the
downstream LSTM receives the same latent_z interface as the AE and CNN
variants.

Observation vector layout (LOBMarketMakingEnv, n_lob_levels=3):
    idx  0      bid-ask spread (normalised)
    idx  1      mid log-return (normalised)
    idx  2      queue imbalance ∈ [−1, 1]
    idx  3      signed volume (normalised)
    idx  4      realised volatility (normalised)
    idx  5      RSI (normalised to [−1, +1])
    idx  6–8    LOB bid depth L1–L3 (per-side proportions)
    idx  9–11   LOB ask depth L1–L3 (per-side proportions)
    idx 12      inventory q / Q_max ∈ [−1, +1]
    idx 13      active bid distance / max_offset ∈ [0, 1]
    idx 14      active ask distance / max_offset ∈ [0, 1]
    idx 15      outstanding bid offset from mid (normalised)
    idx 16      outstanding ask offset from mid (normalised)
    idx 17      time remaining τ = (T−t)/T ∈ [0, 1]
    ──────────────────────────────────────────────────────
    Total: 18 dims  (= 6 + 2·n_lob_levels + 6, with n_lob_levels=3)

latent_dim property returns obs_dim (18) so that recurrent_base.py can
set lstm.input_size dynamically without special-casing this encoder.

Reference
---------
Feature set follows Spooner et al. (2018), Huang et al. (2015),
Sun et al. (2022), and Patel (2018) — see configs/encoder/handcrafted.yaml.

Week 5 deliverable.
"""

import torch

from encoders.base import BaseEncoder


class HandcraftedEncoder(BaseEncoder):
    """
    Identity encoder for the handcrafted LOB feature vector.

    No learned parameters. Passes the normalised observation vector
    directly to the LSTM as latent_z.

    Parameters
    ----------
    obs_dim : int — dimension of the input observation vector (default 18)

    Usage
    -----
        enc = HandcraftedEncoder(obs_dim=18)
        z   = enc(obs)   # z.shape == obs.shape == (B, 18)
    """

    def __init__(self, obs_dim: int = 18):
        super().__init__()
        self.obs_dim = obs_dim

    @property
    def latent_dim(self) -> int:
        """
        Output dimension fed to the LSTM.

        For the handcrafted encoder this equals obs_dim — no compression.
        RecurrentBase reads this property to set lstm.input_size.
        """
        return self.obs_dim

    def _encode_batch(self, x: torch.Tensor) -> torch.Tensor:
        """Pass a 2D observation batch through unchanged"""
        return x.float()

    def __repr__(self) -> str:
        return f"HandcraftedEncoder(obs_dim={self.obs_dim}, latent_dim={self.latent_dim})"