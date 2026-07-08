"""
tests/test_encoder_handcrafted.py

Unit tests for encoders/handcrafted.py.

Run with:
    pytest tests/test_encoder_handcrafted.py -v
"""

import pytest
import torch
from encoders.handcrafted import HandcraftedEncoder


OBS_DIM   = 18   # 6 + 2*n_lob_levels + 6, n_lob_levels=3
BATCH     = 4
SEQ_LEN   = 30


@pytest.fixture
def enc():
    return HandcraftedEncoder(obs_dim=OBS_DIM)


# ── Construction ──────────────────────────────────────────────────────────────

class TestHandcraftedConstruction:
    def test_default_obs_dim(self):
        enc = HandcraftedEncoder()
        assert enc.obs_dim == 18

    def test_custom_obs_dim(self):
        enc = HandcraftedEncoder(obs_dim=32)
        assert enc.obs_dim == 32

    def test_latent_dim_equals_obs_dim(self, enc):
        """latent_dim must equal obs_dim — no compression."""
        assert enc.latent_dim == OBS_DIM

    def test_no_parameters(self, enc):
        """Identity encoder has no learnable parameters."""
        assert sum(p.numel() for p in enc.parameters()) == 0

    def test_repr(self, enc):
        r = repr(enc)
        assert "HandcraftedEncoder" in r
        assert str(OBS_DIM) in r


# ── Single-step forward ───────────────────────────────────────────────────────

class TestHandcraftedSingleStep:
    def test_output_shape(self, enc):
        x = torch.randn(BATCH, OBS_DIM)
        z = enc(x)
        assert z.shape == (BATCH, OBS_DIM)

    def test_output_equals_input(self, enc):
        """Identity: output must be bit-identical to input."""
        x = torch.randn(BATCH, OBS_DIM)
        z = enc(x)
        assert torch.allclose(z, x.float())

    def test_output_dtype_float32(self, enc):
        x = torch.randn(BATCH, OBS_DIM).double()
        z = enc(x)
        assert z.dtype == torch.float32

    def test_batch_size_1(self, enc):
        x = torch.randn(1, OBS_DIM)
        assert enc(x).shape == (1, OBS_DIM)

    def test_large_batch(self, enc):
        x = torch.randn(256, OBS_DIM)
        assert enc(x).shape == (256, OBS_DIM)


# ── Sequence forward ──────────────────────────────────────────────────────────

class TestHandcraftedSequence:
    def test_sequence_output_shape(self, enc):
        x = torch.randn(BATCH, SEQ_LEN, OBS_DIM)
        z = enc(x)
        assert z.shape == (BATCH, SEQ_LEN, OBS_DIM)

    def test_sequence_output_equals_input(self, enc):
        x = torch.randn(BATCH, SEQ_LEN, OBS_DIM)
        z = enc(x)
        assert torch.allclose(z, x.float())

    def test_single_step_seq(self, enc):
        """T=1 sequence should work."""
        x = torch.randn(BATCH, 1, OBS_DIM)
        assert enc(x).shape == (BATCH, 1, OBS_DIM)


# ── Gradient flow ─────────────────────────────────────────────────────────────

class TestHandcraftedGradients:
    def test_gradients_flow_through(self, enc):
        """Gradients must pass through — downstream LSTM can learn."""
        x = torch.randn(BATCH, OBS_DIM, requires_grad=True)
        z = enc(x)
        z.sum().backward()
        assert x.grad is not None

    def test_no_encoder_grad(self, enc):
        """Encoder itself has no parameters to update."""
        x = torch.randn(BATCH, OBS_DIM, requires_grad=True)
        z = enc(x)
        z.sum().backward()
        # No encoder parameters → nothing to check, just confirm no error
        assert True


# ── latent_dim contract ───────────────────────────────────────────────────────

class TestHandcraftedLatentDim:
    @pytest.mark.parametrize("obs_dim", [18, 12, 24, 32])
    def test_latent_dim_tracks_obs_dim(self, obs_dim):
        enc = HandcraftedEncoder(obs_dim=obs_dim)
        assert enc.latent_dim == obs_dim

    def test_output_dim_matches_latent_dim(self, enc):
        x = torch.randn(BATCH, OBS_DIM)
        z = enc(x)
        assert z.shape[-1] == enc.latent_dim