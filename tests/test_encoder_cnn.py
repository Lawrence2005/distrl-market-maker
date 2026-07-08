"""
tests/test_encoder_cnn.py

Unit tests for encoders/cnn.py.

Run with:
    pytest tests/test_encoder_cnn.py -v
"""

import pytest
import torch
from encoders.cnn import CNNEncoder


N_LEVELS   = 10
INPUT_DIM  = 2 * N_LEVELS   # = 20
LATENT_DIM = 32
BATCH      = 4
SEQ_LEN    = 30


@pytest.fixture
def enc():
    return CNNEncoder(n_levels=N_LEVELS, latent_dim=LATENT_DIM)


# ── Construction ──────────────────────────────────────────────────────────────

class TestCNNConstruction:
    def test_default_construction(self):
        enc = CNNEncoder()
        assert enc.n_levels == 10
        assert enc.latent_dim == 32
        assert enc.input_dim == 20

    def test_custom_params(self):
        enc = CNNEncoder(n_levels=5, conv1_channels=16,
                         conv2_channels=8, latent_dim=16, kernel_size=3)
        assert enc.latent_dim == 16
        assert enc.input_dim == 10

    def test_latent_dim_property(self, enc):
        assert enc.latent_dim == LATENT_DIM

    def test_input_dim_property(self, enc):
        assert enc.input_dim == INPUT_DIM

    def test_has_parameters(self, enc):
        """CNN encoder has learnable parameters."""
        assert sum(p.numel() for p in enc.parameters()) > 0

    def test_even_kernel_raises(self):
        with pytest.raises(AssertionError):
            CNNEncoder(kernel_size=2)

    def test_repr(self, enc):
        r = repr(enc)
        assert "CNNEncoder" in r
        assert str(LATENT_DIM) in r

    def test_from_config(self):
        cfg = {
            "n_levels": 10,
            "conv1_channels": 32,
            "conv2_channels": 16,
            "latent_dim": 32,
            "kernel_size": 3,
        }
        enc = CNNEncoder.from_config(cfg)
        assert enc.latent_dim == 32
        assert enc.input_dim == 20

    def test_from_config_defaults(self):
        enc = CNNEncoder.from_config({})
        assert enc.latent_dim == 32


# ── Single-step forward ───────────────────────────────────────────────────────

class TestCNNSingleStep:
    def test_output_shape(self, enc):
        x = torch.randn(BATCH, INPUT_DIM)
        z = enc(x)
        assert z.shape == (BATCH, LATENT_DIM)

    def test_output_dtype_float32(self, enc):
        x = torch.randn(BATCH, INPUT_DIM).double()
        z = enc(x)
        assert z.dtype == torch.float32

    def test_batch_size_1(self, enc):
        x = torch.randn(1, INPUT_DIM)
        assert enc(x).shape == (1, LATENT_DIM)

    def test_large_batch(self, enc):
        x = torch.randn(256, INPUT_DIM)
        assert enc(x).shape == (256, LATENT_DIM)

    def test_output_not_equal_input(self, enc):
        """CNN compresses — output must differ from input."""
        x = torch.randn(BATCH, INPUT_DIM)
        z = enc(x)
        assert z.shape != x.shape or not torch.allclose(z, x[:, :LATENT_DIM])

    def test_different_inputs_different_outputs(self, enc):
        x1 = torch.randn(BATCH, INPUT_DIM)
        x2 = torch.randn(BATCH, INPUT_DIM)
        assert not torch.allclose(enc(x1), enc(x2))


# ── Sequence forward ──────────────────────────────────────────────────────────

class TestCNNSequence:
    def test_sequence_output_shape(self, enc):
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        z = enc(x)
        assert z.shape == (BATCH, SEQ_LEN, LATENT_DIM)

    def test_sequence_single_step(self, enc):
        x = torch.randn(BATCH, 1, INPUT_DIM)
        assert enc(x).shape == (BATCH, 1, LATENT_DIM)

    def test_sequence_consistency(self, enc):
        """
        Encoding each step individually must equal encoding as sequence.
        Verifies that sequence reshape logic is correct.
        """
        enc.eval()
        x = torch.randn(2, SEQ_LEN, INPUT_DIM)
        z_seq = enc(x)                          # (2, T, latent_dim)
        z_ind = torch.stack(
            [enc(x[:, t, :]) for t in range(SEQ_LEN)], dim=1
        )                                       # (2, T, latent_dim)
        assert torch.allclose(z_seq, z_ind, atol=1e-5)


# ── Latent dim ablation ───────────────────────────────────────────────────────

class TestCNNLatentDimAblation:
    @pytest.mark.parametrize("latent_dim", [8, 16, 32])
    def test_output_shape_per_latent_dim(self, latent_dim):
        enc = CNNEncoder(latent_dim=latent_dim)
        x = torch.randn(BATCH, INPUT_DIM)
        assert enc(x).shape == (BATCH, latent_dim)

    @pytest.mark.parametrize("latent_dim", [8, 16, 32])
    def test_latent_dim_property_per_variant(self, latent_dim):
        enc = CNNEncoder(latent_dim=latent_dim)
        assert enc.latent_dim == latent_dim


# ── Gradient flow ─────────────────────────────────────────────────────────────

class TestCNNGradients:
    def test_gradients_flow_through(self, enc):
        """CNN is trained end-to-end — gradients must reach encoder."""
        x = torch.randn(BATCH, INPUT_DIM, requires_grad=True)
        z = enc(x)
        z.sum().backward()
        assert x.grad is not None
        assert not torch.all(x.grad == 0)

    def test_encoder_params_get_gradients(self, enc):
        x = torch.randn(BATCH, INPUT_DIM)
        z = enc(x)
        z.sum().backward()
        for p in enc.parameters():
            assert p.grad is not None

    def test_different_inputs_different_grads(self, enc):
        x1 = torch.randn(BATCH, INPUT_DIM, requires_grad=True)
        x2 = torch.randn(BATCH, INPUT_DIM, requires_grad=True)
        enc(x1).sum().backward()
        enc.zero_grad()
        enc(x2).sum().backward()
        assert not torch.allclose(x1.grad, x2.grad)


# ── Same-padding correctness ──────────────────────────────────────────────────

class TestCNNSamePadding:
    @pytest.mark.parametrize("kernel_size", [3, 5, 7])
    def test_latent_dim_independent_of_kernel(self, kernel_size):
        """
        Same-padding preserves spatial length so latent_dim is
        independent of kernel_size.
        """
        enc = CNNEncoder(latent_dim=16, kernel_size=kernel_size)
        x = torch.randn(BATCH, INPUT_DIM)
        assert enc(x).shape == (BATCH, 16)