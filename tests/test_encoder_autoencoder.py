"""
tests/test_encoder_autoencoder.py

Unit tests for encoders/autoencoder.py.

Uses a temporary checkpoint built from training/pretrain_ae.LOBAutoencoder
so tests don't depend on a real pre-trained checkpoint being present.

Run with:
    pytest tests/test_encoder_autoencoder.py -v
"""

import json
from pathlib import Path

import pytest
import torch
from encoders.autoencoder import AEEncoder
from encoders.base import PRIVATE_FEATURE_DIM
from training.pretrain_ae import LOBAutoencoder


MARKET_INPUT_DIM  = 20   # 2 * n_levels, n_levels=10 — what the frozen AE sees
INPUT_DIM         = MARKET_INPUT_DIM + PRIVATE_FEATURE_DIM  # = 26, what enc() takes
MARKET_LATENT_DIM = 16   # the checkpoint's bottleneck dim
LATENT_DIM        = MARKET_LATENT_DIM + PRIVATE_FEATURE_DIM  # the `.latent_dim` property
BATCH             = 4
SEQ_LEN           = 30


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_checkpoint(
    tmp_path: Path,
    latent_dim: int = MARKET_LATENT_DIM,
    input_dim:  int = MARKET_INPUT_DIM,
    filename:   str = None,
) -> Path:
    """Build a temp checkpoint matching pretrain_ae.py's save_encoder format."""
    model = LOBAutoencoder(input_dim=input_dim, latent_dim=latent_dim)
    ckpt = {
        "state_dict": model.encoder.state_dict(),
        "input_dim":  input_dim,
        "latent_dim": latent_dim,
        "metrics":    {"best_val_loss": 0.0123, "test_loss": 0.0145},
    }
    path = tmp_path / (filename or f"ae_encoder_{latent_dim}.pt")
    torch.save(ckpt, path)
    return path


@pytest.fixture
def ckpt_path(tmp_path):
    return _make_checkpoint(tmp_path)


@pytest.fixture
def enc(ckpt_path):
    return AEEncoder.from_checkpoint(ckpt_path)


# ── Checkpoint loading ─────────────────────────────────────────────────────────

class TestAECheckpointLoading:
    def test_loads_successfully(self, ckpt_path):
        enc = AEEncoder.from_checkpoint(ckpt_path)
        assert isinstance(enc, AEEncoder)

    def test_latent_dim_from_checkpoint(self, enc):
        """.latent_dim is the checkpoint's bottleneck plus the private tail
        — RecurrentBase sizes lstm.input_size off this, not off the
        checkpoint's raw stored value."""
        assert enc.latent_dim == LATENT_DIM

    def test_input_dim_from_checkpoint(self, enc):
        """input_dim is the market-only size the frozen AE was pre-trained
        on; the private tail is appended after it, not through it."""
        assert enc.input_dim == MARKET_INPUT_DIM

    def test_accepts_str_path(self, ckpt_path):
        enc = AEEncoder.from_checkpoint(str(ckpt_path))
        assert enc.latent_dim == LATENT_DIM

    def test_accepts_path_object(self, ckpt_path):
        enc = AEEncoder.from_checkpoint(ckpt_path)
        assert enc.latent_dim == LATENT_DIM

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            AEEncoder.from_checkpoint(tmp_path / "does_not_exist.pt")

    def test_missing_file_error_mentions_pretrain_ae(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="pretrain_ae"):
            AEEncoder.from_checkpoint(tmp_path / "ae_encoder_16.pt")

    def test_corrupt_checkpoint_raises(self, tmp_path):
        bad_path = tmp_path / "corrupt.pt"
        torch.save({"state_dict": {}}, bad_path)   # missing input_dim, latent_dim
        with pytest.raises(ValueError, match="missing keys"):
            AEEncoder.from_checkpoint(bad_path)

    @pytest.mark.parametrize("latent_dim", [8, 16, 32])
    def test_loads_each_ablation_variant(self, tmp_path, latent_dim):
        path = _make_checkpoint(tmp_path, latent_dim=latent_dim)
        enc  = AEEncoder.from_checkpoint(path)
        assert enc.latent_dim == latent_dim + PRIVATE_FEATURE_DIM


# ── from_config ────────────────────────────────────────────────────────────────

class TestAEFromConfig:
    def test_from_config_matches_yaml_keys(self, ckpt_path):
        cfg = {
            "type": "autoencoder",
            "latent_dim": MARKET_LATENT_DIM,
            "pretrained_weights": str(ckpt_path),
            "freeze": True,
        }
        enc = AEEncoder.from_config(cfg)
        assert enc.latent_dim == LATENT_DIM

    def test_from_config_missing_key_raises(self):
        with pytest.raises(KeyError):
            AEEncoder.from_config({"latent_dim": 16})   # missing pretrained_weights


# ── Freezing ──────────────────────────────────────────────────────────────────

class TestAEFreezing:
    def test_no_params_require_grad(self, enc):
        for p in enc.parameters():
            assert not p.requires_grad

    def test_encoder_in_eval_mode(self, enc):
        assert not enc._encoder.training

    def test_stays_frozen_after_train_call(self, enc):
        """Calling .train() on the parent module must not unfreeze encoder."""
        enc.train()
        assert not enc._encoder.training
        for p in enc.parameters():
            assert not p.requires_grad

    def test_output_has_no_grad_fn(self, enc):
        """
        Frozen encoder output should have no grad_fn at all (not just
        zero gradients) since every parameter has requires_grad=False
        and the input itself doesn't require grad either.
        """
        x = torch.randn(BATCH, INPUT_DIM)   # requires_grad=False by default
        z = enc(x)
        assert not z.requires_grad
        assert z.grad_fn is None


# ── Single-step forward ───────────────────────────────────────────────────────

class TestAESingleStep:
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

    def test_deterministic_output(self, enc):
        """Frozen encoder must give identical output for identical input."""
        x = torch.randn(BATCH, INPUT_DIM)
        z1 = enc(x)
        z2 = enc(x)
        assert torch.allclose(z1, z2)

    def test_private_tail_passes_through_unchanged(self, enc):
        """The last PRIVATE_FEATURE_DIM columns (inventory, quote
        distances, outstanding offsets, time remaining) must appear
        verbatim in the output — they bypass the frozen AE entirely."""
        x = torch.randn(BATCH, INPUT_DIM)
        z = enc(x)
        assert torch.allclose(z[:, -PRIVATE_FEATURE_DIM:], x[:, -PRIVATE_FEATURE_DIM:])

    def test_private_tail_changes_output_even_with_fixed_market(self, enc):
        """Regression for the bug where this tail was silently discarded —
        the agent was blind to its own inventory and time remaining."""
        market = torch.randn(BATCH, MARKET_INPUT_DIM)
        x1 = torch.cat([market, torch.zeros(BATCH, PRIVATE_FEATURE_DIM)], dim=-1)
        x2 = torch.cat([market, torch.ones(BATCH, PRIVATE_FEATURE_DIM)], dim=-1)
        assert not torch.allclose(enc(x1), enc(x2))


# ── Sequence forward ──────────────────────────────────────────────────────────

class TestAESequence:
    def test_sequence_output_shape(self, enc):
        x = torch.randn(BATCH, SEQ_LEN, INPUT_DIM)
        z = enc(x)
        assert z.shape == (BATCH, SEQ_LEN, LATENT_DIM)

    def test_sequence_consistency(self, enc):
        """Sequence encoding must equal per-step encoding."""
        x = torch.randn(2, SEQ_LEN, INPUT_DIM)
        z_seq = enc(x)
        z_ind = torch.stack(
            [enc(x[:, t, :]) for t in range(SEQ_LEN)], dim=1
        )
        assert torch.allclose(z_seq, z_ind, atol=1e-5)


# ── Repr ──────────────────────────────────────────────────────────────────────

class TestAERepr:
    def test_repr_contains_dims(self, enc):
        r = repr(enc)
        assert str(LATENT_DIM) in r
        assert str(MARKET_INPUT_DIM) in r
        assert "frozen=True" in r
