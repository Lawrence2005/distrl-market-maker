"""
tests/test_recurrent_base.py

Unit tests for agents/recurrent_base.py (RecurrentBase, DuelingHead).

Run with:
    pytest tests/test_recurrent_base.py -v
"""

import pytest
import torch
from agents.recurrent_base import RecurrentBase, DuelingHead
from encoders.handcrafted import HandcraftedEncoder
from encoders.cnn import CNNEncoder


OBS_DIM    = 18
N_ACTIONS  = 121     # MultiDiscrete([11,11]) flattened, or however you encode
HIDDEN_DIM = 128
BATCH      = 4
SEQ_LEN    = 30


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def handcrafted_enc():
    return HandcraftedEncoder(obs_dim=OBS_DIM)


@pytest.fixture
def base(handcrafted_enc):
    return RecurrentBase(
        encoder=handcrafted_enc, n_actions=N_ACTIONS, hidden_dim=HIDDEN_DIM
    )


# ═══════════════════════════════════════════════════════════════════════════════
# DuelingHead
# ═══════════════════════════════════════════════════════════════════════════════

class TestDuelingHead:
    def test_output_shape(self):
        head = DuelingHead(input_dim=HIDDEN_DIM, n_actions=N_ACTIONS)
        h = torch.randn(BATCH, HIDDEN_DIM)
        q = head(h)
        assert q.shape == (BATCH, N_ACTIONS)

    def test_advantage_mean_subtracted(self):
        """
        Q(s,a) = V(s) + A(s,a) - mean(A) implies mean_a(Q - V) ≈ 0.
        Verify this identity holds for the dueling decomposition.
        """
        head = DuelingHead(input_dim=HIDDEN_DIM, n_actions=N_ACTIONS)
        h = torch.randn(BATCH, HIDDEN_DIM)
        V = head.value_stream(h)
        Q = head(h)
        # mean_a(Q) should equal V (since mean_a(A - mean(A)) = 0)
        assert torch.allclose(Q.mean(dim=-1, keepdim=True), V, atol=1e-5)

    def test_different_states_different_q(self):
        head = DuelingHead(input_dim=HIDDEN_DIM, n_actions=N_ACTIONS)
        h1 = torch.randn(BATCH, HIDDEN_DIM)
        h2 = torch.randn(BATCH, HIDDEN_DIM)
        assert not torch.allclose(head(h1), head(h2))

    def test_small_n_actions(self):
        """n_actions smaller than input_dim//2 should not break mid-layer sizing."""
        head = DuelingHead(input_dim=128, n_actions=3)
        h = torch.randn(BATCH, 128)
        assert head(h).shape == (BATCH, 3)

    def test_gradients_flow(self):
        head = DuelingHead(input_dim=HIDDEN_DIM, n_actions=N_ACTIONS)
        h = torch.randn(BATCH, HIDDEN_DIM, requires_grad=True)
        head(h).sum().backward()
        assert h.grad is not None
        for p in head.parameters():
            assert p.grad is not None


# ═══════════════════════════════════════════════════════════════════════════════
# RecurrentBase — construction
# ═══════════════════════════════════════════════════════════════════════════════

class TestRecurrentBaseConstruction:
    def test_default_construction(self, handcrafted_enc):
        base = RecurrentBase(encoder=handcrafted_enc, n_actions=N_ACTIONS)
        assert base.hidden_dim == 128
        assert base.num_layers == 1

    def test_lstm_input_size_matches_encoder_latent_dim(self, handcrafted_enc):
        base = RecurrentBase(encoder=handcrafted_enc, n_actions=N_ACTIONS)
        assert base.lstm.input_size == handcrafted_enc.latent_dim

    def test_lstm_input_size_with_cnn_encoder(self):
        enc  = CNNEncoder(latent_dim=32)
        base = RecurrentBase(encoder=enc, n_actions=N_ACTIONS)
        assert base.lstm.input_size == 32

    def test_dueling_head_created_by_default(self, base):
        assert base.head is not None
        assert isinstance(base.head, DuelingHead)

    def test_dueling_false_no_head(self, handcrafted_enc):
        base = RecurrentBase(
            encoder=handcrafted_enc, n_actions=N_ACTIONS, dueling=False
        )
        assert base.head is None

    def test_custom_hidden_dim(self, handcrafted_enc):
        base = RecurrentBase(
            encoder=handcrafted_enc, n_actions=N_ACTIONS, hidden_dim=256
        )
        assert base.hidden_dim == 256
        assert base.lstm.hidden_size == 256

    def test_custom_num_layers(self, handcrafted_enc):
        base = RecurrentBase(
            encoder=handcrafted_enc, n_actions=N_ACTIONS, num_layers=2
        )
        assert base.lstm.num_layers == 2

    def test_initial_hidden_is_none(self, base):
        assert base.get_hidden() is None

    def test_count_parameters_keys(self, base):
        counts = base.count_parameters()
        assert set(counts.keys()) == {"encoder", "lstm", "head", "total"}

    def test_count_parameters_total_is_sum(self, base):
        counts = base.count_parameters()
        assert counts["total"] == sum(p.numel() for p in base.parameters())

    def test_repr(self, base):
        r = repr(base)
        assert "RecurrentBase" in r
        assert "DuelingHead" in r


# ═══════════════════════════════════════════════════════════════════════════════
# RecurrentBase — hidden state management
# ═══════════════════════════════════════════════════════════════════════════════

class TestHiddenStateManagement:
    def test_reset_hidden_creates_zero_state(self, base):
        base.reset_hidden(batch_size=BATCH)
        h, c = base.get_hidden()
        assert torch.all(h == 0)
        assert torch.all(c == 0)

    def test_reset_hidden_shape(self, base):
        base.reset_hidden(batch_size=BATCH)
        h, c = base.get_hidden()
        assert h.shape == (1, BATCH, HIDDEN_DIM)   # num_layers=1
        assert c.shape == (1, BATCH, HIDDEN_DIM)

    def test_reset_hidden_default_batch_1(self, base):
        base.reset_hidden()
        h, _ = base.get_hidden()
        assert h.shape == (1, 1, HIDDEN_DIM)

    def test_forward_updates_hidden(self, base):
        base.reset_hidden(batch_size=BATCH)
        h_before, _ = base.get_hidden()
        x = torch.randn(BATCH, SEQ_LEN, OBS_DIM)
        base.forward(x)
        h_after, _ = base.get_hidden()
        assert not torch.allclose(h_before, h_after)

    def test_forward_without_reset_auto_initialises(self, handcrafted_enc):
        """forward() without prior reset_hidden() should auto-init to zeros."""
        base = RecurrentBase(encoder=handcrafted_enc, n_actions=N_ACTIONS)
        x = torch.randn(BATCH, SEQ_LEN, OBS_DIM)
        q, hidden = base.forward(x)
        assert q.shape == (BATCH, SEQ_LEN, N_ACTIONS)

    def test_detach_hidden_removes_grad(self, base):
        base.reset_hidden(batch_size=BATCH)
        x = torch.randn(BATCH, SEQ_LEN, OBS_DIM)
        base.forward(x)
        base.detach_hidden()
        h, c = base.get_hidden()
        assert not h.requires_grad
        assert not c.requires_grad

    def test_detach_hidden_noop_when_none(self, base):
        """detach_hidden() before any reset/forward should not crash."""
        base.detach_hidden()   # hidden is None
        assert base.get_hidden() is None

    def test_set_hidden_restores_state(self, base):
        base.reset_hidden(batch_size=BATCH)
        h = torch.randn(1, BATCH, HIDDEN_DIM)
        c = torch.randn(1, BATCH, HIDDEN_DIM)
        base.set_hidden((h, c))
        h2, c2 = base.get_hidden()
        assert torch.equal(h, h2)
        assert torch.equal(c, c2)

    def test_step_maintains_state_across_calls(self, base):
        base.reset_hidden(batch_size=1)
        base.step(torch.randn(1, OBS_DIM))
        h1, _ = base.get_hidden()
        base.step(torch.randn(1, OBS_DIM))
        h2, _ = base.get_hidden()
        assert not torch.allclose(h1, h2)

    def test_hidden_persists_across_forward_calls_in_chunk(self, base):
        """
        Simulates TBPTT: two consecutive forward calls without reset
        should produce different hidden states (state carries over).
        """
        base.reset_hidden(batch_size=BATCH)
        x1 = torch.randn(BATCH, SEQ_LEN, OBS_DIM)
        x2 = torch.randn(BATCH, SEQ_LEN, OBS_DIM)

        base.forward(x1)
        h_after_1, _ = base.get_hidden()
        base.detach_hidden()

        base.forward(x2)
        h_after_2, _ = base.get_hidden()

        assert not torch.allclose(h_after_1, h_after_2)


# ═══════════════════════════════════════════════════════════════════════════════
# RecurrentBase — forward (training, full sequence)
# ═══════════════════════════════════════════════════════════════════════════════

class TestForwardSequence:
    def test_output_shape(self, base):
        x = torch.randn(BATCH, SEQ_LEN, OBS_DIM)
        q, hidden = base.forward(x)
        assert q.shape == (BATCH, SEQ_LEN, N_ACTIONS)

    def test_hidden_shape(self, base):
        x = torch.randn(BATCH, SEQ_LEN, OBS_DIM)
        _, (h, c) = base.forward(x)
        assert h.shape == (1, BATCH, HIDDEN_DIM)
        assert c.shape == (1, BATCH, HIDDEN_DIM)

    def test_explicit_hidden_argument(self, base):
        """Passing hidden explicitly should override stored state."""
        h0 = torch.zeros(1, BATCH, HIDDEN_DIM)
        c0 = torch.zeros(1, BATCH, HIDDEN_DIM)
        x = torch.randn(BATCH, SEQ_LEN, OBS_DIM)
        q, (h, c) = base.forward(x, hidden=(h0, c0))
        assert q.shape == (BATCH, SEQ_LEN, N_ACTIONS)

    def test_single_timestep_sequence(self, base):
        x = torch.randn(BATCH, 1, OBS_DIM)
        q, _ = base.forward(x)
        assert q.shape == (BATCH, 1, N_ACTIONS)

    def test_dueling_false_returns_lstm_output(self, handcrafted_enc):
        base = RecurrentBase(
            encoder=handcrafted_enc, n_actions=N_ACTIONS, dueling=False
        )
        x = torch.randn(BATCH, SEQ_LEN, OBS_DIM)
        out, _ = base.forward(x)
        assert out.shape == (BATCH, SEQ_LEN, HIDDEN_DIM)   # raw LSTM output

    def test_gradients_flow_to_encoder(self, base):
        """Gradients must reach the encoder for end-to-end training."""
        x = torch.randn(BATCH, SEQ_LEN, OBS_DIM, requires_grad=True)
        q, _ = base.forward(x)
        q.sum().backward()
        assert x.grad is not None

    def test_gradients_flow_to_lstm(self, base):
        x = torch.randn(BATCH, SEQ_LEN, OBS_DIM)
        q, _ = base.forward(x)
        q.sum().backward()
        for p in base.lstm.parameters():
            assert p.grad is not None


# ═══════════════════════════════════════════════════════════════════════════════
# RecurrentBase — step (inference, single timestep)
# ═══════════════════════════════════════════════════════════════════════════════

class TestStep:
    def test_output_shape(self, base):
        base.reset_hidden(batch_size=1)
        q = base.step(torch.randn(1, OBS_DIM))
        assert q.shape == (1, N_ACTIONS)

    def test_step_equals_forward_single_step(self, base):
        """step() must be equivalent to forward() with T=1, same hidden state."""
        base.reset_hidden(batch_size=1)
        x = torch.randn(1, OBS_DIM)

        # Save state before either call
        h0, c0 = base.get_hidden()

        q_step = base.step(x)

        base.set_hidden((h0.clone(), c0.clone()))
        q_fwd, _ = base.forward(x.unsqueeze(1))
        q_fwd = q_fwd.squeeze(1)

        assert torch.allclose(q_step, q_fwd, atol=1e-5)

    def test_repeated_steps_different_outputs(self, base):
        """Same input at different hidden states should give different Q."""
        base.reset_hidden(batch_size=1)
        x = torch.randn(1, OBS_DIM)
        q1 = base.step(x)
        q2 = base.step(x)   # same input, but hidden state has advanced
        assert not torch.allclose(q1, q2)


# ═══════════════════════════════════════════════════════════════════════════════
# Encoder swap — RecurrentBase works identically across encoder types
# ═══════════════════════════════════════════════════════════════════════════════

class TestEncoderSwap:
    @pytest.fixture(params=["handcrafted", "cnn"])
    def base_with_encoder(self, request):
        if request.param == "handcrafted":
            enc       = HandcraftedEncoder(obs_dim=OBS_DIM)
            input_dim = OBS_DIM
        else:
            enc       = CNNEncoder(latent_dim=32)
            input_dim = enc.input_dim   # = 20
        base = RecurrentBase(encoder=enc, n_actions=N_ACTIONS)
        return base, input_dim

    def test_forward_works_for_any_encoder(self, base_with_encoder):
        base, input_dim = base_with_encoder
        x = torch.randn(BATCH, SEQ_LEN, input_dim)
        q, _ = base.forward(x)
        assert q.shape == (BATCH, SEQ_LEN, N_ACTIONS)

    def test_step_works_for_any_encoder(self, base_with_encoder):
        base, input_dim = base_with_encoder
        base.reset_hidden(batch_size=1)
        q = base.step(torch.randn(1, input_dim))
        assert q.shape == (1, N_ACTIONS)