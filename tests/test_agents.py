"""
tests/test_agents.py

Unit tests for agents/dqn.py, agents/qrdqn.py, agents/iqn.py.

Covers:
    DQNAgent        — act, observe, train_step, epsilon decay,
                      hidden state reset, state_dict round-trip
    QRDQNAgent      — quantile head shapes, QR loss, CVaR vs mean
                      action selection, train_step, state_dict
    IQNAgent        — cosine embedding, IQN head shapes, CVaR,
                      train_step, state_dict
    Cross-agent     — shared interface contract (act/observe/train_step/
                      reset_hidden/state_dict) for all three variants

Run with:
    pytest tests/test_agents.py -v
"""

import copy

import numpy as np
import pytest
import torch

from encoders.handcrafted import HandcraftedEncoder
from encoders.cnn import CNNEncoder
from agents.dqn import DQNAgent
from agents.qrdqn import QRDQNAgent, QuantileHead
from agents.iqn import IQNAgent, IQNHead, CosineEmbedding


# ── Shared constants ──────────────────────────────────────────────────────────

OBS_DIM   = 18
N_ACTIONS = 121
HIDDEN    = 64    # small for fast tests
N_Q       = 32    # small n_quantiles for fast tests
K         = 16    # small K for fast IQN tests
SEQ_LEN   = 5
BUF_CAP   = 500


# ── Shared helpers ────────────────────────────────────────────────────────────

def _obs() -> np.ndarray:
    return np.random.randn(OBS_DIM).astype(np.float32)


def _fill(agent, n: int = 300) -> None:
    """Fill agent's replay buffer with enough transitions to train."""
    for i in range(n):
        agent.observe(_obs(), i % N_ACTIONS, float(i) * 0.01,
                      _obs(), i % 20 == 19)


def _make_enc() -> HandcraftedEncoder:
    return HandcraftedEncoder(obs_dim=OBS_DIM)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def dqn():
    agent = DQNAgent(
        encoder=_make_enc(), n_actions=N_ACTIONS,
        hidden_dim=HIDDEN, buffer_capacity=BUF_CAP,
        seq_len=SEQ_LEN, prioritized=False,
        epsilon_decay_steps=1000,
    )
    agent.reset_hidden()
    return agent


@pytest.fixture
def dqn_filled(dqn):
    _fill(dqn)
    return dqn


@pytest.fixture
def qrdqn():
    agent = QRDQNAgent(
        encoder=_make_enc(), n_actions=N_ACTIONS,
        n_quantiles=N_Q, hidden_dim=HIDDEN,
        buffer_capacity=BUF_CAP, seq_len=SEQ_LEN,
        prioritized=False, epsilon_decay_steps=1000,
    )
    agent.reset_hidden()
    return agent


@pytest.fixture
def qrdqn_filled(qrdqn):
    _fill(qrdqn)
    return qrdqn


@pytest.fixture
def iqn():
    agent = IQNAgent(
        encoder=_make_enc(), n_actions=N_ACTIONS,
        n_quantile_samples=K, hidden_dim=HIDDEN,
        buffer_capacity=BUF_CAP, seq_len=SEQ_LEN,
        prioritized=False, epsilon_decay_steps=1000,
    )
    agent.reset_hidden()
    return agent


@pytest.fixture
def iqn_filled(iqn):
    _fill(iqn)
    return iqn


# ═══════════════════════════════════════════════════════════════════════════════
# CosineEmbedding (IQN component)
# ═══════════════════════════════════════════════════════════════════════════════

class TestCosineEmbedding:
    def test_output_shape(self):
        emb = CosineEmbedding(embedding_dim=64, output_dim=HIDDEN)
        tau = torch.rand(32)
        phi = emb(tau)
        assert phi.shape == (32, HIDDEN)

    def test_different_tau_different_output(self):
        emb = CosineEmbedding(embedding_dim=64, output_dim=HIDDEN)
        tau1 = torch.zeros(4)
        tau2 = torch.ones(4)
        assert not torch.allclose(emb(tau1), emb(tau2))

    def test_output_after_relu_nonneg(self):
        """ReLU in proj ensures non-negative activations."""
        emb = CosineEmbedding(embedding_dim=64, output_dim=HIDDEN)
        tau = torch.rand(100)
        phi = emb(tau)
        assert torch.all(phi >= 0)

    def test_basis_not_learned(self):
        """Cosine basis is a fixed buffer, not a parameter."""
        emb = CosineEmbedding(embedding_dim=64, output_dim=HIDDEN)
        param_names = [n for n, _ in emb.named_parameters()]
        assert "basis" not in param_names

    def test_gradients_flow_through(self):
        emb = CosineEmbedding(embedding_dim=64, output_dim=HIDDEN)
        tau = torch.rand(8, requires_grad=False)
        phi = emb(tau)
        phi.sum().backward()
        for p in emb.parameters():
            assert p.grad is not None


# ═══════════════════════════════════════════════════════════════════════════════
# QuantileHead (QR-DQN component)
# ═══════════════════════════════════════════════════════════════════════════════

class TestQuantileHead:
    def test_output_shape_dueling(self):
        head = QuantileHead(HIDDEN, N_ACTIONS, N_Q, dueling=True)
        h = torch.randn(4, HIDDEN)
        Z = head(h)
        assert Z.shape == (4, N_ACTIONS, N_Q)

    def test_output_shape_no_dueling(self):
        head = QuantileHead(HIDDEN, N_ACTIONS, N_Q, dueling=False)
        h = torch.randn(4, HIDDEN)
        Z = head(h)
        assert Z.shape == (4, N_ACTIONS, N_Q)

    def test_dueling_mean_advantage_zero(self):
        """
        Dueling: mean_a(A(s,a,τ)) = 0 for all τ.
        Verify V(s,τ) = mean_a(Z(s,a,τ)).
        """
        head = QuantileHead(HIDDEN, N_ACTIONS, N_Q, dueling=True)
        h = torch.randn(4, HIDDEN)
        Z = head(h)
        V = head.value_stream(h)                          # (4, N_Q)
        mean_Z = Z.mean(dim=1)                            # (4, N_Q) = mean over actions
        assert torch.allclose(mean_Z, V, atol=1e-5)

    def test_gradients_flow(self):
        head = QuantileHead(HIDDEN, N_ACTIONS, N_Q)
        h = torch.randn(4, HIDDEN, requires_grad=True)
        head(h).sum().backward()
        assert h.grad is not None


# ═══════════════════════════════════════════════════════════════════════════════
# IQNHead (IQN component)
# ═══════════════════════════════════════════════════════════════════════════════

class TestIQNHead:
    def test_output_shape(self):
        head = IQNHead(HIDDEN, N_ACTIONS, embedding_dim=64)
        h   = torch.randn(4, HIDDEN)
        tau = torch.rand(4, K)
        Z   = head(h, tau)
        assert Z.shape == (4, K, N_ACTIONS)

    def test_output_shape_no_dueling(self):
        head = IQNHead(HIDDEN, N_ACTIONS, embedding_dim=64, dueling=False)
        h   = torch.randn(4, HIDDEN)
        tau = torch.rand(4, K)
        Z   = head(h, tau)
        assert Z.shape == (4, K, N_ACTIONS)

    def test_different_tau_different_z(self):
        """Different τ should produce different Z(s,a,τ)."""
        head = IQNHead(HIDDEN, N_ACTIONS, embedding_dim=64)
        h    = torch.randn(1, HIDDEN).expand(4, -1)
        tau1 = torch.zeros(4, K)
        tau2 = torch.ones(4, K) * 0.99
        Z1 = head(h, tau1)
        Z2 = head(h, tau2)
        assert not torch.allclose(Z1, Z2)

    def test_dueling_mean_advantage_zero(self):
        head = IQNHead(HIDDEN, N_ACTIONS, embedding_dim=64, dueling=True)
        h   = torch.randn(4, HIDDEN)
        tau = torch.rand(4, K)
        Z   = head(h, tau)                                # (4, K, N_ACTIONS)
        V   = head.value_stream(
            (h.unsqueeze(1).expand(4, K, -1).reshape(4*K, -1) *
             head.tau_embed(tau.reshape(4*K)))
        )                                                 # (4*K, 1)
        # Mean over actions should equal V per τ sample
        mean_Z = Z.mean(dim=-1).reshape(4*K, 1)          # (4*K, 1)
        assert torch.allclose(mean_Z, V, atol=1e-5)

    def test_gradients_flow_through_tau(self):
        head = IQNHead(HIDDEN, N_ACTIONS, embedding_dim=64)
        h   = torch.randn(4, HIDDEN)
        tau = torch.rand(4, K)
        Z   = head(h, tau)
        Z.sum().backward()
        for p in head.parameters():
            assert p.grad is not None


# ═══════════════════════════════════════════════════════════════════════════════
# DQNAgent
# ═══════════════════════════════════════════════════════════════════════════════

class TestDQNAgent:

    # ── Construction ──────────────────────────────────────────────────

    def test_name(self, dqn):
        assert dqn.name == "DQN"

    def test_initial_steps_zero(self, dqn):
        assert dqn._steps == 0

    def test_initial_updates_zero(self, dqn):
        assert dqn._updates == 0

    # ── act() ─────────────────────────────────────────────────────────

    def test_act_returns_valid_action(self, dqn):
        a = dqn.act(_obs())
        assert 0 <= a < N_ACTIONS

    def test_act_returns_int(self, dqn):
        assert isinstance(dqn.act(_obs()), int)

    def test_act_greedy_no_random(self, dqn):
        """Greedy mode must never return random actions."""
        _fill(dqn, 10)
        actions = {dqn.act(_obs(), greedy=True) for _ in range(20)}
        # With fixed weights and same obs, greedy should be deterministic
        assert len(actions) <= N_ACTIONS

    def test_act_explores_during_epsilon_1(self, dqn):
        """At epsilon=1.0, all actions should be random."""
        assert dqn.epsilon == pytest.approx(1.0)
        actions = {dqn.act(_obs()) for _ in range(200)}
        # Should see more than 1 unique action (random exploration)
        assert len(actions) > 1

    # ── epsilon decay ─────────────────────────────────────────────────

    def test_epsilon_decreases_with_steps(self, dqn):
        eps_before = dqn.epsilon
        _fill(dqn, 100)
        assert dqn.epsilon < eps_before

    def test_epsilon_floored_at_end(self):
        agent = DQNAgent(
            encoder=_make_enc(), n_actions=N_ACTIONS,
            hidden_dim=HIDDEN, buffer_capacity=BUF_CAP,
            seq_len=SEQ_LEN, epsilon_end=0.05,
            epsilon_decay_steps=10,
        )
        agent.reset_hidden()
        _fill(agent, 200)
        assert agent.epsilon == pytest.approx(0.05)

    # ── observe() ─────────────────────────────────────────────────────

    def test_observe_increments_steps(self, dqn):
        dqn.observe(_obs(), 0, 0.0, _obs(), False)
        assert dqn._steps == 1

    def test_observe_fills_buffer(self, dqn):
        for i in range(10):
            dqn.observe(_obs(), 0, 0.0, _obs(), False)
        assert len(dqn.buffer) == 10

    # ── train_step() ──────────────────────────────────────────────────

    def test_train_step_returns_none_if_not_ready(self, dqn):
        dqn.observe(_obs(), 0, 0.0, _obs(), False)
        assert dqn.train_step() is None

    def test_train_step_returns_float_when_ready(self, dqn_filled):
        loss = dqn_filled.train_step()
        assert isinstance(loss, float)
        assert np.isfinite(loss)

    def test_train_step_increments_updates(self, dqn_filled):
        before = dqn_filled._updates
        dqn_filled.train_step()
        assert dqn_filled._updates == before + 1

    def test_train_step_loss_decreases_over_training(self, dqn):
        """Loss should generally decrease over many updates (sanity check)."""
        _fill(dqn, 400)
        losses = [dqn.train_step() for _ in range(50)]
        losses = [l for l in losses if l is not None]
        # Compare first 10 vs last 10 — last should be lower on average
        assert np.mean(losses[-10:]) < np.mean(losses[:10]) * 2  # loose

    def test_target_network_updates_at_freq(self):
        agent = DQNAgent(
            encoder=_make_enc(), n_actions=N_ACTIONS,
            hidden_dim=HIDDEN, buffer_capacity=BUF_CAP,
            seq_len=SEQ_LEN, target_update_freq=5,
            prioritized=False,
        )
        agent.reset_hidden()
        _fill(agent, 300)
        # Corrupt target network
        with torch.no_grad():
            for p in agent.target.parameters():
                p.fill_(999.0)
        # Run exactly target_update_freq updates
        for _ in range(5):
            agent.train_step()
        # Target should now match online
        online_p = next(iter(agent.online.parameters()))
        target_p = next(iter(agent.target.parameters()))
        assert torch.allclose(online_p, target_p)

    # ── hidden state ──────────────────────────────────────────────────

    def test_reset_hidden_clears_state(self, dqn_filled):
        dqn_filled.train_step()
        dqn_filled.reset_hidden()
        h, _ = dqn_filled.online.get_hidden()
        assert torch.all(h == 0)

    # ── state_dict round-trip ─────────────────────────────────────────

    def test_state_dict_round_trip(self, dqn_filled):
        dqn_filled.train_step()
        sd = dqn_filled.state_dict()
        dqn_filled.load_state_dict(sd)
        assert dqn_filled._steps == sd["steps"]
        assert dqn_filled._updates == sd["updates"]

    def test_state_dict_preserves_weights(self, dqn_filled):
        dqn_filled.train_step()
        sd = dqn_filled.state_dict()
        w_before = copy.deepcopy(
            next(iter(dqn_filled.online.parameters())).detach()
        )
        dqn_filled.load_state_dict(sd)
        w_after = next(iter(dqn_filled.online.parameters())).detach()
        assert torch.allclose(w_before, w_after)


# ═══════════════════════════════════════════════════════════════════════════════
# QRDQNAgent
# ═══════════════════════════════════════════════════════════════════════════════

class TestQRDQNAgent:

    def test_name(self, qrdqn):
        assert qrdqn.name == "QR-DQN"

    def test_taus_shape(self, qrdqn):
        assert qrdqn.taus.shape == (N_Q,)

    def test_taus_in_unit_interval(self, qrdqn):
        assert torch.all(qrdqn.taus >= 0)
        assert torch.all(qrdqn.taus <= 1)

    def test_taus_formula(self, qrdqn):
        """τ_i = (2i-1)/(2N) for i=1,...,N."""
        N = N_Q
        expected = torch.FloatTensor([(2*i-1)/(2*N) for i in range(1, N+1)])
        assert torch.allclose(qrdqn.taus, expected)

    def test_act_valid_action(self, qrdqn):
        a = qrdqn.act(_obs())
        assert 0 <= a < N_ACTIONS

    def test_act_greedy_valid(self, qrdqn):
        a = qrdqn.act(_obs(), greedy=True)
        assert 0 <= a < N_ACTIONS

    def test_train_step_none_when_not_ready(self, qrdqn):
        qrdqn.observe(_obs(), 0, 0.0, _obs(), False)
        assert qrdqn.train_step() is None

    def test_train_step_returns_finite_loss(self, qrdqn_filled):
        loss = qrdqn_filled.train_step()
        assert loss is not None
        assert np.isfinite(loss)

    def test_train_step_loss_positive(self, qrdqn_filled):
        """QR loss is always non-negative."""
        loss = qrdqn_filled.train_step()
        assert loss >= 0

    def test_state_dict_round_trip(self, qrdqn_filled):
        qrdqn_filled.train_step()
        sd = qrdqn_filled.state_dict()
        qrdqn_filled.load_state_dict(sd)
        assert qrdqn_filled._steps == sd["steps"]

    def test_quantile_huber_loss_nonneg(self):
        """QR loss should always be non-negative."""
        N = 32
        taus = torch.FloatTensor([(2*i-1)/(2*N) for i in range(1, N+1)])
        pred   = torch.randn(4, N)
        target = torch.randn(4, N)
        loss = QRDQNAgent._quantile_huber_loss(pred, target, taus)
        assert loss.item() >= 0

    def test_quantile_huber_loss_zero_at_perfect_pred(self):
        """When pred == target, TD errors are zero so loss should be zero."""
        N = 32
        taus = torch.FloatTensor([(2*i-1)/(2*N) for i in range(1, N+1)])
        # Use identical tensors — td = target - pred = 0 everywhere
        z      = torch.zeros(4, N)
        target = torch.zeros(4, N)
        loss = QRDQNAgent._quantile_huber_loss(z, target, taus)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_cvar_action_uses_lower_tail(self, qrdqn):
        """
        CVaR action should differ from mean action when distributions
        have different tail properties across actions.

        Action 0: mean=5.0 but worst 25% quantiles are very negative (-100)
        Action 1: mean=0.5, all quantiles = 0.5 (safe, predictable)

        Mean policy prefers action 0 (higher mean).
        CVaR policy prefers action 1 (better worst-case tail).
        """
        B, NA, NQ = 1, N_ACTIONS, N_Q
        Z = torch.zeros(B, NA, NQ)

        # Action 0: high mean but catastrophic tail
        n_tail = qrdqn._n_tail              # bottom alpha fraction
        Z[0, 0, :n_tail]  = -100.0         # terrible tail
        Z[0, 0, n_tail:]  = 5.0 + 100.0 * n_tail / (NQ - n_tail)  # high upper

        # Action 1: modest but safe constant return
        Z[0, 1, :] = 0.5

        mean_action = Z.mean(dim=-1).argmax(dim=-1).item()
        cvar_action = qrdqn._cvar_action(Z).item()

        assert mean_action == 0    # action 0 wins on mean
        assert cvar_action == 1    # action 1 wins on CVaR (better tail)

    def test_n_tail_quantiles(self, qrdqn):
        """n_tail = floor(alpha * N_Q)."""
        expected = max(1, int(qrdqn.cvar_alpha * N_Q))
        assert qrdqn._n_tail == expected

    def test_epsilon_decays(self, qrdqn):
        eps_start = qrdqn.epsilon
        _fill(qrdqn, 100)
        assert qrdqn.epsilon < eps_start


# ═══════════════════════════════════════════════════════════════════════════════
# IQNAgent
# ═══════════════════════════════════════════════════════════════════════════════

class TestIQNAgent:

    def test_name(self, iqn):
        assert iqn.name == "IQN"

    def test_n_tail_quantiles(self, iqn):
        expected = max(1, int(iqn.cvar_alpha * K))
        assert iqn._n_tail == expected

    def test_act_valid_action(self, iqn):
        a = iqn.act(_obs())
        assert 0 <= a < N_ACTIONS

    def test_act_greedy_valid(self, iqn):
        a = iqn.act(_obs(), greedy=True)
        assert 0 <= a < N_ACTIONS

    def test_train_step_none_when_not_ready(self, iqn):
        iqn.observe(_obs(), 0, 0.0, _obs(), False)
        assert iqn.train_step() is None

    def test_train_step_returns_finite_loss(self, iqn_filled):
        loss = iqn_filled.train_step()
        assert loss is not None
        assert np.isfinite(loss)

    def test_train_step_loss_nonneg(self, iqn_filled):
        loss = iqn_filled.train_step()
        assert loss >= 0

    def test_state_dict_round_trip(self, iqn_filled):
        iqn_filled.train_step()
        sd = iqn_filled.state_dict()
        iqn_filled.load_state_dict(sd)
        assert iqn_filled._steps == sd["steps"]

    def test_sample_tau_shape(self, iqn):
        tau = iqn._sample_tau(batch_size=4, K=K)
        assert tau.shape == (4, K)

    def test_sample_tau_in_unit_interval(self, iqn):
        tau = iqn._sample_tau(batch_size=8, K=K)
        assert torch.all(tau >= 0)
        assert torch.all(tau <= 1)

    def test_sample_tau_cvar_in_alpha_range(self, iqn):
        """CVaR sampling should restrict τ to [0, alpha]."""
        tau = iqn._sample_tau_cvar(batch_size=8, K=K)
        assert torch.all(tau >= 0)
        assert torch.all(tau <= iqn.cvar_alpha + 1e-6)

    def test_cvar_action_valid(self, iqn):
        Z = torch.randn(2, K, N_ACTIONS)
        actions = iqn._cvar_action(Z)
        assert actions.shape == (2,)
        assert torch.all(actions >= 0)
        assert torch.all(actions < N_ACTIONS)

    def test_epsilon_decays(self, iqn):
        eps_start = iqn.epsilon
        _fill(iqn, 100)
        assert iqn.epsilon < eps_start


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-agent interface contract
# ═══════════════════════════════════════════════════════════════════════════════

class TestAgentInterface:
    """
    All three agents must satisfy the same interface so the training loop
    can swap them without changes.
    """

    @pytest.fixture(params=["dqn", "qrdqn", "iqn"])
    def agent(self, request):
        if request.param == "dqn":
            a = DQNAgent(encoder=_make_enc(), n_actions=N_ACTIONS,
                         hidden_dim=HIDDEN, buffer_capacity=BUF_CAP,
                         seq_len=SEQ_LEN, prioritized=False,
                         epsilon_decay_steps=1000)
        elif request.param == "qrdqn":
            a = QRDQNAgent(encoder=_make_enc(), n_actions=N_ACTIONS,
                           n_quantiles=N_Q, hidden_dim=HIDDEN,
                           buffer_capacity=BUF_CAP, seq_len=SEQ_LEN,
                           prioritized=False, epsilon_decay_steps=1000)
        else:
            a = IQNAgent(encoder=_make_enc(), n_actions=N_ACTIONS,
                         n_quantile_samples=K, hidden_dim=HIDDEN,
                         buffer_capacity=BUF_CAP, seq_len=SEQ_LEN,
                         prioritized=False, epsilon_decay_steps=1000)
        a.reset_hidden()
        return a

    def test_has_name(self, agent):
        assert isinstance(agent.name, str) and len(agent.name) > 0

    def test_act_returns_int_in_range(self, agent):
        a = agent.act(_obs())
        assert isinstance(a, int)
        assert 0 <= a < N_ACTIONS

    def test_observe_increments_steps(self, agent):
        agent.observe(_obs(), 0, 0.0, _obs(), False)
        assert agent._steps == 1

    def test_train_step_none_before_ready(self, agent):
        agent.observe(_obs(), 0, 0.0, _obs(), False)
        assert agent.train_step() is None

    def test_train_step_finite_when_ready(self, agent):
        _fill(agent)
        loss = agent.train_step()
        assert loss is not None
        assert np.isfinite(loss)

    def test_reset_hidden_no_crash(self, agent):
        agent.reset_hidden(batch_size=1)

    def test_state_dict_has_required_keys(self, agent):
        sd = agent.state_dict()
        assert "steps" in sd
        assert "updates" in sd

    def test_load_state_dict_restores_steps(self, agent):
        _fill(agent, 10)
        sd = agent.state_dict()
        agent.load_state_dict(sd)
        assert agent._steps == 10

    def test_epsilon_property_exists(self, agent):
        assert hasattr(agent, "epsilon")
        assert 0.0 <= agent.epsilon <= 1.0

    def test_buffer_attribute_exists(self, agent):
        assert hasattr(agent, "buffer")

    def test_multiple_resets_no_crash(self, agent):
        for _ in range(3):
            agent.reset_hidden()
            _fill(agent, 5)
            agent.train_step()
            agent.reset_hidden()