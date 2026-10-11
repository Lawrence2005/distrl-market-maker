"""
tests/test_cvar.py

Unit tests for agents/cvar_policy.py — the standalone CVaR/mean/VaR action
selection functions, and the CVaRPolicy wrapper that uses them.

Tests on toy known distributions where CVaR is analytically computable.
E.g. uniform distribution U[0,1]: CVaR_0.10 = 0.05 exactly.

This file covers the action-SELECTION risk measure. The episode-level
CVaR reported in results (CVaR_0.10 of per-episode P&Ls) is a different
function, evaluation.metrics.cvar() — see tests/test_metrics.py.

Run with:
    pytest tests/test_cvar.py -v
"""

import numpy as np
import pytest
import torch

from agents.cvar_policy import CVaRPolicy, cvar_action, mean_action, var_action
from agents.dqn import DQNAgent
from agents.qrdqn import QRDQNAgent
from agents.iqn import IQNAgent
from encoders.handcrafted import HandcraftedEncoder

OBS_DIM   = 18
N_ACTIONS = 2
HIDDEN    = 32
N_Q       = 100    # large enough that alpha*N_Q lands on an exact integer
K         = 100
BUF_CAP   = 500
SEQ_LEN   = 5


def _make_enc() -> HandcraftedEncoder:
    return HandcraftedEncoder(obs_dim=OBS_DIM)


def _obs() -> np.ndarray:
    return np.random.randn(OBS_DIM).astype(np.float32)


# ── Analytical CVaR on a known uniform grid ────────────────────────────────────

class TestCVaRActionAnalytical:
    """
    The standard QR-DQN quantile midpoints tau_i = (2i-1)/(2N), i=1..N,
    ARE the quantile function of U[0,1] evaluated at those points, i.e.
    Z_(i) = tau_i for a sorted grid. So CVaR_alpha of this grid has a
    closed form: the mean of the lowest floor(alpha*N) midpoints.
    """

    @staticmethod
    def _uniform_grid(n: int) -> torch.Tensor:
        return torch.tensor([(2 * i - 1) / (2 * n) for i in range(1, n + 1)],
                             dtype=torch.float32)

    @pytest.mark.parametrize("alpha,expected", [(0.10, 0.05), (0.25, 0.125), (1.0, 0.5)])
    def test_cvar_of_uniform_grid_matches_closed_form(self, alpha, expected):
        z = self._uniform_grid(N_Q)
        Z = z.view(1, N_Q, 1)   # (B=1, n_quantiles, n_actions=1)
        action = cvar_action(Z, alpha)
        assert action.item() == 0   # only one action, selection is trivial

        # The closed form itself — computed directly from the definition,
        # not by calling cvar_action again, so this isn't circular.
        n_tail = max(1, int(alpha * N_Q))
        closed_form = z.sort().values[:n_tail].mean().item()
        assert closed_form == pytest.approx(expected, abs=1e-6)

    def test_alpha_1_equals_mean_action(self):
        """CVaR_1.0 = E[Z] — the risk-neutral limit — so cvar_action and
        mean_action must agree when alpha=1."""
        torch.manual_seed(0)
        Z = torch.randn(3, N_Q, 5)
        assert torch.equal(cvar_action(Z, alpha=1.0), mean_action(Z))


# ── Multi-action selection: CVaR vs mean disagree on tail risk ────────────────

class TestCVaRVsMeanActionSelection:
    def test_cvar_avoids_catastrophic_tail_mean_does_not(self):
        """
        Action 0: mean 5.0, but its worst 25% of outcomes are catastrophic.
        Action 1: constant 0.5 — no tail risk at all, lower mean.

        A risk-neutral (mean) policy must prefer action 0 (higher mean).
        A CVaR_0.25 policy must prefer action 1 (far better worst case).
        """
        n_tail = max(1, int(0.25 * N_Q))
        remaining = N_Q - n_tail
        high_value = (5.0 * N_Q - (-100.0) * n_tail) / remaining

        Z = torch.zeros(1, N_Q, 2)
        Z[0, :n_tail, 0] = -100.0
        Z[0, n_tail:, 0] = high_value
        Z[0, :, 1] = 0.5

        # Sanity-check the construction before asserting on it:
        assert Z[0, :, 0].mean().item() == pytest.approx(5.0, abs=1e-4)

        assert mean_action(Z).item() == 0
        assert cvar_action(Z, alpha=0.25).item() == 1

    def test_cvar_is_batched_independently(self):
        """Two batch elements with opposite catastrophic-tail actions must
        each be resolved independently."""
        n_tail = max(1, int(0.25 * N_Q))
        remaining = N_Q - n_tail
        high_value = (5.0 * N_Q - (-100.0) * n_tail) / remaining

        Z = torch.zeros(2, N_Q, 2)
        Z[0, :n_tail, 0] = -100.0
        Z[0, n_tail:, 0] = high_value
        Z[0, :, 1] = 0.5
        # Batch element 1: the catastrophic tail is on action 1 instead.
        Z[1, :n_tail, 1] = -100.0
        Z[1, n_tail:, 1] = high_value
        Z[1, :, 0] = 0.5

        actions = cvar_action(Z, alpha=0.25)
        assert actions[0].item() == 1
        assert actions[1].item() == 0


# ── mean_action ─────────────────────────────────────────────────────────────

class TestMeanAction:
    def test_picks_highest_mean(self):
        Z = torch.zeros(1, N_Q, 3)
        Z[0, :, 0] = 1.0
        Z[0, :, 1] = 9.0
        Z[0, :, 2] = 4.0
        assert mean_action(Z).item() == 1

    def test_ignores_distribution_shape_only_mean_matters(self):
        """A wide, risky distribution with the same mean as a flat one must
        be indistinguishable to mean_action."""
        flat = torch.full((1, N_Q, 1), 5.0)
        risky = torch.zeros(1, N_Q, 1)
        risky[0, : N_Q // 2, 0] = 0.0
        risky[0, N_Q // 2 :, 0] = 10.0
        Z = torch.cat([flat, risky], dim=-1)
        m = Z.mean(dim=1)
        assert m[0, 0].item() == pytest.approx(m[0, 1].item(), abs=1e-4)


# ── var_action ──────────────────────────────────────────────────────────────

class TestVarAction:
    def test_var_of_simple_grid_matches_closed_form(self):
        n = 10
        values = torch.arange(1, n + 1, dtype=torch.float32)  # 1..10, sorted
        alpha = 0.3
        idx = max(0, int(alpha * n) - 1)
        expected_var = values[idx].item()
        assert expected_var == pytest.approx(3.0)

        # action 0 has this VaR (3.0); action 1 is flat at 2.5 (lower VaR) —
        # var_action must prefer action 0.
        Z = torch.stack([values, torch.full((n,), 2.5)], dim=-1).unsqueeze(0)
        assert var_action(Z, alpha=alpha).item() == 0

    def test_var_prefers_higher_tail_even_with_lower_mean(self):
        """VaR looks at one order statistic, not the mean — a distribution
        with a lower mean but a higher alpha-quantile must still win."""
        n = N_Q
        alpha = 0.1
        idx = max(0, int(alpha * n) - 1)

        # Action 0: low VaR, high mean (long right tail).
        a0 = torch.zeros(n)
        a0[: idx + 1] = -50.0
        a0[idx + 1 :] = 10.0
        # Action 1: everything flat at 1.0 — VaR = 1.0, which beats
        # action 0's VaR (-50.0), despite action 0's higher mean.
        a1 = torch.full((n,), 1.0)

        assert a0.mean().item() > a1.mean().item()
        Z = torch.stack([a0, a1], dim=-1).unsqueeze(0)
        assert var_action(Z, alpha=alpha).item() == 1


# ── CVaRPolicy construction ───────────────────────────────────────────────────

class TestCVaRPolicyConstruction:
    def _qrdqn(self):
        agent = QRDQNAgent(encoder=_make_enc(), n_actions=N_ACTIONS, n_quantiles=N_Q,
                            hidden_dim=HIDDEN, buffer_capacity=BUF_CAP, seq_len=SEQ_LEN,
                            prioritized=False, epsilon_decay_steps=1000)
        agent.reset_hidden()
        return agent

    def test_rejects_non_distributional_agent(self):
        dqn_agent = DQNAgent(encoder=_make_enc(), n_actions=N_ACTIONS, hidden_dim=HIDDEN,
                              buffer_capacity=BUF_CAP, seq_len=SEQ_LEN, prioritized=False)
        with pytest.raises(AssertionError, match="QRDQNAgent or IQNAgent"):
            CVaRPolicy(agent=dqn_agent)

    @pytest.mark.parametrize("bad_alpha", [0.0, -0.1, 1.5])
    def test_rejects_alpha_out_of_range(self, bad_alpha):
        with pytest.raises(AssertionError, match="alpha"):
            CVaRPolicy(agent=self._qrdqn(), alpha=bad_alpha)

    def test_rejects_unknown_measure(self):
        with pytest.raises(AssertionError, match="Unknown risk measure"):
            CVaRPolicy(agent=self._qrdqn(), measure="bogus")

    def test_accepts_each_known_measure(self):
        for measure in ("cvar", "mean", "var"):
            CVaRPolicy(agent=self._qrdqn(), measure=measure)

    def test_name_includes_measure_and_alpha(self):
        wrapped = CVaRPolicy(agent=self._qrdqn(), alpha=0.1, measure="cvar")
        assert "cvar" in wrapped.name
        assert "0.1" in wrapped.name

    def test_transparent_attribute_passthrough(self):
        """cvar_agent.epsilon, .n_actions, etc. must reach the base agent."""
        agent = self._qrdqn()
        wrapped = CVaRPolicy(agent=agent)
        assert wrapped.n_actions == agent.n_actions
        assert wrapped.epsilon == agent.epsilon


# ── CVaRPolicy.act — QR-DQN ───────────────────────────────────────────────────

class TestCVaRPolicyActQRDQN:
    def _wrapped(self, alpha=0.25, measure="cvar"):
        agent = QRDQNAgent(encoder=_make_enc(), n_actions=N_ACTIONS, n_quantiles=N_Q,
                            hidden_dim=HIDDEN, buffer_capacity=BUF_CAP, seq_len=SEQ_LEN,
                            prioritized=False, epsilon_start=0.0, epsilon_end=0.0,
                            epsilon_decay_steps=1)
        agent.reset_hidden()
        return CVaRPolicy(agent=agent, alpha=alpha, measure=measure)

    @staticmethod
    def _catastrophic_tail_Z():
        n_tail = max(1, int(0.25 * N_Q))
        remaining = N_Q - n_tail
        high_value = (5.0 * N_Q - (-100.0) * n_tail) / remaining
        Z = torch.zeros(1, N_Q, N_ACTIONS)
        Z[0, :n_tail, 0] = -100.0
        Z[0, n_tail:, 0] = high_value
        Z[0, :, 1] = 0.5
        return Z

    def test_cvar_measure_avoids_catastrophic_tail(self, monkeypatch):
        wrapped = self._wrapped(alpha=0.25, measure="cvar")
        monkeypatch.setattr(wrapped, "_get_z", lambda x: self._catastrophic_tail_Z())
        assert wrapped.act(_obs(), greedy=True) == 1

    def test_mean_measure_prefers_higher_average(self, monkeypatch):
        wrapped = self._wrapped(alpha=0.25, measure="mean")
        monkeypatch.setattr(wrapped, "_get_z", lambda x: self._catastrophic_tail_Z())
        assert wrapped.act(_obs(), greedy=True) == 0

    def test_greedy_bypasses_epsilon(self, monkeypatch):
        """epsilon is a read-only property derived from the decay schedule —
        force it to a constant 1.0 via the schedule, not by assignment."""
        agent = QRDQNAgent(encoder=_make_enc(), n_actions=N_ACTIONS, n_quantiles=N_Q,
                            hidden_dim=HIDDEN, buffer_capacity=BUF_CAP, seq_len=SEQ_LEN,
                            prioritized=False, epsilon_start=1.0, epsilon_end=1.0)
        agent.reset_hidden()
        wrapped = CVaRPolicy(agent=agent, alpha=0.25, measure="cvar")
        assert wrapped.epsilon == 1.0   # would explore ~always if not greedy
        Z = torch.zeros(1, N_Q, N_ACTIONS)
        Z[0, :, 0] = 1.0
        monkeypatch.setattr(wrapped, "_get_z", lambda x: Z)
        actions = {wrapped.act(_obs(), greedy=True) for _ in range(20)}
        assert actions == {0}   # deterministic despite epsilon=1.0

    def test_returns_plain_int(self, monkeypatch):
        wrapped = self._wrapped()
        monkeypatch.setattr(wrapped, "_get_z", lambda x: self._catastrophic_tail_Z())
        action = wrapped.act(_obs(), greedy=True)
        assert isinstance(action, int)


# ── CVaRPolicy.act — IQN ──────────────────────────────────────────────────────

class TestCVaRPolicyActIQN:
    def _wrapped(self, alpha=0.25, measure="cvar"):
        agent = IQNAgent(encoder=_make_enc(), n_actions=N_ACTIONS, n_quantile_samples=K,
                          hidden_dim=HIDDEN, buffer_capacity=BUF_CAP, seq_len=SEQ_LEN,
                          prioritized=False, epsilon_start=0.0, epsilon_end=0.0,
                          epsilon_decay_steps=1)
        agent.reset_hidden()
        return CVaRPolicy(agent=agent, alpha=alpha, measure=measure)

    def test_wrapper_samples_full_range_not_a_restricted_tail(self, monkeypatch):
        """
        Regression guard for the bug fixed in IQNAgent.act() (which sampled
        tau ~ U[0, alpha] and then ALSO took the bottom-alpha slice of
        those samples, making the effective tail alpha^2 instead of alpha).

        The wrapper's _get_z must call the base agent's full-range
        _sample_tau (tau ~ U[0, 1]), not _sample_tau_cvar, and apply the
        bottom-alpha slice exactly once, inside cvar_action.
        """
        wrapped = self._wrapped(alpha=0.25)
        captured = {}
        orig_sample_tau = wrapped._agent._sample_tau

        def spy(batch_size, k):
            tau = orig_sample_tau(batch_size, k)
            captured["tau"] = tau
            return tau

        monkeypatch.setattr(wrapped._agent, "_sample_tau", spy)
        wrapped.act(_obs(), greedy=True)

        assert "tau" in captured, "wrapper must call the base agent's _sample_tau"
        # With K=100 samples from U[0,1], the chance every single one falls
        # below alpha=0.25 is 0.25**100 — not a flaky assertion in practice.
        assert captured["tau"].max().item() > wrapped.alpha

    def test_cvar_measure_avoids_catastrophic_tail(self, monkeypatch):
        n_tail = max(1, int(0.25 * K))
        remaining = K - n_tail
        high_value = (5.0 * K - (-100.0) * n_tail) / remaining
        Z = torch.zeros(1, K, N_ACTIONS)
        Z[0, :n_tail, 0] = -100.0
        Z[0, n_tail:, 0] = high_value
        Z[0, :, 1] = 0.5

        wrapped = self._wrapped(alpha=0.25, measure="cvar")
        monkeypatch.setattr(wrapped, "_get_z", lambda x: Z)
        assert wrapped.act(_obs(), greedy=True) == 1
