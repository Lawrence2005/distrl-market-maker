"""load_agent must restore the CVaR alpha a qrdqn/iqn checkpoint was trained at.

It used to build every agent with the constructor default (0.25), so any
post-hoc evaluation of a checkpoint trained at another alpha silently acted at
0.25 (the checkpoint stores alpha only in its saved config).
"""
import pytest
import torch

from agents.iqn import IQNAgent
from agents.qrdqn import QRDQNAgent
from encoders.handcrafted import HandcraftedEncoder
from training.evaluate import load_agent

from envs.lob_env import N_OFFSET_LEVELS

N_ACTIONS = N_OFFSET_LEVELS ** 2

KW = {
    "qrdqn": dict(hidden_dim=32, n_quantiles=8),
    "iqn":   dict(hidden_dim=32, n_quantile_samples=4, embedding_dim=8),
}


def _save(tmp_path, agent_type, config):
    enc = HandcraftedEncoder(obs_dim=18)
    if agent_type == "qrdqn":
        agent = QRDQNAgent(encoder=enc, n_actions=N_ACTIONS, hidden_dim=32, n_quantiles=8,
                           use_lstm=False)
    else:
        agent = IQNAgent(encoder=enc, n_actions=N_ACTIONS, hidden_dim=32,
                         n_quantile_samples=4, embedding_dim=8, use_lstm=False)
    path = tmp_path / f"{agent_type}.pt"
    torch.save({"agent_state": agent.state_dict(), "config": config}, path)
    return path


@pytest.mark.parametrize("agent_type", ["qrdqn", "iqn"])
class TestLoadAgentAlpha:
    def test_restores_trained_alpha(self, tmp_path, agent_type):
        path = _save(tmp_path, agent_type, {"alpha": 0.05})
        agent, _ = load_agent(path, agent_type, "handcrafted", **KW[agent_type])
        assert agent.cvar_alpha == 0.05

    def test_explicit_alpha_overrides_checkpoint(self, tmp_path, agent_type):
        path = _save(tmp_path, agent_type, {"alpha": 0.05})
        agent, _ = load_agent(path, agent_type, "handcrafted", cvar_alpha=0.5, **KW[agent_type])
        assert agent.cvar_alpha == 0.5

    def test_missing_config_falls_back_to_default(self, tmp_path, agent_type):
        path = _save(tmp_path, agent_type, {})
        agent, _ = load_agent(path, agent_type, "handcrafted", **KW[agent_type])
        assert agent.cvar_alpha == 0.25
