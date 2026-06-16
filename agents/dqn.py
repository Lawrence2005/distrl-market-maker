"""
DQN agent.

Two variants: (1) DQN with snapshot encoder (HC/CNN/AE); (2) DRQN-LSTM — LSTM(128) integrated inside the network, LOB → LSTM → Q-head, following Sun et al. (2022) exactly.

Reference: Sun, Huang & Yu (2022) for LSTM encoder architecture.
           Kumar (2019): DRQN outperforms DQN baseline.
Week 5 deliverable.
"""
# TODO: implement
import torch.nn as nn
from agents.recurrent_base import RecurrentBase

class DQN(nn.Module):
    """Snapshot variant — single LOB observation per step."""
    def __init__(self, encoder, n_actions):
        super().__init__()
        self.encoder = encoder
        self.q_head  = nn.Linear(encoder.output_dim, n_actions)
    ...

class DRQN(RecurrentBase):
    """Recurrent variant — LSTM integrated inside network (Sun et al. 2022)."""
    def __init__(self, input_dim, n_actions):
        super().__init__(input_dim, lstm_hidden=128)
        self.q_head = nn.Linear(self.output_dim, n_actions)

    def forward(self, seq, hidden):
        h_t, hidden = self.forward_lstm(seq, hidden)
        return self.q_head(h_t), hidden