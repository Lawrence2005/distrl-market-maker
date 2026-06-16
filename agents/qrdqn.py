"""
QR-DQN — Quantile Regression DQN.

Two variants: (1) snapshot QR-DQN with HC/CNN/AE encoder; (2) recurrent QR-DQN with LSTM integrated inside the architecture. Recurrent + CVaR is the PRIMARY research agent.

Reference: Dabney, Rowland, Bellemare & Munos (2018).
Week 5 deliverable.
"""
# TODO: implement
import torch.nn as nn
from agents.recurrent_base import RecurrentBase

class QRDQN(nn.Module):
    """Snapshot variant."""
    ...

class RecurrentQRDQN(RecurrentBase):
    """Recurrent variant — LSTM + N=200 quantile heads."""
    def __init__(self, input_dim, n_actions, n_quantiles=200):
        super().__init__(input_dim, lstm_hidden=128)
        self.n_actions   = n_actions
        self.n_quantiles = n_quantiles
        # Dueling heads operating on h_t
        self.value_head     = nn.Linear(self.output_dim, n_quantiles)
        self.advantage_head = nn.Linear(self.output_dim, n_actions * n_quantiles)

    def forward(self, seq, hidden):
        h_t, hidden = self.forward_lstm(seq, hidden)
        value     = self.value_head(h_t)                                        # (batch, N)
        advantage = self.advantage_head(h_t).view(-1, self.n_actions, self.n_quantiles)
        quantiles = value.unsqueeze(1) + advantage - advantage.mean(dim=1, keepdim=True)
        return quantiles, hidden   # (batch, n_actions, N)