"""
IQN — Implicit Quantile Network.

Two variants: (1) snapshot IQN with HC/CNN/AE encoder; (2) recurrent IQN with LSTM integrated inside the architecture.

Reference: Dabney, Ostrovski, Silver & Munos (2018).
Week 5 deliverable.
"""
# TODO: implement
import torch
import torch.nn as nn
from agents.recurrent_base import RecurrentBase

class IQN(nn.Module):
    """Snapshot variant."""
    ...

class RecurrentIQN(RecurrentBase):
    """Recurrent variant — LSTM + implicit quantile function."""
    def __init__(self, input_dim, n_actions, embedding_dim=64):
        super().__init__(input_dim, lstm_hidden=128)
        self.n_actions     = n_actions
        self.embedding_dim = embedding_dim
        # Cosine embedding for τ
        self.tau_embed = nn.Linear(embedding_dim, self.output_dim)
        self.q_head    = nn.Linear(self.output_dim, n_actions)

    def forward(self, seq, hidden, tau: torch.Tensor):
        """
        tau : Tensor shape (batch, n_samples) — sampled quantile levels.
              During training: tau ~ U([0,1]).
              During CVaR inference: tau ~ U([0, alpha]).
        """
        h_t, hidden = self.forward_lstm(seq, hidden)
        # Cosine embedding: φ(τ) = ReLU(Σ cos(πjτ) · w_j + b)
        i   = torch.arange(1, self.embedding_dim + 1, device=tau.device).float()
        phi = torch.cos(torch.pi * tau.unsqueeze(-1) * i)  # (batch, n_samples, embed_dim)
        phi = torch.relu(self.tau_embed(phi))               # (batch, n_samples, lstm_hidden)
        # Element-wise product with h_t, then Q-head
        h_t = h_t.unsqueeze(1) * phi                       # (batch, n_samples, lstm_hidden)
        return self.q_head(h_t), hidden                     # (batch, n_samples, n_actions)