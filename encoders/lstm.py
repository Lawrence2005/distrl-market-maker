"""
LSTM sequential encoder.

Produces a fixed-size 128-dim hidden state h_t from a rolling window
of T=30 sequential LOB snapshots. Compatible with ALL neural agents
(DQN, PPO, QR-DQN, IQN) — h_t is consumed identically to a
handcrafted/CNN/AE-encoded vector by any downstream Q-head or policy head.

Architecture:
    Input:  sequence of T LOB depth snapshots, shape (T, 2K)
    LSTM(input_size=2K, hidden_size=128, num_layers=1)
    Output: hidden state h_t, shape (128,)

Training: end-to-end with RL agent via truncated BPTT (window T=30).
          Hidden state zeroed at episode start.

Replay buffer: agents using this encoder must use a sequence replay buffer
(stores T-step sequences of raw LOB snapshots; LSTM hidden state is
recomputed from raw sequences during training to avoid stale hidden
state bias from old network weights).

Why LSTM works as a general encoder:
    Any agent that takes a fixed-size state vector can use h_t instead.
    The sequence replay requirement applies equally to DQN, PPO, QR-DQN,
    and IQN — it is an implementation detail, not a fundamental constraint.

References:
    Sun, Huang & Yu (2022): primary source for DRQN-LSTM architecture
    applied to LOB market making. What they call DRQN-LSTM is DQN + this
    LSTM encoder. The encoder itself is agent-agnostic.

    Kumar (2019): DRQN outperforms DQN — supports sequential encoding
    advantage (no CNN comparison in Kumar's paper).

Config: training/configs/encoder/lstm.yaml
Week 5 deliverable.
"""
import torch
import torch.nn as nn
from typing import Optional, Tuple


class LSTMEncoder(nn.Module):
    """
    LSTM encoder over a rolling window of LOB snapshots.

    Parameters
    ----------
    input_dim  : int   — dimension of each LOB snapshot (2K for K levels per side)
    hidden_dim : int   — LSTM hidden state size (default 128)
    window     : int   — sequence length T (default 30)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, window: int = 30):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.window     = window
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

    def encode(self, sequence: torch.Tensor,
               hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Encode a sequence of LOB snapshots.

        Parameters
        ----------
        sequence : Tensor shape (batch, T, input_dim) or (T, input_dim)
        hidden   : optional (h_0, c_0) from previous step

        Returns
        -------
        h_t   : Tensor shape (batch, hidden_dim) — state vector for agent
        hidden: (h_n, c_n) — carry forward to next step
        """
        if sequence.dim() == 2:
            sequence = sequence.unsqueeze(0)   # add batch dim
        out, hidden = self.lstm(sequence, hidden)
        h_t = out[:, -1, :]   # take last timestep hidden state
        return h_t.squeeze(0), hidden

    @property
    def output_dim(self) -> int:
        return self.hidden_dim

    def init_hidden(self, batch_size: int = 1,
                    device: torch.device = torch.device("cpu")
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialise hidden state to zeros (call at episode start)."""
        h = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        c = torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return h, c
