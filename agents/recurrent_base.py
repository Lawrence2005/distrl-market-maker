"""
Shared LSTM backbone for all recurrent agent variants.

Imported by:
    agents/dqn.py     → DRQN
    agents/qrdqn.py   → RecurrentQRDQN
    agents/iqn.py     → RecurrentIQN

NOT used by:
    agents/sarsa.py       (tile coding — non-neural)
    agents/ppo.py         (use sb3-contrib RecurrentPPO directly)
    agents/cvar_policy.py (CVaR wrapper is architecture-agnostic)

Architecture:
    LOB snapshot sequence (T=30) → LSTM(hidden=128) → h_t (128-dim)
    h_t feeds directly into the downstream Q-head or policy head
    of whichever agent imports this class.

Sequence replay buffer:
    All agents using RecurrentBase must use a sequence replay buffer
    that stores raw T-step LOB sequences rather than single transitions.
    The LSTM hidden state is recomputed from the raw sequence during
    each training update — this avoids the stale hidden state bias
    that arises if you store and replay pre-computed hidden states
    from older network weights.

References:
    Hausknecht & Stone (2015) — original DRQN architecture
    Sun, Huang & Yu (2022)   — DRQN-LSTM applied to LOB market making
    Kumar (2019)             — DRQN outperforms DQN on cross-session generalisation

Week 5 deliverable.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class RecurrentBase(nn.Module):
    """
    Shared LSTM backbone inherited by all recurrent agent variants.

    Subclasses (DRQN, RecurrentQRDQN, RecurrentIQN) call
    self.forward_lstm() to get h_t, then pass h_t into their
    own Q-head or policy head.

    Parameters
    ----------
    input_dim   : int  — dimension of each LOB snapshot fed into the LSTM.
                         Typically the raw feature vector dimension (e.g. ~17
                         for handcrafted features, or 2K for K LOB levels).
    lstm_hidden : int  — LSTM hidden state size. Default 128.
    n_layers    : int  — number of LSTM layers. Default 1.
    """

    def __init__(
        self,
        input_dim: int,
        lstm_hidden: int = 128,
        n_layers: int = 1,
    ):
        super().__init__()
        self.input_dim   = input_dim
        self.lstm_hidden = lstm_hidden
        self.n_layers    = n_layers

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=n_layers,
            batch_first=True,   # input shape: (batch, T, input_dim)
        )

    def forward_lstm(
        self,
        seq: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Run the LSTM over a sequence of LOB snapshots.

        Parameters
        ----------
        seq    : Tensor, shape (batch, T, input_dim) or (T, input_dim)
                 A rolling window of T=30 sequential LOB observations.
                 If unbatched (T, input_dim), a batch dimension is added
                 automatically and squeezed back out before returning.
        hidden : optional (h_0, c_0) from the previous step.
                 Pass None at the start of each episode — init_hidden()
                 will be called automatically.

        Returns
        -------
        h_t    : Tensor, shape (batch, lstm_hidden) or (lstm_hidden,)
                 The hidden state at the final timestep. This is the
                 vector your Q-head or policy head receives as input.
        hidden : (h_n, c_n) — carry forward to the next timestep.
                 Store this in your agent and pass it back in on the
                 next call within the same episode.
        """
        unbatched = (seq.dim() == 2)
        if unbatched:
            seq = seq.unsqueeze(0)          # (1, T, input_dim)

        if hidden is None:
            device = seq.device
            hidden = self.init_hidden(batch_size=seq.size(0), device=device)

        out, hidden = self.lstm(seq, hidden)
        h_t = out[:, -1, :]                # take final timestep: (batch, lstm_hidden)

        if unbatched:
            h_t = h_t.squeeze(0)           # back to (lstm_hidden,)

        return h_t, hidden

    def init_hidden(
        self,
        batch_size: int = 1,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initialise hidden state and cell state to zeros.

        Call this at the START of every episode to reset the agent's
        temporal memory. Do not carry hidden state across episode
        boundaries — each episode is an independent trading day.

        Parameters
        ----------
        batch_size : int           — number of parallel environments.
        device     : torch.device  — must match your model's device.

        Returns
        -------
        (h_0, c_0) : both shape (n_layers, batch_size, lstm_hidden)
        """
        h_0 = torch.zeros(self.n_layers, batch_size, self.lstm_hidden, device=device)
        c_0 = torch.zeros(self.n_layers, batch_size, self.lstm_hidden, device=device)
        return h_0, c_0

    @property
    def output_dim(self) -> int:
        """
        Dimension of h_t — the vector passed to your Q-head or policy head.
        Use this when constructing downstream layers in subclasses so you
        never hardcode 128 in two places.

        Example in a subclass:
            self.q_head = nn.Linear(self.output_dim, n_actions)
        """
        return self.lstm_hidden