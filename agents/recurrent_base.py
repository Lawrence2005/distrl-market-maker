"""
agents/recurrent_base.py

Shared DRQN-style recurrent base for all agent variants.

Architecture
------------
                    ┌─────────────┐
    obs / snapshot  │   Encoder   │  HandcraftedEncoder / AEEncoder / CNNEncoder
                    └──────┬──────┘
                           │ latent_z  (B, latent_dim)
                    ┌──────▼──────┐
                    │    LSTM     │  hidden_dim=128, num_layers=1
                    └──────┬──────┘
                           │ h_t  (B, hidden_dim)  ← last step hidden state
                    ┌──────▼──────┐
                    │ Dueling Head│  Value + Advantage streams
                    └──────┬──────┘
                           │ Q(s,a)  (B, n_actions)
                           ▼

QRDQNAgent and IQNAgent replace the dueling head output with a quantile
projection — RecurrentBase.hidden_dim is exposed so subclasses can add
their own head on top of the LSTM output.

Dueling architecture (Wang et al. 2016)
----------------------------------------
    h_t → [Value stream]     → V(s)        scalar
        → [Advantage stream] → A(s,a)      (n_actions,)
    Q(s,a) = V(s) + A(s,a) − mean_a(A(s,a))

    Each stream: Linear(hidden_dim → hidden_dim//2) → ReLU → Linear → output

Hidden state management
-----------------------
Training (TBPTT — Truncated Backprop Through Time):
    - hidden state is passed between chunks and detached at chunk boundaries
    - call reset_hidden(batch_size) at episode start
    - call detach_hidden() between TBPTT chunks

Inference (single-step rollout):
    - hidden state is maintained across steps automatically
    - call reset_hidden(batch_size=1) when env.reset() is called
    - call step(obs) at each env step — hidden state updates internally

Parameters
----------
encoder    : nn.Module — any encoder with .latent_dim property
n_actions  : int       — size of the discrete action space
hidden_dim : int       — LSTM hidden state dimension (default 128)
num_layers : int       — number of LSTM layers (default 1)

Reference
---------
Hausknecht & Stone (2015) — Deep Recurrent Q-Networks (DRQN)
Wang et al. (2016)        — Dueling Network Architectures
Sun et al. (2022)         — DRQN for market making

Week 5 deliverable.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


# ── Dueling head ──────────────────────────────────────────────────────────────

class DuelingHead(nn.Module):
    """
    Dueling Q-value head (Wang et al. 2016).

    Splits into value and advantage streams to stabilise training by
    separating state value estimation from action advantage estimation.

    Q(s,a) = V(s) + A(s,a) − (1/|A|) Σ_a' A(s,a')

    Parameters
    ----------
    input_dim : int — hidden state dimension from LSTM
    n_actions : int — number of discrete actions
    """

    def __init__(self, input_dim: int, n_actions: int):
        super().__init__()
        mid = max(input_dim // 2, n_actions)

        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, mid),
            nn.ReLU(),
            nn.Linear(mid, n_actions),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h : Tensor shape (B, input_dim)

        Returns
        -------
        Tensor shape (B, n_actions) — Q-values
        """
        V = self.value_stream(h)            # (B, 1)
        A = self.advantage_stream(h)        # (B, n_actions)
        return V + A - A.mean(dim=-1, keepdim=True)


# ── Recurrent base ────────────────────────────────────────────────────────────

class RecurrentBase(nn.Module):
    """
    DRQN-style recurrent base: Encoder → LSTM → Dueling Head.

    Subclassed by QRDQNAgent and IQNAgent, which replace or extend the
    dueling head with quantile projection layers.

    RecurrentBase can also be used directly as a vanilla DQN backbone
    (without distributional outputs) for the DQN ablation variant.

    Parameters
    ----------
    encoder    : nn.Module — encoder with .latent_dim property
    n_actions  : int       — number of discrete actions
    hidden_dim : int       — LSTM hidden state size (default 128)
    num_layers : int       — LSTM depth (default 1)
    dropout    : float     — LSTM dropout between layers (default 0.0)
                             only active when num_layers > 1
    dueling    : bool      — use dueling head (default True)
                             set False to output raw LSTM hidden state
                             (for subclasses that add their own head)
    use_lstm   : bool      — snapshot vs. recurrent ablation switch (default True)
                             True  → Encoder -> LSTM -> head (temporal memory)
                             False → Encoder -> Linear -> head, applied per
                             timestep independently (no hidden state carried
                             across steps) — the "snapshot" variant
    """

    def __init__(
        self,
        encoder:    nn.Module,
        n_actions:  int,
        hidden_dim: int   = 128,
        num_layers: int   = 1,
        dropout:    float = 0.0,
        dueling:    bool  = True,
        use_lstm:   bool  = True,
    ):
        super().__init__()

        self.encoder    = encoder
        self.n_actions  = n_actions
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self._dueling   = dueling
        self.use_lstm   = use_lstm

        if use_lstm:
            # LSTM: takes latent_z sequence, outputs hidden state sequence
            self.lstm = nn.LSTM(
                input_size  = encoder.latent_dim,
                hidden_size = hidden_dim,
                num_layers  = num_layers,
                batch_first = True,         # input shape: (B, T, latent_dim)
                dropout     = dropout if num_layers > 1 else 0.0,
            )
            self.proj = None
        else:
            # Snapshot mode: no temporal memory — per-timestep linear
            # projection replaces the LSTM entirely.
            self.lstm = None
            self.proj = nn.Linear(encoder.latent_dim, hidden_dim)

        # Dueling head on top of LSTM hidden state
        if dueling:
            self.head = DuelingHead(hidden_dim, n_actions)
        else:
            self.head = None   # subclass will add its own head

        # Hidden state: (num_layers, B, hidden_dim)
        # Initialised on first forward pass or explicit reset_hidden() call
        self._hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None

    # ------------------------------------------------------------------
    # Hidden state management
    # ------------------------------------------------------------------

    def reset_hidden(self, batch_size: int = 1, device: Optional[torch.device] = None) -> None:
        """
        Reset LSTM hidden state to zeros.

        Call at the start of each episode (training or inference).

        Parameters
        ----------
        batch_size : int           — B (default 1 for inference)
        device     : torch.device  — if None, infers from LSTM parameters
        """
        if not self.use_lstm:
            self._hidden = None
            return
        if device is None:
            device = next(self.lstm.parameters()).device
        zeros = torch.zeros(
            self.num_layers, batch_size, self.hidden_dim, device=device
        )
        self._hidden = (zeros, zeros.clone())

    def detach_hidden(self) -> None:
        """
        Detach hidden state from the computation graph.

        Call between TBPTT chunks during training to prevent gradients
        flowing back through the full episode history.
        """
        if self._hidden is not None:
            h, c = self._hidden
            self._hidden = (h.detach(), c.detach())

    def get_hidden(self) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Return current (h, c) hidden state tuple — for checkpointing."""
        return self._hidden

    def set_hidden(self, hidden: tuple[torch.Tensor, torch.Tensor]) -> None:
        """Restore hidden state — for checkpointing or multi-env rollouts."""
        self._hidden = hidden

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward(
        self,
        x:      torch.Tensor,
        hidden: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Full sequence forward pass — used during training.

        Parameters
        ----------
        x      : Tensor shape (B, T, input_dim)
                 Sequence of encoder inputs over T timesteps.
                 input_dim = encoder.input_dim (e.g. 18 for handcrafted,
                 20 for AE/CNN raw LOB snapshot).
        hidden : (h, c) tuple or None
                 Initial hidden state. If None, uses self._hidden.
                 If self._hidden is also None, initialises to zeros.

        Returns
        -------
        q      : Tensor shape (B, T, n_actions) — Q-values at each step
                 (only if dueling=True; otherwise returns LSTM output)
        hidden : (h, c) tuple — final hidden state for next chunk
        """
        # Encode sequence: (B, T, input_dim) → (B, T, latent_dim)
        z = self.encoder(x)

        if self.use_lstm:
            # Initialise hidden state if needed
            if hidden is None:
                hidden = self._hidden
            if hidden is None:
                self.reset_hidden(batch_size=x.size(0), device=x.device)
                hidden = self._hidden

            # LSTM: (B, T, latent_dim) → (B, T, hidden_dim)
            lstm_out, new_hidden = self.lstm(z, hidden)

            # Update stored hidden state
            self._hidden = new_hidden
        else:
            # Snapshot mode: per-timestep projection, no temporal memory
            lstm_out   = self.proj(z)   # (B, T, hidden_dim)
            new_hidden = None
            self._hidden = None

        # Dueling head over all timesteps
        if self._dueling and self.head is not None:
            B, T, H = lstm_out.shape
            q = self.head(lstm_out.reshape(B * T, H))   # (B*T, n_actions)
            q = q.reshape(B, T, self.n_actions)          # (B, T, n_actions)
            return q, new_hidden

        return lstm_out, new_hidden

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Single-step inference forward pass — used during rollout.

        Maintains hidden state internally across calls. Call reset_hidden()
        when env.reset() is called to start a new episode.

        Parameters
        ----------
        x : Tensor shape (B, input_dim) or (1, input_dim)
            Single timestep encoder input.

        Returns
        -------
        Tensor shape (B, n_actions) — Q-values for current step
        """
        # Add time dimension: (B, input_dim) → (B, 1, input_dim)
        x_seq = x.unsqueeze(1)
        q_seq, _ = self.forward(x_seq)

        if self._dueling:
            return q_seq.squeeze(1)   # (B, n_actions)
        return q_seq.squeeze(1)       # (B, hidden_dim) for subclass heads

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def count_parameters(self) -> dict[str, int]:
        """Parameter counts by component — useful for ablation reporting."""
        def n(module): return sum(p.numel() for p in module.parameters())
        temporal = self.lstm if self.use_lstm else self.proj
        return {
            "encoder":  n(self.encoder),
            "lstm":     n(temporal),
            "head":     n(self.head) if self.head is not None else 0,
            "total":    n(self),
        }

    def __repr__(self) -> str:
        counts = self.count_parameters()
        temporal_repr = (
            f"LSTM(input={self.encoder.latent_dim}, "
            f"hidden={self.hidden_dim}, layers={self.num_layers})"
            if self.use_lstm else
            f"Linear(input={self.encoder.latent_dim}, hidden={self.hidden_dim})  "
            f"# snapshot — no temporal memory"
        )
        return (
            f"RecurrentBase(\n"
            f"  encoder={self.encoder},\n"
            f"  temporal={temporal_repr},\n"
            f"  head={'DuelingHead' if self._dueling else 'None (subclass)'},\n"
            f"  n_actions={self.n_actions},\n"
            f"  params: encoder={counts['encoder']:,} "
            f"lstm={counts['lstm']:,} "
            f"head={counts['head']:,} "
            f"total={counts['total']:,}\n"
            f")"
        )