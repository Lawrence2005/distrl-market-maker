"""
DQN agent.

Standard DQN (scalar Q output) compatible with all four encoders:
handcrafted / CNN / AE / LSTM.

When using the LSTM encoder, a sequence replay buffer is required
(stores T=30-step sequences; LSTM hidden state recomputed during training).
For handcrafted/CNN/AE encoders, standard single-step replay is used.

The DQN + LSTM combination is what the literature calls DRQN-LSTM
(Sun et al. 2022) — architecturally it is just DQN with the LSTM encoder.

Reference: Sun, Huang & Yu (2022) for LSTM encoder architecture.
           Kumar (2019): DRQN outperforms DQN baseline.
Week 5 deliverable.
"""
# TODO: implement
