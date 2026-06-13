"""
Encoder × agent ablation table builder.

Full ablation matrix:
  rows:    DQN, PPO, QR-DQN (CVaR), IQN (CVaR)
  columns: Handcrafted, CNN, AE, LSTM
  cells:   Sharpe, convergence speed, OOD degradation

All four neural agents use all four encoders.
SARSA (handcrafted only) reported separately as the non-deep baseline.
Also compares three reward formulations on primary QR-DQN CVaR agent.

Total: 1 (SARSA) + 4×4 (neural) = 17 variants.
Week 8 deliverable.
"""
# TODO: implement
