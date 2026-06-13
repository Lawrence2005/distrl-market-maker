"""
QR-DQN — Quantile Regression DQN.

N=200 quantile heads, Huber quantile loss rho_tau,
dueling network architecture (value + advantage streams),
prioritized experience replay.

CVaR policy: at inference, average lowest floor(alpha*N) quantile outputs.
See agents/cvar_policy.py for wrapper.

Reference: Dabney, Rowland, Bellemare & Munos (2018).
PRIMARY research agent.
Week 5 deliverable.
"""
# TODO: implement
