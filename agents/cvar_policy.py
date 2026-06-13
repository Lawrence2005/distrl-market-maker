"""
CVaR policy wrapper for QR-DQN and IQN.

Applies CVaR risk objective at the POLICY SELECTION level only.
Does NOT modify the Bellman update — sidesteps numerical instability
of direct risk-sensitive RL (Spooner & Savani 2020 caveat).

CVaR_alpha(Z(s,a)) = (1 / floor(alpha*N)) * sum_{i <= floor(alpha*N)} theta_i(s,a)

Usage:
    policy = CVaRPolicy(agent=qrdqn_agent, alpha=0.10)
    action = policy.act(state)

Reference: Lim & Auer (2021); Dabney et al. QR-DQN (2018).
Week 5 deliverable.
"""
# TODO: implement
