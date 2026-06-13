# Design Decision Log

Documents architectural choices that were considered and scoped out,
with clear reasoning. Useful for write-up and interviews.

## Scoped Out

### Macro/Micro Hierarchical Agent (Patel 2018)
- **What it is**: Two-timescale framework — macro agent (buy/sell/hold,
  minute data) + micro agent (LOB placement).
- **Why scoped out**: Patel's own results show multi-agent underperforms
  micro-only because macro pushes orders to the wrong LOB side. Coordination
  overhead is orthogonal to our distributional RL question. The LSTM encoder
  applied to DQN/QR-DQN/IQN already subsumes the micro-agent's temporal LOB
  awareness in a single-agent architecture without coordination overhead.
- **Future work**: Could revisit with a learned communication protocol.

### Adversarial RL (Spooner & Savani 2020)
- **What it is**: Training MM vs. co-evolving adversary that controls market
  params (sigma, A, k) in a zero-sum game.
- **Why scoped out**: Requires implementing a second adversary agent in ABIDES
  that learns simultaneously — effectively a separate research project.
  Spooner & Savani themselves note direct risk-sensitive RL is numerically
  unstable. Our CVaR wrapper achieves robustness through a tractable
  alternative mechanism.
- **Our analog**: Flash crash stress test (Week 9) serves as a fixed-adversary
  robustness check. ARL is the natural next step.
- **Future work**: Implement NAC-S(lambda) adversary agent post-project.
