# agents/ — RL Agent Implementations

Five agents total. SARSA is custom non-deep. DQN and PPO use SB3 for
standard replay variants. QR-DQN and IQN are custom PyTorch — the core
research contribution.

| Agent          | Library        | Risk-sensitive? | Encoders supported         |
|----------------|----------------|-----------------|----------------------------|
| sarsa.py       | Custom         | No              | Handcrafted (tile coding)  |
| dqn.py         | SB3 + custom   | No              | HC / CNN / AE / LSTM       |
| ppo.py         | SB3            | No              | HC / CNN / AE / LSTM       |
| qrdqn.py       | Custom PyTorch | CVaR wrapper    | HC / CNN / AE / LSTM       |
| iqn.py         | Custom PyTorch | CVaR wrapper    | HC / CNN / AE / LSTM       |
| cvar_policy.py | —              | —               | CVaR wrapper for QR-DQN/IQN|

All neural agents (DQN, PPO, QR-DQN, IQN) accept any of the four encoders
from `encoders/` via config. The LSTM encoder requires a sequence replay
buffer — implement once in `training/train.py`, shared across all agents
that use it.

SARSA is the only exception: tile coding is incompatible with neural encoders.

## Week 5 deliverable
