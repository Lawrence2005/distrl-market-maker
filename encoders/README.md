# encoders/ — State Representation Modules

Three snapshot encoders are implemented here as external modules. All three
expose the same interface so any agent can swap encoders via a config change:

```python
encoder.encode(obs) -> torch.Tensor   # fixed-size vector
```

## Encoder summary

| Encoder       | Input         | Output dim | Temporal? | Pre-training | Replay buffer |
|---------------|---------------|------------|-----------|--------------|---------------|
| Handcrafted   | ~17-dim vec   | ~17        | No        | None         | Standard      |
| CNN           | LOB snapshot  | 16–32      | No        | End-to-end   | Standard      |
| AE            | LOB snapshot  | 8–32       | No        | Unsupervised | Standard      |

> **LSTM is integrated inside each recurrent agent variant as a first-class architectural component.** See:
> - `agents/recurrent_base.py` — shared LSTM backbone
> - `agents/dqn.py` — DRQN variant
> - `agents/qrdqn.py` — Recurrent QR-DQN variant
> - `agents/iqn.py` — Recurrent IQN variant

## Ablation matrix

Each neural agent (DQN, PPO, QR-DQN, IQN) has three snapshot variants
(one per encoder here) and one recurrent variant (LSTM integrated inside
the agent). SARSA uses tile coding only.

```
                   Handcrafted   CNN   AE    Recurrent (LSTM inside agent)
SARSA              ✓             ✗     ✗     ✗
DQN / DRQN         ✓             ✓     ✓     ✓
PPO / Rec. PPO     ✓             ✓     ✓     ✓
QR-DQN (CVaR_α)    ✓             ✓     ✓     ✓
IQN (CVaR_α)       ✓             ✓     ✓     ✓
```

Total: 1 (SARSA) + 4 agents × 3 snapshot + 4 agents × 1 recurrent = **17 variants**

## Week 4–5 deliverable