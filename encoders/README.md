# encoders/ — State Representation Modules

Four encoders tested as the scientific ablation axis. All four expose the
same interface so any agent can swap encoders via a config change only:

```python
encoder.encode(obs) -> torch.Tensor   # fixed-size vector
```

## Encoder summary

| Encoder       | Input         | Output dim | Temporal? | Pre-training | Replay buffer |
|---------------|---------------|------------|-----------|--------------|---------------|
| Handcrafted   | 11–13 dim vec | 11–13      | No        | None         | Standard      |
| CNN           | LOB snapshot  | 16–32      | No        | End-to-end   | Standard      |
| AE            | LOB snapshot  | 8–32       | No        | Unsupervised | Standard      |

* LSTM is integrated inside each recurrent agent variant. See agents/dqn.py (DRQN), agents/qrdqn.py (recurrent QR-DQN), agents/iqn.py (recurrent IQN).

## Ablation matrix

All neural agents (DQN, PPO, QR-DQN, IQN) use all four encoders.
SARSA uses tile coding only — incompatible with neural encoders.

```
                   Handcrafted   CNN   AE    LSTM
SARSA              ✓             ✗     ✗     ✗
DQN                ✓             ✓     ✓     ✓
PPO                ✓             ✓     ✓     ✓
QR-DQN (CVaR_α)    ✓             ✓     ✓     ✓
IQN (CVaR_α)       ✓             ✓     ✓     ✓
```

Total variants: 1 + (4 × 4) = 17

## LSTM encoder note

The LSTM encoder produces a 128-dim hidden state h_t — a fixed-size vector
that any agent's Q-head or policy head consumes identically to a
handcrafted/CNN/AE vector. There is no fundamental reason to restrict it
to DQN. The only additional requirement is a sequence replay buffer (all
agents using LSTM share the same buffer implementation).

## Week 4–5 deliverable
