# training/ — Training Scripts and Configs

## Files
- `train.py`           — Hydra entry point: episode loop, logging, checkpointing schedule
- `factory.py`         — Builds encoder/agent/env/policy-wrapper from Hydra config groups
- `rollout.py`         — Single-episode execution, episode metrics, checkpoint save/load
- `replay_buffer.py`   — Sequence replay buffer shared by the DRQN-style agents
- `pretrain_ae.py`     — Autoencoder pre-training on LOBSTER data (offline)
- `evaluate.py`        — Loads a checkpoint and runs evaluation rollouts
- `configs/`           — Hydra YAML configs (one per experiment)

## Usage
```bash
# Pre-train autoencoder (run once before RL training)
python training/pretrain_ae.py data.path=data/lobster/AAPL_2012.csv

# Train QR-DQN with CVaR alpha=0.10, AE encoder, asymmetric reward
python training/train.py agent=qrdqn encoder=autoencoder reward=asymmetric alpha=0.10

# Sweep CVaR alpha values
python training/train.py agent=qrdqn encoder=autoencoder reward=asymmetric alpha=0.05,0.10,0.25,0.50,1.0 --multirun
```

## Configs
See `training/configs/` — one YAML per component (agent, encoder, env, reward).
