# training/ — Training Scripts and Configs

## Files
- `train.py`           — Main training entry point (all agents)
- `pretrain_ae.py`     — Autoencoder pre-training on LOBSTER data (offline)
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
