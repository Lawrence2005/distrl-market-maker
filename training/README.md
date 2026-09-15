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

## Long/unattended runs: progress and resuming

Every run writes `logs/<run_tag>/status.json`, updated after every episode,
with episode count, % complete, elapsed/ETA, and the latest eval Sharpe —
check it (or run `../scripts/status.sh` to see every run at once) instead of
tailing a multi-day nohup log.

`train.py` can resume from the latest checkpoint in
`checkpoint_dir/<run_tag>/` rather than restarting at episode 0:

```bash
python training/train.py agent=qrdqn encoder=handcrafted reward=asymmetric seed=42 training.resume=true
```

For a run you want to survive crashes/host restarts unattended, launch it
through `../scripts/run_agent_supervised.sh` instead of a bare
`nohup python training/train.py ... &`. It creates the log directory (a
missing one is why a `nohup ... > ~/logdir/foo.out &` job can die silently
at launch with no trace) and auto-restarts with `training.resume=true` if
the process ever exits non-zero:

```bash
for agent in dqn qrdqn iqn ppo sarsa; do
  nohup ../scripts/run_agent_supervised.sh \
    ~/distrl-logs/w06_${agent}_low_vol_seed42.out \
    agent=$agent encoder=handcrafted reward=asymmetric \
    env.regime=low_vol env.use_abides=true training.n_episodes=1000 seed=42 \
    > /dev/null 2>&1 &
done
```
