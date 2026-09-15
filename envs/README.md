# envs/ — Environment Layer

ABIDES-Gym extensions for this project.

## Files
- `lob_env.py`           — Base ABIDES-Gym market-making environment wrapper
- `background_agents.py` — LOBSTER-calibrated noise/momentum/informed agents
- `stylized_facts.py`    — Post-episode stylized facts validator
- `multi_agent_env.py`   — Multi-agent wrapper (N simultaneous MM agents)

## Usage
```python
from envs.lob_env import LOBMarketMakingEnv
env = LOBMarketMakingEnv(config="configs/env/base.yaml")
obs, info = env.reset()
```

## Calibration
Background agent parameters are calibrated to LOBSTER data.
See `data/calibration/` for fitted parameters and
`notebooks/02_env_calibration.ipynb` for the calibration walkthrough.

## Week 2 deliverable
