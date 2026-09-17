# envs/ — Environment Layer

ABIDES-Gym extensions for this project.

## Files
- `lob_env.py`           — Base ABIDES-Gym market-making environment wrapper
- `stylized_facts.py`    — Post-episode stylized facts validator
- `multi_agent_env.py`   — Multi-agent wrapper (N simultaneous MM agents)

Background agent population comes from ABIDES's own built-in `rmsc04`
config, unmodified — there is no project-specific background-agent module
here.

## Usage
```python
from envs.lob_env import LOBMarketMakingEnv
env = LOBMarketMakingEnv(config="configs/env/base.yaml")
obs, info = env.reset()
```

## Week 2 deliverable
