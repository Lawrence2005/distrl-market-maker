# envs/ — Environment Layer

ABIDES-Gym extensions for this project.

## Files
- `lob_env.py`           — Base ABIDES-Gym market-making environment wrapper
- `hawkes_arrivals.py`   — Hawkes process order arrival model (replaces Poisson)
- `background_agents.py` — LOBSTER-calibrated noise/momentum/informed agents
- `stylized_facts.py`    — Post-episode stylized facts validator
- `multi_agent_env.py`   — Multi-agent wrapper (N simultaneous MM agents)

## Usage
```python
from envs.lob_env import LOBMarketMakingEnv
env = LOBMarketMakingEnv(config="configs/env/base.yaml")
obs, info = env.reset()
```

## Hawkes Process Arrivals

ABIDES uses simple Poisson arrivals by default — constant intensity,
no clustering. Real order flow is bursty and self-exciting: a burst of
trades tends to trigger more trades shortly after.

We replace Poisson with a **Hawkes process** (Bacry et al. 2015):

```
lambda(t) = mu + sum_{t_j < t} alpha * exp(-beta * (t - t_j))
```

Each arrival temporarily raises the intensity of future arrivals.
Parameters (mu, alpha, beta) calibrated via MLE on LOBSTER tick data.

### Citation chain for this design choice:
- **Bacry, Mastromatteo & Muzy (2015)** — *Hawkes Processes in Finance*:
  primary citation. Derives multivariate Hawkes for bid/ask order flow;
  calibration methodology; shows it reproduces LOB stylized facts.
- **Huang, Lehalle & Rosenbaum (2015)** — empirically documents arrival
  clustering in real LOBs via the queue-reactive model. Their finding
  that intensities are state-dependent and bursty is what the Hawkes
  process models mathematically. Cite both together.

## Calibration
Background agent parameters are calibrated to LOBSTER data.
See `data/calibration/` for fitted parameters and
`notebooks/02_env_calibration.ipynb` for the calibration walkthrough.

## Week 2 deliverable
