# baselines/ — Analytical Benchmarks

Classical market-making models used as benchmarks.
Lineage: Fixed Spread → AS (2008) → GLFT (2012).

## Files
- `fixed_spread.py`  — Symmetric fixed-spread, no inventory adjustment
- `avellaneda_stoikov.py` — AS (2008) closed-form
- `glft.py`          — GLFT (2012): FOIC and LIIC variants
- `utils.py`         — Shared helpers (reservation price, spread calcs)

## Usage
```python
from baselines.glft import GLFT_FOIC
agent = GLFT_FOIC(gamma=0.1, sigma=0.02, Q_max=10)
bid, ask = agent.quotes(inventory=3, time_remaining=0.5)
```

## Week 3 deliverable
