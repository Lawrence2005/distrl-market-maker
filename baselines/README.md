# baselines/ — Analytical Benchmarks

Classical market-making models used as benchmarks.
Lineage: Fixed Spread → AS (2008) → GLFT (2012).

## Files
- `fixed_spread.py`  — Symmetric fixed-spread, no inventory adjustment
- `avellaneda_stoikov.py` — AS (2008) closed-form
- `glft.py`          — GLFT (2012), Proposition 5 with market impact (single unified implementation — the FOIC/LIIC split mentioned in earlier drafts of this README was never built)

## Usage
```python
from baselines.glft import GLFTBaseline
agent = GLFTBaseline(gamma=0.1, kappa=15.88, A=0.018, sigma=0.0003, Q_max=10, lot_size=100)
agent.reset()
action = agent.act(obs, info)  # info must have 'mid_price', 'inventory' (real shares)
```
`scripts/run_baseline.py` drives these against the live env and logs metrics
in the same format `training/train.py` uses, for direct comparison.

## Calibration status

`kappa`/`A`/`sigma` (used by `scripts/run_baseline.py`'s `_REGIME_CALIBRATION`
dict) are **empirically calibrated per regime** against this ABIDES instance
via `scripts/calibrate_fill_intensity.py --regime <name>`: quote a fixed
symmetric spread (`FixedSpreadBaseline`) at a grid of tick offsets, measure
the realized fill rate at each, fit `lambda(delta) = A*exp(-kappa*delta)`
(delta in dollars) by log-linear regression. Regimes differ in
`abides_overrides.fund_vol` (see `training/configs/env/regime/*.yaml`),
which materially changes the fill-rate curve, so a single global kappa
doesn't transfer across regimes.

This superseded two earlier, both-wrong single-value guesses:
- The original uncited `kappa=100.0` default implied a ~$0.02 (2-tick)
  spread — implausibly tight.
- AS (2008)'s own paper-illustrative `kappa=1.5` (still used by
  `tests/test_baselines.py`, but only for formula-arithmetic correctness,
  not market validity) implied a ~$0.65 (65-tick) spread that produced
  **zero fills in 6/6 verification episodes** — this ABIDES instance's real
  order flow essentially never reaches that far from mid.

Fitted implied spreads (gamma=0.1) increase monotonically with regime
volatility, as expected:

| regime | fund_vol | kappa | A | R² | implied spread |
|---|---|---|---|---|---|
| low_vol | 1e-5 | 15.88 | 0.0180 | 0.94 | 12.6 ticks ($0.13) |
| normal | 5e-5 | 7.63 | 0.0063 | 0.93 | 26.1 ticks ($0.26) |
| high_vol | 2e-4 | 3.73 | 0.0116 | 0.95 | 52.9 ticks ($0.53) |

(`flash_crash`/`trending` regimes were removed from the project — see git
history — so only these three remain.) Each fit used 5-7 non-saturated grid
points (10-90 ticks, excluding any that saw zero fills in the calibration
run rather than `log(0)`'ing them). `gamma` (risk aversion) isn't part of
the fill-intensity model and stays at AS(2008)'s illustrative 0.1 —
calibrating it would need a P&L-based procedure, not a fill-rate one; out
of scope here.

`sigma` is passed explicitly per this class's own documented ABIDES-typical
range (0.0003-0.001), scaled with the same volatility ordering, rather than
left at the old uncited default (0.01, 10-33x too high) —
`adapt_sigma=True` self-corrects it from realized volatility within an
episode regardless.

Re-run `calibrate_fill_intensity.py --regime <name>` and update
`scripts/run_baseline.py`'s `_REGIME_CALIBRATION` dict if the env's fill
dynamics change materially (order_size, tick range, fund_vol values, etc),
or if a new regime is added.

`GLFTBaseline.Q_max` is in **lots** (default 10), not real shares — it
sizes the `(2·Q_max+1)×(2·Q_max+1)` ODE grid (one `expm` solve per
recompute), which would be computationally infeasible at the env's real
per-share `Q_max` (~1000). `lot_size` (default 100, matching
`envs/lob_env.py`'s `order_size`) converts the env's real inventory into
lots before indexing the grid — `act()` handles this conversion, `Q_max`
and `compute_quotes()` operate purely in lot units.

## Week 3 deliverable
