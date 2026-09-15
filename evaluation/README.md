# evaluation/ — Metrics and Analysis

All evaluation metrics from the research plan's Evaluation Framework sheet.

## Files
- `metrics.py`          — All metric implementations (Sharpe, MAP, CVaR, MDD, PnLMAP...)
- `as_recovery.py`      — AS/GLFT quote-skew R² test
- `efficient_frontier.py` — Return vs. CVaR frontier across alpha sweep
- `ablation.py`         — Encoder x agent ablation table builder
- `visualize.py`        — All plots (frontier, latent space PCA, inventory dist...)

Market-quality/stylized-facts checks (price impact, spread autocorrelation)
live in `envs/stylized_facts.py`, not here — they validate the simulator
itself rather than a trained agent.

## Week 8–9 deliverable

`results_scaled_down_sweep/` holds the current model-comparison results — a
scaled-down local sweep (5 agents x {low_vol, normal}, handcrafted encoder only,
seed=42), not the full 17-variant encoder/recurrent ablation the top-level
`README.md` describes for Week 8. See `results_scaled_down_sweep/README.md` for
scope, key findings, and two SARSA bugs found and fixed while producing these
results.
