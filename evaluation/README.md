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
