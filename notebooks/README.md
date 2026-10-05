# notebooks/ — Exploratory Analysis

| Notebook | Week | Purpose |
|---|---|---|
| 05_convergence_check.ipynb    | 6 | Training curves, sanity checks — reads `logs/` dynamically, current |
| 07_ablation_analysis.ipynb    | 8 | Encoder x agent ablation table — real implementation, waiting on the full 17-variant ablation (not yet run) |
| 08_efficient_frontier.ipynb   | 9 | CVaR alpha sweep, frontier plot — real implementation, waiting on an alpha sweep (not yet run) |

`01_literature_review.ipynb`, `02_env_calibration.ipynb`, `03_stylized_facts.ipynb`, and
`10_final_figures.ipynb` were planned here but never created — remove from this table if
still unplanned when picking this back up, rather than leaving them looking like missing
files. `04_baseline_analysis.ipynb`, `06_as_recovery.ipynb`, and `09_flash_crash.ipynb`
existed and were removed as part of a cleanup pass: 04's baseline-quoting parameters
(kappa derived analytically) were the guessed values later proven wrong by
`scripts/calibrate_fill_intensity.py`'s live-ABIDES calibration (6/6 zero-fill episodes);
06 loaded real ABIDES-trained checkpoints into a `use_abides=False` synthetic zero-fill
environment, which can't produce meaningful fill/skew behavior regardless of its other
stale constants; 09 referenced the flash_crash regime and a `Visualizer.plot_flash_crash`
method both removed from the project. Current baseline analysis and AS-recovery live in
`scripts/run_baseline.py` / `scripts/run_analysis.py`, verified against real ABIDES — see
`results/README.md`.
