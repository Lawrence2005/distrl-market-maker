# experiments/ — Experiment Configs and Results

Each experiment is a named folder with:
  - `config.yaml`  — full reproducible config snapshot
  - `results.json` — aggregated metrics
  - `logs/`        — W&B run links or local TensorBoard logs

## Naming Convention
`{week}_{agent}_{encoder}_{reward}_{regime}_{notes}`

Examples:
  w06_qrdqn_handcrafted_asymmetric_lowvol_convergence_check/
  w07_qrdqn_ae_asymmetric_highvol_alpha0.10/
  w08_ablation_all_agents_all_encoders_lowvol/
  w09_flash_crash_dqn_vs_qrdqn_alpha005/

## Reproducing a Run
```bash
python training/train.py --config-path experiments/<run_name>/config.yaml
```
