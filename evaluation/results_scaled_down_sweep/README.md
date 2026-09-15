# Round 3 — scaled-down local sweep

Results from the local (non-HPC) training sweep referenced in `evaluation/README.md`
as the current Week 8 model-comparison deliverable. **This is not the full 17-variant
state-representation ablation** described in the top-level `README.md` (Week 4/5/8) —
CNN and autoencoder encoders and all four recurrent (LSTM-integrated) agent variants
were never trained. Every result here uses `encoder=handcrafted` only. That larger
ablation remains future work; it was explicitly descoped for local single-machine
hardware (WSL, ~9.7GB RAM), not attempted and abandoned.

## Scope

- **Agents:** sarsa, dqn, ppo, qrdqn, iqn (5) — the non-recurrent, handcrafted-encoder
  variant of each.
- **Regimes:** low_vol, normal. `high_vol` and the megashock-based `flash_crash` were
  run earlier in the project but dropped from scope (see git history and
  `docs/design_decisions/README.md`); `trending` was removed at the same time.
- **Baselines:** fixed-spread, Avellaneda-Stoikov, GLFT — fill-intensity-calibrated
  per regime against this exact ABIDES instance (`scripts/calibrate_fill_intensity.py`;
  calibration constants documented in `scripts/run_baseline.py`).
- **Seed:** 42 only, everywhere. No multi-seed replication — see "What this doesn't
  establish" below.
- **CVaR alpha:** fixed at 0.25 for QR-DQN/IQN throughout. No alpha sweep was run.
- **Training:** 500-episode budget, patience-based early stopping on smoothed eval
  Sharpe (patience=6 checkpoints, min_delta=0.05, smoothed over the last 3 checkpoints,
  no stopping before episode 150 — see `training/train.py`'s `_early_stop_state`).

## Files

| File | What it is |
|---|---|
| `model_comparison.csv` | Mean Sharpe over each run's *entire* logged history — diagnostic only (exploration-contaminated for RL agents; see the dashboard's methodology notes). Not the number to rank agents by. |
| `as_recovery.csv` | R² of each RL agent's realized bid/ask skew against the closed-form GLFT/AS curves, computed on each agent's **best eval checkpoint** (not latest — see "Bugs found" below), 15 greedy rollouts each. |
| `holdout_eval.csv` | Per-episode raw results: every model (5 RL agents' best checkpoint + 3 baselines) run on **15 identical, never-before-seen seeds per regime** — `evaluation` seeds a fixed offset (`+80000`) clear of every training/eval/AS-recovery seed range used anywhere else. This is the fair, paired, head-to-head comparison. |
| `holdout_summary.csv` | `holdout_eval.csv` aggregated to mean/std/n per model x regime. |
| `significance_vs_glft.csv` | Paired *t*-test and Wilcoxon signed-rank test, each RL agent vs. GLFT, on the 15 shared holdout seeds (paired because both see identical seeds). |

Reproduce via `scripts/run_analysis.py` (AS-recovery + model comparison) and
`scripts/run_holdout_eval.py --n_episodes 15` (holdout + significance table computed
inline in the dashboard build — see below).

## Key findings

- **QR-DQN significantly beats GLFT (the strictest baseline) in `normal`**: held-out
  mean Sharpe +2.76 vs. GLFT's -0.77 (paired *t*-test p<0.0001, Wilcoxon p=0.0001,
  n=15). PPO also clears p<0.05 in the same regime (+0.14 vs -0.77, p=0.033/0.041).
  Neither reaches significance in `low_vol`, where GLFT's tight calibrated quoting
  (+1.43 held-out) isn't beaten by any RL agent.
- Every other RL-agent/regime combination either underperforms GLFT or doesn't clear
  significance with n=15 — see `significance_vs_glft.csv` for the full table.
- **AS-recovery is weak-to-absent across the board** (best: DQN `normal` GLFT R²=0.81;
  most agents R²<0.3 or insufficient distinct inventory levels visited) — none of
  these agents robustly rediscovered analytical inventory-skew quoting.

## Bugs found and fixed during this sweep

Two real bugs in `agents/sarsa.py` were caught while building the results dashboard,
both now fixed (archived pre-fix results in `archive/sarsa_is_online_bug/` and
`archive/sarsa_epsilon_bug/` — do not use):

1. **`SARSAAgent` never set `is_online`** (the attribute every other agent sets
   explicitly — see `agents/dqn.py`, `agents/ppo.py`, `agents/iqn.py`,
   `agents/qrdqn.py`). `training/rollout.py`'s `run_episode()` dispatches on this
   attribute to route SARSA through its own `train_step(obs, action, reward,
   next_obs, done)` path instead of the replay-buffer-style path used by the neural
   agents; without it, SARSA fell through to `agent.train_step()` called with no
   arguments every step, which no-ops. Every SARSA checkpoint before this fix
   reflects an untrained, randomly-initialized policy. Caught by noticing
   `mean_loss` was exactly 0.0 for every training episode.
2. **Epsilon never decayed even after the above fix**, because the step counter
   driving epsilon decay lived in `observe()`, which `is_online` agents never call
   (only `train_step()` is called). Fixed by moving the increment into `train_step()`
   itself. Caught by noticing `status.json`'s `epsilon` field frozen at `epsilon_start`
   after hundreds of episodes.

Both fixes are one or two lines each in `agents/sarsa.py`, verified against real
ABIDES rollouts and the full non-ABIDES test suite (549 passed) before SARSA was
retrained a third time to produce the numbers in this directory.

## What this doesn't establish

- **No multi-seed replication.** All results are seed=42. The held-out significance
  test (n=15 paired episodes) is real evidence within that one seed's trained policy,
  but doesn't rule out seed-to-seed variance in which policy training converges to.
- **No CVaR alpha sensitivity.** QR-DQN/IQN were only ever trained at alpha=0.25.
- **No `high_vol` regime.** Only low_vol/normal are covered here.
- **No encoder/recurrent ablation.** See the top-of-file note — this is the
  handcrafted-snapshot slice of the full 17-variant design, not the full ablation.

## Dashboard

An interactive visualization of all of the above (training/eval curves, loss curves,
peak-vs-final drift analysis, the held-out comparison, AS-recovery, and this
significance table) was built as a Claude artifact during this sweep — ask in a
Claude Code session with access to this project's conversation history, or rebuild it
from `evaluation/results_scaled_down_sweep/*.csv` and `logs/*/train_history.json` /
`eval_history.json`.
