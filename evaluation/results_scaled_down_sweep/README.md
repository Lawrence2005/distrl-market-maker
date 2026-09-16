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

Numbers below are post-loss-scale-fix (see "Loss-scale fix" section) — the fix
materially changed the picture, so treat this as the current, not historical, result.

- **DQN and QR-DQN significantly beat GLFT in `normal`, by a wide margin**: QR-DQN's
  held-out mean Sharpe is +4.39 vs. GLFT's -0.71 (paired *t*-test p<0.0001, n=15); DQN's
  is +3.49 (p<0.0001) — DQN went from one of the worst performers pre-fix to the
  second-best model overall, beating every baseline outright. PPO also clears p<0.05
  in `normal` (+0.76, p=0.0045). IQN does not reach significance there (p=0.09).
- **None of the 5 RL agents beat GLFT in `low_vol`** post-fix — all 5 are significantly
  *below* GLFT there (GLFT and AS's held-out Sharpe, +1.34 and +1.59, remain strong and
  uncontested). This regime got uniformly worse for RL agents after the fix; with only
  one seed per agent it isn't possible to tell whether that's the fix's effect or
  ordinary retraining variance — see "What this doesn't establish."
- **AS-recovery improved for several agents** post-fix (DQN `low_vol` GLFT R²=0.64,
  QR-DQN `low_vol` R²=0.85, IQN `normal` R²=0.61 — all now STRONG recovery, versus weak
  or absent before) but the pattern isn't uniform: IQN `low_vol` and QR-DQN `normal`
  now show *no* recovery despite the corresponding agent's strong held-out Sharpe in
  QR-DQN `normal`'s case — held-out performance and AS-style skew recovery are
  evidently measuring different things, not interchangeable indicators of "good policy."
  Full table in `as_recovery.csv`.

## Bugs found and fixed during this sweep

Two real bugs in `agents/sarsa.py` were caught while building the results dashboard,
both now fixed (pre-fix results were archived locally then later cleared in a repo
cleanup pass — not in git, not recoverable, don't go looking for them):

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

## Loss-scale fix (post-order_size, pre-existing miscalibration)

A code-review audit (not a bug found via broken output — everything above ran and
produced plausible-looking numbers) found that DQN/QR-DQN/IQN's Huber-loss threshold
(`kappa`/`huber_beta`, hardcoded default 1.0) and PPO's `value_coef` (0.5) were never
rescaled when `order_size` went from 1 to 100 shares (see the environment-fix commit) —
even though the analogous quadratic-reward `lam` *was* rescaled for exactly this reason.
Rewards and TD-errors now run roughly 100x larger, so these losses were spending
training in the wrong regime: DQN/QR-DQN/IQN's Huber loss almost always in its linear
tail rather than quadratic-near-zero, and PPO's plain-MSE value loss scaling
quadratically with the mismatch while sharing a backbone with the policy head — a
mechanistically plausible explanation for PPO's original weak showing.

Fix: `kappa`/`huber_beta` rescaled 100x up (1.0 → 100.0), `value_coef` 100x down
(0.5 → 0.005), wired through config (`training/configs/agent/{dqn,qrdqn,iqn,ppo}.yaml`)
instead of hardcoded. All 4 affected agents (dqn, qrdqn, iqn, ppo — not sarsa, unaffected)
were retrained from scratch on both regimes with the fix; pre-fix results were archived
locally first, then cleared out in a later repo-cleanup pass (not in git either way) —
the pre-fix numbers exist only in this project's own conversation history now, not on
disk.

**Effect was large and agent/regime-specific, not a uniform improvement**: QR-DQN and
DQN's `normal`-regime held-out Sharpe roughly doubled (see Key Findings), while every
agent's `low_vol` performance and IQN's performance in both regimes got worse. This
asymmetry is itself informative — it suggests the original miscalibration was masking
real differences between agents/regimes rather than applying a uniform penalty — but
with single-seed runs on both sides of the fix, some of this could be ordinary
retraining variance rather than the fix's effect. Re-running with multiple seeds (see
"What this doesn't establish") would be needed to separate the two cleanly.

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
