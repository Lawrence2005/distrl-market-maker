# Round 3–4 — scaled-down local sweep

Results from the local (non-HPC) training sweep referenced in `evaluation/README.md`
as the current Week 8 model-comparison deliverable. **This is still not the full
17-variant state-representation ablation** described in the top-level `README.md`
(Week 4/5/8) — Round 4 added a 3-encoder QR-DQN slice (cnn, autoencoder, recurrent,
all `normal` regime only, see "CVaR sweep and encoder ablation" below) but the other
4 agents' recurrent variants and the full 17-cell grid remain untrained. That larger
ablation remains future work; it was explicitly descoped for local single-machine
hardware (WSL, ~9.7GB RAM), not attempted and abandoned.

Round 4 extended Round 3's low_vol/normal sweep with a third regime (`high_vol`), an
out-of-distribution transfer test, a simulator stylized-facts audit, a CVaR alpha
sweep, and an encoder ablation slice — see "Round 4 additions" and "CVaR sweep and
encoder ablation" below.

## Scope

- **Agents:** sarsa, dqn, ppo, qrdqn, iqn (5) — the non-recurrent, handcrafted-encoder
  variant of each.
- **Regimes:** low_vol, normal, high_vol (all three, as of Round 4). The
  megashock-based `flash_crash` was run earlier in the project but dropped from scope
  (see git history and `docs/design_decisions/README.md`); `trending` was removed at
  the same time.
- **Baselines:** fixed-spread, Avellaneda-Stoikov, GLFT — fill-intensity-calibrated
  per regime against this exact ABIDES instance (`scripts/calibrate_fill_intensity.py`;
  calibration constants documented in `scripts/run_baseline.py`).
- **Seed:** 42 only, everywhere. No multi-seed replication — see "What this doesn't
  establish" below.
- **CVaR alpha:** 0.25 throughout for the main sweep; Round 4 added a full sweep
  (α ∈ {0.05, 0.10, 0.25, 0.50, 1.0}) for QR-DQN and IQN, `normal` regime only — see
  "CVaR sweep and encoder ablation" below.
- **Encoder:** `handcrafted` throughout the main sweep; Round 4 added QR-DQN ×
  {cnn, autoencoder, recurrent} at α=0.25, `normal` regime only — see below.
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
| `ood_transfer_episodes.csv` | Per-episode raw results: each agent's already-trained **low_vol** best checkpoint, rolled out greedily in the **high_vol** environment on 15 fresh seeds (`+90000` offset, disjoint from every other seed range used anywhere in this project). |
| `ood_transfer_summary.csv` | Per-agent in-distribution (low_vol) vs. OOD (high_vol) held-out Sharpe, absolute Δ, and a percentage-degradation figure that's unstable near zero (see "Round 4 additions" below — use the absolute Δ). |
| `stylized_facts.json` | 5-check stylized-facts audit (fat tails, volatility clustering, spread autocorrelation, price impact, queue-imbalance predictability) of the raw simulator under a random policy, once per regime. |
| `cvar_sweep_holdout.csv` | Per-episode held-out results for QR-DQN + IQN at α ∈ {0.05, 0.10, 0.50, 1.0} (α=0.25 pulled in from `holdout_eval.csv`, same seed block), `normal` regime. |
| `cvar_sweep_as_recovery.csv` | AS-recovery R² for QR-DQN + IQN at all 5 α values. |
| `cvar_efficient_frontier.csv` | Mean held-out P&L and CVaR₀.₁₀ per agent × α, computed directly from `cvar_sweep_holdout.csv` (not from in-sample train/eval history). |
| `encoder_ablation_holdout.csv` | Per-episode held-out results for QR-DQN with cnn/autoencoder/recurrent encoders at α=0.25 (handcrafted pulled in from `holdout_eval.csv`, same seed block), `normal` regime. |
| `encoder_ablation_as_recovery.csv` | AS-recovery R² for the 3 new encoders. |
| `encoder_ablation_significance.csv` | Paired *t*-test/Wilcoxon of each new encoder vs. handcrafted QR-DQN, same 15 shared seeds. |
| `encoder_ablation_summary.csv` | Held-out Sharpe mean/std per encoder (4 rows: handcrafted + 3 new). |

Reproduce via `scripts/run_analysis.py` (AS-recovery + model comparison),
`scripts/run_holdout_eval.py --n_episodes 15` (holdout eval, all 3 regimes),
`scripts/run_significance_test.py` (paired significance vs. GLFT from holdout_eval.csv),
`scripts/run_ood_transfer.py --n_episodes 15` (OOD transfer),
`scripts/run_stylized_facts.py` (simulator audit),
`scripts/run_cvar_sweep_eval.py --n_episodes 15` (CVaR alpha sweep eval), and
`scripts/run_encoder_ablation_eval.py --n_episodes 15` (encoder ablation eval).
`scripts/build_dashboard_data.py` assembles all of the above plus
`logs/*/train_history.json` /`eval_history.json` into the dashboard's `viz_data.json`.

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

## Round 4 additions: high_vol, OOD transfer, stylized facts

**high_vol held-out results** — DQN and IQN dominate this regime by a wide margin:
held-out Sharpe +4.37 (DQN) and +4.63 (IQN), versus **every baseline strongly
negative** (fixedspread -2.21, AS -1.84, GLFT -2.51). QR-DQN, PPO, and SARSA are all
still negative in absolute terms (-0.16, -1.56, -1.59) but less negative than any
baseline. The paired significance test reflects this: **all 5 RL agents clear
p<0.05 against GLFT in high_vol** — but this is driven as much by GLFT performing
unusually poorly in this regime as by the weaker RL agents doing well; a bare "5/5
significant" headline overstates how strong QR-DQN/PPO/SARSA's high_vol policies
actually are in absolute terms. DQN and IQN's wins are the real result here.

**OOD transfer (low_vol-trained checkpoint → high_vol environment, no fine-tuning)**
— every agent's already-weak low_vol held-out Sharpe (all within ±0.5 of zero
post-kappa-fix) degrades further under the regime shift: IQN least (Δ≈-0.17),
DQN (Δ≈-0.44), QR-DQN (Δ≈-0.78), SARSA (Δ≈-1.84), PPO most (Δ≈-2.49). This is a
directional match to the original hypothesis that IQN's CVaR-tail training
objective generalizes better than the others. The `degradation_pct` column in
`ood_transfer_summary.csv` is included for reference but is not a reliable number —
dividing by a near-zero in-distribution baseline produces numbers like PPO's
+68440%; use the absolute Δ instead. Also worth noting: **this OOD result cannot
be cleanly separated from the fact that these agents' low_vol in-distribution
performance is itself weak** — a policy that isn't doing much of anything in-sample
has less "real" performance to lose out-of-sample, so a small absolute Δ isn't
purely a generalization story.

**Stylized facts** — the live ABIDES-Gym simulator passes 5 well-known microstructure
regularities (fat tails, volatility clustering, spread autocorrelation, price impact,
queue-imbalance predictability) under a random policy in all 3 regimes: normal and
high_vol pass all 5; low_vol passes 4/5, failing only queue-imbalance predictability
(the mildest of the 5 checks by design). Separately, and importantly: **real LOBSTER
tick data was never used anywhere in this project.** `data/lobster/` has been empty
for the project's entire duration (confirmed via `git log` — no commit ever added a
file there), and the project's own background-agent calibration parameters (fit from
a synthetic-LOBSTER generator, not real exchange data — originating commit literally
titled "w02: Synthetic data generated and processed") were never consumed by the live
simulator either, which uses ABIDES's own rmsc04 background config unmodified; that
whole calibration pipeline has since been archived to `archive/lobster_calibration/`.
The stylized-facts checks above are the closest substitute available for the
calibration step that never happened, not a replacement for it.

## CVaR sweep and encoder ablation

**CVaR alpha sweep (QR-DQN + IQN × α ∈ {0.05, 0.10, 0.25, 0.50, 1.0}, `normal`
regime)** — the naive expectation is a smooth efficient frontier: lower α (more
risk-averse) trades lower mean P&L for better tail protection (higher CVaR₀.₁₀),
converging to a risk-neutral mean-maximizer at α=1.0. **The data doesn't show
that.** For QR-DQN, α=0.25 dominates every other α on *both* axes simultaneously
(mean held-out P&L +985, CVaR₀.₁₀ +639 — both the best of the 5 values), while
α=1.0 (nominally risk-neutral) is one of the worst on both (mean −14, CVaR −515).
For IQN, there's no visible relationship with α at all: mean P&L ranges from +293
(α=0.05) down to −951 (α=0.50) with no monotonic trend, and CVaR₀.₁₀ is deeply
negative (−3800 to −6100) across every α value. With one training seed per α, this
is most plausibly ordinary seed-to-seed training variance dominating any real
CVaR-alpha effect at this sample size, not evidence that the CVaR mechanism itself
doesn't work — but it's also not evidence that it produces the textbook frontier.
A genuine answer needs multiple seeds per α to separate signal from noise; see
`cvar_efficient_frontier.csv` for the full table.

**Encoder ablation (QR-DQN × {handcrafted, cnn, autoencoder, recurrent}, α=0.25,
`normal` regime)** — a clean, statistically significant result: **handcrafted
features beat every learned-representation alternative by a wide margin.**
Held-out Sharpe: handcrafted +3.82, cnn +0.48, autoencoder +0.003, recurrent
−0.25 — all three differences vs. handcrafted are significant at p<0.001 (paired
Wilcoxon, n=15). For this project's training budget and reward design, none of
the alternative representations tried come close to the hand-engineered feature
vector. A data-quality caveat worth flagging rather than glossing over: **cnn and
autoencoder both showed a minority of held-out episodes at exactly zero Sharpe**
(2/15 and 6/15 respectively) — the signature of a policy that never participates
in the market for an entire episode, not just a bad one. Handcrafted and recurrent
showed no such episodes. Since cnn and autoencoder are the two encoders that
consume the raw LOB depth snapshot (instead of handcrafted's engineered features),
this looks like an issue specific to that raw-snapshot input path rather than the
autoencoder's reconstruction quality specifically (its pretraining reconstruction
loss was excellent, ≈1.3e-8) — worth a follow-up look at how the CNN/AE-encoded
state interacts with ε-greedy exploration or the replay buffer before concluding
raw-snapshot encoders are simply worse, since a policy that goes silent for a
third of its held-out episodes is a different failure mode than "learns a worse
policy."

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
- **CVaR alpha sweep and encoder ablation are both single-seed, `normal`-regime
  only.** The CVaR sweep's non-monotonic, dominated-frontier result (see above)
  is exactly the kind of finding multi-seed replication would need to confirm
  isn't just noise — as-is, it can't distinguish "CVaR-alpha has no clean effect
  here" from "one unlucky/lucky training seed per alpha." The encoder ablation
  covers only QR-DQN, not the other 4 agents' recurrent variants or the full
  17-variant grid.
- **OOD transfer is one direction only, one seed.** Only low_vol→high_vol was tested
  (not normal→high_vol, high_vol→low_vol, etc.), and like everything else here, only
  at seed=42 — no replication to confirm the IQN/PPO generalization ranking holds
  under a different training seed.
- **The cnn/autoencoder zero-Sharpe episodes are unexplained.** Flagged above as a
  real anomaly (2/15 and 6/15 episodes respectively at exactly zero Sharpe) — this
  session didn't dig into root cause, so treat the cnn/autoencoder Sharpe numbers as
  provisional pending that investigation, not just "worse than handcrafted."

## Dashboard

An interactive visualization of all of the above (training/eval curves, loss curves,
peak-vs-final drift analysis, the held-out comparison, AS-recovery, significance
table, OOD transfer, and the stylized-facts audit) was built as a Claude artifact
during this sweep and is kept up to date in place rather than re-published as a new
link — ask in a Claude Code session with access to this project's conversation
history for the current URL, or rebuild it via `scripts/build_dashboard_data.py`
(assembles `viz_data.json` from `evaluation/results_scaled_down_sweep/*.csv`,
`stylized_facts.json`, and `logs/*/train_history.json` / `eval_history.json`) and the
dashboard HTML template.
