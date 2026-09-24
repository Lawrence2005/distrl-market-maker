# Scaled-down local sweep

Results from the local (non-HPC) training sweep referenced in `evaluation/README.md`
as the current Week 8 model-comparison deliverable. **This is still not the full
17-variant state-representation ablation** described in the top-level `README.md`
(Week 4/5/8) — a 3-encoder QR-DQN slice (cnn, autoencoder, recurrent, all `normal`
regime only, see "CVaR sweep and encoder ablation" below) has been trained and
evaluated, but the other 4 agents' recurrent variants and the full 17-cell grid
remain untrained. That larger ablation remains future work; it was explicitly
descoped for local single-machine hardware (WSL, ~9.7GB RAM), not attempted and
abandoned.

The sweep covers three regimes (low_vol, normal, high_vol), a 3-scenario
out-of-distribution transfer test, a simulator stylized-facts audit, a CVaR alpha
sweep, and an encoder ablation slice — see "high_vol results, OOD transfer, and
stylized facts" and "CVaR sweep and encoder ablation" below.

**Numbers below are post-ABIDES-seeding-fix (see "Bugs found and fixed")** — every
holdout/AS-recovery/significance/CVaR/encoder-ablation/OOD number in this directory
was recomputed after that fix, and none of it should be compared against anything
from before it (that included both a stale loss-scale bug and, unknown until this
pass, real market seeding that never worked at all — see below).

## Scope

- **Agents:** sarsa, dqn, ppo, qrdqn, iqn (5) — the non-recurrent, handcrafted-encoder
  variant of each.
- **Regimes:** low_vol, normal, high_vol (all three). The
  megashock-based `flash_crash` was run earlier in the project but dropped from scope
  (see git history and `docs/design_decisions/README.md`); `trending` was removed at
  the same time.
- **Baselines:** fixed-spread, Avellaneda-Stoikov, GLFT — fill-intensity-calibrated
  per regime against this exact ABIDES instance (`scripts/calibrate_fill_intensity.py`;
  calibration constants documented in `scripts/run_baseline.py`).
- **Seed:** 42 only, everywhere. No multi-seed replication — see "What this doesn't
  establish" below.
- **CVaR alpha:** 0.25 throughout for the main sweep; a full sweep
  (α ∈ {0.05, 0.10, 0.25, 0.50, 1.0}) was also run for QR-DQN and IQN, `normal` regime
  only — see "CVaR sweep and encoder ablation" below.
- **Encoder:** `handcrafted` throughout the main sweep; QR-DQN ×
  {cnn, autoencoder, recurrent} was also evaluated at α=0.25, `normal` regime only —
  see below.
- **Training:** 500-episode budget, patience-based early stopping on smoothed eval
  Sharpe (patience=6 checkpoints, min_delta=0.05, smoothed over the last 3 checkpoints,
  no stopping before episode 150 — see `training/train.py`'s `_early_stop_state`).

## Files

| File | What it is |
|---|---|
| `model_comparison.csv` | Mean Sharpe over each run's *entire* logged history — diagnostic only (exploration-contaminated for RL agents; see the dashboard's methodology notes). Not the number to rank agents by. |
| `as_recovery.csv` | R² of each RL agent's realized bid/ask skew against the closed-form GLFT/AS curves, computed on each agent's **best eval checkpoint** (not latest — see "Bugs found" below), 15 greedy rollouts each. |
| `holdout_eval.csv` | Per-episode raw results: every model (5 RL agents' best checkpoint + 3 baselines) run on **15 identical, never-before-seen seeds per regime** — a fixed offset (`+80000`) clear of every training/eval/AS-recovery seed range used anywhere else. Now a genuinely paired, fair, head-to-head comparison — see "Bugs found." |
| `holdout_summary.csv` | `holdout_eval.csv` aggregated to mean/std/n per model x regime. |
| `significance_vs_glft.csv` | Paired *t*-test and Wilcoxon signed-rank test, each RL agent vs. GLFT, on the 15 shared holdout seeds (paired because both now genuinely see identical market realizations — see "Bugs found"). |
| `ood_transfer_episodes.csv` | Per-episode raw results across 3 scenarios: each agent's already-trained **low_vol** or **normal** best checkpoint, rolled out greedily in a different regime's environment with no fine-tuning, on 15 fresh seeds (`+90000` offset, disjoint from every other seed range used anywhere in this project). |
| `ood_transfer_summary.csv` | Per-(agent, scenario) in-distribution vs. OOD held-out Sharpe, plus a percentage-degradation figure that's unstable near a near-zero in-distribution baseline (see below — several rows exceed 1000%; treat those as "collapsed to negative," not literally). |
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
`scripts/run_ood_transfer.py --n_episodes 15` (OOD transfer, 3 scenarios),
`scripts/run_stylized_facts.py` (simulator audit),
`scripts/run_cvar_sweep_eval.py --n_episodes 15` (CVaR alpha sweep eval), and
`scripts/run_encoder_ablation_eval.py --n_episodes 15` (encoder ablation eval).
`scripts/build_dashboard_data.py` assembles all of the above plus
`logs/*/train_history.json` /`eval_history.json` into the dashboard's `viz_data.json`.

## Key findings

- **QR-DQN significantly beats GLFT in all three regimes** — the only agent to do so
  consistently: low_vol +2.85 vs. GLFT +1.24 (p<0.0001), normal +4.19 vs. −1.16
  (p<0.0001), high_vol +1.79 vs. −2.18 (p<0.0001). DQN beats GLFT in normal (+3.65,
  p<0.0001) and high_vol (+2.93, p<0.0001) but is significantly **worse** than GLFT
  in low_vol (−1.20 mean diff, p=0.0034) — GLFT and AS remain strong and uncontested
  in that regime (+1.24 / +1.24 held-out Sharpe), the only regime where a classical
  baseline outright wins. SARSA and PPO clear p<0.05 against GLFT in normal and
  high_vol but not low_vol; IQN clears normal and high_vol but not low_vol either.
- **In `high_vol`, all three baselines (fixed-spread, AS, GLFT) produce bit-identical
  held-out results** (Sharpe −2.1787 exactly, every single episode) — not a bug.
  Under this regime's calibration (`kappa=3.7333`), AS/GLFT's theoretically-optimal
  half-spread computes to roughly the 25–35 tick range across nearly every
  inventory/time state, which all snaps to the same point on the 10-tick action
  grid (`TICK_OFFSETS = [0,10,...,100]`) that fixed-spread is hardcoded to use (30
  ticks). The three "different" classical strategies are computationally
  indistinguishable here given the grid's coarseness — a genuine methodological
  limitation of the discretized action space at this regime's volatility, not an
  economic finding about the strategies themselves.
- **AS-recovery is highly agent/regime-specific, not correlated with held-out
  Sharpe.** IQN shows STRONG recovery in low_vol (GLFT R²=0.98) despite a modest
  Sharpe there (+0.43); QR-DQN shows only WEAK recovery in low_vol (R²=0.50) despite
  the best Sharpe of any agent in that regime (+2.85). Every agent's ppo AS-recovery
  is INSUFFICIENT DATA in all 3 regimes (too few inventory levels visited — see
  `as_recovery.csv`). Full table there.

## high_vol results, OOD transfer, and stylized facts

**high_vol held-out results** — DQN dominates this regime (+2.93), QR-DQN close
behind (+1.79); both handily beat all three (identical) baselines at −2.18. IQN is
roughly flat (−0.05); SARSA (−1.10) and PPO (−1.42) are negative but still less
negative than the baselines. The significance test reflects this: DQN, QR-DQN, IQN,
and SARSA all clear p<0.05 against GLFT in high_vol; PPO does not (p=0.23, despite
a positive mean diff) — some of this "beats GLFT" pattern is inflated by GLFT
performing unusually poorly in this regime (see the baseline-collapse note above),
not purely strong RL policies.

**OOD transfer (3 scenarios, no fine-tuning)** — **QR-DQN is the only agent that
stays profitable in every scenario tested.** Full results:

| Scenario | Agent | In-dist Sharpe | OOD Sharpe | Δ |
|---|---|---|---|---|
| low_vol → high_vol | qrdqn | +2.85 | **+2.06** | −0.79 |
| low_vol → high_vol | sarsa | +0.58 | −1.71 | −2.29 |
| low_vol → high_vol | iqn | +0.43 | −2.50 | −2.94 |
| low_vol → high_vol | ppo | +0.22 | −2.32 | −2.54 |
| low_vol → high_vol | dqn | +0.04 | −0.84 | −0.89 |
| normal → high_vol | dqn | +3.65 | **+2.40** | −1.25 |
| normal → high_vol | qrdqn | +4.19 | **+0.94** | −3.25 |
| normal → high_vol | iqn | +0.35 | −0.30 | −0.64 |
| normal → high_vol | ppo | −0.28 | −2.18 | −1.90 |
| normal → high_vol | sarsa | +0.11 | −1.37 | −1.48 |
| normal → low_vol | qrdqn | +4.19 | **+3.58** | −0.61 |
| normal → low_vol | dqn | +3.65 | **+3.29** | −0.37 |
| normal → low_vol | sarsa | +0.11 | +0.07 | −0.04 |
| normal → low_vol | iqn | +0.35 | +0.20 | −0.15 |
| normal → low_vol | ppo | −0.28 | +0.34 | +0.61 (improves) |

Transferring into a **harder** regime (either scenario ending in high_vol) is where
most agents collapse to negative Sharpe — only QR-DQN and DQN stay positive, and
only when their in-distribution performance was already strong. Transferring into
an **easier** regime (normal → low_vol) costs the strong agents almost nothing
(QR-DQN −15%, DQN −10%) and PPO actually improves. The `degradation_pct` column in
`ood_transfer_summary.csv` blows up for agents with a near-zero in-distribution
baseline (e.g. dqn low_vol→high_vol at +0.04 in-distribution makes any negative
OOD number a huge percentage) — use the absolute Δ or the table above instead.

**Stylized facts** — the live ABIDES-Gym simulator passes 4-5 of 5 well-known
microstructure regularities (fat tails, volatility clustering, spread autocorrelation,
price impact, queue-imbalance predictability) under a random policy in every regime:
normal and high_vol pass all 5; low_vol passes 4/5, failing only spread
autocorrelation. Separately, and importantly: **real LOBSTER tick data was never
used anywhere in this project.** `data/lobster/` has been empty for the project's
entire duration (confirmed via `git log` — no commit ever added a file there), and
the project's own background-agent calibration parameters (fit from a
synthetic-LOBSTER generator, not real exchange data) were never consumed by the live
simulator either, which uses ABIDES's own rmsc04 background config unmodified; that
whole calibration pipeline has since been archived to `archive/lobster_calibration/`.
The stylized-facts checks above are the closest substitute available for the
calibration step that never happened, not a replacement for it.

## CVaR sweep and encoder ablation

**CVaR alpha sweep (QR-DQN + IQN × α ∈ {0.05, 0.10, 0.25, 0.50, 1.0}, `normal`
regime)** — the naive expectation is a smooth efficient frontier: lower α (more
risk-averse) trades lower mean P&L for better tail protection (higher CVaR₀.₁₀),
converging to a risk-neutral mean-maximizer at α=1.0. **The data doesn't show
that.** For QR-DQN, α=0.10 gives the *best* CVaR₀.₁₀ of any value (+536, still
positive at the 10th percentile) while α=0.25 gives the best mean Sharpe (+4.19,
mean P&L +1298); α=0.05 — nominally the most risk-averse — is the *worst* on both
axes at once (mean P&L −76, CVaR₀.₁₀ −1544, mean Sharpe −0.06). For IQN the pattern
is similarly non-monotonic: α=0.10 gives the best CVaR₀.₁₀ (−599) while α=0.25 gives
the best mean P&L (+528); α=0.05 is again the worst on tail risk (CVaR₀.₁₀ −3912).
With one training seed per α, this is most plausibly ordinary seed-to-seed training
variance dominating any real CVaR-alpha effect at this sample size — a plausible
mechanism is that the most extreme α (0.05, focusing training on only the worst
~5% of quantile outcomes) is simply harder to train stably at this budget, not that
the CVaR mechanism doesn't work. A genuine answer needs multiple seeds per α; see
`cvar_efficient_frontier.csv` for the full table.

**Encoder ablation (QR-DQN × {handcrafted, cnn, autoencoder, recurrent}, α=0.25,
`normal` regime)** — a clean, statistically significant result: **handcrafted
features beat every learned-representation alternative by a wide margin.**
Held-out Sharpe: handcrafted +4.19, cnn +0.45, autoencoder −0.18, recurrent −0.39 —
all three differences vs. handcrafted are significant at p<0.0001 (paired Wilcoxon,
n=15). For this project's training budget and reward design, none of the
alternative representations tried come close to the hand-engineered feature vector.
A separate, interesting decoupling: **recurrent shows the strongest AS-recovery of
any encoder** (GLFT R²=0.67, STRONG recovery) despite having the *worst* held-out
Sharpe of the four — it learns AS/GLFT-consistent inventory-skew behavior more
cleanly than handcrafted does, without that translating into better P&L. cnn shows
essentially zero AS-recovery (R²≈0.00) and autoencoder shows WEAK recovery
(R²=0.32) — no correlation between an encoder's AS-recovery and its Sharpe rank.

## Bugs found and fixed during this sweep

**ABIDES real-market seeding never worked, at any point in this project's history**
(caught while investigating an AS-recovery result that changed dramatically between
two nominally-identical reruns of the same script). Root cause, confirmed by reading
the vendored ABIDES config directly: `abides-jpmc-public/abides-markets/
abides_markets/configs/rmsc04.py`'s `build_config()` defaults its `seed` parameter to
a **wall-clock-derived value** (`int(datetime.now().timestamp()*1e6) % 2**32`) when
not explicitly passed, and `envs/lob_env.py` never passed one — meaning every real
ABIDES environment construction in this project's history used an effectively random
background-agent seed, regardless of any `seed=` argument passed anywhere in this
project's own training/eval code. Confirmed via a controlled two-process experiment
(identical checkpoint + identical seed argument produced different action sequences
and rewards; ruled out `PYTHONHASHSEED` as the cause). Practical effect: episode-level
results were never reproducible across separate script invocations, and the "paired"
comparison design this whole eval methodology relies on (same seed ⇒ same market
realization ⇒ a fair agent-vs-agent/agent-vs-baseline comparison) never actually held
— every "paired" episode was secretly an independent draw. Aggregate results (15-episode
means) computed *before* this fix were still statistically valid as *unpaired* samples
(each episode a genuine, unbiased random market draw), just not reproducible and not
actually paired the way `run_significance_test.py`'s methodology assumed.

Fix: `envs/lob_env.py::LOBMarketMakingEnv.reset()` now sets
`self._abides_env.np_random = np.random.default_rng(seed)` before calling the
underlying ABIDES reset, using `gym.core.Env`'s public `np_random` setter (ABIDES's
own `reset()` draws its background-agent seed from exactly this generator via
`self.np_random.integers(...)`) — no vendored code touched. Verified: same seed
across two separate Python processes now produces byte-identical episode results;
different seeds still produce genuinely different episodes (not accidentally frozen
to one market realization). Full test suite (487 passed) plus
`tests/test_lob_env_abides.py` (11/11, the direct regression check for this code
path) both green. Every number in this directory was recomputed after this fix.

Two earlier, separate bugs in `agents/sarsa.py` were also caught while building the
results dashboard (before the seeding fix above), both fixed and not affected by it:

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

## What this doesn't establish

- **No multi-seed replication.** All results are seed=42. The held-out significance
  test (n=15 paired episodes, now genuinely paired post-seeding-fix) is real evidence
  within that one seed's trained policy, but doesn't rule out seed-to-seed variance
  in which policy training converges to.
- **CVaR alpha sweep and encoder ablation are both single-seed, `normal`-regime
  only.** The CVaR sweep's non-monotonic, dominated-frontier result (see above)
  is exactly the kind of finding multi-seed replication would need to confirm
  isn't just noise — as-is, it can't distinguish "CVaR-alpha has no clean effect
  here" from "one unlucky/lucky training seed per alpha." The encoder ablation
  covers only QR-DQN, not the other 4 agents' recurrent variants or the full
  17-variant grid.
- **OOD transfer covers 3 of the 6 possible directed regime pairs, one seed each.**
  low_vol→high_vol, normal→high_vol, and normal→low_vol were tested; low_vol→normal,
  high_vol→normal, and high_vol→low_vol were not, and like everything else here, only
  at seed=42 — no replication to confirm the QR-DQN/DQN robustness ranking holds
  under a different training seed.

## Dashboard

An interactive visualization of all of the above (training/eval curves, loss curves,
peak-vs-final drift analysis, the held-out comparison, AS-recovery, significance
table, OOD transfer, CVaR efficient frontier, encoder ablation, and the
stylized-facts audit) was built as a Claude artifact during this sweep and is kept
up to date in place rather than re-published as a new link — ask in a Claude Code
session with access to this project's conversation history for the current URL, or
rebuild it via `scripts/build_dashboard_data.py` (assembles `viz_data.json` from
`evaluation/results_scaled_down_sweep/*.csv`, `stylized_facts.json`, and
`logs/*/train_history.json` / `eval_history.json`) and the dashboard HTML template.
