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

The sweep covers three regimes (low_vol, normal, high_vol), a 6-scenario (all
directed pairs) out-of-distribution transfer test, a simulator stylized-facts
audit, a CVaR alpha sweep, and an encoder ablation slice — see "high_vol results,
OOD transfer, and stylized facts" and "CVaR sweep and encoder ablation" below.

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
- **Seed:** 42 only, everywhere, except the CVaR alpha sweep below (4 seeds). No
  other multi-seed replication — see "What this doesn't establish" below.
- **CVaR alpha:** 0.25 throughout for the main sweep; a full sweep
  (α ∈ {0.05, 0.10, 0.25, 0.50, 1.0}) was also run for QR-DQN and IQN, `normal` regime
  only, at 4 seeds each (42, 1042, 2042, 3042) — see "CVaR sweep and encoder ablation"
  below.
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
| `ood_transfer_summary.csv` | Per-(agent, scenario) in-distribution vs. OOD held-out Sharpe, all 6 directed regime pairs (30 rows: 6 scenarios × 5 agents), plus a percentage-degradation figure that's unstable near a near-zero in-distribution baseline (see below — several rows exceed 1000%; treat those as "collapsed to negative," not literally). |
| `stylized_facts.json` | 5-check stylized-facts audit (fat tails, volatility clustering, spread autocorrelation, price impact, queue-imbalance predictability) of the raw simulator under a random policy, once per regime. |
| `cvar_sweep_holdout.csv` | Per-episode held-out results for QR-DQN + IQN at α ∈ {0.05, 0.10, 0.25, 0.50, 1.0} × seed ∈ {42, 1042, 2042, 3042}, `normal` regime — 40 (agent, α, seed) combinations × 15 episodes each. |
| `cvar_sweep_as_recovery.csv` | AS-recovery R² for all 40 (agent, α, seed) combinations. |
| `cvar_efficient_frontier_per_seed.csv` | Mean held-out P&L and CVaR₀.₁₀ per (agent, α, seed), computed directly from `cvar_sweep_holdout.csv` (not from in-sample train/eval history). |
| `cvar_efficient_frontier.csv` | The above aggregated to mean ± std across the 4 seeds, per agent × α — this is the actual efficient-frontier table (genuine seed variance, not a single training run per point). |
| `encoder_ablation_holdout.csv` | Per-episode held-out results for all 8 (agent, encoder) combinations — DQN/PPO/QR-DQN/IQN × {cnn, autoencoder}, `normal` regime, `variant`/`agent`/`encoder` columns on every row — evaluated in one process by `scripts/run_encoder_ablation_eval.py` (completes the 17-variant grid; handcrafted is pulled in from `holdout_eval.csv`, same seed block). The dashboard's one encoder-ablation section is built directly from this file, covering all 4 agents in a single chart and table pair. |
| `encoder_ablation_as_recovery.csv` | AS-recovery R² for all 8 combinations. |
| `encoder_ablation_significance.csv` | Paired *t*-test/Wilcoxon, each combination vs. that same agent's own handcrafted baseline (from `holdout_eval.csv`), same shared seeds. |
| `encoder_ablation_summary.csv` | Held-out Sharpe mean/std per (agent, encoder) combination (8 rows). |
| `recurrent_variant_holdout.csv` | Per-episode held-out results, snapshot-encoder policy vs. the same agent with an LSTM(128) backbone (`agents/recurrent_base.py`) in its place, across 10 variants: QR-DQN in all 3 regimes, DQN/IQN/PPO in `normal`, and QR-DQN/`normal` again at α ∈ {0.05, 0.10, 0.50, 1.00} (the CVaR-interaction extension; α=0.25 is the base `qrdqn_normal` row). Same 15-episode held-out block (`+80042..+80056`) used everywhere else. |
| `recurrent_variant_as_recovery.csv` | AS-recovery R² for all 10 recurrent variants. |
| `recurrent_variant_significance.csv` | Paired *t*-test/Wilcoxon, each recurrent variant vs. its own non-recurrent snapshot baseline, same shared seeds. |
| `cvar_sweep_holdout_qrdqn_alpha_baseline_corrected.csv` | QR-DQN non-recurrent holdout re-eval at α ∈ {0.05, 0.10, 0.50, 1.00}, seed=42, on the corrected `+80042..+80056` block — the baseline the 4 recurrent-interaction rows above are paired against (see "Bugs found" below for why this exists as a separate file from `cvar_sweep_holdout.csv`). |

Reproduce via `scripts/run_analysis.py` (AS-recovery + model comparison),
`scripts/run_holdout_eval.py --n_episodes 15` (holdout eval, all 3 regimes),
`scripts/run_significance_test.py` (paired significance vs. GLFT from holdout_eval.csv),
`scripts/run_ood_transfer.py --n_episodes 15` (OOD transfer, all 6 directed regime pairs),
`scripts/run_stylized_facts.py` (simulator audit),
`scripts/run_cvar_sweep_eval.py --n_episodes 15` (CVaR alpha sweep eval),
`scripts/run_encoder_ablation_eval.py --n_episodes 15` (encoder ablation eval, all 8 agent x encoder combinations in one process), and
`scripts/run_recurrent_variant_eval.py --n_episodes 15` (recurrent vs. snapshot, all 10 variants).
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

**OOD transfer (all 6 directed regime pairs, no fine-tuning)** — the original pass
covered 3 of 6 possible directed pairs; the other 3 (low_vol→normal, high_vol→normal,
high_vol→low_vol) were added later to complete the grid. **Result: QR-DQN is now
confirmed positive in all 6 of 6 directed pairs, not just the original 3 — and in
each of the 3 added pairs it doesn't just hold up, it improves on its own
in-distribution number.** Full results:

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
| high_vol → low_vol | ppo | −1.42 | +0.05 | +1.46 (improves) |
| high_vol → low_vol | sarsa | −1.10 | +0.22 | +1.33 (improves) |
| high_vol → low_vol | iqn | −0.05 | +0.03 | +0.09 (improves) |
| high_vol → low_vol | **qrdqn** | +1.79 | **+2.86** | **+1.07 (improves)** |
| high_vol → low_vol | dqn | +2.93 | +2.56 | −0.37 |
| high_vol → normal | iqn | −0.05 | +0.31 | +0.37 (improves) |
| high_vol → normal | ppo | −1.42 | −0.18 | +1.24 (improves, still negative) |
| high_vol → normal | sarsa | −1.10 | −0.22 | +0.88 (improves, still negative) |
| high_vol → normal | **qrdqn** | +1.79 | **+2.97** | **+1.18 (improves)** |
| high_vol → normal | dqn | +2.93 | +2.30 | −0.63 |
| low_vol → normal | dqn | +0.04 | +0.08 | +0.04 |
| low_vol → normal | iqn | +0.43 | +0.68 | +0.25 (improves) |
| low_vol → normal | **qrdqn** | +2.85 | **+3.65** | **+0.80 (improves)** |
| low_vol → normal | sarsa | +0.58 | +0.50 | −0.08 |
| low_vol → normal | ppo | +0.22 | −0.41 | −0.63 (the only added-pair case that gets meaningfully worse) |

The clearest pattern across all 6 pairs: transferring **into high_vol** is the hard
direction for every agent (the worst degradations all land there). Transferring
**out of high_vol** into a calmer regime, conversely, tends to *help* — a policy
that learned to handle the noisiest regime generally does fine, sometimes better,
once conditions calm down. QR-DQN benefits from this most cleanly (improves in
every added pair), while PPO's low_vol→normal result is the one case among the
3 added pairs where an agent gets meaningfully worse rather than flat or better.
The `degradation_pct` column in `ood_transfer_summary.csv` blows up for agents with
a near-zero in-distribution baseline (e.g. dqn low_vol→high_vol at +0.04
in-distribution, or iqn high_vol→normal at −0.05, make any Sharpe change a huge
percentage) — use the absolute Δ or the table above instead.

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
regime, 4 seeds per α: 42, 1042, 2042, 3042)** — the naive expectation is a smooth
efficient frontier: lower α (more risk-averse) trades lower mean P&L for better
tail protection (higher CVaR₀.₁₀), converging to a risk-neutral mean-maximizer at
α=1.0. The 4-seed result does not show that. **QR-DQN shows no return-for-tail
trade at all:** mean P&L *rises* with α (+777 ± 511 at α=0.05, +675 ± 523 at 0.10,
+1000 ± 309 at 0.25, +1232 ± 567 at 0.50, +1845 ± 242 at α=1.0), while CVaR₀.₁₀ is
best at the interior values (+550 ± 404 at α=0.50, +445 ± 130 at 0.25), still
positive at the risk-neutral α=1.0 (+373 ± 532), and worst at the most
conservative α=0.05 (−80 ± 217); mean Sharpe peaks at α=0.25 (3.89 ± 0.27). The
most tail-averse setting is therefore not the best tail, and the risk-neutral one
is not the worst. **IQN behaves the way the method predicts at one end only:**
α=1.0 has both the highest mean P&L (+574 ± 891) and the most negative CVaR₀.₁₀
(−2795 ± 2014), but α=0.05 has the lowest mean P&L (−45 ± 347) and a CVaR₀.₁₀ of
−1745 ± 2003, worse than α=0.10 (−784 ± 748), so the conservative end does not buy
the best tail; the interior values are noisy (std ≥ the gap to the neighbouring α
throughout, e.g. α=0.25: +348 ± 942, CVaR₀.₁₀ −1559 ± 2297). Per-seed standard
deviations are large relative to the differences between adjacent α for both
agents, so the honest summary is: at this training budget and seed count CVaR-α
does not produce a clean, monotonic return-versus-tail-risk frontier. See
`cvar_efficient_frontier.csv` (aggregate mean ± std across the 4 seeds) and
`cvar_efficient_frontier_per_seed.csv` (per-seed points) for the full tables.
(The α≠0.25 rows were regenerated after `training/evaluate.py::load_agent` was
fixed to restore each checkpoint's trained CVaR α — see "Bugs found".)

## Encoder ablation (DQN, PPO, IQN)

Completes the 17-variant architecture grid: DQN, PPO, and IQN were each trained
with the same 2 learned snapshot encoders (`cnn`, `autoencoder`) already
ablated on QR-DQN above, compared against each agent's own existing
handcrafted result, same 15-episode held-out block.

**The "handcrafted wins" finding does not replicate uniformly across agents.**
It holds strongly for DQN — `cnn` (held-out Sharpe −0.40) and `autoencoder`
(+0.02) are both significantly worse than DQN's own handcrafted baseline
(+3.65; paired Wilcoxon, both p<0.0001) — matching QR-DQN's pattern exactly.
It does **not** hold for PPO or IQN: neither learned encoder differs
significantly from handcrafted for either agent (`ppo_cnn` p=0.12, `ppo_autoencoder`
p=0.30, `iqn_cnn` p=0.64, `iqn_autoencoder` p=0.93). This isn't because the
learned encoders perform better for these agents — it's because PPO's and
IQN's own handcrafted baselines are themselves weak in this regime (PPO −0.28,
IQN +0.35, vs. QR-DQN's +4.19 and DQN's +3.65), leaving too small a gap for a
learned-vs-handcrafted difference to be statistically distinguishable at
n=15. Note also that DQN's training-time "best eval Sharpe" for `autoencoder`
showed a clearly negative number (−0.57) that the real held-out eval doesn't
reproduce in magnitude (actual mean +0.02, still significantly worse than
handcrafted but nowhere near as bad as training-time suggested) — another
instance of this project's standing rule that training-time numbers aren't
trustworthy in isolation.

**Net finding:** hand-engineered features never do *worse* than a learned
encoder for any of the 5 agents tested, but "significantly better" is
agent-dependent — it tracks which agents achieve a genuinely strong
handcrafted policy to begin with (QR-DQN, DQN), not a universal property of
learned representations losing to engineered ones.

AS-recovery for all 6 new combinations: NO RECOVERY or INSUFFICIENT DATA (too
few distinct inventory levels visited) — decoupled from the Sharpe result
exactly as seen elsewhere in this project (e.g. DQN/cnn significantly loses on
Sharpe yet shows the same near-zero recovery, GLFT R²≈0.01, as DQN's own
near-zero-recovery handcrafted baseline, R²=0.16). See
`encoder_ablation_holdout.csv`, `encoder_ablation_significance.csv`,
and `encoder_ablation_as_recovery.csv` (filter to `agent != qrdqn`) for the full tables.
[Note: these findings predate the private-feature fix (see memory/docs) and the
encoder-ablation script merge — numbers here will change once the current eval
chain's rerun lands; don't cite this paragraph until then.]

## Recurrent vs. snapshot

**Each agent's standard snapshot-encoder policy vs. the same agent with an
LSTM(128) backbone in its place** (`agents/recurrent_base.py`, 30-step sequence
windows) — architecturally a different axis from the encoder ablation above (the
LSTM lives inside the agent's network, not as a swappable external encoder; see
CLAUDE.md). 10 variant pairs total, all on the same 15-episode held-out block:
QR-DQN in all 3 regimes, DQN/IQN/PPO in `normal` (6 base combinations), plus
QR-DQN/`normal` again across the CVaR alpha sweep (0.05, 0.10, 0.50, 1.00 — 0.25
is the base `qrdqn_normal` row) to test whether the recurrent effect compounds
with the CVaR objective.

**Base 6: recurrent never significantly helps, and three times significantly hurts.**
`qrdqn_normal` (+4.19 snapshot → +1.34 recurrent), `dqn_normal` (+3.65 → +0.35) and
`iqn_normal` (+0.01 → −0.63) reach significance (paired Wilcoxon p=0.0001, 0.0001
and 0.041), all in the direction of recurrent being *worse*. The other 3
(`qrdqn_low_vol`, `qrdqn_high_vol`, `ppo_normal`) show no significant difference
either way.

**CVaR-alpha extension: recurrent QR-DQN is significantly worse at every α tried.**
Evaluated at each checkpoint's own trained α, recurrent loses at α=0.05 (+1.56
snapshot → −0.06 recurrent, p=0.018), α=0.10 (+2.37 → +0.42, p=0.005), α=0.50
(+3.76 → +2.73, p=0.026) and α=1.00 (+3.11 → +0.84, p=0.0001). Overall, across the 10
pairs, recurrent significantly helps in none and significantly hurts seven. Note
that training-time best-eval-Sharpe had suggested recurrent won at α=0.10
(1.878 vs. 1.805); genuine held-out evaluation on identical market paths shows
the opposite.

AS-recovery decouples from the Sharpe result throughout, same pattern as the
encoder ablation: recurrent's GLFT R² is sometimes much higher than its snapshot
counterpart (`qrdqn_low_vol`: 0.50→0.85 STRONG; `dqn_normal`: 0.16→0.45; the
α=0.10 recurrent variant: 0.72, STRONG, despite losing on Sharpe) and sometimes
much lower (`qrdqn_normal`: 0.32→0.02; `qrdqn_high_vol`: 0.35→0.17; `iqn_normal`:
0.45→0.13; the α=1.00 recurrent variant: 0.01, NO RECOVERY) — no consistent
direction, and no correlation with the Sharpe result.

**Net finding:** for this project's training budget, an LSTM backbone doesn't
reliably buy anything a snapshot encoder doesn't already provide, its effect on
QR-DQN specifically is negative at every CVaR α tested, and it measurably hurts
QR-DQN, DQN and IQN in the `normal` regime. See `recurrent_variant_holdout.csv`,
`recurrent_variant_significance.csv`, and `recurrent_variant_as_recovery.csv` for
the full tables.

## Bugs found and fixed during this sweep

**`load_agent` never restored a checkpoint's CVaR α — every post-hoc evaluation of an
α≠0.25 QR-DQN/IQN checkpoint acted at α=0.25.** `training/evaluate.py::load_agent`
built QR-DQN/IQN with the constructor default `cvar_alpha=0.25`; the trained α is
stored only in the checkpoint's saved config, which it never read. Training and the
evaluations run during training were unaffected (the training factory passes α), as
were all α=0.25 results (reproduced bit-exactly after the fix). Affected: the α∈{0.05,
0.10, 0.50, 1.0} rows of the CVaR sweep (holdout, AS-recovery, efficient frontier) and
the four recurrent QR-DQN α variants. Confirmed by loading the α=0.05 checkpoint with
α forced to 0.25 and reproducing the old holdout P&Ls to the digit. Fixed by restoring
α from the checkpoint config (`load_agent(..., cvar_alpha=None)`; an explicit value
overrides; regression test in `tests/test_load_agent_alpha.py`), then re-running the
affected evaluations (`run_cvar_sweep_eval.py --alphas ... --out_suffix`, merged with
`--merge_suffixes`; `run_recurrent_variant_eval.py --only ...`, then `--sig_only`).
The corrected numbers change the sweep's picture (QR-DQN mean P&L now rises with α
rather than peaking at 0.25; the recurrent-vs-snapshot α=0.50 "win" disappears and
becomes a loss), so the sweep and recurrent sections above reflect the corrected data.

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

**`scripts/run_cvar_sweep_eval.py`'s holdout seed block silently diverged from
every other holdout eval in this project** (caught while adding the recurrent ×
CVaR-alpha interaction variants above: the paired significance test against
`cvar_sweep_holdout.csv` failed for all 4 new variants with zero shared seeds).
Root cause: the holdout-seed list comprehension was `[HOLDOUT_OFFSET + i for i
in range(n)]` — missing the project-standard `BASE_SEED +` term that
`scripts/run_holdout_eval.py` and `scripts/run_recurrent_variant_eval.py` both
include, so it produced seeds 80000-80014 instead of the standard 80042-80056.
Not a correctness bug for the CVaR sweep's *own* internal conclusions (every
one of its 40 combos used the same wrong-but-shared block, so the mean±std
comparisons across α and the efficient frontier are self-consistent) — but it
silently broke paired comparison against every other file in this project that
uses the standard block. Fixed in the script (now `BASE_SEED + HOLDOUT_OFFSET + i`); the full 40-combo
holdout re-eval against the corrected block was then re-run to regenerate
`cvar_sweep_holdout.csv` and `cvar_efficient_frontier*.csv` (AS-recovery numbers
were unaffected either way — AS-recovery uses its own fixed
`AS_RECOVERY_SEED=95000`, not the holdout block, so `cvar_sweep_as_recovery.csv`
didn't need to change). The targeted
`cvar_sweep_holdout_qrdqn_alpha_baseline_corrected.csv` (described in Files
above) was a small, cheap stopgap computed before that full redo finished,
specifically to unblock the recurrent-interaction significance test above —
kept as a historical record even though `cvar_sweep_holdout.csv` now carries
the same (agent=qrdqn, alpha, seed=42) rows on the corrected block too.

## What this doesn't establish

- **No multi-seed replication outside the CVaR sweep.** Every other result (main
  campaign, encoder ablation, OOD transfer) is seed=42. The held-out significance
  test (n=15 paired episodes, now genuinely paired post-seeding-fix) is real evidence
  within that one seed's trained policy, but doesn't rule out seed-to-seed variance
  in which policy training converges to. The CVaR alpha sweep is the one exception —
  see above — and its own multi-seed result is exactly the cautionary tale this bullet
  describes: the single-seed version's "non-monotonic frontier" finding did not survive
  3 more seeds.
- **The full 17-variant architecture grid (1 SARSA + 4 neural agents × {handcrafted,
  cnn, autoencoder, recurrent}) is now complete, but `normal`-regime and
  single-seed only.** cnn/autoencoder now cover all 5 agents (QR-DQN's row plus
  DQN/PPO/IQN added later — see "Encoder ablation (DQN, PPO, IQN)" above), and
  the recurrent axis separately covers QR-DQN in all 3 regimes plus
  DQN/IQN/PPO in `normal` and QR-DQN/`normal` across the CVaR alpha sweep (see
  "Recurrent vs. snapshot" above) — but every cell in the grid is still a single
  training seed, and the cnn/autoencoder cells outside `normal` were never run at
  all (only the recurrent axis was extended to low_vol/high_vol, and only for
  QR-DQN). Precisely: *ablated the full 17-variant encoder/recurrent architecture
  grid in the `normal` regime, and separately benchmarked 5 agent architectures
  across 3 volatility regimes* — these are two distinct axes, not a 17×3 cross
  product.
- **OOD transfer now covers all 6 possible directed regime pairs, but still only
  one seed each (seed=42).** All 6 directed pairs among low_vol/normal/high_vol
  are tested (the original 3 plus low_vol→normal, high_vol→normal, and
  high_vol→low_vol, added later — see above), and QR-DQN's robustness holds up
  across every one of them. What's still missing is replication: no seed variance
  to confirm this ranking (or the "transferring out of high_vol tends to help"
  pattern) holds under a different training seed for the underlying checkpoints.

## Dashboard

An interactive visualization of all of the above (training/eval curves, loss curves,
peak-vs-final drift analysis, the held-out comparison, AS-recovery, significance
table, OOD transfer, CVaR efficient frontier, encoder ablation, recurrent vs.
snapshot, the reward comparison (our adverse-inventory penalty vs. its
reversed-sign comparator), and the stylized-facts audit) is checked into this
directory as a self-contained
[`dashboard.html`](dashboard.html) — open it directly in a browser, no server or
build step needed. It's also published as a Claude artifact and kept up to date in
place rather than re-published as a new link — ask in a Claude Code session with
access to this project's conversation history for the current URL. To regenerate
`dashboard.html` after new results land, rebuild `viz_data.json` via
`scripts/build_dashboard_data.py` (assembles it from
`results/*.csv`, `stylized_facts.json`, and
`logs/*/train_history.json` / `eval_history.json`), splice it into the dashboard HTML
template, and save the result back over `dashboard.html`.
