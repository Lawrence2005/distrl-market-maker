# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A 10-week research project building a distributional RL market maker (QR-DQN, IQN with CVaR tail-risk objectives) evaluated against classical analytical benchmarks (Avellaneda-Stoikov, GLFT) across market regimes, inside an ABIDES-Gym limit-order-book simulator (ABIDES's own `rmsc04` background-agent config, unmodified).

The full week-by-week research plan, literature justification, and success criteria live in `README.md` — read it before making design decisions, since architectural choices here are usually driven by a specific cited paper (Spooner 2018, Sun 2022, Ganesh 2019, Dabney et al., Guéant et al., etc.), not just engineering preference. `docs/mdp_formulation.md` and `docs/evaluation_framework.md` hold the finalized MDP spec and metric definitions.

## Environment Setup

Python 3.12, virtualenv at `venv/`.

```bash
source venv/bin/activate
pip install -r requirements.txt
```

`abides-core`, `abides-markets`, `abides-gym` are a vendored fork in `abides-jpmc-public/` (not installed from PyPI — that package name is a placeholder). Install/develop them from source:

```bash
cd abides-jpmc-public && ./setup-dev.sh   # or install.sh for non-editable install
```

**Fallback behavior:** `envs/lob_env.py` catches the `abides_gym` import and, if it's missing, silently falls back to a synthetic GBM mid-price with zero fills (with a one-time `RuntimeWarning`). Code can appear to run correctly against this fallback while never touching the real simulator — if a test or experiment result looks suspicious, first confirm `abides_gym` is actually importable in the active environment.

## Commands

```bash
# Run the full test suite
pytest tests/

# Exclude slow tests that require the real ABIDES simulator (marked @pytest.mark.abides)
pytest tests/ -v -m "not abides"

# Run a single test file / test
pytest tests/test_baselines.py -v
pytest tests/test_baselines.py::TestGLFT::test_foic_quotes -v

# Pre-train the autoencoder encoder (must run before AE-encoder training)
python training/pretrain_ae.py data.path=data/lobster/AAPL_2012.csv

# Train an agent (Hydra config composition)
python training/train.py agent=qrdqn encoder=autoencoder reward=asymmetric alpha=0.10

# CVaR alpha sweep / multirun
python training/train.py agent=qrdqn encoder=autoencoder reward=asymmetric alpha=0.05,0.10,0.25,0.50,1.0 --multirun

# Evaluate a trained checkpoint (the actual CLI — not --config-path, which
# never existed; there is no experiments/<run_name>/config.yaml convention
# in this codebase, see the Architecture section)
python training/evaluate.py --checkpoint checkpoints/<run_tag>/best.pt --agent qrdqn --encoder handcrafted
```

Training config is Hydra-based (`training/configs/config.yaml`), composed from groups: `agent/` (dqn|qrdqn|iqn|ppo|sarsa), `encoder/` (handcrafted|cnn|autoencoder|lstm), `reward/` (asymmetric|quadratic|sparse), `policy/` (cvar|mean), `env/` (base + regime overlays), `variant/` (recurrent, adds LSTM). Override any group from the CLI as shown above.

## Architecture

**Data/control flow:** `envs/` (simulator) → `encoders/` (state representation) → `agents/` (policy/value learning, optionally wrapped by `agents/cvar_policy.py`) → `training/train.py` (drives episodes, writing to `logs/<run_tag>/` and `checkpoints/<run_tag>/`) → `evaluation/` (metrics, ablations, plots) consumed by `evaluation/results_scaled_down_sweep/` (CSVs/JSON) and the published dashboard.

- **`envs/`** — ABIDES-Gym extensions. `lob_env.py` is the base `LOBMarketMakingEnv` (MultiDiscrete bid/ask tick-offset action space, three reward formulations: asymmetric-η / quadratic-λ / sparse). Background agent population comes from ABIDES's own `rmsc04` config, unmodified. `multi_agent_env.py` runs N simultaneous MM agents. `stylized_facts.py` validates simulator realism post-episode.

- **`encoders/`** — three interchangeable *snapshot* (non-temporal) state representations sharing one interface (`encoder.encode(obs) -> torch.Tensor`): `handcrafted.py` (~17-dim feature vector), `cnn.py` (Conv1D over LOB depth), `autoencoder.py` (unsupervised-pretrained, frozen at RL-train time — pretrain via `training/pretrain_ae.py` first). Any neural agent selects one via `encoder=` config.

- **`agents/`** — five algorithms, not uniformly implemented: SARSA (`sarsa.py`) is a from-scratch tile-coding agent, handcrafted features only, incompatible with the neural encoders. DQN and PPO (`dqn.py`, `ppo.py`) are also from-scratch custom PyTorch (no Stable-Baselines3 dependency anywhere in this repo, despite this file previously claiming otherwise — verify against the actual imports before assuming any SB3-specific behavior/API applies here). QR-DQN and IQN (`qrdqn.py`, `iqn.py`) are custom PyTorch — these are the primary research agents, each optionally wrapped in `cvar_policy.py` for a CVaR_α tail-risk objective (risk is applied at policy-selection time, not inside the Bellman update — direct risk-sensitive RL is numerically unstable per Spooner & Savani 2020, so don't "simplify" this by folding CVaR into the loss). `recurrent_base.py` is a shared LSTM(128) backbone used by the *recurrent* variant of every neural agent (DRQN, Recurrent QR-DQN, Recurrent IQN, Recurrent PPO) — this is architecturally distinct from the snapshot encoders above: the LSTM lives inside the agent's network, not as a swappable external encoder, and requires the sequence replay buffer (`training/replay_buffer.py`, T=30-step windows) rather than the standard one.

- **Ablation surface:** the encoder/recurrent choice is a first-class scientific ablation axis (17 total variants: 1 SARSA + 4 neural agents × [3 snapshot encoders + 1 recurrent]), not an implementation detail — when touching agent or encoder code, preserve the ability to swap independently via config.

- **`baselines/`** — non-RL analytical benchmarks in an explicit lineage: `fixed_spread.py` → `avellaneda_stoikov.py` (AS 2008 closed-form) → `glft.py` (Guéant-Lehalle-Fernandez-Tapia 2012 Proposition 5, hard inventory limits, with an `xi` market-impact toggle — this file does not currently implement separately-labeled "FOIC"/"LIIC" variants despite this section previously claiming so; confirm against the actual class/function names before assuming either acronym names something real in the code). RL agents are expected to beat GLFT specifically, since it's the strictest/most-cited reference; AS-recovery (an RL agent rediscovering AS-like inventory skew in the low-vol regime) is used as a sanity check, not a target to beat.

- **`data/`** — LOBSTER tick data is not checked in (must be obtained from lobsterdata.com and placed in `data/lobster/`, gitignored). `crypto/` provides an alternative real-data source; `process_lobster.py` handles either via `--data_dir`, producing LOB depth snapshots for AE pre-training (`encoders/autoencoder.py`). This project's own synthetic-LOBSTER generator and the background-agent calibration pipeline it fed were archived — see `archive/lobster_calibration/`.

- **`evaluation/`** — `metrics.py` (Sharpe, MAP, CVaR, MDD, PnLMAP, etc.), `as_recovery.py` (fits agent quote-skew against the AS/GLFT closed form, reports R²), `efficient_frontier.py` (mean P&L vs CVaR across the α-sweep), `ablation.py` (encoder × agent ablation tables), `visualize.py`.

- **`checkpoints/`** (gitignored) and **`logs/`** hold model weights and run logs, one subfolder per run named by its run_tag (see `evaluation/ablation.py::_run_tag()`, the single source of truth for the naming convention) — `status.json`/`eval_history.json`/`train_history.json` in `logs/<run_tag>/` are what every eval script actually reads; there is no separate `experiments/` folder (an earlier planned `{week}_{agent}_{encoder}_{reward}_{regime}_{notes}` convention that this campaign never adopted — removed along with its empty skeleton dir and README).

## Working in this repo

- Most subdirectories have their own `README.md` with file-by-file summaries and citation rationale (`envs/README.md`, `agents/README.md`, `encoders/README.md`, `baselines/README.md`, `data/README.md`, `evaluation/README.md`, `training/README.md`) — check the local one before the top-level README when working within a single component.
- Comments and docstrings in this codebase frequently cite the specific paper motivating a design choice (e.g. asymmetric reward η, why CVaR is applied at the policy layer). When modifying that logic, preserve or update the citation rather than deleting it — it's load-bearing documentation for the eventual write-up, not incidental.
- `abides-jpmc-public/` is a vendored third-party fork (JPMC's public ABIDES) — treat it as external code; project-specific changes belong in `envs/`, not inside the vendored package.
