"""
training/train.py

Hydra entry point for distrl-market-maker training runs. Component
construction lives in training/factory.py; episode execution and
checkpoint I/O live in training/rollout.py.

Supports Hydra multirun for ablation sweeps:

    # Single run
    python training/train.py agent=qrdqn encoder=handcrafted reward=asymmetric seed=42

    # Multirun across agents and encoders (12 runs)
    python training/train.py \\
        agent=dqn,qrdqn,iqn \\
        encoder=handcrafted,cnn,autoencoder \\
        reward=asymmetric alpha=0.10 env.regime=high_vol seed=42 --multirun

    # Recurrent variant
    python training/train.py \\
        agent=qrdqn variant=recurrent reward=asymmetric seed=42

    # Resume an interrupted run from its latest checkpoint (same overrides
    # as the original run, so run_tag — and therefore checkpoint_dir — match)
    python training/train.py agent=qrdqn encoder=handcrafted reward=asymmetric \\
        seed=42 training.resume=true

Progress can be checked without tailing logs via logs/<run_tag>/status.json
(or scripts/status.sh for all runs at once). For multi-day unattended runs,
launch through scripts/run_agent_supervised.sh instead of a bare
`nohup ... &` — it auto-restarts with training.resume=true if the process
dies for any reason.

Architecture dispatch:
    handcrafted encoder → HandcraftedEncoder(obs_dim=18)
    cnn encoder         → CNNEncoder(from config)
    autoencoder encoder → AEEncoder.from_checkpoint(pretrained_weights)
    variant=recurrent   → agent backbone uses an LSTM (temporal memory)
    variant=null        → agent backbone uses a per-timestep linear
                           projection instead (snapshot ablation)

Online rollout loop (training/rollout.py:run_episode):
    for each episode:
        env.reset() → obs, info
        agent.reset_hidden()
        for each step:
            action  = agent.act(obs)
            obs', r, done, info = env.step(action)
            agent.observe(obs, action, r, obs', done)
            loss = agent.train_step()          ← every step
        log episode metrics (Sharpe, MAP, MDD, PnL, loss)
        if eval_episode: run evaluation rollout

Encoder input dispatch (training/rollout.py:get_encoder_input):
    HandcraftedEncoder: obs vector (18-dim) from env._get_obs()
    CNNEncoder / AEEncoder: LOB snapshot (20-dim) from info["lob_snapshot"]
"""

from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"]  = ""
os.environ["OMP_NUM_THREADS"]       = "1"
os.environ["MKL_NUM_THREADS"]       = "1"
os.environ["OPENBLAS_NUM_THREADS"]  = "1"

import torch
torch.backends.mkldnn.enabled = False  # required for IQN on this CPU

import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.lob_env import N_OFFSET_LEVELS
from training.factory import build_agent, build_encoder, build_env, wrap_policy
from training.rollout import (  # noqa: F401 — re-exported for external callers
    _NumpyEncoder,
    decode_action,
    find_best_checkpoint,
    find_latest_checkpoint,
    get_encoder_input,
    load_checkpoint,
    run_episode,
    save_checkpoint,
)

_CONFIG_PATH = str(Path(__file__).parent / "configs")


def _early_stop_state(eval_history: list[dict], window: int, min_delta: float) -> tuple[float, int]:
    """
    Replay eval_history from scratch to get (best_smoothed_sharpe,
    checkpoints_without_improvement) for the patience-based early-stopping
    rule below. Replaying (rather than persisting this state separately)
    means a resumed run's existing eval_history naturally produces the
    correct state with no separate resume-specific logic.

    Tracks a trailing mean over the last `window` eval checkpoints (not
    the latest checkpoint alone) since even an eval_episodes-averaged
    Sharpe is still noisy — this second averaging layer is what makes the
    delta comparison trustworthy enough to act on.
    """
    best = float("-inf")
    stale = 0
    for i in range(len(eval_history)):
        recent = eval_history[max(0, i - window + 1): i + 1]
        smoothed = sum(e["sharpe"] for e in recent) / len(recent)
        if smoothed > best + min_delta:
            best = smoothed
            stale = 0
        else:
            stale += 1
    return best, stale


def _write_status(
    log_dir:         Path,
    run_tag:         str,
    episode:         int,
    n_episodes:      int,
    elapsed:         float,
    metrics:         dict,
    last_checkpoint: Optional[str],
    state:           str = "running",
) -> None:
    """
    Overwrite log_dir/status.json with a one-shot summary of run progress.

    This is the file to `cat` when checking "where has this run gotten to"
    — it doesn't require tailing/grepping a multi-day nohup log.
    """
    pct = 100.0 * episode / n_episodes if n_episodes else 0.0
    eta = (elapsed / episode) * (n_episodes - episode) if episode > 0 else None

    status = {
        "run_tag":         run_tag,
        "state":           state,  # "running" | "complete"
        "episode":         episode,
        "n_episodes":      n_episodes,
        "pct_complete":    round(pct, 1),
        "elapsed_sec":     round(elapsed, 1),
        "eta_sec":         round(eta, 1) if eta is not None else None,
        "updated_at":      datetime.now().isoformat(timespec="seconds"),
        "last_checkpoint": last_checkpoint,
        "sharpe":          metrics.get("sharpe"),
        "final_pnl":       metrics.get("final_pnl"),
        "mean_loss":       metrics.get("mean_loss"),
        "epsilon":         metrics.get("epsilon"),
    }
    with open(log_dir / "status.json", "w") as f:
        json.dump(status, f, indent=2)


@hydra.main(config_path=_CONFIG_PATH, config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> float:
    """
    Main Hydra entry point.

    Called once per run. In multirun mode, Hydra calls this function
    once per config combination.
    """

    # ── Setup ─────────────────────────────────────────────────────────
    seed = int(cfg.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(OmegaConf.to_yaml(cfg), flush=True)
    print(f"Device: {device}")
    print(f"{'='*60}\n", flush=True)

    # ── Build components ───────────────────────────────────────────────
    encoder   = build_encoder(cfg.encoder)
    n_actions = N_OFFSET_LEVELS ** 2   # flat MultiDiscrete action space
    alpha     = float(cfg.get("alpha", cfg.agent.get("cvar_alpha", 0.25)))
    use_lstm  = bool(cfg.get("variant"))   # variant=recurrent → True, variant=null → False (snapshot)

    agent = build_agent(
        cfg.agent, encoder, n_actions, alpha, device,
        enc_type     = cfg.encoder.type,
        seed         = seed,
        use_lstm     = use_lstm,
        warmup_steps = int(cfg.training.get("warmup_steps", 0)),
    )
    agent = wrap_policy(agent, cfg.get("policy", {}), alpha)
    env   = build_env(cfg.env, cfg.reward, seed)

    if not cfg.env.get("use_abides", True):
        env._abides_env = None
        print("Running with synthetic GBM (use_abides=false)")

    enc_type = cfg.encoder.type

    # ── Output directories ─────────────────────────────────────────────
    agent_type   = cfg.agent.get("type", "agent")
    encoder_type = cfg.encoder.get("type", "encoder")
    reward_type  = cfg.reward.get("reward_type", "reward")
    regime       = cfg.env.get("regime", "base") or "base"
    variant_tag  = "_recurrent" if use_lstm else ""
    alpha_tag    = f"_alpha{alpha:.2f}" if agent_type in ("qrdqn", "iqn") else ""
    # Sampling-scheme ablation axis (uniform vs. prioritized sequence
    # sampling — see training/replay_buffer.py's sample_sequences()).
    # Only meaningful for the three replay-buffer agents; SARSA/PPO don't
    # have a prioritized_replay config key at all.
    sampling_tag = ""
    if agent_type in ("dqn", "qrdqn", "iqn"):
        prioritized  = bool(cfg.agent.get("prioritized_replay", True))
        sampling_tag = "_per" if prioritized else "_uniform"
    run_tag      = (
        f"{agent_type}_{encoder_type}_{reward_type}"
        f"_{regime}{variant_tag}{alpha_tag}{sampling_tag}_seed{seed}"
    )

    project_root = Path(__file__).resolve().parents[1]
    ckpt_dir     = project_root / cfg.training.checkpoint_dir / run_tag
    log_dir      = project_root / cfg.training.log_dir / run_tag

    # Resume safety net: a run started before sampling_tag existed in the
    # run_tag wrote into a directory without it (e.g. the auto-restart
    # supervisor's retry re-invokes this file fresh, so a code update
    # mid-run changes what run_tag it computes). If resuming and the newly
    # tagged path has no checkpoint yet but the untagged legacy path does,
    # keep using the legacy path instead of silently starting over empty.
    if bool(cfg.training.get("resume", False)) and sampling_tag:
        legacy_run_tag = (
            f"{agent_type}_{encoder_type}_{reward_type}"
            f"_{regime}{variant_tag}{alpha_tag}_seed{seed}"
        )
        legacy_ckpt_dir = project_root / cfg.training.checkpoint_dir / legacy_run_tag
        if (find_latest_checkpoint(legacy_ckpt_dir) is not None
                and find_latest_checkpoint(ckpt_dir) is None):
            run_tag  = legacy_run_tag
            ckpt_dir = legacy_ckpt_dir
            log_dir  = project_root / cfg.training.log_dir / legacy_run_tag

    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run tag:    {run_tag}", flush=True)
    print(f"Checkpoint: {ckpt_dir}", flush=True)
    print(f"Logs:       {log_dir}", flush=True)

    # ── Training state ─────────────────────────────────────────────────
    n_episodes    = int(cfg.training.n_episodes)
    eval_every    = int(cfg.training.eval_every)
    eval_episodes = int(cfg.training.eval_episodes)
    ckpt_every    = int(cfg.training.checkpoint_every)
    console_every = int(cfg.logging.console_every)
    train_every_n = int(cfg.training.get("train_every_n_steps", 1))

    early_stopping    = bool(cfg.training.get("early_stopping", False))
    es_patience       = int(cfg.training.get("early_stopping_patience", 6))
    es_min_delta      = float(cfg.training.get("early_stopping_min_delta", 0.05))
    es_min_episodes   = int(cfg.training.get("early_stopping_min_episodes", 150))
    es_smooth_window  = int(cfg.training.get("early_stopping_smooth_window", 3))

    # ── Resume from checkpoint ───────────────────────────────────────────
    # Resuming (rather than always starting fresh) matters because a run
    # spanning many hours/days can be killed by anything — OOM, an
    # unhandled exception, the host machine restarting — and losing all
    # prior episodes each time makes multi-day training impractical.
    resume         = bool(cfg.training.get("resume", False))
    start_episode  = 0
    train_history: list[dict] = []
    eval_history:  list[dict] = []
    last_ckpt_path: Optional[str] = None

    if resume:
        latest = find_latest_checkpoint(ckpt_dir)
        if latest is not None:
            start_episode  = load_checkpoint(agent, latest)
            last_ckpt_path = str(latest)
            print(f"Resumed from {latest} → episode {start_episode}/{n_episodes}", flush=True)

            hist_path = log_dir / "train_history.json"
            if hist_path.exists():
                with open(hist_path) as f:
                    train_history = json.load(f)
            eval_hist_path = log_dir / "eval_history.json"
            if eval_hist_path.exists():
                with open(eval_hist_path) as f:
                    eval_history = json.load(f)
        else:
            print("resume=true but no checkpoint found in "
                  f"{ckpt_dir} — starting from scratch", flush=True)

    elapsed_offset = train_history[-1]["elapsed"] if train_history else 0.0

    # Seeded from resumed eval_history so a resumed run doesn't re-save
    # best.pt over a genuinely better checkpoint recorded before the resume.
    best_eval_sharpe_so_far = max((e["sharpe"] for e in eval_history), default=float("-inf"))

    # Rolling windows for console logging
    recent_sharpe = deque(maxlen=console_every)
    recent_map    = deque(maxlen=console_every)
    recent_loss   = deque(maxlen=console_every)
    recent_pnl    = deque(maxlen=console_every)

    t_start = time.time()

    _write_status(log_dir, run_tag, start_episode, n_episodes,
                  elapsed_offset, train_history[-1] if train_history else {},
                  last_ckpt_path)

    # Tracks the actual last episode reached, as opposed to n_episodes (the
    # configured target) — the two diverge once early stopping can end a
    # run short. Init to start_episode so an already-complete resume (the
    # loop body never executes) still reports the right value below.
    last_episode = start_episode

    # ── Episode loop ───────────────────────────────────────────────────
    for ep in range(start_episode + 1, n_episodes + 1):
        last_episode = ep
        ep_seed = seed + ep
        metrics = run_episode(env, agent, enc_type, training=True, seed=ep_seed,
                               train_every_n_steps=train_every_n)

        metrics["episode"] = ep
        metrics["elapsed"] = elapsed_offset + (time.time() - t_start)
        train_history.append(metrics)

        recent_sharpe.append(metrics["sharpe"])
        recent_map.append(metrics["map"])
        recent_loss.append(metrics["mean_loss"])
        recent_pnl.append(metrics["final_pnl"])

        _write_status(log_dir, run_tag, ep, n_episodes, metrics["elapsed"],
                      metrics, last_ckpt_path)

        # ── Console log ───────────────────────────────────────────────
        if ep % console_every == 0:
            pct = 100.0 * ep / n_episodes
            eta_h = (metrics["elapsed"] / ep) * (n_episodes - ep) / 3600.0
            print(
                f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                f"ep {ep:5d}/{n_episodes} ({pct:5.1f}%) | "
                f"sharpe {np.mean(recent_sharpe):+.3f} | "
                f"map {np.mean(recent_map):.2f} | "
                f"pnl {np.mean(recent_pnl):+.2f} | "
                f"loss {np.mean(recent_loss):.4f} | "
                f"ε {metrics['epsilon']:.3f} | "
                f"steps {agent._steps:,} | "
                f"elapsed {metrics['elapsed']/3600.0:.1f}h | ETA {eta_h:.1f}h",
                flush=True,
            )

        # ── Evaluation rollout ────────────────────────────────────────
        if ep % eval_every == 0:
            eval_metrics_list = [
                run_episode(env, agent, enc_type, training=False, seed=seed + 10000 + ev)
                for ev in range(eval_episodes)
            ]

            eval_summary = {
                "episode":    ep,
                "sharpe":     float(np.mean([m["sharpe"]    for m in eval_metrics_list])),
                "map":        float(np.mean([m["map"]        for m in eval_metrics_list])),
                "mdd":        float(np.mean([m["mdd"]        for m in eval_metrics_list])),
                "final_pnl":  float(np.mean([m["final_pnl"] for m in eval_metrics_list])),
                "sharpe_std": float(np.std( [m["sharpe"]    for m in eval_metrics_list])),
            }
            eval_history.append(eval_summary)

            print(
                f"  EVAL ep {ep:5d} | "
                f"sharpe {eval_summary['sharpe']:+.3f} ± {eval_summary['sharpe_std']:.3f} | "
                f"map {eval_summary['map']:.2f} | "
                f"mdd {eval_summary['mdd']:.2f} | "
                f"pnl {eval_summary['final_pnl']:+.2f}"
            )

            # Track the best-ever eval checkpoint separately from the
            # periodic ep-numbered ones: patience-based early stopping lets
            # training run `patience` checkpoints past a peak before it
            # stops, and the periodic/final checkpoint can end up well below
            # that peak (see the "peak vs. final" drift analysis) — anything
            # that wants the agent's best behavior, not just its last, should
            # load best.pt/best.npz rather than the latest ep-numbered file.
            #
            # Gated on the same es_min_episodes burn-in floor early stopping
            # uses (epsilon hasn't finished annealing before then for the
            # DQN family — see epsilon_decay_steps=50000 ≈ episode 128) so a
            # lucky high-variance eval under a still-mostly-random policy
            # doesn't get immortalized as "best".
            if ep >= es_min_episodes and eval_summary["sharpe"] > best_eval_sharpe_so_far:
                best_eval_sharpe_so_far = eval_summary["sharpe"]
                best_path = save_checkpoint(agent, cfg, ep, eval_summary, ckpt_dir, tag_override="best")
                print(f"  new best eval Sharpe → {best_path}", flush=True)

            if early_stopping and ep >= es_min_episodes:
                # Same burn-in floor as the best-checkpoint gate above: a
                # lucky high-variance eval from before es_min_episodes (agent
                # still mostly-random) must not count as "best" for patience
                # bookkeeping either, or it can absorb the entire patience
                # budget before the gate even opens.
                gated_history = [e for e in eval_history if e["episode"] >= es_min_episodes]
                _, stale_checkpoints = _early_stop_state(gated_history, es_smooth_window, es_min_delta)
                if stale_checkpoints >= es_patience:
                    print(
                        f"\nEarly stopping at episode {ep}: smoothed eval Sharpe "
                        f"hasn't improved by >{es_min_delta} in {stale_checkpoints} "
                        f"consecutive eval checkpoints (patience={es_patience}).",
                        flush=True,
                    )
                    break

        # ── Checkpoint ────────────────────────────────────────────────
        if ep % ckpt_every == 0:
            path = save_checkpoint(agent, cfg, ep, metrics, ckpt_dir)
            last_ckpt_path = str(path)
            print(f"  checkpoint saved → {path}", flush=True)

            # Flush history now, not just at the end — so a crash after
            # this point still leaves the recorded metrics resume can read.
            with open(log_dir / "train_history.json", "w") as f:
                json.dump(train_history, f, indent=2, cls=_NumpyEncoder)
            with open(log_dir / "eval_history.json", "w") as f:
                json.dump(eval_history, f, indent=2, cls=_NumpyEncoder)

    # ── Final save ─────────────────────────────────────────────────────
    # Uses last_episode (actual last episode reached), not n_episodes (the
    # configured target) — they diverge when early stopping ends a run
    # short; using n_episodes here would mislabel e.g. an ep00300 early
    # stop as ep00500.
    final_path = save_checkpoint(agent, cfg, last_episode, train_history[-1], ckpt_dir)
    last_ckpt_path = str(final_path)
    print(f"\nFinal checkpoint → {final_path}", flush=True)

    with open(log_dir / "train_history.json", "w") as f:
        json.dump(train_history, f, indent=2, cls=_NumpyEncoder)
    with open(log_dir / "eval_history.json", "w") as f:
        json.dump(eval_history, f, indent=2, cls=_NumpyEncoder)

    final_state = "early_stopped" if last_episode < n_episodes else "complete"
    _write_status(log_dir, run_tag, last_episode, n_episodes,
                  train_history[-1]["elapsed"], train_history[-1],
                  last_ckpt_path, state=final_state)

    # ── Final summary ──────────────────────────────────────────────────
    best_sharpe = 0.0
    if eval_history:
        best = max(eval_history, key=lambda x: x["sharpe"])
        best_sharpe = best["sharpe"]
        print(f"\nBest eval Sharpe: {best_sharpe:+.4f} at episode {best['episode']}")

    env.close()
    print("Training complete.", flush=True)

    return best_sharpe  # Hydra multirun optimisation target


if __name__ == "__main__":
    train()
