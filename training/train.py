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
from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.lob_env import N_OFFSET_LEVELS
from training.factory import build_agent, build_encoder, build_env, wrap_policy
from training.rollout import (  # noqa: F401 — re-exported for external callers
    _NumpyEncoder,
    decode_action,
    get_encoder_input,
    run_episode,
    save_checkpoint,
)

_CONFIG_PATH = str(Path(__file__).parent / "configs")


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
        enc_type = cfg.encoder.type,
        seed     = seed,
        use_lstm = use_lstm,
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
    run_tag      = (
        f"{agent_type}_{encoder_type}_{reward_type}"
        f"_{regime}{variant_tag}{alpha_tag}_seed{seed}"
    )

    project_root = Path(__file__).resolve().parents[1]
    ckpt_dir     = project_root / cfg.training.checkpoint_dir / run_tag
    log_dir      = project_root / cfg.training.log_dir / run_tag
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

    train_history: list[dict] = []
    eval_history:  list[dict] = []

    # Rolling windows for console logging
    recent_sharpe = deque(maxlen=console_every)
    recent_map    = deque(maxlen=console_every)
    recent_loss   = deque(maxlen=console_every)
    recent_pnl    = deque(maxlen=console_every)

    t_start = time.time()

    # ── Episode loop ───────────────────────────────────────────────────
    for ep in range(1, n_episodes + 1):
        ep_seed = seed + ep
        metrics = run_episode(env, agent, enc_type, training=True, seed=ep_seed)

        metrics["episode"] = ep
        metrics["elapsed"] = time.time() - t_start
        train_history.append(metrics)

        recent_sharpe.append(metrics["sharpe"])
        recent_map.append(metrics["map"])
        recent_loss.append(metrics["mean_loss"])
        recent_pnl.append(metrics["final_pnl"])

        # ── Console log ───────────────────────────────────────────────
        if ep % console_every == 0:
            print(
                f"ep {ep:5d}/{n_episodes} | "
                f"sharpe {np.mean(recent_sharpe):+.3f} | "
                f"map {np.mean(recent_map):.2f} | "
                f"pnl {np.mean(recent_pnl):+.2f} | "
                f"loss {np.mean(recent_loss):.4f} | "
                f"ε {metrics['epsilon']:.3f} | "
                f"steps {agent._steps:,} | "
                f"t {metrics['elapsed']:.0f}s"
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

        # ── Checkpoint ────────────────────────────────────────────────
        if ep % ckpt_every == 0:
            path = save_checkpoint(agent, cfg, ep, metrics, ckpt_dir)
            print(f"  checkpoint saved → {path}")

    # ── Final save ─────────────────────────────────────────────────────
    final_path = save_checkpoint(agent, cfg, n_episodes, train_history[-1], ckpt_dir)
    print(f"\nFinal checkpoint → {final_path}", flush=True)

    with open(log_dir / "train_history.json", "w") as f:
        json.dump(train_history, f, indent=2, cls=_NumpyEncoder)
    with open(log_dir / "eval_history.json", "w") as f:
        json.dump(eval_history, f, indent=2, cls=_NumpyEncoder)

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
