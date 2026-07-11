"""
training/train.py

Main training loop for distrl-market-maker.

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
    variant=recurrent   → RecurrentBase wraps encoder with LSTM
    variant=null        → RecurrentBase with dueling head only (no LSTM seq)

Online rollout loop:
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

Encoder input dispatch:
    HandcraftedEncoder: obs vector (18-dim) from env._get_obs()
    CNNEncoder / AEEncoder: LOB snapshot (20-dim) from info["lob_snapshot"]
    The training loop extracts the right input per encoder type.

Week 6 deliverable.
"""

from __future__ import annotations

import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# ── Project imports ───────────────────────────────────────────────────────────
from envs.lob_env import LOBMarketMakingEnv, N_OFFSET_LEVELS
from encoders.handcrafted import HandcraftedEncoder
from encoders.cnn import CNNEncoder
from encoders.autoencoder import AEEncoder
from agents.dqn import DQNAgent
from agents.qrdqn import QRDQNAgent
from agents.iqn import IQNAgent


# ══════════════════════════════════════════════════════════════════════════════
# Factory helpers
# ══════════════════════════════════════════════════════════════════════════════

def build_encoder(enc_cfg: DictConfig) -> torch.nn.Module:
    """
    Instantiate encoder from config group encoder/.

    Parameters
    ----------
    enc_cfg : DictConfig — contents of configs/encoder/<name>.yaml

    Returns
    -------
    nn.Module with .latent_dim property
    """
    enc_type = enc_cfg.type

    if enc_type == "handcrafted":
        return HandcraftedEncoder(obs_dim=enc_cfg.get("obs_dim", 18))

    if enc_type == "cnn":
        return CNNEncoder.from_config(OmegaConf.to_container(enc_cfg))

    if enc_type == "autoencoder":
        return AEEncoder.from_checkpoint(enc_cfg.pretrained_weights)

    raise ValueError(
        f"Unknown encoder type '{enc_type}'. "
        f"Choose from: handcrafted, cnn, autoencoder."
    )


def build_agent(
    agent_cfg: DictConfig,
    encoder:   torch.nn.Module,
    n_actions: int,
    alpha:     float,
    device:    str,
) -> DQNAgent | QRDQNAgent | IQNAgent:
    """
    Instantiate agent from config group agent/.

    Parameters
    ----------
    agent_cfg : DictConfig — contents of configs/agent/<name>.yaml
    encoder   : nn.Module  — encoder instance
    n_actions : int        — flat action space size
    alpha     : float      — CVaR tail fraction (overrides agent_cfg.cvar_alpha)
    device    : str        — 'cpu' or 'cuda'

    Returns
    -------
    DQNAgent | QRDQNAgent | IQNAgent
    """
    common = dict(
        encoder        = encoder,
        n_actions      = n_actions,
        hidden_dim     = agent_cfg.get("hidden_dim", 128),
        lr             = agent_cfg.get("lr", 1e-4),
        gamma          = agent_cfg.get("gamma", 0.99),
        batch_size     = agent_cfg.get("batch_size", 256),
        seq_len        = agent_cfg.get("lstm_window", 30),
        target_update_freq = agent_cfg.get("target_update_freq", 1000),
        epsilon_start  = agent_cfg.get("epsilon_start", 1.0),
        epsilon_end    = agent_cfg.get("epsilon_end", 0.05),
        epsilon_decay_steps = agent_cfg.get("epsilon_decay_steps", 50_000),
        buffer_capacity = agent_cfg.get("replay_buffer_size", 100_000),
        prioritized    = agent_cfg.get("prioritized_replay", True),
        device         = device,
    )

    agent_type = agent_cfg.get("type", "qrdqn")

    if agent_type == "dqn":
        return DQNAgent(**common)

    if agent_type == "qrdqn":
        return QRDQNAgent(
            **common,
            n_quantiles = agent_cfg.get("n_quantiles", 200),
            dueling     = agent_cfg.get("dueling", True),
            cvar_alpha  = alpha,
        )

    if agent_type == "iqn":
        return IQNAgent(
            **common,
            n_quantile_samples = agent_cfg.get("n_quantile_samples", 64),
            embedding_dim      = agent_cfg.get("embedding_dim", 64),
            dueling            = agent_cfg.get("dueling", True),
            cvar_alpha         = alpha,
        )

    raise ValueError(
        f"Unknown agent type '{agent_type}'. "
        f"Choose from: dqn, qrdqn, iqn."
    )


def build_env(env_cfg: DictConfig, reward_cfg: DictConfig, seed: int) -> LOBMarketMakingEnv:
    """
    Instantiate LOBMarketMakingEnv from merged env + reward config.

    Parameters
    ----------
    env_cfg    : DictConfig — contents of configs/env/base.yaml (+ regime overlay)
    reward_cfg : DictConfig — contents of configs/reward/<name>.yaml
    seed       : int

    Returns
    -------
    LOBMarketMakingEnv
    """
    return LOBMarketMakingEnv(
        reward_type  = reward_cfg.reward_type,
        eta          = reward_cfg.get("eta",  0.5),
        lam          = reward_cfg.get("lam",  0.1),
        Q_max        = env_cfg.get("Q_max",        10),
        tick_size    = env_cfg.get("tick_size",     0.01),
        episode_len  = env_cfg.get("episode_len",   390),
        kappa        = env_cfg.get("kappa",         1.0),
        n_lob_levels = env_cfg.get("n_lob_levels",  3),
        seed         = seed,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Encoder input extraction
# ══════════════════════════════════════════════════════════════════════════════

def get_encoder_input(
    obs:      np.ndarray,
    info:     dict,
    enc_type: str,
) -> np.ndarray:
    """
    Extract the correct input for the encoder from env step output.

    HandcraftedEncoder : obs vector (18-dim) from _get_obs()
    CNNEncoder         : LOB depth snapshot (20-dim) from info["lob_snapshot"]
    AEEncoder          : LOB depth snapshot (20-dim) from info["lob_snapshot"]

    Parameters
    ----------
    obs      : np.ndarray — handcrafted observation from env.step()
    info     : dict       — info dict from env.step()
    enc_type : str        — 'handcrafted', 'cnn', or 'autoencoder'

    Returns
    -------
    np.ndarray — encoder input
    """
    if enc_type == "handcrafted":
        return obs

    # CNN and AE both use raw LOB depth snapshot
    snap = info.get("lob_snapshot", {})
    bid_sizes = snap.get("bid_sizes", [])
    ask_sizes = snap.get("ask_sizes", [])

    if len(bid_sizes) > 0 and len(ask_sizes) > 0:
        snapshot = np.array(ask_sizes + bid_sizes, dtype=np.float32)
    else:
        # LOB not yet populated (early episode) — return zeros
        n_levels = 10
        snapshot = np.zeros(2 * n_levels, dtype=np.float32)

    return snapshot


# ══════════════════════════════════════════════════════════════════════════════
# Episode metrics
# ══════════════════════════════════════════════════════════════════════════════

def compute_episode_metrics(
    step_pnls:   list[float],
    inventories: list[int],
    cum_pnls:    list[float],
) -> dict:
    """
    Compute Sharpe, MAP, MDD and final PnL for one episode.

    Sharpe = mean(step_pnl) / std(step_pnl) * sqrt(T)
    MAP    = mean(|inventory|)
    MDD    = min(cum_pnl - running_max(cum_pnl))
    """
    if len(step_pnls) < 2:
        return {"sharpe": 0.0, "map": 0.0, "mdd": 0.0, "final_pnl": 0.0}

    r    = np.array(step_pnls,   dtype=np.float32)
    inv  = np.array(inventories, dtype=np.float32)
    cpnl = np.array(cum_pnls,    dtype=np.float32)

    mu  = np.mean(r)
    sig = np.std(r) + 1e-10
    sharpe = float(mu / sig * np.sqrt(len(r)))

    map_ = float(np.mean(np.abs(inv)))

    rolling_max = np.maximum.accumulate(cpnl)
    mdd = float(np.min(cpnl - rolling_max))

    return {
        "sharpe":    sharpe,
        "map":       map_,
        "mdd":       mdd,
        "final_pnl": float(cpnl[-1]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Single episode rollout
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(
    env:      LOBMarketMakingEnv,
    agent:    DQNAgent | QRDQNAgent | IQNAgent,
    enc_type: str,
    training: bool = True,
    seed:     Optional[int] = None,
) -> dict:
    """
    Run one episode and return metrics.

    Parameters
    ----------
    env      : LOBMarketMakingEnv
    agent    : any agent with act/observe/train_step interface
    enc_type : str  — encoder type for input extraction
    training : bool — if True, call agent.observe() and agent.train_step()
    seed     : int  — episode seed (None = use env default)

    Returns
    -------
    dict with keys: sharpe, map, mdd, final_pnl, mean_loss, steps, epsilon
    """
    obs, info = env.reset(seed=seed)
    agent.reset_hidden(batch_size=1)

    step_pnls   = []
    inventories = []
    cum_pnls    = []
    losses      = []

    cum_pnl  = 0.0
    prev_mid = info["mid_price"]
    prev_inv = 0

    terminated = truncated = False

    while not (terminated or truncated):
        enc_input = get_encoder_input(obs, info, enc_type)
        action    = agent.act(enc_input, greedy=not training)

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        next_enc = get_encoder_input(next_obs, next_info, enc_type)

        # Step PnL: spread capture + inventory mark-to-market
        inv      = next_info["inventory"]
        mid      = next_info["mid_price"]
        step_pnl = next_info.get("spread_pnl", 0.0) + prev_inv * (mid - prev_mid)
        cum_pnl += step_pnl

        step_pnls.append(step_pnl)
        inventories.append(inv)
        cum_pnls.append(cum_pnl)

        if training:
            agent.observe(enc_input, action, reward, next_enc,
                          terminated or truncated)
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)

        obs      = next_obs
        info     = next_info
        prev_mid = mid
        prev_inv = inv

    metrics = compute_episode_metrics(step_pnls, inventories, cum_pnls)
    metrics["mean_loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["steps"]     = len(step_pnls)
    metrics["epsilon"]   = getattr(agent, "epsilon", 0.0)

    return metrics


# ══════════════════════════════════════════════════════════════════════════════
# Checkpoint helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_checkpoint(
    agent:      DQNAgent | QRDQNAgent | IQNAgent,
    cfg:        DictConfig,
    episode:    int,
    metrics:    dict,
    ckpt_dir:   Path,
) -> Path:
    """Save agent state dict + config + metrics to checkpoint file."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag  = f"ep{episode:05d}"
    path = ckpt_dir / f"{tag}.pt"
    torch.save({
        "agent_state":  agent.state_dict(),
        "episode":      episode,
        "metrics":      metrics,
        "config":       OmegaConf.to_container(cfg, resolve=True),
    }, path)
    return path


def load_checkpoint(
    agent:    DQNAgent | QRDQNAgent | IQNAgent,
    path:     str | Path,
) -> int:
    """Load agent state from checkpoint. Returns episode number."""
    ckpt = torch.load(path, map_location="cpu")
    agent.load_state_dict(ckpt["agent_state"])
    return ckpt["episode"]


# ══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ══════════════════════════════════════════════════════════════════════════════

@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
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
    print(OmegaConf.to_yaml(cfg))
    print(f"Device: {device}")
    print(f"{'='*60}\n")

    # ── Build components ───────────────────────────────────────────────
    encoder   = build_encoder(cfg.encoder)
    n_actions = N_OFFSET_LEVELS ** 2   # flat MultiDiscrete action space

    # Override CVaR alpha from top-level config
    alpha = float(cfg.get("alpha", cfg.agent.get("cvar_alpha", 0.25)))

    agent = build_agent(cfg.agent, encoder, n_actions, alpha, device)
    env   = build_env(cfg.env, cfg.reward, seed)

    enc_type = cfg.encoder.type

    # ── Output directories ─────────────────────────────────────────────
    # Hydra sets cwd to outputs/<date>/<time>/ for each run
    run_dir  = Path(".")
    ckpt_dir = run_dir / cfg.training.checkpoint_dir
    log_dir  = run_dir / cfg.training.log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Training state ─────────────────────────────────────────────────
    n_episodes      = int(cfg.training.n_episodes)
    eval_every      = int(cfg.training.eval_every)
    eval_episodes   = int(cfg.training.eval_episodes)
    ckpt_every      = int(cfg.training.checkpoint_every)
    console_every   = int(cfg.logging.console_every)

    train_history: list[dict] = []
    eval_history:  list[dict] = []

    # Rolling windows for console logging
    recent_sharpe  = deque(maxlen=console_every)
    recent_map     = deque(maxlen=console_every)
    recent_loss    = deque(maxlen=console_every)
    recent_pnl     = deque(maxlen=console_every)

    t_start = time.time()

    # ── Episode loop ───────────────────────────────────────────────────
    for ep in range(1, n_episodes + 1):
        ep_seed = seed + ep
        metrics = run_episode(env, agent, enc_type,
                              training=True, seed=ep_seed)

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
            eval_metrics_list = []
            for ev in range(eval_episodes):
                em = run_episode(env, agent, enc_type,
                                 training=False, seed=seed + 10000 + ev)
                eval_metrics_list.append(em)

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
    final_path = save_checkpoint(agent, cfg, n_episodes,
                                 train_history[-1], ckpt_dir)
    print(f"\nFinal checkpoint → {final_path}")

    # ── Save histories ─────────────────────────────────────────────────
    with open(log_dir / "train_history.json", "w") as f:
        json.dump(train_history, f, indent=2)
    with open(log_dir / "eval_history.json", "w") as f:
        json.dump(eval_history, f, indent=2)

    # ── Final summary ──────────────────────────────────────────────────
    if eval_history:
        best = max(eval_history, key=lambda x: x["sharpe"])
        print(f"\nBest eval Sharpe: {best['sharpe']:+.4f} at episode {best['episode']}")

    env.close()
    print("Training complete.")

    # Return best eval Sharpe for Hydra multirun optimisation
    return best["sharpe"] if eval_history else 0.0


if __name__ == "__main__":
    train()