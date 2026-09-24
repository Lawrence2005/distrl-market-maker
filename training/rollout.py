"""
training/rollout.py

Single-episode execution, episode metrics, and checkpoint I/O.
See training/factory.py for component construction and training/train.py
for the Hydra entry point that drives the episode loop.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from envs.lob_env import LOBMarketMakingEnv, N_OFFSET_LEVELS


def decode_action(flat_action: int) -> np.ndarray:
    """
    Decode flat action index → [bid_idx, ask_idx] for MultiDiscrete env.

    flat_action = bid_idx * N_OFFSET_LEVELS + ask_idx
    """
    bid_idx, ask_idx = divmod(flat_action, N_OFFSET_LEVELS)
    return np.array([bid_idx, ask_idx], dtype=np.int64)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def get_encoder_input(
    obs:      np.ndarray,
    info:     dict,
    enc_type: str,
) -> np.ndarray:
    """
    Extract the correct input for the encoder from env step output.

    HandcraftedEncoder : obs vector (18-dim) from _get_obs()
    CNNEncoder         : LOB depth snapshot (20-dim) from info["lob_snapshot"]
    AEEncoder          : LOB depth snapshot (20-dim) from info["lob_snapshot"],
                          per-side normalized (see below)

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
    n_levels = 10
    snap = info.get("lob_snapshot", {})
    bid_sizes = list(snap.get("bid_sizes", []))
    ask_sizes = list(snap.get("ask_sizes", []))

    # Real order books are frequently thinner than n_levels on one or both
    # sides (or fully empty early in an episode) — pad each side to a fixed
    # n_levels independently so the encoder always receives a constant-shape
    # 2*n_levels vector. Truncate defensively in case a side ever exceeds
    # n_levels (shouldn't happen, since _parse_abides_step already slices to
    # K=n_lob_levels, but keeps this robust to a config mismatch).
    bid_sizes = (bid_sizes + [0.0] * n_levels)[:n_levels]
    ask_sizes = (ask_sizes + [0.0] * n_levels)[:n_levels]

    if enc_type == "autoencoder":
        # Rescale each side to sum to 1.0 — mirrors
        # encoders/autoencoder.py::AEEncoder._normalize() exactly (same
        # 1e-8 floor), so AE-pretraining data collected via this function
        # (scripts/collect_abides_snapshots.py) already matches the
        # distribution the frozen encoder sees at RL-rollout time. Safe
        # division means a fully-empty side — which occurs occasionally
        # under a random exploration policy — yields zeros instead of
        # NaN/inf, so callers no longer need to drop these snapshots as
        # un-normalizable.
        ask_sum = max(sum(ask_sizes), 1e-8)
        bid_sum = max(sum(bid_sizes), 1e-8)
        ask_sizes = [v / ask_sum for v in ask_sizes]
        bid_sizes = [v / bid_sum for v in bid_sizes]

    snapshot = np.array(ask_sizes + bid_sizes, dtype=np.float32)

    return snapshot


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


def run_episode(
    env:                 LOBMarketMakingEnv,
    agent:               Any,
    enc_type:            str,
    training:            bool = True,
    seed:                Optional[int] = None,
    train_every_n_steps: int = 1,
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
    train_every_n_steps : int — for the DQN/QR-DQN/IQN branch (SARSA and PPO
        are unaffected), only call agent.train_step() every Nth env step.
        agent.observe() (and therefore replay-buffer storage, _steps count,
        and epsilon decay) still runs every step regardless — this matches
        Nature DQN's own update cadence (Mnih et al. 2015 store a transition
        every frame but only run an SGD step every 4th), it isn't a
        fidelity cut invented for this codebase.

    Returns
    -------
    dict with keys: sharpe, map, mdd, final_pnl, mean_loss, steps, epsilon
    """
    obs, info = env.reset(seed=seed)
    agent.reset_hidden(batch_size=1)

    from agents.ppo import PPOAgent
    is_ppo   = isinstance(agent, PPOAgent)
    is_sarsa = getattr(agent, 'is_online', False)

    step_pnls   = []
    inventories = []
    cum_pnls    = []
    losses      = []

    cum_pnl  = 0.0
    prev_mid = info["mid_price"]
    prev_inv = 0
    step_idx = 0

    # SARSA(λ)'s on-policy action, carried from one iteration's train_step
    # bootstrap into the next iteration's actually-executed action — see
    # SARSAAgent.train_step()'s next_action docstring for why this must be
    # the SAME sample, not independently re-drawn each place it's needed.
    pending_sarsa_action = None

    terminated = truncated = False

    while not (terminated or truncated):
        step_idx += 1
        enc_input = get_encoder_input(obs, info, enc_type)
        if is_ppo:
            action, value, log_prob = agent.act_with_value(enc_input)
        elif is_sarsa and training and pending_sarsa_action is not None:
            action = pending_sarsa_action
        else:
            action = agent.act(enc_input, greedy=not training)

        next_obs, reward, terminated, truncated, next_info = env.step(decode_action(action))
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
            if is_sarsa:
                done = terminated or truncated
                # Sample a' ONCE here — reused both for this update's
                # bootstrap and as next iteration's actually-executed
                # action (see docstrings on SARSAAgent.train_step and
                # pending_sarsa_action above).
                next_action = None if done else agent.act(next_enc, greedy=not training)
                loss = agent.train_step(
                    obs=enc_input, action=action, reward=reward,
                    next_obs=next_enc, next_action=next_action, done=done,
                )
                if loss is not None:
                    losses.append(loss)
                pending_sarsa_action = next_action
            elif is_ppo:
                # PPO: store with value and log_prob, no per-step update
                agent.observe(enc_input, action, reward, next_enc,
                            terminated or truncated,
                            value=value, log_prob=log_prob)
            else:
                # DQN/QR-DQN/IQN: buffer every step, gradient update every
                # train_every_n_steps-th step (see docstring above).
                agent.observe(enc_input, action, reward, next_enc,
                            terminated or truncated)
                if step_idx % train_every_n_steps == 0:
                    loss = agent.train_step()
                    if loss is not None:
                        losses.append(loss)

        obs      = next_obs
        info     = next_info
        prev_mid = mid
        prev_inv = inv

    if training and is_ppo:
        loss = agent.train_step()   # PPO updates once per episode
        if loss is not None:
            losses.append(loss)

    metrics = compute_episode_metrics(step_pnls, inventories, cum_pnls)
    metrics["mean_loss"] = float(np.mean(losses)) if losses else 0.0
    metrics["steps"]     = len(step_pnls)
    metrics["epsilon"]   = getattr(agent, "epsilon", 0.0)

    return metrics


def save_checkpoint(
    agent, cfg: DictConfig, episode: int, metrics: dict, ckpt_dir: Path,
    tag_override: str | None = None,
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = tag_override or f"ep{episode:05d}"

    try:
        config_dict = OmegaConf.to_container(cfg, resolve=True)
    except Exception as e:
        print(f"Warning: could not resolve config for checkpoint metadata: {e}")
        config_dict = {}

    from agents.sarsa import SARSAAgent
    if isinstance(agent, SARSAAgent):
        # state_dict() called once, saved directly to disk — no tolist()
        sd   = agent.state_dict()
        path = ckpt_dir / f"{tag}.npz"
        np.savez_compressed(
            str(path),
            W0=sd["W"][0], W1=sd["W"][1], W2=sd["W"][2],
            E0=sd["E"][0], E1=sd["E"][1], E2=sd["E"][2],
            steps=np.array(sd["steps"]),
            updates=np.array(sd["updates"]),
        )
        meta_path = ckpt_dir / f"{tag}_meta.json"
        with open(meta_path, "w") as f:
            json.dump({
                "episode": episode,
                "metrics": metrics,
                "config":  config_dict,
            }, f, indent=2, cls=_NumpyEncoder)
    else:
        sd = agent.state_dict()
        path = ckpt_dir / f"{tag}.pt"
        torch.save({
            "agent_state": sd,
            "episode":     episode,
            "metrics":     metrics,
            "config":      config_dict,
        }, path)

    return path


def find_latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """
    Return the checkpoint for the highest episode number in ckpt_dir, or
    None if it doesn't exist or has no checkpoints yet.

    Used by train.py's resume path: sort by the numeric episode tag rather
    than filename/mtime, since those aren't guaranteed stable across a
    crash + restart.
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None

    candidates = list(ckpt_dir.glob("ep*.pt")) + list(ckpt_dir.glob("ep*.npz"))
    if not candidates:
        return None

    return max(candidates, key=lambda p: int(p.stem.replace("ep", "")))


def find_best_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """
    Return the `best.pt`/`best.npz` checkpoint saved whenever eval Sharpe hit
    a new high during training (see train.py), or None if it doesn't exist —
    e.g. a run from before this tracking was added, or one that never
    completed a single eval checkpoint. Falls back to find_latest_checkpoint
    at the call site, not here, so callers can log which path was used.
    """
    ckpt_dir = Path(ckpt_dir)
    for ext in ("pt", "npz"):
        path = ckpt_dir / f"best.{ext}"
        if path.exists():
            return path
    return None


def load_checkpoint(agent, path):
    from agents.sarsa import SARSAAgent
    if isinstance(agent, SARSAAgent):
        data     = np.load(str(path))
        meta_path = Path(str(path).replace(".npz", "_meta.json"))
        with open(meta_path) as f:
            meta = json.load(f)
        agent.load_state_dict({
            "W":       [data["W0"], data["W1"], data["W2"]],
            "E":       [data["E0"], data["E1"], data["E2"]],
            "steps":   int(data["steps"]),
            "updates": int(data["updates"]),
        })
        return meta["episode"]
    else:
        # weights_only=False: these are our own checkpoint files, never
        # external/untrusted ones, and the replay-buffer state embedded in
        # agent_state (numpy arrays + Transition namedtuples, since this
        # session's checkpointing fix) isn't in torch's weights_only=True
        # (the default since torch 2.6) allowed-globals list — the default
        # raises UnpicklingError on load for exactly the checkpoints this
        # fix produces. Caught via an actual on-disk save/load smoke test,
        # not by the test suite, which only round-trips state_dict() in
        # memory and never exercises torch.load from a real file.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        agent.load_state_dict(ckpt["agent_state"])
        return ckpt["episode"]
