"""
training/evaluate.py

Checkpoint evaluation script for distrl-market-maker.

Loads a trained agent checkpoint, runs N evaluation episodes,
computes all metrics, and optionally generates Figure 2 (quote skew
overlaid on Week 3 baseline curves).

Called by analysis notebooks after training completes. Also runnable
as a standalone script for quick checkpoint inspection.

Usage
-----
    # Standalone
    python training/evaluate.py \\
        --checkpoint checkpoints/qrdqn_handcrafted_asymmetric_low_vol_seed42/ep01000.pt \\
        --agent qrdqn \\
        --encoder handcrafted \\
        --n_episodes 20 \\
        --regime low_vol \\
        --seed 100

    # From notebook
    from training.evaluate import evaluate_checkpoint, load_agent

    agent, enc_type = load_agent(
        checkpoint="checkpoints/qrdqn_handcrafted_asymmetric_low_vol_seed42/ep01000.pt",
        agent_type="qrdqn",
        encoder_type="handcrafted",
    )
    results = evaluate_checkpoint(agent, env, enc_type, n_episodes=20)
    print(results["summary"])

Output
------
    {
        "metrics":      {sharpe, map, mdd, cvar_10, final_pnl, ...},
        "per_episode":  list of per-episode metric dicts,
        "skew_data":    {inv_levels, mean_offsets, std_offsets},
        "summary":      human-readable string,
    }

Week 6 deliverable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ── Project path ──────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.base import AgentBase
from envs.lob_env import LOBMarketMakingEnv, N_OFFSET_LEVELS, TICK_OFFSETS
from evaluation.metrics import episode_metrics, aggregate_episodes
from evaluation.visualize import quote_skew_curve


# ══════════════════════════════════════════════════════════════════════════════
# Agent + encoder loading
# ══════════════════════════════════════════════════════════════════════════════

def load_agent(
    checkpoint:   str | Path,
    agent_type:   str,
    encoder_type: str,
    device:       str = "cpu",
    # encoder kwargs
    obs_dim:      int = 18,
    n_levels:     int = 10,
    latent_dim:   int = 16,
    ae_checkpoint: Optional[str] = None,
    # agent kwargs
    n_actions:    int = None,
    hidden_dim:   int = 128,
    n_quantiles:  int = 200,
    n_quantile_samples: int = 64,
) -> tuple[Any, str]:
    """
    Load a trained agent from checkpoint.

    Reconstructs the encoder and agent architecture from the provided
    type strings, then loads weights from the checkpoint file.

    Parameters
    ----------
    checkpoint    : str | Path — path to .pt or .npz checkpoint
    agent_type    : str — 'dqn' | 'qrdqn' | 'iqn' | 'ppo' | 'sarsa'
    encoder_type  : str — 'handcrafted' | 'cnn' | 'autoencoder'
    device        : str — 'cpu' or 'cuda'
    obs_dim       : int — handcrafted obs dimension (default 18)
    n_levels      : int — LOB depth levels for CNN/AE (default 10)
    latent_dim    : int — AE/CNN latent dimension (default 16)
    ae_checkpoint : str — AE encoder checkpoint path (for autoencoder type)
    n_actions     : int — flat action space size (default N_OFFSET_LEVELS^2)
    hidden_dim    : int — LSTM hidden size (default 128)
    n_quantiles   : int — QR-DQN quantile atoms (default 200)
    n_quantile_samples : int — IQN quantile samples (default 64)

    Returns
    -------
    (agent, enc_type) : tuple
        agent    — loaded agent ready for inference
        enc_type — encoder type string for get_encoder_input()
    """
    import torch

    checkpoint = Path(checkpoint)
    if n_actions is None:
        n_actions = N_OFFSET_LEVELS ** 2

    # ── Build encoder ─────────────────────────────────────────────────
    if encoder_type == "handcrafted":
        from encoders.handcrafted import HandcraftedEncoder
        encoder = HandcraftedEncoder(obs_dim=obs_dim)

    elif encoder_type == "cnn":
        from encoders.cnn import CNNEncoder
        encoder = CNNEncoder(n_levels=n_levels, latent_dim=latent_dim)

    elif encoder_type == "autoencoder":
        from encoders.autoencoder import AEEncoder
        ae_ckpt = ae_checkpoint or f"checkpoints/ae_encoder_{latent_dim}.pt"
        encoder = AEEncoder.from_checkpoint(ae_ckpt)

    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")

    # ── Build agent ───────────────────────────────────────────────────
    if agent_type == "sarsa":
        from agents.sarsa import SARSAAgent
        agent = SARSAAgent(
            obs_dim   = obs_dim,
            n_actions = n_actions,
        )
        # Load SARSA weights from .npz
        data = np.load(str(checkpoint))
        agent.load_state_dict({
            "W":       [data["W0"], data["W1"], data["W2"]],
            "E":       [data["E0"], data["E1"], data["E2"]],
            "steps":   int(data["steps"]),
            "updates": int(data["updates"]),
        })
        return agent, "handcrafted"

    # Neural agents — load from .pt
    ckpt = torch.load(str(checkpoint), map_location=device)
    agent_state = ckpt.get("agent_state", ckpt)

    if agent_type == "dqn":
        from agents.dqn import DQNAgent
        agent = DQNAgent(
            encoder    = encoder,
            n_actions  = n_actions,
            hidden_dim = hidden_dim,
            device     = device,
        )

    elif agent_type == "qrdqn":
        from agents.qrdqn import QRDQNAgent
        agent = QRDQNAgent(
            encoder     = encoder,
            n_actions   = n_actions,
            n_quantiles = n_quantiles,
            hidden_dim  = hidden_dim,
            device      = device,
        )

    elif agent_type == "iqn":
        from agents.iqn import IQNAgent
        agent = IQNAgent(
            encoder            = encoder,
            n_actions          = n_actions,
            n_quantile_samples = n_quantile_samples,
            hidden_dim         = hidden_dim,
            device             = device,
        )

    elif agent_type == "ppo":
        from agents.ppo import PPOAgent
        agent = PPOAgent(
            encoder   = encoder,
            n_actions = n_actions,
            hidden_dim = hidden_dim,
            device    = device,
        )

    else:
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Choose from: dqn, qrdqn, iqn, ppo, sarsa."
        )

    agent.load_state_dict(agent_state)
    return agent, encoder_type


# ══════════════════════════════════════════════════════════════════════════════
# Single-episode evaluation rollout
# ══════════════════════════════════════════════════════════════════════════════

def _eval_episode(
    agent:    Any,
    env:      LOBMarketMakingEnv,
    enc_type: str,
    seed:     int,
) -> dict:
    """
    Run one greedy evaluation episode and return raw step data + metrics.

    Parameters
    ----------
    agent    : trained agent
    env      : LOBMarketMakingEnv
    enc_type : str — encoder type for input extraction
    seed     : int — episode seed

    Returns
    -------
    dict with keys: step_pnls, inventories, cum_pnls, bid_offsets,
                    ask_offsets, bid_fills, ask_fills, metrics
    """
    from training.train import get_encoder_input, decode_action

    obs, info = env.reset(seed=seed)
    agent.reset_hidden(batch_size=1)

    step_pnls   = []
    inventories = []
    cum_pnls    = []
    bid_offsets = []
    ask_offsets = []
    bid_fills   = []
    ask_fills   = []

    cum_pnl  = 0.0
    prev_mid = info["mid_price"]
    prev_inv = 0

    terminated = truncated = False

    while not (terminated or truncated):
        enc_input   = get_encoder_input(obs, info, enc_type)
        flat_action = agent.act(enc_input, greedy=True)
        action      = decode_action(flat_action)

        obs, reward, terminated, truncated, info = env.step(action)

        inv      = int(info["inventory"])
        mid      = info["mid_price"]
        step_pnl = info.get("spread_pnl", 0.0) + prev_inv * (mid - prev_mid)
        cum_pnl += step_pnl

        step_pnls.append(step_pnl)
        inventories.append(inv)
        cum_pnls.append(cum_pnl)
        bid_offsets.append(int(TICK_OFFSETS[action[0]]))
        ask_offsets.append(int(TICK_OFFSETS[action[1]]))
        bid_fills.append(float(info.get("bid_filled", 0)))
        ask_fills.append(float(info.get("ask_filled", 0)))

        prev_mid = mid
        prev_inv = inv

    m = episode_metrics(
        step_pnls   = np.array(step_pnls),
        inventories = np.array(inventories),
        cum_pnls    = np.array(cum_pnls),
        q_max       = env.Q_max,
        bid_fills   = np.array(bid_fills),
        ask_fills   = np.array(ask_fills),
    )

    return {
        "step_pnls":   np.array(step_pnls),
        "inventories": np.array(inventories),
        "cum_pnls":    np.array(cum_pnls),
        "bid_offsets": np.array(bid_offsets),
        "ask_offsets": np.array(ask_offsets),
        "bid_fills":   np.array(bid_fills),
        "ask_fills":   np.array(ask_fills),
        "metrics":     m,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Multi-episode evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_checkpoint(
    agent:      AgentBase,
    env:        LOBMarketMakingEnv,
    enc_type:   str,
    n_episodes: int = 20,
    seed:       int = 1000,
    q_max:      int = 10,
) -> dict:
    """
    Run N greedy evaluation episodes and aggregate results.

    Parameters
    ----------
    agent      : trained agent (from load_agent())
    env        : LOBMarketMakingEnv
    enc_type   : str — encoder type
    n_episodes : int — number of evaluation episodes (default 20)
    seed       : int — base seed (each episode uses seed + ep_idx)
    q_max      : int — inventory constraint for boundary filter

    Returns
    -------
    dict with keys:
        metrics     : aggregated metrics (mean ± std across episodes)
        per_episode : list of per-episode metric dicts
        skew_data   : dict with inv_levels, mean_offsets, std_offsets
                      (pooled across all episodes)
        raw         : pooled step-level arrays (inventories, bid_offsets, ...)
        summary     : human-readable string
    """
    print(f"Evaluating {n_episodes} episodes...", flush=True)

    per_episode        = []
    all_inventories    = []
    all_bid_offsets    = []
    all_step_pnls      = []

    for ep in range(n_episodes):
        ep_data = _eval_episode(agent, env, enc_type, seed=seed + ep)

        per_episode.append(ep_data["metrics"])
        all_inventories.extend(ep_data["inventories"].tolist())
        all_bid_offsets.extend(ep_data["bid_offsets"].tolist())
        all_step_pnls.extend(ep_data["step_pnls"].tolist())

        if (ep + 1) % 5 == 0:
            print(f"  ep {ep+1}/{n_episodes} | "
                  f"sharpe={ep_data['metrics']['sharpe']:+.3f} | "
                  f"pnl={ep_data['metrics']['final_pnl']:+.2f}",
                  flush=True)

    # Aggregate metrics
    agg = aggregate_episodes(per_episode)

    # Quote skew curve (pooled)
    inv_arr = np.array(all_inventories)
    off_arr = np.array(all_bid_offsets)
    inv_levels, mean_offsets, std_offsets = quote_skew_curve(
        inv_arr, off_arr, q_max=q_max
    )

    skew_data = {
        "inv_levels":   inv_levels,
        "mean_offsets": mean_offsets,
        "std_offsets":  std_offsets,
    }

    # Human-readable summary
    summary_lines = [
        f"Evaluation Results ({n_episodes} episodes)",
        f"{'─'*40}",
        f"Sharpe:    {agg.get('sharpe_mean', np.nan):+.4f} ± {agg.get('sharpe_std', np.nan):.4f}",
        f"MAP:       {agg.get('map_mean', np.nan):.4f} ± {agg.get('map_std', np.nan):.4f}",
        f"MDD:       {agg.get('mdd_mean', np.nan):.4f}",
        f"CVaR_10:   {agg.get('cvar_10_mean', np.nan):.4f}",
        f"Final PnL: {agg.get('final_pnl_mean', np.nan):+.4f} ± {agg.get('final_pnl_std', np.nan):.4f}",
        f"Win Rate:  {agg.get('win_rate_mean', np.nan):.1%}",
        f"Inv@Bounds:{agg.get('inventory_at_bounds_mean', np.nan):.1%}",
    ]
    if "fill_rate_mean" in agg:
        summary_lines.append(f"Fill Rate: {agg['fill_rate_mean']:.1%}")

    summary_lines.append(f"{'─'*40}")
    summary_lines.append(f"Skew curve inventory range: "
                         f"[{inv_levels.min() if len(inv_levels) else 'N/A'}, "
                         f"{inv_levels.max() if len(inv_levels) else 'N/A'}]")

    return {
        "metrics":     agg,
        "per_episode": per_episode,
        "skew_data":   skew_data,
        "raw": {
            "inventories": np.array(all_inventories),
            "bid_offsets": np.array(all_bid_offsets),
            "step_pnls":   np.array(all_step_pnls),
        },
        "summary": "\n".join(summary_lines),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Convenience: evaluate all agents in an experiment directory
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_all(
    ckpt_root:    str | Path,
    env:          LOBMarketMakingEnv,
    agent_types:  list[str] = None,
    encoder_type: str = "handcrafted",
    regime:       str = "low_vol",
    reward:       str = "asymmetric",
    seed:         int = 42,
    n_episodes:   int = 20,
    eval_seed:    int = 1000,
) -> dict[str, dict]:
    """
    Evaluate all agent checkpoints matching the naming convention.

    Parameters
    ----------
    ckpt_root    : str | Path — path to checkpoints/ directory
    env          : LOBMarketMakingEnv
    agent_types  : list of agent names (default all five)
    encoder_type : str — encoder type (default 'handcrafted')
    regime       : str — regime used in training
    reward       : str — reward type
    seed         : int — training seed
    n_episodes   : int — eval episodes per agent
    eval_seed    : int — base seed for eval episodes

    Returns
    -------
    dict mapping agent_name → evaluate_checkpoint() result dict
    """
    if agent_types is None:
        agent_types = ["sarsa", "dqn", "ppo", "qrdqn", "iqn"]

    ckpt_root = Path(ckpt_root)
    results   = {}

    for agent_type in agent_types:
        run_tag  = (f"{agent_type}_{encoder_type}_{reward}"
                    f"_{regime}_seed{seed}")
        ckpt_dir = ckpt_root / run_tag

        if not ckpt_dir.exists():
            print(f"[skip] {run_tag} — checkpoint dir not found")
            continue

        # Find latest checkpoint
        ext      = ".npz" if agent_type == "sarsa" else ".pt"
        ckpts    = sorted(ckpt_dir.glob(f"*{ext}"))
        # Exclude meta files
        ckpts    = [c for c in ckpts if "_meta" not in c.name]

        if not ckpts:
            print(f"[skip] {run_tag} — no checkpoints found")
            continue

        latest = ckpts[-1]
        print(f"\n{'─'*50}")
        print(f"Evaluating {agent_type} from {latest.name}")

        try:
            agent, enc_type = load_agent(
                checkpoint   = latest,
                agent_type   = agent_type,
                encoder_type = encoder_type,
            )
            result = evaluate_checkpoint(
                agent      = agent,
                env        = env,
                enc_type   = enc_type,
                n_episodes = n_episodes,
                seed       = eval_seed,
                q_max      = env.Q_max,
            )
            results[agent_type] = result
            print(result["summary"])

        except Exception as e:
            print(f"  ERROR evaluating {agent_type}: {e}")
            continue

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained distrl-market-maker checkpoint."
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pt or .npz checkpoint file")
    parser.add_argument("--agent",    required=True,
                        choices=["dqn", "qrdqn", "iqn", "ppo", "sarsa"],
                        help="Agent type")
    parser.add_argument("--encoder",  required=True,
                        choices=["handcrafted", "cnn", "autoencoder"],
                        help="Encoder type")
    parser.add_argument("--n_episodes", type=int, default=20,
                        help="Number of evaluation episodes (default 20)")
    parser.add_argument("--regime",   default="low_vol",
                        help="Environment regime (default low_vol)")
    parser.add_argument("--seed",     type=int, default=1000,
                        help="Base seed for eval episodes (default 1000)")
    parser.add_argument("--episode_len", type=int, default=390,
                        help="Episode length (default 390)")
    parser.add_argument("--use_abides", action="store_true", default=False,
                        help="Use ABIDES simulator (default: synthetic GBM)")
    parser.add_argument("--out",      default=None,
                        help="Save results JSON to this path (optional)")
    args = parser.parse_args()

    env = LOBMarketMakingEnv(
        reward_type = "asymmetric",
        episode_len = args.episode_len,
        Q_max       = 10,
        tick_size   = 0.01,
        seed        = args.seed,
        use_abides  = args.use_abides,
    )

    agent, enc_type = load_agent(
        checkpoint   = args.checkpoint,
        agent_type   = args.agent,
        encoder_type = args.encoder,
    )

    results = evaluate_checkpoint(
        agent      = agent,
        env        = env,
        enc_type   = enc_type,
        n_episodes = args.n_episodes,
        seed       = args.seed,
        q_max      = env.Q_max,
    )

    print("\n" + results["summary"])

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Convert numpy arrays to lists for JSON serialisation
        serialisable = {
            "metrics":     results["metrics"],
            "per_episode": results["per_episode"],
            "skew_data": {
                k: v.tolist() for k, v in results["skew_data"].items()
            },
        }
        with open(out_path, "w") as f:
            json.dump(serialisable, f, indent=2)
        print(f"\nResults saved → {out_path}")

    env.close()


if __name__ == "__main__":
    main()