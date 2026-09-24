"""
scripts/run_multiagent_tournament.py

Multi-agent tournament: N=2 independent QR-DQN
agents trained simultaneously in one shared ABIDES simulation via
envs.multi_agent_env.MultiAgentMarketEnv — normal regime, no shared
replay buffer or coordination mechanism (the standard independent-
learners MARL setup). Each agent runs its own act()/observe()/
train_step() against its own transition each step.

Research questions (per the project's Week 6/8 plan):
    - Does competition tighten spreads relative to the single-agent case?
    - Does it destabilise the market (higher price volatility)?
    - Do the two agents converge to a stable quoting equilibrium or keep
      undercutting each other episode over episode?

Usage:
    # Smoke test (synthetic fallback, no ABIDES, few episodes):
    python scripts/run_multiagent_tournament.py --n_episodes 3 --use_abides false

    # Real run:
    python scripts/run_multiagent_tournament.py --n_episodes 500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from envs.multi_agent_env import MultiAgentMarketEnv
from envs.lob_env import N_OFFSET_LEVELS
from encoders.handcrafted import HandcraftedEncoder
from agents.qrdqn import QRDQNAgent
from training.rollout import decode_action
from evaluation.metrics import episode_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_AGENTS     = 2
REGIME       = "normal"
BASE_SEED    = 42
N_ACTIONS    = N_OFFSET_LEVELS ** 2
TRAIN_EVERY_N_STEPS = 4

RUN_TAG    = f"multiagent_qrdqn_{REGIME}_n{N_AGENTS}_seed{BASE_SEED}"
LOG_DIR    = PROJECT_ROOT / "logs" / RUN_TAG
CKPT_DIR   = PROJECT_ROOT / "checkpoints" / RUN_TAG


def build_agent(seed_offset: int) -> QRDQNAgent:
    torch.manual_seed(BASE_SEED + seed_offset)
    encoder = HandcraftedEncoder(obs_dim=18)
    return QRDQNAgent(
        encoder=encoder, n_actions=N_ACTIONS,
        n_quantiles=32, hidden_dim=256, cvar_alpha=0.25, kappa=100.0,
        use_lstm=False, device="cpu",
    )


def run_episode(env: MultiAgentMarketEnv, agents: list[QRDQNAgent], seed: int, training: bool):
    obs_n = env.reset(seed=seed)
    prev_mids = [env._mid_price] * N_AGENTS
    prev_invs = [0] * N_AGENTS

    step_pnls   = [[] for _ in range(N_AGENTS)]
    inventories = [[] for _ in range(N_AGENTS)]
    cum_pnls    = [[] for _ in range(N_AGENTS)]
    bid_fills   = [[] for _ in range(N_AGENTS)]
    ask_fills   = [[] for _ in range(N_AGENTS)]
    cum_pnl     = [0.0] * N_AGENTS
    spreads     = []

    losses = [[] for _ in range(N_AGENTS)]
    step_i = 0
    done = False
    while not done:
        flat_actions = [agents[i].act(obs_n[i], greedy=not training) for i in range(N_AGENTS)]
        actions_n = [decode_action(a) for a in flat_actions]
        next_obs_n, rewards_n, dones_n, infos_n = env.step(actions_n)
        done = dones_n[0]
        step_i += 1

        mkt = env.market_metrics(infos_n)
        if mkt:
            spreads.append(mkt["market_spread"])

        for i in range(N_AGENTS):
            info = infos_n[i]
            mid = info["mid_price"]
            step_pnl = info.get("spread_pnl", 0.0) + prev_invs[i] * (mid - prev_mids[i])
            cum_pnl[i] += step_pnl
            step_pnls[i].append(step_pnl)
            inventories[i].append(info["inventory"])
            cum_pnls[i].append(cum_pnl[i])
            bid_fills[i].append(info["bid_filled"])
            ask_fills[i].append(info["ask_filled"])
            prev_mids[i] = mid
            prev_invs[i] = info["inventory"]

            if training:
                agents[i].observe(obs_n[i], flat_actions[i], rewards_n[i], next_obs_n[i], dones_n[i])
                if step_i % TRAIN_EVERY_N_STEPS == 0:
                    loss = agents[i].train_step()
                    if loss is not None:
                        losses[i].append(loss)

        obs_n = next_obs_n

    per_agent_metrics = []
    for i in range(N_AGENTS):
        m = episode_metrics(
            np.array(step_pnls[i]), np.array(inventories[i]), np.array(cum_pnls[i]),
            q_max=env.Q_max, bid_fills=np.array(bid_fills[i]), ask_fills=np.array(ask_fills[i]),
        )
        m["mean_loss"] = float(np.mean(losses[i])) if losses[i] else 0.0
        per_agent_metrics.append(m)

    market = {"mean_spread": float(np.mean(spreads)) if spreads else 0.0,
              "spread_std": float(np.std(spreads)) if spreads else 0.0}
    return per_agent_metrics, market


def find_latest_common_checkpoint(ckpt_dir: Path, n_agents: int) -> int:
    """Highest episode number where EVERY agent has a saved checkpoint —
    a mid-write restart could leave one agent's file ahead of another's."""
    if not ckpt_dir.exists():
        return 0
    eps_per_agent = []
    for i in range(n_agents):
        eps = {int(p.stem.split("_ep")[1]) for p in ckpt_dir.glob(f"agent{i}_ep*.pt")}
        if not eps:
            return 0
        eps_per_agent.append(eps)
    common = set.intersection(*eps_per_agent)
    return max(common) if common else 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n_episodes", type=int, default=500)
    p.add_argument("--use_abides", type=str, default="true")
    p.add_argument("--checkpoint_every", type=int, default=25)
    p.add_argument("--resume", type=str, default="true",
                    help="Resume from the latest checkpoint both agents share, "
                         "if one exists (default true) — this script has no "
                         "supervisor auto-restart like scripts/run_job.sh's "
                         "agents do, so an environment restart mid-run used to "
                         "silently discard all progress; added after exactly "
                         "that happened once this session.")
    args = p.parse_args()
    use_abides = args.use_abides.lower() == "true"

    # NOTE: MultiAgentMarketEnv has no `regime` parameter — unlike the
    # single-agent LOBMarketMakingEnv (built via Hydra env/regime/*.yaml
    # overlays), it always uses _MultiAgentAbidesEnv's hardcoded
    # background_config="rmsc04" with no per-regime fund_vol override. So
    # "normal regime" here is really just "whatever rmsc04 defaults to" —
    # not calibrated to match the single-agent normal-regime baseline. This
    # is a real gap in the existing multi-agent env, not something fixed
    # here; flagged in the README/dashboard writeup.
    env = MultiAgentMarketEnv(n_agents=N_AGENTS, episode_len=390, order_size=100,
                               reward_type="asymmetric", eta=0.5,
                               use_abides=use_abides, seed=BASE_SEED)

    agents = [build_agent(seed_offset=i) for i in range(N_AGENTS)]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    train_history = [[] for _ in range(N_AGENTS)]
    market_history = []
    start_episode = 0

    if args.resume.lower() == "true":
        last_ep = find_latest_common_checkpoint(CKPT_DIR, N_AGENTS)
        if last_ep > 0:
            for i in range(N_AGENTS):
                ckpt = torch.load(CKPT_DIR / f"agent{i}_ep{last_ep:05d}.pt",
                                   map_location="cpu", weights_only=False)
                agents[i].load_state_dict(ckpt["agent_state"])
            hist_path = LOG_DIR / "train_history.json"
            if hist_path.exists():
                with open(hist_path) as f:
                    data = json.load(f)
                # Trim any rows past last_ep in case a checkpoint save and a
                # history-file save raced against the restart.
                train_history = [[row for row in agent_hist if row["episode"] <= last_ep]
                                  for agent_hist in data["agents"]]
                market_history = [row for row in data["market"] if row["episode"] <= last_ep]
            start_episode = last_ep
            print(f"Resumed from episode {last_ep} (both agents' state + buffers restored)")

    print(f"Run tag: {RUN_TAG}  |  use_abides={use_abides}  |  n_episodes={args.n_episodes}")
    for ep in range(start_episode + 1, args.n_episodes + 1):
        t0 = time.time()
        per_agent_metrics, market = run_episode(env, agents, seed=BASE_SEED + ep, training=True)
        elapsed = time.time() - t0

        for i in range(N_AGENTS):
            row = dict(per_agent_metrics[i])
            row["episode"] = ep
            row["elapsed"] = elapsed
            train_history[i].append(row)
        market_row = dict(market)
        market_row["episode"] = ep
        market_history.append(market_row)

        if ep % 10 == 0 or ep == 1:
            sh = [f"{per_agent_metrics[i]['sharpe']:+.3f}" for i in range(N_AGENTS)]
            print(f"[{time.strftime('%H:%M:%S')}] ep {ep:4d}/{args.n_episodes} "
                  f"sharpe(agents)={sh} mean_spread={market['mean_spread']:.4f} "
                  f"eps={[round(a.epsilon,3) for a in agents]} ({elapsed:.0f}s)", flush=True)

        if ep % args.checkpoint_every == 0 or ep == args.n_episodes:
            for i in range(N_AGENTS):
                torch.save({"agent_state": agents[i].state_dict()},
                           CKPT_DIR / f"agent{i}_ep{ep:05d}.pt")
            with open(LOG_DIR / "train_history.json", "w") as f:
                json.dump({"agents": train_history, "market": market_history}, f)

    env.close()
    with open(LOG_DIR / "train_history.json", "w") as f:
        json.dump({"agents": train_history, "market": market_history}, f)
    print(f"Training complete. Saved history and checkpoints under {LOG_DIR} / {CKPT_DIR}")


if __name__ == "__main__":
    main()
