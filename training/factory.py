"""
training/factory.py

Construction of encoders, agents, policy wrappers, and the environment
from Hydra config groups. No training-loop logic lives here — see
training/rollout.py for episode execution and training/train.py for the
Hydra entry point.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from agents.base import AgentBase
from encoders.base import BaseEncoder
from envs.lob_env import LOBMarketMakingEnv

_REGIME_CONFIG_DIR = Path(__file__).resolve().parent / "configs" / "env" / "regime"


def build_encoder(enc_cfg: DictConfig) -> BaseEncoder:
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
        from encoders.handcrafted import HandcraftedEncoder
        return HandcraftedEncoder(obs_dim=enc_cfg.get("obs_dim", 18))

    if enc_type == "cnn":
        from encoders.cnn import CNNEncoder
        return CNNEncoder.from_config(OmegaConf.to_container(enc_cfg))

    if enc_type == "autoencoder":
        from encoders.autoencoder import AEEncoder
        return AEEncoder.from_checkpoint(enc_cfg.pretrained_weights)

    raise ValueError(
        f"Unknown encoder type '{enc_type}'. "
        f"Choose from: handcrafted, cnn, autoencoder."
    )


def build_agent(
    agent_cfg: DictConfig,
    encoder:   BaseEncoder,
    n_actions: int,
    alpha:     float,
    device:    str,
    enc_type:    str,
    seed:        int = 42,
    use_lstm:    bool = True,
) -> AgentBase:
    """
    Instantiate agent from config group agent/.

    Parameters
    ----------
    agent_cfg : DictConfig — contents of configs/agent/<name>.yaml
    encoder   : nn.Module  — encoder instance
    n_actions : int        — flat action space size
    alpha     : float      — CVaR tail fraction (overrides agent_cfg.cvar_alpha)
    device    : str        — 'cpu' or 'cuda'
    use_lstm  : bool       — True (variant=recurrent) for temporal memory,
                             False (variant=null) for the snapshot ablation

    Returns
    -------
    Any
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
        use_lstm       = use_lstm,
        device         = device,
    )

    agent_type = agent_cfg.get("type", "qrdqn")

    if agent_type == "sarsa":
        from agents.sarsa import SARSAAgent

        assert enc_type == "handcrafted", (
            "SARSAAgent only supports handcrafted encoder. "
            "Run with encoder=handcrafted."
        )
        return SARSAAgent(
            obs_dim       = int(agent_cfg.get("obs_dim", 18)),
            n_actions     = n_actions,
            n_tilings     = agent_cfg.get("n_tilings",     16),
            n_tiles       = agent_cfg.get("n_tiles",        8),
            memory_size   = agent_cfg.get("memory_size",   2**14),
            alpha         = agent_cfg.get("alpha",         0.001),
            gamma         = agent_cfg.get("gamma",         0.97),
            lambda_trace  = agent_cfg.get("lambda_trace",  0.96),
            lambda_i      = tuple(agent_cfg.get("lambda_i", [0.6, 0.1, 0.3])),
            epsilon       = agent_cfg.get("epsilon",        0.7),
            epsilon_floor = agent_cfg.get("epsilon_floor", 0.0001),
            epsilon_T     = agent_cfg.get("epsilon_T",     1000),
            seed          = seed,
        )

    if agent_type == "dqn":
        from agents.dqn import DQNAgent
        return DQNAgent(**common)

    if agent_type == "qrdqn":
        from agents.qrdqn import QRDQNAgent
        return QRDQNAgent(
            **common,
            n_quantiles = agent_cfg.get("n_quantiles", 200),
            dueling     = agent_cfg.get("dueling", True),
            cvar_alpha  = alpha,
        )

    if agent_type == "iqn":
        from agents.iqn import IQNAgent
        return IQNAgent(
            **common,
            n_quantile_samples = agent_cfg.get("n_quantile_samples", 64),
            embedding_dim      = agent_cfg.get("embedding_dim", 64),
            dueling            = agent_cfg.get("dueling", True),
            cvar_alpha         = alpha,
        )

    if agent_type == "ppo":
        from agents.ppo import PPOAgent
        return PPOAgent(
            encoder        = encoder,
            n_actions      = n_actions,
            hidden_dim     = agent_cfg.get("hidden_dim",     128),
            gamma          = agent_cfg.get("gamma",          0.99),
            gae_lambda     = agent_cfg.get("gae_lambda",     0.95),
            beta_init      = agent_cfg.get("beta_init",      0.5),
            delta_target   = agent_cfg.get("delta_target",   0.01),
            k_epochs       = agent_cfg.get("k_epochs",       4),
            minibatch_size = agent_cfg.get("minibatch_size", 64),
            entropy_coef   = agent_cfg.get("entropy_coef",   0.01),
            value_coef     = agent_cfg.get("value_coef",     0.5),
            lr             = agent_cfg.get("lr",             3e-4),
            max_grad_norm  = agent_cfg.get("max_grad_norm",  0.5),
            use_lstm       = use_lstm,
            device         = device,
        )

    raise ValueError(
        f"Unknown agent type '{agent_type}'. "
        f"Choose from: dqn, qrdqn, iqn, ppo, sarsa."
    )


def wrap_policy(
    agent:      Any,
    policy_cfg: DictConfig,
    alpha:      float,
) -> Any:
    """
    Optionally wrap agent with CVaRPolicy.
    Only applies to distributional agents (QR-DQN, IQN).
    For DQN, SARSA, PPO: returns agent unchanged.
    """
    if not policy_cfg:
        return agent

    measure = policy_cfg.get("measure", "cvar") if hasattr(policy_cfg, "get") else "cvar"

    from agents.qrdqn import QRDQNAgent
    from agents.iqn import IQNAgent

    if not isinstance(agent, (QRDQNAgent, IQNAgent)):
        return agent   # non-distributional agents: no wrapper

    if measure == "mean":
        return agent   # mean is the default for distributional — no wrapper needed

    from agents.cvar_policy import CVaRPolicy

    return CVaRPolicy(agent=agent, alpha=alpha, measure=measure)


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
    regime = env_cfg.get("regime")
    if regime:
        regime_path = _REGIME_CONFIG_DIR / f"{regime}.yaml"
        if not regime_path.exists():
            raise ValueError(f"Unknown env.regime '{regime}' — no config at {regime_path}")
        overlay = OmegaConf.load(regime_path)
        OmegaConf.set_struct(overlay, False)
        OmegaConf.set_struct(env_cfg, False)
        env_cfg = OmegaConf.merge(env_cfg, overlay)

    use_abides = env_cfg.get("use_abides", True)
    env = LOBMarketMakingEnv(
        reward_type  = reward_cfg.reward_type,
        eta          = reward_cfg.get("eta",  0.5),
        lam          = reward_cfg.get("lam",  0.1),
        Q_max        = env_cfg.get("Q_max",        10),
        tick_size    = env_cfg.get("tick_size",     0.01),
        episode_len  = env_cfg.get("episode_len",   390),
        kappa        = env_cfg.get("kappa",         1.0),
        n_lob_levels = env_cfg.get("n_lob_levels",  3),
        seed         = seed,
        use_abides   = use_abides,
    )

    sigma_override = env_cfg.get("sigma_override", None)
    if sigma_override is not None:
        env._sigma_override = float(sigma_override)

    if regime == "flash_crash":
        fc = env_cfg.get("flash_crash", {})
        env._crash_start   = fc.get("crash_start_step", 150)
        env._crash_mag     = fc.get("crash_magnitude",   0.10)
        env._crash_dur     = fc.get("crash_duration",    20)
        env._recovery_frac = fc.get("recovery_frac",     0.50)
        env._recovery_dur  = fc.get("recovery_duration", 80)
        env._post_sigma    = fc.get("post_crash_sigma",  0.015)

    if regime == "trending":
        env._drift = env_cfg.get("drift", 0.0002)

    return env
