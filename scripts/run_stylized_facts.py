"""
scripts/run_stylized_facts.py

Round 4 workstream E: stylized-facts audit of the live ABIDES simulator. Uses a random policy (envs.stylized_facts.run_stylized_facts_audit's own design — this
validates the SIMULATOR's realism) across all three regimes.

Usage:
    python scripts/run_stylized_facts.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from omegaconf import OmegaConf

from training.factory import build_env
from envs.stylized_facts import run_stylized_facts_audit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR   = PROJECT_ROOT / "training" / "configs"
OUT_DIR      = PROJECT_ROOT / "evaluation" / "results_scaled_down_sweep"


def build_regime_env(regime: str, seed: int):
    env_cfg = OmegaConf.load(CONFIG_DIR / "env" / "base.yaml")
    OmegaConf.set_struct(env_cfg, False)
    env_cfg.regime     = regime
    env_cfg.use_abides = True
    reward_cfg = OmegaConf.load(CONFIG_DIR / "reward" / "asymmetric.yaml")
    return build_env(env_cfg, reward_cfg, seed)


def main() -> None:
    all_results = {}
    for regime in ["low_vol", "normal", "high_vol"]:
        print(f"\n{'='*78}\nSTYLIZED FACTS AUDIT — {regime}\n{'='*78}")
        env = build_regime_env(regime, seed=900)
        overall_pass, results = run_stylized_facts_audit(env, n_episodes=10, verbose=True)
        env.close()
        all_results[regime] = {
            "overall_pass": overall_pass,
            "facts": {
                name: {"passed": bool(r["passed"])}
                for name, r in results.items()
            },
        }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "stylized_facts.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
