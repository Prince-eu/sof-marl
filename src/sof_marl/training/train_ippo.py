"""Train IPPO (independent per-agent PPO, no coordination) for >= 5 seeds.

ARCHITECTURE.md section 6: four independent per-agent actor-critics,
``shared_reward_weight = 0`` (no shared reward term, no centralized critic) --
isolates the value of coordination against MAPPO (train_mappo.py), which uses the
identical trainer with a centralized critic and shared_reward_weight > 0.

Usage:
    python -m sof_marl.training.train_ippo [--config config/train.yaml]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from sof_marl.agents.spaces import AGENT_NAMES
from sof_marl.env.sme_treasury_env import SMETreasuryEnv
from sof_marl.training.multi_agent_ppo import (
    IndependentCritics,
    load_ppo_hyperparameters,
    train_multi_agent_ppo,
)

POLICY_NAME = "ippo"
CENTRALIZED_CRITIC = False
SHARED_REWARD_WEIGHT = 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train.yaml")
    args = parser.parse_args()

    train_cfg: dict[str, Any] = yaml.safe_load(Path(args.config).read_text())
    seeds: list[int] = train_cfg["seeds"]
    hp = load_ppo_hyperparameters(args.config)

    models_dir = Path("data/processed/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("reports/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_seed_results = []
    for seed in seeds:
        print(f"[{POLICY_NAME}] seed {seed}: training for {hp.total_timesteps} timesteps ...")
        t0 = time.time()
        result = train_multi_agent_ppo(
            env_factory=SMETreasuryEnv,
            centralized_critic=CENTRALIZED_CRITIC,
            shared_reward_weight=SHARED_REWARD_WEIGHT,
            seed=seed,
            hp=hp,
        )
        elapsed = time.time() - t0
        print(
            f"[{POLICY_NAME}] seed {seed}: done in {elapsed:.1f}s, "
            f"{result['total_timesteps']} steps"
        )

        actor_state = {agent: result["actors"][agent].state_dict() for agent in AGENT_NAMES}
        torch.save(actor_state, models_dir / f"{POLICY_NAME}_seed{seed}_actors.pt")
        critic_model = result["critic_model"]
        assert isinstance(critic_model, IndependentCritics)
        critic_state = {agent: c.state_dict() for agent, c in critic_model.critics.items()}
        torch.save(critic_state, models_dir / f"{POLICY_NAME}_seed{seed}_critics.pt")

        all_seed_results.append(
            {
                "seed": seed,
                "total_timesteps": result["total_timesteps"],
                "wall_clock_seconds": result["wall_clock_seconds"],
                "learning_curve": result["learning_curve"],
            }
        )

    out_path = results_dir / f"learning_curve_{POLICY_NAME}.json"
    out_path.write_text(json.dumps(all_seed_results, indent=2))
    print(f"Wrote {out_path} and model weights to {models_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
