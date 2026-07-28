"""Train MAPPO (centralized-critic, coordinated) for >= 5 seeds.

ARCHITECTURE.md section 6: the Exhibit C.2 design. Same trainer as IPPO
(train_ippo.py), but with a centralized critic over the joint (concatenated)
observation with a per-agent output head, and ``shared_reward_weight > 0``
(``config/train.yaml`` ``mappo.shared_reward_weight``) so each agent's reward
includes ``lambda * R_shared`` (ARCHITECTURE.md section 4.5).

Usage:
    python -m sof_marl.training.train_mappo [--config config/train.yaml]
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
    CentralizedCritic,
    load_ppo_hyperparameters,
    train_multi_agent_ppo,
)

POLICY_NAME = "mappo"
CENTRALIZED_CRITIC = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train.yaml")
    args = parser.parse_args()

    train_cfg: dict[str, Any] = yaml.safe_load(Path(args.config).read_text())
    seeds: list[int] = train_cfg["seeds"]
    shared_reward_weight: float = train_cfg["mappo"]["shared_reward_weight"]
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
            shared_reward_weight=shared_reward_weight,
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
        assert isinstance(critic_model, CentralizedCritic)
        torch.save(
            critic_model.net.state_dict(), models_dir / f"{POLICY_NAME}_seed{seed}_critic.pt"
        )

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
