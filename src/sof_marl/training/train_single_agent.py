"""Train the single-agent PPO baseline for >= 5 seeds, same budget as IPPO/MAPPO.

EVALUATION.md section 1: "the single-agent baseline must be trained with the same
budget as the MARL policies" -- a weak or under-trained baseline is the most
common way a POC like this loses credibility.

Usage:
    python -m sof_marl.training.train_single_agent [--config config/train.yaml]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from sof_marl.baselines.single_agent import SingleAgentTreasuryEnv

POLICY_NAME = "single_agent_ppo"


class FirmValueLoggerCallback(BaseCallback):
    """Logs terminal firm value at each episode end, for a learning curve
    comparable to IPPO/MAPPO's mean_firm_value (training.multi_agent_ppo),
    independent of reward scale/shaping differences between policies."""

    def __init__(self) -> None:
        super().__init__()
        self.episode_firm_values: list[float] = []
        self.episode_timesteps: list[int] = []

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"], strict=True):
            if done:
                self.episode_firm_values.append(float(info["firm_value"]))
                self.episode_timesteps.append(int(self.num_timesteps))
        return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train.yaml")
    args = parser.parse_args()

    train_cfg: dict[str, Any] = yaml.safe_load(Path(args.config).read_text())
    seeds: list[int] = train_cfg["seeds"]
    ppo_cfg: dict[str, Any] = train_cfg["ppo"]

    models_dir = Path("data/processed/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir = Path("reports/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    all_seed_results = []
    for seed in seeds:
        print(
            f"[{POLICY_NAME}] seed {seed}: training for {ppo_cfg['total_timesteps']} timesteps ..."
        )
        env: Monitor = Monitor(SingleAgentTreasuryEnv())
        env.reset(seed=seed)
        model = PPO(
            "MlpPolicy",
            env,
            seed=seed,
            n_steps=ppo_cfg["n_steps"],
            batch_size=ppo_cfg["batch_size"],
            n_epochs=ppo_cfg["n_epochs"],
            gamma=ppo_cfg["gamma"],
            gae_lambda=ppo_cfg["gae_lambda"],
            learning_rate=ppo_cfg["learning_rate"],
            clip_range=ppo_cfg["clip_range"],
            ent_coef=ppo_cfg["ent_coef"],
            vf_coef=ppo_cfg["vf_coef"],
            max_grad_norm=ppo_cfg["max_grad_norm"],
            verbose=0,
        )
        callback = FirmValueLoggerCallback()
        t0 = time.time()
        model.learn(total_timesteps=ppo_cfg["total_timesteps"], callback=callback)
        elapsed = time.time() - t0
        print(f"[{POLICY_NAME}] seed {seed}: done in {elapsed:.1f}s")
        model.save(str(models_dir / f"{POLICY_NAME}_seed{seed}"))

        episode_rewards = env.get_episode_rewards()
        all_seed_results.append(
            {
                "seed": seed,
                "total_timesteps": ppo_cfg["total_timesteps"],
                "wall_clock_seconds": elapsed,
                "episode_rewards": episode_rewards,
                "episode_firm_values": callback.episode_firm_values,
                "episode_timesteps": callback.episode_timesteps,
            }
        )

    out_path = results_dir / f"learning_curve_{POLICY_NAME}.json"
    out_path.write_text(json.dumps(all_seed_results, indent=2))
    print(f"Wrote {out_path} and model weights to {models_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
