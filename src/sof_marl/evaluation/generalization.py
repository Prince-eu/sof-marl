"""Cross-sector generalization experiment (report section 5.6).

Moves from one representative SME to four stylized sector profiles
(config/sectors/{general,retail,manufacturing,services}.yaml), and asks whether the
coordinated policy generalizes across them. Two arms:

- **Zero-shot** (`--phase zeroshot`): evaluate the *baseline-trained* policies
  (trained on the general profile only, section 5.3) on each sector without
  retraining -- the "before", measuring off-distribution transfer.
- **Domain randomization** (`--phase train` then `--phase eval`): retrain all three
  learned policies with the sector profile sampled uniformly at each episode
  (`SMETreasuryEnv(sector_config_paths=...)`), then evaluate the domain-randomized
  policies per sector -- the "after".

Because initial firm value V_0 differs across sectors (e.g. manufacturing carries
more term debt), the combined objective J is only compared *within* a sector
(zero-shot vs domain-randomized for the same sector), never across sectors.

Usage:
    python -m sof_marl.evaluation.generalization --phase zeroshot
    python -m sof_marl.evaluation.generalization --phase train
    python -m sof_marl.evaluation.generalization --phase eval
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from sof_marl.agents.spaces import AGENT_NAMES
from sof_marl.baselines.single_agent import SingleAgentTreasuryEnv
from sof_marl.env.sme_treasury_env import SMETreasuryEnv
from sof_marl.evaluation.metrics import combined_objective, episode_solvency_breach_rate
from sof_marl.evaluation.rollout import (
    EpisodeRecord,
    evaluate_marl_policy,
    evaluate_rule_based,
    evaluate_single_agent,
)
from sof_marl.training.multi_agent_ppo import (
    CentralizedCritic,
    IndependentCritics,
    load_ppo_hyperparameters,
    train_multi_agent_ppo,
)

SECTORS = ["general", "retail", "manufacturing", "services"]
SECTOR_PATHS = [f"config/sectors/{s}.yaml" for s in SECTORS]
MODELS_DIR = Path("data/processed/models")
RESULTS_DIR = Path("reports/results")


def _flatten(records: list[EpisodeRecord] | dict[int, list[EpisodeRecord]]) -> list[EpisodeRecord]:
    if isinstance(records, dict):
        return [r for seed_recs in records.values() for r in seed_recs]
    return records


def _summary(records: list[EpisodeRecord], kappa: float, lam: float) -> dict[str, float]:
    n = len(records)
    return {
        "combined_objective": sum(combined_objective(r, kappa, lam) for r in records) / n,
        "solvency_breach_rate": sum(episode_solvency_breach_rate(r) for r in records) / n,
        "terminal_firm_value": sum(r.firm_values[-1] for r in records) / n,
        "n_episodes": n,
    }


def evaluate_all_on_sector(
    sector_path: str,
    tag: str,
    seeds: list[int],
    n_episodes: int,
    seed_base: int,
    kappa: float,
    lam: float,
) -> dict[str, dict[str, float]]:
    """Evaluate rule-based + the three learned policies (given model tag) on one fixed sector."""
    suffix = f"_{tag}" if tag else ""
    rb = evaluate_rule_based(n_episodes, seed_base, sector_path)
    sa = evaluate_single_agent(
        MODELS_DIR, seeds, n_episodes, seed_base, sector_path, f"single_agent_ppo{suffix}"
    )
    ippo = evaluate_marl_policy(
        MODELS_DIR, f"ippo{suffix}", seeds, n_episodes, seed_base, sector_path
    )
    mappo = evaluate_marl_policy(
        MODELS_DIR, f"mappo{suffix}", seeds, n_episodes, seed_base, sector_path
    )
    return {
        "rule_based": _summary(_flatten(rb), kappa, lam),
        "single_agent_ppo": _summary(_flatten(sa), kappa, lam),
        "ippo": _summary(_flatten(ippo), kappa, lam),
        "mappo": _summary(_flatten(mappo), kappa, lam),
    }


def train_domain_randomized(seeds: list[int], train_cfg: dict[str, Any]) -> None:
    """Retrain single-agent, IPPO, and MAPPO with the sector profile sampled each episode."""
    hp = load_ppo_hyperparameters("config/train.yaml")
    ppo_cfg = train_cfg["ppo"]
    shared_reward_weight = train_cfg["mappo"]["shared_reward_weight"]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for seed in seeds:
        print(f"[single_agent_ppo_dr] seed {seed} ...", flush=True)
        env: Monitor = Monitor(SingleAgentTreasuryEnv(sector_config_paths=SECTOR_PATHS))
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
        model.learn(total_timesteps=ppo_cfg["total_timesteps"])
        model.save(str(MODELS_DIR / f"single_agent_ppo_dr_seed{seed}"))

    def dr_factory() -> SMETreasuryEnv:
        return SMETreasuryEnv(sector_config_paths=SECTOR_PATHS)

    for name, centralized, srw in [("ippo", False, 0.0), ("mappo", True, shared_reward_weight)]:
        for seed in seeds:
            print(f"[{name}_dr] seed {seed} ...", flush=True)
            t0 = time.time()
            result = train_multi_agent_ppo(dr_factory, centralized, srw, seed, hp)
            actor_state = {a: result["actors"][a].state_dict() for a in AGENT_NAMES}
            torch.save(actor_state, MODELS_DIR / f"{name}_dr_seed{seed}_actors.pt")
            cm = result["critic_model"]
            if isinstance(cm, CentralizedCritic):
                torch.save(cm.net.state_dict(), MODELS_DIR / f"{name}_dr_seed{seed}_critic.pt")
            elif isinstance(cm, IndependentCritics):
                torch.save(
                    {a: c.state_dict() for a, c in cm.critics.items()},
                    MODELS_DIR / f"{name}_dr_seed{seed}_critics.pt",
                )
            print(f"  done in {time.time() - t0:.0f}s", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--phase", choices=["zeroshot", "train", "eval"], required=True)
    args = parser.parse_args()

    train_cfg: dict[str, Any] = yaml.safe_load(Path(args.config).read_text())
    seeds: list[int] = train_cfg["seeds"]
    n_episodes: int = train_cfg["eval"]["n_episodes"]
    seed_base: int = train_cfg["eval"]["seed_base"]
    kappa = train_cfg["objective"]["kappa_volatility"]
    lam = train_cfg["objective"]["lambda_solvency"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "generalization_metrics.json"
    existing = json.loads(out_path.read_text()) if out_path.exists() else {}

    if args.phase == "train":
        train_domain_randomized(seeds, train_cfg)
        print("Domain-randomized training complete.")
        return 0

    tag = "" if args.phase == "zeroshot" else "dr"
    key = "zero_shot" if args.phase == "zeroshot" else "domain_randomized"
    per_sector: dict[str, Any] = {}
    for sector, path in zip(SECTORS, SECTOR_PATHS, strict=True):
        print(f"[{args.phase}] evaluating on sector '{sector}' ...", flush=True)
        per_sector[sector] = evaluate_all_on_sector(
            path, tag, seeds, n_episodes, seed_base, kappa, lam
        )
    existing[key] = per_sector
    out_path.write_text(json.dumps(existing, indent=2))
    print(f"Wrote {out_path} [{key}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
