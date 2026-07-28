"""Sensitivity analysis (EVALUATION.md section 3): vary the most consequential
assumed calibration constants one at a time and check whether the qualitative
coordination-lift conclusion (MAPPO's combined objective exceeds every other
policy) is stable.

Given the wall-clock cost of retraining (docs/technical_report.md section 4.4),
this re-evaluates the already-trained seed-0 policies -- not a full 5-seed
retrain -- against each perturbed environment, at a reduced episode count. It is
a robustness check on the trained policies' behavior under different calibration
assumptions, not a re-optimization against them; this is disclosed in the report.

Usage:
    python -m sof_marl.evaluation.sensitivity [--env-config config/env.yaml]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml
from stable_baselines3 import PPO

from sof_marl.baselines.rule_based import RuleBasedPolicy
from sof_marl.baselines.single_agent import SingleAgentTreasuryEnv
from sof_marl.env.sme_treasury_env import SMETreasuryEnv
from sof_marl.evaluation.metrics import combined_objective
from sof_marl.evaluation.rollout import (
    EpisodeRecord,
    load_actors,
    make_actor_act_fn,
    make_rule_based_act_fn,
    run_multiagent_episode,
    run_single_agent_episode,
)

# (dotted config key, perturbed values) -- baseline values (config/env.yaml):
# revenue.weekly_sigma_frac=0.15, exposure_resolution_weeks=12, buffer_days_target=27.
PERTURBATIONS: dict[str, list[float]] = {
    "revenue.weekly_sigma_frac": [0.075, 0.30],
    "exposure_resolution_weeks": [6, 24],
    "buffer_days_target": [15, 40],
}
N_EVAL_EPISODES = 25
SEED_BASE = 200000  # disjoint from training seeds (0-4) and the main eval seeds (100000+)
KAPPA_VOLATILITY = 1.0
LAMBDA_SOLVENCY = 3.0


def _write_perturbed_config(base_path: Path, dotted_key: str, value: float, out_path: Path) -> None:
    cfg: dict[str, Any] = yaml.safe_load(base_path.read_text())
    node = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value
    out_path.write_text(yaml.safe_dump(cfg))


def _mean_objective(records: list[EpisodeRecord]) -> float:
    return sum(combined_objective(r, KAPPA_VOLATILITY, LAMBDA_SOLVENCY) for r in records) / len(
        records
    )


def evaluate_all_policies_seed0(
    config_path: Path, models_dir: Path, n_episodes: int, seed_base: int
) -> dict[str, float]:
    rb_env = SMETreasuryEnv(config_path=str(config_path))
    rb_records = [
        run_multiagent_episode(rb_env, make_rule_based_act_fn(RuleBasedPolicy()), seed_base + i)
        for i in range(n_episodes)
    ]

    sa_env = SingleAgentTreasuryEnv(config_path=str(config_path))
    sa_model = PPO.load(str(models_dir / "single_agent_ppo_seed0.zip"))
    sa_records = [
        run_single_agent_episode(sa_env, sa_model, seed_base + i) for i in range(n_episodes)
    ]

    ippo_env = SMETreasuryEnv(config_path=str(config_path))
    ippo_actors = load_actors(models_dir / "ippo_seed0_actors.pt", ippo_env.cfg)
    ippo_records = [
        run_multiagent_episode(ippo_env, make_actor_act_fn(ippo_actors), seed_base + i)
        for i in range(n_episodes)
    ]

    mappo_env = SMETreasuryEnv(config_path=str(config_path))
    mappo_actors = load_actors(models_dir / "mappo_seed0_actors.pt", mappo_env.cfg)
    mappo_records = [
        run_multiagent_episode(mappo_env, make_actor_act_fn(mappo_actors), seed_base + i)
        for i in range(n_episodes)
    ]

    return {
        "rule_based": _mean_objective(rb_records),
        "single_agent_ppo": _mean_objective(sa_records),
        "ippo": _mean_objective(ippo_records),
        "mappo": _mean_objective(mappo_records),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", default="config/env.yaml")
    parser.add_argument("--models-dir", default="data/processed/models")
    args = parser.parse_args()

    base_path = Path(args.env_config)
    models_dir = Path(args.models_dir)
    tmp_dir = Path("data/processed/sensitivity_configs")
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print("Baseline (unperturbed) ...")
    baseline = evaluate_all_policies_seed0(base_path, models_dir, N_EVAL_EPISODES, SEED_BASE)
    print(f"  {baseline}")

    results: dict[str, Any] = {
        "n_episodes": N_EVAL_EPISODES,
        "seed": 0,
        "baseline": baseline,
        "perturbations": {},
    }
    for key, values in PERTURBATIONS.items():
        results["perturbations"][key] = []
        for value in values:
            out_path = tmp_dir / f"{key.replace('.', '_')}_{value}.yaml"
            _write_perturbed_config(base_path, key, value, out_path)
            print(f"{key} = {value} ...")
            policy_metrics = evaluate_all_policies_seed0(
                out_path, models_dir, N_EVAL_EPISODES, SEED_BASE
            )
            others = [
                policy_metrics["rule_based"],
                policy_metrics["single_agent_ppo"],
                policy_metrics["ippo"],
            ]
            mappo_still_best = policy_metrics["mappo"] > max(others)
            print(f"  {policy_metrics}  mappo_still_best={mappo_still_best}")
            results["perturbations"][key].append(
                {"value": value, "metrics": policy_metrics, "mappo_still_best": mappo_still_best}
            )

    out_json = Path("reports/results/sensitivity_analysis.json")
    out_json.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
