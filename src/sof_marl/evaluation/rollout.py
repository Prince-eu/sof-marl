"""Deterministic held-out evaluation rollouts (EVALUATION.md sections 1-3).

All four policies -- rule-based, single-agent PPO, IPPO, MAPPO -- run on the
identical set of held-out evaluation episodes: fixed seeds disjoint from
training (``config/train.yaml`` ``eval.seed_base``, ``eval.n_episodes``),
deterministic (greedy) actions. Learned policies are evaluated once per trained
seed (``config/train.yaml`` ``seeds``, >= 5), each against the same eval
episodes, so every metric is reported as mean +/- std across training seeds;
rule-based has no training-seed axis (a fixed heuristic, not learned) and is
evaluated once, directly on the eval episodes.

Running this module (``python -m sof_marl.evaluation.rollout``) executes every
rollout, computes the full metric suite (``evaluation/metrics.py``), and writes
``reports/results/eval_episodes_<policy>.json`` (raw per-episode records) and
``reports/results/eval_metrics.json`` (the metric-suite summary table +
coordination-lift test), which ``evaluation/figures.py`` reads to build the
figures.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml
from stable_baselines3 import PPO

from sof_marl.agents.spaces import AGENT_NAMES
from sof_marl.baselines.rule_based import RuleBasedPolicy
from sof_marl.baselines.single_agent import SingleAgentTreasuryEnv
from sof_marl.env import dynamics
from sof_marl.env.sme_treasury_env import SMETreasuryEnv
from sof_marl.training.policy import build_actor

ActFn = Callable[[SMETreasuryEnv, dict[str, Any]], dict[str, Any]]


@dataclass
class EpisodeRecord:
    """Everything the metric suite (EVALUATION.md section 2) needs from one episode."""

    firm_values: list[float] = field(default_factory=list)  # length horizon_weeks+1 (incl. week 0)
    cash_values: list[float] = field(default_factory=list)
    invested_values: list[float] = field(default_factory=list)
    reserve_values: list[float] = field(default_factory=list)
    term_debt_values: list[float] = field(default_factory=list)
    solvency_breaches: list[bool] = field(default_factory=list)  # length horizon_weeks
    shortfall_severities: list[float] = field(default_factory=list)
    financing_costs: list[float] = field(default_factory=list)
    credit_decisions: list[dict[str, Any]] = field(default_factory=list)
    credit_resolutions: list[dict[str, Any]] = field(default_factory=list)


def _initial_record(state: dynamics.FirmState, cfg: Any) -> EpisodeRecord:
    return EpisodeRecord(
        firm_values=[dynamics.firm_value(state, cfg)],
        cash_values=[state.cash],
        invested_values=[state.invested],
        reserve_values=[state.reserve],
        term_debt_values=[state.term_debt],
    )


def run_multiagent_episode(env: SMETreasuryEnv, act_fn: ActFn, seed: int) -> EpisodeRecord:
    obs, _ = env.reset(seed=seed)
    record = _initial_record(env.state, env.cfg)
    while env.agents:
        actions = act_fn(env, obs)
        obs, _rewards, _terms, _truncs, infos = env.step(actions)
        diag = infos[AGENT_NAMES[0]]
        record.firm_values.append(diag["firm_value"])
        record.cash_values.append(env.state.cash)
        record.invested_values.append(env.state.invested)
        record.reserve_values.append(env.state.reserve)
        record.term_debt_values.append(env.state.term_debt)
        record.solvency_breaches.append(diag["solvency_breach"])
        record.shortfall_severities.append(diag["shortfall_severity"])
        record.financing_costs.append(diag["financing_cost"])
        record.credit_decisions.extend(diag["credit_decisions"])
        record.credit_resolutions.extend(diag["credit_resolutions"])
    return record


def run_single_agent_episode(env: SingleAgentTreasuryEnv, model: PPO, seed: int) -> EpisodeRecord:
    obs, info = env.reset(seed=seed)
    record = _initial_record(env._env.state, env._env.cfg)
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        record.firm_values.append(info["firm_value"])
        record.cash_values.append(env._env.state.cash)
        record.invested_values.append(env._env.state.invested)
        record.reserve_values.append(env._env.state.reserve)
        record.term_debt_values.append(env._env.state.term_debt)
        record.solvency_breaches.append(info["solvency_breach"])
        record.shortfall_severities.append(info["shortfall_severity"])
        record.financing_costs.append(info["financing_cost"])
        record.credit_decisions.extend(info["credit_decisions"])
        record.credit_resolutions.extend(info["credit_resolutions"])
    return record


def make_rule_based_act_fn(policy: RuleBasedPolicy) -> ActFn:
    def act_fn(env: SMETreasuryEnv, _obs: dict[str, Any]) -> dict[str, Any]:
        return policy.act(env)

    return act_fn


def make_actor_act_fn(actors: dict[str, nn.Module]) -> ActFn:
    """Deterministic (greedy) actions from a trained actor set (IPPO or MAPPO)."""

    def act_fn(_env: SMETreasuryEnv, obs: dict[str, Any]) -> dict[str, Any]:
        actions: dict[str, Any] = {}
        for agent in AGENT_NAMES:
            obs_t = torch.as_tensor(obs[agent], dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action_t = actors[agent].deterministic_action(obs_t).squeeze(0)
            clipped = actors[agent].clip_to_bounds(action_t)
            actions[agent] = clipped.numpy()
        return actions

    return act_fn


def load_actors(state_dict_path: Path, cfg: Any) -> dict[str, nn.Module]:
    actors = {agent: build_actor(agent, cfg) for agent in AGENT_NAMES}
    state = torch.load(state_dict_path, map_location="cpu")
    for agent in AGENT_NAMES:
        actors[agent].load_state_dict(state[agent])
        actors[agent].eval()
    return actors


def evaluate_rule_based(
    n_episodes: int, seed_base: int, config_path: str = "config/env.yaml"
) -> list[EpisodeRecord]:
    env = SMETreasuryEnv(config_path=config_path)
    policy = RuleBasedPolicy()
    act_fn = make_rule_based_act_fn(policy)
    return [run_multiagent_episode(env, act_fn, seed_base + i) for i in range(n_episodes)]


def evaluate_marl_policy(
    models_dir: Path,
    policy_name: str,
    seeds: list[int],
    n_episodes: int,
    seed_base: int,
    config_path: str = "config/env.yaml",
) -> dict[int, list[EpisodeRecord]]:
    env = SMETreasuryEnv(config_path=config_path)
    out: dict[int, list[EpisodeRecord]] = {}
    for seed in seeds:
        actors = load_actors(models_dir / f"{policy_name}_seed{seed}_actors.pt", env.cfg)
        act_fn = make_actor_act_fn(actors)
        out[seed] = [run_multiagent_episode(env, act_fn, seed_base + i) for i in range(n_episodes)]
    return out


def evaluate_single_agent(
    models_dir: Path,
    seeds: list[int],
    n_episodes: int,
    seed_base: int,
    config_path: str = "config/env.yaml",
    model_stem: str = "single_agent_ppo",
) -> dict[int, list[EpisodeRecord]]:
    out: dict[int, list[EpisodeRecord]] = {}
    for seed in seeds:
        env = SingleAgentTreasuryEnv(config_path=config_path)
        model = PPO.load(str(models_dir / f"{model_stem}_seed{seed}.zip"))
        out[seed] = [run_single_agent_episode(env, model, seed_base + i) for i in range(n_episodes)]
    return out


def _episode_to_dict(record: EpisodeRecord) -> dict[str, Any]:
    return asdict(record)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/train.yaml")
    parser.add_argument("--env-config", default="config/env.yaml")
    parser.add_argument("--tag", default="", help="model/output suffix, e.g. 'stress'")
    args = parser.parse_args()

    train_cfg: dict[str, Any] = yaml.safe_load(Path(args.config).read_text())
    seeds: list[int] = train_cfg["seeds"]
    n_episodes: int = train_cfg["eval"]["n_episodes"]
    seed_base: int = train_cfg["eval"]["seed_base"]
    env_config: str = args.env_config
    suffix = f"_{args.tag}" if args.tag else ""
    models_dir = Path("data/processed/models")
    results_dir = Path("reports/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Evaluating rule_based{suffix} on {n_episodes} held-out episodes ...")
    rule_based_episodes = evaluate_rule_based(n_episodes, seed_base, env_config)
    (results_dir / f"eval_episodes_rule_based{suffix}.json").write_text(
        json.dumps([_episode_to_dict(r) for r in rule_based_episodes], indent=2)
    )

    print(f"Evaluating single_agent_ppo{suffix} ({len(seeds)} seeds x {n_episodes} episodes) ...")
    single_agent_episodes = evaluate_single_agent(
        models_dir, seeds, n_episodes, seed_base, env_config, f"single_agent_ppo{suffix}"
    )
    (results_dir / f"eval_episodes_single_agent_ppo{suffix}.json").write_text(
        json.dumps(
            {
                seed: [_episode_to_dict(r) for r in eps]
                for seed, eps in single_agent_episodes.items()
            },
            indent=2,
        )
    )

    print(f"Evaluating ippo{suffix} ({len(seeds)} seeds x {n_episodes} episodes) ...")
    ippo_episodes = evaluate_marl_policy(
        models_dir, f"ippo{suffix}", seeds, n_episodes, seed_base, env_config
    )
    (results_dir / f"eval_episodes_ippo{suffix}.json").write_text(
        json.dumps(
            {seed: [_episode_to_dict(r) for r in eps] for seed, eps in ippo_episodes.items()},
            indent=2,
        )
    )

    print(f"Evaluating mappo{suffix} ({len(seeds)} seeds x {n_episodes} episodes) ...")
    mappo_episodes = evaluate_marl_policy(
        models_dir, f"mappo{suffix}", seeds, n_episodes, seed_base, env_config
    )
    (results_dir / f"eval_episodes_mappo{suffix}.json").write_text(
        json.dumps(
            {seed: [_episode_to_dict(r) for r in eps] for seed, eps in mappo_episodes.items()},
            indent=2,
        )
    )

    print("All rollouts complete. Computing metric suite ...")
    from sof_marl.evaluation import metrics as metrics_module

    summary = metrics_module.build_evaluation_summary(
        rule_based_episodes=rule_based_episodes,
        single_agent_episodes=single_agent_episodes,
        ippo_episodes=ippo_episodes,
        mappo_episodes=mappo_episodes,
        cfg_path=args.config,
        env_cfg_path=env_config,
    )
    (results_dir / f"eval_metrics{suffix}.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote eval_episodes_*{suffix}.json and {results_dir / f'eval_metrics{suffix}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
