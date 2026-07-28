"""Tests for evaluation.rollout: episode mechanics, independent of any specific
pre-trained checkpoint (consistent with the rest of the suite -- models are
gitignored, so tests build tiny actors/policies on the fly rather than depending
on data/processed/models/ existing)."""

from __future__ import annotations

from pathlib import Path

import torch
from stable_baselines3 import PPO

from sof_marl.agents.spaces import AGENT_NAMES
from sof_marl.baselines.rule_based import RuleBasedPolicy
from sof_marl.baselines.single_agent import SingleAgentTreasuryEnv
from sof_marl.env.sme_treasury_env import SMETreasuryEnv
from sof_marl.evaluation.rollout import (
    load_actors,
    make_actor_act_fn,
    make_rule_based_act_fn,
    run_multiagent_episode,
    run_single_agent_episode,
)
from sof_marl.training.policy import build_actor


def test_run_multiagent_episode_with_rule_based_has_correct_lengths() -> None:
    env = SMETreasuryEnv()
    act_fn = make_rule_based_act_fn(RuleBasedPolicy())
    record = run_multiagent_episode(env, act_fn, seed=100000)
    assert len(record.firm_values) == env.cfg.horizon_weeks + 1
    assert len(record.cash_values) == env.cfg.horizon_weeks + 1
    assert len(record.solvency_breaches) == env.cfg.horizon_weeks
    assert len(record.financing_costs) == env.cfg.horizon_weeks


def test_run_multiagent_episode_is_deterministic_for_rule_based() -> None:
    env = SMETreasuryEnv()
    act_fn = make_rule_based_act_fn(RuleBasedPolicy())
    record_a = run_multiagent_episode(env, act_fn, seed=100000)
    record_b = run_multiagent_episode(env, act_fn, seed=100000)
    assert record_a.firm_values == record_b.firm_values
    assert record_a.cash_values == record_b.cash_values


def test_run_multiagent_episode_with_fresh_actors() -> None:
    env = SMETreasuryEnv()
    actors = {agent: build_actor(agent, env.cfg) for agent in AGENT_NAMES}
    for actor in actors.values():
        actor.eval()
    act_fn = make_actor_act_fn(actors)
    record = run_multiagent_episode(env, act_fn, seed=100000)
    assert len(record.firm_values) == env.cfg.horizon_weeks + 1
    assert len(record.solvency_breaches) == env.cfg.horizon_weeks
    assert set(record.solvency_breaches) <= {True, False}


def test_load_actors_roundtrip_preserves_behavior(tmp_path: Path) -> None:
    env = SMETreasuryEnv()
    actors = {agent: build_actor(agent, env.cfg) for agent in AGENT_NAMES}
    for actor in actors.values():
        actor.eval()
    state = {agent: actors[agent].state_dict() for agent in AGENT_NAMES}
    path = tmp_path / "actors.pt"
    torch.save(state, path)

    loaded = load_actors(path, env.cfg)
    obs, _ = env.reset(seed=100000)
    for agent in AGENT_NAMES:
        obs_t = torch.as_tensor(obs[agent], dtype=torch.float32).unsqueeze(0)
        original_out = actors[agent].deterministic_action(obs_t)
        loaded_out = loaded[agent].deterministic_action(obs_t)
        torch.testing.assert_close(original_out, loaded_out)


def test_run_single_agent_episode_produces_correct_lengths(tmp_path: Path) -> None:
    env = SingleAgentTreasuryEnv()
    model = PPO("MlpPolicy", env, n_steps=64, batch_size=32, n_epochs=1, verbose=0)
    model.learn(total_timesteps=64)

    record = run_single_agent_episode(env, model, seed=100000)
    assert len(record.firm_values) == env._env.cfg.horizon_weeks + 1
    assert len(record.solvency_breaches) == env._env.cfg.horizon_weeks
    assert len(record.credit_decisions) >= 0
