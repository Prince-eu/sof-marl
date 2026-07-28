"""PettingZoo ParallelEnv API conformance and basic episode-lifecycle checks."""

from __future__ import annotations

import numpy as np
from pettingzoo.test import parallel_api_test

from sof_marl.env.sme_treasury_env import SMETreasuryEnv


def test_parallel_api_conformance() -> None:
    env = SMETreasuryEnv(shared_reward_weight=1.0)
    parallel_api_test(env, num_cycles=60)


def test_episode_runs_full_horizon_then_all_agents_finish() -> None:
    env = SMETreasuryEnv()
    obs, infos = env.reset(seed=0)
    assert set(env.agents) == set(env.possible_agents)

    week = 0
    while env.agents:
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)
        week += 1
        assert week <= env.cfg.horizon_weeks
    assert week == env.cfg.horizon_weeks
    assert env.agents == []


def test_reset_is_reproducible_given_same_seed() -> None:
    env_a = SMETreasuryEnv()
    env_b = SMETreasuryEnv()
    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)
    for agent in env_a.possible_agents:
        np.testing.assert_array_equal(obs_a[agent], obs_b[agent])


def test_observation_and_action_space_identity_stable() -> None:
    env = SMETreasuryEnv()
    for agent in env.possible_agents:
        assert env.observation_space(agent) is env.observation_space(agent)
        assert env.action_space(agent) is env.action_space(agent)


def test_observations_match_declared_spaces() -> None:
    env = SMETreasuryEnv()
    obs, _ = env.reset(seed=1)
    for agent in env.possible_agents:
        assert env.observation_space(agent).contains(obs[agent]), agent


def test_credit_risk_pending_requests_are_bounded_and_masked() -> None:
    """No more than max_pending_credit_requests slots are ever marked valid."""
    env = SMETreasuryEnv()
    obs, _ = env.reset(seed=3)
    n_slots = env.cfg.max_pending_credit_requests
    for _ in range(20):
        credit_obs = obs["credit_risk"]
        valid_mask = credit_obs[2 : n_slots * 3 : 3]
        assert valid_mask.sum() <= n_slots
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, _, _, _, _ = env.step(actions)
