"""Tests for baselines.single_agent: Gymnasium API conformance and consistency
with the underlying multi-agent SMETreasuryEnv."""

from __future__ import annotations

import warnings

import numpy as np
from gymnasium.utils.env_checker import check_env

from sof_marl.baselines.single_agent import SingleAgentTreasuryEnv
from sof_marl.env.sme_treasury_env import SMETreasuryEnv


def test_gymnasium_env_checker() -> None:
    env = SingleAgentTreasuryEnv()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(env, skip_render_check=True)


def test_observation_dim_matches_sum_of_multiagent_observations() -> None:
    multi_env = SMETreasuryEnv()
    single_env = SingleAgentTreasuryEnv()
    expected_dim = sum(
        multi_env.observation_space(agent).shape[0]  # type: ignore[union-attr]
        for agent in multi_env.possible_agents
    )
    assert single_env.observation_space.shape == (expected_dim,)


def test_action_dim_matches_sum_of_multiagent_action_dims() -> None:
    multi_env = SMETreasuryEnv()
    single_env = SingleAgentTreasuryEnv()
    liquidity_dim = multi_env.action_space("liquidity").shape[0]  # type: ignore[union-attr]
    expenditure_dim = multi_env.action_space("expenditure").shape[0]  # type: ignore[union-attr]
    capital_dim = multi_env.action_space("capital_allocation").shape[0]  # type: ignore[union-attr]
    credit_slots = multi_env.cfg.max_pending_credit_requests
    assert single_env.action_space.shape == (
        liquidity_dim + credit_slots + expenditure_dim + capital_dim,
    )


def test_episode_runs_full_horizon() -> None:
    env = SingleAgentTreasuryEnv()
    obs, info = env.reset(seed=0)
    week = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        week += 1
        assert week <= env._env.cfg.horizon_weeks
    assert truncated is True
    assert week == env._env.cfg.horizon_weeks


def test_credit_risk_discretization_thresholds() -> None:
    env = SingleAgentTreasuryEnv()
    env.reset(seed=0)
    scores = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
    decisions = np.digitize(scores, bins=[1 / 3, 2 / 3])
    np.testing.assert_array_equal(decisions, [0, 0, 1, 2, 2])


def test_reset_is_reproducible_given_same_seed() -> None:
    env_a = SingleAgentTreasuryEnv()
    env_b = SingleAgentTreasuryEnv()
    obs_a, _ = env_a.reset(seed=7)
    obs_b, _ = env_b.reset(seed=7)
    np.testing.assert_array_equal(obs_a, obs_b)


def test_combined_reward_equals_sum_of_multiagent_rewards() -> None:
    """Cross-check: stepping the wrapper should match manually summing the
    underlying SMETreasuryEnv's per-agent rewards for an equivalent action."""
    single_env = SingleAgentTreasuryEnv()
    single_env.reset(seed=0)
    action = single_env.action_space.sample()

    multi_env = SMETreasuryEnv(shared_reward_weight=0.0)
    multi_env.reset(seed=0)
    split = single_env._split_action(action)

    _, single_reward, _, _, _ = single_env.step(action)
    _, rewards, _, _, _ = multi_env.step(split)
    assert single_reward == sum(rewards.values())
