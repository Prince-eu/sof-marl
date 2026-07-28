"""Tests for training.rollout: GAE correctness and rollout collection shapes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from sof_marl.agents.spaces import AGENT_NAMES
from sof_marl.env.sme_treasury_env import SMETreasuryEnv
from sof_marl.training.policy import build_actor
from sof_marl.training.rollout import collect_rollout, compute_gae


def test_gae_reduces_to_monte_carlo_with_gamma_and_lambda_one() -> None:
    rewards = np.array([1.0, 1.0, 1.0])
    values = np.zeros(3)
    dones = np.array([False, False, True])
    bootstrap_values = np.array([0.0, 0.0, 0.0])
    adv, ret = compute_gae(rewards, values, dones, bootstrap_values, 0.0, gamma=1.0, gae_lambda=1.0)
    np.testing.assert_allclose(ret, [3.0, 2.0, 1.0])


def test_gae_does_not_leak_across_episode_boundary() -> None:
    """A hand-verified two-episode case: steps [0,1] form episode A (bootstrap=5
    at its terminal step 1), steps [2,3,4] form episode B (bootstrap=0)."""
    rewards = np.array([1.0, 1.0, 1.0, 10.0, 10.0])
    values = np.zeros(5)
    dones = np.array([False, True, False, False, True])
    bootstrap_values = np.array([0.0, 5.0, 0.0, 0.0, 0.0])
    adv, ret = compute_gae(rewards, values, dones, bootstrap_values, 0.0, gamma=1.0, gae_lambda=1.0)
    np.testing.assert_allclose(adv, [7.0, 6.0, 21.0, 20.0, 10.0])
    # episode B's advantage at its first step must equal the sum of its own
    # rewards only (1+10+10), not anything from episode A.
    assert adv[2] == pytest.approx(1.0 + 10.0 + 10.0)


def test_gae_with_discounting_matches_hand_calculation() -> None:
    rewards = np.array([1.0, 2.0])
    values = np.array([0.5, 0.5])
    dones = np.array([False, True])
    bootstrap_values = np.array([0.0, 0.0])
    gamma, lam = 0.9, 0.8
    adv, ret = compute_gae(rewards, values, dones, bootstrap_values, 0.0, gamma, lam)
    delta1 = rewards[1] + gamma * 0.0 - values[1]
    delta0 = rewards[0] + gamma * values[1] - values[0]
    expected_adv1 = delta1
    expected_adv0 = delta0 + gamma * lam * expected_adv1
    np.testing.assert_allclose(adv, [expected_adv0, expected_adv1])


def test_collect_rollout_produces_correct_length_trajectories() -> None:
    cfg_env = SMETreasuryEnv()
    actors = {agent: build_actor(agent, cfg_env.cfg) for agent in AGENT_NAMES}

    def value_fn(obs: dict[str, np.ndarray]) -> dict[str, float]:
        return {agent: 0.0 for agent in AGENT_NAMES}

    obs, _ = cfg_env.reset(seed=0)
    n_steps = 75
    batch, obs_out = collect_rollout(cfg_env, actors, value_fn, n_steps, obs)
    for agent in AGENT_NAMES:
        traj = batch.trajectories[agent]
        assert len(traj.obs) == n_steps
        assert len(traj.actions) == n_steps
        assert len(traj.rewards) == n_steps
        assert len(traj.dones) == n_steps
    # 75 steps > one 52-week episode, so at least one episode should complete
    assert len(batch.episode_returns) >= 1
    assert all(agent in obs_out for agent in AGENT_NAMES)


def test_collect_rollout_actions_are_within_declared_spaces() -> None:
    env = SMETreasuryEnv()
    actors = {agent: build_actor(agent, env.cfg) for agent in AGENT_NAMES}

    def value_fn(obs: dict[str, np.ndarray]) -> dict[str, float]:
        return {agent: 0.0 for agent in AGENT_NAMES}

    obs, _ = env.reset(seed=0)
    batch, _ = collect_rollout(env, actors, value_fn, 20, obs)
    for agent in AGENT_NAMES:
        space = env.action_space(agent)
        for action in batch.trajectories[agent].actions:
            clipped = actors[agent].clip_to_bounds(torch.as_tensor(action)).numpy()
            assert space.contains(clipped.astype(space.dtype))
