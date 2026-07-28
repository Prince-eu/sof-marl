"""Tests for training.policy: actor/critic shapes and log-prob self-consistency."""

from __future__ import annotations

import torch

from sof_marl.agents.spaces import AGENT_NAMES, action_spaces, observation_spaces
from sof_marl.env.calibration import load_env_config
from sof_marl.training.policy import Critic, GaussianActor, MultiDiscreteActor, build_actor


def test_build_actor_dispatches_by_space_type() -> None:
    cfg = load_env_config()
    act_spaces = action_spaces(cfg)
    for agent in AGENT_NAMES:
        actor = build_actor(agent, cfg)
        if agent == "credit_risk":
            assert isinstance(actor, MultiDiscreteActor)
        else:
            assert isinstance(actor, GaussianActor)
        assert actor is not None
        assert act_spaces[agent] is not None


def test_gaussian_actor_log_prob_matches_sampled_action() -> None:
    cfg = load_env_config()
    actor = build_actor("liquidity", cfg)
    obs_dim = observation_spaces(cfg)["liquidity"].shape[0]
    obs = torch.randn(8, obs_dim)
    action, log_prob = actor.sample(obs)
    log_prob_recomputed, entropy = actor.log_prob_entropy(obs, action)
    assert torch.allclose(log_prob, log_prob_recomputed)
    assert torch.isfinite(entropy).all()


def test_gaussian_actor_deterministic_action_within_bounds() -> None:
    cfg = load_env_config()
    actor = build_actor("capital_allocation", cfg)
    obs_dim = observation_spaces(cfg)["capital_allocation"].shape[0]
    obs = torch.randn(20, obs_dim) * 5  # push mean_net outputs toward extremes
    action = actor.deterministic_action(obs)
    assert torch.all(action >= actor.low)
    assert torch.all(action <= actor.high)


def test_multidiscrete_actor_actions_are_valid_categories() -> None:
    cfg = load_env_config()
    actor = build_actor("credit_risk", cfg)
    obs_dim = observation_spaces(cfg)["credit_risk"].shape[0]
    obs = torch.randn(10, obs_dim)
    action, log_prob = actor.sample(obs)
    assert action.shape == (10, cfg.max_pending_credit_requests)
    assert torch.all(action >= 0)
    assert torch.all(action < 3)
    log_prob_recomputed, entropy = actor.log_prob_entropy(obs, action)
    assert torch.allclose(log_prob, log_prob_recomputed)
    assert torch.isfinite(entropy).all()


def test_critic_output_shape_matches_n_heads() -> None:
    critic = Critic(obs_dim=10, n_heads=4)
    out = critic(torch.randn(6, 10))
    assert out.shape == (6, 4)
