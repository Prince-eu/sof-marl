"""Tests for training.multi_agent_ppo: end-to-end IPPO/MAPPO training sanity."""

from __future__ import annotations

import copy

import torch

from sof_marl.agents.spaces import AGENT_NAMES
from sof_marl.env.sme_treasury_env import SMETreasuryEnv
from sof_marl.training.multi_agent_ppo import (
    CentralizedCritic,
    IndependentCritics,
    load_ppo_hyperparameters,
    train_multi_agent_ppo,
)


def _tiny_hyperparameters() -> object:
    hp = load_ppo_hyperparameters()
    return hp.__class__(**{**hp.__dict__, "n_steps": 128, "batch_size": 64, "n_epochs": 2})


def test_ippo_training_produces_finite_parameters_and_independent_critics() -> None:
    hp = _tiny_hyperparameters()
    result = train_multi_agent_ppo(
        env_factory=lambda: SMETreasuryEnv(),
        centralized_critic=False,
        shared_reward_weight=0.0,
        seed=0,
        hp=hp,  # type: ignore[arg-type]
        total_timesteps=hp.n_steps * 2,  # type: ignore[attr-defined]
    )
    assert isinstance(result["critic_model"], IndependentCritics)
    for agent in AGENT_NAMES:
        for param in result["actors"][agent].parameters():
            assert torch.isfinite(param).all()
    assert result["total_timesteps"] == hp.n_steps * 2  # type: ignore[attr-defined]
    assert result["wall_clock_seconds"] > 0


def test_mappo_training_produces_finite_parameters_and_centralized_critic() -> None:
    hp = _tiny_hyperparameters()
    result = train_multi_agent_ppo(
        env_factory=lambda: SMETreasuryEnv(),
        centralized_critic=True,
        shared_reward_weight=1.0,
        seed=0,
        hp=hp,  # type: ignore[arg-type]
        total_timesteps=hp.n_steps * 2,  # type: ignore[attr-defined]
    )
    assert isinstance(result["critic_model"], CentralizedCritic)
    for agent in AGENT_NAMES:
        for param in result["actors"][agent].parameters():
            assert torch.isfinite(param).all()
    for param in result["critic_model"].net.parameters():
        assert torch.isfinite(param).all()


def test_training_actually_updates_actor_parameters() -> None:
    hp = _tiny_hyperparameters()
    torch.manual_seed(0)
    from sof_marl.training.policy import build_actor

    env = SMETreasuryEnv()
    initial_params = {
        agent: copy.deepcopy(list(build_actor(agent, env.cfg).parameters()))
        for agent in AGENT_NAMES
    }

    result = train_multi_agent_ppo(
        env_factory=lambda: SMETreasuryEnv(),
        centralized_critic=False,
        shared_reward_weight=0.0,
        seed=0,
        hp=hp,  # type: ignore[arg-type]
        total_timesteps=hp.n_steps * 2,  # type: ignore[attr-defined]
    )
    # at least one agent's parameters should have moved from a fresh random init
    any_changed = False
    for agent in AGENT_NAMES:
        trained = list(result["actors"][agent].parameters())
        for p_init, p_trained in zip(initial_params[agent], trained, strict=False):
            if not torch.allclose(p_init, p_trained):
                any_changed = True
    assert any_changed


def test_learning_curve_has_expected_structure() -> None:
    hp = _tiny_hyperparameters()
    result = train_multi_agent_ppo(
        env_factory=lambda: SMETreasuryEnv(),
        centralized_critic=False,
        shared_reward_weight=0.0,
        seed=0,
        hp=hp,  # type: ignore[arg-type]
        total_timesteps=hp.n_steps * 3,  # type: ignore[attr-defined]
    )
    curve = result["learning_curve"]
    assert isinstance(curve, list)
    for entry in curve:
        assert "timesteps" in entry
        assert "mean_episode_return" in entry
        assert "mean_firm_value" in entry
