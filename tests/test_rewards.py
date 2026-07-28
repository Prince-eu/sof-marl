"""Tests for sof_marl.agents.rewards: per-agent and shared reward terms."""

from __future__ import annotations

import numpy as np
import pytest

from sof_marl.agents import rewards
from sof_marl.env import calibration, dynamics


@pytest.fixture
def cfg() -> calibration.EnvConfig:
    return calibration.load_env_config("config/env.yaml")


@pytest.fixture
def agents_cfg() -> rewards.AgentsConfig:
    return rewards.load_agents_config("config/agents.yaml")


def _step_once(
    cfg: calibration.EnvConfig, actions: dynamics.WeekActions, seed: int = 0
) -> tuple[dynamics.FirmState, dynamics.StepResult]:
    rng = np.random.default_rng(seed)
    state = dynamics.initial_state(cfg)
    result = dynamics.step(state, actions, cfg, rng)
    return state, result


def test_liquidity_reward_penalizes_shortfall_more_than_a_healthy_week(
    cfg: calibration.EnvConfig, agents_cfg: rewards.AgentsConfig
) -> None:
    healthy_actions = dynamics.WeekActions(1.0, 0.0, 0.0, 0.0, 0.3, np.array([0.4, 0.3, 0.3]))
    _, healthy_result = _step_once(cfg, healthy_actions)
    r_healthy = rewards.liquidity_reward(healthy_result, cfg, agents_cfg.liquidity_rewards)

    # Force a shortfall: draw nothing, defer nothing, but simulate a large default loss
    # dragging cash deeply negative via the credit_default_loss channel.
    distressed_actions = dynamics.WeekActions(
        1.0, 0.0, 0.0, 0.0, 0.3, np.array([0.4, 0.3, 0.3]), credit_default_loss=500_000.0
    )
    _, distressed_result = _step_once(cfg, distressed_actions)
    r_distressed = rewards.liquidity_reward(distressed_result, cfg, agents_cfg.liquidity_rewards)

    assert distressed_result.shortfall_severity > 0.0
    assert r_distressed < r_healthy


def test_credit_risk_reward_rewards_margin_and_penalizes_loss(
    cfg: calibration.EnvConfig, agents_cfg: rewards.AgentsConfig
) -> None:
    profit_actions = dynamics.WeekActions(
        1.0, 0.0, 0.0, 0.0, 0.3, np.array([0.4, 0.3, 0.3]), credit_income=1000.0
    )
    _, profit_result = _step_once(cfg, profit_actions)
    r_profit = rewards.credit_risk_reward(
        profit_result, cfg, agents_cfg.credit_risk_rewards, outstanding_exposure_total=0.0
    )

    loss_actions = dynamics.WeekActions(
        1.0, 0.0, 0.0, 0.0, 0.3, np.array([0.4, 0.3, 0.3]), credit_default_loss=1000.0
    )
    _, loss_result = _step_once(cfg, loss_actions)
    r_loss = rewards.credit_risk_reward(
        loss_result, cfg, agents_cfg.credit_risk_rewards, outstanding_exposure_total=0.0
    )

    assert r_profit > 0.0
    assert r_loss < 0.0


def test_credit_risk_reward_penalizes_concentration(
    cfg: calibration.EnvConfig, agents_cfg: rewards.AgentsConfig
) -> None:
    actions = dynamics.WeekActions(1.0, 0.0, 0.0, 0.0, 0.3, np.array([0.4, 0.3, 0.3]))
    _, result = _step_once(cfg, actions)
    r_low_concentration = rewards.credit_risk_reward(
        result, cfg, agents_cfg.credit_risk_rewards, outstanding_exposure_total=0.0
    )
    r_high_concentration = rewards.credit_risk_reward(
        result, cfg, agents_cfg.credit_risk_rewards, outstanding_exposure_total=50_000.0
    )
    assert r_high_concentration < r_low_concentration


def test_expenditure_reward_has_diminishing_returns(
    cfg: calibration.EnvConfig, agents_cfg: rewards.AgentsConfig
) -> None:
    low_spend = dynamics.WeekActions(1.0, 0.0, 0.0, 0.0, 0.25, np.array([0.4, 0.3, 0.3]))
    high_spend = dynamics.WeekActions(1.0, 0.0, 0.0, 0.0, 0.5, np.array([0.4, 0.3, 0.3]))
    _, r1 = _step_once(cfg, low_spend)
    _, r2 = _step_once(cfg, high_spend)
    v1 = rewards.expenditure_reward(
        r1, dynamics.initial_state(cfg), cfg, agents_cfg.expenditure_rewards
    )
    v2 = rewards.expenditure_reward(
        r2, dynamics.initial_state(cfg), cfg, agents_cfg.expenditure_rewards
    )
    # doubling spend should less-than-double the value term (concave, k < 1)
    assert v2 < 2 * v1
    assert v2 > v1


def test_shared_reward_penalizes_solvency_breach(
    cfg: calibration.EnvConfig, agents_cfg: rewards.AgentsConfig
) -> None:
    old_state = dynamics.initial_state(cfg)
    healthy_actions = dynamics.WeekActions(1.0, 0.0, 0.0, 0.0, 0.3, np.array([0.4, 0.3, 0.3]))
    _, healthy_result = _step_once(cfg, healthy_actions)
    r_healthy = rewards.shared_reward(healthy_result, old_state, cfg, agents_cfg.shared)

    catastrophic_actions = dynamics.WeekActions(
        1.0, 0.0, 0.0, 0.0, 0.3, np.array([0.4, 0.3, 0.3]), credit_default_loss=10_000_000.0
    )
    _, breach_result = _step_once(cfg, catastrophic_actions)
    r_breach = rewards.shared_reward(breach_result, old_state, cfg, agents_cfg.shared)

    assert breach_result.solvency_breach is True
    assert r_breach < r_healthy


def test_per_agent_rewards_includes_shared_scaled_by_weight(
    cfg: calibration.EnvConfig, agents_cfg: rewards.AgentsConfig
) -> None:
    old_state = dynamics.initial_state(cfg)
    actions = dynamics.WeekActions(1.0, 0.0, 0.0, 0.0, 0.3, np.array([0.4, 0.3, 0.3]))
    _, result = _step_once(cfg, actions)

    r_zero = rewards.per_agent_rewards(
        result, old_state, cfg, agents_cfg, 0.0, shared_reward_weight=0.0
    )
    r_one = rewards.per_agent_rewards(
        result, old_state, cfg, agents_cfg, 0.0, shared_reward_weight=1.0
    )

    for agent in ("liquidity", "credit_risk", "expenditure", "capital_allocation"):
        assert r_one[agent] == pytest.approx(r_zero[agent] + r_one["shared"])
