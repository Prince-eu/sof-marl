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


def test_credit_risk_reward_rewards_positive_ev_and_penalizes_negative_ev(
    cfg: calibration.EnvConfig, agents_cfg: rewards.AgentsConfig
) -> None:
    """R_cred now tracks the immediate expected value of the week's decisions:
    positive EV (approving profitable exposures) is rewarded, negative EV penalized."""
    r_profit = rewards.credit_risk_reward(
        cfg,
        agents_cfg.credit_risk_rewards,
        credit_decision_ev=1000.0,
        outstanding_exposure_total=0.0,
    )
    r_loss = rewards.credit_risk_reward(
        cfg,
        agents_cfg.credit_risk_rewards,
        credit_decision_ev=-1000.0,
        outstanding_exposure_total=0.0,
    )
    assert r_profit > 0.0
    assert r_loss < 0.0
    # deny-everything (zero EV, no outstanding) is exactly neutral, not negative:
    r_deny = rewards.credit_risk_reward(
        cfg, agents_cfg.credit_risk_rewards, credit_decision_ev=0.0, outstanding_exposure_total=0.0
    )
    assert r_deny == pytest.approx(0.0)


def test_credit_risk_concentration_penalty_is_threshold_based(
    cfg: calibration.EnvConfig, agents_cfg: rewards.AgentsConfig
) -> None:
    """Prudent lending below the budget is free; only over-concentration is penalized."""
    w = agents_cfg.credit_risk_rewards
    budget = w.concentration_budget_frac * cfg.credit_line.limit

    r_below = rewards.credit_risk_reward(cfg, w, 0.0, outstanding_exposure_total=budget * 0.5)
    r_at = rewards.credit_risk_reward(cfg, w, 0.0, outstanding_exposure_total=budget)
    r_above = rewards.credit_risk_reward(cfg, w, 0.0, outstanding_exposure_total=budget * 2.0)

    assert r_below == pytest.approx(0.0)  # no penalty below the budget
    assert r_at == pytest.approx(0.0)  # no penalty at the budget
    assert r_above < 0.0  # penalized only above the budget
    assert r_above < r_below


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
        result, old_state, cfg, agents_cfg, 0.0, 0.0, shared_reward_weight=0.0
    )
    r_one = rewards.per_agent_rewards(
        result, old_state, cfg, agents_cfg, 0.0, 0.0, shared_reward_weight=1.0
    )

    for agent in ("liquidity", "credit_risk", "expenditure", "capital_allocation"):
        assert r_one[agent] == pytest.approx(r_zero[agent] + r_one["shared"])
