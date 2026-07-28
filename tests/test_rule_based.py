"""Tests for baselines.rule_based: sensible, in-bounds, deterministic behavior."""

from __future__ import annotations

import numpy as np
import pytest

from sof_marl.baselines.rule_based import RuleBasedPolicy
from sof_marl.env.sme_treasury_env import SMETreasuryEnv


def test_actions_are_within_declared_spaces() -> None:
    env = SMETreasuryEnv()
    policy = RuleBasedPolicy()
    env.reset(seed=0)
    for _ in range(10):
        actions = policy.act(env)
        for agent in env.possible_agents:
            assert env.action_space(agent).contains(actions[agent]), agent
        env.step(actions)


def test_policy_is_deterministic_given_same_state() -> None:
    env = SMETreasuryEnv()
    policy = RuleBasedPolicy()
    env.reset(seed=0)
    actions_a = policy.act(env)
    actions_b = policy.act(env)
    for agent in env.possible_agents:
        np.testing.assert_array_equal(actions_a[agent], actions_b[agent])


def test_liquidity_pays_payables_on_due_date_and_never_accelerates_receivables() -> None:
    env = SMETreasuryEnv()
    policy = RuleBasedPolicy()
    env.reset(seed=0)
    actions = policy.act(env)
    defer_frac = actions["liquidity"][2]
    accelerate_frac = actions["liquidity"][3]
    assert defer_frac == 0.0
    assert accelerate_frac == 0.0


def test_liquidity_uses_the_calibrated_buffer_target() -> None:
    env = SMETreasuryEnv()
    policy = RuleBasedPolicy()
    env.reset(seed=0)
    actions = policy.act(env)
    assert actions["liquidity"][0] == policy.config.target_buffer_frac == 1.0


def test_credit_risk_never_uses_the_premium_tier() -> None:
    """The rule-based policy is a single-cutoff approve/deny rule (no premium tier)."""
    env = SMETreasuryEnv()
    policy = RuleBasedPolicy()
    env.reset(seed=0)
    for _ in range(52):
        actions = policy.act(env)
        assert not np.any(actions["credit_risk"] == 2)
        if not env.agents:
            break
        env.step(actions)


def test_credit_risk_approves_low_risk_and_denies_high_risk() -> None:
    env = SMETreasuryEnv()
    policy = RuleBasedPolicy()
    env.reset(seed=0)

    class _FakeEnv:
        cfg = env.cfg

        @staticmethod
        def pending_exposures() -> list[dict[str, float]]:
            return [
                {"predicted_prob": 0.01, "exposure_amount": 1000.0},
                {"predicted_prob": 0.9, "exposure_amount": 1000.0},
            ]

    action = policy._credit_risk_action(_FakeEnv())  # type: ignore[arg-type]
    assert action[0] == 1  # approve, well below cutoff
    assert action[1] == 0  # deny, well above cutoff


def test_capital_allocation_is_a_fixed_simplex() -> None:
    env = SMETreasuryEnv()
    policy = RuleBasedPolicy()
    env.reset(seed=0)
    actions = policy.act(env)
    alloc = actions["capital_allocation"]
    assert alloc.sum() == pytest.approx(1.0)
    np.testing.assert_allclose(alloc, np.array(policy.config.capital_allocation_split), rtol=1e-6)


def test_full_episode_runs_without_solvency_breach_under_default_calibration() -> None:
    """Not a correctness requirement in general, but a smoke check that this
    baseline is not a strawman that immediately breaches under the shipped
    calibration and a fixed seed."""
    env = SMETreasuryEnv()
    policy = RuleBasedPolicy()
    env.reset(seed=0)
    breaches = 0
    while env.agents:
        actions = policy.act(env)
        _, _, _, _, infos = env.step(actions)
        if infos["liquidity"]["solvency_breach"]:
            breaches += 1
    assert breaches == 0
