"""Tests for sof_marl.env.dynamics: the cash-conservation invariant and related.

ARCHITECTURE.md section 3: cash_{t+1} == cash_t + inflows_t - outflows_t to floating
tolerance, with every term a named inflow/outflow -- no term may inject or destroy
value outside one.
"""

from __future__ import annotations

import numpy as np
import pytest

from sof_marl.env import calibration, dynamics


@pytest.fixture
def cfg() -> calibration.EnvConfig:
    return calibration.load_env_config("config/env.yaml")


def _random_actions(rng: np.random.Generator) -> dynamics.WeekActions:
    alloc = rng.uniform(0, 1, size=3)
    return dynamics.WeekActions(
        target_buffer_frac=float(rng.uniform(0.0, 2.0)),
        credit_draw_frac=float(rng.uniform(-1.0, 1.0)),
        defer_payables_frac=float(rng.uniform(0.0, 1.0)),
        accelerate_receivables_frac=float(rng.uniform(0.0, 1.0)),
        spend_now_frac=float(rng.uniform(0.0, 1.0)),
        capital_allocation=alloc,
        credit_income=float(rng.uniform(0.0, 500.0)),
        credit_default_loss=float(rng.uniform(0.0, 500.0)),
    )


def test_initial_state_matches_config(cfg: calibration.EnvConfig) -> None:
    state = dynamics.initial_state(cfg)
    assert state.week == 0
    assert state.cash == cfg.initial.cash
    assert state.term_debt == cfg.initial.term_debt
    assert state.credit_drawn == cfg.initial.credit_drawn
    assert state.reserve == cfg.initial.reserve
    assert state.invested == cfg.initial.invested
    assert state.ar_buckets.sum() == pytest.approx(cfg.initial.accounts_receivable)
    assert state.ap_due.sum() == pytest.approx(cfg.initial.accounts_payable)
    assert state.ops_health == 1.0


def test_cash_invariant_holds_over_many_random_steps(cfg: calibration.EnvConfig) -> None:
    rng = np.random.default_rng(12345)
    state = dynamics.initial_state(cfg)
    for _ in range(200):
        actions = _random_actions(rng)
        result = dynamics.step(state, actions, cfg, rng)
        inflow_total = sum(result.flows[k] for k in dynamics.INFLOW_KEYS)
        outflow_total = sum(result.flows[k] for k in dynamics.OUTFLOW_KEYS)
        expected_cash = state.cash + inflow_total - outflow_total
        assert result.state.cash == pytest.approx(expected_cash, abs=1e-6)
        state = result.state


def test_cash_invariant_holds_with_extreme_actions(cfg: calibration.EnvConfig) -> None:
    """Boundary actions (max draw, max defer, max spend, all-in allocation) must still close."""
    rng = np.random.default_rng(7)
    state = dynamics.initial_state(cfg)
    extreme_actions = [
        dynamics.WeekActions(2.0, 1.0, 1.0, 1.0, 1.0, np.array([1.0, 0.0, 0.0]), 0.0, 0.0),
        dynamics.WeekActions(0.0, -1.0, 0.0, 0.0, 0.0, np.array([0.0, 1.0, 0.0]), 0.0, 0.0),
        dynamics.WeekActions(1.0, 0.0, 0.0, 0.0, 0.0, np.array([0.0, 0.0, 1.0]), 300.0, 900.0),
    ]
    for actions in extreme_actions:
        result = dynamics.step(state, actions, cfg, rng)
        inflow_total = sum(result.flows[k] for k in dynamics.INFLOW_KEYS)
        outflow_total = sum(result.flows[k] for k in dynamics.OUTFLOW_KEYS)
        assert result.state.cash == pytest.approx(
            state.cash + inflow_total - outflow_total, abs=1e-6
        )
        state = result.state


def test_no_negative_credit_line_or_negative_term_debt(cfg: calibration.EnvConfig) -> None:
    """Repay/paydown actions may never push balances below zero."""
    rng = np.random.default_rng(1)
    state = dynamics.initial_state(cfg)
    full_repay = dynamics.WeekActions(1.0, -1.0, 0.0, 0.0, 0.0, np.array([1.0, 0.0, 0.0]), 0.0, 0.0)
    for _ in range(10):
        result = dynamics.step(state, full_repay, cfg, rng)
        assert result.state.credit_drawn >= 0.0
        assert result.state.term_debt >= 0.0
        state = result.state


def test_deny_all_credit_income_and_loss_are_zero_by_default() -> None:
    actions = dynamics.WeekActions(1.0, 0.0, 0.0, 0.0, 0.5, np.array([0.4, 0.3, 0.3]))
    assert actions.credit_income == 0.0
    assert actions.credit_default_loss == 0.0


def test_firm_value_matches_manual_formula(cfg: calibration.EnvConfig) -> None:
    state = dynamics.initial_state(cfg)
    expected = (
        state.cash
        + state.reserve
        + state.invested
        + state.ar_buckets.sum() * (1 - cfg.bad_debt_frac)
        - state.term_debt
        - state.credit_drawn
    )
    assert dynamics.firm_value(state, cfg) == pytest.approx(expected)


def test_essential_bias_affects_ops_health(cfg: calibration.EnvConfig) -> None:
    """Higher essential_bias should protect ops_health more for the same spend rate."""
    rng_low = np.random.default_rng(0)
    rng_high = np.random.default_rng(0)
    state = dynamics.initial_state(cfg)
    low_bias = dynamics.WeekActions(
        1.0, 0.0, 0.0, 0.0, 0.9, np.array([0.4, 0.3, 0.3]), essential_bias=0.0
    )
    high_bias = dynamics.WeekActions(
        1.0, 0.0, 0.0, 0.0, 0.9, np.array([0.4, 0.3, 0.3]), essential_bias=1.0
    )
    result_low = dynamics.step(state, low_bias, cfg, rng_low)
    result_high = dynamics.step(state, high_bias, cfg, rng_high)
    assert result_high.state.ops_health > result_low.state.ops_health


def test_ap_schedule_shifts_and_accrues_new_expenses(cfg: calibration.EnvConfig) -> None:
    """Deferring 100% of what's due should roll it into next week's due-now slot."""
    rng = np.random.default_rng(0)
    state = dynamics.initial_state(cfg)
    due_before = float(state.ap_due[0])
    actions = dynamics.WeekActions(1.0, 0.0, 1.0, 0.0, 0.0, np.array([1.0, 0.0, 0.0]), 0.0, 0.0)
    result = dynamics.step(state, actions, cfg, rng)
    # nothing paid; the full amount plus late penalty context should appear, deferred, in ap_due[0]
    assert result.flows["payables_paid"] == pytest.approx(0.0)
    assert result.state.ap_due[0] >= due_before - 1e-6
