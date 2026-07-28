"""Weekly cash-flow, receivables/payables, and financing dynamics.

Implements docs/ARCHITECTURE.md section 3 in order: revenue realizes; receivables
age and collect; agent actions apply (draws/repayments, payables timing,
discretionary spend, capital allocation); payables settle; financing costs accrue;
credit exposures resolve (handled by the caller -- see sme_treasury_env.py -- and
passed in as already-known dollar amounts, keeping this module free of the credit
exposure object model); state advances; solvency is checked.

``step`` is a pure function: given a state, a week's actions, the config, and an RNG,
it returns the next state plus every named cash inflow/outflow for that week. The
cash-conservation invariant this module must satisfy -- enforced in
tests/test_dynamics.py -- is:

    new_state.cash == state.cash + sum(inflows) - sum(outflows)

with every term in ``inflows``/``outflows`` named and accounted for; no term may
inject or destroy value.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from sof_marl.env.calibration import EnvConfig, buffer_target_dollars, weekly_revenue_mu

INFLOW_KEYS = (
    "receivables_accelerated",
    "receivables_collected",
    "credit_line_draw",
    "credit_income",
)
OUTFLOW_KEYS = (
    "credit_line_repay",
    "payables_paid",
    "late_payment_penalty",
    "discretionary_spent",
    "debt_paydown",
    "reserve_allocated",
    "invested_allocated",
    "credit_line_interest",
    "term_debt_interest",
    "credit_default_loss",
)


@dataclass
class FirmState:
    """The firm's full financial state (ARCHITECTURE.md section 2)."""

    week: int
    cash: float
    ar_buckets: NDArray[np.float64]  # shape (3,): [0-30, 31-60, 61-90 day aging stages]
    ap_due: NDArray[np.float64]  # shape (ap_schedule_weeks,): [due this week, due next week, ...]
    credit_drawn: float
    term_debt: float
    reserve: float
    invested: float
    buffer_frac_multiplier: float  # liquidity agent's last-chosen target_buffer_frac
    discretionary_carryover: float
    ops_health: float  # latent operations-health term in [0, 1] (ARCHITECTURE.md section 4.3)


@dataclass
class WeekActions:
    """This week's already-interpreted agent decisions, consumed by ``step``."""

    target_buffer_frac: float  # liquidity, [0, 2]
    credit_draw_frac: float  # liquidity, [-1, 1]; negative = repay
    defer_payables_frac: float  # liquidity, [0, 1]
    accelerate_receivables_frac: float  # liquidity, [0, 1]
    spend_now_frac: float  # expenditure, [0, 1]
    capital_allocation: NDArray[np.float64]  # capital allocation simplex over 3 choices
    credit_income: float = 0.0  # this week's realized margin/premium income from exposures
    credit_default_loss: float = 0.0  # this week's realized default loss from exposures
    essential_bias: float = 0.5  # expenditure, [0, 1]; share of spend treated as essential-adjacent


def initial_state(cfg: EnvConfig) -> FirmState:
    ar_buckets = np.array(cfg.ar_aging_init, dtype=np.float64) * cfg.initial.accounts_receivable
    ap_due = np.zeros(cfg.ap_schedule_weeks, dtype=np.float64)
    ap_due[0] = cfg.initial.accounts_payable
    return FirmState(
        week=0,
        cash=cfg.initial.cash,
        ar_buckets=ar_buckets,
        ap_due=ap_due,
        credit_drawn=cfg.initial.credit_drawn,
        term_debt=cfg.initial.term_debt,
        reserve=cfg.initial.reserve,
        invested=cfg.initial.invested,
        buffer_frac_multiplier=1.0,
        discretionary_carryover=0.0,
        ops_health=1.0,
    )


def collectible_ar(state: FirmState, cfg: EnvConfig) -> float:
    """Expected collectible AR, net of the structural bad-debt drag (ARCHITECTURE.md 4.5)."""
    return float(state.ar_buckets.sum()) * (1.0 - cfg.bad_debt_frac)


def firm_value(state: FirmState, cfg: EnvConfig) -> float:
    """V = cash + reserve + invested + collectible_AR - term_debt - credit_drawn."""
    return (
        state.cash
        + state.reserve
        + state.invested
        + collectible_ar(state, cfg)
        - state.term_debt
        - state.credit_drawn
    )


@dataclass
class StepResult:
    state: FirmState
    flows: dict[str, float]  # every named inflow/outflow this week (see INFLOW_KEYS/OUTFLOW_KEYS)
    shortfall_severity: float  # max(0, -cash) after this week's flows
    solvency_breach: bool  # true cash shortfall even after remaining credit headroom
    revenue_realized: float
    investment_return_dollars: float  # this week's return on the pre-existing invested balance
    debt_paydown: float
    invested_allocated: float
    buffer_target: float  # dollar buffer target implied by this week's target_buffer_frac
    financing_cost_total: float = field(init=False)

    def __post_init__(self) -> None:
        self.financing_cost_total = (
            self.flows["credit_line_interest"]
            + self.flows["term_debt_interest"]
            + self.flows["late_payment_penalty"]
        )


def step(
    state: FirmState, actions: WeekActions, cfg: EnvConfig, rng: np.random.Generator
) -> StepResult:
    week = state.week

    # 1. Revenue realizes (ARCHITECTURE.md 3.1).
    mu = weekly_revenue_mu(week, cfg)
    sigma = cfg.revenue.weekly_sigma_frac * mu
    revenue_t = max(0.0, float(rng.normal(mu, sigma)))

    # 2. Receivables age and collect (ARCHITECTURE.md 3.2). Acceleration (liquidity
    # action) pulls forward a fraction of every bucket at a discount before the
    # normal collection/aging waterfall runs on what remains.
    accelerated = actions.accelerate_receivables_frac * state.ar_buckets
    ar_after_accel = state.ar_buckets - accelerated
    receivables_accelerated = float(accelerated.sum()) * (1.0 - cfg.early_pay_discount_frac)

    collection_curve = np.array(cfg.collection_curve, dtype=np.float64)
    collected = collection_curve * ar_after_accel
    receivables_collected = float(collected.sum())
    remaining = ar_after_accel - collected
    remaining_after_writeoff = remaining * (1.0 - cfg.bad_debt_frac)
    new_ar_buckets = np.array(
        [
            revenue_t,
            remaining_after_writeoff[0],
            remaining_after_writeoff[1] + remaining_after_writeoff[2],
        ],
        dtype=np.float64,
    )

    # 3. Agent actions apply.
    # 3a. Liquidity: credit-line draw/repay.
    headroom = cfg.credit_line.limit - state.credit_drawn
    if actions.credit_draw_frac >= 0:
        credit_line_draw = actions.credit_draw_frac * headroom
        credit_line_repay = 0.0
    else:
        credit_line_draw = 0.0
        credit_line_repay = min(-actions.credit_draw_frac * state.credit_drawn, state.credit_drawn)
    new_credit_drawn = state.credit_drawn + credit_line_draw - credit_line_repay

    # 3b. Liquidity: accounts payable settlement.
    due_now = float(state.ap_due[0])
    deferred = actions.defer_payables_frac * due_now
    payables_paid = due_now - deferred
    late_payment_penalty = deferred * cfg.late_payment_penalty_apr / 52.0
    fixed_expense_new = cfg.expenses.fixed_weekly_frac * mu
    new_ap_due = np.concatenate([state.ap_due[1:], np.zeros(1)])
    new_ap_due[0] += deferred
    new_ap_due[-1] += fixed_expense_new

    # 3c. Expenditure: discretionary spend.
    discretionary_available = (
        cfg.expenses.discretionary_weekly_frac * mu + state.discretionary_carryover
    )
    discretionary_spent = actions.spend_now_frac * discretionary_available
    new_discretionary_carryover = discretionary_available - discretionary_spent
    # essential-adjacent spend protects operations more per dollar than purely
    # deferrable spend: at essential_bias=1.0 spend counts fully toward ops health,
    # at essential_bias=0.0 only half as much (ARCHITECTURE.md section 4.3).
    effective_health_spend_frac = actions.spend_now_frac * (0.5 + 0.5 * actions.essential_bias)
    new_ops_health = float(
        np.clip(
            state.ops_health
            + cfg.ops_health_adjustment_rate
            * (effective_health_spend_frac - cfg.ops_health_target_spend_rate),
            0.0,
            1.0,
        )
    )

    # 3d. Capital allocation: surplus above the (agent-chosen) buffer target.
    buffer_target = buffer_target_dollars(week, cfg, actions.target_buffer_frac)
    surplus = max(0.0, state.cash - buffer_target)
    alloc_sum = float(actions.capital_allocation.sum())
    alloc = (
        actions.capital_allocation / alloc_sum if alloc_sum > 1e-8 else np.array([1.0, 0.0, 0.0])
    )
    debt_paydown = min(float(alloc[0]) * surplus, state.term_debt)
    reserve_allocated = float(alloc[1]) * surplus
    invested_allocated = float(alloc[2]) * surplus
    new_term_debt = state.term_debt - debt_paydown

    # 4. Financing costs accrue on post-action balances.
    credit_line_interest = new_credit_drawn * cfg.credit_line.apr / 52.0
    term_debt_interest = new_term_debt * cfg.term_debt_apr / 52.0

    # 5. Credit exposures resolve: income/loss are supplied by the caller (see
    # sme_treasury_env.py), which owns the exposure pool/queue object model.
    credit_income = actions.credit_income
    credit_default_loss = actions.credit_default_loss

    # 6. Investment return realizes on the existing invested balance (stochastic).
    investment_return_rate = float(
        rng.normal(cfg.investment_return.weekly_mean, cfg.investment_return.weekly_sigma)
    )
    investment_return_dollars = state.invested * investment_return_rate
    new_invested = max(0.0, state.invested + invested_allocated + investment_return_dollars)
    new_reserve = state.reserve + reserve_allocated

    flows = {
        "receivables_accelerated": receivables_accelerated,
        "receivables_collected": receivables_collected,
        "credit_line_draw": credit_line_draw,
        "credit_income": credit_income,
        "credit_line_repay": credit_line_repay,
        "payables_paid": payables_paid,
        "late_payment_penalty": late_payment_penalty,
        "discretionary_spent": discretionary_spent,
        "debt_paydown": debt_paydown,
        "reserve_allocated": reserve_allocated,
        "invested_allocated": invested_allocated,
        "credit_line_interest": credit_line_interest,
        "term_debt_interest": term_debt_interest,
        "credit_default_loss": credit_default_loss,
    }
    inflow_total = sum(flows[k] for k in INFLOW_KEYS)
    outflow_total = sum(flows[k] for k in OUTFLOW_KEYS)
    new_cash = state.cash + inflow_total - outflow_total

    new_state = FirmState(
        week=week + 1,
        cash=new_cash,
        ar_buckets=new_ar_buckets,
        ap_due=new_ap_due,
        credit_drawn=new_credit_drawn,
        term_debt=new_term_debt,
        reserve=new_reserve,
        invested=new_invested,
        buffer_frac_multiplier=actions.target_buffer_frac,
        discretionary_carryover=new_discretionary_carryover,
        ops_health=new_ops_health,
    )

    remaining_headroom = cfg.credit_line.limit - new_credit_drawn
    shortfall_severity = max(0.0, -new_cash)
    solvency_breach = (new_cash + remaining_headroom) < 0.0

    return StepResult(
        state=new_state,
        flows=flows,
        shortfall_severity=shortfall_severity,
        solvency_breach=solvency_breach,
        revenue_realized=revenue_t,
        investment_return_dollars=investment_return_dollars,
        debt_paydown=debt_paydown,
        invested_allocated=invested_allocated,
        buffer_target=buffer_target,
    )
