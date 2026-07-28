"""Per-agent reward terms and the shared coordination reward (ARCHITECTURE.md 4).

Every reward is computed from named, already-known quantities produced by
``dynamics.step`` (and the credit-exposure bookkeeping in ``sme_treasury_env.py``);
no reward function performs its own accounting. Dollar-valued terms are normalized
by the calibrated weekly revenue mean (``cfg.revenue.weekly_mean``) so reward
magnitudes are comparable across the four agents and stable for PPO. Weights are
loaded from ``config/agents.yaml`` -- no magic numbers here (ARCHITECTURE.md
section 7).

Each agent's total reward is ``R_i + shared_reward_weight * R_shared``
(ARCHITECTURE.md 4.5); ``shared_reward_weight`` is 0 for the IPPO baseline and > 0
for MAPPO (config/train.yaml ``mappo.shared_reward_weight``), so it is a parameter
of the environment/training setup, not of this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sof_marl.env.calibration import EnvConfig, weekly_revenue_mu
from sof_marl.env.dynamics import FirmState, StepResult, firm_value


@dataclass(frozen=True)
class LiquidityRewardWeights:
    w_shortfall: float
    w_financing_cost: float
    w_idle_cash: float
    w_late_payment: float


@dataclass(frozen=True)
class CreditRiskRewardWeights:
    margin_bps: float
    w_loss_given_default: float
    w_concentration: float
    concentration_budget_frac: float
    credit_reward_scale: float


@dataclass(frozen=True)
class ExpenditureRewardWeights:
    value_scale: float
    diminishing_returns_k: float
    w_over_deferral: float


@dataclass(frozen=True)
class CapitalAllocationRewardWeights:
    w_return: float
    w_interest_saved: float
    w_liquidity_risk: float


@dataclass(frozen=True)
class SharedRewardWeights:
    alpha_firm_value: float
    beta_solvency_breach: float


@dataclass(frozen=True)
class AgentsConfig:
    liquidity_rewards: LiquidityRewardWeights
    credit_risk_premium_bps: float
    credit_risk_rewards: CreditRiskRewardWeights
    expenditure_rewards: ExpenditureRewardWeights
    capital_allocation_rewards: CapitalAllocationRewardWeights
    shared: SharedRewardWeights


def load_agents_config(path: str | Path = "config/agents.yaml") -> AgentsConfig:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    return AgentsConfig(
        liquidity_rewards=LiquidityRewardWeights(**raw["liquidity"]["rewards"]),
        credit_risk_premium_bps=raw["credit_risk"]["premium_bps"],
        credit_risk_rewards=CreditRiskRewardWeights(**raw["credit_risk"]["rewards"]),
        expenditure_rewards=ExpenditureRewardWeights(**raw["expenditure"]["rewards"]),
        capital_allocation_rewards=CapitalAllocationRewardWeights(
            **raw["capital_allocation"]["rewards"]
        ),
        shared=SharedRewardWeights(**raw["shared"]),
    )


def liquidity_reward(result: StepResult, cfg: EnvConfig, weights: LiquidityRewardWeights) -> float:
    """R_liq = -w1*shortfall - w2*financing_cost - w3*idle_cash - w4*late_payment."""
    scale = cfg.revenue.weekly_mean
    financing_cost = result.flows["credit_line_interest"] + result.flows["term_debt_interest"]
    idle_cash = max(0.0, result.state.cash - 2.0 * result.buffer_target)
    return -(
        weights.w_shortfall * (result.shortfall_severity / scale)
        + weights.w_financing_cost * (financing_cost / scale)
        + weights.w_idle_cash * (idle_cash / scale)
        + weights.w_late_payment * (result.flows["late_payment_penalty"] / scale)
    )


def credit_risk_reward(
    cfg: EnvConfig,
    weights: CreditRiskRewardWeights,
    credit_decision_ev: float,
    outstanding_exposure_total: float,
) -> float:
    """R_cred = credit_reward_scale * (expected value of this week's decisions) - concentration.

    ``credit_decision_ev`` is the *immediate* expected value of this week's
    approve/deny decisions, computed by the environment from the real model's
    predicted default probability at decision time (see sme_treasury_env.py):
    for each approved exposure, margin*(1-p) - w_lgd*LGD*p, in dollars; denials
    contribute 0. This dense, immediate signal replaces the original reward, which
    only resolved 12 weeks later and was too small to learn against R_shared
    (docs/ARCHITECTURE.md section 4.2, reports/technical_report.md 5.3/7). Realized
    cash flows still hit firm value via the environment dynamics, unchanged.

    The concentration penalty is a threshold: it fires only on outstanding exposure
    above ``concentration_budget_frac * credit_line.limit``, so ordinary prudent
    lending is free and only genuine over-concentration is penalized.
    """
    scale = cfg.revenue.weekly_mean
    margin_signal = weights.credit_reward_scale * (credit_decision_ev / scale)
    budget = weights.concentration_budget_frac * cfg.credit_line.limit
    over_budget = max(0.0, outstanding_exposure_total - budget)
    concentration = weights.w_concentration * (over_budget / scale)
    return margin_signal - concentration


def expenditure_reward(
    result: StepResult,
    old_state: FirmState,
    cfg: EnvConfig,
    weights: ExpenditureRewardWeights,
) -> float:
    """R_exp = +value_of_spend(diminishing returns) - w*operational_penalty_for_over_deferral."""
    scale = cfg.revenue.weekly_mean
    spend_frac_of_scale = max(0.0, result.flows["discretionary_spent"] / scale)
    value_of_spend = weights.value_scale * spend_frac_of_scale**weights.diminishing_returns_k
    over_deferral_penalty = weights.w_over_deferral * (1.0 - result.state.ops_health)
    return value_of_spend - over_deferral_penalty


def capital_allocation_reward(
    result: StepResult,
    cfg: EnvConfig,
    weights: CapitalAllocationRewardWeights,
) -> float:
    """R_cap = +risk_adjusted_return + interest_saved_on_paydown - liquidity_risk_penalty.

    liquidity_risk_penalty is a forward-looking proxy: it fires when a naive,
    deterministic projection of next week's cash (this week's ending cash plus
    expected revenue minus expected fixed and discretionary expenses, ignoring
    collection lag) would be negative -- i.e. surplus was allocated away from cash
    while a shortfall was already foreseeable. This is a reward-shaping heuristic
    only; it does not affect the accounted cash balance.
    """
    scale = cfg.revenue.weekly_mean
    risk_adjusted_return = weights.w_return * (result.investment_return_dollars / scale)
    interest_saved = weights.w_interest_saved * (
        result.debt_paydown * cfg.term_debt_apr / 52.0 / scale
    )
    next_mu = weekly_revenue_mu(result.state.week, cfg)
    projected_next_cash = (
        result.state.cash
        + next_mu
        - (cfg.expenses.fixed_weekly_frac + cfg.expenses.discretionary_weekly_frac) * next_mu
    )
    over_allocated = projected_next_cash < 0.0 and result.invested_allocated > 0.0
    liquidity_risk_penalty = (
        weights.w_liquidity_risk * (result.invested_allocated / scale) if over_allocated else 0.0
    )
    return risk_adjusted_return + interest_saved - liquidity_risk_penalty


def shared_reward(
    result: StepResult, old_state: FirmState, cfg: EnvConfig, weights: SharedRewardWeights
) -> float:
    """R_shared = alpha*firm_value_change - beta*solvency_breach_indicator (ARCHITECTURE.md 4.5)."""
    scale = cfg.revenue.weekly_mean
    firm_value_change = (firm_value(result.state, cfg) - firm_value(old_state, cfg)) / scale
    breach_indicator = 1.0 if result.solvency_breach else 0.0
    return (
        weights.alpha_firm_value * firm_value_change
        - weights.beta_solvency_breach * breach_indicator
    )


def per_agent_rewards(
    result: StepResult,
    old_state: FirmState,
    cfg: EnvConfig,
    agents_cfg: AgentsConfig,
    outstanding_exposure_total: float,
    credit_decision_ev: float,
    shared_reward_weight: float,
) -> dict[str, float]:
    """R_i + shared_reward_weight * R_shared for each of the four agents, plus 'shared'."""
    r_shared = shared_reward(result, old_state, cfg, agents_cfg.shared)
    r_liq = liquidity_reward(result, cfg, agents_cfg.liquidity_rewards)
    r_cred = credit_risk_reward(
        cfg, agents_cfg.credit_risk_rewards, credit_decision_ev, outstanding_exposure_total
    )
    r_exp = expenditure_reward(result, old_state, cfg, agents_cfg.expenditure_rewards)
    r_cap = capital_allocation_reward(result, cfg, agents_cfg.capital_allocation_rewards)
    return {
        "liquidity": r_liq + shared_reward_weight * r_shared,
        "credit_risk": r_cred + shared_reward_weight * r_shared,
        "expenditure": r_exp + shared_reward_weight * r_shared,
        "capital_allocation": r_cap + shared_reward_weight * r_shared,
        "shared": r_shared,
    }
