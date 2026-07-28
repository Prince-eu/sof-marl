"""Per-agent observation and action spaces (docs/ARCHITECTURE.md section 4).

Every agent observes a function-specific slice plus a small shared vector
``[cash, projected_cash_next_k, credit_headroom, week/T]`` (ARCHITECTURE.md section
2). Dollar-valued fields are normalized by dividing by the calibrated weekly revenue
mean (``cfg.revenue.weekly_mean``) so magnitudes are roughly O(1) for network
training; ``build_shared_observation`` in ``sme_treasury_env.py`` documents the
exact scaling applied at runtime. Action ranges match ``config/agents.yaml``.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from sof_marl.env.calibration import EnvConfig

AGENT_NAMES: tuple[str, ...] = ("liquidity", "credit_risk", "expenditure", "capital_allocation")

SHARED_OBS_DIM = 4  # [cash_norm, projected_cash_next_k_norm, credit_headroom_norm, week_frac]

# --- liquidity agent ---
# Agent-specific obs: [cash_norm, ar0_norm, ar1_norm, ar2_norm, *ap_due_norm,
# credit_headroom_norm, revenue_forecast_norm]
LIQUIDITY_ACTION_LOW = np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float32)
LIQUIDITY_ACTION_HIGH = np.array([2.0, 1.0, 1.0, 1.0], dtype=np.float32)


def liquidity_agent_obs_dim(cfg: EnvConfig) -> int:
    return 4 + cfg.ap_schedule_weeks + 2  # cash,ar(3) + ap_due(n) + headroom,forecast


def liquidity_observation_space(cfg: EnvConfig) -> spaces.Box:
    dim = liquidity_agent_obs_dim(cfg) + SHARED_OBS_DIM
    return spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def liquidity_action_space() -> spaces.Box:
    return spaces.Box(low=LIQUIDITY_ACTION_LOW, high=LIQUIDITY_ACTION_HIGH, dtype=np.float32)


# --- credit-risk agent ---
# Agent-specific obs: per pending slot [predicted_prob, exposure_amount_norm, valid_mask]
# (max_pending_credit_requests slots, zero-padded), then [cash_norm,
# credit_headroom_norm, outstanding_exposure_norm].
CREDIT_REQUEST_FEATURES = 3  # predicted_prob, exposure_amount_norm, valid_mask
CREDIT_RISK_ACTIONS = ("deny", "approve", "approve_with_premium")  # config/agents.yaml order


def credit_risk_agent_obs_dim(cfg: EnvConfig) -> int:
    return cfg.max_pending_credit_requests * CREDIT_REQUEST_FEATURES + 3


def credit_risk_observation_space(cfg: EnvConfig) -> spaces.Box:
    dim = credit_risk_agent_obs_dim(cfg) + SHARED_OBS_DIM
    return spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def credit_risk_action_space(cfg: EnvConfig) -> spaces.MultiDiscrete:
    return spaces.MultiDiscrete([len(CREDIT_RISK_ACTIONS)] * cfg.max_pending_credit_requests)


# --- expenditure agent ---
# Agent-specific obs: [discretionary_available_norm, cash_norm, ap_due_total_norm,
# season_sin, season_cos, ops_health]
EXPENDITURE_AGENT_OBS_DIM = 6
EXPENDITURE_ACTION_LOW = np.array([0.0, 0.0], dtype=np.float32)
EXPENDITURE_ACTION_HIGH = np.array([1.0, 1.0], dtype=np.float32)


def expenditure_observation_space() -> spaces.Box:
    dim = EXPENDITURE_AGENT_OBS_DIM + SHARED_OBS_DIM
    return spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def expenditure_action_space() -> spaces.Box:
    return spaces.Box(low=EXPENDITURE_ACTION_LOW, high=EXPENDITURE_ACTION_HIGH, dtype=np.float32)


# --- capital-allocation agent ---
# Agent-specific obs: [surplus_norm, term_debt_norm, term_debt_apr, reserve_norm,
# invested_norm, investment_return_weekly_mean]
CAPITAL_ALLOCATION_AGENT_OBS_DIM = 6
CAPITAL_ALLOCATION_CHOICES = ("debt_paydown", "reserve", "investment")  # config/agents.yaml order


def capital_allocation_observation_space() -> spaces.Box:
    dim = CAPITAL_ALLOCATION_AGENT_OBS_DIM + SHARED_OBS_DIM
    return spaces.Box(low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32)


def capital_allocation_action_space() -> spaces.Box:
    return spaces.Box(low=0.0, high=1.0, shape=(len(CAPITAL_ALLOCATION_CHOICES),), dtype=np.float32)


def observation_spaces(cfg: EnvConfig) -> dict[str, spaces.Space]:
    return {
        "liquidity": liquidity_observation_space(cfg),
        "credit_risk": credit_risk_observation_space(cfg),
        "expenditure": expenditure_observation_space(),
        "capital_allocation": capital_allocation_observation_space(),
    }


def action_spaces(cfg: EnvConfig) -> dict[str, spaces.Space]:
    return {
        "liquidity": liquidity_action_space(),
        "credit_risk": credit_risk_action_space(cfg),
        "expenditure": expenditure_action_space(),
        "capital_allocation": capital_allocation_action_space(),
    }
