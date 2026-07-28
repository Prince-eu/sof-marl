"""Loads config/env.yaml and exposes typed accessors, plus the credit exposure pool.

docs/ARCHITECTURE.md section 7: every calibration constant is a named, sourced
config value; there are no magic numbers inside src/sof_marl/env or
src/sof_marl/agents -- everything traces back to a field read here.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray

EXPOSURE_POOL_CACHE_NAME = "credit_exposure_pool.csv"


@dataclass(frozen=True)
class InitialBalanceSheet:
    cash: float
    reserve: float
    invested: float
    accounts_receivable: float
    accounts_payable: float
    term_debt: float
    credit_drawn: float


@dataclass(frozen=True)
class RevenueConfig:
    annual_mean: float
    weekly_sigma_frac: float
    seasonal_amplitude: float

    @property
    def weekly_mean(self) -> float:
        return self.annual_mean / 52.0


@dataclass(frozen=True)
class ExpensesConfig:
    fixed_weekly_frac: float
    discretionary_weekly_frac: float


@dataclass(frozen=True)
class CreditLineConfig:
    limit: float
    apr: float


@dataclass(frozen=True)
class InvestmentReturnConfig:
    weekly_mean: float
    weekly_sigma: float


@dataclass(frozen=True)
class EnvConfig:
    horizon_weeks: int
    initial: InitialBalanceSheet
    ar_aging_init: tuple[float, float, float]
    revenue: RevenueConfig
    expenses: ExpensesConfig
    credit_line: CreditLineConfig
    term_debt_apr: float
    late_payment_penalty_apr: float
    buffer_days_target: float
    collection_curve: tuple[float, float, float]
    bad_debt_frac: float
    credit_exposure_pool_size: int
    exposure_resolution_weeks: int
    early_pay_discount_frac: float
    investment_return: InvestmentReturnConfig
    ap_schedule_weeks: int
    credit_exposure_scale_frac: float
    loss_given_default_frac: float
    max_pending_credit_requests: int
    ops_health_target_spend_rate: float
    ops_health_adjustment_rate: float
    processed_dir: Path


def load_env_config(path: str | Path = "config/env.yaml") -> EnvConfig:
    raw: dict[str, Any] = yaml.safe_load(Path(path).read_text())
    return EnvConfig(
        horizon_weeks=raw["horizon_weeks"],
        initial=InitialBalanceSheet(**raw["initial"]),
        ar_aging_init=tuple(raw["ar_aging_init"]),
        revenue=RevenueConfig(**raw["revenue"]),
        expenses=ExpensesConfig(**raw["expenses"]),
        credit_line=CreditLineConfig(**raw["credit_line"]),
        term_debt_apr=raw["term_debt_apr"],
        late_payment_penalty_apr=raw["late_payment_penalty_apr"],
        buffer_days_target=raw["buffer_days_target"],
        collection_curve=tuple(raw["collection_curve"]),
        bad_debt_frac=raw["bad_debt_frac"],
        credit_exposure_pool_size=raw["credit_exposure_pool_size"],
        exposure_resolution_weeks=raw["exposure_resolution_weeks"],
        early_pay_discount_frac=raw["early_pay_discount_frac"],
        investment_return=InvestmentReturnConfig(**raw["investment_return"]),
        ap_schedule_weeks=raw["ap_schedule_weeks"],
        credit_exposure_scale_frac=raw["credit_exposure_scale_frac"],
        loss_given_default_frac=raw["loss_given_default_frac"],
        max_pending_credit_requests=raw["max_pending_credit_requests"],
        ops_health_target_spend_rate=raw["ops_health_target_spend_rate"],
        ops_health_adjustment_rate=raw["ops_health_adjustment_rate"],
        processed_dir=Path(raw["sba"]["processed_dir"]),
    )


def weekly_revenue_mu(week: int, cfg: EnvConfig) -> float:
    """Seasonal expected weekly revenue for episode week `week` (ARCHITECTURE.md section 3)."""
    season_index = week % 52
    seasonal_factor = 1.0 + cfg.revenue.seasonal_amplitude * float(
        np.sin(2 * np.pi * season_index / 52)
    )
    return cfg.revenue.weekly_mean * seasonal_factor


def buffer_target_dollars(week: int, cfg: EnvConfig, buffer_frac_multiplier: float) -> float:
    """Cash buffer target in dollars: buffer_days_target days of weekly operating outflow,
    scaled by the liquidity agent's chosen target_buffer_frac (1.0 = the calibrated anchor).
    """
    mu = weekly_revenue_mu(week, cfg)
    weekly_outflow = (cfg.expenses.fixed_weekly_frac + cfg.expenses.discretionary_weekly_frac) * mu
    daily_outflow = weekly_outflow / 7.0
    return buffer_frac_multiplier * cfg.buffer_days_target * daily_outflow


def build_or_load_credit_exposure_pool(cfg: EnvConfig, force_rebuild: bool = False) -> pd.DataFrame:
    """Score the Phase-A held-out test split with the calibrated model; cache to disk.

    Returns columns [gross_approval, default, predicted_prob], one row per held-out
    test loan (ARCHITECTURE.md section 5: "sample a pool of exposures from the
    held-out real records; each carries its true label and the model's predicted
    default probability"). Episodes sample credit_exposure_pool_size rows from this
    table at reset time (see sample_episode_exposures).
    """
    cache_path = cfg.processed_dir / EXPOSURE_POOL_CACHE_NAME
    if cache_path.exists() and not force_rebuild:
        return pd.read_csv(cache_path)

    manifest = json.loads((cfg.processed_dir / "manifest.json").read_text())
    feature_columns: list[str] = manifest["feature_columns"]
    test = pd.read_csv(cfg.processed_dir / "test.csv")
    with (cfg.processed_dir / "credit_model_calibrated.pkl").open("rb") as f:
        model = pickle.load(f)
    predicted_prob = model.predict_proba(test[feature_columns])[:, 1]

    pool = pd.DataFrame(
        {
            "gross_approval": test["gross_approval"].to_numpy(),
            "default": test["default"].to_numpy(),
            "predicted_prob": predicted_prob,
        }
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pool.to_csv(cache_path, index=False)
    return pool


def sample_episode_exposures(
    pool: pd.DataFrame, cfg: EnvConfig, rng: np.random.Generator
) -> pd.DataFrame:
    """Sample credit_exposure_pool_size exposures for one episode with random arrival weeks.

    Real SBA gross_approval amounts are scaled by credit_exposure_scale_frac to a
    trade-credit-sized exposure relative to this firm's revenue (ARCHITECTURE.md
    section 5; config/env.yaml).
    """
    n = min(cfg.credit_exposure_pool_size, len(pool))
    idx: NDArray[np.int_] = rng.choice(len(pool), size=n, replace=False)
    sampled = pool.iloc[idx].reset_index(drop=True).copy()
    sampled["exposure_amount"] = sampled["gross_approval"] * cfg.credit_exposure_scale_frac
    sampled["arrival_week"] = rng.integers(0, cfg.horizon_weeks, size=n)
    return sampled
