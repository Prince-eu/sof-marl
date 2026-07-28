"""Tests for evaluation.metrics: the metric suite and coordination-lift test."""

from __future__ import annotations

import numpy as np
import pytest

from sof_marl.env.calibration import load_env_config
from sof_marl.evaluation import metrics
from sof_marl.evaluation.rollout import EpisodeRecord


def _make_record(
    firm_values: list[float],
    cash_values: list[float] | None = None,
    solvency_breaches: list[bool] | None = None,
    financing_costs: list[float] | None = None,
    credit_decisions: list[dict[str, object]] | None = None,
    credit_resolutions: list[dict[str, object]] | None = None,
) -> EpisodeRecord:
    n = len(firm_values) - 1
    return EpisodeRecord(
        firm_values=firm_values,
        cash_values=cash_values if cash_values is not None else list(firm_values),
        invested_values=[0.0] * len(firm_values),
        reserve_values=[0.0] * len(firm_values),
        term_debt_values=[0.0] * len(firm_values),
        solvency_breaches=solvency_breaches if solvency_breaches is not None else [False] * n,
        shortfall_severities=[0.0] * n,
        financing_costs=financing_costs if financing_costs is not None else [0.0] * n,
        credit_decisions=credit_decisions or [],
        credit_resolutions=credit_resolutions or [],
    )


def test_episode_return_and_volatility() -> None:
    record = _make_record([100.0, 110.0, 120.0, 130.0])
    assert metrics.episode_return(record) == pytest.approx(0.3)
    diffs = np.array([10.0, 10.0, 10.0])
    expected_vol = float(np.std(diffs)) / 100.0
    assert metrics.episode_volatility(record) == pytest.approx(expected_vol)


def test_episode_solvency_breach_rate() -> None:
    record = _make_record([100.0] * 5, solvency_breaches=[True, False, True, False])
    assert metrics.episode_solvency_breach_rate(record) == pytest.approx(0.5)


def test_combined_objective_penalizes_volatility_and_breaches() -> None:
    stable = _make_record([100.0, 105.0, 110.0, 115.0])
    volatile = _make_record([100.0, 130.0, 90.0, 115.0])
    breached = _make_record([100.0, 105.0, 110.0, 115.0], solvency_breaches=[True, True, False])

    j_stable = metrics.combined_objective(stable, kappa=1.0, lam=3.0)
    j_volatile = metrics.combined_objective(volatile, kappa=1.0, lam=3.0)
    j_breached = metrics.combined_objective(breached, kappa=1.0, lam=3.0)

    assert j_stable > j_volatile
    assert j_stable > j_breached


def test_credit_metrics_precision_recall_and_loss_rate() -> None:
    decisions = [
        {"decision": 1, "true_label": 0},  # approved, good -> TP
        {"decision": 2, "true_label": 0},  # approved w/ premium, good -> TP
        {"decision": 1, "true_label": 1},  # approved, bad -> FP
        {"decision": 0, "true_label": 0},  # denied, good -> FN
        {"decision": 0, "true_label": 1},  # denied, bad -> TN (not in precision/recall)
    ]
    resolutions = [
        {"true_label": 0, "amount": 1000.0, "income": 25.0, "loss": 0.0},
        {"true_label": 1, "amount": 500.0, "income": 0.0, "loss": 300.0},
    ]
    record = _make_record(
        [100.0, 100.0], credit_decisions=decisions, credit_resolutions=resolutions
    )
    result = metrics.credit_metrics([record])

    assert result["approval_precision"] == pytest.approx(2 / 3)
    assert result["approval_recall"] == pytest.approx(2 / 3)
    assert result["realized_default_rate"] == pytest.approx(0.5)
    assert result["realized_loss_rate_dollar"] == pytest.approx(300.0 / 1500.0)
    assert result["portfolio_yield_net"] == pytest.approx((25.0 - 300.0) / 1500.0)


def test_summarize_policy_aggregates_mean_and_std_across_seeds() -> None:
    cfg = load_env_config()
    episodes_by_seed = {
        0: [_make_record([100.0, 110.0])],
        1: [_make_record([100.0, 120.0])],
    }
    summary = metrics.summarize_policy(episodes_by_seed, cfg, kappa=1.0, lam=3.0)
    assert summary["n_seeds"] == 2
    expected_mean = (0.1 + 0.2) / 2
    assert summary["aggregated"]["return_r"]["mean"] == pytest.approx(expected_mean)
    assert summary["aggregated"]["return_r"]["std"] > 0.0


def test_summarize_policy_handles_rule_based_single_list() -> None:
    cfg = load_env_config()
    episodes = [_make_record([100.0, 110.0]), _make_record([100.0, 90.0])]
    summary = metrics.summarize_policy(episodes, cfg, kappa=1.0, lam=3.0)
    assert summary["n_seeds"] == 1
    assert summary["aggregated"]["return_r"]["std"] == 0.0


def test_coordination_lift_detects_consistent_advantage() -> None:
    better = {0: [_make_record([100.0, 130.0]) for _ in range(20)]}
    worse = {0: [_make_record([100.0, 105.0]) for _ in range(20)]}
    result = metrics.coordination_lift(better, worse, kappa=1.0, lam=3.0, alpha=0.05)
    assert result["delta_combined_objective_mean"] > 0
    assert result["p_value"] < 0.05
    assert result["significant_at_alpha"] is True


def test_coordination_lift_no_difference_is_not_significant() -> None:
    same_a = {0: [_make_record([100.0, 110.0]) for _ in range(10)]}
    same_b = {0: [_make_record([100.0, 110.0]) for _ in range(10)]}
    result = metrics.coordination_lift(same_a, same_b, kappa=1.0, lam=3.0, alpha=0.05)
    assert result["delta_combined_objective_mean"] == pytest.approx(0.0)
    assert result["significant_at_alpha"] is False
