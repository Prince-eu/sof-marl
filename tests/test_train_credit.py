"""Tests for the statistical helper functions in sof_marl.credit.train_credit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from sof_marl.credit.train_credit import bootstrap_ci, scale_pos_weight, term_bucket_default_rates


def test_scale_pos_weight() -> None:
    y = np.array([0, 0, 0, 0, 1, 1])
    assert scale_pos_weight(y) == pytest.approx(4 / 2)


def test_scale_pos_weight_balanced() -> None:
    y = np.array([0, 1, 0, 1])
    assert scale_pos_weight(y) == pytest.approx(1.0)


def test_bootstrap_ci_contains_point_estimate_for_perfect_separation() -> None:
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 500 + [1] * 500)
    y_score = np.concatenate([rng.uniform(0, 0.4, 500), rng.uniform(0.6, 1.0, 500)])
    point_estimate = roc_auc_score(y_true, y_score)
    lo, hi = bootstrap_ci(y_true, y_score, roc_auc_score, n_boot=200, seed=1)
    assert lo <= point_estimate <= hi
    assert lo < hi


def test_bootstrap_ci_is_deterministic_given_seed() -> None:
    y_true = np.array([0, 1] * 200)
    y_score = np.tile(np.linspace(0, 1, 2), 200)
    ci_a = bootstrap_ci(y_true, y_score, roc_auc_score, n_boot=100, seed=42)
    ci_b = bootstrap_ci(y_true, y_score, roc_auc_score, n_boot=100, seed=42)
    assert ci_a == ci_b


def test_term_bucket_default_rates_buckets_and_counts() -> None:
    df = pd.DataFrame(
        {
            "term_months": [6, 6, 30, 300, 300, 300],
            "default": [1, 0, 0, 0, 0, 1],
        }
    )
    buckets = term_bucket_default_rates(df, "default")
    total_n = sum(b["n"] for b in buckets)
    assert total_n == len(df)

    short_term = next(b for b in buckets if b["term_months_range"] == "(0, 12]")
    assert short_term["n"] == 2
    assert short_term["default_rate"] == pytest.approx(0.5)

    long_term = next(b for b in buckets if b["term_months_range"] == "(240, 400]")
    assert long_term["n"] == 3
    assert long_term["default_rate"] == pytest.approx(1 / 3)
