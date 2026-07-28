"""Tests for sof_marl.credit.survival: censored duration/event construction.

Uses a small synthetic CSV in the raw FOIA schema (with the resolution-date columns)
rather than the real gitignored snapshot, so these run fast on a clean checkout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sof_marl.credit.survival import (
    RAW_USECOLS_SURVIVAL,
    _aft_dmatrix,
    build_survival_frame,
    survival_feature_columns,
)


def _row(status, approval, pif, chgoff, fy):
    return {
        "loanstatus": status,
        "approvalfy": fy,
        "grossapproval": 100000,
        "sbaguaranteedapproval": 50000,
        "terminmonths": 84,
        "initialinterestrate": 6.0,
        "fixedorvariableinterestind": "V",
        "jobssupported": 5,
        "naicscode": "722110",
        "borrstate": "CA",
        "businesstype": "CORPORATION",
        "businessage": "Existing, 5 or more years",
        "processingmethod": "7a General",
        "revolverstatus": "FALSE",
        "collateralind": "TRUE",
        "franchisecode": "",
        "approvaldate": approval,
        "asofdate": "3/31/2026",
        "paidinfulldate": pif,
        "chargeoffdate": chgoff,
    }


@pytest.fixture
def frame(tmp_path: Path) -> pd.DataFrame:
    rows = [
        # CHGOFF: event, duration approval->chargeoff (~30 months)
        _row("CHGOFF", "1/1/2016", "", "7/1/2018", 2016),
        # PIF: censored, duration approval->pif (~48 months)
        _row("PIF", "1/1/2016", "1/1/2020", "", 2016),
        # EXEMPT (still active): censored at asofdate (~99 months)
        _row("EXEMPT", "1/1/2018", "", "", 2018),
        # CANCLD and COMMIT: excluded entirely
        _row("CANCLD", "1/1/2016", "", "", 2016),
        _row("COMMIT", "1/1/2016", "", "", 2016),
        # CHGOFF with missing chargeoff date -> NaT duration -> dropped
        _row("CHGOFF", "1/1/2016", "", "", 2016),
    ]
    df = pd.DataFrame(rows, columns=RAW_USECOLS_SURVIVAL)
    path = tmp_path / "raw_survival.csv"
    df.to_csv(path, index=False)
    return build_survival_frame(path)


def test_excludes_cancelled_and_undisbursed_and_bad_durations(frame: pd.DataFrame) -> None:
    # 6 raw rows: CANCLD + COMMIT excluded (2), CHGOFF-with-missing-date dropped (1) -> 3 remain.
    assert len(frame) == 3


def test_event_flag_only_for_charge_off(frame: pd.DataFrame) -> None:
    assert set(frame["event"].unique()) <= {0, 1}
    assert int(frame["event"].sum()) == 1  # only the resolved CHGOFF row is an event


def test_durations_match_expected_endpoints(frame: pd.DataFrame) -> None:
    durs = sorted(frame["duration"].tolist())
    # CHGOFF ~30 months, PIF ~48 months, EXEMPT ~99 months (approx, /30.44)
    assert durs[0] == pytest.approx(30.0, abs=1.5)
    assert durs[1] == pytest.approx(48.0, abs=1.5)
    assert durs[2] == pytest.approx(99.0, abs=1.5)
    assert all(d > 0 for d in durs)


def test_features_present_and_no_target_leakage(frame: pd.DataFrame) -> None:
    cols = survival_feature_columns(frame)
    assert "term_months" in cols
    assert "duration" not in cols
    assert "event" not in cols
    # no resolution-date columns leaked into features
    for c in ("chargeoffdate", "paidinfulldate", "asofdate", "approvaldate"):
        assert c not in frame.columns


def test_aft_dmatrix_interval_labels() -> None:
    x = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.0, 1.0, 0.0]})
    duration = np.array([10.0, 20.0, 30.0])
    event = np.array([1, 0, 1])
    dmat = _aft_dmatrix(x, duration, event)
    lower = dmat.get_float_info("label_lower_bound")
    upper = dmat.get_float_info("label_upper_bound")
    np.testing.assert_allclose(lower, [10.0, 20.0, 30.0])
    # events: upper == lower; censored (index 1): upper == +inf
    assert upper[0] == pytest.approx(10.0)
    assert np.isinf(upper[1])
    assert upper[2] == pytest.approx(30.0)
