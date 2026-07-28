"""Tests for sof_marl.credit.build_dataset: label mapping, leakage, and the split.

Uses a small synthetic CSV in the raw FOIA schema rather than the real (gitignored,
223MB) snapshot, so these tests are fast and run on a clean checkout without
requiring data/download_sba.py to have been run first.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sof_marl.credit.build_dataset import (
    LABEL_COLUMN,
    RAW_USECOLS,
    build_features,
    feature_columns,
    load_raw,
    split_summary,
    time_split,
)

# Fields present in the real FOIA file that are known only at or after loan
# resolution, or that identify the borrower/lender. None of these may ever appear
# in the built feature table.
LEAKAGE_AND_IDENTIFYING_COLUMNS = [
    "paidinfulldate",
    "chargeoffdate",
    "grosschargeoffamount",
    "soldsecmrktind",
    "borrname",
    "borrstreet",
    "borrcity",
    "borrzip",
    "bankname",
    "bankfdicnumber",
    "bankncuanumber",
    "bankstreet",
    "bankcity",
    "bankstate",
    "bankzip",
    "locationid",
]

RAW_ROWS = [
    # loanstatus, approvalfy, grossapproval, sbaguaranteedapproval, terminmonths,
    # initialinterestrate, fixedorvariableinterestind, jobssupported, naicscode,
    # borrstate, businesstype, businessage, processingmethod, revolverstatus,
    # collateralind, franchisecode
    (
        "PIF",
        2012,
        100000,
        50000,
        60,
        6.0,
        "F",
        5,
        "722110",
        "CA",
        "CORPORATION",
        "Existing, 5 or more years",
        "7a General",
        "FALSE",
        "TRUE",
        "",
    ),
    (
        "CHGOFF",
        2013,
        50000,
        25000,
        84,
        7.5,
        "V",
        3,
        "445110",
        "TX",
        "INDIVIDUAL",
        "New, Less than 1 Year old",
        "SBA Express Program",
        "TRUE",
        "FALSE",
        "",
    ),
    (
        "CANCLD",
        2014,
        20000,
        10000,
        36,
        6.5,
        "V",
        2,
        "541990",
        "NY",
        "CORPORATION",
        "Existing, 5 or more years",
        "7a General",
        "FALSE",
        "TRUE",
        "",
    ),
    (
        "EXEMPT",
        2014,
        30000,
        15000,
        36,
        6.5,
        "V",
        2,
        "541990",
        "NY",
        "CORPORATION",
        "Existing, 5 or more years",
        "7a General",
        "FALSE",
        "TRUE",
        "",
    ),
    (
        "COMMIT",
        2015,
        40000,
        20000,
        36,
        6.5,
        "V",
        2,
        "541990",
        "NY",
        "CORPORATION",
        "Existing, 5 or more years",
        "7a General",
        "FALSE",
        "TRUE",
        "",
    ),
    (
        "PIF",
        2016,
        200000,
        150000,
        120,
        5.5,
        "F",
        10,
        "236220",
        "FL",
        "CORPORATION",
        "Existing or more than 2 years old",
        "Preferred Lenders Program",
        "FALSE",
        "TRUE",
        "78760",
    ),
    (
        "CHGOFF",
        2018,
        75000,
        37500,
        84,
        8.0,
        "V",
        4,
        "722511",
        "AZ",
        "PARTNERSHIP",
        "Startup, Loan Funds will Open Business",
        "SBA Express Program",
        "TRUE",
        "FALSE",
        "",
    ),
    (
        "PIF",
        2019,
        300000,
        225000,
        300,
        4.75,
        "F",
        15,
        "531210",
        "WA",
        "CORPORATION",
        "Existing, 5 or more years",
        "7a General",
        "FALSE",
        "TRUE",
        "",
    ),
    # missing naicscode -> dropped
    (
        "PIF",
        2013,
        60000,
        30000,
        60,
        6.0,
        "F",
        5,
        "",
        "CA",
        "CORPORATION",
        "Existing, 5 or more years",
        "7a General",
        "FALSE",
        "TRUE",
        "",
    ),
    # missing jobssupported -> dropped
    (
        "PIF",
        2013,
        60000,
        30000,
        60,
        6.0,
        "F",
        "",
        "722110",
        "CA",
        "CORPORATION",
        "Existing, 5 or more years",
        "7a General",
        "FALSE",
        "TRUE",
        "",
    ),
    # blank business_age / business_type -> "missing" category, not dropped
    (
        "PIF",
        2017,
        90000,
        45000,
        60,
        6.25,
        "V",
        6,
        "541511",
        "OR",
        "",
        "",
        "7a General",
        "FALSE",
        "TRUE",
        "",
    ),
]


@pytest.fixture
def raw_csv(tmp_path: Path) -> Path:
    df = pd.DataFrame(RAW_ROWS, columns=RAW_USECOLS)
    path = tmp_path / "sba_7a_synthetic.csv"
    df.to_csv(path, index=False)
    return path


@pytest.fixture
def features(raw_csv: Path) -> pd.DataFrame:
    return build_features(load_raw(raw_csv))


def test_drops_unresolved_statuses(features: pd.DataFrame) -> None:
    # 11 raw rows: CANCLD, EXEMPT, COMMIT dropped (3); missing naicscode/jobssupported
    # dropped (2). 6 resolved, complete rows remain.
    assert len(features) == 6


def test_label_mapping(features: pd.DataFrame) -> None:
    assert set(features[LABEL_COLUMN].unique()) <= {0, 1}
    # 2 CHGOFF rows in the surviving rows (fy=2013 and fy=2018).
    assert features[LABEL_COLUMN].sum() == 2


def test_no_leakage_or_identifying_columns(features: pd.DataFrame) -> None:
    for col in LEAKAGE_AND_IDENTIFYING_COLUMNS:
        assert col not in features.columns, f"leakage/identifying column present: {col}"


def test_engineered_numeric_features(features: pd.DataFrame) -> None:
    row = features[features["gross_approval"] == 100000].iloc[0]
    assert row["sba_guaranteed_approval"] == 50000
    assert row["sba_guaranty_pct"] == pytest.approx(0.5)
    assert row["revolver_status"] == 0
    assert row["collateral_ind"] == 1
    assert row["is_franchise"] == 0

    franchised = features[features["gross_approval"] == 200000].iloc[0]
    assert franchised["is_franchise"] == 1


def test_naics_sector_is_two_digit_prefix(features: pd.DataFrame) -> None:
    row = features[features["gross_approval"] == 100000].iloc[0]
    assert row["naics_sector_72"] == 1


def test_missing_categorical_becomes_missing_category(features: pd.DataFrame) -> None:
    row = features[features["gross_approval"] == 90000].iloc[0]
    assert row["business_type_missing"] == 1
    assert row["business_age_missing"] == 1
    assert not any(c.endswith("_nan") for c in features.columns)


def test_feature_columns_include_label_only_once(features: pd.DataFrame) -> None:
    cols = feature_columns(features)
    assert LABEL_COLUMN not in cols
    assert len(cols) == len(set(cols))
    assert "approval_fy" in cols  # both a feature and the split key


def test_time_split_is_disjoint_and_exhaustive(features: pd.DataFrame) -> None:
    fit, calib, test = time_split(features, fit_fy_max=2015, calib_fy_max=2017)
    assert len(fit) + len(calib) + len(test) == len(features)
    assert fit["approval_fy"].max() <= 2015
    assert calib["approval_fy"].min() > 2015
    assert calib["approval_fy"].max() <= 2017
    assert test["approval_fy"].min() > 2017


def test_split_summary_matches_default_rate(features: pd.DataFrame) -> None:
    fit, calib, test = time_split(features, fit_fy_max=2015, calib_fy_max=2017)
    summary = split_summary("test", test)
    assert summary["n_rows"] == len(test)
    assert summary["default_rate"] == pytest.approx(test[LABEL_COLUMN].mean())
