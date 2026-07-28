"""Build the leakage-checked SBA 7(a) credit-risk dataset with a time-based split.

Reads the raw FOIA CSV fetched by ``data/download_sba.py``, keeps only loans with a
resolved outcome (paid in full vs charged off), engineers the feature set documented
in ``docs/DATA.md`` section 1.2, and writes three CSV splits to
``data/processed/`` per the fit/calibration/test cohorts in ``config/env.yaml``
(``sba.fit_fy_max``, ``sba.calib_fy_max``):

- ``fit.csv``   -- approval fiscal year <= ``fit_fy_max``, used to fit the model.
- ``calib.csv`` -- approval fiscal year in (``fit_fy_max``, ``calib_fy_max``], used
  only to calibrate predicted probabilities.
- ``test.csv``  -- approval fiscal year > ``calib_fy_max``, held out until final
  evaluation.

Every raw field that is only known at or after loan resolution (``paidinfulldate``,
``chargeoffdate``, ``grosschargeoffamount``) is excluded by never being read in the
first place (see ``RAW_USECOLS``), so leakage cannot happen even by mistake.

Usage:
    python -m sof_marl.credit.build_dataset [--config config/env.yaml]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

LABEL_COLUMN = "default"

# Raw loanstatus values that represent a *resolved* outcome, and their binary label.
# All other statuses (CANCLD, EXEMPT, COMMIT -- undisbursed, cancelled, or otherwise
# unresolved loans) are dropped: they carry no default/non-default ground truth.
RESOLVED_STATUS_LABELS = {"PIF": 0, "CHGOFF": 1}

# Raw FOIA columns this pipeline reads. Deliberately excludes any field populated
# only at or after resolution (paidinfulldate, chargeoffdate, grosschargeoffamount)
# and any borrower/lender-identifying field (name, street, zip, bank identifiers)
# not needed for a generalizable underwriting-time feature set.
RAW_USECOLS = [
    "loanstatus",
    "approvalfy",
    "grossapproval",
    "sbaguaranteedapproval",
    "terminmonths",
    "initialinterestrate",
    "fixedorvariableinterestind",
    "jobssupported",
    "naicscode",
    "borrstate",
    "businesstype",
    "businessage",
    "processingmethod",
    "revolverstatus",
    "collateralind",
    "franchisecode",
]

# Numeric features, all known at or before loan origination.
NUMERIC_FEATURES = [
    "gross_approval",
    "sba_guaranteed_approval",
    "sba_guaranty_pct",
    "term_months",
    "initial_interest_rate",
    "jobs_supported",
    "approval_fy",
    "revolver_status",
    "collateral_ind",
    "is_franchise",
]

# Categorical features, one-hot encoded over the full resolved population (before
# the time split) so the fit/calib/test splits share an identical column set.
CATEGORICAL_SOURCE_COLUMNS = [
    "naics_sector",
    "borrower_state",
    "business_type",
    "business_age",
    "fixed_or_variable",
    "processing_method",
]


def load_raw(raw_path: Path) -> pd.DataFrame:
    """Read the raw FOIA CSV, restricted to the columns this pipeline uses."""
    return pd.read_csv(raw_path, usecols=RAW_USECOLS, low_memory=False)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Filter to resolved loans and engineer the leakage-checked feature table."""
    df = raw[raw["loanstatus"].isin(RESOLVED_STATUS_LABELS)].copy()
    # naicscode and jobssupported are each missing on a handful of rows (4 total in
    # the FY2010-FY2019 snapshot); drop rather than impute a systemic risk field.
    df = df.dropna(subset=["naicscode", "jobssupported"])

    out = pd.DataFrame(index=df.index)
    out[LABEL_COLUMN] = df["loanstatus"].map(RESOLVED_STATUS_LABELS).astype("int8")

    out["gross_approval"] = df["grossapproval"].astype("float64")
    out["sba_guaranteed_approval"] = df["sbaguaranteedapproval"].astype("float64")
    out["sba_guaranty_pct"] = out["sba_guaranteed_approval"] / out["gross_approval"]
    out["term_months"] = df["terminmonths"].astype("float64")
    out["initial_interest_rate"] = df["initialinterestrate"].astype("float64")
    out["jobs_supported"] = df["jobssupported"].astype("float64")
    out["approval_fy"] = df["approvalfy"].astype("int32")
    # pandas' CSV parser infers a native bool dtype for these columns when every
    # value is TRUE/FALSE (no missing values), so compare via a normalized string
    # form rather than the literal string "TRUE" -- that would silently compare a
    # Python bool to a str and always be False.
    out["revolver_status"] = df["revolverstatus"].astype(str).str.upper().eq("TRUE").astype("int8")
    out["collateral_ind"] = df["collateralind"].astype(str).str.upper().eq("TRUE").astype("int8")
    out["is_franchise"] = (
        df["franchisecode"].fillna("").astype(str).str.strip().ne("").astype("int8")
    )

    out["naics_sector"] = df["naicscode"].astype(str).str[:2]
    out["borrower_state"] = df["borrstate"].fillna("missing").astype(str)
    out["business_type"] = df["businesstype"].fillna("missing").astype(str)
    out["business_age"] = df["businessage"].fillna("missing").astype(str)
    out["fixed_or_variable"] = df["fixedorvariableinterestind"].fillna("missing").astype(str)
    out["processing_method"] = df["processingmethod"].fillna("missing").astype(str)

    dummies = pd.get_dummies(
        out[CATEGORICAL_SOURCE_COLUMNS], prefix=CATEGORICAL_SOURCE_COLUMNS, dtype="int8"
    )
    out = pd.concat([out.drop(columns=CATEGORICAL_SOURCE_COLUMNS), dummies], axis=1)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All model feature columns: the fixed numeric set plus the one-hot dummies."""
    dummy_cols = sorted(c for c in df.columns if c not in {LABEL_COLUMN, *NUMERIC_FEATURES})
    return NUMERIC_FEATURES + dummy_cols


def time_split(
    df: pd.DataFrame, fit_fy_max: int, calib_fy_max: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by approval fiscal year: fit <= fit_fy_max < calib <= calib_fy_max < test."""
    fit = df[df["approval_fy"] <= fit_fy_max]
    calib = df[(df["approval_fy"] > fit_fy_max) & (df["approval_fy"] <= calib_fy_max)]
    test = df[df["approval_fy"] > calib_fy_max]
    return fit, calib, test


def split_summary(name: str, split: pd.DataFrame) -> dict[str, str | float | int]:
    n = len(split)
    n_default = int(split[LABEL_COLUMN].sum())
    return {
        "split": name,
        "n_rows": n,
        "n_default": n_default,
        "default_rate": n_default / n if n else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/env.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())["sba"]
    raw_path = Path(cfg["raw_path"])
    processed_dir = Path(cfg["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {raw_path}")
    raw = load_raw(raw_path)
    df = build_features(raw)
    cols = feature_columns(df)

    fit, calib, test = time_split(df, cfg["fit_fy_max"], cfg["calib_fy_max"])

    summaries = [
        split_summary("fit", fit),
        split_summary("calib", calib),
        split_summary("test", test),
    ]
    for s in summaries:
        print(
            f"{s['split']:>5}: n={s['n_rows']:>7}  "
            f"defaults={s['n_default']:>6}  rate={s['default_rate']:.4f}"
        )

    ordered_cols = [LABEL_COLUMN, *cols]
    fit[ordered_cols].to_csv(processed_dir / "fit.csv", index=False)
    calib[ordered_cols].to_csv(processed_dir / "calib.csv", index=False)
    test[ordered_cols].to_csv(processed_dir / "test.csv", index=False)

    manifest = {
        "raw_path": str(raw_path),
        "fit_fy_max": cfg["fit_fy_max"],
        "calib_fy_max": cfg["calib_fy_max"],
        "label_column": LABEL_COLUMN,
        "feature_columns": cols,
        "n_features": len(cols),
        "splits": summaries,
    }
    (processed_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote fit/calib/test CSVs and manifest.json to {processed_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
