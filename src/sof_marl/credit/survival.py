"""Survival-analysis reframe of the SBA 7(a) credit-risk model.

The binary classifier in ``train_credit.py`` reports a held-out ROC-AUC of ~0.94,
which is inflated by a selection bias: the classifier is trained only on *resolved*
loans (paid in full or charged off), so ``term_months`` becomes an artificial
shortcut (long-term loans that have resolved must have resolved early). This module
addresses that objection with the methodologically correct treatment -- a survival
model with right-censoring:

- Event = charge-off (default). Duration = months from approval to the event.
- Censored observations (no default observed):
  - ``PIF`` loans are censored at their paid-in-full date (they left the risk set by
    paying off, not defaulting -- prepayment is treated as non-informative censoring,
    a standard if imperfect simplification; a full competing-risks model is future
    work).
  - ``EXEMPT`` loans -- disbursed but not yet resolved as of the snapshot -- are the
    still-active loans the binary classifier *dropped*. Including them, censored at
    the snapshot date, is exactly what corrects the resolved-loan-only bias.

Model: XGBoost accelerated-failure-time (``survival:aft``), which natively handles
right-censored interval labels. Evaluation: Harrell's concordance index (the
survival analog of ROC-AUC, censoring-aware; lifelines) on a held-out later cohort,
with a bootstrap CI. Output: metrics JSON and a Kaplan-Meier figure stratifying the
held-out set by the model's predicted risk.

Usage:
    python -m sof_marl.credit.survival [--config config/env.yaml]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from numpy.typing import NDArray

from sof_marl.credit.build_dataset import NUMERIC_FEATURES, engineer_features

# Statuses kept for survival modeling and how each maps to (event, censoring endpoint).
# CANCLD (cancelled) and COMMIT (undisbursed) are dropped -- no disbursed exposure.
SURVIVAL_STATUSES = ("PIF", "CHGOFF", "EXEMPT")
RANDOM_SEED = 0
# 300 percentile-bootstrap resamples on the full held-out set: enough for a stable
# 95% CI on the concordance index without the runtime of 1000 on ~100k rows.
N_BOOTSTRAP = 300
DATE_COLUMNS = ["approvaldate", "asofdate", "paidinfulldate", "chargeoffdate"]

RAW_USECOLS_SURVIVAL = [
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
    *DATE_COLUMNS,
]

AFT_PARAMS: dict[str, Any] = {
    "objective": "survival:aft",
    "eval_metric": "aft-nloglik",
    "aft_loss_distribution": "normal",
    "aft_loss_distribution_scale": 1.20,
    "tree_method": "hist",
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "seed": RANDOM_SEED,
}
NUM_BOOST_ROUND = 300


def build_survival_frame(raw_path: Path) -> pd.DataFrame:
    """Return a frame with [duration, event, approval_fy, *features] for survival modeling."""
    raw = pd.read_csv(raw_path, usecols=RAW_USECOLS_SURVIVAL, low_memory=False)
    df = raw[raw["loanstatus"].isin(SURVIVAL_STATUSES)].copy()
    df = df.dropna(subset=["naicscode", "jobssupported"])
    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col].astype(str).str.strip(), errors="coerce")

    status = df["loanstatus"].to_numpy()
    # Event flag: 1 for charge-off, 0 for censored (PIF or EXEMPT).
    event = (status == "CHGOFF").astype("int8")
    # Censoring/event endpoint per status.
    end = df["paidinfulldate"].copy()
    end = end.where(status != "CHGOFF", df["chargeoffdate"])
    end = end.where(status != "EXEMPT", df["asofdate"])
    duration = (end - df["approvaldate"]).dt.days / 30.44

    features = engineer_features(df)
    out = features.copy()
    out.insert(0, "duration", duration.to_numpy())
    out.insert(1, "event", event)
    out = out[(out["duration"] > 0) & out["duration"].notna()]
    return out


def survival_feature_columns(frame: pd.DataFrame) -> list[str]:
    exclude = {"duration", "event", *NUMERIC_FEATURES}
    dummy_cols = sorted(c for c in frame.columns if c not in exclude)
    return NUMERIC_FEATURES + dummy_cols


def _aft_dmatrix(
    x: pd.DataFrame, duration: NDArray[np.float64], event: NDArray[np.int_]
) -> xgb.DMatrix:
    """DMatrix with AFT interval labels: [d, d] for events, [d, +inf] for right-censored."""
    lower = duration.astype(np.float64)
    upper = np.where(event == 1, duration, np.inf).astype(np.float64)
    dmat = xgb.DMatrix(x)
    dmat.set_float_info("label_lower_bound", lower)
    dmat.set_float_info("label_upper_bound", upper)
    return dmat


def bootstrap_cindex_ci(
    duration: NDArray[np.float64],
    predicted_time: NDArray[np.float64],
    event: NDArray[np.int_],
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(duration)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[i] = concordance_index(duration[idx], predicted_time[idx], event[idx])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def plot_km_by_risk(
    duration: NDArray[np.float64],
    event: NDArray[np.int_],
    predicted_time: NDArray[np.float64],
    out_path: Path,
) -> None:
    """Kaplan-Meier survival curves for held-out loans stratified into risk quartiles
    by the model's predicted survival time (lower predicted time = higher risk)."""
    quartiles = np.quantile(predicted_time, [0.25, 0.5, 0.75])
    groups = np.digitize(predicted_time, quartiles)  # 0=highest risk .. 3=lowest risk
    labels = [
        "Q1 (highest predicted risk)",
        "Q2",
        "Q3",
        "Q4 (lowest predicted risk)",
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    kmf = KaplanMeierFitter()
    for g in range(4):
        mask = groups == g
        kmf.fit(duration[mask], event_observed=event[mask], label=labels[g])
        kmf.plot_survival_function(ax=ax, ci_show=False)
    ax.set_xlabel("Months since approval")
    ax.set_ylabel("Survival probability (not charged off)")
    ax.set_title("Held-out survival by model-predicted risk quartile (real SBA 7(a) data)")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/env.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())["sba"]
    raw_path = Path(cfg["raw_path"])
    calib_fy_max = int(cfg["calib_fy_max"])  # train on <= this; test on the newer cohort

    print(f"Building survival frame from {raw_path} ...")
    frame = build_survival_frame(raw_path)
    cols = survival_feature_columns(frame)

    train = frame[frame["approval_fy"] <= calib_fy_max]
    test = frame[frame["approval_fy"] > calib_fy_max]
    print(
        f"train(<= FY{calib_fy_max}) n={len(train)} events={int(train['event'].sum())} | "
        f"test(> FY{calib_fy_max}) n={len(test)} events={int(test['event'].sum())} | "
        f"features={len(cols)}"
    )

    dtrain = _aft_dmatrix(train[cols], train["duration"].to_numpy(), train["event"].to_numpy())
    booster = xgb.train(AFT_PARAMS, dtrain, num_boost_round=NUM_BOOST_ROUND)

    dtest = xgb.DMatrix(test[cols])
    predicted_time = booster.predict(dtest)  # AFT predicts survival time (higher = safer)
    test_duration = test["duration"].to_numpy()
    test_event = test["event"].to_numpy()

    c_index = float(concordance_index(test_duration, predicted_time, test_event))
    ci_low, ci_high = bootstrap_cindex_ci(test_duration, predicted_time, test_event)
    print(f"Held-out concordance index: {c_index:.3f}  (95% CI {ci_low:.3f}-{ci_high:.3f})")

    # term_months ablation: refit without it and re-evaluate, to parallel the binary
    # classifier's ablation (0.939 -> 0.622) and quantify residual term-dependence.
    cols_no_term = [c for c in cols if c != "term_months"]
    dtrain_nt = _aft_dmatrix(
        train[cols_no_term], train["duration"].to_numpy(), train["event"].to_numpy()
    )
    booster_nt = xgb.train(AFT_PARAMS, dtrain_nt, num_boost_round=NUM_BOOST_ROUND)
    pred_nt = booster_nt.predict(xgb.DMatrix(test[cols_no_term]))
    c_index_no_term = float(concordance_index(test_duration, pred_nt, test_event))
    print(f"Held-out concordance index without term_months: {c_index_no_term:.3f}")

    # Feature importance (gain) -- to show term_months is no longer a runaway shortcut.
    gain_raw = booster.get_score(importance_type="gain")
    gain: dict[str, float] = {
        k: (float(sum(v)) if isinstance(v, list) else float(v)) for k, v in gain_raw.items()
    }
    top = sorted(gain.items(), key=lambda kv: -kv[1])[:8]
    term_share = gain.get("term_months", 0.0) / (sum(gain.values()) or 1.0)

    results_dir = Path("reports/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "n_train": len(train),
        "n_test": len(test),
        "n_events_train": int(train["event"].sum()),
        "n_events_test": int(test["event"].sum()),
        "n_features": len(cols),
        "train_fy_max": calib_fy_max,
        "concordance_index": c_index,
        "concordance_index_ci": [ci_low, ci_high],
        "concordance_index_without_term_months": c_index_no_term,
        "term_months_gain_share": term_share,
        "top_features_by_gain": [{"feature": f, "gain": float(g)} for f, g in top],
        "n_bootstrap": N_BOOTSTRAP,
        "aft_params": AFT_PARAMS,
        "included_exempt_as_censored": True,
    }
    (results_dir / "credit_survival_metrics.json").write_text(json.dumps(metrics, indent=2))

    figures_dir = Path("reports/figures")
    plot_km_by_risk(
        test_duration, test_event, predicted_time, figures_dir / "credit_survival_km.png"
    )
    print(
        f"term_months gain share: {term_share:.3f}. "
        f"Wrote {results_dir / 'credit_survival_metrics.json'} and the Kaplan-Meier figure."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
