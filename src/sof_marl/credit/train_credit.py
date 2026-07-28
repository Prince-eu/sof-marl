"""Train and calibrate the SBA 7(a) credit-risk model; report held-out metrics.

Pipeline (docs/ARCHITECTURE.md section 5, docs/DATA.md):

1. Load the fit/calib/test splits written by ``build_dataset.py``.
2. Select XGBoost hyperparameters via a small, documented grid, scored by mean
   stratified 3-fold ROC-AUC on the fit split only -- calib and test are never
   used for model selection.
3. Refit the chosen configuration on the full fit split (this is the base model
   used later for SHAP in ``explain.py``).
4. Calibrate predicted probabilities with isotonic regression fit on the calib
   split, a cohort the base model never trained on.
5. Evaluate once, on the held-out test split (approval FY > calib_fy_max, never
   touched until here): ROC-AUC, PR-AUC, and Brier score, each with a bootstrap
   95% CI, plus a reliability (calibration) curve.
6. Save the base model, the calibrated model, a metrics JSON, and a two-panel
   ROC + calibration figure.

Usage:
    python -m sof_marl.credit.train_credit [--config config/env.yaml]
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import yaml
from numpy.typing import NDArray
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

RANDOM_SEED = 0
N_BOOTSTRAP = 1000
N_CV_FOLDS = 3

# Small, documented hyperparameter grid (docs/technical_report.md Appendix A).
# Selected by mean 3-fold stratified ROC-AUC on the fit split only.
HYPERPARAMETER_GRID: list[dict[str, Any]] = [
    {"max_depth": max_depth, "learning_rate": learning_rate}
    for max_depth in (4, 6, 8)
    for learning_rate in (0.03, 0.1)
]

# Fixed XGBoost settings, held constant across the grid.
FIXED_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "n_jobs": -1,
    "eval_metric": "auc",
    "random_state": RANDOM_SEED,
}


def load_split(
    processed_dir: Path, name: str, feature_columns: list[str], label_column: str
) -> tuple[pd.DataFrame, NDArray[np.int_]]:
    df = pd.read_csv(processed_dir / f"{name}.csv")
    return df[feature_columns], df[label_column].to_numpy()


def scale_pos_weight(y: NDArray[np.int_]) -> float:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return n_neg / n_pos


def select_hyperparameters(
    X: pd.DataFrame, y: NDArray[np.int_]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Small grid search scored by mean stratified k-fold ROC-AUC on the fit split."""
    spw = scale_pos_weight(y)
    skf = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    results: list[dict[str, Any]] = []
    for params in HYPERPARAMETER_GRID:
        fold_aucs: list[float] = []
        for train_idx, val_idx in skf.split(X, y):
            model = xgb.XGBClassifier(**FIXED_PARAMS, **params, scale_pos_weight=spw)
            model.fit(X.iloc[train_idx], y[train_idx])
            preds = model.predict_proba(X.iloc[val_idx])[:, 1]
            fold_aucs.append(float(roc_auc_score(y[val_idx], preds)))
        results.append(
            {
                **params,
                "cv_roc_auc_mean": float(np.mean(fold_aucs)),
                "cv_roc_auc_std": float(np.std(fold_aucs)),
            }
        )
    best = max(results, key=lambda r: float(r["cv_roc_auc_mean"]))
    best_params = {"max_depth": best["max_depth"], "learning_rate": best["learning_rate"]}
    return best_params, results


def bootstrap_ci(
    y_true: NDArray[np.int_],
    y_score: NDArray[np.float64],
    metric_fn: Any,
    n_boot: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for a metric computed on (y_true, y_score)."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats[i] = metric_fn(y_true[idx], y_score[idx])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def evaluate(y_true: NDArray[np.int_], y_score: NDArray[np.float64]) -> dict[str, dict[str, float]]:
    roc_auc = float(roc_auc_score(y_true, y_score))
    pr_auc = float(average_precision_score(y_true, y_score))
    brier = float(brier_score_loss(y_true, y_score))
    roc_lo, roc_hi = bootstrap_ci(y_true, y_score, roc_auc_score)
    pr_lo, pr_hi = bootstrap_ci(y_true, y_score, average_precision_score)
    brier_lo, brier_hi = bootstrap_ci(y_true, y_score, brier_score_loss)
    return {
        "roc_auc": {"value": roc_auc, "ci_low": roc_lo, "ci_high": roc_hi},
        "pr_auc": {"value": pr_auc, "ci_low": pr_lo, "ci_high": pr_hi},
        "brier_score": {"value": brier, "ci_low": brier_lo, "ci_high": brier_hi},
    }


def term_bucket_default_rates(df: pd.DataFrame, label_column: str) -> list[dict[str, Any]]:
    """Observed default rate by term-length bucket, for the term_months sensitivity note."""
    bin_edges = [0, 12, 36, 60, 84, 120, 180, 240, 400]
    buckets = pd.cut(df["term_months"], bin_edges)
    grouped = df.groupby(buckets, observed=True)[label_column].agg(["mean", "count"])
    return [
        {"term_months_range": str(idx), "default_rate": float(row["mean"]), "n": int(row["count"])}
        for idx, row in grouped.iterrows()
    ]


def term_months_sensitivity(
    X_fit: pd.DataFrame,
    y_fit: NDArray[np.int_],
    X_calib: pd.DataFrame,
    y_calib: NDArray[np.int_],
    X_test: pd.DataFrame,
    y_test: NDArray[np.int_],
    best_params: dict[str, Any],
) -> dict[str, Any]:
    """Refit excluding term_months to quantify its share of the headline ROC-AUC.

    ``term_months`` is legitimately known at loan origination (not leakage in the
    strict sense), but this FOIA snapshot only includes loans with a *resolved*
    status: loans still active as of the snapshot date are excluded. Long-term
    loans in this FY2010-FY2019 file have mostly not yet reached natural maturity,
    so nearly every resolved long-term loan is an early exit (usually an early
    payoff), while short-duration express/bridge-style loans resolve quickly either
    way. This couples term length to the resolution-timing mechanism itself, not
    only to origination-time underwriting risk, and its removal materially changes
    the headline number -- see docs/technical_report.md sections 5.1 and 7.
    """
    cols = [c for c in X_fit.columns if c != "term_months"]
    spw = scale_pos_weight(y_fit)
    model = xgb.XGBClassifier(**FIXED_PARAMS, **best_params, scale_pos_weight=spw)
    model.fit(X_fit[cols], y_fit)
    calibrated = CalibratedClassifierCV(estimator=model, method="isotonic", cv="prefit")
    calibrated.fit(X_calib[cols], y_calib)
    score = calibrated.predict_proba(X_test[cols])[:, 1]
    roc_auc = float(roc_auc_score(y_test, score))
    roc_lo, roc_hi = bootstrap_ci(y_test, score, roc_auc_score)
    return {
        "roc_auc_excluding_term_months": {"value": roc_auc, "ci_low": roc_lo, "ci_high": roc_hi},
        "note": (
            "term_months alone accounts for the majority of the headline model's "
            "discriminative power (see docs/technical_report.md 5.1/7): excluding it "
            "and refitting the identical pipeline drops ROC-AUC to the value above. "
            "This is a known characteristic of resolved-loan-only FOIA snapshots, "
            "where long-duration loans are structurally under-represented among "
            "observed defaults because most have not yet reached natural maturity."
        ),
    }


def plot_roc_and_calibration(
    y_true: NDArray[np.int_],
    y_score_uncalibrated: NDArray[np.float64],
    y_score_calibrated: NDArray[np.float64],
    roc_auc: float,
    roc_ci: tuple[float, float],
    out_path: Path,
) -> None:
    fig, (ax_roc, ax_cal) = plt.subplots(1, 2, figsize=(11, 5))

    fpr, tpr, _ = roc_curve(y_true, y_score_calibrated)
    ax_roc.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f} (95% CI {roc_ci[0]:.3f}-{roc_ci[1]:.3f})")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title("Held-out test ROC (real SBA 7(a) data)")
    ax_roc.legend(loc="lower right", fontsize=8)

    frac_pos_uncal, mean_pred_uncal = calibration_curve(
        y_true, y_score_uncalibrated, n_bins=10, strategy="quantile"
    )
    frac_pos_cal, mean_pred_cal = calibration_curve(
        y_true, y_score_calibrated, n_bins=10, strategy="quantile"
    )
    ax_cal.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax_cal.plot(mean_pred_uncal, frac_pos_uncal, marker="o", label="Uncalibrated")
    ax_cal.plot(mean_pred_cal, frac_pos_cal, marker="o", label="Isotonic-calibrated")
    ax_cal.set_xlabel("Mean predicted probability")
    ax_cal.set_ylabel("Observed default rate")
    ax_cal.set_title("Reliability curve, held-out test")
    ax_cal.legend(loc="upper left", fontsize=8)

    fig.suptitle("Credit-risk model: real held-out SBA 7(a) results (not simulated)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/env.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())["sba"]
    processed_dir = Path(cfg["processed_dir"])
    manifest = json.loads((processed_dir / "manifest.json").read_text())
    feature_columns: list[str] = manifest["feature_columns"]
    label_column: str = manifest["label_column"]

    X_fit, y_fit = load_split(processed_dir, "fit", feature_columns, label_column)
    X_calib, y_calib = load_split(processed_dir, "calib", feature_columns, label_column)
    X_test, y_test = load_split(processed_dir, "test", feature_columns, label_column)

    print(
        f"fit={len(X_fit)} calib={len(X_calib)} test={len(X_test)} "
        f"features={len(feature_columns)}"
    )

    print(f"Selecting hyperparameters via {N_CV_FOLDS}-fold CV on the fit split ...")
    best_params, grid_results = select_hyperparameters(X_fit, y_fit)
    print(f"Selected: {best_params}")

    base_model = xgb.XGBClassifier(
        **FIXED_PARAMS, **best_params, scale_pos_weight=scale_pos_weight(y_fit)
    )
    base_model.fit(X_fit, y_fit)

    calibrated_model = CalibratedClassifierCV(estimator=base_model, method="isotonic", cv="prefit")
    calibrated_model.fit(X_calib, y_calib)

    y_score_test_uncalibrated = base_model.predict_proba(X_test)[:, 1]
    y_score_test_calibrated = calibrated_model.predict_proba(X_test)[:, 1]

    metrics = evaluate(y_test, y_score_test_calibrated)
    print("Held-out test metrics (calibrated model):")
    for name, m in metrics.items():
        print(f"  {name}: {m['value']:.4f}  (95% CI {m['ci_low']:.4f}-{m['ci_high']:.4f})")

    print("Running term_months sensitivity check (docs/technical_report.md 5.1/7) ...")
    sensitivity = term_months_sensitivity(
        X_fit, y_fit, X_calib, y_calib, X_test, y_test, best_params
    )
    abl = sensitivity["roc_auc_excluding_term_months"]
    print(
        f"  ROC-AUC excluding term_months: {abl['value']:.4f}  "
        f"(95% CI {abl['ci_low']:.4f}-{abl['ci_high']:.4f})"
    )

    all_splits = pd.concat(
        [
            X_fit[["term_months"]].assign(**{label_column: y_fit}),
            X_calib[["term_months"]].assign(**{label_column: y_calib}),
            X_test[["term_months"]].assign(**{label_column: y_test}),
        ]
    )
    sensitivity["term_bucket_default_rates_all_splits"] = term_bucket_default_rates(
        all_splits, label_column
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    with (processed_dir / "credit_model_base.pkl").open("wb") as f:
        pickle.dump(base_model, f)
    with (processed_dir / "credit_model_calibrated.pkl").open("wb") as f:
        pickle.dump(calibrated_model, f)

    results_dir = Path("reports/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_out = {
        "n_fit": len(X_fit),
        "n_calib": len(X_calib),
        "n_test": len(X_test),
        "n_features": len(feature_columns),
        "fit_default_rate": float(y_fit.mean()),
        "calib_default_rate": float(y_calib.mean()),
        "test_default_rate": float(y_test.mean()),
        "hyperparameter_grid_results": grid_results,
        "selected_hyperparameters": {**FIXED_PARAMS, **best_params},
        "metrics_test_calibrated": metrics,
        "term_months_sensitivity": sensitivity,
        "n_bootstrap": N_BOOTSTRAP,
        "random_seed": RANDOM_SEED,
    }
    (results_dir / "credit_metrics.json").write_text(json.dumps(manifest_out, indent=2))

    figures_dir = Path("reports/figures")
    plot_roc_and_calibration(
        y_test,
        y_score_test_uncalibrated,
        y_score_test_calibrated,
        metrics["roc_auc"]["value"],
        (metrics["roc_auc"]["ci_low"], metrics["roc_auc"]["ci_high"]),
        figures_dir / "credit_roc_calibration.png",
    )
    print(
        f"Wrote {results_dir / 'credit_metrics.json'} and "
        f"{figures_dir / 'credit_roc_calibration.png'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
