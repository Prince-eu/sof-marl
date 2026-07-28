"""SHAP explanations for the SBA credit-risk model (docs/ARCHITECTURE.md section 5).

Loads the uncalibrated base XGBoost model saved by ``train_credit.py`` and computes
SHAP values on a fixed random sample of the held-out test split. SHAP is computed on
the *uncalibrated* base model: isotonic calibration is a monotonic, per-score
recalibration and does not change which features drive a prediction or their
relative ranking, so explaining the base model is equivalent and avoids SHAP's lack
of native support for the ``CalibratedClassifierCV`` meta-estimator.

Produces:

- ``reports/figures/credit_shap_summary.png`` -- global feature-importance summary
  over a sample of the held-out test set.
- ``reports/figures/credit_shap_example_high_risk.png`` and
  ``..._low_risk.png`` -- two per-decision waterfall explanations, for the
  highest- and lowest-scored loans in the test set.
- ``reports/results/credit_shap_examples.json`` -- the same two examples as a table
  of feature values and SHAP contributions, for the report narrative.

Usage:
    python -m sof_marl.credit.explain [--config config/env.yaml]
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
import shap
import yaml
from numpy.typing import NDArray

SAMPLE_SIZE = 2000
RANDOM_SEED = 0


def load_model_and_test(processed_dir: Path) -> tuple[Any, pd.DataFrame, list[str]]:
    manifest = json.loads((processed_dir / "manifest.json").read_text())
    feature_columns: list[str] = manifest["feature_columns"]
    with (processed_dir / "credit_model_base.pkl").open("rb") as f:
        model = pickle.load(f)
    test = pd.read_csv(processed_dir / "test.csv")
    return model, test[feature_columns], feature_columns


def make_summary_plot(explainer: Any, x_sample: pd.DataFrame, out_path: Path) -> None:
    shap_values = explainer(x_sample)
    plt.figure()
    shap.summary_plot(shap_values, x_sample, show=False, max_display=20)
    plt.title("SHAP summary: SBA 7(a) credit-risk model (held-out test sample)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def explain_example(
    explainer: Any, model: Any, row: pd.DataFrame, label: str, out_path: Path
) -> dict[str, Any]:
    """Waterfall figure + top-10 feature-contribution table for one test loan."""
    sv = explainer(row)
    predicted_prob = float(model.predict_proba(row)[0, 1])

    plt.figure()
    shap.plots.waterfall(sv[0], show=False, max_display=15)
    plt.title(f"SHAP explanation: {label} example (predicted p={predicted_prob:.3f})")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()

    contributions = sorted(
        zip(row.columns, sv.values[0], row.iloc[0].to_numpy(), strict=False),
        key=lambda t: -abs(float(t[1])),
    )[:10]
    return {
        "predicted_probability": predicted_prob,
        "top_contributions": [
            {"feature": f, "shap_value": float(v), "feature_value": float(x)}
            for f, v, x in contributions
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/env.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())["sba"]
    processed_dir = Path(cfg["processed_dir"])

    model, x_test, _feature_columns = load_model_and_test(processed_dir)
    x_sample = x_test.sample(n=min(SAMPLE_SIZE, len(x_test)), random_state=RANDOM_SEED)

    explainer = shap.TreeExplainer(model)
    figures_dir = Path("reports/figures")

    print(f"Computing SHAP values on a sample of {len(x_sample)} held-out test rows ...")
    make_summary_plot(explainer, x_sample, figures_dir / "credit_shap_summary.png")

    scores: NDArray[np.float64] = model.predict_proba(x_test)[:, 1]
    high_idx = int(np.argmax(scores))
    low_idx = int(np.argmin(scores))

    print("Explaining highest- and lowest-scored test loans ...")
    examples = {
        "high_risk": explain_example(
            explainer,
            model,
            x_test.iloc[[high_idx]],
            "high-risk",
            figures_dir / "credit_shap_example_high_risk.png",
        ),
        "low_risk": explain_example(
            explainer,
            model,
            x_test.iloc[[low_idx]],
            "low-risk",
            figures_dir / "credit_shap_example_low_risk.png",
        ),
    }

    results_dir = Path("reports/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "credit_shap_examples.json").write_text(json.dumps(examples, indent=2))
    print(f"Wrote SHAP figures to {figures_dir} and {results_dir / 'credit_shap_examples.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
