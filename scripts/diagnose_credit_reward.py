"""Quantitative diagnosis of the credit-risk agent's reward economics.

Reproduces the analysis that motivated the credit-agent reward redesign (see
docs/ARCHITECTURE.md section 4.2 and reports/technical_report.md 5.3/7): computes
the break-even predicted default probability, the fraction of the real exposure
pool that is profitable to lend to, the normalized reward magnitudes, and -- the
key finding -- that the original per-week concentration penalty taxed holding any
exposure book at roughly 100x the margin it earned.

Usage:
    python scripts/diagnose_credit_reward.py
"""

from __future__ import annotations

import pandas as pd

from sof_marl.agents import rewards as R
from sof_marl.env import calibration


def main() -> int:
    cfg = calibration.load_env_config()
    ac = R.load_agents_config()
    scale = cfg.revenue.weekly_mean
    margin_bps = ac.credit_risk_rewards.margin_bps
    premium_bps = ac.credit_risk_premium_bps
    lgd = cfg.loss_given_default_frac

    be_plain = (margin_bps / 10000) / ((margin_bps / 10000) + lgd)
    be_prem = ((margin_bps + premium_bps) / 10000) / (((margin_bps + premium_bps) / 10000) + lgd)
    print(f"reward scale (weekly revenue mean): ${scale:,.0f}")
    print(f"break-even predicted default prob: plain {be_plain:.3f} | premium {be_prem:.3f}")

    pool = pd.read_csv(cfg.processed_dir / "credit_exposure_pool.csv")
    pool["exposure_amount"] = pool["gross_approval"] * cfg.credit_exposure_scale_frac
    print(f"\nexposure pool n={len(pool)}  true default rate={pool['default'].mean():.3f}")
    print(f"predicted_prob median={pool.predicted_prob.median():.4f}  mean={pool.predicted_prob.mean():.4f}")
    print(f"fraction below plain break-even: {(pool.predicted_prob < be_plain).mean():.3f}")

    amt = pool.exposure_amount.median()
    margin_one = amt * margin_bps / 10000 / scale
    conc_one_week = 0.2 * amt / scale  # original w_concentration = 0.2, per week
    print(f"\nper median exposure (${amt:,.0f}):")
    print(f"  margin if repaid (normalized):            {margin_one:+.5f}")
    print(f"  ORIGINAL concentration penalty PER WEEK:  {conc_one_week:.5f}")
    print(f"  original penalty over 12-week hold:       {conc_one_week * 12:.5f}")
    print(f"  -> holding one good exposure cost ~{conc_one_week * 12 / margin_one:.0f}x its margin (the bug)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
