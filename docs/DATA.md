# DATA — sources, licensing, and calibration

Two kinds of data feed the POC: (1) **real** public loan-outcome data that trains the
credit-risk model, and (2) **public summary statistics** used to calibrate the
simulated treasury environment. The first is genuine ground truth; the second only
sets realistic constants for a disclosed simulation. Keep the two clearly separated
in code, in figures, and in the report.

---

## 1. Real data: SBA 7(a) FOIA loan-level dataset (credit-risk model)

- **What:** loan-level records for the SBA 7(a) program, including loan amount, term,
  industry (NAICS), borrower location, jobs, business age, and — critically — the
  resolved **loan status** (paid in full vs charged off), which is the supervised
  target.
- **Where:** SBA Open Data portal, "7(a) & 504 FOIA" dataset. Files are segmented by
  period (e.g., FY2010–FY2019, FY2020–present) as CSV, with a published data
  dictionary. Updated quarterly.
  - Dataset: https://data.sba.gov/en/dataset/7-a-504-foia
  - Mirror / catalog entry: https://catalog.data.gov/dataset/sba-7a-and-504-loan-data-reports
- **License / use:** U.S. Government public-domain open data (released under FOIA on
  the federal open-data portal). Free to use; cite the SBA as source. Record the
  exact file name and "as of" date you downloaded in `data/README.md` and in the
  report, because the portal refreshes quarterly and results must be reproducible
  against a specific snapshot.
- **Acquisition:** `data/download_sba.py` downloads the chosen CSV to `data/raw/`
  (git-ignored). Pin the file name and an expected row count / checksum so a
  reviewer can confirm they have the same snapshot.

### 1.1 Label
- Keep rows with a resolved status: `PIF` (paid in full) → 0, `CHGOFF` (charged off)
  → 1. Drop `EXEMPT`, active/undisbursed, cancelled, or otherwise unresolved rows.
- Document the resulting class balance; charge-offs are the minority class, so use
  class weighting or `scale_pos_weight`, and evaluate with PR-AUC in addition to
  ROC-AUC.

### 1.2 Features (avoid leakage)
Use only fields knowable at or before origination. Reasonable set:
- Numeric: gross approval amount, SBA-guaranteed amount, term (months), number of
  employees / jobs supported.
- Categorical (one-hot or target-encode with care): NAICS 2-digit sector, borrower
  state, urban/rural flag, business type (new vs existing), revolver flag.
- **Exclude** anything populated only at resolution (e.g., charge-off amount, gross
  disbursement post-outcome, paid-in-full date). Leakage here would inflate AUC and
  destroy credibility — check each field against the data dictionary.

### 1.3 Split
- Time-based split: train on earlier approval-year cohorts, test on later ones
  (e.g., train ≤ FY2017, test FY2018+). This mimics real forward prediction and
  avoids look-ahead. State the exact cutoff.
- Optionally restrict to a recent, homogeneous window to reduce macro-regime drift;
  if you do, justify it and report dataset size.

### 1.4 Honest expectations
Published and community work on this dataset lands ROC-AUC roughly in the
**0.65–0.78** range depending on features, window, and encoding. Report your real
number with a confidence interval (bootstrap on the test set). Do not tune on the
test set. If the number is modest, that is fine and expected — it is real.

---

## 2. Calibration data: public statistics for the simulation

The environment is a **calibrated simulation**, not real firm data. Each constant in
`config/env.yaml` must cite a public source in a comment. Use figures the petition
already relies on where possible, so the simulation is consistent with the brief.

Suggested calibration anchors (verify current values when you pull them):

| Constant | Anchor | Source (already in the petition) |
|---|---|---|
| Median cash buffer days for small firms (sets default buffer target and shortfall frequency) | ~27 days median (varies by sector) | JPMorgan Chase Institute, *Cash is King* [Exhibit B.18] |
| Share of small firms seeking external financing; approval/shortfall rates | Fed Small Business Credit Survey figures | *2026 Report on Employer Firms* / SBCS [Exhibits B.5, B.6] |
| SME population scale / sector mix context | 36.2M small businesses; sector shares | SBA Office of Advocacy 2025 profile [Exhibit B.7] |
| Financing gap / constraint framing | qualitative realism of credit scarcity | BCG/Biz2X, IFC [Exhibits B.16, B.19] |
| Interest-rate / financing-cost level | current small-business loan rate range | FRED series (cite series ID and pull date) |

Rules:
- Every calibration constant has a source comment. If no public anchor exists for a
  constant, mark it `# assumption (no public anchor); sensitivity-tested` and include
  it in the sensitivity analysis (EVALUATION.md).
- Seasonality, volatility, and collection-curve shapes that are assumptions get
  disclosed as assumptions and, where they matter, sensitivity-tested.
- Never present a simulated quantity as an empirical result. Figure captions on
  simulated outputs must read "simulated (calibrated to public statistics)."

---

## 3. Data hygiene checklist (before training anything)

- [ ] Snapshot recorded: file name, "as of" date, row count, checksum.
- [ ] Label mapping and dropped statuses documented; class balance reported.
- [ ] Every feature checked against the data dictionary for leakage.
- [ ] Time-based split defined and stated.
- [ ] All `env.yaml` constants carry a source comment or an `# assumption` tag.
- [ ] `data/` raw and processed are git-ignored; only derived, shareable artifacts
      (metrics, figures) are committed.
