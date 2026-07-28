# EVALUATION — baselines, metrics, protocol, honest reporting

The evaluation exists to answer one question credibly: **does the coordinated
multi-agent design (Exhibit C.2) manage an SME's treasury better than the sensible
alternatives, and can a reviewer reproduce that finding?** Everything here is built
so the answer is defensible either way it comes out.

---

## 1. Policies compared

All four run in the identical environment on identical evaluation seeds:

1. Rule-based treasury heuristic (documented, non-straw-man).
2. Single-agent PPO (one controller, all actions).
3. IPPO — independent multi-agent, no coordination (`lambda = 0`).
4. MAPPO — coordinated multi-agent, centralized critic (`lambda > 0`) — the C.2 design.

A weak baseline is the most common way a POC like this loses credibility. The
rule-based policy must be one a competent treasurer would actually use, and the
single-agent baseline must be trained with the same budget as the MARL policies.

---

## 2. Metrics

Report all of these; do not select the flattering subset.

**Solvency / liquidity**
- Solvency-breach rate: fraction of weeks with an uncovered cash shortfall.
- Max drawdown: most negative cash position over an episode.
- Buffer-day distribution: mean and 10th percentile.
- Total financing cost incurred.

**Credit**
- Realized default-loss rate on accepted exposures.
- Approval precision / recall against true labels.
- Portfolio yield net of losses.
- (Model-level, from Phase A) held-out ROC-AUC, PR-AUC, calibration, with bootstrap CIs.

**Firm value / capital efficiency**
- Terminal firm value (cash + reserve + invested + collectible AR − debt).
- Risk-adjusted objective (mean − kappa·std of weekly firm-value change).
- Return on deployed capital.

**Coordination lift (the headline)**
- Δ(combined objective) of MAPPO vs each other policy, with significance.
- Δ(solvency-breach rate) of MAPPO vs IPPO (isolates coordination, since IPPO is the
  same agents without the shared critic/reward).

Define one **combined objective** up front in `config/train.yaml` (a weighted sum of
terminal firm value, negative solvency-breach rate, and negative financing cost, with
stated weights) and pre-register it in the report so the headline metric is not
chosen after seeing results.

---

## 3. Protocol

- **Seeds:** ≥5 training seeds per learned policy. Evaluate every policy on the same
  fixed set of ≥100 held-out episodes (evaluation seeds disjoint from training).
- **Reporting:** mean ± standard deviation across training seeds for every metric.
  For the headline lift, add a paired test across evaluation episodes (e.g., Wilcoxon
  signed-rank) and report the p-value and effect size, not just the mean gap.
- **Determinism:** evaluation rollouts are deterministic (greedy action) with fixed
  env seeds so results are reproducible bit-for-bit given the seed.
- **Compute disclosure:** report hardware, total training timesteps, and wall-clock.
- **Sensitivity analysis:** vary the 2–3 most consequential assumed constants
  (e.g., revenue volatility, default resolution schedule, buffer target) one at a
  time and show the qualitative conclusion is stable. If it is not stable, say so and
  scope the claim accordingly.

---

## 4. Figures (committed to `reports/figures/`)

1. Learning curves (combined objective vs timesteps) for single-agent, IPPO, MAPPO,
   with seed variance bands; rule-based as a horizontal reference line.
2. Coordination-lift bar chart: combined objective per policy, error bars over seeds.
3. Credit model: ROC curve with AUC + CI, calibration curve, and a SHAP summary plot.
4. Representative cash trajectory over the 52-week episode under each policy, marking
   solvency breaches.

Clean, labeled axes, real units, captions that state simulated-vs-real. No decorative
styling, no truncated axes that exaggerate differences.

---

## 5. Honest-reporting rules (mandatory)

- Report the result you got, including a null or negative coordination result. A
  cleanly reported "coordination gave a small, non-significant improvement in this
  POC" is credible and still demonstrates real, substantive progress on the system.
  A fabricated or cherry-picked win is disqualifying and, in a federal filing,
  dangerous.
- No superlatives the numbers do not support. Prefer "reduced simulated solvency
  breaches from X% to Y% (mean over 5 seeds, p = …)" over "dramatically improved."
- Separate the real credit-model result from the simulated treasury result in every
  place they appear.
- State limitations in the report body, not buried in a footnote: single SME profile
  class, simulated dynamics, proxy default labels, modest training budget, no live
  deployment.
- Reproducibility statement: exact SBA snapshot, pinned dependencies, seed list, and
  `scripts/reproduce.sh`. The headline number must regenerate from a clean checkout.

---

## 6. What "success" means for the filing

Success is **not** "coordination won by a big margin." Success is a real,
reproducible, honestly documented implementation of the petitioner's architecture
that produces measured results on public U.S. SME data. That artifact satisfies D-1
and upgrades Prong 1 merit regardless of the size of the coordination lift, provided
the work is genuine and clearly reported.
