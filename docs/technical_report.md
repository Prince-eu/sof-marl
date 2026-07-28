# SOF-MARL: A Multi-Agent Reinforcement-Learning Prototype for Autonomous SME Treasury Management

**Technical Report**

Author: Prince Enyiorji
Version: 0.1 (draft scaffold; bracketed items are filled from the actual run)
Date: [[FILL: report date]]
Code snapshot: [[FILL: git commit hash]]
Data snapshot: SBA 7(a) FOIA, file [[FILL: file name]], as of [[FILL: date]], [[FILL: row count]] rows

> Scaffold note (delete before finalizing): every `[[FILL: ...]]` is a number, figure,
> or statement produced by the run. Do not write any results sentence until the
> corresponding number exists. Nothing in this report describes work that was not done.
> The treasury environment is a calibrated simulation and is labeled as such wherever
> it appears; the credit-risk model is trained on real public data.

---

## Abstract

This report documents a proof-of-concept implementation of the self-optimizing,
cloud-native autonomous finance architecture for small and medium enterprises (SMEs)
proposed in Enyiorji (2025). The system uses multi-agent reinforcement learning
(MARL) to manage an SME's weekly treasury decisions across four functions: liquidity
forecasting, credit-risk assessment, expenditure optimization, and capital allocation.
The credit-risk function is grounded in a supervised model trained on the U.S. Small
Business Administration (SBA) 7(a) public loan-outcome dataset, which achieves a
held-out ROC-AUC of [[FILL]] ([[FILL: 95% CI]]). The four functions operate as
cooperating agents in a treasury environment calibrated to public U.S. small-business
statistics. We compare four control policies on identical evaluation episodes: a
rule-based heuristic, a single-agent controller, independent multi-agent RL (IPPO),
and coordinated multi-agent RL (MAPPO). Coordinated control [[FILL: changed]] the
combined treasury objective by [[FILL: value]] relative to the strongest baseline
(mean over [[FILL: N]] seeds, [[FILL: test, p-value]]) and [[FILL: changed]] the
simulated solvency-breach rate from [[FILL]] to [[FILL]]. We report interpretability
(SHAP) for the credit model, a sensitivity analysis over key assumptions, limitations,
and a full reproducibility statement.

---

## 1. Objective and scope

### 1.1 Goal
Demonstrate, with measured and reproducible results on public U.S. SME data, that the
MARL architecture in Enyiorji (2025) can be implemented and can manage an SME's
treasury, and test the paper's central claim that coordinated multi-agent control
outperforms non-coordinated and single-agent alternatives.

### 1.2 What this prototype is and is not
In scope: one representative SME profile class; a 52-week (one fiscal year) episode;
four cooperating agents; a real-data credit-risk model; a four-way policy comparison;
interpretability; sensitivity analysis.

Out of scope (stated as future work, not implied to exist): real firm or bank data
beyond the public SBA dataset; live or streaming deployment; a user interface;
regulatory certification or production MLOps; large-scale hyperparameter search.

### 1.3 Design basis
The architecture follows Enyiorji (2025), *Designing a Self-Optimizing Cloud-Native
Autonomous Finance System for SMEs Using Multi-Agent Reinforcement Learning*. This
report implements a scoped subset sufficient to test the core claim.

---

## 2. System architecture

Brief prose summary here; full specification in the project's ARCHITECTURE.md.

- **Environment.** The SME treasury is modeled as a Markov game with one shared cash
  pool and four agents acting each week. State includes cash, receivables aging,
  payables schedule, revolving credit and term debt, seasonal revenue and expenses,
  pending credit exposures, reserve, and invested capital. Dynamics enforce an
  accounting identity every step (cash conservation), verified by unit tests.
- **Agents.** Liquidity forecasting, credit-risk assessment, expenditure
  optimization, capital allocation. Each has a defined observation slice, action
  space, and reward. [[FILL: one-line note on which agents were fully trained vs
  simplified in this POC, if any.]]
- **Coordination.** A shared firm-level reward plus a centralized critic (MAPPO)
  provides the coordination signal; the independent baseline (IPPO) removes it. This
  isolates the value of coordination.

Figure 1. System diagram. [[FILL: figure reference]]

---

## 3. Data

### 3.1 Real data: SBA 7(a) FOIA loan outcomes (credit-risk model)
- Source: SBA Open Data portal, 7(a) and 504 FOIA dataset (public-domain U.S.
  Government open data). https://data.sba.gov/en/dataset/7-a-504-foia
- Snapshot: file [[FILL]], as of [[FILL]], [[FILL: rows]], checksum [[FILL]].
- Target: charge-off (1) vs paid-in-full (0); unresolved statuses dropped. Class
  balance: [[FILL: % charged off]].
- Features: [[FILL: final feature list]]; leakage-checked against the data
  dictionary (fields known only at resolution excluded).
- Split: time-based; train on approval years [[FILL]], test on [[FILL]].

### 3.2 Calibration data: public statistics for the simulation
The treasury environment is a simulation calibrated to public statistics. Each
constant is sourced.

| Constant | Value used | Source |
|---|---|---|
| Median cash buffer days | [[FILL]] | JPMorgan Chase Institute, *Cash is King* |
| Financing-seeking / approval rates | [[FILL]] | Fed Small Business Credit Survey |
| Sector / population context | [[FILL]] | SBA Office of Advocacy 2025 profile |
| Small-business loan rate | [[FILL]] | FRED series [[FILL: series ID]], pulled [[FILL]] |
| [[FILL: any assumption]] | [[FILL]] | assumption (no public anchor); sensitivity-tested |

---

## 4. Methods

### 4.1 Credit-risk model
Gradient-boosted trees (XGBoost), probability-calibrated. Metrics: held-out ROC-AUC,
PR-AUC, calibration (Brier score / reliability curve), with bootstrap confidence
intervals. Hyperparameters and the small grid searched: Appendix A.

### 4.2 Environment and reward
Reward terms and weights per ARCHITECTURE.md, fixed in `config/agents.yaml`.

**Combined treasury objective (pre-registered).** All components are computed per
episode and are dimensionless, so the weights are directly interpretable. Let `V_t` be
firm value in week `t` and `V_0` the initial firm value.

- Return: `r = (V_52 - V_0) / V_0`
- Volatility: `s = stdev(weekly change in V) / V_0`
- Solvency-breach rate: `B = (weeks with an uncovered cash shortfall) / 52`

Risk-adjusted return: `RAR = r - kappa * s`.
Combined objective (headline): `J = r - kappa * s - lambda * B`.

Weights (balanced posture, pre-registered): `lambda = 3.0` (solvency), `kappa = 1.0`
(volatility), encoding the priority ordering solvency > return > volatility. These are
set in `config/train.yaml`. Financing cost is reported as a diagnostic and is not a term
in `J`, because it is already reflected in `V_52` (interest paid reduces cash and
therefore firm value); adding it separately would double-count it.

**Scale check (pre-registered, run before any learned-policy result).** After the
environment is built, verify on the rule-based baseline alone that no single term of `J`
dominates the others by an order of magnitude. If it does, adjust the weights once,
record that the adjustment was made before any RL results were seen, and freeze the
definition. Report the final weights and whether any adjustment occurred.

### 4.3 Policies compared
1. Rule-based treasury heuristic (fixed buffer, pay-on-due, fixed credit cutoff,
   fixed surplus split). Documented in Appendix B.
2. Single-agent PPO (one controller, full action vector).
3. IPPO (independent multi-agent, no coordination).
4. MAPPO (coordinated multi-agent, centralized critic).

### 4.4 Training and evaluation protocol
- Seeds: [[FILL: N >= 5]] training seeds per learned policy.
- Evaluation: [[FILL: >= 100]] held-out episodes, fixed eval seeds shared across all
  policies, deterministic (greedy) rollouts.
- Reporting: mean and standard deviation across seeds; paired test
  ([[FILL: e.g., Wilcoxon signed-rank]]) for the coordination lift with effect size.
- Compute: [[FILL: hardware]], [[FILL: total timesteps]], [[FILL: wall-clock]].

---

## 5. Results

### 5.1 Credit-risk model (real data)
| Metric | Value | 95% CI |
|---|---|---|
| ROC-AUC | [[FILL]] | [[FILL]] |
| PR-AUC | [[FILL]] | [[FILL]] |
| Brier score | [[FILL]] | [[FILL]] |

Figure 2. ROC curve and calibration curve. [[FILL]]

Narrative: [[FILL: one honest paragraph. State the AUC plainly. A value in the mid-0.6s
to mid-0.7s is a normal result on this dataset; do not editorialize it upward.]]

### 5.2 Treasury environment and baseline behavior
Narrative on how the rule-based and single-agent policies behave, to establish that
the baselines are credible and the task is non-trivial. [[FILL]]

Figure 3. Representative 52-week cash trajectory per policy, solvency breaches marked.
[[FILL]]

### 5.3 Policy comparison (headline)
All values are mean +/- std over [[FILL: N]] seeds, on the shared evaluation episodes.

| Policy | Solvency-breach rate (B) | Max drawdown | Financing cost (diag.) | Default-loss rate | Return (r) | Risk-adj. return (RAR) | Combined objective (J) |
|---|---|---|---|---|---|---|---|
| Rule-based | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] |
| Single-agent PPO | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] |
| IPPO (independent) | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] |
| MAPPO (coordinated) | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] |

Coordination lift (MAPPO vs each):

| Comparison | Delta (combined objective) | Delta (solvency-breach rate) | Test statistic | p-value | Effect size |
|---|---|---|---|---|---|
| MAPPO vs rule-based | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] |
| MAPPO vs single-agent | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] |
| MAPPO vs IPPO | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] | [[FILL]] |

Figure 4. Learning curves (combined objective vs timesteps) with seed bands, and a
coordination-lift bar chart. [[FILL]]

Narrative: [[FILL: report exactly what happened, including a null or negative result.
"Coordinated control improved the combined objective by X (p = ...)" or "the
coordination advantage was small and not statistically significant in this POC,
consistent with ...". Both are valid outcomes for this report.]]

### 5.4 Sensitivity analysis
Vary the most consequential assumed constants one at a time; show whether the
qualitative conclusion holds.

| Constant varied | Range | Effect on headline conclusion |
|---|---|---|
| [[FILL: e.g., revenue volatility]] | [[FILL]] | [[FILL]] |
| [[FILL: e.g., default resolution schedule]] | [[FILL]] | [[FILL]] |
| [[FILL: e.g., buffer target]] | [[FILL]] | [[FILL]] |

---

## 6. Interpretability

SHAP analysis of the credit-risk model: global feature importance and example
per-decision explanations. This addresses a known concern about reinforcement-learning
and machine-learning financial systems, that opaque models complicate managerial
trust, and it follows the explainability approach the author has applied in prior
production credit-model work.

Figure 5. SHAP summary plot. [[FILL]]
Narrative: [[FILL: which features drive predicted default risk, and whether the
directions are economically sensible.]]

---

## 7. Limitations

State plainly, in the body:
- The treasury environment is a calibrated simulation, not observed firm data.
- The credit model uses SBA 7(a) loan outcomes as a proxy for SME default risk; it is
  real but not identical to the trade-credit exposures modeled in the environment.
- A single SME profile class is modeled; generalization across sectors and sizes is
  future work.
- Training budget is POC-scale; results are not a production performance ceiling.
- [[FILL: any others surfaced during the build.]]

---

## 8. Reproducibility

- Data: exact SBA snapshot named above; `data/download_sba.py` fetches it; checksum
  provided.
- Code: commit [[FILL]]; pinned `requirements.txt`; fixed seed list [[FILL]].
- One command: `scripts/reproduce.sh` runs data preparation, credit-model training,
  MARL training, evaluation, and figure generation, and regenerates the headline
  numbers in this report.
- Environment: [[FILL: Python version, OS, hardware]].

---

## 9. Conclusion

[[FILL: two to four honest sentences. What was implemented, what the real credit result
was, what the coordination experiment showed, and what it establishes about the
feasibility of the proposed architecture. Do not overclaim. Tie back to the design
basis in Enyiorji (2025), and point to the limitations and future work above.]]

---

## References

- Enyiorji, P. (2025). Designing a self-optimizing cloud-native autonomous finance
  system for SMEs using multi-agent reinforcement learning. *International Journal of
  Financial Management and Economics*, 8(1), 596 to 605.
- U.S. Small Business Administration. 7(a) and 504 FOIA loan data.
  https://data.sba.gov/en/dataset/7-a-504-foia
- [[FILL: JPMorgan Chase Institute, Federal Reserve SBCS, SBA Office of Advocacy, FRED
  series, and any library/method citations actually used.]]

---

## Appendix A. Credit-model hyperparameters and grid
[[FILL]]

## Appendix B. Rule-based policy specification
[[FILL]]

## Appendix C. Full configuration files
[[FILL: env.yaml, agents.yaml, train.yaml as used for the reported run.]]
