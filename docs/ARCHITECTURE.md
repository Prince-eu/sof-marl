# ARCHITECTURE — SOF-MARL Prototype

Technical design for the multi-agent environment, the four agents, the credit-risk
model integration, and the coordination mechanism. This is the contract the code in
`src/sof_marl/` implements. Keep this file and the code in sync; if a design choice
changes during the build, change it here first.

Notation: time step `t` is one week; an episode is `T = 52` weeks (one fiscal year).
All monetary quantities are in USD and, in code, held as float and rounded only for
display.

---

## 1. The firm as a Markov game

We model an SME's treasury as a partially decomposable Markov game with a shared
environment state and four agents acting simultaneously each week. The firm has one
cash pool; the agents pursue different objectives over that pool, which is what
creates both the need for and the payoff from coordination.

Formally: `(N, S, {A_i}, P, {R_i}, R_shared, gamma)` with `N = 4` agents, shared
state space `S`, per-agent action spaces `A_i`, transition kernel `P`, per-agent
rewards `R_i`, a shared firm-level reward `R_shared`, and discount `gamma = 0.99`.

---

## 2. State

The environment maintains a full financial state; each agent observes the slice
relevant to its function plus shared solvency signals (so coordination is possible).

Core state variables (updated each week by `dynamics.py`):

| Variable | Meaning |
|---|---|
| `cash` | cash on hand |
| `ar_buckets` | accounts receivable in aging buckets [0–30, 31–60, 61–90] |
| `ap_schedule` | accounts payable with due weeks |
| `credit_line_limit`, `credit_drawn` | revolving line capacity and current draw |
| `term_debt`, `interest_rate` | outstanding term debt and its rate |
| `revenue_mu(t)`, `revenue_sigma` | seasonal expected revenue and volatility |
| `fixed_expenses`, `discretionary_budget` | committed vs deferrable outflows |
| `pending_credit_requests` | list of trade-credit/financing exposures to score this week (see §4) |
| `reserve`, `invested` | capital held in reserve vs deployed |
| `week`, `season_index` | calendar features |

Observation for agent `i` = its function-specific variables + a small shared vector
`[cash, projected_cash_next_k, credit_headroom, week/T]`. Normalize observations
(running mean/std or fixed scales from `env.yaml`); document the scaling.

**Implementation (Phase B), `src/sof_marl/env/`.** `ar_buckets` age by one stage per
*week* (not per 30 calendar days): `collection_curve` is specified per week in
`DATA.md`, so the `[0-30, 31-60, 61-90]`-day labels are the standard AR-aging-report
terminology for three successive weekly aging stages, not a literal 30-day-per-tick
mapping. `ap_schedule` is a rolling `ap_schedule_weeks = 2` array `[due this week, due
next week]`: new fixed expenses are scheduled one week out, giving the liquidity
agent's `defer_payables_frac` a real future slot to defer into (deferred amounts
roll into next week's due-now slot and accrue `late_payment_penalty_apr`). All
dollar-valued observation fields are normalized by dividing by the calibrated
weekly revenue mean (`cfg.revenue.weekly_mean = annual_mean / 52`); `week/T` is
already in `[0, 1]`. `projected_cash_next_k` uses `k = 4` weeks, projected linearly
from the current cash balance and the calibrated net weekly flow (revenue mean minus
fixed and discretionary expense fractions) -- a deliberately simple, deterministic
lookahead, not a forecast model.

---

## 3. Dynamics (accounting must close every step)

Each week, in order:

1. Revenue realizes: `rev_t ~ max(0, Normal(revenue_mu(t), revenue_sigma))`,
   seasonal `revenue_mu(t)` from calibration (DATA.md).
2. Receivables age; a fraction collects into cash per the collection curve;
   a small fraction becomes bad debt.
3. Agents' actions apply (see §4): draws/repayments, payables timing, discretionary
   spend, credit approvals, capital moves.
4. Expenses settle; payables due this week are paid (or deferred if the liquidity
   agent chose to, incurring a late penalty).
5. Financing costs accrue on `credit_drawn` and `term_debt`.
6. Accepted credit exposures resolve over subsequent weeks: each repays on schedule
   unless it defaults, with default drawn against the **real credit model's**
   predicted probability for that exposure (see §5).
7. State advances; solvency checked.

**Invariant (unit-tested in `test_dynamics.py`):**
`cash_{t+1} == cash_t + inflows_t - outflows_t` to floating tolerance. No term may
inject or destroy value outside a named inflow/outflow. A shortfall (`cash < 0`)
is allowed but penalized and, if uncovered by available credit headroom, triggers a
solvency-breach flag used in metrics.

**Implementation (Phase B).** `dynamics.step` is a pure function
`(FirmState, WeekActions, EnvConfig, rng) -> StepResult`, returning every named
cash inflow/outflow for the week (`INFLOW_KEYS`/`OUTFLOW_KEYS` in
`src/sof_marl/env/dynamics.py`) so the invariant is directly testable. Concrete
mechanics for parts the design left as prose:

- **Receivable acceleration** (liquidity action) pulls forward a fraction of every
  AR bucket at a discount (`early_pay_discount_frac`, config/env.yaml) *before* the
  normal collection/aging waterfall runs on what remains that week.
- **Credit-line draw/repay**: `credit_draw_frac >= 0` draws that fraction of
  remaining headroom; `< 0` repays that fraction of the current drawn balance
  (capped at the balance). Interest accrues weekly on the post-draw/repay balance.
- **Capital allocation surplus** = `max(0, cash - buffer_target)`, where
  `buffer_target = target_buffer_frac * buffer_days_target * (weekly fixed +
  discretionary expense) / 7` -- i.e. the liquidity agent's `target_buffer_frac`
  action scales the calibrated JPMorgan-anchor buffer, and that scaled target is
  what the capital-allocation agent's surplus is measured against (the coordination
  coupling ARCHITECTURE.md section 1 describes). `debt_paydown` is capped at the
  outstanding `term_debt` balance.
- **Solvency breach** = `(cash + remaining credit headroom) < 0` after this week's
  flows, i.e. even drawing the rest of the line could not cover the shortfall;
  `shortfall_severity = max(0, -cash)` is reported separately (continuous, for
  reward shaping) regardless of headroom.
- Credit-exposure income/loss (section 5) are computed by the environment (which
  owns the exposure pool/queue) and passed into `dynamics.step` as already-known
  dollar amounts, keeping this module free of the exposure object model.

---

## 4. Agents: observations, actions, rewards

Action spaces are kept low-dimensional and mostly discrete/bounded-continuous so PPO
trains reliably in a short budget. Reward weights live in `config/agents.yaml`.

### 4.1 Liquidity-forecasting agent
- Observes: cash, AR aging, AP due schedule, credit headroom, revenue forecast.
- Actions (bounded continuous, scaled): target buffer level; amount to draw/repay on
  the line; fraction of due payables to defer (0–1); receivable acceleration
  (offer early-pay discount, 0–1).
- Reward `R_liq`:
  `- w1 * shortfall_severity  - w2 * financing_cost  - w3 * idle_cash_penalty  - w4 * late_payment_penalty`.
  Rewards keeping the firm solvent at least cost, penalizing both running dry and
  hoarding.

### 4.2 Credit-risk assessment agent
- Observes: this week's `pending_credit_requests` (each a feature vector), current
  cash/headroom, exposure concentration.
- Actions (per request, discrete): `deny | approve | approve_with_premium`.
- The default probability of each request is **not** learned by this agent from
  scratch; it is supplied by the real SBA-trained model (§5). The agent learns a
  *policy over* those risk scores given the firm's liquidity and concentration
  state (e.g., approve fewer risky exposures when cash is tight).
- Reward `R_cred`:
  `+ margin_on_repaid  - loss_given_default_on_defaulted  - concentration_penalty`.

### 4.3 Expenditure-optimization agent
- Observes: discretionary budget, cash, upcoming obligations, season.
- Actions (bounded continuous): fraction of discretionary budget to spend now vs
  defer; split across essential-adjacent vs deferrable categories.
- Reward `R_exp`:
  `+ value_of_spend(diminishing returns)  - w * operational_penalty_for_over_deferral`.
  Over-deferral degrades a latent "operations health" term to prevent the trivial
  policy of never spending.

### 4.4 Capital-allocation agent
- Observes: surplus cash above buffer, debt level and rate, reserve, invested,
  expected returns.
- Actions (simplex over three choices): allocate surplus to {debt paydown, reserve,
  investment}; investment earns a stochastic risk-adjusted return.
- Reward `R_cap`:
  `+ risk_adjusted_return_on_invested  + interest_saved_on_paydown  - liquidity_risk_penalty_if_over_allocated`.

### 4.5 Shared reward (the coordination signal)
`R_shared = alpha * firm_value_change - beta * solvency_breach_indicator`
where `firm_value = cash + reserve + invested + AR_expected_collectible - debt`.
Each agent's total reward is `R_i + lambda * R_shared`. Setting `lambda > 0` and
using a centralized critic (MAPPO) is what lets agents trade off against each other
over the shared cash pool. The IPPO baseline uses `lambda = 0` and no shared critic,
isolating the value of coordination.

**Implementation (Phase B), `src/sof_marl/agents/rewards.py`.** `lambda` is
`shared_reward_weight`, a parameter of `SMETreasuryEnv.__init__` (default `0.0`),
not hardcoded here -- the same environment serves both the IPPO (`lambda = 0`) and
MAPPO (`lambda > 0`, `config/train.yaml` `mappo.shared_reward_weight`) arms in Phase
D. All dollar-valued reward terms are normalized by `cfg.revenue.weekly_mean` for
scale consistency across agents. Concrete proxies for terms the design left as
prose:

- **Idle-cash penalty** (liquidity, `R_liq`): `max(0, cash - 2 * buffer_target)`,
  i.e. only cash well beyond (2x) the agent's own chosen buffer target counts as
  idle, so this does not fight the capital-allocation agent's ordinary surplus use.
- **Credit-risk reward `R_cred` (redesigned after the Phase-E credit-agent finding;
  see reports/technical_report.md 5.3/7).** The initial POC used the realized
  margin/loss at the exposure's 12-week resolution, minus a per-week concentration
  penalty linear in total outstanding notional. That collapsed the credit agent to
  "always deny": a quantitative diagnosis (`scripts/diagnose_credit_reward.py`)
  showed the concentration penalty taxed holding *any* book at ~100x the margin it
  earned, and the realized signal was both delayed 12 weeks and too small to learn
  against `R_shared`. `R_cred` is now
  `credit_reward_scale * (decision_EV / weekly_revenue) - concentration`, where:
  - **`decision_EV`** is the *immediate* expected value of this week's approve/deny
    decisions, computed at decision time from the real model's predicted default
    probability `p`: for each approved exposure,
    `margin*(1-p) - w_lgd*LGD*p` (premium margin if approved with premium); a denial
    contributes 0. Dense, immediate, and directly incentivizes approving positive-EV
    exposures. Realized cash flows still resolve at 12 weeks and hit firm value
    (hence `R_shared`) unchanged -- only the credit agent's *own* learning signal
    moved to decision time.
  - **`credit_reward_scale`** (`config/agents.yaml`) lifts this signal to a magnitude
    learnable against `R_shared` (the agent-specific normalization).
  - **concentration** is now a *threshold* penalty: `w_concentration * max(0,
    outstanding - concentration_budget_frac * credit_line.limit) / weekly_revenue`,
    so prudent lending below the budget is free and only genuine over-concentration
    is penalized.
- **Operational penalty for over-deferral** (expenditure, `R_exp`):
  `w_over_deferral * (1 - ops_health)`, directly using the latent `ops_health` term,
  which mean-reverts toward 1.0 when an *effective* spend rate is at or above
  `ops_health_target_spend_rate` and decays otherwise
  (`ops_health_adjustment_rate`, both in config/env.yaml). The effective rate is
  `spend_now_frac * (0.5 + 0.5 * essential_bias)`: essential-adjacent spend
  (`essential_bias` near 1) protects operations fully per dollar, purely
  deferrable spend (`essential_bias` near 0) only half as much -- this is what
  the expenditure agent's second action dimension (ARCHITECTURE.md section 4.3)
  actually controls.
- **Liquidity-risk penalty** (capital allocation, `R_cap`): fires only when a naive,
  deterministic one-week-ahead cash projection (this week's ending cash plus
  expected revenue minus expected fixed and discretionary expense, ignoring
  collection lag) would be negative *and* the agent allocated surplus to
  investment anyway -- a forward-looking reward-shaping heuristic, not part of the
  accounted cash balance.

---

## 5. Real-data credit model (the anchor)

The credit agent's risk scores come from a supervised model trained on the SBA 7(a)
FOIA loan-outcome data (DATA.md). This is the one component grounded in real U.S.
SME outcomes and the POC's strongest evidentiary element.

- Target: binary charge-off (`CHGOFF`) vs paid-in-full (`PIF`); drop unresolved
  statuses.
- Features: loan amount, term, employees/jobs, industry (NAICS sector), borrower
  state, urban/rural, revolving-vs-term, new-vs-existing business, and other fields
  present in the FOIA dictionary. No leakage: exclude any field only known at or
  after resolution.
- Model: gradient-boosted trees (`xgboost`), probability-calibrated
  (`sklearn` isotonic/Platt). Report held-out ROC-AUC, PR-AUC, and a calibration
  curve. Time-based split (train on older cohorts, test on newer) to avoid temporal
  leakage; document the split.
- Explainability: `shap` summary plot over the test set + example per-request
  explanations. This directly addresses the interpretability concern raised in the
  independent literature about this architecture class and mirrors the petitioner's
  documented SHAP work at Deloitte.
- Integration: at environment build time, sample a pool of "exposures" from the
  held-out real records; each carries its true label and the model's predicted
  default probability. The environment reveals features + model score to the agent;
  resolution uses the true label. This keeps the credit dynamics tied to real data.

**Implementation (Phase B), `src/sof_marl/env/calibration.py` +
`sme_treasury_env.py`.** The Phase-A held-out test split (`data/processed/test.csv`)
is scored once by the calibrated model and cached
(`data/processed/credit_exposure_pool.csv`); each episode samples
`credit_exposure_pool_size` (50) rows without replacement, using the episode's own
RNG seed for reproducibility. Real SBA `gross_approval` amounts (median ~$100k in
the held-out split) are far larger than a single trade-credit exposure should be
relative to this firm's ~$23k/week revenue, so exposure notional is
`gross_approval * credit_exposure_scale_frac` (0.02; assumption, sensitivity-tested).
Each sampled exposure gets a uniformly random arrival week in `[0, horizon_weeks)`;
arrivals queue in FIFO order and up to `max_pending_credit_requests` (4) are
presented per week -- overflow simply queues to the following week, so no exposure
is dropped, only delayed. An accepted exposure resolves exactly
`exposure_resolution_weeks` (12) weeks later: if the true label is paid-in-full, the
firm earns `margin_bps` (`+ premium_bps` if `approve_with_premium`) of notional; if
charged-off, the firm loses `loss_given_default_frac` (0.6; typical commercial-LGD
assumption) of notional. Both `margin_bps`/`premium_bps` are in `config/agents.yaml`;
`credit_exposure_scale_frac` and `loss_given_default_frac` are in `config/env.yaml`.

Do not overstate this model. On the SBA dataset an honest AUC is typically in the
mid-0.6s to mid-0.7s. Report what you get.

---

## 6. Algorithms and the coordination experiment

Four policies, one environment, one metric suite:

1. **Rule-based** (`baselines/rule_based.py`): maintain a fixed buffer (calibrated
   to public median buffer days), pay payables on due date, draw the line only to
   avoid shortfall, approve credit below a fixed risk cutoff, split surplus by a
   fixed rule. A reasonable treasurer's heuristic, documented and defensible.
2. **Single-agent PPO** (`baselines/single_agent.py` + SB3): one agent observes the
   full state and emits the concatenated action vector. Tests whether decomposition
   into specialized agents helps at all.
3. **IPPO** (`training/multi_agent_ppo.py`): four independent per-agent PPO
   learners (own actor-critic each), `lambda = 0`, no centralized critic.
   Multi-agent but uncoordinated. (Not SuperSuit parameter-sharing -- see
   CLAUDE.md's amended locked decision: SuperSuit requires identical obs/action
   spaces across agents, which this environment's four heterogeneous agents
   cannot satisfy by design.)
4. **MAPPO** (`training/multi_agent_ppo.py`, same trainer as IPPO): four agents,
   centralized critic over joint observations, `lambda > 0`. This is the Exhibit
   C.2 design.

The headline finding is the **coordination lift**: MAPPO vs the best of
{rule-based, single-agent, IPPO} on the combined objective and on solvency. Predict
nothing; measure it. If MAPPO wins, that is the demonstrated merit. If it ties or
loses, report it honestly and discuss why (the report is still valid demonstrated
progress; a null result cleanly reported is credible, a faked win is fatal).

Training hygiene: fixed seed list in `train.yaml` (≥5), identical env seeds across
policies for evaluation, logged learning curves, early-stopping on a validation
metric, and a documented, small hyperparameter grid only.

**Implementation (Phase C).** `baselines/rule_based.py`'s fixed parameters (buffer
multiplier, risk cutoff, surplus split, spend rate) are named, documented constants
in `config/baselines.yaml`, none tuned against an evaluation result -- the risk
cutoff (0.15) is set relative to the real held-out portfolio base rate (9.74%,
`reports/results/credit_metrics.json`), consistent with EVALUATION.md's requirement
that this baseline not be a strawman. It reads `env.state`/`env.cfg` directly rather
than the normalized observation vectors (a real treasurer works from the books, not
a feature encoding), while the learned policies are properly observation-limited.

`baselines/single_agent.py` wraps `SMETreasuryEnv` as a single `gymnasium.Env` by
concatenating all four agents' observation vectors and combining their action
spaces into one `Box`; SB3's PPO requires a single homogeneous action space, but the
four agents' native spaces mix continuous (`Box`) and discrete (`MultiDiscrete`)
types, so the credit-risk agent's four decisions are represented as continuous
scores in `[0, 1]` and discretized on each step (`< 1/3` deny, `< 2/3` approve, else
approve_with_premium). Its reward is the unweighted sum of the four base per-function
rewards (`shared_reward_weight = 0`): a single controller that already sees and
decides everything has no separate identity for a coordination term to act against.

---

## 7. Configuration surface

Everything a reviewer might question is a named, sourced constant in `config/`:
buffer targets, seasonality, volatility, collection curve, default resolution
schedule, reward weights, discount, training budget, seeds. No magic numbers inside
`src/`. `calibration.py` loads `env.yaml` and exposes typed accessors. Every
calibration constant carries a comment citing its source (DATA.md).

**Implementation (Phase B).** Beyond the constants already listed in `DATA.md`,
building a working weekly simulation required a handful of additional named
constants with no direct public-statistics source, added to `config/env.yaml` and
tagged `assumption` (candidates for the sensitivity analysis, EVALUATION.md
section 3, alongside revenue volatility and the buffer target): `ap_schedule_weeks`,
`early_pay_discount_frac`, `credit_exposure_scale_frac`, `loss_given_default_frac`,
`max_pending_credit_requests`, `ops_health_target_spend_rate`,
`ops_health_adjustment_rate`. `agents/spaces.py` defines the four agents'
`gymnasium` observation/action spaces (dimensions and action ordering match
`config/agents.yaml`); `agents/rewards.py` loads `config/agents.yaml` into a typed
`AgentsConfig` and computes `R_liq`, `R_cred`, `R_exp`, `R_cap`, and `R_shared`.
