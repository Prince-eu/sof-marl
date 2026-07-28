---
name: sof-marl-prototype
description: >
  Build guide for SOF-MARL, a proof-of-concept implementation of the
  Self-Optimizing Cloud-Native Autonomous Finance System for SMEs described in
  Enyiorji (2025) [Exhibit C.2]. Use this when implementing, training, evaluating,
  or documenting the multi-agent reinforcement-learning treasury prototype in
  Claude Code. Covers repository scaffolding, the multi-agent environment, the
  four agents, the real-data credit-risk model (SBA 7(a) FOIA data), the
  coordination experiment, evaluation against baselines, and the technical report
  that becomes the petition's demonstrated-progress exhibit.
---

# SOF-MARL Prototype — Master Build Guide

## 0. Read this first

This skill drives the build of a **credible proof-of-concept (POC)**, not a toy and
not a production system. It implements the architecture documented in the
petitioner's sole-authored paper, *Designing a Self-Optimizing Cloud-Native
Autonomous Finance System for SMEs Using Multi-Agent Reinforcement Learning*
(Exhibit C.2), at a scope that one engineer can build, train, and document in a
few days, and that stands up to a technically literate reader.

The POC exists to satisfy deficiency **D-1** (no demonstrated progress on the
proposed system) and to strengthen **Prong 1 substantial merit** by moving it from
"recognized by others" to "demonstrated to work." It also converts the Prong 2
plan of action from planned to in-progress.

**Quality bar (non-negotiable).** The reviewer of the resulting exhibit may be a
skilled adjudicator, an RFE-writing officer, or opposing technical counsel. The
work must read as the output of a competent ML engineer:

- Real, cited public data for the credit-risk model. No invented numbers.
- The simulation is disclosed as a calibrated simulation, everywhere it appears.
- Results are reported with mean and standard deviation over multiple seeds, with
  limitations stated plainly. No cherry-picking, no rounding up, no "up to X%."
- The repository runs end to end from a clean checkout with one command.
- Code is typed, documented, tested, and formatted. No dead code, no notebook
  sprawl, no placeholder comments left in.

If any step cannot be done honestly, cut the claim, not the corner. A smaller
true result beats a larger unverifiable one.

## 1. What is being built

A multi-agent reinforcement-learning system that manages the weekly treasury
decisions of a small or medium enterprise. Four cooperating agents, one per
function in Exhibit C.2, act on a shared financial state:

1. **Liquidity-forecasting agent** — keeps the firm solvent at least cost.
2. **Credit-risk assessment agent** — decides which trade-credit / financing
   exposures to accept, scored by a model trained on real SBA loan-outcome data.
3. **Expenditure-optimization agent** — times and prioritizes discretionary spend.
4. **Capital-allocation agent** — deploys surplus across debt paydown, reserve,
   and investment.

The scientific claim the POC must actually demonstrate is the thesis of C.2:
**coordinated multi-agent control outperforms the alternatives.** The headline
experiment compares, on the same environment and metrics:

- a rule-based treasury policy (industry-standard heuristics),
- a single-agent RL controller that owns all decisions,
- independent multi-agent RL (IPPO, no coordination signal), and
- coordinated multi-agent RL (MAPPO / shared-objective, the C.2 design).

If coordination does not help, report that honestly — but the design below is
built so that coordination has a real mechanism to exploit (shared solvency
constraint, competing objectives over one cash pool).

Full technical spec: **ARCHITECTURE.md**.

## 2. Scope boundaries (what is in, what is out)

In scope:
- One representative SME profile class, calibrated to public statistics.
- 52 weekly steps per episode (one fiscal year).
- Four agents; the credit agent grounded in a real SBA-data model.
- The four-way baseline comparison above.
- SHAP explanations for the credit model (interpretability is a first-class
  result, not an afterthought — it answers the exact critique an independent
  citing paper raised about this architecture, and mirrors the petitioner's
  documented Deloitte work).
- A written technical report with figures.

Out of scope (state these as future work in the report, do not fake them):
- Real customer or bank data beyond the public SBA dataset.
- Live deployment, streaming data, or a UI.
- Regulatory certification, PII handling, production MLOps.
- Hyperparameter searches beyond a small, documented grid.

## 3. Tech stack

Pin exact versions in `requirements.txt`; do not float. Recommended baseline:

- Python 3.11
- `pettingzoo` — multi-agent environment API
- `stable-baselines3` — PPO building blocks; used directly for the single-agent
  baseline. SuperSuit's parameter-sharing was evaluated and dropped (CLAUDE.md's
  amended locked decision): it requires every agent to share one identical
  observation and action space, which this environment's four heterogeneous
  agents cannot satisfy by design (a genuine mix of continuous and discrete
  actions). IPPO and MAPPO are instead one compact custom multi-agent PPO trainer
  in `src/sof_marl/training/` (PyTorch), not Ray/RLlib.
- `gymnasium` — single-agent baseline env wrapper
- `numpy`, `pandas`, `scipy` — data and simulation
- `scikit-learn`, `xgboost` — credit-risk model
- `shap` — credit-model explanations
- `matplotlib` — figures (no seaborn styling gimmicks; clean, labeled axes)
- `pyyaml` / `pydantic` — typed configuration
- Tooling: `pytest`, `ruff`, `black`, `mypy`, `pip-tools`

Choose one MARL path (RLlib MAPPO **or** the self-contained MAPPO) and remove the
other from the dependency list. Do not ship both half-wired.

## 4. Repository layout

```
sof-marl/
  README.md                 # what it is, how to run, headline result, honest caveats
  LICENSE                   # MIT for the code
  requirements.txt          # pinned
  Makefile                  # make setup | data | credit-model | train | eval | report
  pyproject.toml            # ruff/black/mypy config
  config/
    env.yaml                # environment + calibration constants (sourced, see DATA.md)
    agents.yaml             # action spaces, reward weights
    train.yaml              # algorithm, seeds, timesteps
  data/
    download_sba.py         # fetch SBA 7(a) FOIA csv -> data/raw/
    raw/                    # .gitignored
    processed/              # .gitignored
  src/
    sof_marl/
      __init__.py
      env/
        sme_treasury_env.py # PettingZoo ParallelEnv
        dynamics.py         # cash-flow, receivables/payables, financing
        calibration.py      # loads env.yaml, exposes sampled SME profiles
      agents/
        spaces.py           # per-agent observation/action spaces
        rewards.py          # per-agent + shared reward terms
      credit/
        build_dataset.py    # SBA raw -> features + charge-off label
        train_credit.py     # xgboost + calibration; saves model + metrics
        explain.py          # SHAP summary + per-decision explanations
      baselines/
        rule_based.py       # heuristic treasury policy
        single_agent.py     # gymnasium wrapper: one agent, all actions
      training/
        train_ippo.py
        train_mappo.py      # or rllib_mappo.py
      evaluation/
        rollout.py          # deterministic eval rollouts, fixed eval seeds
        metrics.py          # solvency, credit, value, coordination-lift
        figures.py          # learning curves, lift bars, ROC+SHAP, trajectories
  reports/
    technical_report.md     # the exhibit source (convert to PDF at the end)
    figures/                # generated, committed
    results/                # generated tables (csv/json), committed
  tests/
    test_env_api.py         # PettingZoo API conformance
    test_dynamics.py        # accounting identities hold every step
    test_rewards.py
    test_credit_dataset.py
  scripts/
    reproduce.sh            # one command: data -> model -> train -> eval -> report
```

Keep notebooks out of the deliverable. If you explore in a notebook, port the
kept logic into `src/` and delete the notebook.

## 5. Build phases

Work in this order; each phase ends with something real and testable. Detailed
specs live in the referenced notes.

**Phase A — Scaffold + real-data anchor (do this first).**
Stand up the repo, tooling, CI-style `make` targets. Then build the credit-risk
model on the SBA 7(a) FOIA data end to end: `download_sba.py` → `build_dataset.py`
→ `train_credit.py` → `explain.py`. This produces the POC's one unambiguously real
result (held-out AUC, calibration curve, SHAP summary) before any RL exists, so
even under worst-case time pressure there is a genuine artifact. See **DATA.md**.

**Phase B — Environment + accounting.**
Implement `sme_treasury_env.py` as a PettingZoo `ParallelEnv` with the state,
per-agent action spaces, and dynamics from **ARCHITECTURE.md**. Enforce accounting
identities as tests (cash_{t+1} equals cash_t plus inflows minus outflows, no
value created from nothing). Wire the credit agent's decisions to the Phase-A
model's default probabilities.

**Phase C — Baselines.**
Implement the rule-based policy and the single-agent gymnasium wrapper. These must
be genuinely reasonable, not straw men — a weak baseline invalidates the whole
comparison. Document the heuristics and their basis.

**Phase D — MARL training.**
Train IPPO (independent) and MAPPO (coordinated). Use the shared/eval seeds from
`train.yaml`. Log learning curves. Keep training budgets modest but sufficient for
convergence; report wall-clock and hardware.

**Phase E — Evaluation.**
Run all four policies through the same evaluation rollouts (fixed eval seeds,
held-out episodes). Compute the metric suite in **EVALUATION.md**, over ≥5 training
seeds, reporting mean ± std. Generate the figure set.

**Phase F — Report.**
Write `technical_report.md`: system overview, data and calibration with sources,
method, results (tables + figures), interpretability, limitations, reproducibility
statement, and author attribution (the petitioner as architect/lead; note any
tools used). Convert to PDF. This file is the exhibit.

Day-by-day allocation and a minimum-viable-result fallback: **ROADMAP.md**.

## 6. How it enters the petition

The POC yields, for the filing:
- `reports/technical_report.pdf` — the demonstrated-progress exhibit.
- The repository as an appendix or an accessible archive (zip or a link; if a
  link, ensure it resolves and is stable).
- Two or three figures suitable for inline reference.

In the brief, it supports:
- **Prong 1 (merit):** the specific architecture has been implemented and produces
  measured results on public U.S. SME data — merit demonstrated, not only cited.
- **Prong 2 (well-positioned / plan of action):** the foundation and build stages
  are underway with concrete output, not merely proposed.

Draft the petition language from the report's actual results **after** the run.
Do not write results-dependent sentences before the numbers exist. Every figure
cited in the brief must exist in `reports/figures/` and say what the brief says.

## 7. Honesty and attribution guardrails (mandatory)

- The environment is a **calibrated simulation**. Say so in the report abstract,
  in every figure caption that uses simulated data, and in the petition text.
- The SBA credit result is **real**; report the true held-out metric even if
  modest (an AUC around 0.65–0.75 is a normal, honest result on this dataset —
  do not inflate it).
- Report **mean ± std over seeds**; never a single lucky run.
- State limitations explicitly: single SME profile class, simulated dynamics,
  proxy default labels, POC-scale training.
- Attribution: this is the petitioner's architecture (Exhibit C.2) and endeavor;
  he directs and owns the build. If AI coding assistance is used, that is fine and
  ordinary, but nothing in the report may claim more than was actually done.
- Reproducibility is part of honesty: pinned deps, fixed seeds, `scripts/reproduce.sh`,
  and a stated environment. A reader must be able to regenerate the headline number.

Companion notes: **ARCHITECTURE.md** (design), **DATA.md** (data + calibration),
**EVALUATION.md** (baselines, metrics, honest reporting), **ROADMAP.md** (schedule
and fallback).
