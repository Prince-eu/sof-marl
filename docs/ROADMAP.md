# ROADMAP — building this before you file

You intend to file this week. A full four-agent MARL system with a clean coordination
result is ambitious on that timeline, so this plan is ordered by **evidentiary value
per hour**: the real-data result comes first and is self-sufficient, and each later
phase adds strength but is not required for the exhibit to be genuine. If you run out
of time at any checkpoint, you stop there and document exactly what exists — nothing
in the report ever describes work that was not done.

Assume ~3–4 focused working days. Adjust to your actual availability.

---

## Day 1 — Scaffold + the real result

- Repo scaffold, tooling (`ruff`/`black`/`mypy`/`pytest`), `Makefile`, pinned
  `requirements.txt`, `README` skeleton.
- `data/download_sba.py`: pull one SBA 7(a) FOIA CSV; record snapshot name, date,
  row count, checksum.
- `credit/build_dataset.py` + `train_credit.py`: label, leakage-checked features,
  time-based split, XGBoost, calibration, held-out ROC-AUC / PR-AUC with bootstrap CI.
- `credit/explain.py`: SHAP summary + two example explanations.
- **Checkpoint 1 (minimum viable exhibit):** a real, reproducible credit-risk result
  on public U.S. SME data with explanations. If everything else fell through, this
  alone is honest demonstrated progress on the credit-assessment function of the C.2
  system.

## Day 2 — Environment + baselines

- `env/sme_treasury_env.py` (PettingZoo ParallelEnv), `dynamics.py`, `calibration.py`
  loading `env.yaml` (every constant sourced per DATA.md).
- Wire the credit agent to the Day-1 model's probabilities and real labels.
- Tests: PettingZoo API conformance; accounting invariant holds every step.
- `baselines/rule_based.py` and `baselines/single_agent.py` (documented, fair).
- **Checkpoint 2:** a working, tested simulation with two credible baselines and the
  real credit model embedded. Already a substantive artifact.

## Day 3 — MARL + evaluation

- `training/train_ippo.py` (SB3 + SuperSuit, parameter-shared) and
  `training/train_mappo.py` (RLlib or the compact MAPPO).
- `evaluation/rollout.py`, `metrics.py`, `figures.py`.
- Train ≥5 seeds each within a modest budget; run all four policies through the same
  eval episodes; compute the metric suite and the coordination lift with a paired
  test.
- **Checkpoint 3:** the headline experiment exists, whatever it shows.

## Day 4 — Report + reproducibility

- `reports/technical_report.md`: overview, data + calibration with sources, method,
  results (tables + the four figures), interpretability, limitations, reproducibility
  statement, author attribution. Convert to PDF.
- `scripts/reproduce.sh` verified from a clean checkout; regenerate the headline
  number.
- Polish: remove dead code, run `ruff`/`black`/`mypy`/`pytest` green, finalize README.
- **Checkpoint 4:** filing-ready exhibit.

---

## Minimum-viable-result ladder (if time runs short)

Each rung is a genuine, defensible exhibit on its own. Stop at the highest rung you
reach and document only that.

1. **Rung 1 (Day 1):** real SBA credit-risk model + SHAP. "The credit-assessment
   component of the proposed system, implemented and validated on public U.S. SME
   loan data."
2. **Rung 2 (Day 2):** + working multi-agent treasury simulation with the credit
   model embedded and rule-based/single-agent baselines. "The architecture is
   implemented; the credit function is grounded in real data."
3. **Rung 3 (Day 3):** + trained IPPO/MAPPO and the coordination comparison.
   "Coordinated multi-agent control implemented and evaluated against baselines."
4. **Rung 4 (Day 4):** + full report and reproducibility. Filing-ready.

## A candid note on timing

If the week is genuinely too tight for Rung 3–4 done honestly, it is better to file
with a Rung 1–2 exhibit plus the plan-of-action narrative than to rush a shaky MARL
result. The petition does not need a large coordination win; it needs real,
demonstrated progress on the actual system. A modest, true, reproducible artifact
strengthens the case. A hurried, fragile, or overstated one is worse than none,
because a filing puts your credibility on the record. Build to the rung you can
defend, and describe it exactly.

If you want, I can also help build any of these phases directly in a sandbox under
your direction, or draft the `reports/technical_report.md` scaffold so it is ready to
fill with the run's actual numbers.
