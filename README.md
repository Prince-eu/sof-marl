# SOF-MARL

A proof-of-concept implementation of the self-optimizing, cloud-native autonomous
finance architecture for SMEs described in Enyiorji (2025), *Designing a
Self-Optimizing Cloud-Native Autonomous Finance System for SMEs Using Multi-Agent
Reinforcement Learning*.

Four cooperating agents (liquidity forecasting, credit-risk assessment, expenditure
optimization, capital allocation) manage an SME's weekly treasury in a shared
environment. The credit-risk function is grounded in a model trained on real public
U.S. SME loan-outcome data (SBA 7(a) FOIA). The headline experiment tests whether
coordinated multi-agent control (MAPPO) outperforms a rule-based heuristic, a
single-agent controller, and independent multi-agent RL (IPPO).

> The treasury environment is a **calibrated simulation** and is labeled as such
> wherever it appears. The credit-risk model is trained on **real** public data. See
> `docs/` for the full specifications.

## Result

`reports/technical_report.pdf` is the full exhibit. Headline: on 100 shared,
held-out evaluation episodes, MAPPO increased the pre-registered combined treasury
objective by 9.45-9.51 over every other policy (mean over 5 training seeds, paired
Wilcoxon p < 0.001), stable across six calibration perturbations. The credit-risk
model achieves a held-out ROC-AUC of 0.939 (0.622 on a feature-ablated, more
conservative reading; see the report for why both numbers are reported). One
important limitation, investigated and disclosed rather than hidden: the
credit-risk agent converged to denying nearly every exposure under all three
learned policies, so the measured MAPPO lift is attributable to the other three
agents coordinating well, not to better credit decisions -- see report sections
5.3 and 7.

## Quickstart

```bash
make setup            # venv + pinned deps + editable install
make data             # download the SBA 7(a) FOIA snapshot (config/env.yaml sba.csv_url)
make credit-model     # Phase A: real credit-risk model + SHAP
make train            # Phase D: single-agent PPO + IPPO + MAPPO, 5 seeds each
make eval             # Phase E: evaluation + sensitivity analysis + figures
make check            # ruff + black + mypy + pytest
```

Or run everything in one command: `bash scripts/reproduce.sh` (~2.5 hours, mostly
`make train`).

## Layout

```
config/    env, agents, and training configuration (objective weights pre-registered)
data/      SBA download script (raw/processed are git-ignored)
src/sof_marl/
  env/         PettingZoo treasury environment + dynamics + calibration
  agents/      per-agent observation/action spaces and rewards
  credit/      SBA dataset build, XGBoost training, SHAP explanations
  baselines/   rule-based and single-agent policies
  training/    custom multi-agent PPO trainer (PyTorch) serving both IPPO and MAPPO
  evaluation/  rollouts, metrics, sensitivity analysis, figures
tests/     env API conformance, accounting invariant, reward/dataset/policy checks
reports/   technical_report source + PDF, figures, and result tables (JSON)
docs/      SKILL, ARCHITECTURE, DATA, EVALUATION, ROADMAP, technical_report scaffold
```

## Build guide

Start with `CLAUDE.md`, then `docs/SKILL.md`. The combined objective is
`J = r - 1.0*s - 3.0*B` (return, minus normalized volatility, minus solvency-breach
rate), pre-registered in `config/train.yaml`. Results are reported as mean and
standard deviation over at least five seeds, with limitations stated plainly.
`docs/technical_report.md` is the unfilled scaffold this build guide describes;
`reports/technical_report.md` (and its PDF) is the filled exhibit with the real
numbers above.

Author: Prince Enyiorji.
