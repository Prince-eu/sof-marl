# CLAUDE.md — build instructions for this repository

You are implementing **SOF-MARL**, a proof-of-concept of the multi-agent
reinforcement-learning autonomous-finance architecture in Enyiorji (2025). Read the
full specifications in `docs/` before writing code, in this order:
`docs/SKILL.md` (master guide), `docs/ARCHITECTURE.md` (design contract),
`docs/DATA.md` (data + calibration), `docs/EVALUATION.md` (metrics + honesty),
`docs/ROADMAP.md` (order of work), `docs/technical_report.md` (the report you will
fill with real numbers at the end).

## Non-negotiable quality bar

This code becomes an evidentiary exhibit in a federal filing. It must read as the
work of a competent ML engineer, not a demo.

- Real, cited public data for the credit model (SBA 7(a) FOIA). No invented numbers.
- The treasury environment is a **calibrated simulation** and is labeled as such in
  every figure caption and report sentence that uses it.
- Report **mean +/- std over >= 5 seeds**, never a single run. Report null or negative
  results honestly. A cleanly reported small effect is fine; a fabricated or
  cherry-picked one is disqualifying.
- The repo runs end to end from a clean checkout via `scripts/reproduce.sh`.
- Typed, documented, tested, formatted. `ruff`, `black`, `mypy`, `pytest` all green.
  No dead code, no leftover TODOs, no placeholder comments in the final state.
- If a claim cannot be made honestly, cut the claim, not the corner.

## Locked decisions (do not silently change; if you must, update docs first)

- **Combined objective (headline):** `J = r - kappa*s - lambda*B`, with
  `r` = (V_52 - V_0)/V_0, `s` = stdev(weekly change in V)/V_0,
  `B` = weeks with uncovered shortfall / 52. Weights are **balanced**:
  `lambda = 3.0`, `kappa = 1.0`. Set in `config/train.yaml`. Financing cost is a
  reported diagnostic, not a term in J (already inside V_52).
- **Pre-registered scale check:** after the env is built, verify on the rule-based
  baseline (before any RL result) that no term of J dominates by an order of
  magnitude; adjust weights at most once, record that it was pre-results, freeze.
- **MARL path:** Stable-Baselines3 directly (no SuperSuit) for the single-agent
  baseline (`baselines/single_agent.py`). **Amended during Phase D build:**
  SuperSuit's parameter-sharing (`MarkovVectorEnv`, `pad_action_space_v0`) requires
  every agent to share one identical observation *and* action space, and its own
  padding utilities only bridge Box-with-Box or Discrete-with-Discrete ("not a
  mix") -- confirmed by reading its source. The four agents are heterogeneous by
  ARCHITECTURE.md's own design (different obs dims; a genuine mix of continuous
  Box and discrete MultiDiscrete actions), so literal SuperSuit parameter-sharing
  cannot apply; forcing it would mean flattening the agents into an artificial
  common space for no real benefit. IPPO and MAPPO are therefore both a single
  compact custom multi-agent PPO trainer in `src/sof_marl/training/` (PyTorch,
  reusing Stable-Baselines3's `ActorCriticPolicy`/`RolloutBuffer` building blocks
  where convenient): four independent per-agent actor-critics for IPPO
  (`shared_reward_weight = 0`, per-agent critic), the same four actors plus one
  centralized critic over joint observations for MAPPO
  (`shared_reward_weight > 0`). Do **not** add Ray/RLlib; keep the dependency set
  lean.
- **Credit model:** XGBoost on SBA 7(a) FOIA data, probability-calibrated, evaluated
  with time-based split, ROC-AUC + PR-AUC + calibration + bootstrap CIs, plus SHAP.
- **Orchestration layer:** out of scope for now (coordination lives in MAPPO's shared
  reward + centralized critic). Leave the seam clean for a future `orchestration/`
  module but do not build it.

## Build order (see docs/ROADMAP.md for the day-by-day and the fallback ladder)

1. **Phase A — real result first.** `data/download_sba.py` (URL is in
   `config/env.yaml`, verify it resolves) then `src/sof_marl/credit/`:
   `build_dataset.py` (leakage-checked features, time split), `train_credit.py`
   (XGBoost + calibration + metrics), `explain.py` (SHAP). This alone is a genuine
   exhibit; do it fully before touching the RL.
2. **Phase B — environment.** `src/sof_marl/env/` per ARCHITECTURE.md; enforce the
   cash-conservation invariant as a test; wire the credit agent to Phase-A scores.
3. **Phase C — baselines.** Rule-based and single-agent (fair, documented).
4. **Phase D — MARL.** IPPO and MAPPO; >= 5 seeds.
5. **Phase E — evaluation.** Shared eval episodes, the full metric suite, coordination
   lift with a paired test, the four figures.
6. **Phase F — report.** Fill `docs/technical_report.md` (or copy to
   `reports/technical_report.md`) with real numbers; convert to PDF.

## Running

`make setup` install deps. `make data` fetch SBA. `make credit-model` Phase A.
`make train` Phase D. `make eval` Phase E. `make report` build the PDF.
`make check` runs ruff + black --check + mypy + pytest. `scripts/reproduce.sh` runs
the whole pipeline and regenerates the headline numbers.

## Attribution and honesty

This is Prince Enyiorji's architecture (Exhibit C.2) and endeavor; he directs and owns
the build. AI coding assistance is ordinary and fine, but nothing in the report may
claim more than was actually done, and every reported number must regenerate from a
clean checkout.
