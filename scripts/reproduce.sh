#!/usr/bin/env bash
# Full pipeline: setup -> data -> credit model -> MARL training -> evaluation ->
# sensitivity analysis -> figures. Regenerates every number reported in
# reports/technical_report.md, from a clean checkout, with one command.
#
# Warning: this trains single-agent PPO, IPPO, and MAPPO for 5 seeds each at the
# full 1,000,000-timestep budget (config/train.yaml). Expect roughly 2.5 hours of
# wall-clock on the hardware this was authored against (see technical_report.md
# section 4.4) -- most of that is the `make train` step.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  make setup
fi
# shellcheck disable=SC1091
source .venv/bin/activate
# The editable install (`pip install -e .`) is unreliable on some Python builds
# (see Makefile); export PYTHONPATH so the direct `python -m` calls below resolve
# the package the same way the `make` targets do.
export PYTHONPATH=src

make data
make credit-model
make train
python -m sof_marl.evaluation.rollout
python -m sof_marl.evaluation.sensitivity
python -m sof_marl.evaluation.figures

echo "Pipeline complete. Real numbers are in reports/results/*.json and reports/figures/*.png."
