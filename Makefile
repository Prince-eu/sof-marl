.PHONY: setup data credit-model train eval report check clean

# The editable install (`pip install -e .`) is unreliable on some Python builds
# (observed: Homebrew python@3.11 framework build silently drops the src/ path
# injection). Set PYTHONPATH explicitly so `python -m sof_marl...` and pytest
# resolve the package the same way regardless of editable-install behavior.
export PYTHONPATH := src

setup:
	python3.11 -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -r requirements.txt && pip install -e .

data:
	python data/download_sba.py

credit-model:
	python -m sof_marl.credit.build_dataset
	python -m sof_marl.credit.train_credit
	python -m sof_marl.credit.explain

train:
	python -m sof_marl.training.train_single_agent
	python -m sof_marl.training.train_ippo
	python -m sof_marl.training.train_mappo

eval:
	python -m sof_marl.evaluation.rollout
	python -m sof_marl.evaluation.sensitivity
	python -m sof_marl.evaluation.figures

report:
	@echo "reports/technical_report.md is filled with real results. Convert to PDF"
	@echo "(e.g. pandoc reports/technical_report.md -o reports/technical_report.pdf)."

check:
	ruff check src tests
	black --check src tests
	mypy src
	pytest

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
