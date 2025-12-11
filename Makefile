# Makefile for the grokking-multitask experiments.
# All targets assume the project root as the working directory.

PY      := ./.venv/bin/python
PYPATH  := PYTHONPATH=.

.PHONY: install test smoke replicate multitask analysis figures all clean clean-runs paper

install:
	python3 -m venv .venv
	./.venv/bin/pip install --upgrade pip
	./.venv/bin/pip install -r requirements.txt

test:
	$(PYPATH) $(PY) tests/test_pipeline.py

smoke:
	$(PYPATH) $(PY) -c "from data.tasks import make_task; from src.train import TrainConfig, train; \
		d = make_task('+', p=23, train_frac=0.5, seed=0); \
		c = TrainConfig(name='_smoke', ops=('+',), p=23, train_frac=0.5, seed=0, \
		                steps=1000, log_every=500, eval_every=500, save_dir='/tmp/grokking_smoke'); \
		train(c, d)"

# Single-task baselines.
replicate:
	$(PYPATH) $(PY) experiments/01_baseline_addition.py
	$(PYPATH) $(PY) experiments/02_baseline_subtraction.py
	$(PYPATH) $(PY) experiments/03_baseline_multiplication.py

# Multi-task experiments + seed sweep.
multitask:
	$(PYPATH) $(PY) experiments/04_multitask_two.py
	$(PYPATH) $(PY) experiments/05_multitask_three.py
	$(PYPATH) $(PY) experiments/07_seed_sweep.py

analysis:
	$(PYPATH) $(PY) experiments/06_curriculum.py
	$(PYPATH) $(PY) experiments/08_robustness_sweeps.py

# Generate every figure and table from the saved metric histories.
figures:
	$(PYPATH) $(PY) experiments/plot_results.py

# Build the paper PDF (requires tectonic).
paper:
	cd paper && tectonic main.tex

# Run everything end-to-end.
all:
	$(PYPATH) $(PY) experiments/run_all.py
	$(PYPATH) $(PY) experiments/plot_results.py
	cd paper && tectonic main.tex

clean:
	rm -rf paper/main.aux paper/main.log paper/main.out paper/main.pdf paper/main.bbl
	find . -name __pycache__ -type d -exec rm -rf {} +

clean-runs:
	rm -rf runs/* checkpoints/*
