SHELL := /bin/bash
PYTHON := python

.PHONY: install lint test bench-smoke format typecheck

install:
	$(PYTHON) -m pip install -r requirements.txt

lint:
	$(PYTHON) -m ruff check --fix src tests

format:
	$(PYTHON) -m ruff format src tests

typecheck:
	$(PYTHON) -m mypy src

test:
	PYTHONPATH=src $(PYTHON) -m pytest

bench-smoke:
	$(PYTHON) -m kvbench.cli bench-kv-scaling --config configs/bench/kv_scaling.yaml
