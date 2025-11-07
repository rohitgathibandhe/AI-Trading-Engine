PYTHON ?= /Users/Rohit/.venvs/data_engine_py310/bin/python
PYTEST ?= PYTHONPATH=data_engine $(PYTHON) -m pytest

.PHONY: test-fast test-all test-cov

test-fast:
	$(PYTEST) -m "unit or integration" -q

test-all:
	$(PYTEST) -m "unit or integration or backtest" -q

test-cov:
	$(PYTEST) -m "unit or integration" --cov=data_engine --cov-report=term-missing
