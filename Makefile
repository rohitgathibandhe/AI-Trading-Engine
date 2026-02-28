PYTHON ?= /Users/Rohit/.venvs/data_engine_py310/bin/python
PYTEST ?= PYTHONPATH=data_engine $(PYTHON) -m pytest

.PHONY: test-fast test-all test-cov ui-start ui-restart ui-stop ui-status ui-health ui-logs

test-fast:
	$(PYTEST) -m "unit or integration" -q

test-all:
	$(PYTEST) -m "unit or integration or backtest" -q

test-cov:
	$(PYTEST) -m "unit or integration" --cov=data_engine --cov-report=term-missing

ui-start:
	./scripts/launchd/ui_server_ctl.sh start --port 8000

ui-restart:
	./scripts/launchd/ui_server_ctl.sh restart --port 8000

ui-stop:
	./scripts/launchd/ui_server_ctl.sh stop

ui-status:
	./scripts/launchd/ui_server_ctl.sh status --port 8000

ui-health:
	./scripts/launchd/ui_server_ctl.sh health --port 8000

ui-logs:
	./scripts/launchd/ui_server_ctl.sh logs
