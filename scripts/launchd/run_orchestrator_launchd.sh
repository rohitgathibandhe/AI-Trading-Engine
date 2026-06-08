#!/usr/bin/env bash
# KeepAlive launcher for the regime-aware orchestrator (start_agent.py).
# Deploys the weekly Batman BKM strategy (15:16 IST) + the intraday-AI path.
#
# The UI server (paper_pnl_server.py) can reset last_strategy.json to the v83
# selection, which makes start_agent.py refuse to start ("requires
# strategy_file='batman_bkm_monthly'"). Force the correct selection on every
# (re)launch so a KeepAlive restart is self-healing.
set -euo pipefail

ROOT="/Users/Rohit/AI-Trading-Engine"
STATE="$ROOT/data_engine/market_ai/state"
PY="/Users/Rohit/.venvs/data_engine_py310/bin/python3"

printf '{\n  "strategy_file": "batman_bkm_monthly"\n}\n' > "$STATE/last_strategy.json"

cd "$ROOT/data_engine"
export PYTHONPATH="$ROOT/data_engine"
export PYTHONDONTWRITEBYTECODE=1
export TRADE_MODE="${TRADE_MODE:-paper}"
export AGENT_SETTINGS_JSON="$STATE/agent_settings.json"

exec "$PY" -u market_ai/start_agent.py
