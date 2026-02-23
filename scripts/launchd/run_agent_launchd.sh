#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY_BIN="$PYTHON_BIN"
elif [[ -x "$ROOT_DIR/.venv/bin/python3" ]]; then
  PY_BIN="$ROOT_DIR/.venv/bin/python3"
else
  PY_BIN="$(command -v python3)"
fi

if [[ -z "${TRADE_MODE:-}" ]]; then
  export TRADE_MODE="live"
fi
export PYTHONDONTWRITEBYTECODE=1

exec "$PY_BIN" "$ROOT_DIR/data_engine/market_ai/start_agent.py"

