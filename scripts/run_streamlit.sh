#!/usr/bin/env bash
set -euo pipefail

# Launch Streamlit dashboard with repo on PYTHONPATH.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
APP_FILE="${REPO_ROOT}/data_engine/market_ai/ui/app.py"
PYTHON="${PYTHON:-${VENV_PYTHON:-python3}}"

if [[ ! -f "$APP_FILE" ]]; then
  echo "Streamlit app not found at $APP_FILE" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/data_engine:${PYTHONPATH:-}"
HOST="${HOST:-localhost}"
PORT="${PORT:-8501}"

exec "$PYTHON" -m streamlit run "$APP_FILE" \
  --server.address="$HOST" \
  --server.port="$PORT"
