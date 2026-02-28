#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
INSTALL_SCRIPT="${ROOT_DIR}/scripts/launchd/install_launchagents.sh"
LABEL="com.rohit.ai-trading.ui"
UID_NUM="$(id -u)"
PLIST="${HOME}/Library/LaunchAgents/${LABEL}.plist"
LOG_STDOUT="${ROOT_DIR}/data_engine/market_ai/state/launchd_ui.stdout.log"
LOG_STDERR="${ROOT_DIR}/data_engine/market_ai/state/launchd_ui.stderr.log"

CMD="${1:-help}"
if [[ $# -gt 0 ]]; then
  shift
fi

PORT="${PAPER_PNL_PORT:-8000}"
PY_BIN="${PYTHON_BIN:-}"
FOLLOW_LOGS=0

usage() {
  cat <<EOF
Usage: scripts/launchd/ui_server_ctl.sh <command> [options]

Commands:
  start|up            Install/load launchd UI service and start it
  restart             Restart UI service (kickstart)
  stop|down           Stop/unload UI service
  status              Show service + listener + health
  health              Query /api/health
  logs                Show recent UI logs
  install             Install/load UI launchd service only
  help                Show this help

Options:
  --port <port>       UI port (default: 8000)
  --python <path>     Python interpreter for launchd plist
  --follow|-f         Follow logs (for logs command)
EOF
}

resolve_py() {
  if [[ -n "${PY_BIN}" ]]; then
    return 0
  fi
  if [[ -x "${ROOT_DIR}/.venv/bin/python3" ]]; then
    PY_BIN="${ROOT_DIR}/.venv/bin/python3"
    return 0
  fi
  if [[ -x "${HOME}/.venvs/data_engine_py310/bin/python3" ]]; then
    PY_BIN="${HOME}/.venvs/data_engine_py310/bin/python3"
    return 0
  fi
  PY_BIN="$(command -v python3)"
}

service_loaded() {
  launchctl print "gui/${UID_NUM}/${LABEL}" >/dev/null 2>&1
}

bootstrap_from_plist() {
  launchctl bootstrap "gui/${UID_NUM}" "${PLIST}" >/dev/null 2>&1 || true
  launchctl enable "gui/${UID_NUM}/${LABEL}" >/dev/null 2>&1 || true
  launchctl kickstart -k "gui/${UID_NUM}/${LABEL}" >/dev/null 2>&1 || true
}

install_cmd() {
  resolve_py
  "${INSTALL_SCRIPT}" --no-agent --ui-port "${PORT}" --python "${PY_BIN}" --load
}

start_cmd() {
  if [[ ! -f "${PLIST}" ]]; then
    install_cmd
  elif ! service_loaded; then
    bootstrap_from_plist
  else
    launchctl kickstart -k "gui/${UID_NUM}/${LABEL}" >/dev/null 2>&1 || true
  fi
  status_cmd
}

restart_cmd() {
  if ! service_loaded; then
    start_cmd
    return 0
  fi
  launchctl kickstart -k "gui/${UID_NUM}/${LABEL}" >/dev/null 2>&1 || true
  status_cmd
}

stop_cmd() {
  launchctl bootout "gui/${UID_NUM}/${LABEL}" >/dev/null 2>&1 || true
  echo "Stopped ${LABEL} (if it was running)."
}

health_cmd() {
  curl -sS --max-time 5 "http://127.0.0.1:${PORT}/api/health"
  echo
}

logs_cmd() {
  if [[ "${FOLLOW_LOGS}" -eq 1 ]]; then
    tail -f "${LOG_STDERR}" "${LOG_STDOUT}"
    return 0
  fi
  echo "== stderr (${LOG_STDERR}) =="
  tail -n 80 "${LOG_STDERR}" 2>/dev/null || true
  echo
  echo "== stdout (${LOG_STDOUT}) =="
  tail -n 80 "${LOG_STDOUT}" 2>/dev/null || true
}

status_cmd() {
  local tmp
  tmp="/tmp/${LABEL}.status.$$"
  echo "== Launchd service =="
  if launchctl print "gui/${UID_NUM}/${LABEL}" >"${tmp}" 2>&1; then
    grep -En "state =|pid =|path =|working directory =|stdout path =|stderr path =|runs =" "${tmp}" || true
  else
    echo "Not loaded: ${LABEL}"
  fi
  rm -f "${tmp}" >/dev/null 2>&1 || true

  echo
  echo "== Listener =="
  lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN || echo "No process is listening on port ${PORT}"

  echo
  echo "== Health =="
  health_cmd || true
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)
      PORT="${2:-}"
      shift 2
      ;;
    --python)
      PY_BIN="${2:-}"
      shift 2
      ;;
    --follow|-f)
      FOLLOW_LOGS=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

case "${CMD}" in
  start|up)
    start_cmd
    ;;
  restart)
    restart_cmd
    ;;
  stop|down)
    stop_cmd
    ;;
  status)
    status_cmd
    ;;
  health)
    health_cmd
    ;;
  logs)
    logs_cmd
    ;;
  install)
    install_cmd
    ;;
  help|--help|-h)
    usage
    ;;
  *)
    echo "Unknown command: ${CMD}" >&2
    usage
    exit 1
    ;;
esac
