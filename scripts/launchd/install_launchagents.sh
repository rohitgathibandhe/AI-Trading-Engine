#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCH_AGENTS_DIR="${HOME}/Library/LaunchAgents"

AGENT_LABEL="com.rohit.ai-trading.agent"
UI_LABEL="com.rohit.ai-trading.ui"
AGENT_PLIST="${LAUNCH_AGENTS_DIR}/${AGENT_LABEL}.plist"
UI_PLIST="${LAUNCH_AGENTS_DIR}/${UI_LABEL}.plist"

INSTALL_AGENT=1
INSTALL_UI=1
LOAD_NOW=0
AGENT_MODE="live"
UI_PORT="8000"
PY_BIN="${PYTHON_BIN:-}"
POLL_SEC="${POLL_SEC:-10}"
THROTTLE_SEC="${THROTTLE_SEC:-15}"

usage() {
  cat <<'EOF'
Usage: scripts/launchd/install_launchagents.sh [options]

Options:
  --agent-mode <live|paper>   Agent TRADE_MODE (default: live)
  --ui-port <port>            UI server port (default: 8000)
  --python <path>             Python interpreter path (default: .venv/bin/python3 or python3)
  --poll-sec <sec>            POLL_SEC for start_agent (default: 10)
  --throttle-sec <sec>        launchd ThrottleInterval (default: 15)
  --no-agent                  Skip installing agent launch agent
  --no-ui                     Skip installing UI launch agent
  --load                      Bootstrap + enable + kickstart immediately
  --help                      Show this help

Examples:
  scripts/launchd/install_launchagents.sh --agent-mode live --ui-port 8000
  scripts/launchd/install_launchagents.sh --agent-mode paper --load
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --agent-mode)
      AGENT_MODE="${2:-}"
      shift 2
      ;;
    --ui-port)
      UI_PORT="${2:-}"
      shift 2
      ;;
    --python)
      PY_BIN="${2:-}"
      shift 2
      ;;
    --poll-sec)
      POLL_SEC="${2:-}"
      shift 2
      ;;
    --throttle-sec)
      THROTTLE_SEC="${2:-}"
      shift 2
      ;;
    --no-agent)
      INSTALL_AGENT=0
      shift
      ;;
    --no-ui)
      INSTALL_UI=0
      shift
      ;;
    --load)
      LOAD_NOW=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ "$AGENT_MODE" != "live" && "$AGENT_MODE" != "paper" ]]; then
  echo "Invalid --agent-mode: $AGENT_MODE (must be live or paper)" >&2
  exit 1
fi

if [[ -z "$PY_BIN" ]]; then
  if [[ -x "${ROOT_DIR}/.venv/bin/python3" ]]; then
    PY_BIN="${ROOT_DIR}/.venv/bin/python3"
  else
    PY_BIN="$(command -v python3)"
  fi
fi

mkdir -p "$LAUNCH_AGENTS_DIR"
mkdir -p "${ROOT_DIR}/data_engine/market_ai/state"

PATH_ENV="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

write_agent_plist() {
  cat >"$AGENT_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${AGENT_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${ROOT_DIR}/scripts/launchd/run_agent_launchd.sh</string>
  </array>
  <key>WorkingDirectory</key>
  <string>${ROOT_DIR}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>${PATH_ENV}</string>
    <key>PYTHON_BIN</key>
    <string>${PY_BIN}</string>
    <key>TRADE_MODE</key>
    <string>${AGENT_MODE}</string>
    <key>POLL_SEC</key>
    <string>${POLL_SEC}</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key>
    <false/>
  </dict>
  <key>ThrottleInterval</key>
  <integer>${THROTTLE_SEC}</integer>
  <key>ProcessType</key>
  <string>Background</string>
  <key>StandardOutPath</key>
  <string>${ROOT_DIR}/data_engine/market_ai/state/launchd_agent.stdout.log</string>
  <key>StandardErrorPath</key>
  <string>${ROOT_DIR}/data_engine/market_ai/state/launchd_agent.stderr.log</string>
</dict>
</plist>
EOF
}

write_ui_plist() {
  cat >"$UI_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>${UI_LABEL}</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>${ROOT_DIR}/scripts/launchd/run_ui_server_launchd.sh</string>
  </array>
  <key>WorkingDirectory</key>
  <string>${ROOT_DIR}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>PATH</key>
    <string>${PATH_ENV}</string>
    <key>PYTHON_BIN</key>
    <string>${PY_BIN}</string>
    <key>PAPER_PNL_PORT</key>
    <string>${UI_PORT}</string>
  </dict>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <dict>
    <key>SuccessfulExit</key>
    <false/>
  </dict>
  <key>ThrottleInterval</key>
  <integer>${THROTTLE_SEC}</integer>
  <key>ProcessType</key>
  <string>Background</string>
  <key>StandardOutPath</key>
  <string>${ROOT_DIR}/data_engine/market_ai/state/launchd_ui.stdout.log</string>
  <key>StandardErrorPath</key>
  <string>${ROOT_DIR}/data_engine/market_ai/state/launchd_ui.stderr.log</string>
</dict>
</plist>
EOF
}

bootout_if_loaded() {
  local label="$1"
  launchctl bootout "gui/${UID}/${label}" >/dev/null 2>&1 || true
}

bootstrap_and_enable() {
  local label="$1"
  local plist="$2"
  bootout_if_loaded "$label"
  launchctl bootstrap "gui/${UID}" "$plist"
  launchctl enable "gui/${UID}/${label}" || true
  launchctl kickstart -k "gui/${UID}/${label}" || true
}

if [[ "$INSTALL_AGENT" -eq 1 ]]; then
  write_agent_plist
  echo "Wrote ${AGENT_PLIST}"
fi
if [[ "$INSTALL_UI" -eq 1 ]]; then
  write_ui_plist
  echo "Wrote ${UI_PLIST}"
fi

if [[ "$LOAD_NOW" -eq 1 ]]; then
  if [[ "$INSTALL_AGENT" -eq 1 ]]; then
    bootstrap_and_enable "$AGENT_LABEL" "$AGENT_PLIST"
    echo "Loaded ${AGENT_LABEL}"
  fi
  if [[ "$INSTALL_UI" -eq 1 ]]; then
    bootstrap_and_enable "$UI_LABEL" "$UI_PLIST"
    echo "Loaded ${UI_LABEL}"
  fi
else
  echo "Plists installed only (not loaded). Use --load to bootstrap now."
fi

cat <<EOF

Next checks:
  launchctl print gui/${UID}/${AGENT_LABEL}   # if agent installed
  launchctl print gui/${UID}/${UI_LABEL}      # if UI installed
  tail -f ${ROOT_DIR}/data_engine/market_ai/state/launchd_agent.stderr.log
  tail -f ${ROOT_DIR}/data_engine/market_ai/state/launchd_ui.stderr.log
EOF
