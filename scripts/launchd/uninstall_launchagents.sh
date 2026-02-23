#!/usr/bin/env bash
set -euo pipefail

LAUNCH_AGENTS_DIR="${HOME}/Library/LaunchAgents"
AGENT_LABEL="com.rohit.ai-trading.agent"
UI_LABEL="com.rohit.ai-trading.ui"
AGENT_PLIST="${LAUNCH_AGENTS_DIR}/${AGENT_LABEL}.plist"
UI_PLIST="${LAUNCH_AGENTS_DIR}/${UI_LABEL}.plist"

REMOVE_AGENT=1
REMOVE_UI=1

usage() {
  cat <<'EOF'
Usage: scripts/launchd/uninstall_launchagents.sh [--no-agent] [--no-ui]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-agent)
      REMOVE_AGENT=0
      shift
      ;;
    --no-ui)
      REMOVE_UI=0
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

bootout_if_loaded() {
  local label="$1"
  launchctl bootout "gui/${UID}/${label}" >/dev/null 2>&1 || true
}

if [[ "$REMOVE_AGENT" -eq 1 ]]; then
  bootout_if_loaded "$AGENT_LABEL"
  rm -f "$AGENT_PLIST"
  echo "Removed ${AGENT_PLIST}"
fi

if [[ "$REMOVE_UI" -eq 1 ]]; then
  bootout_if_loaded "$UI_LABEL"
  rm -f "$UI_PLIST"
  echo "Removed ${UI_PLIST}"
fi

