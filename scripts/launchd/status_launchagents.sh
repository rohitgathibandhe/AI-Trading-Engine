#!/usr/bin/env bash
set -euo pipefail

AGENT_LABEL="com.rohit.ai-trading.agent"
UI_LABEL="com.rohit.ai-trading.ui"

print_status() {
  local label="$1"
  echo
  echo "=== ${label} ==="
  if launchctl print "gui/${UID}/${label}" >/tmp/"${label}".launchctl.$$ 2>&1; then
    sed -n '1,80p' /tmp/"${label}".launchctl.$$
  else
    cat /tmp/"${label}".launchctl.$$ 2>/dev/null || true
  fi
  rm -f /tmp/"${label}".launchctl.$$ >/dev/null 2>&1 || true
}

print_status "$AGENT_LABEL"
print_status "$UI_LABEL"

