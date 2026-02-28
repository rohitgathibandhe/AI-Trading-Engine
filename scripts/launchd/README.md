# launchd Auto-Start / Auto-Restart (macOS)

This folder provides `launchd` helpers to run the trading agent and UI server as
user `LaunchAgents` with auto-start on login and auto-restart on crash.

## Services

- `com.rohit.ai-trading.agent`
  - Runs `data_engine/market_ai/start_agent.py`
  - `TRADE_MODE` configurable (`live` or `paper`)
- `com.rohit.ai-trading.ui`
  - Runs `scripts/paper_pnl_server.py` (default port `8000`)

## Files

- `run_agent_launchd.sh` : wrapper that resolves Python and runs the agent
- `run_ui_server_launchd.sh` : wrapper that runs the UI/API server
- `ui_server_ctl.sh` : single command wrapper for UI start/restart/stop/status/health/logs
- `install_launchagents.sh` : generate/install plists (optionally load now)
- `status_launchagents.sh` : inspect `launchctl` status
- `uninstall_launchagents.sh` : unload/remove plists

## UI one-command controls

```bash
scripts/launchd/ui_server_ctl.sh start --port 8000
scripts/launchd/ui_server_ctl.sh status --port 8000
scripts/launchd/ui_server_ctl.sh restart --port 8000
scripts/launchd/ui_server_ctl.sh health --port 8000
scripts/launchd/ui_server_ctl.sh logs
scripts/launchd/ui_server_ctl.sh stop
```

## Recommended sequence

### 1. Install plists only (safe)

```bash
scripts/launchd/install_launchagents.sh --agent-mode paper
```

This writes plists to `~/Library/LaunchAgents` but does not start them.

### 2. Load and start (paper rehearsal)

```bash
scripts/launchd/install_launchagents.sh --agent-mode paper --load
```

### 3. Check status

```bash
scripts/launchd/status_launchagents.sh
```

### 4. Review logs

```bash
tail -f data_engine/market_ai/state/launchd_agent.stderr.log
tail -f data_engine/market_ai/state/launchd_ui.stderr.log
```

## Live mode install (only after checks)

```bash
scripts/launchd/install_launchagents.sh --agent-mode live --load
```

## Uninstall / disable

```bash
scripts/launchd/uninstall_launchagents.sh
```

## Important notes

- These are `LaunchAgents` (user session). They start when you log in.
- If you need start-on-boot before login, that requires a `LaunchDaemon`
  design and a different security model.
- The agent loads credentials from `data_engine/market_ai/state/creds.json`.
- Do not store secrets in the launchd plist.

## Pre-live checklist (minimum)

1. Dhan access token is fresh in `data_engine/market_ai/state/creds.json`
2. Telegram alerts tested (`POST /api/alerts/telegram/test`)
3. `GET /api/health` returns non-stale heartbeat during paper run
4. `GET /api/control/status` shows:
   - watchdog not stale
   - Telegram configured
   - no active locks (unless intentional)
5. Paper run under launchd for at least one session before switching to live
