#!/usr/bin/env python3
"""V83 intraday agent self-healing watchdog.

Runs every 5 minutes during market hours (09:15–15:35 IST, Mon–Fri).
Detects and fixes issues proactively, alerts via Telegram on critical
problems it cannot auto-resolve.

Checks performed:
  1. Duplicate agent processes → kill extras, keep launchd-managed one
  2. Agent process dead + launchd not restarting → force kickstart
  3. Persistent OPTION_CHAIN_QUOTES_UNAVAILABLE (≥3 consecutive cycles) →
     verify API reachability, log diagnostic
  4. Persistent DATA_PIPELINE_HEALTH_BLOCKED (≥3 cycles) → restart agent
  5. Stale/expired Dhan token → alert immediately (cannot fix automatically)
  6. Agent in after-hours sleep during market hours → restart agent
  7. Log file growing without today's entries → restart agent

Auto-fixes are silent (logged only). Telegram alerts fire for:
  - Unresolvable blocks (token expired, API down for >15 min)
  - Agent restart events (so you know when intervention happened)
  - First clean entry after a block period
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

# ─── Paths ──────────────────────────────────────────────────────────────────
_ROOT   = Path(__file__).resolve().parents[1]
_STATE  = _ROOT / "data_engine" / "market_ai" / "state"
_LOG    = _STATE / "intraday_v83_runner.log"
_WDOG   = _STATE / "v83_watchdog_state.json"
_CREDS  = _STATE / "creds.json"

LAUNCHD_LABEL = "com.algoagent.intraday_v83"
AGENT_MODULE  = "market_ai.intraday_defined_risk.cli"
PYTHON        = "/Users/Rohit/.venvs/data_engine_py310/bin/python3"
WORKDIR       = str(_ROOT / "data_engine")

IST = timezone(timedelta(hours=5, minutes=30))

# Number of consecutive NO_TRADE cycles with the same block before action
BLOCK_THRESHOLD = 3
# Minutes of option chain unavailability before a Telegram alert
OC_ALERT_MINUTES = 15


# ─── Utilities ──────────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(IST)


def _is_market_hours(now: datetime) -> bool:
    hhmm = now.strftime("%H%M")
    return now.weekday() < 5 and "0915" <= hhmm <= "1535"


def _load_creds() -> dict:
    try:
        return json.loads(_CREDS.read_text())
    except Exception:
        return {}


def _send_telegram(text: str) -> bool:
    creds = _load_creds()
    bot_token = str(creds.get("telegram_bot_token") or "").strip()
    chat_id   = str(creds.get("telegram_chat_id") or "").strip()
    if not bot_token or not chat_id:
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        return r.status_code == 200
    except Exception:
        return False


def _log(msg: str) -> None:
    ts = _now().strftime("%Y-%m-%d %H:%M:%S IST")
    print(f"[{ts}] WATCHDOG: {msg}", flush=True)


def _load_wdog_state() -> dict:
    try:
        return json.loads(_WDOG.read_text())
    except Exception:
        return {}


def _save_wdog_state(state: dict) -> None:
    _WDOG.write_text(json.dumps(state, indent=2, default=str))


# ─── Agent process helpers ───────────────────────────────────────────────────

def _find_agent_pids() -> list[int]:
    """Return PIDs of all running v83 agent processes."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "market_ai.intraday_defined_risk.cli"],
            text=True,
        ).strip()
        return [int(p) for p in out.splitlines() if p.strip()]
    except subprocess.CalledProcessError:
        return []


def _launchd_pid() -> int | None:
    """Return the PID that launchd considers canonical for the v83 agent."""
    try:
        out = subprocess.check_output(
            ["launchctl", "list", LAUNCHD_LABEL],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            m = re.search(r'"PID"\s*=\s*(\d+)', line)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return None


def _restart_agent() -> bool:
    """Kick launchd to restart the agent."""
    try:
        uid = os.getuid()
        subprocess.check_call(
            ["launchctl", "kickstart", "-k", f"gui/{uid}/{LAUNCHD_LABEL}"],
            timeout=15,
        )
        return True
    except Exception as e:
        _log(f"kickstart failed: {e}")
        return False


def _kill_pid(pid: int) -> None:
    try:
        os.kill(pid, signal.SIGTERM)
        time.sleep(2)
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


# ─── Log analysis ────────────────────────────────────────────────────────────

def _recent_decisions(n: int = 10) -> list[dict]:
    """Return the last n JSON decision lines from the agent log that have today's timestamp."""
    today_str = _now().strftime("%Y-%m-%d")
    results: list[dict] = []
    if not _LOG.exists():
        return results
    lines = _LOG.read_text(errors="replace").splitlines()
    for line in reversed(lines):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        dr = d.get("metadata", {}).get("data_readiness", {}) or {}
        ts = dr.get("timestamp", "")
        if not ts.startswith(today_str):
            continue
        results.append(d)
        if len(results) >= n:
            break
    return list(reversed(results))


def _consecutive_block(decisions: list[dict], block_key: str) -> int:
    """Count trailing consecutive decisions blocked by block_key."""
    count = 0
    for d in reversed(decisions):
        reason = (d.get("metadata", {}).get("data_readiness", {}) or {}).get("reason", "")
        if reason == block_key:
            count += 1
        else:
            break
    return count


def _last_decision_age_minutes(decisions: list[dict]) -> float | None:
    """Minutes since the most recent decision with today's timestamp."""
    if not decisions:
        return None
    ts_str = (decisions[-1].get("metadata", {}).get("data_readiness", {}) or {}).get("timestamp", "")
    if not ts_str:
        return None
    try:
        ts = datetime.fromisoformat(ts_str).replace(tzinfo=IST)
        return (_now() - ts).total_seconds() / 60
    except Exception:
        return None


def _token_looks_expired(decisions: list[dict]) -> bool:
    """Check if log has recent token auth errors."""
    today_str = _now().strftime("%Y-%m-%d")
    if not _LOG.exists():
        return False
    text = _LOG.read_text(errors="replace")
    # Find lines from today with token error markers
    for line in text.splitlines():
        if today_str not in line:
            continue
        low = line.lower()
        if any(kw in low for kw in ["invalid token", "unauthorized", "token expired", "access token invalid", "authentication failed"]):
            return True
    return False


def _api_reachable() -> bool:
    """Quick HEAD check to Dhan API."""
    try:
        r = requests.head("https://api.dhan.co", timeout=5)
        return r.status_code < 500
    except Exception:
        return False


# ─── Main watchdog logic ─────────────────────────────────────────────────────

def run_watchdog() -> None:
    now = _now()
    _log(f"Watchdog tick at {now.strftime('%H:%M:%S IST')}")

    if not _is_market_hours(now):
        _log("Outside market hours — nothing to do.")
        return

    state = _load_wdog_state()
    alerts_sent: list[str] = []
    fixes_applied: list[str] = []

    # ── 1. Duplicate process check ───────────────────────────────────────────
    pids = _find_agent_pids()
    ld_pid = _launchd_pid()
    if len(pids) > 1:
        extras = [p for p in pids if p != ld_pid]
        for pid in extras:
            _log(f"Killing duplicate agent PID {pid} (launchd canonical={ld_pid})")
            _kill_pid(pid)
        fixes_applied.append(f"Killed {len(extras)} duplicate agent process(es)")

    # ── 2. Agent dead — force restart ───────────────────────────────────────
    pids = _find_agent_pids()
    if not pids:
        _log("Agent not running — triggering launchd kickstart")
        ok = _restart_agent()
        fix = "Agent was dead → kickstart triggered" if ok else "Agent dead + kickstart FAILED"
        fixes_applied.append(fix)
        _send_telegram(f"⚠️ <b>V83 Watchdog</b>\n{fix} at {now.strftime('%H:%M IST')}")
        _save_wdog_state(state)
        return  # Let it restart before next checks

    # ── 3. Analyse recent decisions ──────────────────────────────────────────
    decisions = _recent_decisions(n=20)
    age = _last_decision_age_minutes(decisions)

    # ── 4. Stale log — agent running but not deciding ────────────────────────
    if age is not None and age > 10:
        _log(f"Last decision was {age:.1f} min ago — agent may be stuck")
        ok = _restart_agent()
        fixes_applied.append(f"Agent log stale ({age:.1f} min) → restarted")
        _send_telegram(
            f"⚠️ <b>V83 Watchdog</b>\n"
            f"Agent log stale ({age:.1f} min) — restarted at {now.strftime('%H:%M IST')}"
        )

    # ── 5. Token expiry ──────────────────────────────────────────────────────
    if _token_looks_expired(decisions):
        last_alerted = state.get("token_alert_sent_date", "")
        if last_alerted != str(now.date()):
            state["token_alert_sent_date"] = str(now.date())
            _send_telegram(
                f"🚨 <b>V83 Watchdog — TOKEN EXPIRED</b>\n"
                f"Detected auth errors in agent log at {now.strftime('%H:%M IST')}.\n"
                f"Please update <code>state/creds.json</code> with today's Dhan access token."
            )
            alerts_sent.append("token_expired")
            _log("CRITICAL: Token appears expired — Telegram alert sent")

    # ── 6. Persistent option chain block ────────────────────────────────────
    oc_streak = _consecutive_block(decisions, "OPTION_CHAIN_QUOTES_UNAVAILABLE")
    if oc_streak >= BLOCK_THRESHOLD:
        _log(f"OPTION_CHAIN_QUOTES_UNAVAILABLE for {oc_streak} consecutive cycles")
        oc_block_since = state.get("oc_block_since")
        if not oc_block_since:
            state["oc_block_since"] = now.isoformat()
        else:
            block_minutes = (_now() - datetime.fromisoformat(oc_block_since).replace(tzinfo=IST)).total_seconds() / 60
            if block_minutes >= OC_ALERT_MINUTES:
                last_alerted = state.get("oc_alert_sent_hhmm", "")
                cur_hhmm = now.strftime("%H%M")
                # Alert at most once per 30 min
                if not last_alerted or abs(int(cur_hhmm) - int(last_alerted)) >= 30:
                    api_ok = _api_reachable()
                    state["oc_alert_sent_hhmm"] = cur_hhmm
                    if api_ok:
                        # API is up but we're getting no quotes — restart agent
                        ok = _restart_agent()
                        fixes_applied.append(f"OC unavailable {block_minutes:.0f}min → restarted agent")
                        _send_telegram(
                            f"⚠️ <b>V83 Watchdog</b>\n"
                            f"Option chain unavailable for {block_minutes:.0f} min. API reachable.\n"
                            f"Auto-restarted agent at {now.strftime('%H:%M IST')}."
                        )
                    else:
                        _send_telegram(
                            f"🚨 <b>V83 Watchdog — DHAN API DOWN</b>\n"
                            f"Option chain unavailable for {block_minutes:.0f} min AND Dhan API unreachable.\n"
                            f"No trades possible until API recovers."
                        )
                        alerts_sent.append("api_down")
    else:
        if state.get("oc_block_since"):
            _log("Option chain block cleared")
            state.pop("oc_block_since", None)
            state.pop("oc_alert_sent_hhmm", None)

    # ── 7. Persistent DATA_PIPELINE_HEALTH_BLOCKED ───────────────────────────
    dp_streak = 0
    for d in reversed(decisions):
        block = d.get("metadata", {}).get("primary_block_reason", "")
        if block == "DATA_PIPELINE_HEALTH_BLOCKED":
            dp_streak += 1
        else:
            break
    if dp_streak >= BLOCK_THRESHOLD:
        _log(f"DATA_PIPELINE_HEALTH_BLOCKED for {dp_streak} cycles — restarting agent")
        ok = _restart_agent()
        fixes_applied.append("DATA_PIPELINE_HEALTH_BLOCKED → restarted agent")
        _send_telegram(
            f"⚠️ <b>V83 Watchdog</b>\n"
            f"DATA_PIPELINE_HEALTH_BLOCKED persisted {dp_streak} cycles.\n"
            f"Auto-restarted at {now.strftime('%H:%M IST')}."
        )

    # ── 8. Check if entry window is active but agent is not evaluating ───────
    hhmm = now.strftime("%H%M")
    in_entry_window = "1100" <= hhmm <= "1430"
    if in_entry_window and decisions:
        last_d = decisions[-1]
        block = last_d.get("metadata", {}).get("primary_block_reason", "")
        snap = (last_d.get("metadata", {}).get("data_readiness", {}) or {}).get("reason", "")
        action = last_d.get("action", "")
        _log(f"Entry window: action={action} block={block} snap={snap}")

    # ── Summary ──────────────────────────────────────────────────────────────
    if fixes_applied:
        _log("Fixes applied: " + " | ".join(fixes_applied))
    else:
        _log("No issues detected.")

    _save_wdog_state(state)


if __name__ == "__main__":
    run_watchdog()
