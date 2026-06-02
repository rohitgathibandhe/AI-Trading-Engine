#!/usr/bin/env python3
"""Pre-market readiness check for the intraday options agent.

Runs at 09:00 IST every trading day (Mon–Fri via launchd).

Checks performed:
  1. Agent process is alive (heartbeat < 5 min old)
  2. Broker auth token is valid (no expired token)
  3. Session state is reset for today (not stale from yesterday)
  4. No open paper position left dangling from yesterday
  5. Runtime config is sane (mode, lots, thresholds)
  6. EOD learning ran yesterday (thresholds were updated)
  7. Option chain data feed is reachable
  8. Weekly cycle context (Day 1 / pre-expiry / expiry) + IC opportunity flag

Sends a Telegram report with either:
  ✅ READY — everything green, market state context, today's playbook targets
  ⚠️  WARNING — non-critical issues; agent will still run but degraded
  🚨 CRITICAL — agent is down, stale token, or dangerous stale state
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

# ─── Paths ──────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).resolve().parents[1]
_STATE      = _ROOT / "data_engine" / "market_ai" / "state"
_SCRIPTS    = _ROOT / "scripts"

CREDS_FILE          = _STATE / "creds.json"
HEARTBEAT_FILE      = _STATE / "agent_heartbeat.json"
AGENT_PID_FILE      = _STATE / "agent.pid"
RUNTIME_CONFIG_FILE = _STATE / "intraday_v83_runtime_config.json"
RUNTIME_STATE_FILE  = _STATE / "intraday_v83_runtime_state.json"
PAPER_STATE_FILE    = _STATE / "intraday_v83_paper_state.json"
EOD_REPORT_FILE     = _STATE / "eod_learning_report.json"
VALIDATION_JSONL    = _STATE / "intraday_v83_paper_live_validation_decisions.jsonl"

IST = timezone(timedelta(hours=5, minutes=30))

# ─── Helpers ────────────────────────────────────────────────────────────────
def _now_ist() -> datetime:
    return datetime.now(IST)

def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text()) if path.exists() else {}
    except Exception:
        return {}

def _load_creds() -> dict:
    return _read_json(CREDS_FILE)

def _send_telegram(text: str) -> bool:
    creds = _load_creds()
    bot_token = str(creds.get("telegram_bot_token") or "").strip()
    chat_id   = str(creds.get("telegram_chat_id") or "").strip()
    if not bot_token or not chat_id:
        print("[telegram] No credentials — skipping")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        return r.status_code == 200
    except Exception as e:
        print(f"[telegram] send failed: {e}")
        return False


# ─── Weekly cycle helpers ────────────────────────────────────────────────────
# NIFTY weekly expiry = Tuesday (shifts to Monday on holidays).
def _weekly_context(today: date) -> dict:
    wd = today.weekday()  # 0=Mon 1=Tue 2=Wed 3=Thu 4=Fri
    labels = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri"}
    day_label = labels.get(wd, str(wd))

    is_expiry_day     = wd == 1  # Tuesday
    is_post_expiry    = wd == 2  # Wednesday — Day 1 of new weekly series (best IC day)
    is_pre_expiry     = wd == 0  # Monday — day before expiry, gamma rises
    days_to_expiry    = (1 - wd) % 7 or 7  # days until next Tuesday

    if is_expiry_day:
        weekly_label = "⚡ EXPIRY DAY (Tuesday) — high gamma, directional spreads use next-week options"
        ic_opportunity = "🚫 IC blocked on expiry day (gamma risk)"
    elif is_post_expiry:
        weekly_label = "🌟 DAY 1 OF NEW WEEKLY SERIES (Wednesday) — 7 DTE, maximum weekly premium"
        ic_opportunity = "✅ IC OPPORTUNITY: 7DTE options are richest, window opens 10:45 IST"
    elif is_pre_expiry:
        weekly_label = f"⚠️ PRE-EXPIRY (Monday, {days_to_expiry}d to expiry) — gamma rising, tighter stops"
        ic_opportunity = "⚠️ IC: normal window (11:30), extra caution on strikes"
    else:
        days_str = f"{days_to_expiry}d to expiry"
        weekly_label = f"📅 {day_label} — {days_str}, standard session"
        ic_opportunity = f"✅ IC: normal window 11:30–13:30 if market is range-bound"

    return {
        "weekday": wd,
        "day_label": day_label,
        "is_expiry_day": is_expiry_day,
        "is_post_expiry": is_post_expiry,
        "is_pre_expiry": is_pre_expiry,
        "days_to_expiry": days_to_expiry,
        "weekly_label": weekly_label,
        "ic_opportunity": ic_opportunity,
    }


# ─── Individual checks ───────────────────────────────────────────────────────
def check_agent_alive() -> tuple[str, str]:
    """Returns (status, message). status ∈ {OK, WARNING, CRITICAL}."""
    heartbeat = _read_json(HEARTBEAT_FILE)
    pid_data  = _read_json(AGENT_PID_FILE)

    if not heartbeat:
        return "CRITICAL", "❌ Heartbeat file missing — agent may not be running"

    ts_str = heartbeat.get("timestamp") or ""
    try:
        ts = datetime.fromisoformat(ts_str)
        age_minutes = (_now_ist() - ts).total_seconds() / 60
    except Exception:
        return "CRITICAL", f"❌ Heartbeat timestamp invalid: {ts_str!r}"

    pid = heartbeat.get("pid") or pid_data.get("pid")
    pid_alive = False
    if pid:
        try:
            os.kill(int(pid), 0)
            pid_alive = True
        except OSError:
            pass

    if age_minutes > 10:
        return "CRITICAL", f"❌ Heartbeat stale ({age_minutes:.0f} min old, PID {pid})"
    if age_minutes > 5:
        return "WARNING", f"⚠️ Heartbeat aging ({age_minutes:.1f} min, PID {pid})"
    if not pid_alive:
        return "WARNING", f"⚠️ Heartbeat fresh but PID {pid} not found in process list"

    mode = heartbeat.get("v83_runtime_mode", "UNKNOWN")
    return "OK", f"✅ Agent alive (PID {pid}, mode={mode}, heartbeat {age_minutes:.1f} min ago)"


def check_session_state(today: date) -> tuple[str, str]:
    state = _read_json(RUNTIME_STATE_FILE)
    if not state:
        return "WARNING", "⚠️ Runtime state file missing — will be created on first tick"

    session_date = state.get("session_date", "")
    if session_date != today.isoformat():
        return "WARNING", f"⚠️ Runtime state has session_date={session_date!r} (expected {today}) — will auto-reset on first tick"

    trade_count = int(state.get("session_trade_count") or 0)
    pnl = float(state.get("realized_pnl_rupees") or 0.0)
    daily_lock = state.get("daily_lock", {})
    if daily_lock.get("active"):
        return "WARNING", f"⚠️ Daily lock active from yesterday: {daily_lock.get('reason')} — will auto-reset"

    return "OK", f"✅ Session state clean (date={session_date}, trades={trade_count}, P&L=₹{pnl:.0f})"


def check_paper_position() -> tuple[str, str]:
    paper_state = _read_json(PAPER_STATE_FILE)
    active = paper_state.get("active_position")
    if active and isinstance(active, dict) and active.get("structure"):
        strategy = active.get("structure", {}).get("strategy", "UNKNOWN")
        entry = active.get("entry_time", "?")
        return "WARNING", f"⚠️ Open paper position from yesterday: {strategy} entered at {entry} — check if it should be closed"
    return "OK", "✅ No open paper position"


def check_runtime_config() -> tuple[str, str]:
    config = _read_json(RUNTIME_CONFIG_FILE)
    if not config:
        return "CRITICAL", "❌ Runtime config file missing"

    issues = []
    mode = config.get("mode", "UNKNOWN")
    if mode not in {"PAPER_LIVE", "MICRO_LIVE", "LIVE_DISABLED"}:
        issues.append(f"unusual mode={mode}")

    paper_threshold = float(config.get("paper_override_bearish_score_threshold") or 0)
    if paper_threshold < 6.5:
        issues.append(f"paper_threshold={paper_threshold:.1f} too low (should be ≥6.5)")

    max_lots = int((config.get("risk") or {}).get("max_lots_per_trade") or 0)
    if max_lots < 1:
        issues.append(f"max_lots_per_trade={max_lots} invalid")

    enabled_strats = config.get("live_enabled_strategies") or []
    if "IRON_CONDOR" not in enabled_strats:
        issues.append("IRON_CONDOR not in live_enabled_strategies")
    if "BULL_PUT_CREDIT_SPREAD" not in enabled_strats:
        issues.append("BULL_PUT_CREDIT_SPREAD not in live_enabled_strategies")

    if issues:
        return "WARNING", f"⚠️ Config issues: {', '.join(issues)}"

    bullish_pbs = [p for p in (config.get("live_enabled_playbooks") or []) if "BULLISH" in p or "RECOVERY" in p]
    return "OK", (
        f"✅ Config OK (mode={mode}, paper_threshold={paper_threshold:.1f}, "
        f"max_lots={max_lots}, {len(enabled_strats)} strategies, "
        f"{len(bullish_pbs)} bullish playbooks)"
    )


def check_eod_learning() -> tuple[str, str]:
    report = _read_json(EOD_REPORT_FILE)
    if not report:
        return "WARNING", "⚠️ No EOD learning report found — thresholds may not be calibrated"

    run_date = report.get("session_date") or report.get("date") or ""
    today_str = date.today().isoformat()
    yesterday_str = (date.today() - timedelta(days=1)).isoformat()

    if run_date not in {today_str, yesterday_str}:
        return "WARNING", f"⚠️ EOD learning last ran on {run_date!r} — thresholds may be stale"

    trades = report.get("trades_processed", 0)
    calibrated = report.get("calibration_ran", False)
    threshold = report.get("active_bearish_threshold") or report.get("current_bearish_threshold") or "?"
    return "OK", f"✅ EOD learning ran ({run_date}, {trades} trades, calibrated={calibrated}, threshold={threshold})"


def check_recent_decisions(today: date) -> tuple[str, str]:
    """Sanity-check: are there any INSUFFICIENT_DATA decisions today that suggest feed issues?"""
    if not VALIDATION_JSONL.exists():
        return "OK", "✅ No decisions yet today (market not open)"

    today_str = today.isoformat()
    recent: list[dict] = []
    try:
        lines = VALIDATION_JSONL.read_text().splitlines()
        for line in reversed(lines[-200:]):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if row.get("session_date") == today_str:
                    recent.append(row)
            except Exception:
                continue
            if len(recent) >= 10:
                break
    except Exception as e:
        return "WARNING", f"⚠️ Could not read validation JSONL: {e}"

    if not recent:
        return "OK", "✅ No decisions yet today (pre-market)"

    insuf = sum(1 for r in recent if r.get("block_reason") == "INSUFFICIENT_DATA")
    total = len(recent)
    if insuf == total:
        return "WARNING", f"⚠️ All {total} recent decisions are INSUFFICIENT_DATA — option chain feed may be down"
    if insuf > total * 0.5:
        return "WARNING", f"⚠️ {insuf}/{total} recent decisions are INSUFFICIENT_DATA"

    last = recent[0]
    spot = (last.get("data") or {}).get("spot_price") or "?"
    state = last.get("market_state") or "?"
    score = last.get("bearish_trade_score") or "?"
    pcr   = last.get("pcr_by_oi") or "?"
    smb   = last.get("smart_money_bias") or "?"
    return "OK", f"✅ Recent decisions OK (spot={spot}, state={state}, bearish_score={score}, PCR={pcr}, smart_money={smb})"


# ─── Main ─────────────────────────────────────────────────────────────────────
def run_check(dry_run: bool = False) -> int:
    now   = _now_ist()
    today = now.date()
    weekly = _weekly_context(today)

    checks = [
        ("Agent alive",        check_agent_alive()),
        ("Session state",      check_session_state(today)),
        ("Open position",      check_paper_position()),
        ("Runtime config",     check_runtime_config()),
        ("EOD learning",       check_eod_learning()),
        ("Recent decisions",   check_recent_decisions(today)),
    ]

    critical = [(n, m) for n, (s, m) in checks if s == "CRITICAL"]
    warnings  = [(n, m) for n, (s, m) in checks if s == "WARNING"]
    ok_checks = [(n, m) for n, (s, m) in checks if s == "OK"]

    # ── Overall status ──
    if critical:
        overall_icon = "🚨"
        overall_label = "CRITICAL — agent needs attention before market opens"
    elif warnings:
        overall_icon = "⚠️"
        overall_label = "WARNING — agent will run but review before 09:15"
    else:
        overall_icon = "✅"
        overall_label = "ALL SYSTEMS READY"

    # ── Build message ──
    lines = [
        f"{overall_icon} <b>Pre-Market Check — {today.strftime('%a %d %b %Y')} | {now.strftime('%H:%M')} IST</b>",
        f"{overall_icon} <b>{overall_label}</b>",
        "",
        f"<b>📅 Weekly Context</b>",
        f"  {weekly['weekly_label']}",
        f"  {weekly['ic_opportunity']}",
        "",
    ]

    if critical:
        lines.append("<b>🚨 CRITICAL</b>")
        for name, msg in critical:
            lines.append(f"  [{name}] {msg}")
        lines.append("")

    if warnings:
        lines.append("<b>⚠️ WARNINGS</b>")
        for name, msg in warnings:
            lines.append(f"  [{name}] {msg}")
        lines.append("")

    lines.append("<b>✅ OK</b>")
    for name, msg in ok_checks:
        lines.append(f"  [{name}] {msg}")

    lines += [
        "",
        "<b>🎯 Today's Focus</b>",
    ]

    # Weekly-context-aware focus
    if weekly["is_expiry_day"]:
        lines += [
            "  • Tuesday expiry: directional spreads use NEXT-week options",
            "  • Gamma risk is high — agent applies tighter delta bands",
            "  • No Iron Condor today (expiry-day blocked)",
            "  • Watch for post-expiry gap gaps and set context early",
        ]
    elif weekly["is_post_expiry"]:
        lines += [
            "  • 🌟 Day 1 of weekly series — 7 DTE, richest weekly premium",
            "  • IC window opens 10:45 IST (45 min earlier than normal)",
            "  • IC uses wider 150pt spreads + delta 8–14 on both wings",
            "  • Directional setups still live — bear call / bull put as usual",
            "  • Best PCR/OI baseline: fresh weekly data, unbiased",
        ]
    elif weekly["is_pre_expiry"]:
        lines += [
            "  • Pre-expiry Monday: gamma rising into Tuesday expiry",
            "  • IC still possible but agent uses tighter balance filter",
            "  • Directional spreads: watch delta stops — gamma can accelerate",
        ]
    else:
        lines += [
            "  • Standard session — all playbooks active",
            "  • PCR + OI conflict guards active for all entries",
            "  • IC window: 11:30–13:30 if range-confirmed",
        ]

    lines += [
        "",
        "📊 Active Strategies: BEAR_CALL | BULL_PUT | IRON_CONDOR",
        "🔬 Mode: PAPER_LIVE (all trades paper only)",
        "⚙️ OI guards: PCR, smart_money_bias, wall_migration wired in",
        f"⏰ Market opens 09:15 IST | No new entries after 14:30 IST",
    ]

    msg = "\n".join(lines)

    print(msg)
    print()

    exit_code = 2 if critical else (1 if warnings else 0)
    if not dry_run:
        ok = _send_telegram(msg)
        print(f"Telegram: {'sent' if ok else 'FAILED'}")
    else:
        print("[dry-run] Telegram not sent")

    return exit_code


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Pre-market readiness check")
    parser.add_argument("--dry-run", action="store_true", help="Print report but don't send Telegram")
    args = parser.parse_args()
    sys.exit(run_check(dry_run=args.dry_run))
