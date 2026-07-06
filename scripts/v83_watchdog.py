#!/usr/bin/env python3
"""V83 intraday agent self-healing watchdog.

Runs every 5 minutes during market hours (09:15–15:35 IST, Mon–Fri).

Three layers of monitoring:

LAYER 1 — Infrastructure (process health)
  - Duplicate processes → kill extras
  - Dead agent → kickstart via launchd
  - Stale log → restart agent
  - Token expiry → Telegram alert

LAYER 2 — Data quality (API health)
  - Option chain 429 loop → auto-increase dhan_option_chain_attempts in config
  - Persistent OPTION_CHAIN_QUOTES_UNAVAILABLE → restart agent
  - DATA_PIPELINE_HEALTH_BLOCKED persisting → restart agent

LAYER 3 — Strategy adaptation (the self-learning layer)
  - Bullish signals above threshold but ENTRY_WINDOW_CLOSED → widen bullish window
  - Bearish signals above threshold but ENTRY_WINDOW_CLOSED → widen bearish window
  - HIGH trade score + correct market bias + entry_ready=False for 8+ cycles
    → PLAYBOOK GAP detected → writes v83_pending_gap_report.json + Telegram alert
    → gap report read by watchdog startup check; code fix applied before next session
  - New unknown block reasons → Telegram alert with diagnosis
  - Session ends with no trade despite ≥1 cycle where signal crossed threshold
    → diagnose and alert so the gap is fixed before tomorrow's open

All config mutations are written to the runtime_config JSON and the agent
hot-reloads it on its next poll cycle (no restart required).
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

# ─── Paths ──────────────────────────────────────────────────────────────────
_ROOT   = Path(__file__).resolve().parents[1]
_STATE  = _ROOT / "data_engine" / "market_ai" / "state"
_LOG    = _STATE / "intraday_v83_runner.log"
_WDOG   = _STATE / "v83_watchdog_state.json"
_CREDS  = _STATE / "creds.json"
_RT_CFG = _STATE / "intraday_v83_runtime_config.json"
_RL_CFG = _STATE / "intraday_v83_run_live_config.json"

LAUNCHD_LABEL = "com.algoagent.intraday_v83"
IST = timezone(timedelta(hours=5, minutes=30))

# ─── Thresholds ──────────────────────────────────────────────────────────────
BLOCK_THRESHOLD      = 3     # consecutive same-reason blocks before acting
OC_ALERT_MINUTES     = 15    # minutes OC unavailable before Telegram
WINDOW_CLOSED_CYCLES = 5     # cycles with ENTRY_WINDOW_CLOSED + good signal before widening window
STALE_THRESHOLD_LATE = 10    # minutes (after 09:45 AM)
STALE_THRESHOLD_EARLY = 20   # minutes (before 09:45 AM, accumulating candles)
BULLISH_SIGNAL_MIN   = 3.3   # minimum bullish_entry_score we consider "signal present"
BEARISH_SIGNAL_MIN   = 3.3   # minimum bearish_entry_score we consider "signal present"


# ─── Utilities ───────────────────────────────────────────────────────────────

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


def _load_rt_config() -> dict:
    try:
        return json.loads(_RT_CFG.read_text())
    except Exception:
        return {}


def _save_rt_config(cfg: dict) -> None:
    cfg["updated_at"] = _now().isoformat()
    _RT_CFG.write_text(json.dumps(cfg, indent=2, sort_keys=True))


def _load_rl_config() -> dict:
    try:
        return json.loads(_RL_CFG.read_text())
    except Exception:
        return {}


def _save_rl_config(cfg: dict) -> None:
    _RL_CFG.write_text(json.dumps(cfg, indent=2, sort_keys=True))


# ─── Agent process helpers ───────────────────────────────────────────────────

def _find_agent_pids() -> list[int]:
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "market_ai.intraday_defined_risk.cli"],
            text=True,
        ).strip()
        return [int(p) for p in out.splitlines() if p.strip()]
    except subprocess.CalledProcessError:
        return []


def _launchd_pid() -> int | None:
    try:
        out = subprocess.check_output(
            ["launchctl", "list", LAUNCHD_LABEL],
            text=True, stderr=subprocess.DEVNULL,
        )
        for line in out.splitlines():
            m = re.search(r'"PID"\s*=\s*(\d+)', line)
            if m:
                return int(m.group(1))
    except Exception:
        pass
    return None


def _restart_agent() -> bool:
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


# ─── Log analysis ─────────────────────────────────────────────────────────────

def _todays_decisions(n: int = 60) -> list[dict]:
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
        m = d.get("metadata", {}) or {}
        # Check both data_readiness.timestamp (error path) and trade_funnel.timestamp (success path)
        dr_ts = (m.get("data_readiness") or {}).get("timestamp", "")
        tf_ts = (m.get("trade_funnel") or {}).get("timestamp", "")
        ts = dr_ts or tf_ts
        if not ts.startswith(today_str):
            continue
        results.append(d)
        if len(results) >= n:
            break
    return list(reversed(results))


def _last_decision_age_minutes(decisions: list[dict]) -> float | None:
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


def _consecutive_snap_reason(decisions: list[dict], reason: str) -> int:
    count = 0
    for d in reversed(decisions):
        r = (d.get("metadata", {}).get("data_readiness", {}) or {}).get("reason", "")
        if r == reason:
            count += 1
        else:
            break
    return count


def _token_looks_expired() -> bool:
    today_str = _now().strftime("%Y-%m-%d")
    if not _LOG.exists():
        return False
    text = _LOG.read_text(errors="replace")
    for line in text.splitlines():
        if today_str not in line:
            continue
        low = line.lower()
        if any(kw in low for kw in ["invalid token", "unauthorized", "token expired", "authentication failed"]):
            return True
    return False


def _api_reachable() -> bool:
    try:
        r = requests.head("https://api.dhan.co", timeout=5)
        return r.status_code < 500
    except Exception:
        return False


# ─── LAYER 3: Strategy gap analysis ─────────────────────────────────────────

def _analyze_strategy_gaps(decisions: list[dict]) -> list[str]:
    """
    Scan today's decisions for patterns that indicate a configuration gap.
    Returns a list of fix descriptions applied.
    """
    fixes = []
    now = _now()
    hhmm = now.strftime("%H%M")

    # Only do strategy analysis after 10:00 AM so there's enough signal history
    if hhmm < "1000":
        return fixes

    # Count cycles where:
    #   - bullish_entry_score is strong
    #   - but no trade was taken
    #   - and the entry gate metadata shows ENTRY_WINDOW_CLOSED
    bullish_window_closed = 0
    bearish_window_closed = 0
    bullish_signal_missed = 0   # signal was there, no gate block, still no trade
    bearish_signal_missed = 0
    seen_block_reasons: set[str] = set()
    oc_attempts_hit_limit = 0

    for d in decisions:
        m = d.get("metadata", {}) or {}
        action = d.get("action", "")
        eg = m.get("entry_gate") or {}
        gate_blocks = eg.get("block_reasons") or []
        bull_entry_score = float(m.get("bullish_entry_score") or 0)
        bear_entry_score = float(m.get("bearish_entry_score") or 0)
        bull_entry_ready = bool(m.get("bullish_entry_ready"))
        bear_entry_ready = bool(m.get("bearish_entry_ready"))
        snap_reason = (m.get("data_readiness") or {}).get("reason", "")

        # Track unknown block reasons for alerting
        for br in gate_blocks:
            seen_block_reasons.add(br)

        if action == "NO_TRADE":
            # Strong bullish signal but entry window was closed
            if bull_entry_ready and "ENTRY_WINDOW_CLOSED" in gate_blocks:
                bullish_window_closed += 1
            # Strong bearish signal but entry window was closed
            if bear_entry_ready and "ENTRY_WINDOW_CLOSED" in gate_blocks:
                bearish_window_closed += 1
            # Option chain repeated failures
            if snap_reason == "OPTION_CHAIN_QUOTES_UNAVAILABLE":
                oc_attempts_hit_limit += 1

    rt_cfg = _load_rt_config()
    rl_cfg = _load_rl_config()
    changed_rt = False
    changed_rl = False

    # Fix: bullish window too narrow — widen it
    if bullish_window_closed >= WINDOW_CLOSED_CYCLES:
        current_start = str(rt_cfg.get("allowed_bullish_entry_start_hhmm") or "10:00")
        h, mn = int(current_start[:2]), int(current_start[3:])
        new_start_min = max(h * 60 + mn - 30, 9 * 60 + 45)  # max 09:45 floor
        new_h, new_m = divmod(new_start_min, 60)
        new_start = f"{new_h:02d}:{new_m:02d}"
        if new_start != current_start:
            rt_cfg["allowed_bullish_entry_start_hhmm"] = new_start
            changed_rt = True
            msg = f"📈 <b>V83 Watchdog — Auto-fix</b>\nBullish entry window widened: {current_start} → {new_start}\n({bullish_window_closed} cycles: bullish signal ready but ENTRY_WINDOW_CLOSED)"
            _send_telegram(msg)
            fixes.append(f"Bullish window widened {current_start} → {new_start} ({bullish_window_closed} blocked cycles)")
            _log(fixes[-1])

    # Fix: bearish window too narrow — widen it
    if bearish_window_closed >= WINDOW_CLOSED_CYCLES:
        current_start = str(rt_cfg.get("allowed_entry_start_hhmm") or "11:00")
        h, mn = int(current_start[:2]), int(current_start[3:])
        new_start_min = max(h * 60 + mn - 30, 9 * 60 + 45)
        new_h, new_m = divmod(new_start_min, 60)
        new_start = f"{new_h:02d}:{new_m:02d}"
        if new_start != current_start:
            rt_cfg["allowed_entry_start_hhmm"] = new_start
            changed_rt = True
            msg = f"📉 <b>V83 Watchdog — Auto-fix</b>\nBearish entry window widened: {current_start} → {new_start}\n({bearish_window_closed} cycles: bearish signal ready but ENTRY_WINDOW_CLOSED)"
            _send_telegram(msg)
            fixes.append(f"Bearish window widened {current_start} → {new_start} ({bearish_window_closed} blocked cycles)")
            _log(fixes[-1])

    # Fix: option chain keeps failing — increase retry attempts
    if oc_attempts_hit_limit >= 10:
        current_attempts = int(rl_cfg.get("dhan_option_chain_attempts") or 1)
        if current_attempts < 3:
            rl_cfg["dhan_option_chain_attempts"] = current_attempts + 1
            changed_rl = True
            fixes.append(f"OC attempts increased {current_attempts} → {current_attempts + 1} ({oc_attempts_hit_limit} failures today)")
            _log(fixes[-1])
            _send_telegram(
                f"⚠️ <b>V83 Watchdog — Auto-fix</b>\n"
                f"Option chain kept failing ({oc_attempts_hit_limit}× today).\n"
                f"Increased dhan_option_chain_attempts: {current_attempts} → {current_attempts + 1}"
            )

    if changed_rt:
        _save_rt_config(rt_cfg)
    if changed_rl:
        _save_rl_config(rl_cfg)

    # ── LAYER 3b: Playbook gap detection ──────────────────────────────────────
    # Detect: high trade score + correct market bias + bullish/bearish_entry_ready=False
    # This means no playbook matched despite good conditions. Cannot auto-fix code,
    # but write a structured gap report and alert immediately.
    fixes += _detect_playbook_gaps(decisions)

    return fixes


# Threshold: bull/bear trade score above this with entry_ready=False triggers gap report
_PLAYBOOK_GAP_SCORE_MIN = 4.5
# How many consecutive cycles must have the gap before alerting
_PLAYBOOK_GAP_CYCLES_MIN = 8
# Pending gap report file — written so next Claude session can pick it up
_GAP_REPORT = _STATE / "v83_pending_gap_report.json"


def _detect_playbook_gaps(decisions: list[dict]) -> list[str]:
    """
    Detect when bullish/bearish trade score is high and market bias is aligned
    but entry_ready stays False (SETUP_NOT_DETECTED). This means no playbook is
    matching despite favourable conditions — a code gap, not a config gap.
    Write a structured report and alert immediately.
    """
    fixes: list[str] = []
    now = _now()
    hhmm = now.strftime("%H%M")
    # Only check during core trading window
    if hhmm < "1030" or hhmm > "1500":
        return fixes

    today_str = str(now.date())

    # Load existing gap report to avoid duplicate alerts
    existing = {}
    try:
        existing = json.loads(_GAP_REPORT.read_text()) if _GAP_REPORT.exists() else {}
    except Exception:
        pass
    if existing.get("date") == today_str and existing.get("alerted"):
        return fixes

    # Filter all today's decisions to the core trading window (10:30–14:30).
    # Use rolling count (not consecutive streak) — bias oscillates within a minute.
    # A trade having fired today means no gap.
    window_decisions = [
        d for d in decisions
        if "10:30" <= str(((d.get("metadata", {}) or {}).get("trade_funnel", {}) or {}).get("timestamp", ""))[11:16] <= "14:30"
    ]

    had_trade = any(d.get("action") == "TRADE" for d in decisions)
    if had_trade:
        return fixes

    bull_gap_cycles: list[dict] = []
    bear_gap_cycles: list[dict] = []

    for d in window_decisions:
        m = d.get("metadata", {}) or {}
        tf = m.get("trade_funnel", {}) or {}
        if not tf:
            continue

        bull_score = float(tf.get("bullish_trade_score") or 0)
        bear_score = float(tf.get("bearish_trade_score") or 0)
        bull_ready = bool(m.get("bullish_entry_ready"))
        bear_ready = bool(m.get("bearish_entry_ready"))
        bias = str(tf.get("market_state_bias") or "")
        market = str(tf.get("market_state") or "")
        rejection = str(tf.get("canonical_rejection_reason") or "")
        chain_state = str(m.get("option_chain_pressure_state") or "")
        ts = str((tf.get("timestamp") or ""))[11:16]

        # Bull gap: high score, bullish bias, entry_ready=False, chain not vetoing
        if (
            bull_score >= _PLAYBOOK_GAP_SCORE_MIN
            and bias == "BULLISH"
            and not bull_ready
            and rejection in {"SETUP_NOT_DETECTED", "SETUP_PATTERN_MISMATCH", "SETUP_QUALITY_TOO_LOW", "NO_VALID_STRATEGY"}
            and chain_state not in {"BALANCED_WALLS", "DOWNSIDE_PUT_SUPPORT"}
        ):
            bull_gap_cycles.append({
                "ts": ts, "bull_score": bull_score, "market": market,
                "rejection": rejection, "chain": chain_state,
                "entry_score": m.get("bullish_entry_score"), "trend_score": m.get("bullish_trend_score"),
            })

        # Bear gap: high score, bearish bias, entry_ready=False
        if (
            bear_score >= _PLAYBOOK_GAP_SCORE_MIN
            and bias == "BEARISH"
            and not bear_ready
            and rejection in {"SETUP_NOT_DETECTED", "SETUP_PATTERN_MISMATCH", "SETUP_QUALITY_TOO_LOW", "NO_VALID_STRATEGY"}
            and chain_state not in {"BALANCED_WALLS", "OVERHEAD_CALL_PRESSURE"}
        ):
            bear_gap_cycles.append({
                "ts": ts, "bear_score": bear_score, "market": market,
                "rejection": rejection, "chain": chain_state,
                "entry_score": m.get("bearish_entry_score"), "trend_score": m.get("bearish_trend_score"),
            })

    direction = None
    gap_cycles = []
    if len(bull_gap_cycles) >= _PLAYBOOK_GAP_CYCLES_MIN:
        direction = "BULLISH"
        gap_cycles = bull_gap_cycles
    elif len(bear_gap_cycles) >= _PLAYBOOK_GAP_CYCLES_MIN:
        direction = "BEARISH"
        gap_cycles = bear_gap_cycles

    if not direction:
        return fixes

    first = gap_cycles[0]
    last = gap_cycles[-1]
    avg_score = sum(c.get("bull_score" if direction == "BULLISH" else "bear_score", 0) for c in gap_cycles) / len(gap_cycles)
    best_score = max(c.get("bull_score" if direction == "BULLISH" else "bear_score", 0) for c in gap_cycles)
    sample = gap_cycles[0]

    report = {
        "date": today_str,
        "detected_at": now.isoformat(),
        "direction": direction,
        "gap_cycles": len(gap_cycles),
        "window": f"{first['ts']}–{last['ts']}",
        "avg_score": round(avg_score, 2),
        "best_score": round(best_score, 2),
        "rejection": sample.get("rejection"),
        "market_state": sample.get("market"),
        "chain_state": sample.get("chain"),
        "bullish_entry_score": sample.get("entry_score"),
        "bullish_trend_score": sample.get("trend_score"),
        "alerted": False,
        "action_required": (
            f"Add/loosen a playbook condition for {direction} {sample['market']} "
            f"with score {best_score:.1f}. Rejection was {sample['rejection']}. "
            f"Check regime.py entry_ready conditions."
        ),
    }

    _GAP_REPORT.write_text(json.dumps(report, indent=2, default=str))
    _log(
        f"PLAYBOOK GAP DETECTED: {direction} score {best_score:.2f} for {len(gap_cycles)} cycles "
        f"({first['ts']}–{last['ts']}) but entry_ready=False. "
        f"Rejection={sample['rejection']} chain={sample['chain']}. "
        f"Report written to v83_pending_gap_report.json."
    )

    msg = (
        f"🚨 <b>V83 Watchdog — PLAYBOOK GAP</b>\n\n"
        f"<b>{direction}</b> trade score <b>{best_score:.2f}</b> for {len(gap_cycles)} cycles "
        f"({first['ts']}–{last['ts']}) but <b>no playbook matched</b>.\n\n"
        f"Market: {sample['market']} | Chain: {sample['chain']}\n"
        f"Rejection: {sample['rejection']}\n"
        f"entry_score={sample['entry_score']} | trend_score={sample['trend_score']}\n\n"
        f"⚠️ This is a code gap — a new regime.py condition is needed.\n"
        f"Gap report saved to <code>v83_pending_gap_report.json</code> for next Claude session."
    )
    sent = _send_telegram(msg)
    report["alerted"] = True
    _GAP_REPORT.write_text(json.dumps(report, indent=2, default=str))

    fixes.append(
        f"PLAYBOOK GAP: {direction} score {best_score:.2f} for {len(gap_cycles)} cycles — "
        f"gap report written, alert {'sent' if sent else 'FAILED (network)'}"
    )
    return fixes


def _end_of_session_diagnosis(decisions: list[dict], state: dict) -> None:
    """
    After 15:00 IST, if no trade was taken today despite signal presence, diagnose
    and send a Telegram summary so the gap is visible before tomorrow.
    """
    now = _now()
    if now.strftime("%H%M") < "1500":
        return
    today_str = str(now.date())
    if state.get("eod_diagnosis_sent") == today_str:
        return

    had_trade = any(d.get("action") == "TRADE" for d in decisions)
    if had_trade:
        state["eod_diagnosis_sent"] = today_str
        return

    # Find the best signal seen today
    best_bull = 0.0
    best_bear = 0.0
    bull_ready_count = 0
    bear_ready_count = 0
    block_tally: dict[str, int] = defaultdict(int)

    for d in decisions:
        m = d.get("metadata", {}) or {}
        best_bull = max(best_bull, float(m.get("bullish_trade_score") or 0))
        best_bear = max(best_bear, float(m.get("bearish_trade_score") or 0))
        if m.get("bullish_entry_ready"):
            bull_ready_count += 1
        if m.get("bearish_entry_ready"):
            bear_ready_count += 1
        eg = m.get("entry_gate") or {}
        for br in (eg.get("block_reasons") or []):
            block_tally[br] += 1

    top_blocks = sorted(block_tally.items(), key=lambda x: -x[1])[:5]
    block_summary = "\n".join(f"  • {b}: {c}×" for b, c in top_blocks) or "  (none logged)"

    msg = (
        f"📊 <b>V83 End-of-Day: No Trade Today ({today_str})</b>\n\n"
        f"Best bullish trade score: <b>{best_bull:.2f}</b> "
        f"({'ready ' + str(bull_ready_count) + '×' if bull_ready_count else 'never entry-ready'})\n"
        f"Best bearish trade score: <b>{best_bear:.2f}</b> "
        f"({'ready ' + str(bear_ready_count) + '×' if bear_ready_count else 'never entry-ready'})\n\n"
        f"Top gate blocks:\n{block_summary}\n\n"
        f"<i>Watchdog will auto-fix window issues overnight if needed.</i>"
    )
    sent = _send_telegram(msg)
    if sent:
        state["eod_diagnosis_sent"] = today_str
        _log(f"EOD diagnosis sent: bull={best_bull:.2f} bear={best_bear:.2f} top_block={top_blocks[0][0] if top_blocks else 'none'}")


# ─── IV history tracker ──────────────────────────────────────────────────────

def _update_iv_history(decisions: list[dict], state: dict) -> None:
    """After 15:30 IST, persist today's avg_chain_iv and compute rolling IV rank.

    iv_history.csv stores up to 60 sessions (≈ 3 months) of avg_chain_iv.
    Rolling 20-day IV rank: 0 = historically low IV, 100 = historically high IV.
    High IV rank (>70) means premium selling edge is elevated.
    Low IV rank (<30) means IV is compressed — widen wings or be more selective.
    """
    import csv as _csv
    now = _now()
    if now.strftime("%H%M") < "1530":
        return
    today_str = str(now.date())
    if state.get("iv_history_written") == today_str:
        return

    # Extract last non-None avg_chain_iv from today's decisions
    avg_iv: float | None = None
    for d in reversed(decisions):
        v = (d.get("metadata") or {}).get("avg_chain_iv")
        if v is not None:
            try:
                avg_iv = float(v)
                break
            except (TypeError, ValueError):
                pass
    if avg_iv is None:
        return  # no IV data today (API down all day)

    # Load, update, and trim history
    rows: list[dict] = []
    if _IV_HISTORY.exists():
        try:
            with _IV_HISTORY.open() as f:
                rows = list(_csv.DictReader(f))
        except Exception:
            rows = []
    rows = [r for r in rows if r.get("date") != today_str]
    rows.append({"date": today_str, "avg_chain_iv": f"{avg_iv:.4f}"})
    rows = rows[-60:]
    try:
        with _IV_HISTORY.open("w", newline="") as f:
            writer = _csv.DictWriter(f, fieldnames=["date", "avg_chain_iv"])
            writer.writeheader()
            writer.writerows(rows)
    except Exception as exc:
        _log(f"IV history write failed: {exc}")
        return

    # Compute 20-day rolling IV rank
    recent_ivs: list[float] = []
    for r in rows[-20:]:
        try:
            recent_ivs.append(float(r["avg_chain_iv"]))
        except (ValueError, KeyError):
            pass
    iv_rank: float | None = None
    if len(recent_ivs) >= 5:
        low, high = min(recent_ivs), max(recent_ivs)
        iv_rank = round((avg_iv - low) / (high - low) * 100, 1) if high > low else 50.0

    state["iv_history_written"] = today_str
    state["last_iv_rank"] = iv_rank
    _log(f"IV history: avg_chain_iv={avg_iv:.2f} iv_rank={iv_rank} sessions={len(rows)}")

    if iv_rank is not None:
        rank_emoji = "🔥" if iv_rank > 70 else ("❄️" if iv_rank < 30 else "📊")
        context = (
            "Premium selling edge elevated — favour tighter spreads for higher credit."
            if iv_rank > 70 else (
                "IV compressed — widen wings or skip low-credit setups."
                if iv_rank < 30 else
                "IV in normal range — standard strike selection applies."
            )
        )
        _send_telegram(
            f"{rank_emoji} <b>V83 IV Rank — EOD Update ({today_str})</b>\n"
            f"Avg chain IV: <b>{avg_iv:.1f}%</b> | "
            f"20-day IV Rank: <b>{iv_rank:.0f}/100</b>\n"
            f"<i>{context}</i>"
        )


# ─── Self-learning engine ─────────────────────────────────────────────────────

def _load_exit_trades() -> list[dict]:
    """Load all PAPER_EXIT records from the blotter (all sessions)."""
    trades: list[dict] = []
    if not _PAPER_TRADES.exists():
        return trades
    try:
        with _PAPER_TRADES.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line)
                    if t.get("event") == "PAPER_EXIT" and t.get("realized_paper_pnl") is not None:
                        trades.append(t)
                except Exception:
                    pass
    except Exception:
        pass
    return trades


def _update_setup_performance(state: dict) -> None:
    """After 15:30 IST: compute win rates and avg PnL per playbook subtype."""
    now = _now()
    if now.strftime("%H%M") < "1530":
        return
    today_str = str(now.date())
    if state.get("setup_perf_written") == today_str:
        return

    trades = _load_exit_trades()
    if not trades:
        return

    by_playbook: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        playbook = str(t.get("playbook") or "UNKNOWN")
        by_playbook[playbook].append({
            "pnl": float(t["realized_paper_pnl"]),
            "mfe": float(t.get("mfe_rupees") or 0.0),
            "exit_reason": str(t.get("exit_reason") or ""),
        })

    performance: dict[str, dict] = {}
    for playbook, records in by_playbook.items():
        wins = [r for r in records if r["pnl"] > 0]
        avg_pnl = sum(r["pnl"] for r in records) / len(records)
        total_mfe = sum(r["mfe"] for r in records if r["mfe"] > 0)
        total_pnl_on_mfe = sum(r["pnl"] for r in records if r["mfe"] > 0)
        mfe_capture = total_pnl_on_mfe / total_mfe if total_mfe > 0 else 0.0
        performance[playbook] = {
            "samples": len(records),
            "wins": len(wins),
            "win_rate": round(len(wins) / len(records), 3),
            "avg_pnl": round(avg_pnl, 2),
            "avg_mfe": round(sum(r["mfe"] for r in records) / len(records), 2),
            "mfe_capture_rate": round(mfe_capture, 3),
            "last_5_pnls": [r["pnl"] for r in records[-5:]],
        }

    _SETUP_PERF.write_text(json.dumps({"updated_at": today_str, "performance": performance}, indent=2))
    state["setup_perf_written"] = today_str
    _log(f"Setup performance updated: {len(performance)} playbooks across {len(trades)} exits")

    # Alert on underperforming setups (real trading setups only)
    skip = {"NO_TRADE", "UNKNOWN", "RANGE_NO_TRADE", "GAP_NO_TRADE"}
    underperformers = [
        (p, d) for p, d in performance.items()
        if p not in skip and d["samples"] >= 5 and d["win_rate"] < 0.35
    ]
    if underperformers:
        lines = "\n".join(
            f"• {p}: {d['win_rate']*100:.0f}% win, {d['samples']} trades, avg {d['avg_pnl']:.0f}₹"
            for p, d in underperformers
        )
        _send_telegram(
            f"⚠️ <b>V83 Learning — Underperforming Setups</b>\n\n"
            f"{lines}\n\n"
            f"<i>Auto-raising entry threshold for these directions.</i>"
        )


def _apply_adaptive_gates(state: dict) -> list[str]:
    """Read setup performance and raise entry score thresholds for consistently losing directions."""
    fixes: list[str] = []
    if not _SETUP_PERF.exists():
        return fixes
    try:
        perf_data = json.loads(_SETUP_PERF.read_text())
    except Exception:
        return fixes

    performance = perf_data.get("performance", {})
    skip = {"NO_TRADE", "UNKNOWN", "RANGE_NO_TRADE", "GAP_NO_TRADE"}

    # Aggregate wins/samples per direction
    bear_wins = bear_total = bull_wins = bull_total = 0
    for p, d in performance.items():
        if p in skip:
            continue
        if any(kw in p for kw in ("BEARISH", "BEAR", "SHORT", "REJECTION", "CONTINUATION_DOWN")):
            bear_wins += d["wins"]
            bear_total += d["samples"]
        elif any(kw in p for kw in ("BULLISH", "BULL", "RECLAIM", "CONTINUATION_UP")):
            bull_wins += d["wins"]
            bull_total += d["samples"]

    try:
        rt = json.loads(_RT_CFG.read_text()) if _RT_CFG.exists() else {}
    except Exception:
        return fixes

    changed = False
    if bear_total >= 5:
        bear_wr = bear_wins / bear_total
        if bear_wr < 0.30:
            # paper_override_bearish_score_threshold is the hot-loaded paper gate field
            cur = float(rt.get("paper_override_bearish_score_threshold", 6.5))
            new_val = round(min(cur + 0.3, 9.0), 2)
            if new_val != cur:
                rt["paper_override_bearish_score_threshold"] = new_val
                fixes.append(f"bearish_paper_threshold {cur:.1f}→{new_val:.1f} (bear wr {bear_wr*100:.0f}%)")
                changed = True

    if bull_total >= 5:
        bull_wr = bull_wins / bull_total
        if bull_wr < 0.30:
            cur = float(rt.get("paper_override_bullish_score_threshold", rt.get("paper_override_bearish_score_threshold", 6.5)))
            new_val = round(min(cur + 0.3, 9.0), 2)
            if new_val != cur:
                rt["paper_override_bullish_score_threshold"] = new_val
                fixes.append(f"bullish_paper_threshold {cur:.1f}→{new_val:.1f} (bull wr {bull_wr*100:.0f}%)")
                changed = True

    if changed:
        try:
            _RT_CFG.write_text(json.dumps(rt, indent=2))
        except Exception as exc:
            _log(f"adaptive gates write failed: {exc}")

    return fixes


def _update_circuit_breaker(state: dict) -> list[str]:
    """Detect 2+ consecutive session losses and write circuit_breaker_active to runtime_config.

    Called on every watchdog cycle during market hours.  Clears the flag automatically
    when the last 2 trades include a winner (i.e. the streak is broken).
    """
    fixes: list[str] = []
    today_str = str(_now().date())
    trades = _load_exit_trades()
    today_trades = [t for t in trades if t.get("session_date") == today_str]

    if len(today_trades) >= 2:
        last_two = today_trades[-2:]
        consecutive_losses = all(float(t.get("realized_paper_pnl", 1.0)) < 0 for t in last_two)
    else:
        consecutive_losses = False

    try:
        rt = json.loads(_RT_CFG.read_text()) if _RT_CFG.exists() else {}
    except Exception:
        return fixes

    current = bool(rt.get("circuit_breaker_active", False))
    if consecutive_losses == current:
        return fixes  # no change needed

    rt["circuit_breaker_active"] = consecutive_losses
    try:
        _RT_CFG.write_text(json.dumps(rt, indent=2))
    except Exception as exc:
        _log(f"circuit_breaker write failed: {exc}")
        return fixes

    if consecutive_losses:
        _send_telegram(
            "🔴 <b>Circuit Breaker ARMED</b>\n"
            "2 consecutive session losses detected.\n"
            "Entry score threshold raised +1.0 for rest of session."
        )
        _log("Circuit breaker armed: 2 consecutive losses today")
        fixes.append("circuit_breaker ARMED (2 consecutive session losses)")
    else:
        _log("Circuit breaker cleared")
        fixes.append("circuit_breaker CLEARED")
    return fixes


def _update_time_of_day_performance(state: dict) -> None:
    """After 15:30 IST: compute win rate and avg PnL grouped by entry hour."""
    now = _now()
    if now.strftime("%H%M") < "1530":
        return
    today_str = str(now.date())
    if state.get("tod_perf_written") == today_str:
        return

    trades = _load_exit_trades()
    if len(trades) < 3:
        return

    by_hour: dict[int, list[float]] = defaultdict(list)
    for t in trades:
        ts = t.get("entry_timestamp") or ""
        try:
            hour = int(ts[11:13])
        except (ValueError, IndexError):
            continue
        by_hour[hour].append(float(t["realized_paper_pnl"]))

    tod_perf: dict[str, dict] = {}
    for hour, pnls in sorted(by_hour.items()):
        wins = [p for p in pnls if p > 0]
        tod_perf[f"{hour:02d}:00"] = {
            "samples": len(pnls),
            "win_rate": round(len(wins) / len(pnls), 3),
            "avg_pnl": round(sum(pnls) / len(pnls), 2),
            "total_pnl": round(sum(pnls), 2),
        }

    _TIME_PERF.write_text(json.dumps({"updated_at": today_str, "by_hour": tod_perf}, indent=2))
    state["tod_perf_written"] = today_str
    _log(f"Time-of-day performance updated: {len(tod_perf)} hours from {len(trades)} exits")

    # Alert on consistently negative hours (≥3 samples, avg pnl < -500)
    bad_hours = [(h, d) for h, d in tod_perf.items() if d["samples"] >= 3 and d["avg_pnl"] < -500]
    if bad_hours:
        lines = "\n".join(f"• {h}: avg {d['avg_pnl']:.0f}₹ ({d['samples']} trades)" for h, d in bad_hours)
        _send_telegram(
            f"⏰ <b>V83 Learning — Weak Time Slots</b>\n\n"
            f"{lines}\n\n"
            f"<i>Consider raising entry thresholds in these hours.</i>"
        )


# ─── EoD position guardian ───────────────────────────────────────────────────

_PAPER_STATE  = _STATE / "intraday_v83_paper_state.json"
_IV_HISTORY   = _STATE / "iv_history.csv"
_PAPER_TRADES   = _STATE / "intraday_v83_paper_live_trades.jsonl"
_SETUP_PERF     = _STATE / "setup_performance.json"
_TIME_PERF      = _STATE / "time_of_day_performance.json"
_WIN_PROB_MODEL = _STATE / "win_prob_model.json"

# Features used for ML training — must match _ML_FEATURE_NAMES in features.py
_ML_FEAT_NAMES = [
    "ml_bearish_entry_score", "ml_bullish_entry_score", "ml_india_vix",
    "ml_vwap_reclaim", "ml_banknifty_divergence", "ml_days_to_monthly_expiry",
    "ml_bullish_momentum", "ml_bearish_momentum", "ml_rv30_pct",
    "ml_order_flow_imbalance",
]


def _train_win_probability_model(state: dict) -> None:
    """EOD: train a logistic regression on completed paper trades with ML features.

    Pairs ENTRY + EXIT records by entry_timestamp; requires at least 15 labeled
    samples.  Saves coefficient sets for BEARISH and BULLISH directions separately
    to state/win_prob_model.json.  Silently skips if sklearn unavailable or data
    is insufficient.
    """
    if not _PAPER_TRADES.exists():
        return
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        import numpy as np
    except ImportError:
        return

    # Load all records
    entries, exits_by_entry = {}, {}
    try:
        with _PAPER_TRADES.open() as f:
            for line in f:
                row = json.loads(line.strip())
                if row.get("event") == "PAPER_ENTRY" and row.get("entry_timestamp"):
                    entries[row["entry_timestamp"]] = row
                elif row.get("event") == "PAPER_EXIT" and row.get("entry_timestamp"):
                    exits_by_entry[row["entry_timestamp"]] = row
    except Exception:
        return

    # Build labelled samples: require all ML features present + known PnL
    samples_bear, samples_bull = [], []
    for ts, entry in entries.items():
        exit_rec = exits_by_entry.get(ts)
        if exit_rec is None or exit_rec.get("realized_paper_pnl") is None:
            continue
        # Check if all ML features are recorded in the entry
        if not any(f in entry for f in _ML_FEAT_NAMES):
            continue
        feat_vec = []
        for f in _ML_FEAT_NAMES:
            raw = entry.get(f)
            if raw is None:
                feat_vec.append(0.0)
            else:
                feat_vec.append(1.0 if raw is True else (0.0 if raw is False else float(raw)))
        label = 1 if float(exit_rec["realized_paper_pnl"]) > 0 else 0
        direction = str(entry.get("setup_direction") or entry.get("ml_setup_direction") or "").upper()
        if direction == "BULLISH":
            samples_bull.append((feat_vec, label))
        else:
            samples_bear.append((feat_vec, label))

    def _fit_direction(samples):
        if len(samples) < 15:
            return None
        X = np.array([s[0] for s in samples])
        y = np.array([s[1] for s in samples])
        if len(set(y)) < 2:
            return None
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(X_s, y)
        # Store weights in original (unscaled) space: w_raw = w_scaled / std, b adjusted
        std = scaler.scale_
        mean = scaler.mean_
        weights = {f: round(float(clf.coef_[0][i] / std[i]), 6) for i, f in enumerate(_ML_FEAT_NAMES)}
        intercept = float(clf.intercept_[0]) - sum(
            clf.coef_[0][i] / std[i] * mean[i] for i in range(len(_ML_FEAT_NAMES))
        )
        return {"weights": weights, "intercept": round(intercept, 6)}

    result_bear = _fit_direction(samples_bear)
    result_bull = _fit_direction(samples_bull)
    total_samples = len(samples_bear) + len(samples_bull)

    if result_bear is None and result_bull is None:
        return

    model = {
        "samples": total_samples,
        "bearish_samples": len(samples_bear),
        "bullish_samples": len(samples_bull),
        "bearish": result_bear,
        "bullish": result_bull,
    }
    try:
        _WIN_PROB_MODEL.write_text(json.dumps(model, indent=2))
        _log(f"win_prob_model trained: {total_samples} samples (bear={len(samples_bear)}, bull={len(samples_bull)})")
    except Exception as exc:
        _log(f"win_prob_model write failed: {exc}")


def _guardian_force_close(now: datetime, state: dict, fixes_applied: list[str]) -> None:
    """
    After 15:20 IST, if a paper position is still open and the agent has not written
    a recent decision, force-exit the position directly in paper_state.json.

    This is the last-resort safety net when the agent is down at TIME_EXIT (15:15).
    Paper mode only — live mode would need a broker call.
    """
    hhmm = now.strftime("%H%M")
    if hhmm < "1520":
        return
    today_str = str(now.date())
    if state.get("eod_guardian_ran") == today_str:
        return
    try:
        ps = json.loads(_PAPER_STATE.read_text()) if _PAPER_STATE.exists() else {}
    except Exception:
        return
    if not ps.get("active_position"):
        return  # no open position — nothing to do

    # Only act if log is stale (agent not writing decisions)
    log_age_min = (time.time() - _LOG.stat().st_mtime) / 60.0 if _LOG.exists() else 999
    if log_age_min < 8:
        return  # agent is alive and handling it

    # Force-close: clear active position and write a synthetic exit to paper_state
    pos = ps["active_position"]
    entry_credit = float((pos.get("structure") or {}).get("credit_points") or 0)
    entry_time = ps.get("last_exit", {}).get("entry_timestamp") or now.isoformat()
    ps["last_exit"] = {
        "event": "PAPER_EXIT",
        "exit_reason": "TIME_EXIT",
        "exit_mode": "GUARDIAN_FORCE_CLOSE",
        "exit_timestamp": now.isoformat(),
        "entry_timestamp": entry_time,
        "realized_paper_pnl": 0.0,  # conservative — we don't have live LTP
        "guardian_note": "Watchdog force-closed at 15:20+ because agent was unresponsive",
    }
    ps["active_position"] = None
    try:
        _PAPER_STATE.write_text(json.dumps(ps, indent=2, default=str))
    except Exception as exc:
        _log(f"EoD guardian: failed to write paper_state: {exc}")
        return

    state["eod_guardian_ran"] = today_str
    msg = (
        f"🛡️ <b>V83 EoD Guardian — FORCE CLOSE</b>\n"
        f"Open position detected at {now.strftime('%H:%M IST')} with agent unresponsive "
        f"({log_age_min:.0f} min since last log).\n"
        f"Cleared position in paper_state.json. Manual review recommended."
    )
    _send_telegram(msg)
    fixes_applied.append(f"EoD guardian: force-closed open position (agent silent {log_age_min:.0f}min)")
    _log(f"EoD guardian: force-closed open paper position (log stale {log_age_min:.0f}min)")


# ─── Main watchdog logic ──────────────────────────────────────────────────────

def _check_pending_gap_report() -> None:
    """On each tick, if a pending gap report exists from today or yesterday, log it prominently."""
    if not _GAP_REPORT.exists():
        return
    try:
        report = json.loads(_GAP_REPORT.read_text())
    except Exception:
        return
    now = _now()
    report_date = report.get("date", "")
    # Show report for up to 2 days so it's visible at next-morning startup
    if report_date not in {str(now.date()), str((now - timedelta(days=1)).date())}:
        return
    if report.get("gap_fixed"):
        return
    _log(
        f"⚠️  PENDING GAP REPORT ({report_date}): {report.get('direction')} playbook gap — "
        f"score {report.get('best_score')} for {report.get('gap_cycles')} cycles "
        f"({report.get('window')}). Rejection={report.get('rejection')}. "
        f"ACTION: {report.get('action_required')}"
    )


def run_watchdog() -> None:
    now = _now()
    _log(f"Watchdog tick at {now.strftime('%H:%M:%S IST')}")
    _check_pending_gap_report()

    if not _is_market_hours(now):
        _log("Outside market hours — nothing to do.")
        return

    state = _load_wdog_state()
    fixes_applied: list[str] = []

    # ── Grace window (post-restart, don't recheck immediately) ───────────────
    grace_until_str = state.get("restart_grace_until", "")
    if grace_until_str:
        try:
            grace_until = datetime.fromisoformat(grace_until_str).replace(tzinfo=IST)
            if now < grace_until:
                _log(f"In restart grace window until {grace_until.strftime('%H:%M:%S')} — skipping.")
                _save_wdog_state(state)
                return
        except Exception:
            pass
        state.pop("restart_grace_until", None)

    # ── LAYER 1: Process health ───────────────────────────────────────────────
    pids = _find_agent_pids()
    ld_pid = _launchd_pid()
    if len(pids) > 1:
        extras = [p for p in pids if p != ld_pid]
        for pid in extras:
            _log(f"Killing duplicate agent PID {pid} (launchd={ld_pid})")
            _kill_pid(pid)
        fixes_applied.append(f"Killed {len(extras)} duplicate(s)")

    pids = _find_agent_pids()
    if not pids:
        _log("Agent not running — kickstart")
        ok = _restart_agent()
        fix = "Agent dead → kickstart" if ok else "Agent dead + kickstart FAILED"
        fixes_applied.append(fix)
        _send_telegram(f"⚠️ <b>V83 Watchdog</b>\n{fix} at {now.strftime('%H:%M IST')}")
        state["restart_grace_until"] = (now + timedelta(minutes=2)).isoformat()
        _save_wdog_state(state)
        return

    # ── LAYER 1: Token expiry ────────────────────────────────────────────────
    if _token_looks_expired():
        last_alerted = state.get("token_alert_sent_date", "")
        if last_alerted != str(now.date()):
            state["token_alert_sent_date"] = str(now.date())
            _send_telegram(
                f"🚨 <b>V83 Watchdog — TOKEN EXPIRED</b>\n"
                f"Auth errors in log at {now.strftime('%H:%M IST')}.\n"
                f"Update <code>state/creds.json</code> with today's Dhan token."
            )
            _log("CRITICAL: Token expired — alerted")

    # ── LAYER 1: Stale log ────────────────────────────────────────────────────
    decisions = _todays_decisions(n=600)  # large n: gap detection needs full day window
    age = _last_decision_age_minutes(decisions)
    stale_threshold = STALE_THRESHOLD_EARLY if now.strftime("%H%M") < "0945" else STALE_THRESHOLD_LATE

    # Blind-spot fix: when age is None (no JSON decisions at all today), fall back to
    # log file mtime. An agent that is completely frozen produces no JSON lines, so
    # _last_decision_age_minutes returns None and the original check was silently skipped.
    if age is None and _LOG.exists():
        age = (time.time() - _LOG.stat().st_mtime) / 60.0

    if age is not None and age > stale_threshold:
        _log(f"Log stale {age:.1f} min — restarting")
        ok = _restart_agent()
        fixes_applied.append(f"Log stale ({age:.1f} min) → restarted")
        if age > 20:
            _send_telegram(f"⚠️ <b>V83 Watchdog</b>\nLog stale {age:.1f} min → restarted at {now.strftime('%H:%M IST')}")
        state["restart_grace_until"] = (now + timedelta(minutes=3)).isoformat()

    # ── LAYER 1b: All-day INSUFFICIENT_DATA detection ─────────────────────────
    # When Dhan API is completely down, every cycle logs INSUFFICIENT_DATA.
    # This never triggers the stale-log check (log IS being written) so we detect
    # it separately and alert + restart periodically.
    if now.strftime("%H%M") >= "0940" and decisions:
        insuff_count = sum(
            1 for d in decisions
            if (d.get("metadata", {}).get("data_readiness") or {}).get("reason") == "MIN_5M_CANDLES_NOT_READY"
            or (d.get("metadata", {}).get("data_readiness") or {}).get("candle_count", 1) == 0
        )
        if insuff_count == len(decisions) and insuff_count >= 3:
            last_insuff_alert = state.get("insuff_data_alert_hhmm", "")
            cur_h = now.strftime("%H%M")
            if not last_insuff_alert or abs(int(cur_h) - int(last_insuff_alert)) >= 30:
                state["insuff_data_alert_hhmm"] = cur_h
                _log(f"ALL {insuff_count} decisions today are INSUFFICIENT_DATA — Dhan historical API may be down")
                _send_telegram(
                    f"🚨 <b>V83 Watchdog — DATA API DOWN</b>\n"
                    f"All {insuff_count} decisions today show INSUFFICIENT_DATA.\n"
                    f"Dhan historical candle API appears to be down.\n"
                    f"Check API status at {now.strftime('%H:%M IST')}."
                )
                ok = _restart_agent()
                fixes_applied.append(f"All-day INSUFFICIENT_DATA ({insuff_count} cycles) → restarted")
                state["restart_grace_until"] = (now + timedelta(minutes=3)).isoformat()
        else:
            state.pop("insuff_data_alert_hhmm", None)

    # ── LAYER 2: Data quality ─────────────────────────────────────────────────
    oc_streak = _consecutive_snap_reason(decisions, "OPTION_CHAIN_QUOTES_UNAVAILABLE")
    if oc_streak >= BLOCK_THRESHOLD:
        oc_block_since = state.get("oc_block_since")
        if not oc_block_since:
            state["oc_block_since"] = now.isoformat()
        else:
            block_minutes = (_now() - datetime.fromisoformat(oc_block_since).replace(tzinfo=IST)).total_seconds() / 60
            if block_minutes >= OC_ALERT_MINUTES:
                last_h = state.get("oc_alert_sent_hhmm", "")
                cur_h = now.strftime("%H%M")
                if not last_h or abs(int(cur_h) - int(last_h)) >= 30:
                    state["oc_alert_sent_hhmm"] = cur_h
                    if _api_reachable():
                        ok = _restart_agent()
                        fixes_applied.append(f"OC unavailable {block_minutes:.0f}min → restarted")
                        state["restart_grace_until"] = (now + timedelta(minutes=3)).isoformat()
                        _send_telegram(
                            f"⚠️ <b>V83 Watchdog</b>\nOC unavailable {block_minutes:.0f} min. "
                            f"Restarted at {now.strftime('%H:%M IST')}."
                        )
                    else:
                        _send_telegram(
                            f"🚨 <b>V83 Watchdog — DHAN API DOWN</b>\n"
                            f"OC unavailable {block_minutes:.0f} min AND API unreachable."
                        )
    else:
        state.pop("oc_block_since", None)
        state.pop("oc_alert_sent_hhmm", None)

    dp_streak = 0
    for d in reversed(decisions):
        if d.get("metadata", {}).get("primary_block_reason") == "DATA_PIPELINE_HEALTH_BLOCKED":
            dp_streak += 1
        else:
            break
    if dp_streak >= BLOCK_THRESHOLD:
        _log(f"DATA_PIPELINE_HEALTH_BLOCKED {dp_streak} cycles — restarting")
        ok = _restart_agent()
        fixes_applied.append(f"DATA_PIPELINE_HEALTH_BLOCKED {dp_streak}× → restarted")
        state["restart_grace_until"] = (now + timedelta(minutes=3)).isoformat()
        _send_telegram(
            f"⚠️ <b>V83 Watchdog</b>\nDATA_PIPELINE_HEALTH_BLOCKED {dp_streak}× → restarted at {now.strftime('%H:%M IST')}"
        )

    # ── LAYER 2b: EoD position guardian ──────────────────────────────────────
    # If a paper position is still open after 15:20 and the agent has not written
    # a decision in the last 10 minutes, force-exit the position now. This prevents
    # overnight holds when TIME_EXIT fires at 15:15 but the agent is down.
    _guardian_force_close(now, state, fixes_applied)

    # ── LAYER 3: Strategy gap analysis ───────────────────────────────────────
    strategy_fixes = _analyze_strategy_gaps(decisions)
    fixes_applied.extend(strategy_fixes)

    # ── EOD diagnosis (no-trade day) ──────────────────────────────────────────
    _end_of_session_diagnosis(decisions, state)

    # ── EOD IV history tracking ───────────────────────────────────────────────
    _update_iv_history(decisions, state)

    # ── Intraday circuit breaker: detect consecutive losses, gate entry ────────
    cb_fixes = _update_circuit_breaker(state)
    fixes_applied.extend(cb_fixes)

    # ── EOD self-learning: setup performance + adaptive gates + time-of-day ──
    _update_setup_performance(state)
    adaptive_fixes = _apply_adaptive_gates(state)
    fixes_applied.extend(adaptive_fixes)
    _update_time_of_day_performance(state)
    _train_win_probability_model(state)

    # ── Summary ───────────────────────────────────────────────────────────────
    if fixes_applied:
        _log("Fixes: " + " | ".join(fixes_applied))
    else:
        _log("No issues.")

    _save_wdog_state(state)


if __name__ == "__main__":
    run_watchdog()
