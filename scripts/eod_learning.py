#!/usr/bin/env python3
"""End-of-day learning loop for the intraday options agent.

Runs at 15:35 IST every trading day (via launchd).

What it does:
  1. Parses today's paper trade outcomes from the JSONL log
  2. Feeds outcomes into the LearningStore SQLite database
  3. Runs parameter calibration (adjusts score thresholds if enough data)
  4. Promotes shadow parameters to active if they pass the guard rails
  5. Writes updated thresholds back to the runtime config for tomorrow
  6. Computes per-playbook rolling stats over the last 30 sessions
  7. Identifies missed opportunities (TRADABLE setups that were blocked)
  8. Sends a concise Telegram report
  9. Writes a JSON report to state/ for the dashboard

Usage:
  python scripts/eod_learning.py              # runs for today
  python scripts/eod_learning.py --date 2026-06-01  # backfill a specific date
  python scripts/eod_learning.py --dry-run    # compute + report, no writes
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import date, datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import requests

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_ENGINE = ROOT / "data_engine"
sys.path.insert(0, str(DATA_ENGINE))

STATE = DATA_ENGINE / "market_ai" / "state"
TRADES_JSONL         = STATE / "intraday_v83_paper_live_trades.jsonl"
DECISIONS_JSONL      = STATE / "intraday_v83_paper_live_validation_decisions.jsonl"
SHADOW_JSONL         = STATE / "intraday_v83_shadow_live_decisions.jsonl"
RUNTIME_CONFIG       = STATE / "intraday_v83_runtime_config.json"
LEARNING_DB          = STATE / "intraday_v83_paper_live_learning.sqlite3"
CREDS_FILE           = STATE / "creds.json"
EOD_REPORT_FILE      = STATE / "eod_learning_report.json"
PLAYBOOK_STATS_FILE  = STATE / "playbook_performance_stats.json"
LOG_FILE             = ROOT / "logs" / "eod_learning.log"

IST = timezone(timedelta(hours=5, minutes=30))

# ── guardrails for threshold adjustment ──────────────────────────────────────
# Thresholds are nudged by ±STEP, clamped within [MIN, MAX]
THRESHOLD_STEP_UP   = 0.30   # raise when profit_factor < UNDERPERFORM_PF
THRESHOLD_STEP_DOWN = 0.20   # lower when profit_factor > OUTPERFORM_PF
UNDERPERFORM_PF     = 1.20   # below this → tighten entry bar
OUTPERFORM_PF       = 2.20   # above this → loosen entry bar
MIN_SAMPLES_TO_ADAPT = 5     # minimum trades before adjusting thresholds
BEARISH_THRESHOLD_MIN = 5.50
BEARISH_THRESHOLD_MAX = 9.00
BULLISH_THRESHOLD_MIN = 6.00
BULLISH_THRESHOLD_MAX = 9.50
ROLLING_WINDOW_SESSIONS = 30  # sessions used for rolling stats


# ── helpers ──────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    ts = datetime.now(IST).strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a") as f:
            f.write(line + "\n")
    except Exception:
        pass


def _load_creds() -> dict:
    try:
        return json.loads(CREDS_FILE.read_text())
    except Exception:
        return {}


def _send_telegram(text: str) -> bool:
    creds = _load_creds()
    bot_token = str(creds.get("telegram_bot_token") or "").strip()
    chat_id   = str(creds.get("telegram_chat_id") or "").strip()
    if not bot_token or not chat_id:
        _log("[telegram] No credentials — skipping alert")
        return False
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        return r.status_code == 200
    except Exception as e:
        _log(f"[telegram] send failed: {e}")
        return False


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            pass
    return records


def _profit_factor(pnls: list[float]) -> float:
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss   = abs(sum(p for p in pnls if p < 0))
    if gross_loss == 0:
        return gross_profit if gross_profit > 0 else 0.0
    return round(gross_profit / gross_loss, 3)


def _max_drawdown(pnls: list[float]) -> float:
    equity = 0.0
    peak   = 0.0
    dd     = 0.0
    for p in pnls:
        equity += p
        peak    = max(peak, equity)
        dd      = max(dd, peak - equity)
    return round(dd, 2)


# ── core data extraction ─────────────────────────────────────────────────────

def _parse_completed_trades(session_date: str) -> list[dict]:
    """Pair ENTRY + EXIT events from the trades JSONL for a given session."""
    all_records = _load_jsonl(TRADES_JSONL)
    # collect entries and exits for this session
    entries: dict[str, dict] = {}  # keyed by entry_timestamp
    exits: list[dict] = []
    for rec in all_records:
        if rec.get("session_date") != session_date:
            continue
        if rec.get("event") == "PAPER_ENTRY":
            key = rec.get("entry_timestamp", "")
            entries[key] = rec
        elif rec.get("event") == "PAPER_EXIT":
            exits.append(rec)

    trades = []
    for ex in exits:
        entry_ts = ex.get("entry_timestamp", "")
        en = entries.get(entry_ts, {})
        gate = en.get("gate") or {}
        pnl = float(ex.get("realized_paper_pnl") or 0.0)
        trades.append({
            "session_date": session_date,
            "entry_timestamp": entry_ts,
            "exit_timestamp": ex.get("exit_timestamp"),
            "playbook": ex.get("playbook") or gate.get("playbook") or "UNKNOWN",
            "strategy": ex.get("strategy") or gate.get("strategy") or "UNKNOWN",
            "pnl_rupees": pnl,
            "exit_reason": ex.get("exit_reason"),
            "mae_rupees": float(ex.get("mae_rupees") or 0.0),
            "mfe_rupees": float(ex.get("mfe_rupees") or 0.0),
            "attribution": ex.get("paper_trade_attribution") or ex.get("paper_candidate_reason"),
            "market_state": gate.get("market_state"),
            "bearish_trade_score": float(gate.get("bearish_trade_score") or 0.0),
            "tradability_class": gate.get("tradability_class"),
        })
    return trades


def _parse_missed_opportunities(session_date: str) -> list[dict]:
    """Find TRADABLE setups that were blocked (not traded)."""
    records = _load_jsonl(DECISIONS_JSONL)
    missed = []
    seen_playbooks: set[str] = set()
    for rec in records:
        if rec.get("session_date") != session_date:
            continue
        if rec.get("tradability_class") != "TRADABLE":
            continue
        if rec.get("v83_decision") == "TRADE":
            continue  # was traded, not missed
        pb = rec.get("playbook") or "UNKNOWN"
        block = rec.get("block_reason") or "UNKNOWN"
        score = rec.get("bearish_trade_score") or rec.get("bullish_trade_score") or 0.0
        # deduplicate — keep highest score per playbook
        key = f"{pb}:{block}"
        if key not in seen_playbooks:
            seen_playbooks.add(key)
            missed.append({
                "playbook": pb,
                "block_reason": block,
                "score": float(score),
                "market_state": rec.get("market_state"),
                "timestamp": rec.get("timestamp"),
            })
    # Sort by score descending — highest conviction misses first
    return sorted(missed, key=lambda x: x["score"], reverse=True)[:10]


def _compute_rolling_playbook_stats(
    window_sessions: int = ROLLING_WINDOW_SESSIONS,
) -> dict[str, dict]:
    """Rolling per-playbook stats across last N sessions."""
    all_records = _load_jsonl(TRADES_JSONL)
    # collect all exit events grouped by (session_date, playbook)
    by_pb: dict[str, list[float]] = defaultdict(list)
    session_dates: set[str] = set()
    for rec in sorted(all_records, key=lambda r: r.get("timestamp", "")):
        if rec.get("event") != "PAPER_EXIT":
            continue
        session_date = rec.get("session_date", "")
        session_dates.add(session_date)
        pb  = rec.get("playbook") or "UNKNOWN"
        pnl = float(rec.get("realized_paper_pnl") or 0.0)
        by_pb[pb].append(pnl)

    # Limit to last N sessions
    sorted_dates = sorted(session_dates)[-window_sessions:]
    by_pb_windowed: dict[str, list[float]] = defaultdict(list)
    for rec in all_records:
        if rec.get("event") != "PAPER_EXIT":
            continue
        if rec.get("session_date") not in sorted_dates:
            continue
        pb  = rec.get("playbook") or "UNKNOWN"
        pnl = float(rec.get("realized_paper_pnl") or 0.0)
        by_pb_windowed[pb].append(pnl)

    stats = {}
    for pb, pnls in by_pb_windowed.items():
        wins = sum(1 for p in pnls if p > 0)
        total = len(pnls)
        stats[pb] = {
            "samples":        total,
            "win_rate":       round(wins / total, 3) if total > 0 else 0.0,
            "total_pnl":      round(sum(pnls), 2),
            "avg_pnl":        round(sum(pnls) / total, 2) if total > 0 else 0.0,
            "profit_factor":  _profit_factor(pnls),
            "max_drawdown":   _max_drawdown(pnls),
            "best_trade":     round(max(pnls), 2) if pnls else 0.0,
            "worst_trade":    round(min(pnls), 2) if pnls else 0.0,
        }
    return stats


# ── threshold adaptation ──────────────────────────────────────────────────────

def _compute_threshold_adjustments(
    playbook_stats: dict[str, dict],
    current_bearish_threshold: float,
    current_bullish_threshold: float,
) -> dict[str, Any]:
    """
    Compute new global score thresholds based on recent playbook performance.

    Logic:
    - Look at all BEARISH playbooks collectively → adjust bearish_threshold
    - Look at all BULLISH playbooks collectively → adjust bullish_threshold
    - Only adjust if we have MIN_SAMPLES_TO_ADAPT trades
    - Adjustment is bounded and stepped conservatively
    """
    BEARISH_PLAYBOOKS = {
        "SIDEWAYS_TO_BEARISH_REJECTION", "GAP_DOWN_BEARISH_CONTINUATION",
        "GAP_UP_BEARISH_FAILURE", "HIGH_CONFLUENCE_BEARISH_CONTINUATION",
        "EARLY_BALANCE_BEARISH_FAILED_RECLAIM",
    }
    BULLISH_PLAYBOOKS = {
        "OPEN_DRIVE_BULLISH", "GAP_UP_BULLISH_CONTINUATION",
        "GAP_DOWN_BULLISH_RECOVERY", "SIDEWAYS_TO_BULLISH_RECLAIM",
        "HIGH_CONFLUENCE_BULLISH_CONTINUATION", "EARLY_BALANCE_BULLISH_RECLAIM",
        "AFTERNOON_TREND_HOLD_BULLISH",
    }

    def _combined_stats(playbooks: set[str]) -> dict | None:
        pnls = []
        for pb in playbooks:
            if pb in playbook_stats:
                s = playbook_stats[pb]
                # Reconstruct approximate pnl list from stats
                # (we use total_pnl and samples as a proxy)
                pnls.extend([s["avg_pnl"]] * int(s["samples"]))
        if len(pnls) < MIN_SAMPLES_TO_ADAPT:
            return None
        return {
            "samples": len(pnls),
            "profit_factor": _profit_factor(pnls),
            "total_pnl": sum(pnls),
        }

    adjustments = {
        "bearish_threshold_old": current_bearish_threshold,
        "bullish_threshold_old": current_bullish_threshold,
        "bearish_threshold_new": current_bearish_threshold,
        "bullish_threshold_new": current_bullish_threshold,
        "bearish_reason": "no_change",
        "bullish_reason": "no_change",
    }

    # Bearish
    bs = _combined_stats(BEARISH_PLAYBOOKS)
    if bs:
        pf = bs["profit_factor"]
        if pf < UNDERPERFORM_PF:
            new = min(current_bearish_threshold + THRESHOLD_STEP_UP, BEARISH_THRESHOLD_MAX)
            adjustments["bearish_threshold_new"] = round(new, 2)
            adjustments["bearish_reason"] = f"underperforming pf={pf:.2f} → raised threshold"
        elif pf > OUTPERFORM_PF:
            new = max(current_bearish_threshold - THRESHOLD_STEP_DOWN, BEARISH_THRESHOLD_MIN)
            adjustments["bearish_threshold_new"] = round(new, 2)
            adjustments["bearish_reason"] = f"outperforming pf={pf:.2f} → relaxed threshold"
        else:
            adjustments["bearish_reason"] = f"pf={pf:.2f} within target range — no change"
    else:
        adjustments["bearish_reason"] = f"insufficient samples (<{MIN_SAMPLES_TO_ADAPT})"

    # Bullish
    bs2 = _combined_stats(BULLISH_PLAYBOOKS)
    if bs2:
        pf = bs2["profit_factor"]
        if pf < UNDERPERFORM_PF:
            new = min(current_bullish_threshold + THRESHOLD_STEP_UP, BULLISH_THRESHOLD_MAX)
            adjustments["bullish_threshold_new"] = round(new, 2)
            adjustments["bullish_reason"] = f"underperforming pf={pf:.2f} → raised threshold"
        elif pf > OUTPERFORM_PF:
            new = max(current_bullish_threshold - THRESHOLD_STEP_DOWN, BULLISH_THRESHOLD_MIN)
            adjustments["bullish_threshold_new"] = round(new, 2)
            adjustments["bullish_reason"] = f"outperforming pf={pf:.2f} → relaxed threshold"
        else:
            adjustments["bullish_reason"] = f"pf={pf:.2f} within target range — no change"
    else:
        adjustments["bullish_reason"] = f"insufficient samples (<{MIN_SAMPLES_TO_ADAPT})"

    return adjustments


# ── learning store integration ────────────────────────────────────────────────

def _feed_outcomes_to_learning_store(trades: list[dict], session_date: str) -> int:
    """Log today's trade outcomes into the LearningStore SQLite DB."""
    try:
        from market_ai.intraday_defined_risk.learning import LearningStore
        store = LearningStore(db_path=LEARNING_DB)
    except Exception as e:
        _log(f"[learning] Failed to import LearningStore: {e}")
        return 0

    logged = 0
    for t in trades:
        try:
            store.log_outcome(
                session_date=session_date,
                strategy=str(t.get("strategy") or "UNKNOWN"),
                pnl_rupees=float(t.get("pnl_rupees") or 0.0),
                max_drawdown_rupees=abs(float(t.get("mae_rupees") or 0.0)),
                margin_utilisation=0.05,  # approximate — refine when margin data is available
                features={
                    "playbook":           t.get("playbook"),
                    "market_state":       t.get("market_state"),
                    "bearish_trade_score": t.get("bearish_trade_score"),
                    "exit_reason":        t.get("exit_reason"),
                    "rv30_pct":           0.0,   # populated when feature history available
                    "credit_width_ratio": 0.0,
                },
            )
            logged += 1
        except Exception as e:
            _log(f"[learning] log_outcome error: {e}")

    return logged


def _run_calibration(dry_run: bool = False) -> dict:
    """Run the calibration engine and return a status dict."""
    try:
        from market_ai.intraday_defined_risk.learning import LearningStore
        store = LearningStore(db_path=LEARNING_DB)
        result = store.calibrate(window=60)
        status = {
            "status": result.status,
            "reasons": result.reasons,
            "objective_before": result.objective_before,
            "objective_after": result.objective_after,
        }
        if result.candidate_parameters:
            status["candidate"] = {
                "bearish_trade_score_threshold": result.candidate_parameters.bearish_trade_score_threshold,
                "bullish_trade_score_threshold": result.candidate_parameters.bullish_trade_score_threshold,
            }
        return status
    except Exception as e:
        return {"status": "ERROR", "reasons": [str(e)]}


def _get_current_thresholds() -> tuple[float, float]:
    """Read current thresholds from active AdaptiveParameters in SQLite."""
    try:
        from market_ai.intraday_defined_risk.learning import LearningStore
        store = LearningStore(db_path=LEARNING_DB)
        params = store.active_parameters()
        return params.bearish_trade_score_threshold, params.bullish_trade_score_threshold
    except Exception:
        return 6.5, 7.5


def _apply_threshold_update(
    adjustments: dict,
    dry_run: bool = False,
) -> bool:
    """Write new thresholds into AdaptiveParameters SQLite AND runtime config."""
    bear_new  = adjustments["bearish_threshold_new"]
    bull_new  = adjustments["bullish_threshold_new"]
    bear_old  = adjustments["bearish_threshold_old"]
    bull_old  = adjustments["bullish_threshold_old"]

    if bear_new == bear_old and bull_new == bull_old:
        _log("[threshold] No changes to apply.")
        return False

    if dry_run:
        _log(f"[threshold] DRY-RUN — would set bear={bear_new}, bull={bull_new}")
        return True

    # 1. Update AdaptiveParameters in SQLite
    try:
        import sqlite3, json as _json
        from dataclasses import asdict
        from market_ai.intraday_defined_risk.learning import LearningStore
        from market_ai.intraday_defined_risk.data_models import AdaptiveParameters

        store = LearningStore(db_path=LEARNING_DB)
        current = store.active_parameters()
        updated = AdaptiveParameters(
            **{**asdict(current),
               "bearish_trade_score_threshold": bear_new,
               "bullish_trade_score_threshold": bull_new}
        )
        with sqlite3.connect(LEARNING_DB) as conn:
            conn.execute(
                "UPDATE parameter_state SET active_json = ? WHERE singleton_id = 1",
                (_json.dumps(asdict(updated)),),
            )
            conn.commit()
        _log(f"[threshold] SQLite updated: bear {bear_old}→{bear_new}, bull {bull_old}→{bull_new}")
    except Exception as e:
        _log(f"[threshold] SQLite update error: {e}")

    # 2. Update runtime config JSON (belt-and-suspenders)
    try:
        cfg = json.loads(RUNTIME_CONFIG.read_text())
        cfg["bearish_score_margin"]         = bear_new  # used by evaluate_entry_gate
        cfg["paper_override_bearish_score_threshold"] = max(bear_new - 0.5, BEARISH_THRESHOLD_MIN)
        cfg["updated_at"] = datetime.now(IST).isoformat()
        RUNTIME_CONFIG.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
        _log(f"[threshold] runtime_config updated")
    except Exception as e:
        _log(f"[threshold] runtime_config update error: {e}")

    return True


# ── Telegram report ───────────────────────────────────────────────────────────

def _build_telegram_report(
    session_date: str,
    trades: list[dict],
    missed: list[dict],
    playbook_stats: dict[str, dict],
    adjustments: dict,
    calibration: dict,
) -> str:
    today_pnl = sum(t["pnl_rupees"] for t in trades)
    n_trades  = len(trades)
    n_wins    = sum(1 for t in trades if t["pnl_rupees"] > 0)
    n_losses  = sum(1 for t in trades if t["pnl_rupees"] < 0)
    pnl_emoji = "🟢" if today_pnl >= 0 else "🔴"

    lines = [
        f"📊 <b>EOD Learning Report — {session_date}</b>",
        "",
        f"{'─'*36}",
        f"<b>Today's Performance</b>",
        f"  Trades: {n_trades}  |  W: {n_wins}  L: {n_losses}",
        f"  {pnl_emoji} Net P&amp;L: ₹{today_pnl:+,.0f}",
    ]

    # Per-trade breakdown
    if trades:
        lines.append("")
        for t in trades:
            emoji = "✅" if t["pnl_rupees"] > 0 else "❌"
            exit_t = (t.get("exit_timestamp") or "")[-8:-3] if t.get("exit_timestamp") else "?"
            lines.append(
                f"  {emoji} [{exit_t}] {t['playbook']} | {t['strategy']}"
                f"\n      P&amp;L: ₹{t['pnl_rupees']:+,.0f}  |  Exit: {t['exit_reason']}"
            )

    if n_trades == 0:
        lines.append("  ⚠️ No paper trades executed today.")

    # Missed opportunities
    high_missed = [m for m in missed if m["score"] >= 6.0]
    if high_missed:
        lines.append("")
        lines.append(f"<b>Missed Opportunities</b> (score ≥ 6.0)")
        for m in high_missed[:4]:
            lines.append(
                f"  ⚠️ {m['playbook']} | score={m['score']:.2f}"
                f"\n      Blocked: {m['block_reason']}"
            )

    # Rolling playbook stats (only playbooks with ≥2 trades)
    notable = {pb: s for pb, s in playbook_stats.items() if s["samples"] >= 2}
    if notable:
        lines.append("")
        lines.append(f"<b>Playbook Stats (last {ROLLING_WINDOW_SESSIONS}d)</b>")
        for pb, s in sorted(notable.items(), key=lambda x: x[1]["total_pnl"], reverse=True):
            emoji = "🟢" if s["total_pnl"] >= 0 else "🔴"
            lines.append(
                f"  {emoji} {pb[:30]}"
                f"\n      PF={s['profit_factor']:.2f}  WR={s['win_rate']*100:.0f}%"
                f"  n={s['samples']}  P&amp;L=₹{s['total_pnl']:+,.0f}"
            )

    # Threshold changes
    lines.append("")
    lines.append("<b>Threshold Adaptation</b>")
    bear_changed = adjustments["bearish_threshold_new"] != adjustments["bearish_threshold_old"]
    bull_changed = adjustments["bullish_threshold_new"] != adjustments["bullish_threshold_old"]
    if bear_changed:
        lines.append(
            f"  📐 Bearish: {adjustments['bearish_threshold_old']} → {adjustments['bearish_threshold_new']}"
            f"\n      ({adjustments['bearish_reason']})"
        )
    else:
        lines.append(f"  Bearish: {adjustments['bearish_threshold_old']} (no change)")
    if bull_changed:
        lines.append(
            f"  📐 Bullish: {adjustments['bullish_threshold_old']} → {adjustments['bullish_threshold_new']}"
            f"\n      ({adjustments['bullish_reason']})"
        )
    else:
        lines.append(f"  Bullish: {adjustments['bullish_threshold_old']} (no change)")

    # Calibration status
    cal_status = calibration.get("status", "?")
    cal_emoji  = {"PROMOTED": "🚀", "SHADOW": "🌒", "REJECTED": "🚫", "NO_SHADOW": "—"}.get(cal_status, "ℹ️")
    lines.append("")
    lines.append(f"<b>Calibration</b>: {cal_emoji} {cal_status}")
    for reason in calibration.get("reasons", [])[:2]:
        lines.append(f"  {reason}")

    lines.append("")
    lines.append("─" * 36)
    lines.append("🤖 Agent learns. Agent improves.")

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="End-of-day learning loop")
    parser.add_argument("--date",    default=None, help="Session date YYYY-MM-DD (default: today)")
    parser.add_argument("--dry-run", action="store_true", help="Compute but don't write changes")
    args = parser.parse_args()

    session_date = args.date or date.today().isoformat()
    dry_run      = args.dry_run

    _log(f"{'='*50}")
    _log(f"EOD Learning Loop — session={session_date}  dry_run={dry_run}")
    _log(f"{'='*50}")

    # 1. Parse today's trades
    trades = _parse_completed_trades(session_date)
    _log(f"Completed trades today: {len(trades)}")
    for t in trades:
        _log(f"  {t['playbook']}  pnl=₹{t['pnl_rupees']:+,.0f}  exit={t['exit_reason']}")

    # 2. Identify missed opportunities
    missed = _parse_missed_opportunities(session_date)
    _log(f"Missed TRADABLE setups: {len(missed)}")
    for m in missed[:5]:
        _log(f"  {m['playbook']}  score={m['score']:.2f}  block={m['block_reason']}")

    # 3. Feed outcomes to learning store
    logged = 0
    if trades and not dry_run:
        logged = _feed_outcomes_to_learning_store(trades, session_date)
        _log(f"Outcomes logged to SQLite: {logged}")
    elif trades and dry_run:
        _log(f"DRY-RUN: would log {len(trades)} outcomes to SQLite")

    # 4. Rolling playbook stats
    playbook_stats = _compute_rolling_playbook_stats()
    _log(f"Playbook stats computed for {len(playbook_stats)} playbooks")

    # 5. Compute threshold adjustments
    bear_current, bull_current = _get_current_thresholds()
    _log(f"Current thresholds: bear={bear_current}, bull={bull_current}")
    adjustments = _compute_threshold_adjustments(playbook_stats, bear_current, bull_current)
    _log(f"Threshold adjustments: bear {adjustments['bearish_threshold_old']}→{adjustments['bearish_threshold_new']} ({adjustments['bearish_reason']})")
    _log(f"Threshold adjustments: bull {adjustments['bullish_threshold_old']}→{adjustments['bullish_threshold_new']} ({adjustments['bullish_reason']})")

    # 6. Apply threshold changes
    changed = _apply_threshold_update(adjustments, dry_run=dry_run)

    # 7. Run calibration engine
    calibration = _run_calibration(dry_run=dry_run)
    _log(f"Calibration: {calibration.get('status')} — {calibration.get('reasons', [])}")

    # 8. Write JSON report
    report = {
        "session_date":    session_date,
        "generated_at":    datetime.now(IST).isoformat(),
        "dry_run":         dry_run,
        "trades":          trades,
        "missed_setups":   missed,
        "playbook_stats":  playbook_stats,
        "threshold_adjustments": adjustments,
        "threshold_changed": changed,
        "calibration":     calibration,
        "outcomes_logged": logged,
        "today_pnl":       sum(t["pnl_rupees"] for t in trades),
        "today_trades":    len(trades),
        "today_wins":      sum(1 for t in trades if t["pnl_rupees"] > 0),
    }
    if not dry_run:
        EOD_REPORT_FILE.write_text(json.dumps(report, indent=2, default=str))
        PLAYBOOK_STATS_FILE.write_text(json.dumps(playbook_stats, indent=2, default=str))
        _log(f"Reports written to {EOD_REPORT_FILE.name} and {PLAYBOOK_STATS_FILE.name}")

    # 9. Telegram report
    msg = _build_telegram_report(
        session_date=session_date,
        trades=trades,
        missed=missed,
        playbook_stats=playbook_stats,
        adjustments=adjustments,
        calibration=calibration,
    )
    _log("Sending Telegram report...")
    if not dry_run:
        ok = _send_telegram(msg)
        _log(f"Telegram: {'sent ✓' if ok else 'failed ✗'}")
    else:
        _log("DRY-RUN: Telegram message preview:")
        _log(msg)

    _log("EOD learning loop complete.")
    today_pnl = sum(t["pnl_rupees"] for t in trades)
    _log(f"Summary: {len(trades)} trades  |  P&L: ₹{today_pnl:+,.0f}  |  calibration={calibration.get('status')}")


if __name__ == "__main__":
    main()
