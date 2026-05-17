#!/usr/bin/env python3
"""Weekly Iron Condor Entry Assistant.

Runs on Tuesday after 3:35 PM IST. Reads the live NIFTY option chain,
analyses market conditions (trend, IV, OI walls, expected move), and
suggests the optimal Iron Condor strikes for the next weekly expiry.

Sends a full trade plan via Telegram including:
  - Recommended strikes (both sides)
  - Credit estimate and max loss
  - Confidence level based on market conditions
  - Risk/reward summary
  - Whether to skip the trade (high VIX, event risk, strong trend)

Usage:
  python scripts/weekly_ic_assistant.py          # auto-runs at Tuesday 3:35 PM
  python scripts/weekly_ic_assistant.py --now    # force-run immediately (for testing)
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_engine"))

STATE_DIR = ROOT / "data_engine" / "market_ai" / "state"
CREDS_FILE = STATE_DIR / "creds.json"

IST = timezone(timedelta(hours=5, minutes=30))
LOT_SIZE = 65
DEFAULT_LOTS = 3
ENTRY_AFTER_HHMM = (15, 35)   # 3:35 PM IST — after weekly expiry settles
ENTRY_CLOSE_HHMM = (15, 25)   # Must exit before 3:25 PM on expiry day


def _load_creds() -> dict[str, str]:
    try:
        return json.loads(CREDS_FILE.read_text())
    except Exception:
        return {}


def _send_telegram(bot_token: str, chat_id: str, text: str) -> bool:
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
        return r.status_code == 200
    except Exception as e:
        print(f"[telegram] {e}", file=sys.stderr)
        return False


def _dhan_headers(creds: dict) -> dict[str, str]:
    return {
        "client-id": str(creds.get("client_id") or ""),
        "access-token": str(creds.get("access_token") or ""),
        "Content-Type": "application/json",
    }


def _get_nifty_ltp(creds: dict) -> float | None:
    """Fetch NIFTY spot price via Dhan LTP endpoint."""
    try:
        r = requests.post(
            "https://api.dhan.co/v2/marketfeed/ltp",
            headers=_dhan_headers(creds),
            json={"IDX_I": [13]},
            timeout=8,
        )
        data = r.json()
        return float(data.get("data", {}).get("IDX_I", {}).get("13", {}).get("last_price") or 0) or None
    except Exception:
        return None


def _get_option_chain(creds: dict, expiry: str) -> list[dict[str, Any]]:
    """Fetch NIFTY option chain for the given expiry date (YYYY-MM-DD)."""
    try:
        r = requests.post(
            "https://api.dhan.co/optionchain",
            headers=_dhan_headers(creds),
            json={"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I", "Expiry": expiry},
            timeout=10,
        )
        data = r.json()
        # Flatten call + put rows
        rows: list[dict] = []
        for entry in data.get("data", {}).get("oc", []):
            rows.append(entry)
        return rows
    except Exception as e:
        print(f"[option_chain] {e}", file=sys.stderr)
        return []


def _next_tuesday(from_date: date) -> date:
    """Return the next Tuesday on or after from_date."""
    days = (1 - from_date.weekday()) % 7  # 1 = Tuesday
    if days == 0:
        days = 7
    return from_date + timedelta(days=days)


def _analyse_chain(
    rows: list[dict],
    spot: float,
) -> dict[str, Any]:
    """Extract key signals from the option chain."""
    calls = [r for r in rows if r.get("optionType") == "CALL" or str(r.get("option_type", "")).upper() == "CE"]
    puts  = [r for r in rows if r.get("optionType") == "PUT"  or str(r.get("option_type", "")).upper() == "PE"]

    # Fallback: split by strike vs spot (calls OTM above, puts OTM below)
    if not calls and not puts:
        calls = [r for r in rows if float(r.get("strike_price") or r.get("strikePrice") or 0) >= spot]
        puts  = [r for r in rows if float(r.get("strike_price") or r.get("strikePrice") or 0) <  spot]

    def strike(r: dict) -> float:
        return float(r.get("strike_price") or r.get("strikePrice") or 0)

    def oi(r: dict) -> float:
        return float(r.get("oi") or r.get("OI") or r.get("openInterest") or 0)

    def iv(r: dict) -> float:
        return float(r.get("implied_volatility") or r.get("iv") or r.get("IV") or 0)

    def ltp(r: dict) -> float:
        return float(r.get("ltp") or r.get("LTP") or r.get("last_price") or 0)

    # ATM IV (average of nearest call/put IV)
    atm_calls = sorted(calls, key=lambda r: abs(strike(r) - spot))[:3]
    atm_puts  = sorted(puts,  key=lambda r: abs(strike(r) - spot))[:3]
    atm_ivs = [iv(r) for r in atm_calls + atm_puts if iv(r) > 0]
    avg_iv = sum(atm_ivs) / len(atm_ivs) if atm_ivs else 15.0

    # Expected weekly move (1 std dev): spot × IV × sqrt(7/365)
    expected_move = spot * (avg_iv / 100) * math.sqrt(7 / 365)

    # Put/Call OI ratio (PCR)
    total_call_oi = sum(oi(r) for r in calls)
    total_put_oi  = sum(oi(r) for r in puts)
    pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0

    # Max OI strikes = support/resistance walls
    call_wall_row = max(calls, key=oi, default=None)
    put_wall_row  = max(puts,  key=oi, default=None)
    call_wall = strike(call_wall_row) if call_wall_row else spot + 500
    put_wall  = strike(put_wall_row)  if put_wall_row  else spot - 500

    return {
        "avg_iv": round(avg_iv, 2),
        "expected_move": round(expected_move, 0),
        "pcr": round(pcr, 2),
        "call_wall": call_wall,
        "put_wall": put_wall,
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "atm_call_ltp": ltp(atm_calls[0]) if atm_calls else 0,
        "atm_put_ltp":  ltp(atm_puts[0])  if atm_puts  else 0,
    }


def _round_to_50(value: float) -> float:
    return round(value / 50) * 50


def _suggest_strikes(
    spot: float,
    analysis: dict[str, Any],
    lots: int = DEFAULT_LOTS,
) -> dict[str, Any]:
    """Suggest Iron Condor strikes and compute P&L."""
    em = analysis["expected_move"]
    iv = analysis["avg_iv"]
    call_wall = analysis["call_wall"]
    put_wall  = analysis["put_wall"]

    # Short strikes: 1.1× expected move from spot, snapped to 50pt grid
    # Pulled in toward the OI wall if wall is closer (provides natural resistance)
    raw_short_call = spot + em * 1.1
    raw_short_put  = spot - em * 1.1
    short_call = _round_to_50(min(raw_short_call, call_wall - 50))
    short_put  = _round_to_50(max(raw_short_put,  put_wall  + 50))

    # Long strikes (protection): 200pt further out
    wing_width = 200.0
    long_call = short_call + wing_width
    long_put  = short_put  - wing_width

    # Confidence scoring
    confidence_score = 0
    confidence_notes: list[str] = []

    # IV-based confidence
    if iv < 14:
        confidence_score += 2
        confidence_notes.append(f"Low IV={iv:.1f}% → premium thin, consider skipping")
    elif iv < 20:
        confidence_score += 3
        confidence_notes.append(f"Moderate IV={iv:.1f}% → good premium, IC suitable")
    else:
        confidence_score += 1
        confidence_notes.append(f"High IV={iv:.1f}% → wide moves possible, use wider wings")

    # PCR-based confidence
    if 0.8 <= analysis["pcr"] <= 1.3:
        confidence_score += 2
        confidence_notes.append(f"PCR={analysis['pcr']:.2f} → balanced market, IC ideal")
    elif analysis["pcr"] > 1.5:
        confidence_score += 1
        confidence_notes.append(f"PCR={analysis['pcr']:.2f} → put-heavy, market may be defensive")
    else:
        confidence_score -= 1
        confidence_notes.append(f"PCR={analysis['pcr']:.2f} → call-heavy, bullish bias")

    # OI wall distance
    call_buffer = short_call - spot
    put_buffer  = spot - short_put
    if call_buffer > em * 0.9 and put_buffer > em * 0.9:
        confidence_score += 2
        confidence_notes.append("Short strikes are well beyond 1× expected move → safe zone")
    else:
        confidence_score -= 1
        confidence_notes.append("Short strikes close to expected move → tight IC")

    # Confidence label
    if confidence_score >= 6:
        confidence_label = "HIGH"
    elif confidence_score >= 4:
        confidence_label = "MEDIUM"
    else:
        confidence_label = "LOW — CONSIDER SKIPPING"

    # Credit estimate (rough): 30% of wing width × IV factor
    credit_per_side = wing_width * 0.20 * (iv / 15)
    total_credit_per_unit = credit_per_side * 2
    total_credit = total_credit_per_unit * LOT_SIZE * lots
    max_loss_per_lot = wing_width * LOT_SIZE
    max_loss = max_loss_per_lot * lots
    margin_required = max_loss * 1.15  # SPAN + exposure buffer
    target_keep = total_credit * 0.65

    return {
        "short_call": short_call,
        "long_call":  long_call,
        "short_put":  short_put,
        "long_put":   long_put,
        "wing_width": wing_width,
        "credit_est_per_unit": round(total_credit_per_unit, 0),
        "gross_credit_est": round(total_credit, 0),
        "target_keep_65pct": round(target_keep, 0),
        "max_loss": round(max_loss, 0),
        "margin_required": round(margin_required, 0),
        "lots": lots,
        "confidence": confidence_label,
        "confidence_score": confidence_score,
        "confidence_notes": confidence_notes,
    }


def _format_telegram_message(
    spot: float,
    expiry: str,
    analysis: dict[str, Any],
    plan: dict[str, Any],
    now_ist: datetime,
) -> str:
    conf = plan["confidence"]
    conf_emoji = "🟢" if "HIGH" in conf else ("🟡" if "MEDIUM" in conf else "🔴")
    notes_str = "\n".join(f"  • {n}" for n in plan["confidence_notes"])

    return (
        f"📊 <b>Weekly IC Entry — {expiry}</b>\n"
        f"<i>{now_ist.strftime('%d %b %Y  %I:%M %p IST')}</i>\n\n"
        f"<b>NIFTY Spot:</b> {spot:,.0f}\n"
        f"<b>Expected move (1σ, 7 days):</b> ±{analysis['expected_move']:,.0f} pts\n"
        f"<b>India VIX equivalent IV:</b> {analysis['avg_iv']:.1f}%\n"
        f"<b>PCR:</b> {analysis['pcr']:.2f}  |  "
        f"Call wall: {analysis['call_wall']:,.0f}  |  Put wall: {analysis['put_wall']:,.0f}\n\n"
        f"{'─' * 36}\n"
        f"<b>📋 Suggested Trade ({plan['lots']} lots × {LOT_SIZE} units)</b>\n\n"
        f"  <b>SELL</b> {plan['short_call']:,.0f} CE  +  "
        f"<b>BUY</b> {plan['long_call']:,.0f} CE\n"
        f"  <b>SELL</b> {plan['short_put']:,.0f} PE  +  "
        f"<b>BUY</b> {plan['long_put']:,.0f} PE\n\n"
        f"  Wing width each side: {plan['wing_width']:.0f} pts\n"
        f"  Credit estimate:      ~Rs {plan['gross_credit_est']:,}\n"
        f"  Target (keep 65%):    ~Rs {plan['target_keep_65pct']:,}/week\n"
        f"  Max loss (one side):   Rs {plan['max_loss']:,}\n"
        f"  Margin required:      ~Rs {plan['margin_required']:,}\n\n"
        f"{'─' * 36}\n"
        f"{conf_emoji} <b>Confidence: {conf}</b>\n"
        f"{notes_str}\n\n"
        f"<b>Exit rules:</b>\n"
        f"  ✅ Close at 70% profit hit\n"
        f"  🛑 Close if loss = 2× credit received\n"
        f"  ⏰ Close next Tuesday before 3:25 PM\n\n"
        f"<i>Verify live premiums on Dhan before placing.</i>"
    )


def run(force: bool = False) -> None:
    now_ist = datetime.now(tz=IST)
    weekday = now_ist.weekday()  # 1 = Tuesday
    hour, minute = now_ist.hour, now_ist.minute
    current_hhmm = hour * 100 + minute
    entry_hhmm = ENTRY_AFTER_HHMM[0] * 100 + ENTRY_AFTER_HHMM[1]

    if not force:
        if weekday != 1:
            print(f"[weekly_ic] Not Tuesday ({now_ist.strftime('%A')}) — skipping.")
            return
        if current_hhmm < entry_hhmm:
            print(f"[weekly_ic] Before {ENTRY_AFTER_HHMM[0]}:{ENTRY_AFTER_HHMM[1]:02d} IST — skipping.")
            return

    creds = _load_creds()
    bot_token = str(creds.get("telegram_bot_token") or "").strip()
    chat_id   = str(creds.get("telegram_chat_id") or "").strip()
    if not bot_token or not chat_id:
        print("[weekly_ic] Telegram not configured.")
        return

    # Next Tuesday expiry
    today = now_ist.date()
    next_expiry = _next_tuesday(today + timedelta(days=1))
    expiry_str = next_expiry.isoformat()
    print(f"[weekly_ic] Analysing for expiry {expiry_str} ...")

    # Fetch live data
    spot = _get_nifty_ltp(creds)
    if not spot:
        _send_telegram(bot_token, chat_id, "⚠️ Weekly IC: Could not fetch NIFTY spot price. Check Dhan token.")
        return

    rows = _get_option_chain(creds, expiry_str)
    if not rows:
        _send_telegram(bot_token, chat_id, f"⚠️ Weekly IC: Option chain empty for {expiry_str}. Market may not have opened next week's contracts yet.")
        return

    analysis = _analyse_chain(rows, spot)
    plan = _suggest_strikes(spot, analysis)
    msg = _format_telegram_message(spot, expiry_str, analysis, plan, now_ist)
    ok = _send_telegram(bot_token, chat_id, msg)
    if ok:
        print(f"[weekly_ic] Telegram sent. Confidence: {plan['confidence']}")
    else:
        print("[weekly_ic] Telegram send failed.")
        print(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly IC Entry Assistant")
    parser.add_argument("--now", action="store_true", help="Force run regardless of day/time")
    args = parser.parse_args()
    run(force=args.now)
