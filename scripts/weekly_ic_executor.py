#!/usr/bin/env python3
"""Weekly Iron Condor Executor.

Runs Tuesday 3:35 PM IST (after weekly expiry settles).

What it does:
  1. Reads live NIFTY option chain
  2. Analyses market: IV, PCR, OI walls, daily trend
  3. Selects optimal Iron Condor strikes (sell near-OTM, buy far protection)
  4. Resolves Dhan security IDs for all 4 legs
  5. Saves pending trade plan to state/weekly_ic_pending.json
  6. Sends Telegram with full plan + one-click deploy link

The deploy link (http://localhost:8000/api/weekly_ic/deploy?token=<uuid>)
opens in your browser and places all 4 orders via Dhan API.

Usage:
  python scripts/weekly_ic_executor.py           # runs only on Tuesday after 3:35 PM
  python scripts/weekly_ic_executor.py --now     # force-run immediately (testing)
  python scripts/weekly_ic_executor.py --close   # build and send close plan for open position
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_engine"))

STATE_DIR  = ROOT / "data_engine" / "market_ai" / "state"
CREDS_FILE = STATE_DIR / "creds.json"
PENDING_FILE  = STATE_DIR / "weekly_ic_pending.json"
POSITION_FILE = STATE_DIR / "weekly_ic_position.json"

IST = timezone(timedelta(hours=5, minutes=30))
LOT_SIZE        = 65
DEFAULT_LOTS    = 3
WING_WIDTH_CALL = 200.0   # points — call side protection width
WING_WIDTH_PUT  = 200.0   # points — put side protection width
MARKET_CLOSE    = (15, 30)
ENTRY_AFTER     = (15, 35)
EXIT_BEFORE     = (15, 20)   # exit legs before 3:20 PM on expiry day

UI_BASE = "http://localhost:8000"


# ── helpers ──────────────────────────────────────────────────────────────────

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


def _next_tuesday(from_date: date) -> date:
    days = (1 - from_date.weekday()) % 7
    return from_date + timedelta(days=(days or 7))


def _round_strike(value: float, grid: float = 50.0) -> float:
    return round(value / grid) * grid


# ── market data ──────────────────────────────────────────────────────────────

def _get_nifty_ltp(creds: dict) -> float | None:
    try:
        r = requests.post(
            "https://api.dhan.co/v2/marketfeed/ltp",
            headers=_dhan_headers(creds),
            json={"IDX_I": [13]},
            timeout=8,
        )
        price = r.json().get("data", {}).get("IDX_I", {}).get("13", {}).get("last_price")
        return float(price) if price else None
    except Exception:
        return None


def _get_raw_chain(creds: dict, expiry: str) -> dict[str, Any]:
    try:
        r = requests.post(
            "https://api.dhan.co/optionchain",
            headers=_dhan_headers(creds),
            json={"UnderlyingScrip": 13, "UnderlyingSeg": "IDX_I", "Expiry": expiry},
            timeout=10,
        )
        return r.json()
    except Exception as e:
        print(f"[chain] {e}", file=sys.stderr)
        return {}


def _parse_chain(raw: dict, spot: float) -> dict[str, Any]:
    """Parse raw Dhan option chain into a strike-keyed dict with ltp, oi, iv, security_id."""
    data = raw.get("data") or raw
    if isinstance(data.get("data"), dict):
        data = data["data"]
    spot_live = float(data.get("last_price") or data.get("lastPrice") or spot)
    oc = data.get("oc") or {}

    strikes: dict[float, dict[str, Any]] = {}
    for strike_key, node in oc.items():
        try:
            k = float(strike_key)
        except Exception:
            continue
        ce = node.get("ce") or {}
        pe = node.get("pe") or {}
        greeks_ce = ce.get("greeks") or {}
        greeks_pe = pe.get("greeks") or {}
        strikes[k] = {
            "ce_ltp":  float(ce.get("last_price") or ce.get("ltp") or 0),
            "ce_bid":  float(ce.get("top_bid_price") or 0),
            "ce_ask":  float(ce.get("top_ask_price") or 0),
            "ce_oi":   float(ce.get("oi") or 0),
            "ce_iv":   float(ce.get("implied_volatility") or greeks_ce.get("iv") or 0),
            "ce_sid":  ce.get("SEM_SMST_SECURITY_ID") or ce.get("security_id"),
            "pe_ltp":  float(pe.get("last_price") or pe.get("ltp") or 0),
            "pe_bid":  float(pe.get("top_bid_price") or 0),
            "pe_ask":  float(pe.get("top_ask_price") or 0),
            "pe_oi":   float(pe.get("oi") or 0),
            "pe_iv":   float(pe.get("implied_volatility") or greeks_pe.get("iv") or 0),
            "pe_sid":  pe.get("SEM_SMST_SECURITY_ID") or pe.get("security_id"),
        }
    return {"spot": spot_live, "strikes": strikes}


def _resolve_sid_from_scrip(symbol: str, expiry: str, strike: float, opt: str) -> int | None:
    """Fallback: resolve security ID from Dhan scrip master CSV."""
    try:
        from market_ai.modules.data_fetch.dhan_scrip_cache import resolve_option_security_id
        return resolve_option_security_id(symbol, expiry, strike, opt)
    except Exception:
        return None


# ── strategy selection ────────────────────────────────────────────────────────

def _analyse(parsed: dict) -> dict[str, Any]:
    spot    = parsed["spot"]
    strikes = parsed["strikes"]

    if not strikes:
        return {"ok": False, "reason": "Empty option chain"}

    available = sorted(strikes.keys())

    # ATM IV
    atm_keys = sorted(available, key=lambda k: abs(k - spot))[:6]
    ivs = []
    for k in atm_keys:
        s = strikes[k]
        if s["ce_iv"] > 0: ivs.append(s["ce_iv"])
        if s["pe_iv"] > 0: ivs.append(s["pe_iv"])
    avg_iv = sum(ivs) / len(ivs) if ivs else 15.0

    # Expected move (1σ, 7 calendar days)
    expected_move = spot * (avg_iv / 100) * math.sqrt(7 / 365)

    # PCR
    total_call_oi = sum(s["ce_oi"] for s in strikes.values())
    total_put_oi  = sum(s["pe_oi"] for s in strikes.values())
    pcr = total_put_oi / total_call_oi if total_call_oi > 0 else 1.0

    # OI walls
    call_wall = max(available, key=lambda k: strikes[k]["ce_oi"], default=spot + 500)
    put_wall  = min(available, key=lambda k: strikes[k]["pe_oi"], default=spot - 500)
    # Correct direction
    call_wall_oi_strikes = [k for k in available if k >= spot]
    put_wall_oi_strikes  = [k for k in available if k <= spot]
    if call_wall_oi_strikes:
        call_wall = max(call_wall_oi_strikes, key=lambda k: strikes[k]["ce_oi"])
    if put_wall_oi_strikes:
        put_wall  = max(put_wall_oi_strikes,  key=lambda k: strikes[k]["pe_oi"])

    return {
        "ok": True,
        "spot": spot,
        "avg_iv": round(avg_iv, 2),
        "expected_move": round(expected_move, 0),
        "pcr": round(pcr, 2),
        "call_wall": call_wall,
        "put_wall": put_wall,
        "available_strikes": available,
    }


def _select_legs(
    parsed: dict,
    analysis: dict,
    expiry: str,
    lots: int = DEFAULT_LOTS,
) -> dict[str, Any] | None:
    spot     = analysis["spot"]
    em       = analysis["expected_move"]
    iv       = analysis["avg_iv"]
    strikes  = parsed["strikes"]
    available = analysis["available_strikes"]

    # Short strikes: 1.15× expected move, pulled in toward OI wall, snapped to 50pt grid
    raw_sc = spot + em * 1.15
    raw_sp = spot - em * 1.15
    short_call_raw = min(raw_sc, analysis["call_wall"] - 50)
    short_put_raw  = max(raw_sp, analysis["put_wall"]  + 50)
    short_call = _round_strike(short_call_raw)
    short_put  = _round_strike(short_put_raw)
    long_call  = short_call + WING_WIDTH_CALL
    long_put   = short_put  - WING_WIDTH_PUT

    # Snap to nearest available strikes
    def nearest(target: float) -> float:
        return min(available, key=lambda k: abs(k - target))

    short_call = nearest(short_call)
    short_put  = nearest(short_put)
    long_call  = nearest(long_call)
    long_put   = nearest(long_put)

    # Safety checks
    if short_call >= long_call or short_put <= long_put:
        return None
    if (short_call - spot) < em * 0.7 or (spot - short_put) < em * 0.7:
        return None   # strikes too close to spot

    sc = strikes.get(short_call, {})
    lc = strikes.get(long_call,  {})
    sp = strikes.get(short_put,  {})
    lp = strikes.get(long_put,   {})

    # Credit: sell bid, buy ask (conservative)
    sc_credit = sc.get("ce_bid") or sc.get("ce_ltp") or 0
    lc_cost   = lc.get("ce_ask") or lc.get("ce_ltp") or 0
    sp_credit = sp.get("pe_bid") or sp.get("pe_ltp") or 0
    lp_cost   = lp.get("pe_ask") or lp.get("pe_ltp") or 0

    net_credit_per_unit = (sc_credit - lc_cost) + (sp_credit - lp_cost)
    if net_credit_per_unit <= 0:
        return None

    gross_credit  = net_credit_per_unit * LOT_SIZE * lots
    max_loss_call = (short_call - long_call + net_credit_per_unit) * LOT_SIZE * lots  # negative = capped loss
    max_loss      = abs((WING_WIDTH_CALL - net_credit_per_unit) * LOT_SIZE * lots)
    target_keep   = gross_credit * 0.70

    # Security IDs — from chain first, scrip master fallback
    def sid(data: dict, key: str, strike: float, opt: str) -> int | None:
        v = data.get(key)
        if v:
            try: return int(v)
            except Exception: pass
        return _resolve_sid_from_scrip("NIFTY", expiry, strike, opt)

    sc_sid = sid(sc, "ce_sid", short_call, "CE")
    lc_sid = sid(lc, "ce_sid", long_call,  "CE")
    sp_sid = sid(sp, "pe_sid", short_put,  "PE")
    lp_sid = sid(lp, "pe_sid", long_put,   "PE")

    quantity = LOT_SIZE * lots

    # Confidence
    score, notes = _score_confidence(analysis)

    return {
        "lots": lots,
        "quantity": quantity,
        "expiry": expiry,
        "spot_at_entry": spot,
        "short_call": short_call, "long_call": long_call,
        "short_put":  short_put,  "long_put":  long_put,
        "wing_width": WING_WIDTH_CALL,
        "sc_ltp": sc_credit, "lc_ltp": lc_cost,
        "sp_ltp": sp_credit, "lp_ltp": lp_cost,
        "net_credit_per_unit": round(net_credit_per_unit, 2),
        "gross_credit": round(gross_credit, 0),
        "target_keep": round(target_keep, 0),
        "max_loss": round(max_loss, 0),
        "stop_trigger_loss": round(gross_credit * 2, 0),
        "margin_required": round(max_loss * 1.15, 0),
        "sc_sid": sc_sid, "lc_sid": lc_sid,
        "sp_sid": sp_sid, "lp_sid": lp_sid,
        "confidence": score,
        "confidence_label": "HIGH" if score >= 6 else ("MEDIUM" if score >= 4 else "LOW"),
        "confidence_notes": notes,
    }


def _score_confidence(a: dict) -> tuple[int, list[str]]:
    score, notes = 0, []
    iv = a["avg_iv"]
    if 14 <= iv <= 22:
        score += 3; notes.append(f"✅ IV={iv:.1f}% is in the sweet spot (14–22%)")
    elif iv < 14:
        score += 1; notes.append(f"⚠️ IV={iv:.1f}% is low — premium will be thin")
    else:
        score += 1; notes.append(f"⚠️ IV={iv:.1f}% is elevated — wider moves possible")

    pcr = a["pcr"]
    if 0.8 <= pcr <= 1.4:
        score += 2; notes.append(f"✅ PCR={pcr:.2f} — market is balanced, IC ideal")
    elif pcr > 1.5:
        score += 1; notes.append(f"🔵 PCR={pcr:.2f} — put-heavy, market may be cautious")
    else:
        score -= 1; notes.append(f"🔴 PCR={pcr:.2f} — call-heavy, bullish pressure")

    em = a["expected_move"]
    if em < 400:
        score += 2; notes.append(f"✅ Expected move ±{em:.0f}pts — low vol, IC very suitable")
    elif em < 600:
        score += 1; notes.append(f"✅ Expected move ±{em:.0f}pts — moderate, IC suitable")
    else:
        score -= 1; notes.append(f"⚠️ Expected move ±{em:.0f}pts — high vol week, widen strikes")

    return score, notes


# ── pending trade plan ────────────────────────────────────────────────────────

def _save_pending(plan: dict, token: str) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {**plan, "token": token, "created_at": datetime.now(IST).isoformat(), "status": "PENDING"}
    PENDING_FILE.write_text(json.dumps(payload, indent=2))


def _load_pending() -> dict:
    try:
        return json.loads(PENDING_FILE.read_text())
    except Exception:
        return {}


# ── Telegram message ──────────────────────────────────────────────────────────

def _telegram_entry_msg(plan: dict, token: str) -> str:
    conf_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(plan["confidence_label"], "⚪")
    notes = "\n".join(f"  {n}" for n in plan["confidence_notes"])
    deploy_url = f"{UI_BASE}/api/weekly_ic/deploy?token={token}"
    close_later_url = f"{UI_BASE}/api/weekly_ic/close?token={token}"

    sid_status = "✅" if all([plan.get("sc_sid"), plan.get("lc_sid"), plan.get("sp_sid"), plan.get("lp_sid")]) else "⚠️ (some IDs missing — verify manually)"

    return (
        f"📊 <b>Weekly IC Plan — Expiry {plan['expiry']}</b>\n"
        f"NIFTY spot: <b>{plan['spot_at_entry']:,.0f}</b>  |  "
        f"Expected move: ±{plan.get('expected_move', '?')} pts\n\n"
        f"<b>4 Legs ({plan['lots']} lots × {LOT_SIZE} units = {plan['quantity']} qty each)</b>\n\n"
        f"  📉 SELL {plan['short_call']:,.0f} CE  @ ~{plan['sc_ltp']:.1f}\n"
        f"  🛡 BUY  {plan['long_call']:,.0f} CE  @ ~{plan['lc_ltp']:.1f}\n"
        f"  📈 SELL {plan['short_put']:,.0f} PE  @ ~{plan['sp_ltp']:.1f}\n"
        f"  🛡 BUY  {plan['long_put']:,.0f} PE  @ ~{plan['lp_ltp']:.1f}\n\n"
        f"Security IDs: {sid_status}\n\n"
        f"Net credit:      ~Rs {plan['gross_credit']:,.0f}\n"
        f"Target (70%):    ~Rs {plan['target_keep']:,.0f}/week\n"
        f"Max loss:         Rs {plan['max_loss']:,.0f} (one side breach)\n"
        f"Stop at loss:     Rs {plan['stop_trigger_loss']:,.0f}\n"
        f"Margin needed:   ~Rs {plan['margin_required']:,.0f}\n\n"
        f"{conf_emoji} <b>Confidence: {plan['confidence_label']} ({plan['confidence']}/8)</b>\n"
        f"{notes}\n\n"
        f"{'─'*36}\n"
        f"<b>👇 Tap to deploy (places all 4 orders):</b>\n"
        f"<a href='{deploy_url}'>✅ DEPLOY TRADE</a>\n\n"
        f"<i>Exit: Next Tuesday before 3:20 PM\n"
        f"Or tap when ready: <a href='{close_later_url}'>🔴 CLOSE ALL LEGS</a></i>"
    )


def _telegram_skip_msg(reason: str) -> str:
    return f"⏭ <b>Weekly IC — SKIPPED</b>\n\nReason: {reason}\n\nNext opportunity: Next Tuesday 3:35 PM."


# ── close plan ────────────────────────────────────────────────────────────────

def _telegram_close_msg(position: dict, token: str) -> str:
    close_url = f"{UI_BASE}/api/weekly_ic/close?token={token}"
    entry_credit = position.get("gross_credit", 0)
    expiry = position.get("expiry", "?")
    return (
        f"⏰ <b>Weekly IC Exit — {expiry}</b>\n\n"
        f"Entry credit: Rs {entry_credit:,.0f}\n\n"
        f"<b>Legs to close (BUY back short, SELL long):</b>\n"
        f"  BUY  {position.get('short_call'):,.0f} CE\n"
        f"  SELL {position.get('long_call'):,.0f} CE\n"
        f"  BUY  {position.get('short_put'):,.0f} PE\n"
        f"  SELL {position.get('long_put'):,.0f} PE\n\n"
        f"<b>👇 Tap to close all 4 legs:</b>\n"
        f"<a href='{close_url}'>🔴 CLOSE ALL LEGS NOW</a>"
    )


# ── entry flow ────────────────────────────────────────────────────────────────

def run_entry(force: bool = False) -> None:
    now = datetime.now(IST)
    weekday = now.weekday()     # 1 = Tuesday
    hhmm = now.hour * 100 + now.minute
    entry_hhmm = ENTRY_AFTER[0] * 100 + ENTRY_AFTER[1]

    if not force:
        if weekday != 1:
            print(f"[weekly_ic] Not Tuesday — skipping (today={now.strftime('%A')}).")
            return
        if hhmm < entry_hhmm:
            print(f"[weekly_ic] Before 3:35 PM IST — skipping.")
            return

    creds = _load_creds()
    bot_token = str(creds.get("telegram_bot_token") or "").strip()
    chat_id   = str(creds.get("telegram_chat_id") or "").strip()

    if not bot_token or not chat_id:
        print("[weekly_ic] Telegram not configured.")
        return

    today = now.date()
    next_expiry = _next_tuesday(today + timedelta(days=1))
    expiry_str  = next_expiry.isoformat()
    print(f"[weekly_ic] Building IC plan for expiry {expiry_str} ...")

    spot = _get_nifty_ltp(creds)
    if not spot:
        _send_telegram(bot_token, chat_id, _telegram_skip_msg("Could not fetch NIFTY spot — check Dhan token."))
        return

    raw = _get_raw_chain(creds, expiry_str)
    parsed = _parse_chain(raw, spot)

    if not parsed["strikes"]:
        _send_telegram(bot_token, chat_id, _telegram_skip_msg(
            f"Option chain empty for {expiry_str}. Next week's contracts may not be listed yet."
        ))
        return

    analysis = _analyse(parsed)
    if not analysis["ok"]:
        _send_telegram(bot_token, chat_id, _telegram_skip_msg(analysis.get("reason", "Analysis failed.")))
        return

    # Skip if IV is very high (>28%) — high gap risk
    if analysis["avg_iv"] > 28:
        _send_telegram(bot_token, chat_id, _telegram_skip_msg(
            f"IV={analysis['avg_iv']:.1f}% is too high (>28%). Holding cash this week to avoid gap risk."
        ))
        return

    plan = _select_legs(parsed, analysis, expiry_str, lots=DEFAULT_LOTS)
    if plan is None:
        _send_telegram(bot_token, chat_id, _telegram_skip_msg(
            "Could not build valid IC legs — strikes too close to spot or zero credit. Skip this week."
        ))
        return

    # Add analysis fields to plan for display
    plan["avg_iv"]        = analysis["avg_iv"]
    plan["expected_move"] = analysis["expected_move"]
    plan["pcr"]           = analysis["pcr"]

    token = str(uuid.uuid4()).replace("-", "")[:16]
    _save_pending(plan, token)
    msg = _telegram_entry_msg(plan, token)
    ok = _send_telegram(bot_token, chat_id, msg)
    print(f"[weekly_ic] Plan sent. Confidence={plan['confidence_label']}. Token={token}. Telegram={'OK' if ok else 'FAILED'}.")


# ── close flow ────────────────────────────────────────────────────────────────

def run_close_alert() -> None:
    """Send a Telegram close alert if there's an open weekly IC position."""
    creds = _load_creds()
    bot_token = str(creds.get("telegram_bot_token") or "").strip()
    chat_id   = str(creds.get("telegram_chat_id") or "").strip()
    if not bot_token or not chat_id:
        return

    position = {}
    if POSITION_FILE.exists():
        try:
            position = json.loads(POSITION_FILE.read_text())
        except Exception:
            pass
    if not position or position.get("status") != "OPEN":
        pending = _load_pending()
        if pending and pending.get("status") == "DEPLOYED":
            position = pending
    if not position:
        print("[weekly_ic_close] No open position found.")
        return

    token = str(uuid.uuid4()).replace("-", "")[:16]
    # Save a close token
    close_state = {**position, "close_token": token, "close_created_at": datetime.now(IST).isoformat()}
    PENDING_FILE.write_text(json.dumps(close_state, indent=2))
    msg = _telegram_close_msg(position, token)
    _send_telegram(bot_token, chat_id, msg)
    print(f"[weekly_ic_close] Close alert sent. Token={token}.")


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly IC Executor")
    parser.add_argument("--now",   action="store_true", help="Force entry run (ignore day/time check)")
    parser.add_argument("--close", action="store_true", help="Send close alert for open position")
    args = parser.parse_args()

    if args.close:
        run_close_alert()
    else:
        run_entry(force=args.now)
