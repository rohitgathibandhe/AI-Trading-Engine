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

from market_ai.dhan_wrapper import DhanWrapper  # noqa: E402

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
        dw = DhanWrapper(
            dhan_client_id=str(creds.get("client_id") or "").strip(),
            access_token=str(creds.get("access_token") or "").strip(),
        )
        return dw.get_option_chain(13, "IDX_I", expiry) or {}
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

    # Short strikes: 1.15× expected move, pulled toward OI wall but never closer than 0.9× em
    raw_sc = spot + em * 1.15
    raw_sp = spot - em * 1.15
    min_sc = spot + em * 0.9   # floor: never let wall drag short call too close to spot
    max_sp = spot - em * 0.9   # ceiling: symmetric for puts
    short_call_raw = max(min_sc, min(raw_sc, analysis["call_wall"] - 50))
    short_put_raw  = min(max_sp, max(raw_sp, analysis["put_wall"]  + 50))
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
        f"📋 <b>Weekly IC Preview — Expiry {plan['expiry']}</b>\n"
        f"NIFTY spot at close: <b>{plan['spot_at_entry']:,.0f}</b>  |  "
        f"Expected move: ±{plan.get('expected_move', '?')} pts\n\n"
        f"<b>Tentative legs ({plan['lots']} lots × {LOT_SIZE} = {plan['quantity']} qty each)</b>\n\n"
        f"  📉 SELL {plan['short_call']:,.0f} CE  @ ~{plan['sc_ltp']:.1f}\n"
        f"  🛡 BUY  {plan['long_call']:,.0f} CE  @ ~{plan['lc_ltp']:.1f}\n"
        f"  📈 SELL {plan['short_put']:,.0f} PE  @ ~{plan['sp_ltp']:.1f}\n"
        f"  🛡 BUY  {plan['long_put']:,.0f} PE  @ ~{plan['lp_ltp']:.1f}\n\n"
        f"Net credit:    ~₹{plan['gross_credit']:,.0f}\n"
        f"Target (70%):  ~₹{plan['target_keep']:,.0f}/week\n"
        f"Max loss:       ₹{plan['max_loss']:,.0f}  |  Stop: ₹{plan['stop_trigger_loss']:,.0f}\n"
        f"Margin needed: ~₹{plan['margin_required']:,.0f}\n\n"
        f"{conf_emoji} <b>Confidence: {plan['confidence_label']} ({plan['confidence']}/8)</b>\n"
        f"{notes}\n\n"
        f"{'─'*36}\n"
        f"⏳ <b>DO NOT deploy yet.</b>\n"
        f"A fresh morning review fires at <b>9:00 AM tomorrow</b> with updated strikes "
        f"based on live market data.\n\n"
        f"<i>Overnight gaps or news can shift optimal strikes significantly. "
        f"Deploy only after the 9 AM refresh confirms the plan.</i>\n\n"
        f"<i>Close link (use next Tuesday morning): <a href='{close_later_url}'>🔴 CLOSE ALL LEGS</a></i>"
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


# ── morning review flow ───────────────────────────────────────────────────────

def run_morning_review(force: bool = False) -> None:
    """Wednesday 9:00 AM IST — refresh pending IC plan with live market data.

    The entry plan is built Tuesday at 3:35 PM on post-close data.  Overnight
    gaps, news, or regime shifts can make those strikes dangerous by Wednesday
    morning.  This step:

      1. Checks whether a PENDING plan exists (from the previous evening).
      2. Re-fetches live spot + option chain for next Tuesday's expiry.
      3. Re-selects optimal strikes using *current* market data.
      4. Saves an updated pending plan (overwrites the stale one).
      5. Sends a fresh Telegram with updated strikes + a live deploy link.
         The user decides whether to deploy based on the morning context.

    If market conditions have deteriorated (IV > 28%, chain not available, etc.)
    it sends a SKIP alert and marks the plan as CANCELLED so the stale link
    can no longer deploy.
    """
    now = datetime.now(IST)
    weekday = now.weekday()   # 2 = Wednesday

    if not force:
        if weekday != 2:
            print(f"[weekly_ic_review] Not Wednesday — skipping (today={now.strftime('%A')}).")
            return
        hhmm = now.hour * 100 + now.minute
        if hhmm < 900:
            print("[weekly_ic_review] Before 9:00 AM IST — skipping.")
            return

    # Load the pending plan from last evening (also refresh ASSESSMENT_HOLD plans)
    pending = _load_pending()
    if not pending or pending.get("status") not in ("PENDING", "ASSESSMENT_HOLD", None):
        print(f"[weekly_ic_review] Status={pending.get('status') if pending else 'none'} — nothing to review.")
        return

    plan_age_hours = 0.0
    try:
        created = datetime.fromisoformat(str(pending.get("created_at") or ""))
        plan_age_hours = (now - created).total_seconds() / 3600.0
    except Exception:
        pass

    creds = _load_creds()
    bot_token = str(creds.get("telegram_bot_token") or "").strip()
    chat_id   = str(creds.get("telegram_chat_id") or "").strip()
    if not bot_token or not chat_id:
        print("[weekly_ic_review] Telegram not configured.")
        return

    expiry_str = str(pending.get("expiry") or "")
    old_sc = pending.get("short_call")
    old_sp = pending.get("short_put")
    old_spot = pending.get("spot_at_entry")

    print(f"[weekly_ic_review] Refreshing plan for {expiry_str} (plan was {plan_age_hours:.1f}h old) ...")

    # Re-fetch live market data
    # Market data is not available before 9:15 AM IST — if we're pre-market,
    # send a plan-summary (no fresh data needed) and defer to the 10:15 AM assessment.
    pre_market = (now.hour * 100 + now.minute) < 915
    spot = _get_nifty_ltp(creds)
    if not spot:
        if pre_market:
            # Normal — market hasn't opened yet. Just remind about the plan.
            _send_telegram(bot_token, chat_id,
                f"🌅 <b>Weekly IC Plan Ready — {expiry_str}</b>\n\n"
                f"Market not yet open (data available after 9:15 AM).\n\n"
                f"<b>Tentative strikes (built on Tuesday close data):</b>\n"
                f"  📉 SELL {old_sc:,.0f} CE  |  📈 SELL {old_sp:,.0f} PE\n"
                f"  Spot at plan: {old_spot:,.0f}  |  Credit: ~₹{pending.get('gross_credit',0):,.0f}\n\n"
                f"⏳ <b>Market assessment fires at 10:15 AM</b> with live data.\n"
                f"Strikes will be re-validated before any deploy link is issued.\n\n"
                f"<i>No action needed — assessment runs automatically.</i>"
            )
            print(f"[weekly_ic_review] Pre-market (before 9:15 AM) — no live data. "
                  f"Sent plan reminder. Assessment fires at 10:15 AM.")
            return
        # Post 9:15 AM with no spot = real failure (token issue)
        _send_telegram(bot_token, chat_id,
            f"⚠️ <b>Weekly IC Morning Review — FAILED</b>\n\n"
            f"Could not fetch live NIFTY spot. Dhan token may have expired.\n\n"
            f"<i>Stale plan (spot={old_spot:,.0f}, SC={old_sc}, SP={old_sp}) is still pending "
            f"but NOT recommended to deploy without fresh data.</i>"
        )
        return

    spot_move = round(spot - old_spot, 1) if old_spot else 0
    spot_move_str = f"+{spot_move:+,.0f}" if spot_move >= 0 else f"{spot_move:,.0f}"

    raw = _get_raw_chain(creds, expiry_str)
    parsed = _parse_chain(raw, spot)

    if not parsed["strikes"]:
        # Cancel the stale plan so the old link cannot fire
        pending["status"] = "CANCELLED"
        PENDING_FILE.write_text(json.dumps(pending, indent=2))
        _send_telegram(bot_token, chat_id,
            f"⏭ <b>Weekly IC Morning Review — SKIPPED</b>\n\n"
            f"Option chain for {expiry_str} is not available.\n"
            f"Stale plan cancelled — do NOT deploy.\n\n"
            f"Spot now: {spot:,.0f}  (was {old_spot:,.0f}, move={spot_move_str})"
        )
        return

    analysis = _analyse(parsed)
    if not analysis["ok"] or analysis["avg_iv"] > 28:
        pending["status"] = "CANCELLED"
        PENDING_FILE.write_text(json.dumps(pending, indent=2))
        reason = f"IV={analysis['avg_iv']:.1f}% too high (>28%)" if analysis["avg_iv"] > 28 else analysis.get("reason", "Analysis failed")
        _send_telegram(bot_token, chat_id,
            f"⏭ <b>Weekly IC Morning Review — CONDITIONS CHANGED, SKIP</b>\n\n"
            f"{reason}\n"
            f"Stale plan cancelled — holding cash this week.\n\n"
            f"Spot now: {spot:,.0f}  (was {old_spot:,.0f}, move={spot_move_str})"
        )
        return

    # Rebuild fresh plan
    new_plan = _select_legs(parsed, analysis, expiry_str, lots=DEFAULT_LOTS)
    if new_plan is None:
        pending["status"] = "CANCELLED"
        PENDING_FILE.write_text(json.dumps(pending, indent=2))
        _send_telegram(bot_token, chat_id,
            f"⏭ <b>Weekly IC Morning Review — NO VALID LEGS</b>\n\n"
            f"Spot moved to {spot:,.0f} (was {old_spot:,.0f}, {spot_move_str}).\n"
            f"Could not build valid IC legs — strikes too close or zero credit.\n"
            f"Stale plan cancelled."
        )
        return

    new_plan["avg_iv"]        = analysis["avg_iv"]
    new_plan["expected_move"] = analysis["expected_move"]
    new_plan["pcr"]           = analysis["pcr"]

    # Generate a fresh token (invalidates the old deploy link)
    new_token = str(uuid.uuid4()).replace("-", "")[:16]
    _save_pending(new_plan, new_token)

    # Strike change summary
    sc_chg = f"{new_plan['short_call']:,.0f} (was {old_sc:,.0f})" if old_sc != new_plan["short_call"] else f"{new_plan['short_call']:,.0f} ✓ unchanged"
    sp_chg = f"{new_plan['short_put']:,.0f} (was {old_sp:,.0f})" if old_sp != new_plan["short_put"] else f"{new_plan['short_put']:,.0f} ✓ unchanged"

    conf_emoji = {"HIGH": "🟢", "MEDIUM": "🟡", "LOW": "🔴"}.get(new_plan["confidence_label"], "⚪")
    notes = "\n".join(f"  {n}" for n in new_plan["confidence_notes"])
    deploy_url = f"{UI_BASE}/api/weekly_ic/deploy?token={new_token}"

    msg = (
        f"🌅 <b>Weekly IC Morning Review — {expiry_str}</b>\n"
        f"Spot: <b>{spot:,.0f}</b>  (overnight move: {spot_move_str} pts)\n\n"
        f"<b>Updated strikes (refreshed from live chain):</b>\n"
        f"  📉 SELL {new_plan['short_call']:,.0f} CE  →  {sc_chg}\n"
        f"  🛡 BUY  {new_plan['long_call']:,.0f} CE\n"
        f"  📈 SELL {new_plan['short_put']:,.0f} PE  →  {sp_chg}\n"
        f"  🛡 BUY  {new_plan['long_put']:,.0f} PE\n\n"
        f"Net credit: ~₹{new_plan['gross_credit']:,.0f}  |  "
        f"Max loss: ₹{new_plan['max_loss']:,.0f}\n"
        f"IV: {new_plan['avg_iv']:.1f}%  |  Expected move: ±{new_plan['expected_move']:.0f} pts\n\n"
        f"{conf_emoji} <b>Confidence: {new_plan['confidence_label']} ({new_plan['confidence']}/8)</b>\n"
        f"{notes}\n\n"
        f"{'─'*36}\n"
        f"⏳ <b>Market assessment fires at 10:15 AM.</b>\n"
        f"If structure looks good, a deploy link will be sent then.\n"
        f"<i>Opening range needs to settle before entry — avoid 9:15 AM volatility.</i>"
    )
    ok = _send_telegram(bot_token, chat_id, msg)
    print(f"[weekly_ic_review] Refreshed. SC={new_plan['short_call']}, SP={new_plan['short_put']}. "
          f"Spot move={spot_move_str}. Token={new_token}. Telegram={'OK' if ok else 'FAILED'}. "
          f"Deploy link withheld — assessment fires at 10:15 AM.")


# ── 10:15 AM market assessment ────────────────────────────────────────────────

# Thresholds for the 10:15 AM gate
_MAX_IV_AT_ENTRY       = 22.0   # % — above this, IC premium looks good but gamma risk is high
_MIN_SPOT_MARGIN_RATIO = 0.80   # spot must be ≥ 80% of expected_move away from both short strikes
_MAX_DAY_RANGE_RATIO   = 0.50   # day's high-low range must be < 50% of expected weekly move
_PCR_LOW               = 0.70   # below this = very bearish (call writers dominating)
_PCR_HIGH              = 1.80   # above this = very complacent (put writers dominating)


def _get_nifty_ohlc_today(creds: dict) -> dict[str, float | None]:
    """Fetch today's OHLC for NIFTY from Dhan intraday endpoint."""
    try:
        r = requests.post(
            "https://api.dhan.co/v2/charts/intraday",
            headers=_dhan_headers(creds),
            json={
                "securityId": "13",
                "exchangeSegment": "IDX_I",
                "instrument": "INDEX",
                "interval": "5",
                "oi": False,
            },
            timeout=10,
        )
        d = r.json()
        opens  = d.get("open")  or []
        highs  = d.get("high")  or []
        lows   = d.get("low")   or []
        closes = d.get("close") or []
        if not closes:
            return {"open": None, "high": None, "low": None, "close": None, "candles": 0}
        return {
            "open":    float(opens[0])   if opens   else None,
            "high":    float(max(highs)) if highs   else None,
            "low":     float(min(lows))  if lows    else None,
            "close":   float(closes[-1]) if closes  else None,
            "candles": len(closes),
        }
    except Exception as e:
        print(f"[ohlc] {e}", file=sys.stderr)
        return {"open": None, "high": None, "low": None, "close": None, "candles": 0}


def _is_paper_mode() -> bool:
    """Return True if the agent is running in PAPER_LIVE mode."""
    try:
        rc_path = STATE_DIR / "intraday_v83_runtime_config.json"
        rc = json.loads(rc_path.read_text())
        return str(rc.get("mode") or "").startswith("PAPER")
    except Exception:
        return False


def _auto_deploy_paper(token: str) -> tuple[bool, str]:
    """Call the local UI server to deploy the IC in paper mode.

    Returns (success, message).
    """
    try:
        r = requests.get(f"{UI_BASE}/api/weekly_ic/deploy?token={token}", timeout=15)
        if r.status_code == 200:
            return True, "Auto-deployed via UI server"
        return False, f"UI server returned HTTP {r.status_code}"
    except Exception as exc:
        return False, f"Auto-deploy call failed: {exc}"


def run_market_assessment(force: bool = False) -> None:
    """Wednesday 10:15 AM IST — multi-factor gate before issuing IC deploy link.

    Checks after the opening range has established (~1 hour of price action):

      1. Opening range contained   — day high-low < 50% of weekly expected move
      2. IV settled                — ATM IV < 22% (not a panic/volatile day)
      3. Spot safety margin        — spot ≥ 80% of expected_move from both short strikes
      4. PCR balanced              — between 0.70 and 1.80 (no extreme skew)
      5. Enough candles            — at least 8 × 5-min candles (= 40 min of data)

    In PAPER_LIVE mode:  auto-deploys immediately when ≥4/5 gates pass.
    In LIVE mode:        sends Telegram deploy link for manual confirmation.
    If 2+ gates fail:   sends a HOLD alert; 11:30 AM re-check fires automatically.
    """
    now = datetime.now(IST)
    weekday = now.weekday()  # 2 = Wednesday

    if not force:
        if weekday != 2:
            print(f"[weekly_ic_assess] Not Wednesday — skipping (today={now.strftime('%A')}).")
            return
        hhmm = now.hour * 100 + now.minute
        if hhmm < 1015:
            print("[weekly_ic_assess] Before 10:15 AM IST — skipping.")
            return

    pending = _load_pending()
    # Accept PENDING, None, and ASSESSMENT_HOLD (re-check after earlier failure)
    if not pending or pending.get("status") not in ("PENDING", "ASSESSMENT_HOLD", None):
        print(f"[weekly_ic_assess] Status={pending.get('status') if pending else 'none'} — nothing to assess.")
        return

    creds = _load_creds()
    bot_token = str(creds.get("telegram_bot_token") or "").strip()
    chat_id   = str(creds.get("telegram_chat_id") or "").strip()
    if not bot_token or not chat_id:
        print("[weekly_ic_assess] Telegram not configured.")
        return

    expiry_str    = str(pending.get("expiry") or "")
    planned_sc    = float(pending.get("short_call") or 0)
    planned_sp    = float(pending.get("short_put")  or 0)
    expected_move = float(pending.get("expected_move") or 400)
    plan_credit   = float(pending.get("gross_credit") or 0)

    print(f"[weekly_ic_assess] Assessing market for {expiry_str} IC entry ...")

    # ── Fetch live data ───────────────────────────────────────────────────────
    spot = _get_nifty_ltp(creds)
    if not spot:
        _send_telegram(bot_token, chat_id,
            f"⚠️ <b>Weekly IC Assessment — FAILED (10:15 AM)</b>\n\n"
            f"Cannot fetch NIFTY spot. Dhan token may have expired.\n"
            f"Please check manually and deploy if conditions look safe."
        )
        return

    ohlc    = _get_nifty_ohlc_today(creds)
    raw     = _get_raw_chain(creds, expiry_str)
    parsed  = _parse_chain(raw, spot)
    analysis = _analyse(parsed) if parsed["strikes"] else {}

    # Current IV from live chain (fallback to plan's IV)
    current_iv  = float(analysis.get("avg_iv") or pending.get("avg_iv") or 15.0)
    current_pcr = float(analysis.get("pcr")    or pending.get("pcr")    or 1.0)

    # Day range
    day_high   = ohlc.get("high")
    day_low    = ohlc.get("low")
    day_open   = ohlc.get("open")
    num_candles = int(ohlc.get("candles") or 0)
    day_range  = (day_high - day_low) if (day_high and day_low) else None

    # Spot distance to short strikes (positive = safe side)
    sc_margin = planned_sc - spot   # call side: spot below SC
    sp_margin = spot - planned_sp   # put side:  spot above SP
    min_required_margin = expected_move * _MIN_SPOT_MARGIN_RATIO

    # ── Gate checks ───────────────────────────────────────────────────────────
    gates: list[tuple[bool, str, str]] = []   # (passed, label, detail)

    # 1. Enough opening data
    gates.append((
        num_candles >= 8,
        "Opening data",
        f"{num_candles} × 5-min candles  {'✅' if num_candles >= 8 else '⚠️ (< 40 min data)'}"
    ))

    # 2. Opening range contained
    if day_range is not None:
        range_ratio = day_range / expected_move
        gates.append((
            range_ratio < _MAX_DAY_RANGE_RATIO,
            "Day range",
            f"Range={day_range:.0f} pts = {range_ratio*100:.0f}% of expected move "
            f"{'✅' if range_ratio < _MAX_DAY_RANGE_RATIO else '⚠️ trending day — IC risky'}"
        ))
    else:
        gates.append((False, "Day range", "⚠️ Could not fetch intraday OHLC"))

    # 3. IV settled
    gates.append((
        current_iv <= _MAX_IV_AT_ENTRY,
        "IV",
        f"ATM IV={current_iv:.1f}%  "
        f"{'✅ settled' if current_iv <= _MAX_IV_AT_ENTRY else f'⚠️ elevated (>{_MAX_IV_AT_ENTRY}%) — wider spreads'}"
    ))

    # 4. Spot margin from short strikes
    call_safe = sc_margin >= min_required_margin
    put_safe  = sp_margin >= min_required_margin
    gates.append((
        call_safe and put_safe,
        "Strike margin",
        f"Call side: {sc_margin:.0f} pts {'✅' if call_safe else '🚨 too close'}  |  "
        f"Put side: {sp_margin:.0f} pts {'✅' if put_safe else '🚨 too close'}  "
        f"(min={min_required_margin:.0f})"
    ))

    # 5. PCR balanced
    gates.append((
        _PCR_LOW <= current_pcr <= _PCR_HIGH,
        "PCR",
        f"PCR={current_pcr:.2f}  "
        f"{'✅ balanced' if _PCR_LOW <= current_pcr <= _PCR_HIGH else ('⚠️ extremely bearish' if current_pcr < _PCR_LOW else '⚠️ extremely complacent')}"
    ))

    passed   = [g for g in gates if g[0]]
    failed   = [g for g in gates if not g[0]]
    n_passed = len(passed)
    n_gates  = len(gates)

    # ── Decision ─────────────────────────────────────────────────────────────
    gate_lines = "\n".join(f"  {'✅' if g[0] else '❌'} {g[1]}: {g[2]}" for g in gates)
    paper_mode = _is_paper_mode()
    assessment_label = now.strftime("%-I:%M %p")   # e.g. "10:15 AM"

    if n_passed >= n_gates - 1:
        # All clear (5/5) or borderline (4/5) — deploy or issue link
        new_token = str(uuid.uuid4()).replace("-", "")[:16]
        pending["token"] = new_token
        pending["assessment_at"] = now.isoformat()
        PENDING_FILE.write_text(json.dumps(pending, indent=2))
        deploy_url = f"{UI_BASE}/api/weekly_ic/deploy?token={new_token}"

        clarity = "ALL CLEAR" if n_passed == n_gates else "BORDERLINE"
        clarity_emoji = "🟢" if n_passed == n_gates else "🟡"
        caution_note = (
            f"\n<b>Note:</b> {failed[0][2]}\n" if failed else ""
        )

        if paper_mode:
            # ── PAPER MODE: auto-deploy immediately ──────────────────────────
            auto_ok, auto_msg = _auto_deploy_paper(new_token)
            if auto_ok:
                msg = (
                    f"{clarity_emoji} <b>Weekly IC — {clarity} — Auto-deployed (PAPER)</b>\n"
                    f"Expiry: {expiry_str}  |  Spot: {spot:,.0f}  |  {assessment_label} assessment\n\n"
                    f"<b>Gates:</b>\n{gate_lines}\n"
                    f"{caution_note}\n"
                    f"<b>Position deployed:</b>\n"
                    f"  📉 SELL {planned_sc:,.0f} CE  |  📈 SELL {planned_sp:,.0f} PE\n"
                    f"  Credit: ~₹{plan_credit:,.0f}  |  IV: {current_iv:.1f}%  |  PCR: {current_pcr:.2f}\n\n"
                    f"✅ <b>All 4 legs submitted automatically in PAPER mode.</b>\n"
                    f"<i>Exit next Tuesday before 3:20 PM IST</i>"
                )
                result_str = f"AUTO_DEPLOYED_{clarity}"
            else:
                # Auto-deploy failed — fall back to manual link
                msg = (
                    f"{clarity_emoji} <b>Weekly IC — {clarity} — Auto-deploy FAILED (PAPER)</b>\n"
                    f"Expiry: {expiry_str}  |  Spot: {spot:,.0f}\n\n"
                    f"<b>Gates:</b>\n{gate_lines}\n"
                    f"{caution_note}\n"
                    f"⚠️ Auto-deploy error: {auto_msg}\n\n"
                    f"<b>👇 Deploy manually:</b>\n"
                    f"<a href='{deploy_url}'>{'✅ DEPLOY IC NOW' if n_passed == n_gates else '⚠️ DEPLOY IC (CAUTION)'}</a>"
                )
                result_str = f"AUTO_DEPLOY_FAILED_{clarity}"
                print(f"[weekly_ic_assess] Auto-deploy failed: {auto_msg}")
        else:
            # ── LIVE MODE: send deploy link for manual confirmation ───────────
            msg = (
                f"{clarity_emoji} <b>Weekly IC — {clarity} — Deploy when ready</b>\n"
                f"Expiry: {expiry_str}  |  Spot: {spot:,.0f}  |  {assessment_label} assessment\n\n"
                f"<b>Gates:</b>\n{gate_lines}\n"
                f"{caution_note}\n"
                f"<b>Position:</b>\n"
                f"  📉 SELL {planned_sc:,.0f} CE  |  📈 SELL {planned_sp:,.0f} PE\n"
                f"  Credit: ~₹{plan_credit:,.0f}  |  IV: {current_iv:.1f}%  |  PCR: {current_pcr:.2f}\n\n"
                f"<b>👇 Tap to deploy:</b>\n"
                f"<a href='{deploy_url}'>{'✅ DEPLOY IC NOW' if n_passed == n_gates else '⚠️ DEPLOY IC (CAUTION)'}</a>\n\n"
                f"<i>Exit next Tuesday before 3:20 PM IST</i>"
            )
            result_str = clarity

    else:
        # 2+ failures
        failed_labels = ", ".join(g[1] for g in failed)
        hhmm = now.hour * 100 + now.minute

        if hhmm >= 1130 or pending.get("status") == "ASSESSMENT_HOLD":
            # This is the 11:30 AM re-check (or a forced re-run after a prior HOLD)
            # Conditions still bad — cancel for the week
            pending["status"] = "CANCELLED"
            PENDING_FILE.write_text(json.dumps(pending, indent=2))
            msg = (
                f"⏭ <b>Weekly IC — CANCELLED for this week ({n_passed}/{n_gates} gates)</b>\n"
                f"Expiry: {expiry_str}  |  Spot: {spot:,.0f}  |  {assessment_label} re-check\n\n"
                f"<b>Gates:</b>\n{gate_lines}\n\n"
                f"<b>Failed:</b> {failed_labels}\n\n"
                f"Conditions did not improve since the earlier assessment.\n"
                f"<b>Plan cancelled — holding cash this week.</b>\n"
                f"Next opportunity: Tuesday after 3:35 PM IST."
            )
            result_str = "CANCELLED"
        else:
            # First assessment at 10:15 AM — hold and wait for 11:30 AM re-check
            pending["status"] = "ASSESSMENT_HOLD"
            PENDING_FILE.write_text(json.dumps(pending, indent=2))
            msg = (
                f"🔴 <b>Weekly IC — NOT SUITABLE YET ({n_passed}/{n_gates} gates)</b>\n"
                f"Expiry: {expiry_str}  |  Spot: {spot:,.0f}  |  {assessment_label} assessment\n\n"
                f"<b>Gates:</b>\n{gate_lines}\n\n"
                f"<b>Failed:</b> {failed_labels}\n"
                f"<b>Re-check fires automatically at 11:30 AM IST.</b>\n"
                f"If still failing at 11:30, the plan will be cancelled for this week.\n\n"
                f"<i>Manual re-check: python scripts/weekly_ic_executor.py --assess --now</i>"
            )
            result_str = "HOLD"

    ok = _send_telegram(bot_token, chat_id, msg)
    print(f"[weekly_ic_assess] Result={result_str} ({n_passed}/{n_gates}). "
          f"Paper={paper_mode} IV={current_iv:.1f}% PCR={current_pcr:.2f} "
          f"SC_margin={sc_margin:.0f} SP_margin={sp_margin:.0f}. "
          f"Telegram={'OK' if ok else 'FAILED'}.")


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
    print(f"[weekly_ic] Plan saved. Confidence={plan['confidence_label']}. Token={token}. "
          f"Telegram={'OK' if ok else 'FAILED'}. Morning review fires at 09:00 AM tomorrow.")


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
    parser.add_argument("--now",            action="store_true", help="Force run (ignore day/time check)")
    parser.add_argument("--close",          action="store_true", help="Send close alert for open position")
    parser.add_argument("--morning-review", action="store_true", help="Wednesday 9:00 AM — refresh pending plan")
    parser.add_argument("--assess",         action="store_true", help="Wednesday 10:15 AM — market assessment gate")
    args = parser.parse_args()

    if args.close:
        run_close_alert()
    elif args.morning_review:
        run_morning_review(force=args.now)
    elif args.assess:
        run_market_assessment(force=args.now)
    else:
        run_entry(force=args.now)
