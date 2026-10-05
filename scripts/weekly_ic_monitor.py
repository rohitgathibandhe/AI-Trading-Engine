#!/usr/bin/env python
"""WEEKLY IC MONITOR — the missing manager for a 3-5 day hold.

The weekly iron condor is NOT an intraday trade. It is opened Wednesday ~10:15 and carried to the
following Tuesday expiry, and the plan already states exactly what it is trying to do:

    target_keep       = 70% of the credit received
    stop_trigger_loss = 2x the credit received

...but until now those two numbers were computed once at plan time, printed into a Telegram
message, and then never looked at again. Nothing marked the open position between deployment and
the Tuesday close alert. So a condor that reached its 70% target on Friday just kept sitting there,
giving the gains back into expiry-week gamma and carrying a weekend gap for premium it had already
earned. The trade had a target; the agent did not know it.

This is that manager. Every run during market hours it:
  1. marks all four legs at REAL exit prices (BUY back the shorts at ASK, SELL the longs at BID —
     the price you would actually pay to get flat, not a flattering mid),
  2. computes open P&L and what fraction of the credit has been captured,
  3. judges it against the plan's OWN target/stop, plus the two risks that only exist on a
     multi-day hold: spot approaching a short strike, and expiry-day gamma,
  4. appends the mark to state/weekly_ic_monitor.jsonl — one row per check, so the hold is a
     forward record we can retrospect instead of a black box between Wed and Tue,
  5. alerts on a CHANGE of state (never re-alerts the same state), and in paper mode closes the
     position on TARGET_HIT / STOP_HIT — mirroring the existing paper auto-deploy, so the trade is
     managed end-to-end rather than only opened automatically.

Holding doctrine this encodes (theta sign decides the hold): a CREDIT structure is a swing/expiry
trade, so it is not flattened intraday — it is carried until the target is hit, the stop is hit, or
expiry forces the issue. That is the opposite of the debit intraday path, and deliberately so.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import weekly_ic_executor as W  # noqa: E402  — reuse creds/chain/telegram helpers

STATE_DIR = W.STATE_DIR
PENDING = STATE_DIR / "weekly_ic_pending.json"
LEDGER = STATE_DIR / "weekly_ic_monitor.jsonl"
MON_STATE = STATE_DIR / "weekly_ic_monitor_state.json"
IST = W.IST

# Multi-day risks that simply do not exist on an intraday trade.
BREACH_BUFFER_PTS = 100.0   # spot this close to a short strike = the condor is under pressure
EXPIRY_DAY_FLATTEN = (15, 20)   # expiry day: be flat before the close


def _load_position() -> dict | None:
    try:
        p = json.loads(PENDING.read_text())
    except (OSError, ValueError):
        return None
    return p if p.get("status") == "DEPLOYED" else None


def _mark(pos: dict, creds: dict) -> dict | None:
    """Mark the 4 legs at the price it would actually cost to get flat."""
    raw = W._get_raw_chain(creds, pos["expiry"])
    if not raw:
        return None
    parsed = W._parse_chain(raw, float(pos.get("spot_at_entry") or 0))
    ks = parsed["strikes"]
    spot = parsed["spot"]
    try:
        sc = ks[float(pos["short_call"])]
        lc = ks[float(pos["long_call"])]
        sp = ks[float(pos["short_put"])]
        lp = ks[float(pos["long_put"])]
    except KeyError:
        return None

    # Cost to close: pay ASK to buy back the shorts, receive BID for the longs.
    def _px(node, side, field, fallback):
        v = float(node.get(f"{side}_{field}") or 0)
        return v if v > 0 else float(node.get(f"{side}_{fallback}") or 0)

    cost = ((_px(sc, "ce", "ask", "ltp") + _px(sp, "pe", "ask", "ltp"))
            - (_px(lc, "ce", "bid", "ltp") + _px(lp, "pe", "bid", "ltp")))
    credit = float(pos["net_credit_per_unit"])
    qty = float(pos["quantity"])
    pnl = (credit - cost) * qty
    return {
        "spot": round(spot, 2),
        "cost_to_close_per_unit": round(cost, 2),
        "open_pnl_rupees": round(pnl, 2),
        "capture_pct": round(100.0 * pnl / float(pos["gross_credit"]), 1) if pos.get("gross_credit") else None,
        "call_margin_pts": round(float(pos["short_call"]) - spot, 1),
        "put_margin_pts": round(spot - float(pos["short_put"]), 1),
    }


def _judge(pos: dict, m: dict, now: datetime) -> tuple[str, str]:
    """Return (state, human reason). Order matters: risk before reward."""
    pnl = m["open_pnl_rupees"]
    exp = str(pos.get("expiry") or "")
    is_expiry_day = exp == now.date().isoformat()
    hhmm = (now.hour, now.minute)

    if pnl <= -abs(float(pos["stop_trigger_loss"])):
        return "STOP_HIT", f"loss ₹{-pnl:,.0f} hit the ₹{float(pos['stop_trigger_loss']):,.0f} stop (2x credit)"
    if is_expiry_day and hhmm >= EXPIRY_DAY_FLATTEN:
        return "EXPIRY_FLATTEN", "expiry day past 15:20 — close rather than carry pin risk"
    if pnl >= float(pos["target_keep"]):
        return "TARGET_HIT", (f"captured {m['capture_pct']:.0f}% of credit "
                              f"(₹{pnl:,.0f} >= ₹{float(pos['target_keep']):,.0f} target)")
    if min(m["call_margin_pts"], m["put_margin_pts"]) <= BREACH_BUFFER_PTS:
        side = "CALL" if m["call_margin_pts"] < m["put_margin_pts"] else "PUT"
        return "BREACH_WARNING", (f"spot {m['spot']:,.0f} is {min(m['call_margin_pts'], m['put_margin_pts']):.0f}pts "
                                  f"from the short {side} — condor under pressure")
    return "HOLD", f"open ₹{pnl:+,.0f} ({m['capture_pct']:+.0f}% of credit)"


def _auto_close_paper(token: str) -> tuple[bool, str]:
    """Mirror of _auto_deploy_paper — the trade should be managed end-to-end, not only opened."""
    try:
        import requests
        r = requests.get(f"{W.UI_BASE}/api/weekly_ic/close?token={token}", timeout=15)
        if r.status_code == 200:
            return True, "Auto-closed via UI server"
        return False, f"UI server returned HTTP {r.status_code}"
    except Exception as exc:
        return False, f"Auto-close call failed: {exc}"


def main() -> int:
    now = datetime.now(IST)
    force = "--now" in sys.argv

    pos = _load_position()
    if not pos:
        print("[weekly_ic_monitor] No DEPLOYED weekly position — nothing to manage.")
        return 0

    hhmm = now.hour * 100 + now.minute
    if not force and (now.weekday() > 4 or hhmm < 915 or hhmm > 1530):
        print(f"[weekly_ic_monitor] Outside market hours ({now:%a %H:%M}) — skipping.")
        return 0

    creds = W._load_creds()
    m = _mark(pos, creds)
    if not m:
        print("[weekly_ic_monitor] Could not mark position (no chain) — no action.", file=sys.stderr)
        return 0

    state, reason = _judge(pos, m, now)
    entry_day = str(pos.get("deployed_at") or pos.get("created_at") or "")[:10]
    try:
        days_held = (now.date() - date.fromisoformat(entry_day)).days
    except ValueError:
        days_held = None

    row = {"at": now.isoformat(timespec="seconds"), "expiry": pos.get("expiry"),
           "days_held": days_held, "state": state, "reason": reason,
           "gross_credit": pos.get("gross_credit"), "target_keep": pos.get("target_keep"),
           "stop_trigger_loss": pos.get("stop_trigger_loss"), **m}
    with LEDGER.open("a") as f:
        f.write(json.dumps(row) + "\n")

    print(f"[weekly_ic_monitor] {state}: {reason}")
    print(f"  spot {m['spot']:,.0f} | cost-to-close {m['cost_to_close_per_unit']:.2f} "
          f"| open ₹{m['open_pnl_rupees']:+,.0f} | day {days_held}")

    # alert/act only on a CHANGE of state — never spam the same condition every 15 minutes
    try:
        prev = json.loads(MON_STATE.read_text()).get("state")
    except (OSError, ValueError):
        prev = None
    if state == prev:
        return 0
    MON_STATE.write_text(json.dumps({"state": state, "at": row["at"]}))

    actionable = state in ("TARGET_HIT", "STOP_HIT", "EXPIRY_FLATTEN")
    icon = {"TARGET_HIT": "🎯", "STOP_HIT": "🛑", "EXPIRY_FLATTEN": "⏰",
            "BREACH_WARNING": "⚠️", "HOLD": "📊"}.get(state, "📊")
    lines = [f"{icon} <b>WEEKLY IC — {state.replace('_', ' ')}</b>", "",
             f"{reason}", "",
             f"Open P&L:  <b>₹{m['open_pnl_rupees']:+,.0f}</b>  ({m['capture_pct']:+.0f}% of credit)",
             f"Spot:      {m['spot']:,.0f}   (call {m['call_margin_pts']:+.0f}pts / put {m['put_margin_pts']:+.0f}pts)",
             f"Held:      day {days_held} of the {pos.get('expiry')} expiry"]

    if actionable:
        ok, msg = _auto_close_paper(str(pos.get("token") or ""))
        lines += ["", ("✅ Position auto-closed (paper)." if ok else f"⚠️ Auto-close failed: {msg}")]
        if ok:
            pos["status"] = "CLOSED"
            pos["closed_at"] = row["at"]
            pos["close_state"] = state
            pos["close_pnl_rupees"] = m["open_pnl_rupees"]
            PENDING.write_text(json.dumps(pos, indent=2))
        print(f"  auto-close: {ok} — {msg}")

    bot = str(creds.get("telegram_bot_token") or "").strip()
    chat = str(creds.get("telegram_chat_id") or "").strip()
    if bot and chat:
        W._send_telegram(bot, chat, "\n".join(lines))
    else:
        W._alert_fallback_log("\n".join(lines), False)   # no creds -> still keep the record
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
