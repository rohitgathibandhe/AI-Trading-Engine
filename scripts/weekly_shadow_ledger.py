#!/usr/bin/env python
"""WEEKLY shadow ledger — forward evidence for the weekly-seller breadth edge.

On each weekly EXPIRY day (front weekly expires today), reconstruct the delta-neutral weekly
condor that WOULD have been entered ~5 DTE (the first captured day this expiry was the front week),
using the entry-day captured chain at REAL FILLS (sell@bid / buy@ask), and settle it at today's
intrinsic. Logs one JSON line per week to state/weekly_shadow_ledger.jsonl.

This is the forward-accumulating validator for the in-sample result (dense 7mo, real fills:
+1,149/wk, 72% win, robust across strikes — see memory/project_weekly_seller.md). NOT a trade;
it never touches the broker or config. Runs daily 15:50 IST and self-skips on non-expiry days.

Config matches the validated sim: short ~0.22 delta, long ~0.12 wings, set-and-forget (rolling
was shown to HURT), intrinsic settlement.
"""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

STATE = Path(__file__).resolve().parents[1] / "data_engine" / "market_ai" / "state"
ROLL = STATE / "rolling_option_live"
IST = timezone(timedelta(hours=5, minutes=30))
LOT, FEE = 75, 0.05
SHORT_D, LONG_D = 0.22, 0.12


def _entry_rows(day):
    """Parse the ~10:00 snapshot for a captured day -> {strike: {ce/pe: {delta,bid,ask}}}, + spot, + expiry."""
    f = ROLL / day / "chain_snapshots.jsonl"
    if not f.exists():
        return None, None, None
    best = None
    for line in f.read_text().splitlines():
        try:
            r = json.loads(line)
        except Exception:
            continue
        t = r.get("captured_at", "")[11:16]
        if t >= "10:00":
            best = r; break
        best = r
    if not best:
        return None, None, None
    oc = (((best.get("response") or {}).get("data") or {}).get("data") or {})
    spot = float(oc.get("last_price") or 0)
    rows = {}
    for k, node in (oc.get("oc") or {}).items():
        strike = float(k); rows[strike] = {}
        for side in ("ce", "pe"):
            leg = node.get(side) or {}; g = leg.get("greeks") or {}
            rows[strike][side] = {"delta": g.get("delta"),
                                  "bid": float(leg.get("top_bid_price") or 0),
                                  "ask": float(leg.get("top_ask_price") or 0)}
    return rows, spot, best.get("expiry")


def _pick(rows, side, dt, tol=0.08):
    best = None; bd = 9
    for sk, sides in rows.items():
        leg = sides.get(side) or {}; d = leg.get("delta")
        if d is None or leg.get("bid", 0) <= 0 or leg.get("ask", 0) <= 0:
            continue
        e = abs(abs(d) - dt)
        if e < bd:
            bd = e; best = (sk, leg)
    return best if bd <= tol else None


def _capture_days():
    if not ROLL.exists():
        return []
    return sorted(d.name for d in ROLL.iterdir() if d.is_dir() and d.name[:4].isdigit())


def main() -> int:
    now = datetime.now(IST); today = now.strftime("%Y-%m-%d")
    rows_t, spot_t, exp_t = _entry_rows(today)
    if not rows_t or not spot_t or not exp_t:
        print(f"[weekly] no capture for {today}; skip"); return 0
    if str(exp_t) != today:          # only settle ON the expiry day
        print(f"[weekly] {today} not an expiry day (front weekly {exp_t}); skip"); return 0

    # entry day = earliest captured day whose front weekly was today's expiry
    entry_day = None
    for d in _capture_days():
        if d >= today:
            break
        _, _, e = _entry_rows(d)
        if str(e) == today:
            entry_day = d; break
    if not entry_day:
        print(f"[weekly] no entry-day capture found for expiry {today}; skip"); return 0

    er, es, _ = _entry_rows(entry_day)
    sp = _pick(er, "pe", SHORT_D); lp = _pick(er, "pe", LONG_D)
    sc = _pick(er, "ce", SHORT_D); lc = _pick(er, "ce", LONG_D)
    if not (sp and lp and sc and lc and lp[0] < sp[0] < sc[0] < lc[0]):
        print(f"[weekly] could not build condor from {entry_day}; skip"); return 0
    # entry credit at real fills: sell shorts @ bid, buy longs @ ask
    credit = (sp[1]["bid"] - lp[1]["ask"]) + (sc[1]["bid"] - lc[1]["ask"]) - 4 * FEE
    # intrinsic settlement at today's expiry spot
    def clamp(x, lo, hi): return max(lo, min(hi, x))
    put_liab = clamp(sp[0] - spot_t, 0, sp[0] - lp[0])
    call_liab = clamp(spot_t - sc[0], 0, lc[0] - sc[0])
    pnl = round((credit - put_liab - call_liab) * LOT)
    dte = (datetime.strptime(today, "%Y-%m-%d") - datetime.strptime(entry_day, "%Y-%m-%d")).days

    rec = {"expiry": today, "entry_day": entry_day, "dte_at_entry": dte,
           "condor": {"sp": sp[0], "lp": lp[0], "sc": sc[0], "lc": lc[0]},
           "entry_spot": round(es, 1), "expiry_spot": round(spot_t, 1),
           "credit_pts": round(credit, 2), "put_liab": round(put_liab, 1),
           "call_liab": round(call_liab, 1), "pnl_rupees": pnl}
    (STATE / "weekly_shadow_ledger.jsonl").open("a").write(json.dumps(rec) + "\n")
    print(f"[weekly] {entry_day}->{today} ({dte}DTE) condor {lp[0]:.0f}/{sp[0]:.0f}-{sc[0]:.0f}/{lc[0]:.0f} "
          f"credit {credit:.1f} settle spot {spot_t:.0f} -> P&L Rs {pnl:+,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
