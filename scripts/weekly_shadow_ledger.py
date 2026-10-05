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


def _pick_strike(rows, side, want, tol=25.0):
    """Nearest usable leg to an exact strike — for EQUAL-WIDTH wings."""
    best = None; bd = 1e9
    for sk, sides in rows.items():
        leg = sides.get(side) or {}
        if leg.get("bid", 0) <= 0 or leg.get("ask", 0) <= 0:
            continue
        e = abs(sk - want)
        if e < bd:
            bd = e; best = (sk, leg)
    return best if bd <= tol else None


def _settle(sp, lp, sc, lc, spot_t):
    """Build one condor variant and settle it at intrinsic. Shorts are shared across variants;
    only the WING rule differs, so this isolates wing design as the single variable."""
    if not (sp and lp and sc and lc and lp[0] < sp[0] < sc[0] < lc[0]):
        return None
    credit = (sp[1]["bid"] - lp[1]["ask"]) + (sc[1]["bid"] - lc[1]["ask"]) - 4 * FEE
    if credit <= 0:
        return None
    clamp = lambda x, lo, hi: max(lo, min(hi, x))  # noqa: E731
    pw, cw = sp[0] - lp[0], lc[0] - sc[0]
    put_liab = clamp(sp[0] - spot_t, 0, pw)
    call_liab = clamp(spot_t - sc[0], 0, cw)
    return {
        "condor": {"lp": lp[0], "sp": sp[0], "sc": sc[0], "lc": lc[0]},
        "put_width": pw, "call_width": cw,
        "credit_pts": round(credit, 2),
        "put_liab": round(put_liab, 1), "call_liab": round(call_liab, 1),
        "pnl_rupees": round((credit - put_liab - call_liab) * LOT),
        # max loss per side — the risk the delta rule silently unbalances
        "put_maxloss_rupees": round((credit - pw) * LOT),
        "call_maxloss_rupees": round((credit - cw) * LOT),
    }


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
    sp = _pick(er, "pe", SHORT_D)
    sc = _pick(er, "ce", SHORT_D)
    if not (sp and sc):
        print(f"[weekly] no ~{SHORT_D} short strikes on {entry_day}; skip"); return 0

    # Three WING designs on the SAME shorts — logged side by side so live evidence, not a
    # backtest row, picks the winner. Found 2026-07-17: delta-selected wings are delta-neutral
    # but NOT risk-neutral. Puts carry richer IV, so delta moves more slowly per strike and
    # "0.22 -> 0.12" spans more points on the put side: the live 07-15 condor came out 200pt put
    # / 100pt call, i.e. -11,768 put max loss vs -4,268 call — ~3x more risk on one side of a
    # structure that looks balanced. On the dense 14 weeks at real spreads, equal-150 beat delta
    # on EVERY metric (+31,643 vs +29,881, worst -6,071 vs -6,377, max loss -5,682 vs -6,744,
    # asym 1.00x vs 1.60x) and equal-200 earned far more (+48,656, 13/13 weeks better) but at a
    # -8,559/side max loss that never materialised in 13 weeks — a tail too fat to price on that
    # sample. So: log all three, decide forward.
    variants = {
        "delta": _settle(sp, _pick(er, "pe", LONG_D), sc, _pick(er, "ce", LONG_D), spot_t),
        "equal_150": _settle(sp, _pick_strike(er, "pe", sp[0] - 150.0),
                             sc, _pick_strike(er, "ce", sc[0] + 150.0), spot_t),
        "equal_200": _settle(sp, _pick_strike(er, "pe", sp[0] - 200.0),
                             sc, _pick_strike(er, "ce", sc[0] + 200.0), spot_t),
    }
    if not variants["delta"]:
        print(f"[weekly] could not build the delta condor from {entry_day}; skip"); return 0
    dte = (datetime.strptime(today, "%Y-%m-%d") - datetime.strptime(entry_day, "%Y-%m-%d")).days

    d = variants["delta"]
    rec = {"expiry": today, "entry_day": entry_day, "dte_at_entry": dte,
           "entry_spot": round(es, 1), "expiry_spot": round(spot_t, 1),
           # top-level mirrors the DELTA variant so the existing record shape still reads
           "condor": d["condor"], "credit_pts": d["credit_pts"],
           "put_liab": d["put_liab"], "call_liab": d["call_liab"],
           "pnl_rupees": d["pnl_rupees"],
           "variants": variants}
    (STATE / "weekly_shadow_ledger.jsonl").open("a").write(json.dumps(rec) + "\n")

    c = d["condor"]
    print(f"[weekly] {entry_day}->{today} ({dte}DTE) settle spot {spot_t:.0f}")
    print(f"[weekly]   delta     {c['lp']:.0f}/{c['sp']:.0f}-{c['sc']:.0f}/{c['lc']:.0f} "
          f"credit {d['credit_pts']:.1f} -> Rs {d['pnl_rupees']:+,}")
    for k in ("equal_150", "equal_200"):
        v = variants[k]
        if not v:
            print(f"[weekly]   {k:<9} not buildable"); continue
        vc = v["condor"]
        print(f"[weekly]   {k:<9} {vc['lp']:.0f}/{vc['sp']:.0f}-{vc['sc']:.0f}/{vc['lc']:.0f} "
              f"credit {v['credit_pts']:.1f} -> Rs {v['pnl_rupees']:+,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
