#!/usr/bin/env python
"""Daily SHADOW BOOK — the honest evidence engine for the seller/all-weather build.

Every day, from the live-captured option chain (rolling_option_live/<date>/chain_snapshots.jsonl),
compute what the REGIME-MATCHED credit trades WOULD have made, bypassing the agent's construction
filters entirely. This accumulates the real, forward, out-of-sample record we need to decide
whether bull-put / bear-call / iron-condor are genuine edges — BEFORE any of them drives live money.

For each day it logs, for bull-put (up/sideways bet), bear-call (down/sideways bet), and
iron-condor (range bet): entry credit, final mark, P&L/lot, and the day's actual open->close
move (so we can later bucket by realized regime). Appends one JSON line to
state/shadow_book.jsonl. Scheduled at 15:40 IST after the capture stops.

NOT a trade and NOT the agent's P&L — it's a what-if ledger for validation. Mid-priced
(real fills cost the spread); treat magnitudes as indicative, direction/consistency as signal.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

STATE = Path(__file__).resolve().parents[1] / "data_engine" / "market_ai" / "state"
IST = timezone(timedelta(hours=5, minutes=30))
LOT = 75
ENTRY_AFTER = "09:30"   # let the opening range form
MARK_BY = "15:20"


def _parse_snapshot(rec):
    oc = (((rec.get("response") or {}).get("data") or {}).get("data") or {})
    spot = float(oc.get("last_price") or 0)
    rows = {}
    for k, node in (oc.get("oc") or {}).items():
        strike = float(k)
        rows[strike] = {}
        for side in ("ce", "pe"):
            leg = node.get(side) or {}
            g = leg.get("greeks") or {}
            b = float(leg.get("top_bid_price") or 0)
            a = float(leg.get("top_ask_price") or 0)
            ltp = float(leg.get("last_price") or 0)
            mid = (b + a) / 2 if b > 0 and a > 0 else ltp
            rows[strike][side] = {"delta": g.get("delta"), "mid": mid, "bid": b, "ask": a}
    t = rec.get("captured_at", "")[11:19]
    return t, spot, rows


def _pick(rows, side, dtarget):
    best = None; bd = 9
    for strike, sides in rows.items():
        leg = sides.get(side) or {}
        d = leg.get("delta")
        if d is None or leg.get("mid", 0) <= 0:
            continue
        e = abs(abs(d) - dtarget)
        if e < bd:
            bd = e; best = (strike, leg)
    return best if bd <= 0.08 else None


def _credit_spread(entry_rows, exit_rows, side):
    """Short ~0.30 delta, long ~0.15 delta. Returns (entry_credit, final_value, pnl_rupees, legs)."""
    sp = _pick(entry_rows, side, 0.30)
    lp = _pick(entry_rows, side, 0.15)
    if not sp or not lp:
        return None
    sk, sv = sp; lk, lv = lp
    credit = sv["mid"] - lv["mid"]
    if credit <= 0:
        return None
    ex_s = (exit_rows.get(sk) or {}).get(side) or {}
    ex_l = (exit_rows.get(lk) or {}).get(side) or {}
    if not ex_s.get("mid") or ex_l.get("mid") is None:
        return None
    final_val = ex_s["mid"] - ex_l["mid"]
    pnl = (credit - final_val) * LOT
    return {"short": sk, "long": lk, "credit_pts": round(credit, 2),
            "final_val_pts": round(final_val, 2), "pnl_rupees": round(pnl), }


def main() -> int:
    now = datetime.now(IST)
    day = now.strftime("%Y-%m-%d")
    snap_file = STATE / "rolling_option_live" / day / "chain_snapshots.jsonl"
    if not snap_file.exists():
        print(f"[shadow] no capture for {day}; skip")
        return 0
    snaps = []
    for line in snap_file.read_text().splitlines():
        try:
            t, spot, rows = _parse_snapshot(json.loads(line))
        except Exception:
            continue
        if t >= "09:15" and spot > 0 and any((s.get("pe") or {}).get("bid", 0) > 0 for s in rows.values()):
            snaps.append((t, spot, rows))
    if len(snaps) < 2:
        print(f"[shadow] <2 live snapshots for {day}; skip")
        return 0

    entry = next((s for s in snaps if s[0] >= ENTRY_AFTER), snaps[0])
    exit_ = next((s for s in reversed(snaps) if s[0] <= MARK_BY), snaps[-1])
    (te, spot_e, rows_e), (tx, spot_x, rows_x) = entry, exit_
    move_pct = (spot_x - spot_e) / spot_e * 100 if spot_e else 0.0
    realized = "UP" if move_pct >= 0.3 else "DOWN" if move_pct <= -0.3 else "RANGE"

    bull_put = _credit_spread(rows_e, rows_x, "pe")   # profits up/sideways
    bear_call = _credit_spread(rows_e, rows_x, "ce")  # profits down/sideways
    condor = None
    if bull_put and bear_call:
        condor = {"pnl_rupees": bull_put["pnl_rupees"] + bear_call["pnl_rupees"],
                  "credit_pts": round(bull_put["credit_pts"] + bear_call["credit_pts"], 2)}

    record = {
        "date": day, "entry_time": te, "exit_time": tx,
        "spot_entry": round(spot_e, 1), "spot_exit": round(spot_x, 1),
        "day_move_pct": round(move_pct, 2), "realized_regime": realized,
        "snapshots": len(snaps),
        "bull_put": bull_put, "bear_call": bear_call, "iron_condor": condor,
    }
    out = STATE / "shadow_book.jsonl"
    with out.open("a") as f:
        f.write(json.dumps(record) + "\n")
    def _p(x): return f"{x['pnl_rupees']:+,}" if x else "n/a"
    print(f"[shadow] {day} {realized} ({move_pct:+.2f}%) | bull_put {_p(bull_put)} | "
          f"bear_call {_p(bear_call)} | condor {_p(condor)} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
