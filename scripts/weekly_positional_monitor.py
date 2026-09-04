#!/usr/bin/env python
"""WEEKLY POSITIONAL — daily monitor + ADJUSTMENT ENGINE.

Marks each open paper position to the live chain, runs the mature defense ladder
(weekly_positional_brain.evaluate_management), and — this is the maturity — actually EXECUTES the
adjustment against the live chain, enforcing the credit-only rule and tracking cumulative credit:

  LEG_INTO_CONDOR  build the opposite-side spread (delta-placed), price it; apply ONLY if it is a
                   credit. Widens the tested breakeven, stays defined-risk. Position -> condor.
  ROLL_OUT         roll the tested spread to next week at the same short; apply ONLY if the new credit
                   covers the cost to close the old (net >= 0). Else escalate to CLOSE.
  HARVEST_UNTESTED buy back the near-worthless untested side, locking its credit and dropping its risk.
  TAKE_PROFIT / STOP_CLOSE / GAMMA_EXIT / CLOSE_* -> close the whole position, realize P&L.

Positions are RECONSTRUCTED from the ledger (PAPER_ENTRY + every PAPER_ADJUST), so the current legs,
cumulative credit and adjustment count are always exact. Writes PAPER_ADJUST / PAPER_EXIT rows keyed by
the entry timestamp, and a snapshot the dashboard serves.

  python scripts/weekly_positional_monitor.py            # mark, decide, and EXECUTE adjustments
  python scripts/weekly_positional_monitor.py --report   # mark + decide only, take no action
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "data_engine"))

import weekly_ic_executor as wx
import weekly_positional_paper as wp
from market_ai.intraday_defined_risk import weekly_positional_brain as brain

STATE = ROOT / "data_engine" / "market_ai" / "state"
LEDGER = STATE / "weekly_positional_paper.jsonl"
LOTS_MULT = wx.LOT_SIZE


def _rows() -> list[dict]:
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()] if LEDGER.exists() else []


def _open_positions() -> list[dict]:
    """Reconstruct each open position by folding PAPER_ADJUST events onto its PAPER_ENTRY. Keyed by the
    entry timestamp; dropped once a PAPER_EXIT for that key appears."""
    positions: dict[str, dict] = {}
    order: list[str] = []
    for r in _rows():
        ev, key = r.get("event"), r.get("entry_key") or r.get("ts")
        if ev == "PAPER_ENTRY":
            p = dict(r)
            p.update({"_key": r.get("ts"), "legs": list(r.get("legs") or []),
                      "total_credit_rupees": float(r.get("credit_rupees") or 0),
                      "adjustments": 0, "adjusted": False, "lots": r.get("lots") or wx.DEFAULT_LOTS})
            positions[r.get("ts")] = p
            order.append(r.get("ts"))
        elif ev == "PAPER_ADJUST" and key in positions:
            p = positions[key]
            if r.get("legs_replace") is not None:
                p["legs"] = list(r["legs_replace"])            # roll: swap the tested spread
            else:
                p["legs"] = p["legs"] + list(r.get("legs_added") or [])
            p["total_credit_rupees"] += float(r.get("credit_added") or 0)
            p["adjustments"] += 1
            p["adjusted"] = True
            if r.get("new_structure"):
                p["structure"] = r["new_structure"]
            if r.get("new_expiry"):
                p["expiry"] = r["new_expiry"]
        elif ev == "PAPER_EXIT" and key in positions:
            positions.pop(key, None)
    return [positions[k] for k in order if k in positions]


def _trading_days_to(expiry: str) -> int:
    d, d1, n = date.today(), date.fromisoformat(expiry), 0
    while d < d1:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


def _side_value_pts(strikes, legs, opt_key: str) -> float:
    """Cost to close one side's spread (buy back short, sell long) in points."""
    val = 0.0
    for lg in legs:
        if ("CE" if opt_key == "ce" else "PE") != lg["opt"]:
            continue
        px = wp._mid(strikes.get(wp._nearest_strike(strikes, lg["strike"]), {}), opt_key)
        val += px if lg["action"] == "SELL" else -px
    return val


def _mark(strikes, deltas, legs):
    """(cost_to_close_pts, per-side max short delta) across the whole position."""
    call_v = _side_value_pts(strikes, legs, "ce")
    put_v = _side_value_pts(strikes, legs, "pe")
    ce_sd = pe_sd = 0.0
    for lg in legs:
        if lg["action"] != "SELL":
            continue
        opt = "ce" if lg["opt"] == "CE" else "pe"
        sd = deltas.get(wp._nearest_strike(strikes, lg["strike"]), {}).get(opt, 0.0)
        if opt == "ce":
            ce_sd = max(ce_sd, sd)
        else:
            pe_sd = max(pe_sd, sd)
    return call_v + put_v, call_v, put_v, ce_sd, pe_sd


def _build_side_spread(parsed, deltas, opt_key: str, spot: float, gap_floor: float):
    """Build a fresh credit spread on one side (delta-placed short, wing-width long). Returns
    (legs, credit_pts) or (None, 0) if it can't be priced for a credit."""
    strikes = parsed["strikes"]
    up = (opt_key == "ce")
    s = wp._short_by_delta(strikes, deltas, opt_key, spot, gap_floor, side_up=up)
    l = wp._nearest_strike(strikes, s + wx.WING_WIDTH_CALL) if up else wp._nearest_strike(strikes, s - wx.WING_WIDTH_PUT)
    credit = wp._mid(strikes[s], opt_key) - wp._mid(strikes[l], opt_key)
    OPT = "CE" if up else "PE"
    legs = [{"action": "SELL", "opt": OPT, "strike": s, "delta": round(deltas.get(s, {}).get(opt_key, 0), 3)},
            {"action": "BUY", "opt": OPT, "strike": l}]
    return (legs, credit) if credit > 0 else (None, 0.0)


def _rupees(pts: float, lots: int) -> float:
    return round(pts * wx.LOT_SIZE * lots, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="report only, take no action")
    args = ap.parse_args()

    positions = _open_positions()
    if not positions:
        print("No open weekly positional paper positions.")
        return 0

    creds = wx._load_creds()
    m = wp._latest_metadata()
    cur_trend = brain.broader_trend(m)
    spot_hint = float(m.get("atm_strike") or 0)
    snapshot = []
    new_events: list[dict] = []

    for pos in positions:
        expiry, lots = pos["expiry"], int(pos.get("lots") or wx.DEFAULT_LOTS)
        raw = wx._get_raw_chain(creds, expiry)
        parsed = wx._parse_chain(raw, spot=spot_hint)
        if not parsed.get("strikes"):
            print(f"[{expiry}] chain unavailable — skipping."); continue
        deltas = wp._deltas_by_strike(raw)
        strikes = parsed["strikes"]
        spot = parsed["spot"]

        cur_val, call_v, put_v, ce_sd, pe_sd = _mark(strikes, deltas, pos["legs"])
        credit_pts_now = pos["total_credit_rupees"] / (wx.LOT_SIZE * lots)
        mtm_rupees = _rupees(credit_pts_now - cur_val, lots)
        max_sd = max(ce_sd, pe_sd)
        tested_side = "CALL" if ce_sd >= pe_sd else "PUT"
        dte = _trading_days_to(expiry)

        # untested-side decay% (condor only): 1 - current untested-side value / its entry credit.
        untested_decay = 0.0
        if pos.get("adjusted"):
            unt_val = put_v if tested_side == "CALL" else call_v
            # entry credit of the untested side ~ the adjustment credit that created it (approx via legs)
            unt_entry = abs(pos.get("_untested_entry_pts") or 0) or None
            if unt_entry:
                untested_decay = max(0.0, 1 - unt_val / unt_entry)

        current = {"mtm_rupees": mtm_rupees, "max_short_delta": max_sd, "days_to_expiry": dte,
                   "broader_trend": cur_trend, "tested_side": tested_side, "untested_decay_pct": untested_decay}
        decision = brain.evaluate_management(pos, current)

        print(f"\n=== {pos['structure']} ({pos['direction']}) exp {expiry}  adj={pos['adjustments']} ===")
        print(f"  cum credit Rs {pos['total_credit_rupees']:,.0f} | MTM Rs {mtm_rupees:+,.0f} | "
              f"short Δ {max_sd:.2f} ({tested_side} tested) | {dte}d left | trend {cur_trend}")
        print(f"  LADDER -> {decision['action']}: {decision['reason']}")

        entry_key = pos["_key"]
        acted = None
        if not args.report:
            act = decision["action"]
            if act == "LEG_INTO_CONDOR":
                opp = "pe" if tested_side == "CALL" else "ce"
                gap_floor = brain.plan_entry(m).get("min_short_distance_pts", 250)
                legs, credit = _build_side_spread(parsed, deltas, opp, spot, gap_floor)
                if legs:
                    new_events.append({"event": "PAPER_ADJUST", "ts": datetime.now().isoformat(timespec="seconds"),
                                       "book": "WEEKLY_POSITIONAL", "entry_key": entry_key, "action": act,
                                       "legs_added": legs, "credit_added": _rupees(credit, lots),
                                       "new_structure": "IRON_CONDOR", "reason": decision["reason"]})
                    acted = f"legged into condor (+Rs {_rupees(credit, lots):,.0f} credit)"
                else:
                    acted = "no credit available for the opposite spread — hold"
            elif act == "ROLL_OUT":
                nxt = (date.fromisoformat(expiry) + timedelta(days=7)).isoformat()
                nraw = wx._get_raw_chain(creds, nxt); nparsed = wx._parse_chain(nraw, spot=spot_hint)
                ndeltas = wp._deltas_by_strike(nraw)
                if nparsed.get("strikes"):
                    opt = "ce" if tested_side == "CALL" else "pe"
                    close_cost = _side_value_pts(strikes, pos["legs"], opt)   # cost to close tested side now
                    nlegs, ncredit = _build_side_spread(nparsed, ndeltas, opt, spot,
                                                        brain.plan_entry(m).get("min_short_distance_pts", 250))
                    if nlegs and (ncredit - close_cost) >= 0:
                        kept = [lg for lg in pos["legs"] if lg["opt"] != ("CE" if opt == "ce" else "PE")]
                        new_events.append({"event": "PAPER_ADJUST", "ts": datetime.now().isoformat(timespec="seconds"),
                                           "book": "WEEKLY_POSITIONAL", "entry_key": entry_key, "action": act,
                                           "legs_replace": kept + nlegs, "credit_added": _rupees(ncredit - close_cost, lots),
                                           "new_expiry": nxt, "reason": decision["reason"]})
                        acted = f"rolled {tested_side} to {nxt} (net +Rs {_rupees(ncredit - close_cost, lots):,.0f})"
                    else:
                        act = "CLOSE_NO_CREDIT_ROLL"
                        decision["reason"] += " — no credit roll available, closing"
                if act in ("CLOSE_NO_CREDIT_ROLL",):
                    new_events.append({"event": "PAPER_EXIT", "ts": datetime.now().isoformat(timespec="seconds"),
                                       "book": "WEEKLY_POSITIONAL", "entry_key": entry_key, "action": act,
                                       "realized_rupees": mtm_rupees, "reason": decision["reason"]})
                    acted = "closed (no credit roll)"
            elif act == "HARVEST_UNTESTED":
                opt = "pe" if tested_side == "CALL" else "ce"   # untested side
                buyback = _side_value_pts(strikes, pos["legs"], opt)
                new_events.append({"event": "PAPER_ADJUST", "ts": datetime.now().isoformat(timespec="seconds"),
                                   "book": "WEEKLY_POSITIONAL", "entry_key": entry_key, "action": act,
                                   "legs_replace": [lg for lg in pos["legs"] if lg["opt"] != ("CE" if opt == "ce" else "PE")],
                                   "credit_added": -_rupees(buyback, lots), "reason": decision["reason"]})
                acted = f"harvested untested side (bought back for Rs {_rupees(buyback, lots):,.0f})"
            elif act in ("TAKE_PROFIT", "STOP_CLOSE", "GAMMA_EXIT", "CLOSE_THESIS_BROKEN", "CLOSE_MAX_ADJUSTED"):
                new_events.append({"event": "PAPER_EXIT", "ts": datetime.now().isoformat(timespec="seconds"),
                                   "book": "WEEKLY_POSITIONAL", "entry_key": entry_key, "action": act,
                                   "realized_rupees": mtm_rupees, "reason": decision["reason"]})
                acted = f"closed, realized Rs {mtm_rupees:+,.0f}"
            if acted:
                print(f"  -> EXECUTED: {acted}")

        snapshot.append({"structure": pos["structure"], "direction": pos["direction"], "expiry": expiry,
                         "legs": pos["legs"], "credit_rupees": pos["total_credit_rupees"],
                         "mtm_rupees": mtm_rupees, "short_delta": round(max_sd, 3), "tested_side": tested_side,
                         "days_to_expiry": dte, "trend_now": cur_trend, "adjustments": pos["adjustments"],
                         "ladder_action": decision["action"], "ladder_reason": decision["reason"],
                         "acted": acted, "entry_ts": pos.get("_key")})

    for e in new_events:
        with LEDGER.open("a") as f:
            f.write(json.dumps(e) + "\n")
    (STATE / "weekly_positional_snapshot.json").write_text(json.dumps(
        {"updated": datetime.now().isoformat(timespec="seconds"), "positions": snapshot}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
