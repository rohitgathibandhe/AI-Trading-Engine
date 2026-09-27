#!/usr/bin/env python
"""WEEKLY POSITIONAL — PAPER entry executor.

Runs the weekly_positional_brain against the LIVE option chain and places a PAPER trade for the
NEXT-week expiry: the directional defined-risk structure the broader trend calls for, with gap-safe
strikes and a premium gate. Writes one PAPER_ENTRY to state/weekly_positional_paper.jsonl. No real
orders. Management (the defense ladder) is a separate daily monitor.

  python scripts/weekly_positional_paper.py            # plan + place a paper entry from live data
  python scripts/weekly_positional_paper.py --dry-run  # plan only, don't write the ledger
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

import weekly_ic_executor as wx  # reuse chain fetch / parse / analyse / expiry / creds
from market_ai.intraday_defined_risk import weekly_positional_brain as brain

STATE = ROOT / "data_engine" / "market_ai" / "state"
LEDGER = STATE / "weekly_positional_paper.jsonl"
LOTS = 3


def _latest_metadata() -> dict:
    """The brain's inputs (daily trend, EMAs, walls, expected move, ATR, vol regime) from the newest
    runner-log decision."""
    log = STATE / "intraday_v83_runner.log"
    for line in reversed(log.read_text(errors="ignore").splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        m = (j.get("metadata") or {})
        if len(m) > 50:
            return m
    return {}


def _next_week_expiry() -> str:
    """The NEXT weekly expiry (not the nearest): the Tuesday after the coming one."""
    nearest = wx._next_tuesday(date.today())
    return (nearest + timedelta(days=7)).isoformat()


TARGET_SHORT_DELTA = 0.18   # place the short leg here (standard credit-spread delta), gap-floored below


def _nearest_strike(strikes: dict, target: float) -> float:
    return min(strikes.keys(), key=lambda k: abs(k - target)) if strikes else target


def _mid(node: dict, opt: str) -> float:
    bid, ask, ltp = node.get(f"{opt}_bid", 0), node.get(f"{opt}_ask", 0), node.get(f"{opt}_ltp", 0)
    return (bid + ask) / 2 if (bid > 0 and ask > 0) else ltp


def _deltas_by_strike(raw: dict) -> dict:
    """Extract |delta| per strike from the raw chain greeks (parse_chain drops delta)."""
    data = raw.get("data") or raw
    if isinstance(data.get("data"), dict):
        data = data["data"]
    out = {}
    for sk, node in (data.get("oc") or {}).items():
        try:
            k = float(sk)
        except (TypeError, ValueError):
            continue
        out[k] = {"ce": abs(float(((node.get("ce") or {}).get("greeks") or {}).get("delta") or 0)),
                  "pe": abs(float(((node.get("pe") or {}).get("greeks") or {}).get("delta") or 0))}
    return out


def _short_by_delta(strikes, deltas, opt, spot, gap_floor, side_up: bool) -> float:
    """Choose the short strike nearest TARGET_SHORT_DELTA, but never closer than the gap floor from
    spot (so a typical overnight gap can't breach it). side_up=True for calls (strike above spot)."""
    floor_k = spot + gap_floor if side_up else spot - gap_floor
    cands = [k for k in strikes if (k >= floor_k if side_up else k <= floor_k) and deltas.get(k, {}).get(opt, 0) > 0]
    if not cands:  # low-IV: even the gap-floor strike may be past the delta target — take the floor
        return wx._round_strike(floor_k)
    return min(cands, key=lambda k: abs(deltas[k][opt] - TARGET_SHORT_DELTA))


def build_position(plan: dict, analysis: dict, parsed: dict, deltas: dict) -> dict | None:
    strikes = parsed["strikes"]
    spot = analysis["spot"]
    gap_floor = plan["min_short_distance_pts"]
    wing = wx.WING_WIDTH_CALL
    legs, credit_pts, width = [], 0.0, wing
    struct = plan["structure"]

    if struct in ("BEAR_CALL_CREDIT_SPREAD", "IRON_CONDOR"):
        s = _short_by_delta(strikes, deltas, "ce", spot, gap_floor, side_up=True)
        l = _nearest_strike(strikes, s + wing)
        credit_pts += _mid(strikes[s], "ce") - _mid(strikes[l], "ce")
        legs += [{"action": "SELL", "opt": "CE", "strike": s, "delta": round(deltas.get(s, {}).get("ce", 0), 3)},
                 {"action": "BUY", "opt": "CE", "strike": l}]
    if struct in ("BULL_PUT_CREDIT_SPREAD", "IRON_CONDOR"):
        s = _short_by_delta(strikes, deltas, "pe", spot, gap_floor, side_up=False)
        l = _nearest_strike(strikes, s - wing)
        credit_pts += _mid(strikes[s], "pe") - _mid(strikes[l], "pe")
        legs += [{"action": "SELL", "opt": "PE", "strike": s, "delta": round(deltas.get(s, {}).get("pe", 0), 3)},
                 {"action": "BUY", "opt": "PE", "strike": l}]

    if credit_pts <= 0:
        return {"error": f"no net credit available ({credit_pts:.1f} pts)"}
    credit_ratio = credit_pts / width
    max_loss_pts = width - credit_pts
    return {
        "structure": struct, "direction": plan["direction"], "legs": legs,
        "credit_points": round(credit_pts, 2), "credit_ratio": round(credit_ratio, 3),
        "credit_rupees": round(credit_pts * wx.LOT_SIZE * LOTS, 0),
        "max_loss_rupees": round(max_loss_pts * wx.LOT_SIZE * LOTS, 0),
        "wing_width": width, "lots": LOTS, "spot_at_entry": spot,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # FLAT-GUARD: this executor is scheduled daily, so it must NEVER stack a second position. If a weekly
    # positional trade is already open (the defense-ladder monitor is managing it), stand aside. Reuses the
    # monitor's reconstruction (folds PAPER_ADJUST, drops on PAPER_EXIT) so it reads the ledger identically.
    import weekly_positional_monitor as wm
    _open = wm._open_positions()
    if _open:
        print(f"=> already holding {len(_open)} weekly positional position(s) "
              f"({', '.join(str(p.get('structure', '?')) for p in _open)}) — managed by the monitor; no new entry.")
        return 0

    m = _latest_metadata()
    plan = brain.plan_entry(m)
    print(f"BRAIN: {plan['action']}  {plan.get('structure','')}  ({plan.get('direction','')})")
    print(f"  why: {plan.get('why','')}")
    if plan["action"] != "TRADE":
        print("=> stand aside this week.")
        return 0

    expiry = _next_week_expiry()
    print(f"\nNEXT-WEEK EXPIRY: {expiry}")
    creds = wx._load_creds()
    raw = wx._get_raw_chain(creds, expiry)
    parsed = wx._parse_chain(raw, spot=float(m.get("atm_strike") or 0))
    if not parsed.get("strikes"):
        print("=> option chain unavailable (market closed / data). Re-run in market hours.")
        return 0
    analysis = wx._analyse(parsed)
    if not analysis.get("ok"):
        print(f"=> {analysis.get('reason')}")
        return 0

    deltas = _deltas_by_strike(raw)
    pos = build_position(plan, analysis, parsed, deltas)
    if pos is None or pos.get("error"):
        print(f"=> cannot build: {pos.get('error') if pos else 'none'}")
        return 0

    print(f"\nPOSITION: {pos['structure']} ({pos['direction']}) x{pos['lots']} lots, expiry {expiry}")
    for lg in pos["legs"]:
        print(f"   {lg['action']} {lg['opt']} {lg['strike']:.0f}")
    print(f"  credit: {pos['credit_points']:.1f} pts (ratio {pos['credit_ratio']:.2f}) = Rs {pos['credit_rupees']:,.0f}")
    print(f"  max loss: Rs {pos['max_loss_rupees']:,.0f}   spot {pos['spot_at_entry']:.0f}")

    # Premium gate
    if pos["credit_ratio"] < plan["min_credit_ratio"]:
        print(f"=> STAND ASIDE: credit ratio {pos['credit_ratio']:.2f} < floor {plan['min_credit_ratio']:.2f} "
              f"(premium too thin for the gap risk).")
        return 0

    entry = {
        "event": "PAPER_ENTRY", "ts": datetime.now().isoformat(timespec="seconds"),
        "book": "WEEKLY_POSITIONAL", "expiry": expiry, "adjusted": False,
        "why": plan["why"], **pos,
    }
    if args.dry_run:
        print("\n[dry-run] would record this PAPER_ENTRY.")
        return 0
    with LEDGER.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"\nPAPER_ENTRY recorded -> {LEDGER.name}. Held to next-week expiry, managed by the defense ladder.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
