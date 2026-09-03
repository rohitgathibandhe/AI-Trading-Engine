#!/usr/bin/env python
"""WEEKLY POSITIONAL — daily monitor. Marks each open paper position to the live chain, runs the
defense ladder (weekly_positional_brain.evaluate_management), and reports P&L + the recommended action.
On a management action it appends a PAPER_EXIT / PAPER_ADJUST row to the ledger.

  python scripts/weekly_positional_monitor.py            # report + act
  python scripts/weekly_positional_monitor.py --report   # report only, take no action
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "data_engine"))

import weekly_ic_executor as wx
import weekly_positional_paper as wp
from market_ai.intraday_defined_risk import weekly_positional_brain as brain

STATE = ROOT / "data_engine" / "market_ai" / "state"
LEDGER = STATE / "weekly_positional_paper.jsonl"


def _rows() -> list[dict]:
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()] if LEDGER.exists() else []


def _open_positions() -> list[dict]:
    """Entries with no later EXIT for the same expiry+structure."""
    rows = _rows()
    closed = {(r.get("expiry"), r.get("structure")) for r in rows if r.get("event") in ("PAPER_EXIT",)}
    return [r for r in rows if r.get("event") == "PAPER_ENTRY" and (r.get("expiry"), r.get("structure")) not in closed]


def _trading_days_to(expiry: str) -> int:
    d0, d1 = date.today(), date.fromisoformat(expiry)
    n, d = 0, d0
    from datetime import timedelta
    while d < d1:
        d += timedelta(days=1)
        if d.weekday() < 5:
            n += 1
    return n


def _spread_value_pts(strikes, deltas, legs) -> tuple[float, float]:
    """Current cost-to-close (points) and the max |short delta| across the position."""
    val, max_sd = 0.0, 0.0
    for lg in legs:
        opt = "ce" if lg["opt"] == "CE" else "pe"
        node = strikes.get(wp._nearest_strike(strikes, lg["strike"]), {})
        px = wp._mid(node, opt)
        val += px if lg["action"] == "SELL" else -px      # cost to CLOSE = buy back shorts, sell longs
        if lg["action"] == "SELL":
            max_sd = max(max_sd, deltas.get(wp._nearest_strike(strikes, lg["strike"]), {}).get(opt, 0.0))
    return val, max_sd


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

    for pos in positions:
        expiry = pos["expiry"]
        raw = wx._get_raw_chain(creds, expiry)
        parsed = wx._parse_chain(raw, spot=float(m.get("atm_strike") or 0))
        if not parsed.get("strikes"):
            print(f"[{expiry}] chain unavailable — skipping this cycle."); continue
        deltas = wp._deltas_by_strike(raw)
        cur_val, max_sd = _spread_value_pts(parsed["strikes"], deltas, pos["legs"])

        credit_pts = pos["credit_points"]
        mtm_pts = credit_pts - cur_val                       # + = profit (spread cheaper to close than credit)
        mtm_rupees = round(mtm_pts * wx.LOT_SIZE * pos["lots"], 0)
        dte = _trading_days_to(expiry)

        current = {"mtm_rupees": mtm_rupees, "max_short_delta": max_sd,
                   "days_to_expiry": dte, "broader_trend": cur_trend}
        decision = brain.evaluate_management(pos, current)

        print(f"\n=== {pos['structure']} ({pos['direction']}) exp {expiry} ===")
        print(f"  credit Rs {pos['credit_rupees']:,.0f} | now MTM Rs {mtm_rupees:+,.0f} "
              f"({mtm_pts:+.1f} pts) | short delta {max_sd:.2f} | {dte} trading days left | trend now {cur_trend}")
        print(f"  DEFENSE LADDER -> {decision['action']}: {decision['reason']}")

        if not args.report and decision["action"] != "HOLD":
            evt = {"event": "PAPER_EXIT" if decision["action"] in
                   ("TAKE_PROFIT", "STOP_CLOSE", "GAMMA_EXIT", "CLOSE_THESIS_BROKEN") else "PAPER_ADJUST",
                   "ts": datetime.now().isoformat(timespec="seconds"), "book": "WEEKLY_POSITIONAL",
                   "expiry": expiry, "structure": pos["structure"], "action": decision["action"],
                   "reason": decision["reason"], "realized_rupees": mtm_rupees if "EXIT" in
                   ("PAPER_EXIT" if decision["action"] in
                    ("TAKE_PROFIT", "STOP_CLOSE", "GAMMA_EXIT", "CLOSE_THESIS_BROKEN") else "") else None}
            with LEDGER.open("a") as f:
                f.write(json.dumps(evt) + "\n")
            print(f"  -> recorded {evt['event']} ({decision['action']}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
