#!/usr/bin/env python3
"""Structure-choice scorecard: for each directional read, did BUYING (debit) or SELLING (credit) win?

Reads state/pair_shadow.jsonl (ghost debit+credit pairs, see intraday_defined_risk/pair_shadow.py).
For every direction it shows the head-to-head overall and split by the market features captured at
entry — the decision table a structure-selection rule would be built from. Promotion bar per cell
(project memory project_put_to_call_transition): >= 20 pairs, the winner ahead on total AND on worst
loss within 2x credit; nothing drives live selection until a cell clears it.

  python scripts/pair_shadow_report.py [--since YYYY-MM-DD] [--exit seller|ride]
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

STATE = Path(__file__).resolve().parents[1] / "data_engine" / "market_ai" / "state"
MIN_PAIRS = 20
PAIRS = {"BEARISH": ("PUT_DEBIT", "BEAR_CALL"), "BULLISH": ("CALL_DEBIT", "BULL_PUT")}


def _jsonl(path: Path) -> list[dict]:
    try:
        return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    except OSError:
        return []


def _pnl(leg: dict, mode: str) -> float | None:
    c = (leg or {}).get("candidates") or {}
    if mode == "ride":
        return (c.get("RIDE_TO_CLOSE") or {}).get("pnl_rupees")
    return ((c.get("SELLER_EXIT") or c.get("DEBIT_EXIT") or {}).get("pnl_rupees"))


def _bucket(row: dict, key: str) -> str:
    f = row.get("features") or {}
    if key == "efficiency":
        v = f.get("trend_efficiency_ratio")
        return "?" if v is None else ("<0.35 (no momentum)" if v < 0.35 else ("0.35-0.55" if v < 0.55 else ">=0.55 (strong)"))
    if key == "time":
        m = row.get("minutes_since_open") or 0
        return "09:15-10:30" if m < 75 else ("10:30-12:30" if m < 195 else "12:30-15:15")
    if key == "conviction":
        v = f.get("selector_conviction")
        return "?" if v is None else ("<0.6" if v < 0.6 else ">=0.6")
    if key == "trigger":
        return str(row.get("trigger"))
    return str(f.get(key, "?"))


def _line(label: str, rows: list[dict], debit: str, credit: str, mode: str) -> str:
    d = [x for x in (_pnl(r["structures"].get(debit), mode) for r in rows) if x is not None]
    c = [x for x in (_pnl(r["structures"].get(credit), mode) for r in rows) if x is not None]
    both = [r for r in rows if _pnl(r["structures"].get(debit), mode) is not None
            and _pnl(r["structures"].get(credit), mode) is not None]
    sell_wins = sum(1 for r in both if _pnl(r["structures"][credit], mode) > _pnl(r["structures"][debit], mode))
    f = lambda v: f"{sum(v):>8,.0f} ({100*sum(x > 0 for x in v)/len(v):3.0f}% win)" if v else "      —"
    pick = "—"
    if len(both) >= MIN_PAIRS:
        pick = "SELL" if sum(c) > sum(d) else "BUY"
    return (f"  {label:24} n={len(rows):3d} | {debit:10} {f(d)} | {credit:9} {f(c)} | "
            f"sell better {sell_wins}/{len(both)} | pick: {pick}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="")
    ap.add_argument("--exit", default="seller", choices=["seller", "ride"])
    a = ap.parse_args()
    rows = [r for r in _jsonl(STATE / "pair_shadow.jsonl") if str(r.get("session_date")) >= a.since]
    if not rows:
        print("No structure-choice pairs recorded yet.")
        return 0
    print(f"Exit rule: {'managed (seller / debit exits)' if a.exit == 'seller' else 'hold to 15:15'}   "
          f"— a cell needs >= {MIN_PAIRS} pairs before it may 'pick'")
    for direction, (debit, credit) in PAIRS.items():
        sub = [r for r in rows if r.get("direction") == direction]
        if not sub:
            continue
        print(f"\n=== {direction}: {debit} (buy) vs {credit} (sell)")
        print(_line("ALL", sub, debit, credit, a.exit))
        for key in ("trigger", "efficiency", "conviction", "time", "vol_regime", "selector_iv",
                    "option_chain_pressure_state"):
            groups = defaultdict(list)
            for r in sub:
                groups[_bucket(r, key)].append(r)
            print(f"  -- by {key}")
            for k, v in sorted(groups.items()):
                print(_line(f"  {k}", v, debit, credit, a.exit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
