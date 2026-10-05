#!/usr/bin/env python
"""Score the VWAP ruleset in REAL OPTION P&L (not just index points).

For each shadow-book day we take the ruleset's day-level read (from the candles) and map it to the
defined-risk structure it implies, then look up that structure's ACTUAL fill P&L in the shadow book:

  TREND_UP   -> CALL_DEBIT   (buy the up-move)
  TREND_DOWN -> PUT_DEBIT    (buy the down-move)
  RANGE      -> IRON_CONDOR  (sell the range)

Compares the ruleset's structure-selection to fixed baselines (always-condor, always-put-debit) so we
can see whether reading the tape the ruleset's way actually beats a naive fixed choice — in rupees.

  python scripts/intraday_ruleset_option_score.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import intraday_ruleset_backtest as bt

STATE = ROOT / "data_engine" / "market_ai" / "state"
SHADOW = STATE / "shadow_book.jsonl"

MAP = {"TREND_UP": "call_debit", "TREND_DOWN": "put_debit", "RANGE": "iron_condor"}


def _pnl(structs, key):
    v = (structs or {}).get(key)
    return float(v.get("pnl_rupees")) if isinstance(v, dict) and v.get("pnl_rupees") is not None else None


def main():
    rows = [json.loads(l) for l in SHADOW.read_text().splitlines() if l.strip()]
    ruleset = []          # (date, regime, structure, pnl)
    base_condor = []
    base_putdebit = []
    best = []
    for r in rows:
        day = str(r.get("date"))
        cc = STATE / f"candle_cache_{day}_5m.json"
        if not cc.exists():
            continue
        try:
            sig = bt.day_signal(bt._load(cc))
        except Exception:
            sig = None
        if not sig:
            continue
        s = r.get("structures") or {}
        p = _pnl(s, MAP[sig])
        if p is not None:
            ruleset.append((day, sig, MAP[sig], p))
        pc = _pnl(s, "iron_condor")
        if pc is not None:
            base_condor.append(pc)
        pp = _pnl(s, "put_debit")
        if pp is not None:
            base_putdebit.append(pp)
        # best-possible (hindsight ceiling) among the defined structures
        cand = [x for x in (_pnl(s, k) for k in ("call_debit", "put_debit", "iron_condor", "iron_fly",
                                                 "bull_put", "bear_call")) if x is not None]
        if cand:
            best.append(max(cand))

    def stats(pnls):
        if not pnls:
            return "n=0"
        w = sum(1 for x in pnls if x > 0)
        return f"n={len(pnls)}  total Rs {sum(pnls):+,.0f}  avg {sum(pnls)/len(pnls):+,.0f}  win {100*w/len(pnls):.0f}%"

    rp = [p for _, _, _, p in ruleset]
    print("VWAP RULESET — real option P&L vs the shadow book's actual fills\n")
    print(f"  RULESET structure choice : {stats(rp)}")
    print(f"  baseline always-condor   : {stats(base_condor)}")
    print(f"  baseline always-put-debit: {stats(base_putdebit)}")
    print(f"  hindsight best-possible  : {stats(best)}")
    from collections import Counter
    reg = Counter(sig for _, sig, _, _ in ruleset)
    print(f"\n  ruleset regime mix: {dict(reg)}")
    print("\n  per-day (ruleset pick):")
    for day, sig, st, p in ruleset:
        print(f"   {day}  {sig:10} -> {st:12} Rs {p:+,.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
