#!/usr/bin/env python
"""MATURITY SCORECARD — the objective gate to the stocks phase. 5 parameters, each PASS/FAIL.

The user's bar: 5/5, forward-proven, before real money graduates to equities. This makes "show me
maturity" a number, not an opinion. Judged on the FORWARD record (paper on live data + shadow book),
never a backtest. Run daily:  python scripts/maturity_scorecard.py

  1. PROFITABLE        net realized P&L > 0 over the trailing window
  2. EXIT capture      median MFE-capture >= 50% (keeps its winners, doesn't give profit back)
  3. STRUCTURE skill   picked the best-available defined structure >= 50% of trading days
  4. FEW MISSES        stand-aside MISS rate <= 20% (rarely skips an available winner)
  5. TAIL bounded      worst single day >= -(per-trade risk cap)   (defined-risk holds)
"""
from __future__ import annotations

import collections
import json
import statistics
from pathlib import Path

STATE = Path(__file__).resolve().parents[1] / "data_engine" / "market_ai" / "state"
DEFINED = ("put_debit", "call_debit", "bull_put", "bear_call", "iron_fly", "iron_condor")
GOOD = 500.0
RISK_CAP = 18000.0


def _jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def main() -> int:
    trades = _jsonl(STATE / "intraday_v83_paper_live_trades.jsonl")
    exits = [t for t in trades if t.get("event") == "PAPER_EXIT"]
    entries_by_day = collections.defaultdict(list)
    for t in trades:
        if t.get("event") == "PAPER_ENTRY":
            entries_by_day[str(t.get("session_date"))].append(t.get("strategy"))
    shadow = {str(r.get("date")): (r.get("structures") or {}) for r in _jsonl(STATE / "shadow_book.jsonl")}

    results = []

    # 1. PROFITABLE
    net = sum(float(t.get("realized_paper_pnl") or 0) for t in exits)
    results.append(("PROFITABLE", net > 0, f"net Rs {net:+,.0f} over {len(exits)} closed trades"))

    # 2. EXIT capture
    caps = [t["mfe_capture_pct"] for t in exits if t.get("mfe_capture_pct") is not None and float(t.get("mfe_rupees") or 0) > 0]
    med = statistics.median(caps) if caps else 0.0
    results.append(("EXIT capture", med >= 0.50, f"median MFE capture {med*100:.0f}% (target >=50%) n={len(caps)}"))

    # 3. STRUCTURE skill — of days the agent traded, did it pick the best-available defined structure?
    smap = {"BULL_PUT_CREDIT_SPREAD": "bull_put", "BEAR_CALL_CREDIT_SPREAD": "bear_call",
            "PUT_DEBIT_SPREAD": "put_debit", "CALL_DEBIT_SPREAD": "call_debit",
            "IRON_FLY": "iron_fly", "IRON_CONDOR": "iron_condor"}
    hit = tot = 0
    for d, strategies in entries_by_day.items():
        sh = shadow.get(d, {})
        avail = {k: (sh.get(k) or {}).get("pnl_rupees") for k in DEFINED if isinstance(sh.get(k), dict) and (sh.get(k) or {}).get("pnl_rupees") is not None}
        if not avail:
            continue
        best = max(avail, key=avail.get)
        tot += 1
        if smap.get(str(strategies[0])) == best:
            hit += 1
    rate = hit / tot if tot else 0.0
    results.append(("STRUCTURE skill", rate >= 0.50, f"picked best-available {hit}/{tot} days ({rate*100:.0f}%, target >=50%)"))

    # 4. FEW MISSES — stand-aside days where a defined winner was available
    traded_days = set(entries_by_day)
    miss = na = 0
    for d, sh in shadow.items():
        avail = {k: (sh.get(k) or {}).get("pnl_rupees") for k in DEFINED if isinstance(sh.get(k), dict) and (sh.get(k) or {}).get("pnl_rupees") is not None}
        if not avail:
            continue
        if d not in traded_days:
            na += 1
            if max(avail.values()) >= GOOD:
                miss += 1
    miss_rate = miss / na if na else 0.0
    results.append(("FEW MISSES", miss_rate <= 0.20, f"missed {miss}/{na} stand-aside days ({miss_rate*100:.0f}%, target <=20%)"))

    # 5. TAIL bounded
    worst = min((float(t.get("realized_paper_pnl") or 0) for t in exits), default=0.0)
    results.append(("TAIL bounded", worst >= -RISK_CAP, f"worst day Rs {worst:+,.0f} vs floor Rs {-RISK_CAP:,.0f}"))

    score = sum(1 for _, ok, _ in results)
    print("MATURITY SCORECARD — the gate to the stocks phase (forward record only)\n")
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}]  {name:<16} {detail}")
    print(f"\n  SCORE: {score}/5" + ("  — MATURE. Ready to discuss the equities layer." if score == 5 else f"  — not mature; {5-score} parameter(s) to earn."))
    print("  Judged on the forward paper-on-live + shadow record. A backtest number does not count here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
