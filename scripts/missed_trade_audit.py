#!/usr/bin/env python
"""MISSED-TRADE AUDIT — was standing aside actually correct, or did the agent skip a good trade?

"Do nothing" is a valid decision ONLY when nothing good was available. It is NOT automatically good
discipline — and treating every stand-aside as fine is how an agent quietly skips winners while
looking prudent. This audit tests each no-trade day against the best DEFINED-RISK structure that day
(from the shadow book, real fills) and flags a MISS when the agent stood aside while a good trade
existed — plus WHY it stood aside, so a too-conservative guard gets caught instead of praised.

Naked structures (short strangle/straddle) are excluded from "available" — they are banned, so a day
that only a naked short would have won is a genuine no-trade, not a miss.

Run after the close:  python scripts/missed_trade_audit.py
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

STATE = Path(__file__).resolve().parents[1] / "data_engine" / "market_ai" / "state"
DEFINED = ("put_debit", "call_debit", "bull_put", "bear_call", "iron_fly", "iron_condor")
GOOD_TRADE_RUPEES = 500.0     # a defined-risk structure clearing this = a trade worth having taken


def _jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _agent_traded() -> dict[str, float]:
    """date -> realized agent P&L (only days it actually opened a position)."""
    out: dict[str, float] = collections.defaultdict(float)
    seen = set()
    for r in _jsonl(STATE / "intraday_v83_paper_live_trades.jsonl"):
        if r.get("event") == "PAPER_ENTRY":
            seen.add(str(r.get("session_date")))
        if r.get("event") == "PAPER_EXIT" and r.get("realized_paper_pnl") is not None:
            out[str(r.get("session_date"))] += float(r["realized_paper_pnl"])
    return {d: out.get(d, 0.0) for d in seen}


def _standaside_reason(day: str) -> str:
    """Dominant reason the agent stood aside that day (from the runner-log rationale)."""
    c = collections.Counter()
    f = STATE / "intraday_v83_runner.log"
    if not f.exists():
        return "?"
    for line in f.read_text().splitlines():
        if f"{day}T" not in line or not line.strip().startswith("{"):
            continue
        try:
            j = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (j.get("metadata") or {}).get("selector_family") == "STAND_ASIDE" or j.get("action") == "NO_TRADE":
            rat = (j.get("rationale") or [""])[0]
            for key in ("cheap", "before", "blackout", "OVERHEAD", "VETO", "CHOP", "RANGE_TIGHT", "INSUFFICIENT", "MIN_5M"):
                if key.lower() in rat.lower():
                    c[key] += 1
                    break
    return c.most_common(1)[0][0] if c else "no clear reason"


def main() -> int:
    shadow = _jsonl(STATE / "shadow_book.jsonl")
    traded = _agent_traded()
    if not shadow:
        print("No shadow-book history yet.")
        return 0

    rows = []
    for r in shadow:
        d = str(r.get("date"))
        s = r.get("structures") or {}
        avail = {k: (s.get(k) or {}).get("pnl_rupees") for k in DEFINED if isinstance(s.get(k), dict)}
        avail = {k: v for k, v in avail.items() if v is not None}
        if not avail:
            continue
        best_k = max(avail, key=avail.get)
        best_v = avail[best_k]
        agent_pnl = traded.get(d)
        did_trade = d in traded
        rows.append({"date": d, "did_trade": did_trade, "agent_pnl": agent_pnl,
                     "best_struct": best_k, "best_pnl": best_v,
                     "miss": (not did_trade) and best_v >= GOOD_TRADE_RUPEES})

    print("MISSED-TRADE AUDIT — was standing aside correct, or a skipped winner?\n")
    print(f"  {'date':<12}{'agent':>10}{'best-defined avail':>22}{'verdict':>26}")
    print("  " + "-" * 70)
    misses = 0
    missed_total = 0.0
    for r in sorted(rows, key=lambda x: x["date"]):
        if r["did_trade"]:
            verdict = f"traded ({r['agent_pnl']:+,.0f})"
        elif r["miss"]:
            verdict = "*** MISS — stood aside ***"
            misses += 1
            missed_total += r["best_pnl"]
        else:
            verdict = "ok — nothing good"
        agent = f"{r['agent_pnl']:+,.0f}" if r["did_trade"] else "aside"
        print(f"  {r['date']:<12}{agent:>10}{r['best_struct']+' '+format(r['best_pnl'],'+,.0f'):>22}{verdict:>26}")

    print("  " + "-" * 70)
    na = sum(1 for r in rows if not r["did_trade"])
    print(f"\n  stand-aside days: {na}   of which MISSES (a defined-risk winner was available): {misses}")
    print(f"  total P&L left on the table by standing aside on winners: ~Rs {missed_total:+,.0f}")
    if misses:
        print("\n  WHY it stood aside on the missed days (this is the guard to challenge, not praise):")
        for r in sorted(rows, key=lambda x: x["date"]):
            if r["miss"]:
                print(f"    {r['date']}: skipped {r['best_struct']} {r['best_pnl']:+,.0f}  ->  {_standaside_reason(r['date'])}")
    print("\n  RULE: a stand-aside is only 'good' on the 'nothing good' rows. Every MISS is a guard to")
    print("  re-examine — standing aside is not automatically discipline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
