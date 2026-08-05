#!/usr/bin/env python
"""DID TODAY'S CHANGES ACTUALLY ENGAGE?

Every expensive failure on this project has been the same shape: the analysis was right and a silent
plumbing failure threw it away, while the logs read like a considered decision.
  - the weekly seller vetoed 9 plans on an HTTP 400 dressed as "market not suitable"
  - the intraday agent detected a pin, selected IRON_FLY, and a missing dict key discarded it —
    -84 taken instead of +5,464

So after a day of behavioural changes, "it ran without errors" is not evidence. This checks that each
change actually FIRED, and says plainly when it did not. Run after the close:

    python scripts/did_changes_engage.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

STATE = Path(__file__).resolve().parents[1] / "data_engine" / "market_ai" / "state"
IST = ZoneInfo("Asia/Kolkata")


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


def main() -> int:
    day = sys.argv[1] if len(sys.argv) > 1 else datetime.now(IST).date().isoformat()
    print(f"DID THE CHANGES ENGAGE?  session {day}\n")

    trades = _jsonl(STATE / "intraday_v83_paper_live_trades.jsonl")
    entries = [t for t in trades if t.get("event") == "PAPER_ENTRY" and str(t.get("session_date")) == day]
    exits = [t for t in trades if t.get("event") == "PAPER_EXIT" and str(t.get("session_date")) == day]
    decisions = [d for d in _jsonl(STATE / "intraday_v83_paper_live_validation_decisions.jsonl")
                 if str(d.get("timestamp", ""))[:10] == day]

    # 1. SELLER-FIRST — did it stop buying by default?
    print("1. SELLER-FIRST (seller by default, buy only on an efficient trend)")
    if not entries:
        print("   NO ENTRY TODAY — nothing to judge. Check for an all-day stand-aside.")
    for e in entries:
        s = str(e.get("strategy"))
        kind = "BUYER (pays theta)" if "DEBIT" in s else "SELLER (collects theta)"
        print(f"   {s:<26} {kind}")
    if entries and all("DEBIT" in str(e.get("strategy")) for e in entries):
        print("   -> still buying. If the tape was choppy (low efficiency) the switch did NOT fire.")

    # 2. IRON_FLY reachable? It had NEVER been built live before the _strat_map fix.
    print("\n2. IRON_FLY REACHABLE (was silently unreachable until 13162a2)")
    ever = Counter(str(t.get("strategy")) for t in trades if t.get("event") == "PAPER_ENTRY")
    print(f"   iron flies ever opened live: {ever.get('IRON_FLY', 0)}")
    blocked = [d for d in decisions if "FLY" in str(d.get("canonical_rejection_reason", "")).upper()
               or "FLY" in str(d.get("block_reason", "")).upper()]
    if blocked:
        print(f"   {len(blocked)} fly-related rejections today — read canonical_rejection_reason")
    if ever.get("IRON_FLY", 0) == 0:
        print("   -> STILL UNPROVEN. If a low-efficiency day selected the fly and no trade appeared,")
        print("      strike construction is failing (liquidity / credit floor / delta band).")

    # 3. ENTRY TIMING — did the wait arm, and did it fill or time out?
    print("\n3. ENTRY TIMING (wait for a retrace, timeout so a trend is never missed)")
    waits = [d for d in decisions if str(d.get("block_reason", "")).startswith("ENTRY_WAIT")]
    if not waits:
        print("   never armed — no directional debit signalled today (expected under seller-first)")
    else:
        for k, n in Counter(str(d.get("block_reason", "")).split(":")[0] for d in waits).most_common():
            print(f"   {n:>4}  {k}")
        if not any("FILLED" in str(d.get("block_reason", "")) or "TIMEOUT" in str(d.get("block_reason", ""))
                   for d in waits):
            print("   -> armed but never resolved. It should ALWAYS end in FILLED or TIMEOUT;")
            print("      if not, the wait is eating signals — set ENTRY_WAIT_ENABLED=0.")

    # 4. SIZING
    print("\n4. POSITION SIZE (baseline floor of 2 lots)")
    for e in entries:
        legs = e.get("legs") or []
        q = max((int(l.get("quantity") or 0) for l in legs), default=0)
        print(f"   {str(e.get('strategy')):<26} qty {q}  = {q // 65 if q else 0} lot(s)")
    if entries and all(max((int(l.get("quantity") or 0) for l in (e.get("legs") or [])), default=0) <= 65
                       for e in entries):
        print("   -> still 1 lot. The floor did not apply — check min_lots_per_trade in the run config.")

    # 5. EXIT SHADOW — accumulating the evidence to settle the exit question
    print("\n5. EXIT SHADOW BOOK (needs ~20 closed trades to rule)")
    rows = _jsonl(STATE / "exit_shadow.jsonl")
    print(f"   {len(rows)} closed trades recorded ({len([r for r in rows if r.get('session_date') == day])} today)")
    if exits and not [r for r in rows if r.get("session_date") == day]:
        print("   -> a trade closed today but NO shadow row was written. The recorder is not firing.")

    # 6. WEEKLY SELLER
    print("\n6. WEEKLY SELLER (0/9 deployed before today's two fixes)")
    try:
        p = json.loads((STATE / "weekly_ic_pending.json").read_text())
        print(f"   pending plan: {p.get('status')}  expiry {p.get('expiry')}  credit Rs {p.get('gross_credit', 0):,.0f}")
    except (OSError, ValueError):
        print("   no pending plan")
    gl = _jsonl(STATE / "weekly_ic_gate_log.jsonl")
    if gl:
        last = gl[-1]
        print(f"   last assessment: {last.get('decision')} ({last.get('n_passed')}/{last.get('n_gates')})"
              f"  blocked_by={last.get('blocked_by')}")
    else:
        print("   no gate log yet — first entry appears at the next Wednesday 10:15 assessment")

    print("\n   Anything reading '-> ' above is a change that did NOT engage. Those come first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
