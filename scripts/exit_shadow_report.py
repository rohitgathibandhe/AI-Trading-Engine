#!/usr/bin/env python
"""EXIT SHADOW REPORT — which exit rule is actually winning on the FORWARD live record?

Reads state/exit_shadow.jsonl (one row per closed trade, every candidate rule scored at the same
real live marks the agent traded on) and ranks the rules against the live baseline, RIDE_TO_CLOSE.

Deliberately reports the numbers that decide this, not just the total:
  * total / mean — the headline
  * WORST trade — a rule that wins on total by mutilating the tail shows up here
  * TAIL KEPT   — total P&L of the trades that were the biggest winners under RIDE_TO_CLOSE.
                  ~90% of this edge's net comes from a handful of home runs, so any rule that
                  scores well on total while shrinking TAIL KEPT is the known trap: it banks the
                  faders and decapitates the runners. Read this column before the total.
  * n_changed   — how often the rule actually did something different

A rule earns live promotion only by beating RIDE_TO_CLOSE on total AND holding the tail, over
enough trades. Until then the live path is unchanged. Safe to run any time; decides nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

STATE = Path(__file__).resolve().parents[1] / "data_engine" / "market_ai" / "state"
LEDGER = STATE / "exit_shadow.jsonl"
BASELINE = "RIDE_TO_CLOSE"
MIN_TRADES = 20          # don't call a winner on a handful — the mistake this whole file exists to prevent
TAIL_N = 3               # how many top RIDE_TO_CLOSE trades count as "the tail"


def rows() -> list[dict]:
    if not LEDGER.exists():
        return []
    out = []
    for line in LEDGER.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def main() -> int:
    data = rows()
    if not data:
        print("EXIT SHADOW — no closed trades recorded yet.")
        print(f"  ledger: {LEDGER}")
        print("  The recorder runs inside the live loop; a row appears after each trade closes.")
        return 0

    names: list[str] = []
    for r in data:
        for k in (r.get("candidates") or {}):
            if k not in names:
                names.append(k)

    # index of the tail trades, judged by the BASELINE (these are the ones that carry the net)
    base_pnls = [(i, (r.get("candidates", {}).get(BASELINE) or {}).get("pnl_rupees") or 0.0)
                 for i, r in enumerate(data)]
    tail_idx = {i for i, _ in sorted(base_pnls, key=lambda x: -x[1])[:TAIL_N]}

    print(f"EXIT SHADOW — {len(data)} closed trades on the forward live record")
    print(f"  baseline = {BASELINE} (what the agent does today)\n")
    print(f"  {'rule':<22}{'total':>10}{'mean':>9}{'worst':>9}{'tail kept':>11}{'n_chg':>7}{'vs base':>10}")

    base_total = sum(p for _, p in base_pnls)
    base_tail = sum(p for i, p in base_pnls if i in tail_idx)
    lines = []
    for name in names:
        pnls, changed = [], 0
        for r in data:
            c = (r.get("candidates") or {}).get(name)
            b = (r.get("candidates") or {}).get(BASELINE)
            if not c:
                continue
            pnls.append(float(c.get("pnl_rupees") or 0.0))
            if b and c.get("exit_at") != b.get("exit_at"):
                changed += 1
        if not pnls:
            continue
        total = sum(pnls)
        tail = sum(p for i, p in enumerate(pnls) if i in tail_idx)
        lines.append((name, total, total / len(pnls), min(pnls), tail, changed, total - base_total))

    for name, total, mean, worst, tail, changed, delta in sorted(lines, key=lambda x: -x[1]):
        flag = ""
        if name != BASELINE:
            if tail < base_tail * 0.85:
                flag = "  <-- TAIL DAMAGE"
            elif delta > 0:
                flag = "  <-- beats baseline"
        print(f"  {name:<22}{total:>+10,.0f}{mean:>+9,.0f}{worst:>+9,.0f}{tail:>+11,.0f}"
              f"{changed:>7}{delta:>+10,.0f}{flag}")

    print(f"\n  tail = total of the top {TAIL_N} trades by {BASELINE} (baseline tail {base_tail:+,.0f})")
    if len(data) < MIN_TRADES:
        print(f"  NOT ACTIONABLE YET: {len(data)}/{MIN_TRADES} trades. A fat-tailed edge cannot be")
        print("  judged on a small sample — that is exactly how a trail gets shipped and reverted.")
    else:
        best = max((l for l in lines if l[0] != BASELINE), key=lambda x: x[1], default=None)
        if best and best[1] > base_total and best[4] >= base_tail * 0.85:
            print(f"  CANDIDATE: {best[0]} beats baseline by {best[6]:+,.0f} and holds the tail.")
            print("  Next step is a live A/B, not a direct flip.")
        else:
            print(f"  No rule beats {BASELINE} while holding the tail. Live path stays as-is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
