#!/usr/bin/env python
"""STRATEGY SCORECARD — see WHY the agent picks a structure, and its FORWARD record.

Runs the desk-brain matrix against the latest market read and shows, best-first: each eligible
structure's fit score, its forward-record tilt (from promotion_state), the blended score, and the
reason. Writes state/strategy_scorecard.json for the dashboard. This is the visibility layer over the
autonomy loop: selection = how well it FITS the tape + how it has PERFORMED forward.

  python scripts/strategy_scorecard.py            # print + write the snapshot
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_engine"))
sys.path.insert(0, str(ROOT / "scripts"))

from market_ai.intraday_defined_risk import strategy_matrix as sm
import weekly_positional_paper as wp

STATE = ROOT / "data_engine" / "market_ai" / "state"


def main() -> int:
    m = wp._latest_metadata()
    ranked = sm.evaluate_matrix(m)
    report = sm._forward_report()
    pick = sm.select_from_matrix(m)

    rows = []
    for r in ranked:
        rows.append({"strategy": r["strategy"], "family": r["family"], "hold": r["hold"],
                     "fit_score": r.get("fit_score"), "forward_tilt": r.get("forward_tilt"),
                     "score": r["score"], "why": r["why"], "picked": r["strategy"] == pick.get("strategy")})

    forward = []
    for s, rr in report.items():
        forward.append({"strategy": s, "qualifying_days": rr.get("qualifying_days"),
                        "avg": rr.get("avg"), "win_rate": rr.get("win_rate"), "worst": rr.get("worst"),
                        "status": rr.get("status")})

    snap = {"updated": datetime.now().isoformat(timespec="seconds"),
            "condition": m.get("selector_condition"), "bias": sm._bias(m),
            "picked": pick.get("strategy"), "picked_why": pick.get("why"),
            "ranked": rows, "forward": forward}
    (STATE / "strategy_scorecard.json").write_text(json.dumps(snap, indent=2))

    print(f"bias {snap['bias']} | condition {snap['condition']} | PICK -> {snap['picked']}")
    print(f"{'structure':26}{'fit':>7}{'fwd':>7}{'score':>8}  why")
    for r in rows:
        star = " *" if r["picked"] else "  "
        print(f"{star}{r['strategy']:24}{r['fit_score']:>7.2f}{(r['forward_tilt'] or 0):>7.2f}{r['score']:>8.2f}  {r['why'][:70]}")
    print("\nFORWARD record (drives the tilt as it grows to 20 days):")
    for f in sorted(forward, key=lambda x: -(x.get('avg') or -1e9)):
        print(f"  {f['strategy']:24} {int(f.get('qualifying_days') or 0):>2}d  avg {str(f.get('avg')):>7}  win {str(f.get('win_rate')):>5}  {f.get('status')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
