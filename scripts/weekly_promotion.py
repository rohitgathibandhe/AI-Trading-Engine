#!/usr/bin/env python
"""WEEKLY POSITIONAL — forward-record / go-live gate.

The intraday promotion gate reads the shadow book and never sees the weekly book. This tracks the
weekly positional book's OWN realized record and judges honestly whether it has earned real money —
so 'can I go live?' becomes a number, not a feeling, and it ACCUMULATES with every closed trade.

The bar is deliberately hard, and includes one thing a hot streak can't fake: the DEFENSE must have
been exercised. A book that only ever won in easy trending weeks has NOT proven its adjustment ladder,
so it can't graduate on wins alone — it must have survived tested/adjusted weeks too.

  python scripts/weekly_promotion.py            # print + write weekly_promotion_state.json
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path

STATE = Path(__file__).resolve().parents[1] / "data_engine" / "market_ai" / "state"
LEDGER = STATE / "weekly_positional_paper.jsonl"
OUT = STATE / "weekly_promotion_state.json"

# ── The go-live bar ───────────────────────────────────────────────────────────────────────────
MIN_TRADES   = 15      # enough closed weeklies to trust the record (not a lucky handful)
MIN_AVG      = 0.0     # positive average realized P&L per trade
MIN_WIN_RATE = 60.0    # % of trades green (defined-risk selling should win often)
MIN_RET_TAIL = 3.0     # total edge >= 3x the worst single loss (survives a bad week)
MIN_TESTED   = 3       # trades where the DEFENSE fired (adjusted/tested) — proves the ladder, not just easy wins


def _rows() -> list[dict]:
    return [json.loads(l) for l in LEDGER.read_text().splitlines() if l.strip()] if LEDGER.exists() else []


def evaluate() -> dict:
    rows = _rows()
    # Which entries were ever adjusted (the defense fired)?
    adjusted_keys = {r.get("entry_key") for r in rows if r.get("event") == "PAPER_ADJUST"}
    exits = [r for r in rows if r.get("event") == "PAPER_EXIT"]
    pnls = [float(r.get("realized_rupees") or 0) for r in exits]
    tested = sum(1 for r in exits if r.get("entry_key") in adjusted_keys
                 or r.get("action") in ("STOP_CLOSE", "CLOSE_THESIS_BROKEN", "CLOSE_MAX_ADJUSTED", "ROLL_OUT", "LEG_INTO_CONDOR"))
    n = len(pnls)
    rec: dict = {"n_trades": n, "tested_trades": tested, "updated": datetime.now().isoformat(timespec="seconds")}
    if n == 0:
        rec.update(status="NO_TRADES", eligible=False, total=0, avg=None, win_rate=None,
                   worst=None, ret_tail=None, bar={"min_trades": MIN_TRADES})
        return rec
    total = sum(pnls)
    avg = total / n
    wr = 100.0 * sum(1 for x in pnls if x > 0) / n
    worst = min(pnls)
    ret_tail = (total / abs(worst)) if worst < 0 else float("inf")
    checks = {
        "enough_trades": n >= MIN_TRADES,
        "positive_avg": avg > MIN_AVG,
        "win_rate": wr >= MIN_WIN_RATE,
        "tail_bounded": ret_tail >= MIN_RET_TAIL,
        "defense_proven": tested >= MIN_TESTED,
    }
    eligible = all(checks.values())
    missing = [k for k, v in checks.items() if not v]
    rec.update(total=round(total), avg=round(avg), win_rate=round(wr, 1),
               worst=round(worst), ret_tail=(round(ret_tail, 2) if ret_tail != float("inf") else None),
               checks=checks, eligible=eligible,
               status=("ELIGIBLE_FOR_LIVE" if eligible else "NOT_READY: " + ", ".join(missing)))
    return rec


def main() -> int:
    rec = evaluate()
    OUT.write_text(json.dumps(rec, indent=2))
    print("WEEKLY POSITIONAL — go-live gate (judged on the book's own realized record)\n")
    print(f"  trades {rec['n_trades']} (defense fired on {rec['tested_trades']})  |  "
          f"total Rs {rec.get('total'):+,}  avg Rs {str(rec.get('avg'))}  win {rec.get('win_rate')}%  "
          f"worst Rs {str(rec.get('worst'))}  ret/tail {rec.get('ret_tail')}")
    print(f"  BAR: >= {MIN_TRADES} trades, avg > 0, win >= {MIN_WIN_RATE}%, ret/tail >= {MIN_RET_TAIL}, "
          f">= {MIN_TESTED} tested (defense proven)")
    for k, v in (rec.get("checks") or {}).items():
        print(f"    [{'PASS' if v else 'FAIL'}]  {k}")
    print(f"\n  => {rec['status']}")
    print("  Judged on the forward paper-on-live record. Wins in easy trending weeks alone do NOT graduate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
