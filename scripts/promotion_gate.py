#!/usr/bin/env python
"""PROMOTION GATE — the rule that decides when a precondition→strategy pairing has EARNED the right
to graduate from SHADOW (paper-tested) to the LIVE driver. Replaces "promote when I judge it ready"
with an explicit, evidence-based rule, run against the forward record.

It reads the shadow book (state/shadow_book.jsonl — the daily {preconditions → per-structure real-fill
P&L} rows) and, for each structure, evaluates it ONLY on the days its PRECONDITION fired — using the
SAME precondition functions the matrix selects on. So we measure "does this strategy work WHEN its
preconditions hold," not its average over all days. A pairing graduates only when the forward record
clears every bar (enough qualifying days, positive edge, a win-rate floor, and a return that dwarfs
the worst single day). Until then it stays in shadow.

Outputs a human scorecard + a machine-readable state/promotion_state.json (the ELIGIBLE list the live
selector can consume once auto-promotion is wired). Safe to run daily; it decides nothing on its own.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data_engine"))
from market_ai.intraday_defined_risk import strategy_matrix as M  # noqa: E402

STATE = REPO / "data_engine" / "market_ai" / "state"
SHADOW = STATE / "shadow_book.jsonl"
OUT_STATE = STATE / "promotion_state.json"

# ── Promotion bars (conservative; tune as forward data accumulates) ──────────────────────────
MIN_DAYS = 20         # enough qualifying days to trust the sample (not a lucky handful)
MIN_AVG_PNL = 0.0     # positive average P&L per qualifying day
MIN_WIN_RATE = 55.0   # % of qualifying days green
MIN_RET_TAIL = 3.0    # total edge >= 3x the worst single qualifying day (survives a bad day)

# ── Drift demotion: auto-revert a PROMOTED pairing to shadow when its RECENT edge decays ──────
# A strategy that earned its way to live can stop working when the market regime shifts. Watch the
# most recent qualifying days; if the recent edge turns negative or the recent win-rate collapses,
# demote it back to shadow (remove from eligible) so it must re-earn promotion. This is the
# "re-evaluate on drift" half of the loop — the gate promotes AND demotes, by rule.
DRIFT_WINDOW = 10     # look at the most recent N qualifying days
DRIFT_MIN_DAYS = 5    # need at least this many recent days to call drift
DRIFT_AVG_FLOOR = 0.0 # recent avg P&L must stay above this
DRIFT_WIN_FLOOR = 40.0  # recent win-rate floor (%)

# matrix strategy name -> shadow_book structure key
KEY = {
    "PUT_DEBIT_SPREAD": "put_debit", "CALL_DEBIT_SPREAD": "call_debit",
    "BULL_PUT_CREDIT_SPREAD": "bull_put", "BEAR_CALL_CREDIT_SPREAD": "bear_call",
    "IRON_CONDOR": "iron_condor", "IRON_FLY": "iron_fly",
    "SHORT_STRANGLE": "short_strangle", "SHORT_STRADDLE": "short_straddle",
}


def _rows() -> list[dict]:
    if not SHADOW.exists():
        return []
    rows = []
    for line in SHADOW.read_text().splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _structure_pnl(row: dict, key: str):
    s = row.get("structures") or {}
    if key in s and s[key]:
        return s[key].get("pnl_rupees")
    if key in ("bull_put", "bear_call", "iron_condor") and row.get(key):  # legacy old-format rows
        return row[key].get("pnl_rupees")
    return None


def evaluate() -> list[dict]:
    rows = _rows()
    out = []
    for spec in M.SPECS:
        key = KEY.get(spec.name)
        dp = []                                          # (date, pnl) on qualifying days
        for row in rows:
            pc = row.get("preconditions") or {}
            if not pc:                                   # can't precondition-filter a legacy row
                continue
            ok, _, _ = spec.precondition(pc)
            if not ok:
                continue
            p = _structure_pnl(row, key)
            if p is not None:
                dp.append((str(row.get("date") or ""), float(p)))
        dp.sort(key=lambda x: x[0])                      # chronological — so the recent window is real
        pnls = [p for _, p in dp]
        n = len(pnls)
        rec = {"strategy": spec.name, "is_credit": spec.is_credit, "defined_risk": spec.defined_risk,
               "hold": spec.default_hold, "qualifying_days": n}
        if n == 0:
            rec.update(status="NO_QUALIFYING_DAYS", total=0, avg=None, win_rate=None, worst=None, ret_tail=None)
        else:
            total = sum(pnls); avg = total / n
            wr = 100.0 * sum(1 for x in pnls if x > 0) / n
            worst = min(pnls)
            ret_tail = (total / abs(worst)) if worst < 0 else float("inf")
            passed = (n >= MIN_DAYS and avg > MIN_AVG_PNL and wr >= MIN_WIN_RATE and ret_tail >= MIN_RET_TAIL)
            # DRIFT: a promoted strategy whose recent window has decayed gets demoted back to shadow.
            recent = pnls[-DRIFT_WINDOW:]
            drifted = False
            if passed and len(recent) >= DRIFT_MIN_DAYS:
                r_avg = sum(recent) / len(recent)
                r_wr = 100.0 * sum(1 for x in recent if x > 0) / len(recent)
                if r_avg <= DRIFT_AVG_FLOOR or r_wr < DRIFT_WIN_FLOOR:
                    drifted = True
            if drifted:
                status = "DEMOTED_DRIFT"                  # earned promotion but recent edge decayed
                passed = False
            elif passed:
                status = "ELIGIBLE_FOR_LIVE"
            elif n < MIN_DAYS:
                status = f"NEED_MORE_DATA ({n}/{MIN_DAYS} days)"
            else:
                status = "SHADOW_FAIL"                    # enough data, but edge didn't clear the bar
            rec.update(status=status, total=round(total), avg=round(avg),
                       win_rate=round(wr, 1), worst=round(worst),
                       ret_tail=(round(ret_tail, 2) if ret_tail != float("inf") else None))
        out.append(rec)
    return out


def main() -> int:
    report = evaluate()
    print("PROMOTION GATE — shadow→live eligibility (judged on each strategy's qualifying days)")
    print(f"  bars: >= {MIN_DAYS} days, avg > {MIN_AVG_PNL}, win >= {MIN_WIN_RATE}%, ret/tail >= {MIN_RET_TAIL}\n")
    print(f"  {'strategy':<24}{'days':>5}{'total':>9}{'avg':>7}{'win%':>6}{'worst':>8}{'ret/tail':>9}  status")
    for r in sorted(report, key=lambda x: (x["status"] != "ELIGIBLE_FOR_LIVE", -(x["total"] or 0))):
        f = lambda v, w: (f"{v:>{w}}" if v is not None else " " * (w - 1) + "-")  # noqa: E731
        print(f"  {r['strategy']:<24}{r['qualifying_days']:>5}{f(r['total'],9)}{f(r['avg'],7)}"
              f"{f(r['win_rate'],6)}{f(r['worst'],8)}{f(r['ret_tail'],9)}  {r['status']}")
    eligible = [r["strategy"] for r in report if r["status"] == "ELIGIBLE_FOR_LIVE"]
    OUT_STATE.write_text(json.dumps({"eligible_for_live": eligible, "report": report}, indent=2))
    print(f"\n  ELIGIBLE for live: {eligible or 'none yet — keep accumulating shadow days'}")
    print(f"  -> {OUT_STATE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
