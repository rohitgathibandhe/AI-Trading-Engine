#!/usr/bin/env python
"""Replay the HIGH-PROBABILITY CONFLUENCE GATE across the shadow-book history and score it in REAL
option P&L (the shadow book's actual per-structure fills).

For each shadow-book day we rebuild the decision metadata from the recorded preconditions, run the
live gate (strategy_selector._high_prob_confluence), map its verdict to a structure, and look up that
structure's actual fill:

  BULLISH -> bull_put     BEARISH -> bear_call     None -> NO TRADE (0)

Answers the two questions the desk asked:
  1. Does the gate's selectivity actually make money (vs always-trading, vs hindsight-best)?
  2. WHY does it stand aside so often — which confluence condition is the binding constraint?

  python scripts/confluence_gate_backtest.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data_engine"))

from market_ai.intraday_defined_risk import strategy_selector as ss
from market_ai.intraday_defined_risk.strategy_matrix import _bias, _ema20_bias, _m_pattern_at_resistance, _w_pattern_at_support

STATE = ROOT / "data_engine" / "market_ai" / "state"
SHADOW = STATE / "shadow_book.jsonl"


class _Read:
    """Minimal stand-in for the live MarketRead — the gate only reads .bias."""
    def __init__(self, bias: str):
        self.bias = bias


def _pnl(structs: dict, key: str):
    v = (structs or {}).get(key)
    return float(v["pnl_rupees"]) if isinstance(v, dict) and v.get("pnl_rupees") is not None else None


def _blocking_reason(m: dict, read) -> str:
    """When the gate stands aside, which condition failed FIRST — so we know what to tune."""
    eff = ss._trend_efficiency(m)
    if eff < ss._CONF_MIN_EFF:
        return f"efficiency<{ss._CONF_MIN_EFF}"
    ema = _ema20_bias(m)
    pa_bear = (bool(m.get("accepted_breakdown")) or _m_pattern_at_resistance(m)) and ema in ("BEARISH", "NEUTRAL")
    pa_bull = (bool(m.get("accepted_breakout")) or _w_pattern_at_support(m)) and ema in ("BULLISH", "NEUTRAL")
    if not (pa_bull or pa_bear):
        return "no_price_action_setup"
    if pa_bear and read.bias != "BEARISH":
        return "bias_not_bearish"
    if pa_bull and read.bias != "BULLISH":
        return "bias_not_bullish"
    if pa_bear and not ss._chain_confirms(m, "BEARISH"):
        return "chain_not_bearish"
    if pa_bull and not ss._chain_confirms(m, "BULLISH"):
        return "chain_not_bullish"
    return "other"


def main() -> int:
    rows = [json.loads(l) for l in SHADOW.read_text().splitlines() if l.strip()]
    traded, blocked = [], Counter()
    n_days = 0
    for r in rows:
        m = dict(r.get("preconditions") or {})
        if not m:
            continue
        n_days += 1
        read = _Read(_bias(m))
        direction, _why = ss._high_prob_confluence(read, m)
        if direction is None:
            blocked[_blocking_reason(m, read)] += 1
            continue
        key = "bull_put" if direction == "BULLISH" else "bear_call"
        p = _pnl(r.get("structures") or {}, key)
        traded.append((str(r.get("date")), direction, key, p))

    def stats(pnls):
        pnls = [x for x in pnls if x is not None]
        if not pnls:
            return "n=0"
        w = sum(1 for x in pnls if x > 0)
        return f"n={len(pnls)}  total Rs {sum(pnls):+,.0f}  avg {sum(pnls)/len(pnls):+,.0f}  win {100*w/len(pnls):.0f}%"

    # Baselines over the SAME day set, for honest comparison
    all_bp = [_pnl(r.get("structures") or {}, "bull_put") for r in rows if r.get("preconditions")]
    all_bc = [_pnl(r.get("structures") or {}, "bear_call") for r in rows if r.get("preconditions")]

    print("HIGH-PROBABILITY CONFLUENCE GATE — replayed on the shadow book\n")
    print(f"  days with preconditions : {n_days}")
    print(f"  gate TRADED             : {len([t for t in traded if t[3] is not None])} days "
          f"({100*len(traded)/max(n_days,1):.0f}% of days)")
    print(f"  gate STOOD ASIDE        : {sum(blocked.values())} days\n")
    print(f"  GATE trades (real P&L)  : {stats([p for _,_,_,p in traded])}")
    print(f"  baseline always bull_put: {stats(all_bp)}")
    print(f"  baseline always bear_call:{stats(all_bc)}\n")
    print("  WHY it stood aside (binding constraint):")
    for reason, n in blocked.most_common():
        print(f"    {reason:24} {n:4}  ({100*n/max(sum(blocked.values()),1):.0f}%)")
    print("\n  gate trades taken:")
    for day, d, k, p in traded:
        print(f"    {day}  {d:8} -> {k:10} Rs {'n/a' if p is None else format(p,'+,.0f')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
