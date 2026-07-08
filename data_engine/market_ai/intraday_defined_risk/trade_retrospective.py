"""
Trade Retrospective — introspect + retrospect every trade to improve.

This is the learning loop the top desks run: every closed trade gets a post-mortem
(did the thesis play out? was the exit good or did we give it back? was the loss
controlled?), and the post-mortems are mined into CONCRETE rule adjustments that
feed back into strategy selection and entry/exit timing.

It is deterministic and evidence-only — it never invents a story, it reads what the
price actually did between entry and exit and grades the trade against its own plan.

Verdicts per trade:
  WIN_AS_PLANNED   — thesis played out (price moved the trade's way), booked a win
  WIN_GIVEBACK     — won, but gave back a large share of peak profit (exit too late/loose)
  LOSS_CONTROLLED  — thesis failed but the loss was cut small (good risk control)
  LOSS_RAN         — thesis failed and the loss was large (stop too loose / strike too close)
  LOSS_REVERSAL    — entered with a move that then reversed against us (entry-timing miss)
  SCRATCH_MISMARK  — booked ~0 with no real move (data/marking artifact — exclude from learning)

The aggregate report groups verdicts by condition, strategy, direction, and
time-of-day, and emits ranked, actionable recommendations.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TradeVerdict:
    verdict: str
    lesson: str
    spot_move: float | None = None
    with_thesis: bool | None = None


def _is_bearish(strategy: str) -> bool:
    return "BEAR_CALL" in strategy or "PUT_DEBIT" in strategy


def _is_bullish(strategy: str) -> bool:
    return "BULL_PUT" in strategy or "CALL_DEBIT" in strategy


def grade_trade(trade: dict[str, Any], entry_spot: float | None, exit_spot: float | None) -> TradeVerdict:
    """Grade one trade against what the price actually did."""
    pnl = float(trade.get("pnl_rupees") or 0.0)
    strategy = str(trade.get("strategy") or "")
    exit_reason = str(trade.get("exit_reason") or "")

    move = None if (entry_spot is None or exit_spot is None) else (exit_spot - entry_spot)

    # with-thesis = did spot move the direction the trade wanted?
    with_thesis = None
    if move is not None:
        if _is_bearish(strategy):
            with_thesis = move < 0
        elif _is_bullish(strategy):
            with_thesis = move > 0
        else:  # range (condor/strangle): wanted small move
            with_thesis = abs(move) < 60

    # mismark / scratch
    if abs(pnl) < 1.0 and (move is None or abs(move) < 15):
        return TradeVerdict("SCRATCH_MISMARK", "No real move + ~0 P&L — likely a marking/data artifact; exclude from learning.", move, with_thesis)

    if pnl > 0:
        # winner — check if it was clean or gave back
        mfe = float(trade.get("mfe_rupees") or 0.0)
        if mfe > 0 and pnl < 0.5 * mfe:
            return TradeVerdict("WIN_GIVEBACK", "Won but kept <50% of peak — exit is too loose/late; tighten the profit trail.", move, with_thesis)
        return TradeVerdict("WIN_AS_PLANNED", "Thesis played out and was captured — the pattern to reinforce.", move, with_thesis)

    # loser
    if with_thesis is False:
        # entered, price went the other way
        if abs(pnl) <= 1500:
            return TradeVerdict("LOSS_CONTROLLED", "Direction wrong but loss cut small — good risk control; the miss is ENTRY timing.", move, with_thesis)
        return TradeVerdict("LOSS_RAN", "Direction wrong AND loss large — strike too close / stop too loose; widen strikes or tighten stop.", move, with_thesis)
    if with_thesis is True:
        # price moved our way but we still lost — strike/structure/exit problem
        return TradeVerdict("LOSS_REVERSAL", "Price ended our way yet we lost — intraday whipsaw stopped us out, or strikes/mark wrong. Review exit + entry timing.", move, with_thesis)
    return TradeVerdict("LOSS_CONTROLLED" if abs(pnl) <= 1500 else "LOSS_RAN", "Loss (no spot reference).", move, with_thesis)


def _tod_bucket(ts: str) -> str:
    hhmm = str(ts)[11:16]
    if hhmm < "10:00":
        return "OPEN(9:15-10)"
    if hhmm < "12:00":
        return "MORN(10-12)"
    if hhmm < "13:30":
        return "MID(12-13:30)"
    return "LATE(13:30-15:15)"


def retrospect(trades: list[dict[str, Any]], spot_by_min: dict[str, float]) -> dict[str, Any]:
    """Grade every trade and mine actionable patterns."""
    def spot(ts): return spot_by_min.get(str(ts)[:16])

    graded = []
    for t in trades:
        v = grade_trade(t, spot(t.get("entry_timestamp")), spot(t.get("timestamp")))
        graded.append((t, v))

    verdicts = Counter(v.verdict for _, v in graded)

    # bucket P&L + win rate by (strategy, condition, time-of-day)
    def bucket_stats(keyfn):
        agg = defaultdict(lambda: {"n": 0, "pnl": 0.0, "wins": 0})
        for t, v in graded:
            if v.verdict == "SCRATCH_MISMARK":
                continue
            k = keyfn(t)
            a = agg[k]
            a["n"] += 1
            a["pnl"] += float(t.get("pnl_rupees") or 0.0)
            a["wins"] += 1 if float(t.get("pnl_rupees") or 0.0) > 0 else 0
        return {k: {**a, "win_rate": round(a["wins"] / a["n"], 2) if a["n"] else 0.0} for k, a in agg.items()}

    by_strategy = bucket_stats(lambda t: str(t.get("strategy") or "?"))
    by_state = bucket_stats(lambda t: str(t.get("market_state") or "?"))
    by_tod = bucket_stats(lambda t: _tod_bucket(t.get("entry_timestamp") or ""))
    by_strategy_tod = bucket_stats(lambda t: f"{str(t.get('strategy') or '?')[:10]} @ {_tod_bucket(t.get('entry_timestamp') or '')}")

    # ── mine recommendations ────────────────────────────────────────────────
    recs = []
    # losing buckets with enough sample
    for k, a in sorted(by_strategy_tod.items(), key=lambda x: x[1]["pnl"]):
        if a["n"] >= 4 and a["pnl"] < 0:
            recs.append(f"CUT/FIX: '{k}' — {a['n']} trades, Rs{a['pnl']:.0f}, win {a['win_rate']:.0%}. Negative expectancy; restrict this condition/time or fix entry.")
    # WIN_GIVEBACK → tighten trail
    if verdicts.get("WIN_GIVEBACK", 0) >= 3:
        recs.append(f"TIGHTEN EXIT: {verdicts['WIN_GIVEBACK']} winners gave back >50% of peak — tighten the profit trail / add a partial-take.")
    # LOSS_RAN → strikes/stop
    if verdicts.get("LOSS_RAN", 0) >= 3:
        recs.append(f"WIDEN STRIKES / TIGHTEN STOP: {verdicts['LOSS_RAN']} losers ran large — sell further OTM (lower delta) or cut sooner.")
    # LOSS_REVERSAL → entry timing
    if verdicts.get("LOSS_REVERSAL", 0) >= 3:
        recs.append(f"ENTRY TIMING: {verdicts['LOSS_REVERSAL']} trades were right-direction-by-close but lost — improve entry trigger (wait for pattern confirmation) to avoid whipsaw stops.")
    # scratch marks
    if verdicts.get("SCRATCH_MISMARK", 0) >= 3:
        recs.append(f"DATA/MARKING: {verdicts['SCRATCH_MISMARK']} trades booked ~0 with no move — fix mid-day repricing (intrinsic-value fallback) so trades are marked & managed.")

    return {
        "n_trades": len(trades),
        "verdicts": dict(verdicts),
        "by_strategy": by_strategy,
        "by_market_state": by_state,
        "by_time_of_day": by_tod,
        "by_strategy_time": by_strategy_tod,
        "recommendations": recs,
        "graded": [(str(t.get("entry_timestamp"))[:16], str(t.get("strategy"))[:12], v.verdict,
                    round(v.spot_move) if v.spot_move is not None else None,
                    round(float(t.get("pnl_rupees") or 0))) for t, v in graded],
    }
