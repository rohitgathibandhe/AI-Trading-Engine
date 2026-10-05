"""
Daily Market Study — the brain trains from the MARKET every day, not just its trades.

A professional options trader studies the tape daily whether or not they traded:
how did price behave, did it gap and did the gap fill, and — most important for a
premium seller — how did IMPLIED VOL / PREMIUM behave? The classic edge the operator
described: a sharp down day spikes put premium (fear bid); the next morning gaps up
and that premium CRUSHES (vol mean-reverts). A seller who knows this sells the rich
post-spike premium and lets the crush pay them.

This module distils each session into a structured DayStudy — market character, gap
behavior, and seller-perspective volatility/premium observations — and mines cross-day
patterns (e.g. "after a >1.5% down day, the next open gaps up and IV crushes N/M
times"). Run EOD, it accumulates a knowledge base the selector can lean on.

Deterministic, evidence-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DayStudy:
    date: str
    prev_close: float
    open: float
    high: float
    low: float
    close: float
    gap_pct: float = 0.0
    day_move_pct: float = 0.0
    range_pct: float = 0.0
    gap_type: str = "FLAT"          # GAP_UP / GAP_DOWN / FLAT
    day_type: str = "RANGE"         # TREND_UP / TREND_DOWN / RANGE / VOLATILE
    gap_filled: bool = False
    # premium / IV lens (filled when a premium proxy is available)
    atm_premium_open: float | None = None
    atm_premium_close: float | None = None
    iv_open: float | None = None
    iv_close: float | None = None
    premium_behavior: str = "UNKNOWN"   # CRUSHED / EXPANDED / STEADY
    seller_notes: list[str] = field(default_factory=list)


def study_day(
    date: str, prev_close: float, o: float, h: float, l: float, c: float,
    *, prev_day_move_pct: float | None = None,
    atm_premium_open: float | None = None, atm_premium_close: float | None = None,
    iv_open: float | None = None, iv_close: float | None = None,
) -> DayStudy:
    gap_pct = (o - prev_close) / prev_close * 100.0 if prev_close else 0.0
    day_move_pct = (c - o) / o * 100.0 if o else 0.0
    range_pct = (h - l) / o * 100.0 if o else 0.0

    gap_type = "GAP_UP" if gap_pct >= 0.3 else "GAP_DOWN" if gap_pct <= -0.3 else "FLAT"
    if abs(day_move_pct) >= 0.8 and abs(c - o) >= 0.6 * (h - l):
        day_type = "TREND_UP" if day_move_pct > 0 else "TREND_DOWN"
    elif range_pct >= 1.5:
        day_type = "VOLATILE"
    else:
        day_type = "RANGE"

    # gap fill: did price trade back to the prior close during the day?
    gap_filled = (l <= prev_close <= h)

    s = DayStudy(date=date, prev_close=prev_close, open=o, high=h, low=l, close=c,
                 gap_pct=round(gap_pct, 2), day_move_pct=round(day_move_pct, 2),
                 range_pct=round(range_pct, 2), gap_type=gap_type, day_type=day_type,
                 gap_filled=gap_filled,
                 atm_premium_open=atm_premium_open, atm_premium_close=atm_premium_close,
                 iv_open=iv_open, iv_close=iv_close)

    # ── premium / IV behaviour (seller's lens) ──────────────────────────────
    prem_o = iv_open if iv_open is not None else atm_premium_open
    prem_c = iv_close if iv_close is not None else atm_premium_close
    if prem_o is not None and prem_c is not None and prem_o > 0:
        chg = (prem_c - prem_o) / prem_o * 100.0
        if chg <= -12:
            s.premium_behavior = "CRUSHED"
        elif chg >= 12:
            s.premium_behavior = "EXPANDED"
        else:
            s.premium_behavior = "STEADY"

    # ── seller notes: the lessons a premium seller would write down ─────────
    if prev_day_move_pct is not None and prev_day_move_pct <= -1.5 and gap_type == "GAP_UP":
        s.seller_notes.append(
            "Gap-UP after a >1.5% down day: yesterday's fear-bid put premium is rich and "
            "prone to CRUSH — favour selling puts / bull-put on the open strength."
            + (f" (premium {s.premium_behavior.lower()} today)" if s.premium_behavior != 'UNKNOWN' else "")
        )
    if day_type == "VOLATILE":
        s.seller_notes.append("Volatile wide-range day — premium sellers get whipsawed; buyers/debit favoured.")
    if day_type in ("TREND_UP", "TREND_DOWN"):
        d = "down" if day_type == "TREND_DOWN" else "up"
        s.seller_notes.append(f"Directional {d}-trend — with-trend debit (buy) beat premium selling into the move.")
    if gap_type != "FLAT" and gap_filled:
        s.seller_notes.append(f"{gap_type} that FILLED — fade-the-gap / range behaviour; premium selling around the fill worked.")
    if s.premium_behavior == "CRUSHED":
        s.seller_notes.append("IV/premium CRUSHED intraday — a seller holding through the crush was paid; a buyer bled theta+vega.")
    elif s.premium_behavior == "EXPANDED":
        s.seller_notes.append("IV/premium EXPANDED — a naked seller was hurt; defined-risk or long-vol was the safer side.")
    return s


def mine_patterns(studies: list[DayStudy]) -> dict[str, Any]:
    """Mine cross-day seller-relevant statistics from a run of DayStudies."""
    n = len(studies)
    out: dict[str, Any] = {"days": n}
    if n < 5:
        return out

    # after a big down day, what does the next day do?
    after_down = []
    for i in range(1, n):
        if studies[i - 1].day_move_pct <= -1.5:
            after_down.append(studies[i])
    if after_down:
        gap_ups = sum(1 for d in after_down if d.gap_type == "GAP_UP")
        crushed = sum(1 for d in after_down if d.premium_behavior == "CRUSHED")
        out["after_big_down_day"] = {
            "n": len(after_down),
            "gap_up_rate": round(gap_ups / len(after_down), 2),
            "premium_crush_rate": round(crushed / len([d for d in after_down if d.premium_behavior != 'UNKNOWN']), 2) if any(d.premium_behavior != 'UNKNOWN' for d in after_down) else None,
            "lesson": "Sharp down day -> next open often gaps up and premium crushes -> sell the rich put premium.",
        }

    # gap fill frequency (fade-the-gap edge)
    gaps = [d for d in studies if d.gap_type != "FLAT"]
    if gaps:
        out["gap_fill_rate"] = round(sum(1 for d in gaps if d.gap_filled) / len(gaps), 2)

    # day-type distribution (what kind of market to expect)
    from collections import Counter
    out["day_type_mix"] = dict(Counter(d.day_type for d in studies))
    out["premium_behavior_mix"] = dict(Counter(d.premium_behavior for d in studies))
    return out


def load_prev_day_context(jsonl_path, today: str | None = None) -> dict:
    """Read the most recent completed DayStudy (before `today`) so the NEXT session's
    decisions can SEE yesterday's character. Advisory context — attaches to metadata,
    does NOT gate (the premium-crush lesson has too few sharp-down samples to gate on;
    it accumulates here for forward validation). Returns {} when unavailable.
    """
    import json
    import os

    try:
        if not os.path.exists(jsonl_path):
            return {}
        rows = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
        if today is not None:
            rows = [r for r in rows if str(r.get("date") or "") < str(today)]
        if not rows:
            return {}
        last = rows[-1]
        mv = last.get("day_move_pct")
        ctx = {
            "prev_day_date": last.get("date"),
            "prev_day_move_pct": mv,
            "prev_day_type": last.get("day_type"),
            "prev_day_gap_type": last.get("gap_type"),
            "prev_day_premium_behavior": last.get("premium_behavior"),
        }
        if isinstance(mv, (int, float)) and mv <= -1.5:
            ctx["prev_day_lesson"] = (
                "after a sharp down day: next-open premium tends rich and CRUSHES — "
                "favour selling / do not overpay to buy debits at the open"
            )
        elif isinstance(mv, (int, float)) and mv >= 1.5:
            ctx["prev_day_lesson"] = "after a sharp up day: watch for gap-and-fade / mean reversion"
        return ctx
    except Exception:
        return {}
