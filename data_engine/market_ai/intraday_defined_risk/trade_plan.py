"""Trader playbook pieces decoded from 2026-10-19 (owner request): structure trail, next-wall target, wall-proximity
veto, failed-retest entry. Pure functions on completed 5m bars + the current chain; every rule is env-toggled in the
callers (execution.py / strategy_selector.py) and A/B tested on real chains + the 18-month replay before going live.

10-19 read: 23,000 put wall broke 11:30-11:35, the 11:40-11:50 retests failed at 23,010-23,016 (entry ~23,000, stop
above the retest), the grind made lower highs 22,979 -> 22,961 -> 22,941 -> 22,924 (trail above the last one), and
price stopped at the 22,850-22,800 put walls (target / no new short into it)."""
from __future__ import annotations

from datetime import timedelta


def closed_bars(snapshot):
    try:
        now = snapshot.option_chain.timestamp.replace(tzinfo=None)
        return [b for b in snapshot.nifty_5m.bars if b.timestamp.replace(tzinfo=None) + timedelta(minutes=5) <= now
                and b.timestamp.date() == now.date()]
    except Exception:
        return []


def swing_stop(bars, entry_time, short: bool) -> float | None:
    """Most recent confirmed 5m pivot formed AFTER entry: a lower high for a short (high above both neighbours), a
    higher low for a long. None until one exists."""
    after = [b for b in bars if b.timestamp.replace(tzinfo=None) >= entry_time.replace(tzinfo=None)]
    piv = None
    for i in range(1, len(after) - 1):
        a, b, c = after[i - 1], after[i], after[i + 1]
        if short and b.high >= a.high and b.high >= c.high:
            piv = b.high
        if not short and b.low <= a.low and b.low <= c.low:
            piv = b.low
    if piv is None:   # before the first post-entry pivot: the structure stop is the extreme of the 3 bars into entry
        pre = [b for b in bars if b.timestamp.replace(tzinfo=None) < entry_time.replace(tzinfo=None)][-3:]
        if pre:
            piv = max(b.high for b in pre) if short else min(b.low for b in pre)
    return piv


def _book(chain):
    calls, puts = {}, {}
    for q in chain.quotes:
        if q.oi is None:
            continue
        if str(getattr(q.option_type, "value", q.option_type)).upper().startswith("C"):
            calls[float(q.strike)] = int(q.oi)
        else:
            puts[float(q.strike)] = int(q.oi)
    return calls, puts


def next_wall(chain, short: bool, spot: float, min_gap: float = 40.0, span: float = 300.0) -> float | None:
    """The opposite-side OI wall in the trade's path: biggest PUT OI strike 40-300 pts below spot for a short, biggest
    CALL OI strike 40-300 above for a long — only if it is a real wall (>= 1.3x the median OI of that side nearby)."""
    calls, puts = _book(chain)
    book = puts if short else calls
    cands = {k: v for k, v in book.items() if (spot - span <= k <= spot - min_gap) if short} if short else \
            {k: v for k, v in book.items() if spot + min_gap <= k <= spot + span}
    if not cands:
        return None
    vals = sorted(book[k] for k in book if abs(k - spot) <= span)
    med = vals[len(vals) // 2] if vals else 0
    k, v = max(cands.items(), key=lambda kv: kv[1])
    return k if med and v >= 1.3 * med else None


def nearest_wall_distance(chain, short: bool, spot: float, span: float = 150.0) -> float | None:
    """Distance to the biggest opposite wall within `span` in the trade's direction (put wall below for a short)."""
    calls, puts = _book(chain)
    book = puts if short else calls
    cands = {k: v for k, v in book.items() if (spot - span <= k < spot) if short} if short else \
            {k: v for k, v in book.items() if spot < k <= spot + span}
    if not cands:
        return None
    k = max(cands.items(), key=lambda kv: kv[1])[0]
    return abs(spot - k)


def failed_retest(bars, short: bool, lookback: int = 20, near: float = 20.0) -> bool:
    """True when the latest bars show a broken level that was retested and rejected: level L = the extreme of the
    `lookback` bars before the last 6 (support low for a short); price closed beyond L, a later bar came back within
    `near` pts of L, and the last close is beyond L again."""
    if len(bars) < lookback + 6:
        return False
    base, recent = bars[-(lookback + 6):-6], bars[-6:]
    if short:
        L = min(b.low for b in base)
        broke = any(b.close < L - 3 for b in recent[:-1])
        retest = any(b.high >= L - near for b in recent[1:])
        return broke and retest and recent[-1].close < L - 3
    L = max(b.high for b in base)
    broke = any(b.close > L + 3 for b in recent[:-1])
    retest = any(b.low <= L + near for b in recent[1:])
    return broke and retest and recent[-1].close > L + 3
