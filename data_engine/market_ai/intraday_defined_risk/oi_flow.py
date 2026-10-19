"""OI-FLOW tracker — who is defending / abandoning the strikes around price, over the last ~30 minutes.

Owner, 2026-10-19 expiry: the 12:08 breakdown put-debit exited +84 on a 25-pt bounce (12:37) while the chain said the
move was intact — 23,000 put OI had fallen 29.8L -> 15.1L (writers abandoning support) and call OI at 23,000 / 22,950 /
22,900 kept RISING (writers moving the ceiling down). Nifty fell another ~90 pts to the next put wall (22,850-22,800).

  update(chain)                  record OI per strike near ATM (call each cycle; keyed by chain timestamp)
  wall_flip(spot)                'BEARISH' | 'BULLISH' | None — at a strike just above/below spot, one side's writers
                                 leaving while the other side's pile in (the level is changing hands)
  supports(direction, spot)      True while the flow still backs a LONG ('BULLISH') / SHORT ('BEARISH') position
  next_wall(direction, spot)     nearest opposite wall in the trade's direction (target), or None
Pure state in-process; empty/neutral when the chain carries no OI (synthetic replays)."""
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta

_HIST: deque = deque(maxlen=80)          # (ts, spot, {strike: (call_oi, put_oi)})
WINDOW = timedelta(minutes=30)
SHORT = timedelta(minutes=15)


def reset() -> None:
    _HIST.clear()


def update(chain) -> None:
    try:
        ts = chain.timestamp.replace(tzinfo=None)
        spot = float(chain.spot)
    except Exception:
        return
    if _HIST and (_HIST[-1][0] == ts or _HIST[-1][0].date() != ts.date()):
        if _HIST[-1][0] == ts:
            return
        _HIST.clear()                      # new session
    book: dict[float, list[int]] = {}
    for q in chain.quotes:
        if q.oi is None or abs(q.strike - spot) > 400:
            continue
        b = book.setdefault(float(q.strike), [0, 0])
        if str(getattr(q.option_type, "value", q.option_type)).upper().startswith("C"):
            b[0] = int(q.oi)
        else:
            b[1] = int(q.oi)
    if sum(c + p for c, p in book.values()) > 0:
        _HIST.append((ts, spot, {k: tuple(v) for k, v in book.items()}))


def _then(delta: timedelta):
    if len(_HIST) < 2:
        return None, None
    now = _HIST[-1]
    old = next((h for h in _HIST if now[0] - h[0] <= delta), None)
    return (old, now) if old is not None and old is not now else (None, None)


def _chg(old, now, k, side):
    a = old[2].get(k, (0, 0))[side]
    b = now[2].get(k, (0, 0))[side]
    return b - a, (b - a) / a if a > 0 else 0.0


def wall_flip(spot: float, min_drop: float = 0.15, min_rise: float = 0.30) -> str | None:
    old, now = _then(WINDOW)
    if old is None:
        return None
    base = round(spot / 50) * 50
    for k in (base - 50, base, base + 50):
        _, put_pct = _chg(old, now, k, 1)
        _, call_pct = _chg(old, now, k, 0)
        # support breaking: price at/below k, put writers leaving k while call writers arrive at k
        if k >= spot - 25 and put_pct <= -min_drop and call_pct >= min_rise:
            return "BEARISH"
        # resistance breaking: price at/above k, call writers leaving while put writers arrive
        if k <= spot + 25 and call_pct <= -min_drop and put_pct >= min_rise:
            return "BULLISH"
    return None


def supports(direction: str, spot: float) -> bool | None:
    """Over the last 15 min: SHORT is backed while net call OI ADDED within 100 pts above price exceeds net put OI added
    within 100 pts below (writers leaning on the ceiling); LONG is the mirror. None if no OI history."""
    old, now = _then(SHORT)
    if old is None:
        return None
    calls = sum(_chg(old, now, k, 0)[0] for k in now[2] if spot <= k <= spot + 100)
    puts = sum(_chg(old, now, k, 1)[0] for k in now[2] if spot - 100 <= k <= spot)
    if direction == "BEARISH":
        return calls > puts and calls > 0
    return puts > calls and puts > 0


def next_wall(direction: str, spot: float, min_gap: float = 40.0) -> float | None:
    """Biggest opposite-side OI strike beyond price in the trade's direction (put wall below for a short)."""
    if not _HIST:
        return None
    book = _HIST[-1][2]
    if direction == "BEARISH":
        cands = [(v[1], k) for k, v in book.items() if k <= spot - min_gap]
    else:
        cands = [(v[0], k) for k, v in book.items() if k >= spot + min_gap]
    return max(cands)[1] if cands else None
