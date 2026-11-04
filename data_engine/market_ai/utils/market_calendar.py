"""NSE market calendar: trading days, holiday-adjusted weekly/monthly expiries, special sessions and scheduled events.

2026-11-05 audit: the agent hard-coded "expiry = Tuesday" (regime is_expiry_day / pin risk) while Tue 2026-11-10 and
Tue 2026-11-24 are NSE holidays (weekly expiry -> Mon 11-09, MONTHLY expiry -> Mon 11-23); the Sun 2026-11-08 Muhurat
session would become Monday's "previous day"; and no event (RBI policy, Budget) was known to any decision.
Extra years / events can be added in state/market_calendar.json without code changes:
  {"holidays": ["2027-01-26", ...], "special_sessions": ["2027-..."], "events": [{"date": ..., "time": "10:00", ...}]}
"""
from __future__ import annotations

import json
from datetime import date, datetime, time, timedelta
from pathlib import Path

from .nifty_expiry_calendar import NIFTY_TUESDAY_EXPIRY_EFFECTIVE_DATE

# NSE equity/F&O trading holidays falling on weekdays (source: NSE 2026 circular via zerodha.com/marketintel/holiday-calendar)
NSE_HOLIDAYS = {
    date(2026, 1, 15), date(2026, 1, 26), date(2026, 3, 3), date(2026, 3, 26), date(2026, 3, 31), date(2026, 4, 3),
    date(2026, 4, 14), date(2026, 5, 1), date(2026, 5, 28), date(2026, 6, 26), date(2026, 9, 14), date(2026, 10, 2),
    date(2026, 10, 20), date(2026, 11, 10), date(2026, 11, 24), date(2026, 12, 25),
}
# short ceremonial sessions that must never become a "previous day" (Muhurat ~1 hour on Diwali)
SPECIAL_SESSIONS = {date(2026, 11, 8)}
# scheduled high-impact events; time is IST; block = [time - pre, time + post] minutes for NEW entries
EVENTS = [
    {"date": "2026-12-04", "time": "10:00", "label": "RBI MPC policy", "severity": "high", "pre": 15, "post": 45},
    {"date": "2027-02-01", "time": "11:00", "label": "Union Budget", "severity": "high", "pre": 105, "post": 150},
    {"date": "2027-02-05", "time": "10:00", "label": "RBI MPC policy", "severity": "high", "pre": 15, "post": 45},
]

_STATE = Path(__file__).resolve().parents[1] / "state" / "market_calendar.json"
_extra: dict | None = None


def _load_extra() -> dict:
    global _extra
    if _extra is None:
        try:
            _extra = json.loads(_STATE.read_text()) if _STATE.exists() else {}
        except Exception:  # noqa: BLE001
            _extra = {}
    return _extra


def _holidays() -> set[date]:
    extra = {date.fromisoformat(d) for d in (_load_extra().get("holidays") or [])}
    return NSE_HOLIDAYS | extra


def special_sessions() -> set[date]:
    return SPECIAL_SESSIONS | {date.fromisoformat(d) for d in (_load_extra().get("special_sessions") or [])}


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in _holidays()


def previous_trading_day(d: date) -> date:
    c = d - timedelta(days=1)
    while not is_trading_day(c):
        c -= timedelta(days=1)
    return c


def next_trading_day(d: date) -> date:
    c = d + timedelta(days=1)
    while not is_trading_day(c):
        c += timedelta(days=1)
    return c


def weekly_expiry(d: date) -> date:
    """The weekly expiry of the series trading on d: Tuesday on/after d, moved back over holidays."""
    wd = 1 if d >= NIFTY_TUESDAY_EXPIRY_EFFECTIVE_DATE else 3
    t = d + timedelta(days=(wd - d.weekday()) % 7)
    while not is_trading_day(t):
        t -= timedelta(days=1)
    if t < d:                                   # the shifted expiry fell before d -> next week's
        t = d + timedelta(days=(wd - d.weekday()) % 7 + 7)
        while not is_trading_day(t):
            t -= timedelta(days=1)
    return t


def is_expiry_day(d: date) -> bool:
    return is_trading_day(d) and weekly_expiry(d) == d


def monthly_expiry(d: date) -> date:
    """Last Tuesday (Thursday before 2025-09) of d's month, moved back over holidays."""
    from .index_calendar import monthly_expiry_date
    t = monthly_expiry_date(d)
    while not is_trading_day(t):
        t -= timedelta(days=1)
    return t


def events_on(d: date) -> list[dict]:
    rows = EVENTS + list(_load_extra().get("events") or [])
    return [e for e in rows if str(e.get("date")) == d.isoformat()]


def event_block(ts: datetime) -> dict | None:
    """The scheduled event whose no-new-entry window contains ts, if any."""
    for e in events_on(ts.date()):
        try:
            h, m = str(e.get("time") or "09:15").split(":")
            t0 = datetime.combine(ts.date(), time(int(h), int(m)))
        except ValueError:
            continue
        naive = ts.replace(tzinfo=None)
        if t0 - timedelta(minutes=int(e.get("pre", 15))) <= naive <= t0 + timedelta(minutes=int(e.get("post", 45))):
            return e
    return None
