from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable

NIFTY_LEGACY_WEEKLY_EXPIRY_WEEKDAY = 3  # Thursday
NIFTY_CURRENT_WEEKLY_EXPIRY_WEEKDAY = 1  # Tuesday
# NSE changed NIFTY index derivative expiries to Tuesday for contracts
# expiring on or after 2025-09-01. The first Tuesday weekly expiry in that
# cycle is 2025-09-02.
NIFTY_TUESDAY_EXPIRY_EFFECTIVE_DATE = date(2025, 9, 1)
NIFTY_FIRST_TUESDAY_WEEKLY_EXPIRY = date(2025, 9, 2)

__all__ = [
    "NIFTY_LEGACY_WEEKLY_EXPIRY_WEEKDAY",
    "NIFTY_CURRENT_WEEKLY_EXPIRY_WEEKDAY",
    "NIFTY_TUESDAY_EXPIRY_EFFECTIVE_DATE",
    "NIFTY_FIRST_TUESDAY_WEEKLY_EXPIRY",
    "align_to_previous_trading_day",
    "coerce_to_date",
    "infer_nifty_weekly_expiry",
    "next_weekday_on_or_after",
]


def coerce_to_date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    try:
        text = str(value or "").strip()
        if not text:
            return None
        return date.fromisoformat(text[:10])
    except Exception:
        return None


def next_weekday_on_or_after(anchor: date, weekday: int) -> date:
    days_ahead = (weekday - anchor.weekday()) % 7
    return anchor + timedelta(days=days_ahead)


def align_to_previous_trading_day(target: date, trading_days: Iterable[date] | None = None) -> date:
    if trading_days is None:
        return target
    normalized = {day for day in trading_days if isinstance(day, date)}
    if not normalized or target in normalized:
        return target
    cursor = target
    for _ in range(7):
        cursor -= timedelta(days=1)
        if cursor in normalized:
            return cursor
    return target


def infer_nifty_weekly_expiry(session_date: object, *, trading_days: Iterable[date] | None = None) -> date | None:
    current = coerce_to_date(session_date)
    if current is None:
        return None

    legacy_expiry = next_weekday_on_or_after(current, NIFTY_LEGACY_WEEKLY_EXPIRY_WEEKDAY)
    if legacy_expiry < NIFTY_TUESDAY_EXPIRY_EFFECTIVE_DATE:
        target = legacy_expiry
    else:
        target = next_weekday_on_or_after(current, NIFTY_CURRENT_WEEKLY_EXPIRY_WEEKDAY)
        if target < NIFTY_FIRST_TUESDAY_WEEKLY_EXPIRY:
            target = NIFTY_FIRST_TUESDAY_WEEKLY_EXPIRY
    return align_to_previous_trading_day(target, trading_days)
