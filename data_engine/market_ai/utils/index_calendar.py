"""Index derivatives calendar helpers (monthly expiries, entry dates)."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

__all__ = [
    "last_weekday_of_month",
    "monthly_expiry_date",
    "last_friday_before",
    "batman_entry_datetime",
]


def last_weekday_of_month(year: int, month: int, weekday: int) -> date:
    """Return the last occurrence of `weekday` (0=Mon) for the given month."""

    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    d = next_month - timedelta(days=1)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def monthly_expiry_date(target: date) -> date:
    """
    Return the index monthly expiry for `target`'s month.

    NIFTY monthly expiry = last Tuesday of the month for expiries on/after 2025-09-01 (the exchange
    moved index expiries Thu -> Tue), and the last THURSDAY before that. Always returning Tuesday gave
    the wrong date for every pre-Sep-2025 month (e.g. 2025-06-24 instead of 2025-06-26), which
    mis-dates historical/backtest expiries.
    """
    from .nifty_expiry_calendar import NIFTY_TUESDAY_EXPIRY_EFFECTIVE_DATE

    tuesday = last_weekday_of_month(target.year, target.month, weekday=1)
    if tuesday >= NIFTY_TUESDAY_EXPIRY_EFFECTIVE_DATE:
        return tuesday
    return last_weekday_of_month(target.year, target.month, weekday=3)  # 3 = Thursday


def last_friday_before(expiry: date) -> date:
    """Return the Friday strictly before the given expiry date."""

    d = expiry - timedelta(days=1)
    while d.weekday() != 4:  # 4 = Friday
        d -= timedelta(days=1)
    if d >= expiry:
        raise ValueError("Computed Friday is not before expiry")
    return d


def batman_entry_datetime(expiry: date, entry_time: time = time(15, 15)) -> datetime:
    """Last-Friday-before-expiry 15:15 IST entry timestamp."""

    entry_date = last_friday_before(expiry)
    return datetime.combine(entry_date, entry_time)
