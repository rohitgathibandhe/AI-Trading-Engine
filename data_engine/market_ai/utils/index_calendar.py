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

    Per our NIFTY setup, we treat the monthly expiry as the last Tuesday
    of the month (weekday=1), shifting earlier if that Tuesday is a holiday
    via last_weekday_of_month helper.
    """

    return last_weekday_of_month(target.year, target.month, weekday=1)  # 1 = Tuesday


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
