from __future__ import annotations

from datetime import date, timedelta
from typing import Optional


def _is_tuesday(d: date) -> bool:
    return d.weekday() == 1


def last_tuesday_of_month(y: int, m: int) -> date:
    if m == 12:
        nm = date(y + 1, 1, 1)
    else:
        nm = date(y, m + 1, 1)
    d = nm - timedelta(days=1)
    while not _is_tuesday(d):
        d -= timedelta(days=1)
    return d


def next_tuesday(d: Optional[date] = None) -> date:
    d = d or date.today()
    ahead = (1 - d.weekday()) % 7
    if ahead == 0:
        ahead = 7
    return d + timedelta(days=ahead)


def normalize_expiry(mode: str, user_expiry: Optional[str] = None) -> str:
    if mode == "auto_weekly":
        return next_tuesday().isoformat()
    if mode == "auto_monthly":
        today = date.today()
        d = last_tuesday_of_month(today.year, today.month)
        if d < today:
            yy, mm = (today.year + 1, 1) if today.month == 12 else (today.year, today.month + 1)
            d = last_tuesday_of_month(yy, mm)
        return d.isoformat()
    if user_expiry:
        try:
            y, m, dd = map(int, user_expiry.split("-"))
            return date(y, m, dd).isoformat()
        except Exception:
            pass
    return next_tuesday().isoformat()
