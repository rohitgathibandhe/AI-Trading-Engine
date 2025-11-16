from datetime import date, time

from market_ai.utils.index_calendar import (
    batman_entry_datetime,
    last_friday_before,
    last_weekday_of_month,
    monthly_expiry_date,
)


def test_last_weekday_of_month():
    assert last_weekday_of_month(2024, 11, 3) == date(2024, 11, 28)  # Thu
    assert last_weekday_of_month(2025, 1, 4) == date(2025, 1, 31)  # Fri


def test_monthly_expiry_and_last_friday():
    expiry = monthly_expiry_date(date(2025, 6, 1))
    assert expiry == date(2025, 6, 26)
    entry = last_friday_before(expiry)
    assert entry == date(2025, 6, 20)


def test_batman_entry_datetime():
    expiry = date(2025, 9, 25)
    entry_dt = batman_entry_datetime(expiry)
    assert entry_dt.date() == date(2025, 9, 19)
    assert entry_dt.time() == time(15, 15)
