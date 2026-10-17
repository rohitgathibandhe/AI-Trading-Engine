import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import weekly_positional_paper as wp  # noqa: E402

LISTED = ["2026-10-19", "2026-10-27", "2026-11-03", "2026-11-10"]


def test_trading_dte_counts_weekdays():
    assert wp._trading_dte("2026-10-27", date(2026, 10, 21)) == 4   # Thu,Fri,Mon,Tue
    assert wp._trading_dte("2026-11-03", date(2026, 10, 21)) == 9


def test_window_picks_two_week_expiry_right_after_expiry():
    e, why = wp._entry_expiry(LISTED, date(2026, 10, 21))           # Wed: 10-27 is DTE4 (<6) -> 11-03 DTE9
    assert e == "2026-11-03"


def test_window_waits_when_too_early():
    e, why = wp._entry_expiry(LISTED, date(2026, 10, 20))           # Tue: 10-27 DTE5, 11-03 DTE10 (>9)
    assert e is None and "wait" in why


def test_window_monday_takes_dte6():
    assert wp._entry_expiry(LISTED, date(2026, 10, 26))[0] == "2026-11-03"   # Mon: 11-03 is DTE6
