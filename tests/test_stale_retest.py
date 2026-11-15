from datetime import datetime, timedelta

from market_ai.intraday_defined_risk import strategy_selector as ss
from market_ai.intraday_defined_risk import trend_day


def _bars(rows):
    t0 = datetime(2026, 11, 12, 9, 15)
    return [{"t": t0 + timedelta(minutes=5 * i), "o": o, "h": h, "l": l, "c": c} for i, (o, h, l, c) in enumerate(rows)]


def _choice(struct):
    return ss.StrategyChoice(condition="BREAKOUT_UP", family=ss.FAM_DIRECTIONAL_DEBIT, structures=[struct],
                             rationale="", executable_today=True)


def test_extreme_age():
    b = _bars([(10, 12, 9, 11), (11, 15, 10, 14), (14, 14, 8, 9), (9, 10, 8.5, 9.5)])
    assert trend_day.extreme_age(b) == {"high_bars": 2, "low_bars": 1}


def test_stale_gate(monkeypatch):
    monkeypatch.setenv("SEL_STALE_EXTREME", "4")
    md = {"trend_day": {"extreme_age": {"high_bars": 7, "low_bars": 0}}}
    assert ss._stale_extreme_gate(_choice("CALL_DEBIT_SPREAD"), md).structures == []
    md = {"trend_day": {"extreme_age": {"high_bars": 2, "low_bars": 9}}}
    assert ss._stale_extreme_gate(_choice("CALL_DEBIT_SPREAD"), md).structures == ["CALL_DEBIT_SPREAD"]
    assert ss._stale_extreme_gate(_choice("PUT_DEBIT_SPREAD"), md).structures == []
    md = {"failed_break_entry": "x", "trend_day": {"extreme_age": {"high_bars": 9}}}
    assert ss._stale_extreme_gate(_choice("CALL_DEBIT_SPREAD"), md).structures == ["CALL_DEBIT_SPREAD"]


def test_stale_gate_off(monkeypatch):
    monkeypatch.delenv("SEL_STALE_EXTREME", raising=False)
    md = {"trend_day": {"extreme_age": {"high_bars": 70}}}
    assert ss._stale_extreme_gate(_choice("CALL_DEBIT_SPREAD"), md).structures == ["CALL_DEBIT_SPREAD"]


def test_extended_then_retest(monkeypatch):
    from datetime import time
    monkeypatch.setenv("SEL_RETEST_ENTRY", "30,15")
    monkeypatch.setattr(ss, "_DEBIT_BLACKOUT", None)
    ss._RETEST_ARMED.clear()
    ab = {"dir": "DOWN", "at": "2026-11-11T13:35:00", "or_low": 25264.0, "or_high": 25424.0}
    md = {"trend_day": {"accepted_break": ab, "last_bar": {"o": 25227, "h": 25228, "l": 25179, "c": 25188.5}}}
    c = ss._extended_break_gate(_choice("PUT_DEBIT_SPREAD"), md, time(13, 50))
    assert c.structures == [] and md["extended_break_skip"] == 75.5
    # bounce bar that closes up: no entry
    md = {"trend_day": {"accepted_break": ab, "last_bar": {"o": 25245.8, "h": 25261.5, "l": 25240.5, "c": 25255.5}}}
    c = ss._retest_entry(ss.StrategyChoice(condition="CHOP", family=ss.FAM_STAND_ASIDE, structures=[], rationale=""), md, time(14, 10))
    assert c.structures == []
    # rejection bar back below the level: enter put, stop at the level
    md = {"trend_day": {"accepted_break": ab, "last_bar": {"o": 25255.8, "h": 25255.8, "l": 25226.1, "c": 25228.6}}}
    c = ss._retest_entry(ss.StrategyChoice(condition="CHOP", family=ss.FAM_STAND_ASIDE, structures=[], rationale=""), md, time(14, 15))
    assert c.structures == ["PUT_DEBIT_SPREAD"] and md["level_stop_spot"] == 25264.0
    # once per side
    md = {"trend_day": {"accepted_break": ab, "last_bar": {"o": 25255.8, "h": 25255.8, "l": 25226.1, "c": 25228.6}}}
    c = ss._retest_entry(ss.StrategyChoice(condition="CHOP", family=ss.FAM_STAND_ASIDE, structures=[], rationale=""), md, time(14, 20))
    assert c.structures == []


def test_small_extension_not_blocked(monkeypatch):
    from datetime import time
    monkeypatch.setenv("SEL_RETEST_ENTRY", "30,15")
    ss._RETEST_ARMED.clear()
    ab = {"dir": "DOWN", "at": "2026-11-11T13:35:00", "or_low": 25264.0}
    md = {"trend_day": {"accepted_break": ab, "last_bar": {"o": 25270, "h": 25272, "l": 25240, "c": 25245}}}
    assert ss._extended_break_gate(_choice("PUT_DEBIT_SPREAD"), md, time(13, 40)).structures == ["PUT_DEBIT_SPREAD"]
