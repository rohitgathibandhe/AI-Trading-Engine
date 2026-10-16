from datetime import datetime, time, timedelta

from market_ai.intraday_defined_risk.data_models import OhlcvBar
from market_ai.intraday_defined_risk.features import opening_drive_read
from market_ai.intraday_defined_risk import strategy_selector as sel


def _bars(rows, day=datetime(2026, 10, 15, 9, 15)):
    return [OhlcvBar(timestamp=day + timedelta(minutes=5 * i), open=o, high=max(o, c) + 5, low=min(o, c) - 5,
                     close=c, volume=1.0) for i, (o, c) in enumerate(rows)]


def test_drive_when_gap_and_first15_agree():
    # 2026-10-15: prev close 22,622, opened 22,671 (gap +49), first 15m +79 -> BULLISH (day closed +188 from 09:30)
    r = opening_drive_read(_bars([(22671, 22700), (22700, 22730), (22730, 22750)]), 22622.0)
    assert r["dir"] == "BULLISH" and r["gap"] == 49 and r["body"] == 79


def test_no_drive_when_they_disagree():
    # 2026-10-16: gap +68 then a -113 first 15m (a sweep) -> no drive; the day went UP
    r = opening_drive_read(_bars([(23065, 22956), (22958, 22988), (22988, 22952)]), 22997.0)
    assert r["dir"] is None


def test_needs_the_0915_bars_and_prev_close():
    assert opening_drive_read(_bars([(1, 2), (2, 3)]), 0.5)["dir"] is None
    assert opening_drive_read(_bars([(1, 2), (2, 3), (3, 4)]), None)["dir"] is None
    late = _bars([(1, 2), (2, 3), (3, 4)], day=datetime(2026, 10, 15, 9, 30))
    assert opening_drive_read(late, 0.5)["dir"] is None


def test_selector_fills_direction_only_in_window(monkeypatch):
    monkeypatch.setattr(sel, "_high_prob_confluence", lambda read, md: (None, "no confluence"))
    md = {"opening_drive": {"dir": "BULLISH", "gap": 49.0, "body": 79.0}}
    c = sel.select_strategy(dict(md), 22750.0, now_time=time(9, 36))
    assert c.structures and c.structures[0] in ("BULL_PUT_CREDIT_SPREAD", "CALL_DEBIT_SPREAD")
    c2 = sel.select_strategy(dict(md), 22750.0, now_time=time(10, 5))
    assert not c2.structures or "OPENING DRIVE" not in (c2.rationale or "")
