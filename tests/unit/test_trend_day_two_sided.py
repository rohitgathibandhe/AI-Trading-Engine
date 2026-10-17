from datetime import datetime, timedelta

from market_ai.intraday_defined_risk import trend_day as td


def _bars(closes, start=datetime(2026, 10, 7, 9, 15)):
    out, prev = [], closes[0]
    for i, c in enumerate(closes):
        out.append({"t": start + timedelta(minutes=5 * i), "o": prev, "h": max(prev, c) + 3, "l": min(prev, c) - 3,
                    "c": c, "v": 0.0})
        prev = c
    return out


def test_two_sided_or_break_is_a_range_day_not_a_trend():
    # OR 09:15-09:30 ~22600-22693; closes below the OR low, then above the OR high, then held above VWAP
    closes = [22690, 22618, 22624, 22614, 22588, 22617, 22613, 22614, 22609, 22609, 22623, 22612, 22656, 22667,
              22698, 22705, 22706, 22708, 22710, 22712, 22714, 22716]
    st = td.trend_state(_bars(closes))
    assert st["state"] == "RANGE_TWO_SIDED"


def test_one_sided_break_can_still_be_a_trend(monkeypatch):
    closes = [22650, 22660, 22655, 22670, 22700, 22720, 22735, 22750, 22760, 22772, 22785, 22795, 22805]
    st = td.trend_state(_bars(closes), oi={"smart_money_bias": "BULLISH"})
    assert st["state"] == "TREND_UP"
    monkeypatch.setattr(td, "TWO_SIDED_RANGE", False)
    assert td.trend_state(_bars(closes), oi={"smart_money_bias": "BULLISH"})["state"] == "TREND_UP"
