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


def test_accepted_break_needs_two_closes_and_detects_side():
    from datetime import datetime, timedelta
    from market_ai.intraday_defined_risk import trend_day as td
    t0 = datetime(2026, 10, 1, 9, 15)
    closes = [100, 101, 99, 100, 112, 99, 100, 85, 84, 83]          # OR 09:15-09:25; 09:35 poke up (1 close), then 2 closes down
    bars = [{"t": t0 + timedelta(minutes=5 * k), "o": c, "h": c + 3, "l": c - 3, "c": c, "v": 0} for k, c in enumerate(closes)]
    ab = td.accepted_break(bars)
    assert ab["dir"] == "DOWN" and not ab["two_sided"] and ab["valid"]


def test_failed_break_up_trap_gives_down_reversal():
    from datetime import datetime, timedelta
    from market_ai.intraday_defined_risk import trend_day as td
    t0 = datetime(2026, 10, 21, 9, 15)
    closes = [100, 104, 103, 110, 109, 98, 95]                       # OR high ~107; 09:30/35 above; 09:45 back inside < VWAP
    bars = [{"t": t0 + timedelta(minutes=5 * k), "o": c, "h": c + 3, "l": c - 3, "c": c, "v": 0} for k, c in enumerate(closes)]
    fb = td.failed_break(bars)
    assert fb["dir"] == "DOWN" and fb["valid"] and fb["trap_extreme"] == 113


def test_fib_pullback_long_into_618_zone():
    from datetime import datetime, timedelta
    from market_ai.intraday_defined_risk import trend_day as td
    t0 = datetime(2026, 10, 23, 9, 15)
    # OR, then impulse 1000 -> 1100, pullback to ~1035 (61.8-68% zone 1032-1038), strong green close off it
    seq = [(1000, 1003, 997, 1001)] * 3 + [(1001, 1021, 1000, 1020), (1020, 1051, 1019, 1050), (1050, 1081, 1049, 1080),
          (1080, 1101, 1079, 1100), (1100, 1101, 1070, 1072), (1072, 1074, 1040, 1042), (1042, 1044, 1033, 1036),
          (1036, 1056, 1035, 1055)]
    bars = [{"t": t0 + timedelta(minutes=5 * k), "o": o, "h": h, "l": l, "c": c, "v": 0} for k, (o, h, l, c) in enumerate(seq)]
    f = td.fib_pullback(bars)
    assert f and f["dir"] == "UP" and f["zone"][0] <= 1036 <= f["zone"][1] + 5


def test_failed_up_break_resets_trend_until_trap_is_cleared(monkeypatch):
    from datetime import datetime, timedelta
    from market_ai.intraday_defined_risk import trend_day as td
    monkeypatch.setattr(td, "FAILED_RESETS", True)
    t0 = datetime(2026, 10, 27, 9, 15)
    # OR 100-110; up-break closes 112/115 (trap high ~118); fails to 104 under VWAP; drifts back to 113-114 (< trap+margin)
    closes = [100, 108, 105, 112, 115, 104, 103, 106, 110, 113, 114, 113]
    bars = [{"t": t0 + timedelta(minutes=5 * k), "o": c, "h": c + 3, "l": c - 3, "c": c, "v": 0} for k, c in enumerate(closes)]
    st = td.trend_state(bars)
    assert st["state"] != "TREND_UP"
    ab = td.accepted_break(bars)
    assert not ab or ab["dir"] != "UP" or not ab["valid"]
