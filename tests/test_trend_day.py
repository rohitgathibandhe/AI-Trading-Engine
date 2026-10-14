from datetime import datetime, timedelta

from market_ai.intraday_defined_risk import trend_day as td

D = datetime(2026, 10, 13, 9, 15)


def _bars(closes, start=D, vol=1000.0):
    out, prev = [], closes[0]
    for i, c in enumerate(closes):
        o = prev
        out.append({"t": start + timedelta(minutes=5 * i), "o": o, "h": max(o, c) + 3, "l": min(o, c) - 3, "c": c, "v": vol})
        prev = c
    return out


def _up_day():
    # OR 09:15-09:30 ~22300-22340, breaks up, holds above VWAP, pulls back, resumes
    return _bars([22300, 22320, 22340, 22370, 22400, 22420, 22440, 22460, 22470, 22480, 22490, 22500,
                  22475, 22450, 22432, 22462])       # pullback into the VWAP/EMA20 zone, then a resume bar


def test_up_day_is_detected_on_completed_bars():
    b = _up_day()
    st = td.trend_state(b[:12])
    assert st["state"] == "TREND_UP" and st["checks"]["orb"] and st["checks"]["vwap_hold"]


def test_choppy_day_is_not_a_trend():
    b = _bars([22300, 22340, 22290, 22330, 22295, 22335, 22300, 22320, 22305, 22325, 22300, 22320])
    assert td.trend_state(b)["state"] == "NONE"


def test_pullback_then_resume_triggers_with_the_trend():
    b = _up_day()
    st = td.trend_state(b)
    tg = td.trigger(b, st)
    assert st["state"] == "TREND_UP" and tg is not None and tg["kind"] in ("PULLBACK", "RANGE_BREAKOUT")


def test_no_new_trend_entries_after_14_00():
    b = _bars([22300, 22320, 22340] + [22350 + 5 * i for i in range(60)])      # ends well after 14:00
    st = td.trend_state(b)
    assert st["state"] == "TREND_UP" and td.trigger(b, st) is None


def test_late_safety_net_joins_a_held_trend_after_11_30():
    # steady grind up from 09:30, no pullback/breakout pattern on the final bar, final bar 11:35 closes up
    closes = [22300, 22310, 22320] + [22330 + 2 * i for i in range(26)]
    b = _bars(closes)
    for x in b[-6:-1]:
        x["h"] = x["c"] + 30              # make the last bars a wide (non-tight) box so no RANGE_BREAKOUT
    b[-1]["c"] = b[-2]["c"] + 1           # tiny up-close: not a new day-high breakout with volume
    b[-1]["v"] = 1.0
    st = td.trend_state(b)
    tg = td.trigger(b, st)
    assert b[-1]["t"].time() >= td.SAFETY_FROM and st["state"] == "TREND_UP"
    assert tg is not None and tg["kind"] == "LATE_SAFETY"


def test_evaluate_uses_only_completed_bars():
    b = _up_day()
    now = b[-1]["t"] + timedelta(minutes=2)             # last bar still forming
    ev = td.evaluate(b, now)
    assert ev["trigger"] is None or ev["trigger"]["at"] != b[-1]["t"].isoformat()
