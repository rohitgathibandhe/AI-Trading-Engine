from datetime import datetime, timedelta
from types import SimpleNamespace as NS

from market_ai.intraday_defined_risk import execution as ex
from market_ai.intraday_defined_risk.data_models import StrategyType

DAY = datetime(2026, 11, 6, 9, 15)


def _bars(closes):
    """5m bars from closes; each bar spans 10 pts below its close (a steady up-day has rising lows)."""
    out, prev = [], closes[0] - 5
    for i, c in enumerate(closes):
        lo = min(prev, c) - 6
        out.append(NS(timestamp=DAY + timedelta(minutes=5 * i), open=prev, high=max(prev, c) + 3, low=lo, close=c, volume=0))
        prev = c
    return out


# OR 09:15-09:30 ~24600-24620, break up, then 30+ min of rising closes above VWAP with higher 15m lows
UP = [24610, 24605, 24615, 24640, 24660, 24680, 24700, 24720, 24735, 24750, 24770, 24790, 24810, 24830, 24850]


def _pos():
    return NS(entry_credit_points=-66.5, lot_size=65, lots=4, entry_time=DAY + timedelta(minutes=35),
              structure=NS(strategy=StrategyType.CALL_DEBIT_SPREAD, width_points=250.0), metadata={"is_fade": False})


def _check(monkeypatch, closes, flag):
    monkeypatch.setenv("DEBIT_TRAIL_ON_CLOSE", "1")
    monkeypatch.setenv("DEBIT_TREND_HOLD", "1" if flag else "0")
    monkeypatch.setattr(ex, "_current_debit_value", lambda *a: 120.0)
    pos = _pos()
    bars = _bars(closes)
    pos.metadata["debit_fav_spot"] = max(closes)
    now = bars[-1].timestamp + timedelta(minutes=5)
    snap = NS(option_chain=NS(spot=closes[-1]), nifty_5m=NS(bars=bars))
    return ex._evaluate_debit_exit(pos, snap, None, now)


def test_pullback_inside_a_confirmed_trend_is_held(monkeypatch):
    closes = UP + [24818]          # 32-pt pullback, still above VWAP and the last swing low
    assert _check(monkeypatch, closes, flag=False).reason == "DEBIT_SPOT_REVERSAL"
    assert not _check(monkeypatch, closes, flag=True).should_exit


def test_close_through_the_swing_low_exits(monkeypatch):
    closes = UP + [24800, 24815, 24830, 24785]   # swing low 24794 confirmed by two later bars; 24785 closes through it
    d = _check(monkeypatch, closes, flag=True)
    assert d.should_exit and d.reason == "DEBIT_TREND_SWING"


def test_no_trend_state_keeps_the_normal_trail(monkeypatch):
    flat = [24610, 24605, 24615, 24612, 24618, 24609, 24640, 24600, 24630, 24604, 24635, 24603]
    assert _check(monkeypatch, flat, flag=True).reason == "DEBIT_SPOT_REVERSAL"
