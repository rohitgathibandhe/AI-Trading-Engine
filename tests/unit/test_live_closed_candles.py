from datetime import datetime

from market_ai.intraday_defined_risk.data_models import OhlcvBar
from market_ai.intraday_defined_risk.live_runtime import _closed_only


def _b(hh, mm):
    return OhlcvBar(timestamp=datetime(2026, 10, 22, hh, mm), open=1, high=1, low=1, close=1, volume=0)


def test_forming_candle_is_dropped():
    bars = [_b(14, 15), _b(14, 20), _b(14, 25)]
    now = datetime(2026, 10, 22, 14, 27, 54)          # 14:25 candle still forming
    assert [b.timestamp.minute for b in _closed_only(bars, 5, now)] == [15, 20]
    assert len(_closed_only(bars, 5, datetime(2026, 10, 22, 14, 30, 0))) == 3


def test_toggle_restores_old_behaviour(monkeypatch):
    monkeypatch.setenv("LIVE_CLOSED_CANDLES", "0")
    assert len(_closed_only([_b(14, 25)], 5, datetime(2026, 10, 22, 14, 26))) == 1
