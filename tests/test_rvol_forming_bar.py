from datetime import datetime, timedelta

from market_ai.intraday_defined_risk.data_models import OhlcvBar
from market_ai.intraday_defined_risk.features import compute_volume_signals


def _bars(vols, start=datetime(2026, 10, 12, 11, 0)):
    return [OhlcvBar(start + timedelta(minutes=5 * i), 100, 101, 99, 100.5, v) for i, v in enumerate(vols)]


def test_forming_bar_is_excluded_from_relative_volume():
    bars = _bars([100, 100, 100, 100, 5])          # last bar just opened -> tiny partial volume
    now = bars[-1].timestamp + timedelta(minutes=1)
    assert compute_volume_signals(bars, now=now)["relative_volume"] == 1.0   # uses the last COMPLETED bar
    assert compute_volume_signals(bars)["relative_volume"] == 0.05            # legacy behaviour without `now`


def test_completed_last_bar_is_used():
    bars = _bars([100, 100, 100, 200])
    now = bars[-1].timestamp + timedelta(minutes=5, seconds=1)
    assert compute_volume_signals(bars, now=now)["relative_volume"] == 2.0
