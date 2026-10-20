from datetime import datetime, timedelta
from types import SimpleNamespace

from market_ai.intraday_defined_risk import trade_plan as tp


def _bars(hlc, start=datetime(2026, 10, 19, 11, 30)):
    return [SimpleNamespace(timestamp=start + timedelta(minutes=5 * i), open=c, high=h, low=l, close=c) for i, (h, l, c) in enumerate(hlc)]


def test_swing_stop_tracks_last_lower_high_after_entry():
    b = _bars([(23016, 22992, 23002), (22992, 22949, 22954), (22979, 22939, 22961), (22964, 22942, 22946),
               (22961, 22936, 22946), (22957, 22941, 22945), (22949, 22928, 22929)])
    # monotonic decline after entry -> no pivot yet -> stop = high of the bars into entry (the failed retest)
    assert tp.swing_stop(b, b[1].timestamp, short=True) == 23016
    b2 = b + _bars([(22944, 22924, 22931), (22939, 22914, 22922)], start=b[-1].timestamp + timedelta(minutes=5))
    b2[4] = SimpleNamespace(timestamp=b2[4].timestamp, open=0, high=22965, low=22936, close=22946)   # a lower high
    assert tp.swing_stop(b2, b2[1].timestamp, short=True) == 22965


def test_failed_retest_short():
    base = [(23080, 23040, 23060)] * 20
    recent = [(23030, 23000, 23010), (23020, 22990, 22995), (23028, 23000, 23005), (23034, 23008, 23012), (23030, 22990, 22998), (23000, 22980, 22985)]
    assert tp.failed_retest(_bars(base + recent), short=True) is True


def test_walls_from_chain():
    q = [SimpleNamespace(strike=k, option_type=t, oi=v) for k, t, v in
         [(22850, "PUT", 21_500_000), (22800, "PUT", 29_200_000), (22900, "PUT", 20_400_000), (22950, "PUT", 8_000_000),
          (23000, "PUT", 8_400_000), (23000, "CALL", 35_600_000), (22950, "CALL", 27_000_000), (23050, "CALL", 20_000_000),
          (22700, "PUT", 19_600_000), (23100, "CALL", 27_700_000)]]
    chain = SimpleNamespace(quotes=q)
    assert tp.next_wall(chain, short=True, spot=22950) == 22800
    assert tp.nearest_wall_distance(chain, short=True, spot=22872) is not None
