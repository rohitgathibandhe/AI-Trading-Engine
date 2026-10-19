from datetime import datetime, timedelta
from types import SimpleNamespace

from market_ai.intraday_defined_risk import oi_flow


def _chain(ts, spot, book):
    q = []
    for k, (c, p) in book.items():
        q.append(SimpleNamespace(strike=k, option_type="CALL", oi=c))
        q.append(SimpleNamespace(strike=k, option_type="PUT", oi=p))
    return SimpleNamespace(timestamp=ts, spot=spot, quotes=q)


def test_bearish_wall_flip_and_support():
    oi_flow.reset()
    t0 = datetime(2026, 10, 19, 11, 33)
    oi_flow.update(_chain(t0, 23035, {22950: (2_300_000, 15_900_000), 23000: (11_100_000, 29_800_000), 22850: (1_000_000, 11_300_000)}))
    oi_flow.update(_chain(t0 + timedelta(minutes=30), 22980, {22950: (6_500_000, 17_400_000), 23000: (24_200_000, 24_400_000), 22850: (1_200_000, 15_500_000)}))
    assert oi_flow.wall_flip(22980) == "BEARISH"
    # 10 minutes later: writers keep adding calls just above price (ceiling moving down) -> a short is still backed
    oi_flow.update(_chain(t0 + timedelta(minutes=40), 22947, {22950: (10_100_000, 15_800_000), 23000: (32_800_000, 15_100_000), 22850: (1_300_000, 15_800_000)}))
    assert oi_flow.supports("BEARISH", 22947) is True
    assert oi_flow.next_wall("BEARISH", 22980) == 22850


def test_no_oi_is_neutral():
    oi_flow.reset()
    assert oi_flow.wall_flip(23000) is None and oi_flow.supports("BEARISH", 23000) is None
