from datetime import datetime, timedelta

import pandas as pd

from market_ai.modules.analytics.live_trade_monitor import (
    evaluate_position_health,
    find_multi_tf_zones,
)


def _make_candles(start: datetime, tf: str, prices: list[float]) -> pd.DataFrame:
    rows = []
    delta = {"1m": timedelta(minutes=1), "5m": timedelta(minutes=5), "15m": timedelta(minutes=15)}.get(tf, timedelta(minutes=5))
    t = start
    for p in prices:
        rows.append({"timestamp": t, "high": p * 1.001, "low": p * 0.999})
        t += delta
    return pd.DataFrame(rows)


def test_support_resistance_detection_and_health():
    start = datetime(2025, 1, 1, 9, 15)
    candles = {
        "5m": _make_candles(start, "5m", [100, 102, 104, 103, 105, 103, 101, 99, 100, 98, 97]),
        "15m": _make_candles(start, "15m", [100, 101, 102, 101, 103, 102, 101, 100]),
    }
    zones = find_multi_tf_zones(candles)
    spot = 101.0
    health = evaluate_position_health(pnl=500, pnl_pct=1.0, spot=spot, zones=zones, min_buffer_pct=0.001)
    summary = health.risk_flag
    assert summary in {"safe", "caution", "exit"}
    assert health.nearest_support is None or health.nearest_support.price <= spot
    assert health.nearest_resistance is None or health.nearest_resistance.price >= spot
