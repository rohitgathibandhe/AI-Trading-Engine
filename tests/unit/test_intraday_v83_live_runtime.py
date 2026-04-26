from __future__ import annotations

from datetime import datetime

from market_ai.intraday_defined_risk.data_models import OptionType
from market_ai.intraday_defined_risk.live_runtime import _bars_from_candles, _normalize_timestamp, _quote_from_row, _vwap_from_bars


def test_live_runtime_converts_dhan_candles_to_v83_bars() -> None:
    bars = _bars_from_candles(
        [
            {
                "timestamp": datetime(2026, 4, 21, 9, 15),
                "open": 24000.0,
                "high": 24040.0,
                "low": 23980.0,
                "close": 24020.0,
                "volume": 1000,
            },
            {"timestamp": None, "open": 0},
        ]
    )

    assert len(bars) == 1
    assert bars[0].close == 24020.0
    assert _vwap_from_bars(bars) == 24013.333333333332


def test_live_runtime_parses_dhan_epoch_timestamps_as_ist() -> None:
    parsed = _normalize_timestamp(1776743100.0)

    assert parsed == datetime(2026, 4, 21, 9, 15)


def test_live_runtime_builds_quote_with_ltp_bid_ask_fallback() -> None:
    quote = _quote_from_row(
        {
            "strike": 24500,
            "option_type": "CALL",
            "ltp": 42.5,
            "bid": "",
            "ask": "",
            "delta": 0.22,
            "iv": 13.1,
            "oi": 10000,
        }
    )

    assert quote is not None
    assert quote.option_type == OptionType.CALL
    assert quote.bid > 0.0
    assert quote.ask >= quote.bid
    assert quote.delta == 0.22
