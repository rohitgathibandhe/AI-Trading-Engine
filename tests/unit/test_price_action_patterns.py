from __future__ import annotations

from market_ai.modules.analytics.price_action_patterns import detect_recent_price_action


def test_detect_recent_price_action_flags_double_top_confirmation() -> None:
    candles = [
        {"open": 23000.0, "high": 23060.0, "low": 22980.0, "close": 23040.0},
        {"open": 23040.0, "high": 23180.0, "low": 23020.0, "close": 23160.0},
        {"open": 23160.0, "high": 23170.0, "low": 23060.0, "close": 23090.0},
        {"open": 23090.0, "high": 23175.0, "low": 23070.0, "close": 23140.0},
        {"open": 23140.0, "high": 23145.0, "low": 22990.0, "close": 23020.0},
    ]
    out = detect_recent_price_action(
        candles,
        support=22980.0,
        resistance=23170.0,
        atr_like_points=95.0,
        ema_slow=23095.0,
    )
    assert out["price_action_bias"] == "BEARISH"
    assert "DOUBLE_TOP_M_CONFIRMED" in (out.get("price_action_patterns") or [])
    assert out["retest_status"] == "DOUBLE_TOP_NECKLINE_BREAK"


def test_detect_recent_price_action_flags_fake_breakout_reversal() -> None:
    candles = [
        {"open": 23000.0, "high": 23040.0, "low": 22970.0, "close": 23020.0},
        {"open": 23020.0, "high": 23070.0, "low": 22990.0, "close": 23040.0},
        {"open": 23040.0, "high": 23100.0, "low": 23020.0, "close": 23092.0},
        {"open": 23092.0, "high": 23135.0, "low": 23000.0, "close": 23018.0},
    ]
    out = detect_recent_price_action(
        candles,
        support=22960.0,
        resistance=23080.0,
        atr_like_points=82.0,
        ema_slow=23070.0,
    )
    assert out["price_action_bias"] == "BEARISH"
    assert out["price_action_confirmation"] == "REVERSAL_CONFIRMED"
    assert "FAKE_BREAKOUT_UP" in (out.get("price_action_patterns") or [])
    assert out["retest_status"] == "FAILED_BREAKOUT_RETEST"


def test_detect_recent_price_action_flags_double_bottom_confirmation() -> None:
    candles = [
        {"open": 22980.0, "high": 23010.0, "low": 22920.0, "close": 22950.0},
        {"open": 22950.0, "high": 22970.0, "low": 22820.0, "close": 22840.0},
        {"open": 22840.0, "high": 22960.0, "low": 22830.0, "close": 22930.0},
        {"open": 22930.0, "high": 22940.0, "low": 22825.0, "close": 22870.0},
        {"open": 22870.0, "high": 23025.0, "low": 22860.0, "close": 22990.0},
    ]
    out = detect_recent_price_action(
        candles,
        support=22840.0,
        resistance=23010.0,
        atr_like_points=92.0,
        ema_slow=22910.0,
    )
    assert out["price_action_bias"] == "BULLISH"
    assert "DOUBLE_BOTTOM_W_CONFIRMED" in (out.get("price_action_patterns") or [])
    assert out["retest_status"] == "DOUBLE_BOTTOM_NECKLINE_BREAK"


def test_detect_recent_price_action_surfaces_bullish_fvg_and_fib_pullback() -> None:
    candles = [
        {"open": 100.0, "high": 100.5, "low": 98.8, "close": 99.8},
        {"open": 99.8, "high": 108.8, "low": 99.6, "close": 108.2},
        {"open": 108.3, "high": 110.0, "low": 107.6, "close": 108.4},
        {"open": 108.4, "high": 108.8, "low": 105.2, "close": 106.1},
    ]
    out = detect_recent_price_action(
        candles,
        support=103.5,
        resistance=108.0,
        atr_like_points=3.5,
        ema_slow=103.8,
    )
    assert out["fair_value_gap_bias"] == "BULLISH"
    assert out["fair_value_gap_low"] is not None
    assert out["fair_value_gap_high"] is not None
    assert out["fib_bullish_zone_low"] is not None
    assert out["fib_bullish_zone_high"] is not None
    assert out["fib_bullish_active"] is True


def test_detect_recent_price_action_flags_strong_bullish_5m_candle() -> None:
    candles = [
        {"open": 25480.0, "high": 25510.0, "low": 25460.0, "close": 25490.0},
        {"open": 25490.0, "high": 25505.0, "low": 25440.0, "close": 25455.0},
        {"open": 25455.0, "high": 25590.0, "low": 25445.0, "close": 25580.0},
    ]
    out = detect_recent_price_action(
        candles,
        support=25450.0,
        resistance=25610.0,
        atr_like_points=90.0,
        ema_slow=25510.0,
        ema_base=25480.0,
    )
    assert out["strong_candle_confirmed"] is True
    assert out["strong_candle_bias"] == "BULLISH"
    assert out["strong_candle_status"] == "STRONG_GREEN_5M"
