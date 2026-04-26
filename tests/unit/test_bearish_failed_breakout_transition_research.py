from market_ai.intraday_defined_risk.bearish_failed_breakout_transition_research import (
    matches_bearish_failed_breakout_transition_shadow,
)
from market_ai.intraday_defined_risk.data_models import AdaptiveParameters


def test_matches_failed_breakout_transition_shadow_reference_case() -> None:
    metadata = {
        "market_state": "TRANSITION",
        "failure_type": "FAILED_BREAKOUT",
        "tradability_class": "TRADABLE",
        "opening_range_break_state": "FAILED_UP",
        "bearish_trade_score": 6.0208,
        "bearish_trade_score_threshold": 6.1,
        "setup_quality_score": 21.65,
        "structure_signal": "BEARISH_CHOCH",
        "lower_high_confirmed": True,
        "bearish_candle_quality_score": 4.0,
        "trend_efficiency_ratio": 0.5903,
        "momentum_persistence_score": 0.7143,
        "price_vs_vwap": -16.9,
        "price_vs_ema20": -20.31,
        "price_vs_or_high": -14.45,
    }
    assert matches_bearish_failed_breakout_transition_shadow(
        metadata,
        execution_5m="UNCONFIRMED",
        params=AdaptiveParameters(),
    )


def test_rejects_when_score_is_not_within_shadow_buffer() -> None:
    metadata = {
        "market_state": "TRANSITION",
        "failure_type": "FAILED_BREAKOUT",
        "tradability_class": "TRADABLE",
        "opening_range_break_state": "FAILED_UP",
        "bearish_trade_score": 5.7,
        "bearish_trade_score_threshold": 6.1,
        "setup_quality_score": 21.65,
        "structure_signal": "BEARISH_CHOCH",
        "lower_high_confirmed": True,
        "bearish_candle_quality_score": 4.0,
        "trend_efficiency_ratio": 0.5903,
        "momentum_persistence_score": 0.7143,
        "price_vs_vwap": -16.9,
        "price_vs_ema20": -20.31,
        "price_vs_or_high": -14.45,
    }
    assert not matches_bearish_failed_breakout_transition_shadow(
        metadata,
        execution_5m="UNCONFIRMED",
        params=AdaptiveParameters(),
    )


def test_rejects_when_failed_breakout_has_not_lost_acceptance() -> None:
    metadata = {
        "market_state": "TRANSITION",
        "failure_type": "FAILED_BREAKOUT",
        "tradability_class": "TRADABLE",
        "opening_range_break_state": "FAILED_UP",
        "bearish_trade_score": 6.05,
        "bearish_trade_score_threshold": 6.1,
        "setup_quality_score": 21.65,
        "structure_signal": "BEARISH_CHOCH",
        "lower_high_confirmed": True,
        "bearish_candle_quality_score": 4.0,
        "trend_efficiency_ratio": 0.5903,
        "momentum_persistence_score": 0.7143,
        "price_vs_vwap": 2.0,
        "price_vs_ema20": 1.0,
        "price_vs_or_high": -1.0,
    }
    assert not matches_bearish_failed_breakout_transition_shadow(
        metadata,
        execution_5m="UNCONFIRMED",
        params=AdaptiveParameters(),
    )
