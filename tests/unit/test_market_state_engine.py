from datetime import time

from market_ai.intraday_defined_risk.regime import _market_state_classification


def test_transition_has_priority_over_trend_alignment() -> None:
    market_state, bias, state_quality_score, state_confidence_score, _ = _market_state_classification(
        spot=100.0,
        now_time=time(11, 0),
        vwap=99.0,
        ema_fast=101.0,
        ema_mid=100.0,
        ema_slow=99.0,
        current_structure_signal="BEARISH_CHOCH",
        execution_5m="DOWN_CONFIRMED",
        bullish_pullback=False,
        bullish_shallow=False,
        bullish_vwap_hold=False,
        bearish_pullback=True,
        bearish_shallow=False,
        range_entry_ready=False,
        gap_up_bearish_failure_ready=True,
        gap_down_bearish_continuation_ready=False,
        gap_up_bullish_ready=False,
        gap_down_bullish_recovery_ready=False,
        open_drive_bullish_ready=False,
        open_drive_bearish=False,
        bullish_candle_pattern="NONE",
        bearish_candle_pattern="BEARISH_BREAKOUT_FAILURE",
        around_vwap=False,
    )
    assert market_state == "TRANSITION"
    assert bias == "BEARISH"
    assert state_quality_score > 0
    assert state_confidence_score > 0.7


def test_true_range_state_penalizes_non_range_context() -> None:
    market_state, bias, state_quality_score, _, _ = _market_state_classification(
        spot=100.0,
        now_time=time(12, 0),
        vwap=100.0,
        ema_fast=100.1,
        ema_mid=100.0,
        ema_slow=99.95,
        current_structure_signal="BALANCED",
        execution_5m="RANGE_CONFIRMED",
        bullish_pullback=False,
        bullish_shallow=False,
        bullish_vwap_hold=False,
        bearish_pullback=False,
        bearish_shallow=False,
        range_entry_ready=False,
        gap_up_bearish_failure_ready=False,
        gap_down_bearish_continuation_ready=False,
        gap_up_bullish_ready=False,
        gap_down_bullish_recovery_ready=False,
        open_drive_bullish_ready=False,
        open_drive_bearish=False,
        bullish_candle_pattern="NONE",
        bearish_candle_pattern="NONE",
        around_vwap=True,
        opening_range_break_state="NONE",
        candle_overlap_ratio=0.82,
        trend_efficiency_ratio=0.20,
        momentum_persistence_score=0.18,
    )
    assert market_state == "TRUE_RANGE"
    assert bias == "NEUTRAL"
    assert state_quality_score < 0


def test_directional_balance_identifies_bearish_balance_to_breakdown() -> None:
    market_state, bias, state_quality_score, _, _ = _market_state_classification(
        spot=99.2,
        now_time=time(11, 45),
        vwap=99.8,
        ema_fast=99.5,
        ema_mid=99.45,
        ema_slow=99.55,
        current_structure_signal="BEARISH_BOS",
        execution_5m="DOWN_CONFIRMED",
        bullish_pullback=False,
        bullish_shallow=False,
        bullish_vwap_hold=False,
        bearish_pullback=True,
        bearish_shallow=False,
        range_entry_ready=False,
        gap_up_bearish_failure_ready=False,
        gap_down_bearish_continuation_ready=False,
        gap_up_bullish_ready=False,
        gap_down_bullish_recovery_ready=False,
        open_drive_bullish_ready=False,
        open_drive_bearish=False,
        bullish_candle_pattern="NONE",
        bearish_candle_pattern="BEARISH_REJECTION_WICK",
        around_vwap=True,
        opening_range_break_state="FAILED_UP",
        candle_overlap_ratio=0.62,
        trend_efficiency_ratio=0.39,
        momentum_persistence_score=0.42,
        lower_high_confirmed=True,
        failed_bounce_candidate=False,
    )
    assert market_state == "DIRECTIONAL_BALANCE"
    assert bias == "BEARISH"
    assert state_quality_score > 0


def test_late_session_penalty_reduces_state_quality() -> None:
    earlier = _market_state_classification(
        spot=100.0,
        now_time=time(13, 30),
        vwap=99.0,
        ema_fast=101.0,
        ema_mid=100.0,
        ema_slow=99.0,
        current_structure_signal="BULLISH_BOS",
        execution_5m="UP_CONFIRMED",
        bullish_pullback=True,
        bullish_shallow=False,
        bullish_vwap_hold=False,
        bearish_pullback=False,
        bearish_shallow=False,
        range_entry_ready=False,
        gap_up_bearish_failure_ready=False,
        gap_down_bearish_continuation_ready=False,
        gap_up_bullish_ready=True,
        gap_down_bullish_recovery_ready=False,
        open_drive_bullish_ready=True,
        open_drive_bearish=False,
        bullish_candle_pattern="BULLISH_EXPANSION",
        bearish_candle_pattern="NONE",
        around_vwap=False,
    )
    later = _market_state_classification(
        spot=100.0,
        now_time=time(14, 5),
        vwap=99.0,
        ema_fast=101.0,
        ema_mid=100.0,
        ema_slow=99.0,
        current_structure_signal="BULLISH_BOS",
        execution_5m="UP_CONFIRMED",
        bullish_pullback=True,
        bullish_shallow=False,
        bullish_vwap_hold=False,
        bearish_pullback=False,
        bearish_shallow=False,
        range_entry_ready=False,
        gap_up_bearish_failure_ready=False,
        gap_down_bearish_continuation_ready=False,
        gap_up_bullish_ready=True,
        gap_down_bullish_recovery_ready=False,
        open_drive_bullish_ready=True,
        open_drive_bearish=False,
        bullish_candle_pattern="BULLISH_EXPANSION",
        bearish_candle_pattern="NONE",
        around_vwap=False,
    )
    assert earlier[2] > later[2]
