from market_ai.intraday_defined_risk.data_models import AdaptiveParameters, RegimeLabel, RegimeState
from market_ai.intraday_defined_risk.strategy import (
    REGIME_TRADABILITY_LOW_EDGE,
    REGIME_TRADABILITY_NOT_TRADABLE,
    REGIME_TRADABILITY_SPREAD,
    classify_regime_tradability,
    low_edge_tradability_filter_reason,
)


def _regime_state(**metadata):
    return RegimeState(
        regime=RegimeLabel.UP_TREND,
        trend_15m="TREND_UP",
        execution_5m="UP_CONFIRMED",
        ema20_15m=1.0,
        ema20_slope_15m=1.0,
        rv30_pct=0.2,
        or_length_minutes=30,
        opening_range_high=1.0,
        opening_range_low=0.5,
        vwap=1.0,
        confidence=0.7,
        reasons=[],
        metadata=metadata,
    )


def test_gap_down_failed_bounce_is_tradable_spread() -> None:
    regime_state = _regime_state(
        playbook="GAP_DOWN_BEARISH_CONTINUATION",
        bearish_subtype="GAP_DOWN_FAILED_BOUNCE",
    )
    tradability, _ = classify_regime_tradability(regime_state)
    assert tradability == REGIME_TRADABILITY_SPREAD


def test_bearish_rejection_transition_is_tradable_spread() -> None:
    regime_state = _regime_state(
        playbook="SIDEWAYS_TO_BEARISH_REJECTION",
        bearish_subtype="BEARISH_REJECTION_TRANSITION",
    )
    tradability, _ = classify_regime_tradability(regime_state)
    assert tradability == REGIME_TRADABILITY_SPREAD


def test_late_session_bullish_reclaim_is_low_edge() -> None:
    regime_state = _regime_state(
        playbook="SIDEWAYS_TO_BULLISH_RECLAIM",
        bullish_shadow_subtype="LATE_SESSION_BULLISH_RECLAIM",
    )
    tradability, _ = classify_regime_tradability(regime_state)
    assert tradability == REGIME_TRADABILITY_LOW_EDGE


def test_bullish_expansion_playbook_is_not_tradable() -> None:
    regime_state = _regime_state(
        playbook="OPEN_DRIVE_BULLISH",
    )
    tradability, _ = classify_regime_tradability(regime_state)
    assert tradability == REGIME_TRADABILITY_NOT_TRADABLE


def test_low_edge_filter_requires_strong_quality_and_confidence() -> None:
    regime_state = _regime_state(
        playbook="SIDEWAYS_TO_BULLISH_RECLAIM",
        bullish_shadow_subtype="LATE_SESSION_BULLISH_RECLAIM",
        setup_quality_score=8.0,
    )
    reason = low_edge_tradability_filter_reason(regime_state, AdaptiveParameters())
    assert reason is not None
