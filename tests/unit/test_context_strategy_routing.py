from __future__ import annotations

from datetime import time

from market_ai.intraday_defined_risk.data_models import RegimeLabel, RegimeState, StrategyType
from market_ai.intraday_defined_risk.strategy import (
    REGIME_TRADABILITY_LOW_EDGE,
    REGIME_TRADABILITY_TRADABLE,
    classify_regime_tradability,
    select_strategy,
)


def _bearish_regime(**metadata: object) -> RegimeState:
    base = {
        "day_archetype": "GAP_DOWN_CONTINUATION",
        "playbook": "GAP_DOWN_BEARISH_CONTINUATION",
        "bearish_entry_ready": True,
        "bearish_setup": "GAP_CONTINUATION",
        "market_state": "TREND_DOWN",
        "failure_type": "FAILED_BOUNCE",
        "bearish_failure_score": 4.0,
        "bearish_location_live_score": 4.0,
        "bearish_trade_score": 8.5,
        "no_trade_score": 3.0,
        "trade_score_margin_bearish": 1.5,
        "bearish_trade_score_threshold": 6.5,
        "tradability_class": "TRADABLE",
        "tradability_reason": "Bearish continuation is clean and monetizable.",
    }
    base.update(metadata)
    return RegimeState(
        regime=RegimeLabel.DOWN_TREND,
        trend_15m="TREND_DOWN",
        execution_5m="DOWN_CONFIRMED",
        ema20_15m=22000.0,
        ema20_slope_15m=-5.0,
        rv30_pct=0.45,
        or_length_minutes=30,
        opening_range_high=22100.0,
        opening_range_low=22000.0,
        vwap=22020.0,
        confidence=0.90,
        reasons=["Bearish state aligned."],
        metadata=base,
    )


def test_select_strategy_allows_high_score_bearish_context() -> None:
    regime = _bearish_regime()
    strategy, reasons = select_strategy(regime, time(11, 0), allowed_playbook_tiers=("A", "B"))
    assert strategy == StrategyType.BEAR_CALL_CREDIT_SPREAD
    assert any("Bear Call Credit Spread" in reason for reason in reasons)


def test_select_strategy_blocks_when_no_trade_score_is_too_close() -> None:
    regime = _bearish_regime(bearish_trade_score=7.0, no_trade_score=6.0)
    strategy, reasons = select_strategy(regime, time(11, 0), allowed_playbook_tiers=("A", "B"))
    assert strategy == StrategyType.NO_TRADE
    assert any("does not exceed no-trade score" in reason for reason in reasons)


def test_classify_regime_tradability_uses_context_metadata_first() -> None:
    regime = _bearish_regime(
        tradability_class=REGIME_TRADABILITY_LOW_EDGE,
        tradability_reason="Context is valid but open space is limited.",
    )
    tradability, reason = classify_regime_tradability(regime)
    assert tradability == REGIME_TRADABILITY_LOW_EDGE
    assert "open space is limited" in reason


def test_classify_regime_tradability_returns_tradable_for_explicit_context() -> None:
    regime = _bearish_regime(
        tradability_class=REGIME_TRADABILITY_TRADABLE,
        tradability_reason="Transition bearish failure is tradeable.",
    )
    tradability, reason = classify_regime_tradability(regime)
    assert tradability == REGIME_TRADABILITY_TRADABLE
    assert "tradeable" in reason.lower()


def test_select_strategy_allows_directional_balance_bearish_context() -> None:
    regime = _bearish_regime(
        market_state="DIRECTIONAL_BALANCE",
        bearish_trade_score=5.8,
        no_trade_score=4.7,
        trade_score_margin_bearish=0.8,
        bearish_trade_score_threshold=5.45,
    )
    strategy, _ = select_strategy(regime, time(11, 15), allowed_playbook_tiers=("A", "B"))
    assert strategy == StrategyType.BEAR_CALL_CREDIT_SPREAD


def test_select_strategy_allows_bearish_rejection_transition_context() -> None:
    regime = _bearish_regime(
        day_archetype="SIDEWAYS_TO_BEARISH",
        playbook="SIDEWAYS_TO_BEARISH_REJECTION",
        bearish_setup="PULLBACK_REJECTION",
        market_state="TRANSITION",
        failure_type="BEARISH_REJECTION_TRANSITION",
        bearish_failure_score=4.5,
        bearish_trade_score=6.4,
        no_trade_score=0.0,
        trade_score_margin_bearish=1.0,
        bearish_trade_score_threshold=6.1,
        bearish_upper_wick_fraction=0.60,
        bearish_close_location=0.40,
        bearish_candle_quality_score=0.60,
    )
    strategy, reasons = select_strategy(regime, time(12, 10), allowed_playbook_tiers=("A", "B"))
    assert strategy == StrategyType.BEAR_CALL_CREDIT_SPREAD
    assert any("Bear Call Credit Spread" in reason for reason in reasons)
