from __future__ import annotations

from market_ai.modules.analytics.intraday_trade_planner import (
    build_higher_tf_confluence,
    build_trade_plan,
)


def test_higher_tf_confluence_uses_weighted_bias_score() -> None:
    out = build_higher_tf_confluence(
        snap_15m={"pattern": "DOWNTREND", "ema_alignment": "BEARISH", "ema_20_50_alignment": "BEARISH"},
        snap_60m={"pattern": "UPTREND", "ema_alignment": "NEUTRAL", "ema_20_50_alignment": "NEUTRAL"},
        market_bias="BEARISH",
        rsi_divergence={"bias": "BEARISH"},
    )
    assert out["weighted_ok"] is True
    assert float(out["bias_score"]) > 0.0
    assert out["status"] == "PARTIAL"


def test_trade_plan_accepts_atr_tolerant_pullback_near_zone() -> None:
    htf = build_higher_tf_confluence(
        snap_15m={"pattern": "DOWNTREND", "ema_alignment": "BEARISH", "ema_20_50_alignment": "BEARISH"},
        snap_60m={"pattern": "DOWNTREND", "ema_alignment": "BEARISH", "ema_20_50_alignment": "BEARISH"},
        market_bias="BEARISH",
        rsi_divergence={"bias": "BEARISH", "status": "BEARISH_DIVERGENCE", "description": "Bearish divergence"},
    )
    plan = build_trade_plan(
        spot=25550.0,
        snap_5m={
            "primary_pattern": "DOUBLE_TOP_M_CONFIRMED",
            "price_action_confirmation": "CANDLE_AND_RETEST_CONFIRMED",
            "retest_status": "RESISTANCE_HOLD",
            "ema_slow": 25610.0,
            "ema_base": 25645.0,
            "ema_20_50_alignment": "BEARISH",
            "atr_like_points": 145.0,
            "pattern": "DOWNTREND",
            "strong_candle_bias": "BEARISH",
            "strong_candle_confirmed": True,
            "strong_candle_status": "STRONG_RED_5M",
            "strong_candle_description": "Strong bearish 5m candle closed near the low after the pullback.",
        },
        snap_15m={
            "pattern": "DOWNTREND",
            "resistance": 25640.0,
            "atr_like_points": 145.0,
            "volatility_regime": "NORMAL",
            "ema_alignment": "BEARISH",
            "ema_20_50_alignment": "BEARISH",
        },
        snap_60m={"pattern": "DOWNTREND", "ema_alignment": "BEARISH", "ema_20_50_alignment": "BEARISH"},
        market_bias="BEARISH",
        bias_confidence=0.82,
        orb={"orb_high": 25635.0},
        nearest_support={"price": 24950.0, "side": "support", "timeframe": "intraday"},
        nearest_resistance={"price": 25640.0, "side": "resistance", "timeframe": "intraday"},
        advisor={
            "pcr_bias": "BEARISH",
            "oi_build_bias": "BEARISH_RESISTANCE",
            "market_context": {"option_chain": {"call_wall_above": {"strike": 25750.0}, "put_wall_below": {"strike": 24900.0}}},
        },
        buyer_seller_zones={
            "buyers": {"active": False},
            "sellers": {"active": True, "summary": "Sellers are rejecting near 25640."},
        },
        rsi_divergence={"bias": "BEARISH", "status": "BEARISH_DIVERGENCE", "description": "Bearish divergence"},
        higher_tf_confluence=htf,
    )
    assert plan["setup_type"] == "CALL_CREDIT_SPREAD"
    assert plan["pullback_touched"] is False
    assert plan["pullback_ok"] is True
    assert float(plan["pullback_score"]) > 0.0
    assert "PULLBACK_ZONE_NOT_REACHED" not in (plan.get("entry_block_reasons") or [])
    assert plan["ema_2050_ok"] is True
    assert plan["strong_candle_ok"] is True
    assert plan["entry_ready"] is True


def test_trade_plan_uses_fvg_and_fib_for_bullish_pullback_confirmation() -> None:
    htf = build_higher_tf_confluence(
        snap_15m={"pattern": "UPTREND", "ema_alignment": "BULLISH", "ema_20_50_alignment": "BULLISH"},
        snap_60m={"pattern": "UPTREND", "ema_alignment": "BULLISH", "ema_20_50_alignment": "BULLISH"},
        market_bias="BULLISH",
        rsi_divergence={"bias": "BULLISH", "status": "BULLISH_DIVERGENCE", "description": "Bullish divergence"},
    )
    plan = build_trade_plan(
        spot=25510.0,
        snap_5m={
            "primary_pattern": "BREAKOUT_CONTINUATION_UP",
            "price_action_confirmation": "CANDLE_CONFIRMED",
            "retest_status": "BREAKOUT_RETEST_HOLD",
            "ema_slow": 25490.0,
            "ema_base": 25445.0,
            "ema_20_50_alignment": "BULLISH",
            "atr_like_points": 120.0,
            "pattern": "UPTREND",
            "strong_candle_bias": "BULLISH",
            "strong_candle_confirmed": True,
            "strong_candle_status": "STRONG_GREEN_5M",
            "strong_candle_description": "Strong bullish 5m candle closed near the high after the pullback.",
            "fair_value_gap_bias": "BULLISH",
            "fair_value_gap_status": "BULLISH_FVG_ACTIVE",
            "fair_value_gap_low": 25480.0,
            "fair_value_gap_high": 25525.0,
            "fair_value_gap_active": True,
            "fair_value_gap_description": "Bullish FVG sits around 25480-25525.",
            "fib_bullish_zone_low": 25460.0,
            "fib_bullish_zone_high": 25540.0,
            "fib_bullish_active": True,
            "fib_description": "Fib pullback zone is 25460-25540 for bullish continuation.",
        },
        snap_15m={
            "pattern": "UPTREND",
            "support": 25500.0,
            "atr_like_points": 120.0,
            "volatility_regime": "NORMAL",
            "ema_alignment": "BULLISH",
            "ema_20_50_alignment": "BULLISH",
        },
        snap_60m={"pattern": "UPTREND", "ema_alignment": "BULLISH", "ema_20_50_alignment": "BULLISH"},
        market_bias="BULLISH",
        bias_confidence=0.84,
        orb={"orb_low": 25495.0},
        nearest_support={"price": 25500.0, "side": "support", "timeframe": "intraday"},
        nearest_resistance={"price": 26120.0, "side": "resistance", "timeframe": "intraday"},
        advisor={
            "pcr_bias": "BULLISH",
            "oi_build_bias": "BULLISH_SUPPORT",
            "market_context": {"option_chain": {"put_wall_below": {"strike": 25350.0}, "call_wall_above": {"strike": 26150.0}}},
        },
        buyer_seller_zones={
            "buyers": {"active": False, "summary": "No clear buyer zone yet."},
            "sellers": {"active": False},
        },
        rsi_divergence={"bias": "BULLISH", "status": "BULLISH_DIVERGENCE", "description": "Bullish divergence"},
        higher_tf_confluence=htf,
    )
    assert plan["setup_type"] == "PUT_CREDIT_SPREAD"
    assert plan["fvg_supportive"] is True
    assert plan["fib_supportive"] is True
    assert plan["buyer_seller_ok"] is True
    assert plan["ema_2050_ok"] is True
    assert plan["strong_candle_ok"] is True
    assert "BUYER_ZONE_UNCONFIRMED" not in (plan.get("entry_block_reasons") or [])
    assert plan["entry_ready"] is True


def test_trade_plan_allows_strong_bearish_trend_continuation_without_exact_pullback_touch() -> None:
    htf = build_higher_tf_confluence(
        snap_15m={"pattern": "DOWNTREND", "ema_alignment": "BEARISH", "ema_20_50_alignment": "BEARISH"},
        snap_60m={"pattern": "DOWNTREND", "ema_alignment": "BEARISH", "ema_20_50_alignment": "BEARISH"},
        market_bias="BEARISH",
        rsi_divergence={"bias": "BEARISH", "status": "BEARISH_DIVERGENCE", "description": "Bearish divergence"},
    )
    plan = build_trade_plan(
        spot=25440.0,
        snap_5m={
            "primary_pattern": "BREAKOUT_CONTINUATION_DOWN",
            "price_action_confirmation": "BREAKOUT_CONFIRMED",
            "retest_status": "BREAKDOWN_RETEST_HOLD",
            "ema_slow": 25520.0,
            "ema_base": 25580.0,
            "ema_20_50_alignment": "BEARISH",
            "atr_like_points": 120.0,
            "pattern": "DOWNTREND",
            "strong_candle_bias": "BEARISH",
            "strong_candle_confirmed": True,
            "strong_candle_status": "STRONG_RED_5M",
            "strong_candle_description": "Strong bearish 5m candle closed near the low after the pullback.",
        },
        snap_15m={
            "pattern": "DOWNTREND",
            "resistance": 25620.0,
            "atr_like_points": 120.0,
            "volatility_regime": "NORMAL",
            "ema_alignment": "BEARISH",
            "ema_20_50_alignment": "BEARISH",
        },
        snap_60m={"pattern": "DOWNTREND", "ema_alignment": "BEARISH", "ema_20_50_alignment": "BEARISH"},
        market_bias="BEARISH",
        bias_confidence=0.86,
        orb={"orb_high": 25600.0},
        nearest_support={"price": 25100.0, "side": "support", "timeframe": "intraday"},
        nearest_resistance={"price": 25630.0, "side": "resistance", "timeframe": "intraday"},
        advisor={
            "pcr_bias": "BEARISH",
            "oi_build_bias": "BEARISH_RESISTANCE",
            "market_context": {"option_chain": {"call_wall_above": {"strike": 25750.0}}},
        },
        buyer_seller_zones={
            "buyers": {"active": False},
            "sellers": {"active": True, "summary": "Sellers keep defending lower highs."},
        },
        rsi_divergence={"bias": "BEARISH", "status": "BEARISH_DIVERGENCE", "description": "Bearish divergence"},
        higher_tf_confluence=htf,
    )
    assert plan["continuation_ready"] is True
    assert plan["entry_ready"] is True
    assert plan["posture"] == "READY_TREND_CONTINUATION"
    assert "PULLBACK_ZONE_NOT_REACHED" not in (plan.get("entry_block_reasons") or [])


def test_trade_plan_requires_ema2050_and_strong_candle_for_pullback_entry() -> None:
    htf = build_higher_tf_confluence(
        snap_15m={"pattern": "UPTREND", "ema_alignment": "BULLISH", "ema_20_50_alignment": "BULLISH"},
        snap_60m={"pattern": "UPTREND", "ema_alignment": "BULLISH", "ema_20_50_alignment": "BULLISH"},
        market_bias="BULLISH",
        rsi_divergence={"bias": "BULLISH", "status": "BULLISH_DIVERGENCE", "description": "Bullish divergence"},
    )
    plan = build_trade_plan(
        spot=25510.0,
        snap_5m={
            "primary_pattern": "BREAKOUT_CONTINUATION_UP",
            "price_action_confirmation": "CANDLE_CONFIRMED",
            "retest_status": "BREAKOUT_RETEST_HOLD",
            "ema_slow": 25490.0,
            "ema_base": 25520.0,
            "ema_20_50_alignment": "NEUTRAL",
            "atr_like_points": 120.0,
            "pattern": "UPTREND",
            "strong_candle_bias": "NEUTRAL",
            "strong_candle_confirmed": False,
            "strong_candle_status": "NONE",
            "strong_candle_description": "No strong 5m confirmation candle.",
            "fair_value_gap_bias": "BULLISH",
            "fair_value_gap_status": "BULLISH_FVG_ACTIVE",
            "fair_value_gap_low": 25480.0,
            "fair_value_gap_high": 25525.0,
            "fair_value_gap_active": True,
            "fib_bullish_zone_low": 25460.0,
            "fib_bullish_zone_high": 25540.0,
            "fib_bullish_active": True,
        },
        snap_15m={
            "pattern": "UPTREND",
            "support": 25500.0,
            "atr_like_points": 120.0,
            "volatility_regime": "NORMAL",
            "ema_alignment": "BULLISH",
            "ema_20_50_alignment": "BULLISH",
        },
        snap_60m={"pattern": "UPTREND", "ema_alignment": "BULLISH", "ema_20_50_alignment": "BULLISH"},
        market_bias="BULLISH",
        bias_confidence=0.84,
        orb={"orb_low": 25495.0},
        nearest_support={"price": 25500.0, "side": "support", "timeframe": "intraday"},
        nearest_resistance={"price": 26120.0, "side": "resistance", "timeframe": "intraday"},
        advisor={
            "pcr_bias": "BULLISH",
            "oi_build_bias": "BULLISH_SUPPORT",
            "market_context": {"option_chain": {"put_wall_below": {"strike": 25350.0}, "call_wall_above": {"strike": 26150.0}}},
        },
        buyer_seller_zones={"buyers": {"active": True}, "sellers": {"active": False}},
        rsi_divergence={"bias": "BULLISH", "status": "BULLISH_DIVERGENCE", "description": "Bullish divergence"},
        higher_tf_confluence=htf,
    )
    assert plan["entry_ready"] is False
    assert "EMA2050_CONFIRMATION_MISSING" in (plan.get("entry_block_reasons") or [])
    assert "STRONG_5M_CANDLE_MISSING" in (plan.get("entry_block_reasons") or [])


def test_trade_plan_allows_bullish_continuation_near_zone_without_extreme_candle_when_confluence_is_strong() -> None:
    htf = build_higher_tf_confluence(
        snap_15m={"pattern": "UPTREND", "ema_alignment": "BULLISH", "ema_20_50_alignment": "BULLISH"},
        snap_60m={"pattern": "UPTREND", "ema_alignment": "BULLISH", "ema_20_50_alignment": "BULLISH"},
        market_bias="BULLISH",
        rsi_divergence={"bias": "BULLISH", "status": "BULLISH_DIVERGENCE", "description": "Bullish divergence"},
    )
    plan = build_trade_plan(
        spot=25578.0,
        snap_5m={
            "primary_pattern": "BREAKOUT_CONTINUATION_UP",
            "price_action_confirmation": "CANDLE_CONFIRMED",
            "retest_status": "BREAKOUT_RETEST_HOLD",
            "ema_slow": 25498.0,
            "ema_base": 25460.0,
            "ema_20_50_alignment": "BULLISH",
            "atr_like_points": 120.0,
            "pattern": "UPTREND",
            "strong_candle_bias": "NEUTRAL",
            "strong_candle_confirmed": False,
            "strong_candle_status": "NONE",
            "strong_candle_description": "No extreme 5m candle yet, but buyers are holding continuation.",
            "fair_value_gap_bias": "BULLISH",
            "fair_value_gap_status": "BULLISH_FVG_ACTIVE",
            "fair_value_gap_low": 25480.0,
            "fair_value_gap_high": 25520.0,
            "fair_value_gap_active": True,
            "fib_bullish_zone_low": 25470.0,
            "fib_bullish_zone_high": 25530.0,
            "fib_bullish_active": True,
        },
        snap_15m={
            "pattern": "UPTREND",
            "support": 25500.0,
            "atr_like_points": 120.0,
            "volatility_regime": "NORMAL",
            "ema_alignment": "BULLISH",
            "ema_20_50_alignment": "BULLISH",
        },
        snap_60m={"pattern": "UPTREND", "ema_alignment": "BULLISH", "ema_20_50_alignment": "BULLISH"},
        market_bias="BULLISH",
        bias_confidence=0.86,
        orb={"orb_low": 25495.0},
        nearest_support={"price": 25500.0, "side": "support", "timeframe": "intraday"},
        nearest_resistance={"price": 26080.0, "side": "resistance", "timeframe": "intraday"},
        advisor={
            "pcr_bias": "BULLISH",
            "oi_build_bias": "BULLISH_SUPPORT",
            "market_context": {"option_chain": {"put_wall_below": {"strike": 25350.0}, "call_wall_above": {"strike": 26150.0}}},
        },
        buyer_seller_zones={"buyers": {"active": True, "summary": "Buyers keep defending the shallow pullback."}, "sellers": {"active": False}},
        rsi_divergence={"bias": "BULLISH", "status": "BULLISH_DIVERGENCE", "description": "Bullish divergence"},
        higher_tf_confluence=htf,
    )
    assert plan["continuation_ready"] is True
    assert plan["entry_ready"] is True
    assert "STRONG_5M_CANDLE_MISSING" not in (plan.get("entry_block_reasons") or [])
