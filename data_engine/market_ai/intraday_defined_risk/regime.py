from __future__ import annotations

from datetime import time

from .data_models import AdaptiveParameters, MarketSnapshot, RegimeLabel, RegimeState, ValidationError
from .features import (
    RANGE_GATE_TIME,
    adaptive_or_length_minutes,
    balanced_range_condor_setup,
    bearish_candle_context,
    bearish_failed_reclaim_setup,
    bearish_pullback_rejection_setup,
    bearish_shallow_continuation_setup,
    bearish_tight_breakdown_setup,
    bullish_candle_context,
    bullish_pullback_reclaim_setup,
    bullish_shallow_continuation_setup,
    bullish_vwap_hold_higher_low_setup,
    closes,
    close_location_value,
    compute_opening_range,
    compute_vwap,
    ema,
    ema_distance_pct,
    ema_value,
    ema_slope,
    gap_down_continuation_setup,
    gap_down_recovery_setup,
    gap_up_continuation_setup,
    gap_up_failure_reversal_setup,
    last_n_closes_above,
    last_n_closes_below,
    nearest_resistance,
    nearest_support,
    opening_gap_pct,
    opening_range_whipsaw,
    option_chain_pressure,
    option_chain_oi_flow,
    option_chain_wall_migration,
    oscillates_around_vwap,
    market_structure_state,
    open_drive_bullish_reclaim_setup,
    price_change_pct,
    recent_higher_lows,
    recent_lower_highs,
    realized_volatility_pct,
    remains_inside_opening_range,
    rolling_window_bars,
    session_bars,
    sideways_bullish_reclaim_setup,
)


def _round_level(level: float | None) -> float | None:
    if level is None:
        return None
    return round(float(level), 2)


def _first_present(*levels: float | None) -> float | None:
    for level in levels:
        if level is not None:
            return float(level)
    return None


def _structure_signal(metadata: dict[str, float | str | bool | None]) -> str:
    if bool(metadata.get("bullish_choch")):
        return "BULLISH_CHOCH"
    if bool(metadata.get("bearish_choch")):
        return "BEARISH_CHOCH"
    if bool(metadata.get("bullish_bos")):
        return "BULLISH_BOS"
    if bool(metadata.get("bearish_bos")):
        return "BEARISH_BOS"
    return "BALANCED"


def _build_trade_plan(
    *,
    metadata: dict[str, float | str | bool | None],
    regime: RegimeLabel,
    opening_range_high: float | None,
    opening_range_low: float | None,
    vwap: float | None,
    support_ref: float | None,
    resistance_ref: float | None,
    bullish_support_anchor: float | None,
    bearish_resistance_anchor: float | None,
) -> dict[str, object]:
    gap_pct = float(metadata.get("opening_gap_pct") or 0.0)
    if bool(metadata.get("big_gap_up")):
        market_open_context = "BIG_GAP_UP"
    elif bool(metadata.get("big_gap_down")):
        market_open_context = "BIG_GAP_DOWN"
    elif gap_pct >= 0.25:
        market_open_context = "GAP_UP"
    elif gap_pct <= -0.25:
        market_open_context = "GAP_DOWN"
    else:
        market_open_context = "FLAT_OPEN"

    playbook = str(metadata.get("playbook") or "NO_TRADE")
    structure_signal = _structure_signal(metadata)
    bullish_fvg_context = "BULLISH_FVG_ACTIVE" if bool(metadata.get("bullish_fvg_active")) else "NONE"
    bearish_fvg_context = "BEARISH_FVG_ACTIVE" if bool(metadata.get("bearish_fvg_active")) else "NONE"
    bullish_ob_context = "BULLISH_ORDER_BLOCK_ACTIVE" if bool(metadata.get("bullish_order_block_active")) else "NONE"
    bearish_ob_context = "BEARISH_ORDER_BLOCK_ACTIVE" if bool(metadata.get("bearish_order_block_active")) else "NONE"
    candle_pattern = "NONE"
    confluence_score = 0.0

    if regime == RegimeLabel.UP_TREND:
        entry_zone_low = _first_present(
            metadata.get("bullish_order_block_low"),
            metadata.get("bullish_fvg_low"),
            bullish_support_anchor,
            opening_range_high,
            vwap,
            support_ref,
        )
        entry_zone_high = _first_present(
            metadata.get("bullish_order_block_high"),
            metadata.get("bullish_fvg_high"),
            bullish_support_anchor,
            opening_range_high,
            vwap,
            support_ref,
        )
        invalidation_level = _first_present(
            metadata.get("bullish_order_block_low"),
            metadata.get("bullish_fvg_low"),
            opening_range_high,
            vwap,
            support_ref,
        )
        target_level = _first_present(
            resistance_ref,
            metadata.get("current_call_wall"),
        )
        execution_plan = "SELL_BULL_PUT_ON_RECLAIM"
        management_plan = "Hold while reclaimed support, VWAP, and bullish structure remain intact."
        thesis = (
            f"{metadata.get('day_archetype')} with {playbook}: sell put spread below reclaimed support once trend continuation is confirmed."
        )
        fvg_context = bullish_fvg_context
        order_block_context = bullish_ob_context
        candle_pattern = str(metadata.get("bullish_candle_pattern") or "NONE")
        confluence_score = float(metadata.get("bullish_confluence_score") or 0.0)
    elif regime == RegimeLabel.DOWN_TREND:
        entry_zone_low = _first_present(
            metadata.get("bearish_order_block_low"),
            metadata.get("bearish_fvg_low"),
            bearish_resistance_anchor,
            opening_range_low,
            vwap,
            resistance_ref,
        )
        entry_zone_high = _first_present(
            metadata.get("bearish_order_block_high"),
            metadata.get("bearish_fvg_high"),
            bearish_resistance_anchor,
            opening_range_low,
            vwap,
            resistance_ref,
        )
        invalidation_level = _first_present(
            metadata.get("bearish_order_block_high"),
            metadata.get("bearish_fvg_high"),
            opening_range_low,
            vwap,
            resistance_ref,
        )
        target_level = _first_present(
            support_ref,
            metadata.get("current_put_wall"),
        )
        execution_plan = "SELL_BEAR_CALL_ON_REJECTION"
        management_plan = "Hold while rejection below resistance, EMA20, and VWAP remains valid."
        thesis = (
            f"{metadata.get('day_archetype')} with {playbook}: sell call spread above failed reclaim / rejection zone once downside continuation is confirmed."
        )
        fvg_context = bearish_fvg_context
        order_block_context = bearish_ob_context
        candle_pattern = str(metadata.get("bearish_candle_pattern") or "NONE")
        confluence_score = float(metadata.get("bearish_confluence_score") or 0.0)
    elif regime == RegimeLabel.RANGE:
        range_mid = None
        if opening_range_high is not None and opening_range_low is not None:
            range_mid = (opening_range_high + opening_range_low) / 2.0
        entry_zone_low = opening_range_low
        entry_zone_high = opening_range_high
        invalidation_level = _first_present(opening_range_low, vwap)
        target_level = _first_present(range_mid, vwap)
        execution_plan = "SELL_BALANCED_CONDOR"
        management_plan = "Hold while price remains balanced inside the opening range and VWAP."
        thesis = "Balanced range session: sell a neutral condor only while the range remains intact and option pressure stays neutral."
        fvg_context = "NONE"
        order_block_context = "NONE"
        confluence_score = float(max(metadata.get("bullish_confluence_score") or 0.0, metadata.get("bearish_confluence_score") or 0.0))
    else:
        entry_zone_low = None
        entry_zone_high = None
        invalidation_level = None
        target_level = None
        execution_plan = "STAND_ASIDE"
        management_plan = "Wait for a clearer structure or a dedicated playbook trigger."
        thesis = "Structure is not aligned with an active playbook. Preserve capital and wait."
        fvg_context = bullish_fvg_context if bullish_fvg_context != "NONE" else bearish_fvg_context
        order_block_context = bullish_ob_context if bullish_ob_context != "NONE" else bearish_ob_context
        candle_pattern = str(
            metadata.get("bullish_candle_pattern")
            or metadata.get("bearish_candle_pattern")
            or "NONE"
        )
        confluence_score = float(max(metadata.get("bullish_confluence_score") or 0.0, metadata.get("bearish_confluence_score") or 0.0))

    return {
        "scenario": str(metadata.get("day_archetype") or "UNCLASSIFIED"),
        "playbook": playbook,
        "market_open_context": market_open_context,
        "structure_signal": structure_signal,
        "reversal_context": structure_signal.endswith("CHOCH"),
        "liquidity_bias": str(metadata.get("smart_money_bias") or "UNKNOWN"),
        "candle_pattern": candle_pattern,
        "confluence_score": round(confluence_score, 4),
        "fvg_context": fvg_context,
        "order_block_context": order_block_context,
        "entry_zone_low": _round_level(entry_zone_low),
        "entry_zone_high": _round_level(entry_zone_high),
        "invalidation_level": _round_level(invalidation_level),
        "target_level": _round_level(target_level),
        "execution_plan": execution_plan,
        "management_plan": management_plan,
        "thesis": thesis,
    }


def classify_regime(snapshot: MarketSnapshot, params: AdaptiveParameters | None = None) -> RegimeState:
    params = (params or AdaptiveParameters()).clamped()
    try:
        snapshot.validate()
    except ValidationError as exc:
        return RegimeState(
            regime=RegimeLabel.NO_TRADE,
            trend_15m="INVALID_INPUT",
            execution_5m="INVALID_INPUT",
            ema20_15m=None,
            ema20_slope_15m=None,
            rv30_pct=0.0,
            or_length_minutes=None,
            opening_range_high=None,
            opening_range_low=None,
            vwap=None,
            confidence=0.0,
            reasons=[str(exc)],
        )

    bars_15m = session_bars(snapshot.nifty_15m)
    bars_5m = session_bars(snapshot.nifty_5m)
    close_15m = closes(bars_15m)
    close_5m = closes(bars_5m)
    ema20_series = ema(close_15m, period=20)
    ema20_value = ema20_series[-1] if ema20_series else None
    ema20_slope_value = ema_slope(close_15m, period=20, lookback=3)
    ema20_5m_value = ema_value(close_5m, period=20)
    ema50_5m_value = ema_value(close_5m, period=50)
    trend_15m = "NEUTRAL"
    reasons: list[str] = []
    confidence = 0.0
    spot = snapshot.option_chain.spot
    hour_bars = rolling_window_bars(bars_5m, 12)
    hour_start = hour_bars[0].open if hour_bars else spot
    hour_change_pct = price_change_pct(hour_start, spot)
    support_5m = nearest_support(bars_5m, spot, count=1)
    resistance_5m = nearest_resistance(bars_5m, spot, count=1)
    support_15m = nearest_support(bars_15m, spot, count=1)
    resistance_15m = nearest_resistance(bars_15m, spot, count=1)
    option_pressure = option_chain_pressure(spot, snapshot.option_chain.quotes)
    oi_flow = option_chain_oi_flow(
        spot,
        snapshot.option_chain.quotes,
        snapshot.previous_option_chain.quotes if snapshot.previous_option_chain is not None else None,
    )
    wall_migration = option_chain_wall_migration(
        spot,
        snapshot.option_chain.quotes,
        snapshot.previous_option_chain.quotes if snapshot.previous_option_chain is not None else None,
    )
    support_ref = support_5m[0] if support_5m else (support_15m[0] if support_15m else None)
    resistance_ref = resistance_5m[0] if resistance_5m else (resistance_15m[0] if resistance_15m else None)
    session_open = bars_5m[0].open if bars_5m else None
    gap_pct = opening_gap_pct(snapshot.previous_session_close, session_open)
    metadata: dict[str, float | str | bool | None] = {
        "day_archetype": "UNCLASSIFIED",
        "opening_gap_pct": gap_pct,
        "big_gap_up": False,
        "big_gap_down": False,
        "ema20_5m": ema20_5m_value,
        "ema50_5m": ema50_5m_value,
        "ema_distance_pct_5m": ema_distance_pct(ema20_5m_value, ema50_5m_value, spot),
        "last_hour_change_pct": hour_change_pct,
        "support_5m": support_5m[0] if support_5m else None,
        "resistance_5m": resistance_5m[0] if resistance_5m else None,
        "support_15m": support_15m[0] if support_15m else None,
        "resistance_15m": resistance_15m[0] if resistance_15m else None,
        "bullish_chain_pressure": option_pressure["bullish_pressure"],
        "bearish_chain_pressure": option_pressure["bearish_pressure"],
        "bullish_flow_score": oi_flow["bullish_flow_score"],
        "bearish_flow_score": oi_flow["bearish_flow_score"],
        "smart_money_bias": oi_flow["smart_money_bias"],
        "put_support_strike": oi_flow["put_support_strike"],
        "call_resistance_strike": oi_flow["call_resistance_strike"],
        "put_support_oi_change": oi_flow["put_support_oi_change"],
        "call_resistance_oi_change": oi_flow["call_resistance_oi_change"],
        "current_put_wall": wall_migration["current_put_wall"],
        "previous_put_wall": wall_migration["previous_put_wall"],
        "put_wall_shift": wall_migration["put_wall_shift"],
        "current_call_wall": wall_migration["current_call_wall"],
        "previous_call_wall": wall_migration["previous_call_wall"],
        "call_wall_shift": wall_migration["call_wall_shift"],
        "bullish_wall_score": wall_migration["bullish_wall_score"],
        "bearish_wall_score": wall_migration["bearish_wall_score"],
        "wall_migration_bias": wall_migration["wall_migration_bias"],
        "bullish_entry_ready": False,
        "bullish_setup": None,
        "bullish_support_quality_score": 0.0,
        "bearish_entry_ready": False,
        "bearish_setup": None,
        "playbook": "NO_TRADE",
        "min_short_call_strike": None,
        "max_short_put_strike": None,
        "preferred_width_points": None,
        "allowed_width_points": None,
        "target_short_put_buffer_points": None,
        "minimum_net_edge_rupees": None,
        "trend_follow_ready_bullish": False,
        "trend_follow_ready_bearish": False,
        "bullish_trend_score": 0.0,
        "bearish_trend_score": 0.0,
        "bullish_entry_score": 0.0,
        "bearish_entry_score": 0.0,
        "bullish_candle_pattern": "NONE",
        "bearish_candle_pattern": "NONE",
        "bullish_candle_quality_score": 0.0,
        "bearish_candle_quality_score": 0.0,
        "bullish_location_score": 0.0,
        "bearish_location_score": 0.0,
        "bullish_structure_score": 0.0,
        "bearish_structure_score": 0.0,
        "bullish_confluence_score": 0.0,
        "bearish_confluence_score": 0.0,
        "bullish_planner_alignment": False,
        "bearish_planner_alignment": False,
        "early_sideways_bullish_ready": False,
        "gap_bullish_ready": False,
        "gap_bearish_ready": False,
    }

    if ema20_value is not None:
        latest_15m_close = close_15m[-1]
        trend_down_structure = recent_lower_highs(bars_15m, needed=2)
        trend_up_structure = recent_higher_lows(bars_15m, needed=2)
        fast_uptrend_override = (
            latest_15m_close > ema20_value
            and ema20_slope_value > 0
            and hour_change_pct >= 0.20
            and ema20_5m_value is not None
            and ema50_5m_value is not None
            and spot > ema20_5m_value > ema50_5m_value
        )
        fast_downtrend_override = (
            latest_15m_close < ema20_value
            and ema20_slope_value < 0
            and hour_change_pct <= -0.20
            and ema20_5m_value is not None
            and ema50_5m_value is not None
            and spot < ema20_5m_value < ema50_5m_value
        )
        if latest_15m_close < ema20_value and ema20_slope_value < 0 and (trend_down_structure or fast_downtrend_override):
            trend_15m = "TREND_DOWN"
            reasons.append(
                "15m price below EMA20 with negative slope and either lower swing highs or strong first-hour downside continuation."
            )
            confidence += 0.45
        elif latest_15m_close > ema20_value and ema20_slope_value > 0 and (trend_up_structure or fast_uptrend_override):
            trend_15m = "TREND_UP"
            reasons.append(
                "15m price above EMA20 with positive slope and either higher swing lows or strong first-hour upside continuation."
            )
            confidence += 0.45
        else:
            reasons.append("15m trend gate is neutral.")

    rv30_pct = realized_volatility_pct(bars_5m)
    or_length = adaptive_or_length_minutes(rv30_pct, params.rv_high_cutoff, params.rv_mid_cutoff)
    opening_range = None
    if snapshot.or_levels and or_length in snapshot.or_levels:
        opening_range = snapshot.or_levels[or_length]
    else:
        opening_range = compute_opening_range(bars_5m, or_length)

    vwap = snapshot.live_vwap if snapshot.live_vwap is not None else compute_vwap(bars_5m)
    execution_5m = "UNCONFIRMED"

    if opening_range and vwap:
        if snapshot.timestamp.time() >= RANGE_GATE_TIME and opening_range_whipsaw(bars_5m, opening_range):
            reasons.append("Opening-range whipsaw detected before 10:00; treating session as event risk.")
        elif last_n_closes_below(opening_range.low, bars_5m, n=2) and snapshot.option_chain.spot < vwap:
            execution_5m = "DOWN_CONFIRMED"
            reasons.append("Two consecutive 5m closes below OR low with spot below VWAP.")
            confidence += 0.30
        elif last_n_closes_above(opening_range.high, bars_5m, n=2) and snapshot.option_chain.spot > vwap:
            execution_5m = "UP_CONFIRMED"
            reasons.append("Two consecutive 5m closes above OR high with spot above VWAP.")
            confidence += 0.30
        elif (
            snapshot.timestamp.time() >= RANGE_GATE_TIME
            and rv30_pct < params.rv_mid_cutoff
            and remains_inside_opening_range(bars_5m, opening_range, required_bars=9)
            and oscillates_around_vwap(bars_5m, vwap, required_bars=9)
        ):
            execution_5m = "RANGE_CONFIRMED"
            reasons.append("Price remains inside opening range and oscillates around VWAP after 10:00.")
            confidence += 0.30
        else:
            reasons.append("5m execution trigger is not confirmed.")
    else:
        reasons.append("Opening range or VWAP could not be established.")

    structure_state = market_structure_state(bars_5m, ema20_5m_value, ema50_5m_value, vwap)
    metadata.update(structure_state)
    current_structure_signal = _structure_signal(metadata)
    metadata["structure_signal"] = current_structure_signal
    bullish_candle_state = bullish_candle_context(bars_5m)
    bearish_candle_state = bearish_candle_context(bars_5m)
    strong_bullish_patterns = {
        "BULLISH_EXPANSION",
        "BULLISH_ENGULFING",
        "BULLISH_BREAKDOWN_FAILURE",
        "BULLISH_MORNING_STAR",
        "BULLISH_HAMMER",
    }
    strong_bearish_patterns = {
        "BEARISH_EXPANSION",
        "BEARISH_ENGULFING",
        "BEARISH_BREAKOUT_FAILURE",
        "BEARISH_EVENING_STAR",
        "BEARISH_SHOOTING_STAR",
    }
    metadata["bullish_candle_pattern"] = str(bullish_candle_state["pattern"])
    metadata["bearish_candle_pattern"] = str(bearish_candle_state["pattern"])
    metadata["bullish_candle_quality_score"] = float(bullish_candle_state["quality_score"])
    metadata["bearish_candle_quality_score"] = float(bearish_candle_state["quality_score"])
    bullish_structure_score = 0.0
    bearish_structure_score = 0.0
    if bool(metadata.get("bullish_bos")):
        bullish_structure_score += 1.25
    if bool(metadata.get("bullish_choch")):
        bullish_structure_score += 1.0
    if bool(metadata.get("bullish_fvg_active")):
        bullish_structure_score += 0.75
    if bool(metadata.get("bullish_order_block_active")):
        bullish_structure_score += 0.75
    if bool(metadata.get("bullish_smc_alignment")):
        bullish_structure_score += 0.75
    if bool(metadata.get("bearish_bos")):
        bearish_structure_score += 1.25
    if bool(metadata.get("bearish_choch")):
        bearish_structure_score += 1.0
    if bool(metadata.get("bearish_fvg_active")):
        bearish_structure_score += 0.75
    if bool(metadata.get("bearish_order_block_active")):
        bearish_structure_score += 0.75
    if bool(metadata.get("bearish_smc_alignment")):
        bearish_structure_score += 0.75
    metadata["bullish_structure_score"] = bullish_structure_score
    metadata["bearish_structure_score"] = bearish_structure_score

    range_bias = (
        opening_range is not None
        and vwap is not None
        and snapshot.timestamp.time() >= RANGE_GATE_TIME
        and remains_inside_opening_range(bars_5m, opening_range, required_bars=6)
        and oscillates_around_vwap(bars_5m, vwap, required_bars=6)
    )
    big_gap_up = gap_pct >= 0.60
    big_gap_down = gap_pct <= -0.60
    metadata["big_gap_up"] = big_gap_up
    metadata["big_gap_down"] = big_gap_down
    open_drive_bullish = (
        gap_pct >= 0.25
        and opening_range is not None
        and vwap is not None
        and spot > vwap
        and spot > opening_range.high
        and last_n_closes_above(opening_range.high, bars_5m, n=2)
        and hour_change_pct >= 0.30
    )
    open_drive_bearish = (
        gap_pct <= -0.25
        and opening_range is not None
        and vwap is not None
        and spot < vwap
        and spot < opening_range.low
        and last_n_closes_below(opening_range.low, bars_5m, n=2)
        and hour_change_pct <= -0.30
    )
    sideways_to_bullish = (
        trend_15m == "TREND_UP"
        and execution_5m == "UP_CONFIRMED"
        and not open_drive_bullish
        and snapshot.timestamp.time() >= RANGE_GATE_TIME
        and (range_bias or abs(gap_pct) < 0.25 or hour_change_pct < 0.30)
    )
    sideways_to_bearish = (
        trend_15m == "TREND_DOWN"
        and execution_5m == "DOWN_CONFIRMED"
        and not open_drive_bearish
        and snapshot.timestamp.time() >= RANGE_GATE_TIME
        and (range_bias or abs(gap_pct) < 0.25 or hour_change_pct > -0.30)
    )

    regime = RegimeLabel.NO_TRADE
    bullish_trend_score = 0.0
    bearish_trend_score = 0.0
    if vwap is not None and ema20_5m_value is not None and ema50_5m_value is not None:
        if spot > vwap:
            bullish_trend_score += 1.0
        if spot < vwap:
            bearish_trend_score += 1.0
        if spot > ema20_5m_value > ema50_5m_value:
            bullish_trend_score += 1.5
        if spot < ema20_5m_value < ema50_5m_value:
            bearish_trend_score += 1.5
        if ema_distance_pct(ema20_5m_value, ema50_5m_value, spot) >= 0.10:
            bullish_trend_score += 0.5
        if ema_distance_pct(ema50_5m_value, ema20_5m_value, spot) >= 0.10:
            bearish_trend_score += 0.5
    if hour_change_pct >= 0.20:
        bullish_trend_score += 1.0
    if hour_change_pct <= -0.20:
        bearish_trend_score += 1.0
    if len(hour_bars) >= 6 and recent_higher_lows(hour_bars, needed=2):
        bullish_trend_score += 1.0
    if len(hour_bars) >= 6 and recent_lower_highs(hour_bars, needed=2):
        bearish_trend_score += 1.0
    if support_ref is not None and (spot - support_ref) / max(spot, 1.0) <= 0.004:
        bullish_trend_score += 0.5
    if resistance_ref is not None and (resistance_ref - spot) / max(spot, 1.0) <= 0.004:
        bearish_trend_score += 0.5
    if option_pressure["bullish_pressure"] >= 0.52:
        bullish_trend_score += 0.5
    if option_pressure["bearish_pressure"] >= 0.52:
        bearish_trend_score += 0.5
    if wall_migration["wall_migration_bias"] == "BULLISH":
        bullish_trend_score += 0.5
    elif wall_migration["wall_migration_bias"] == "BEARISH":
        bearish_trend_score += 0.5
    if oi_flow["smart_money_bias"] == "BULLISH":
        bearish_trend_score -= 0.50
    elif oi_flow["smart_money_bias"] == "BEARISH":
        bullish_trend_score -= 0.50
    metadata["bullish_trend_score"] = bullish_trend_score
    metadata["bearish_trend_score"] = bearish_trend_score
    metadata["trend_follow_ready_bullish"] = bullish_trend_score >= 3.5
    metadata["trend_follow_ready_bearish"] = bearish_trend_score >= 3.5
    bullish_support_anchor = max(
        [
            level
            for level in [
                support_ref,
                opening_range.high if opening_range is not None and spot >= opening_range.high else None,
            ]
            if level is not None
        ],
        default=None,
    )
    bearish_resistance_anchor = min(
        [
            level
            for level in [
                resistance_ref,
                opening_range.low if opening_range is not None and spot <= opening_range.low else None,
            ]
            if level is not None
        ],
        default=None,
    )
    bullish_location_score = 0.0
    bearish_location_score = 0.0
    if vwap is not None:
        if spot > vwap:
            bullish_location_score += 1.0
        if spot < vwap:
            bearish_location_score += 1.0
        if last_n_closes_above(vwap, bars_5m, n=2):
            bullish_location_score += 0.50
        if last_n_closes_below(vwap, bars_5m, n=2):
            bearish_location_score += 0.50
    if ema20_5m_value is not None:
        if spot > ema20_5m_value:
            bullish_location_score += 1.0
        if spot < ema20_5m_value:
            bearish_location_score += 1.0
        if last_n_closes_above(ema20_5m_value, bars_5m, n=2):
            bullish_location_score += 0.50
        if last_n_closes_below(ema20_5m_value, bars_5m, n=2):
            bearish_location_score += 0.50
    if bullish_support_anchor is not None and (spot - bullish_support_anchor) / max(spot, 1.0) <= 0.0045:
        bullish_location_score += 0.75
    if bearish_resistance_anchor is not None and (bearish_resistance_anchor - spot) / max(spot, 1.0) <= 0.0045:
        bearish_location_score += 0.75
    if opening_range is not None:
        if spot > opening_range.high:
            bullish_location_score += 0.35
        if spot < opening_range.low:
            bearish_location_score += 0.35
    metadata["bullish_location_score"] = round(bullish_location_score, 4)
    metadata["bearish_location_score"] = round(bearish_location_score, 4)
    bullish_confluence_score = (
        float(metadata["bullish_candle_quality_score"])
        + bullish_location_score
        + float(metadata["bullish_structure_score"])
        + float(metadata.get("bullish_flow_score") or 0.0)
        + float(metadata.get("bullish_wall_score") or 0.0)
    )
    bearish_confluence_score = (
        float(metadata["bearish_candle_quality_score"])
        + bearish_location_score
        + float(metadata["bearish_structure_score"])
        + float(metadata.get("bearish_flow_score") or 0.0)
        + float(metadata.get("bearish_wall_score") or 0.0)
    )
    metadata["bullish_confluence_score"] = round(bullish_confluence_score, 4)
    metadata["bearish_confluence_score"] = round(bearish_confluence_score, 4)
    if bullish_confluence_score >= 5.5:
        reasons.append(
            f"Bullish confluence is strong: candle pattern {metadata['bullish_candle_pattern']}, supportive location around EMA20/VWAP, and aligned structure/OI context."
        )
    if bearish_confluence_score >= 5.5:
        reasons.append(
            f"Bearish confluence is strong: candle pattern {metadata['bearish_candle_pattern']}, rejection below EMA20/VWAP, and aligned structure/OI context."
        )
    bullish_pullback = bullish_pullback_reclaim_setup(
        bars_5m,
        ema20_5m_value,
        ema50_5m_value,
        vwap,
        support_level=bullish_support_anchor,
    )
    bullish_shallow = bullish_shallow_continuation_setup(bars_5m, ema20_5m_value, ema50_5m_value, vwap)
    bullish_vwap_hold = bullish_vwap_hold_higher_low_setup(bars_5m, ema20_5m_value, ema50_5m_value, vwap)
    bearish_pullback = bearish_pullback_rejection_setup(
        bars_5m,
        ema20_5m_value,
        ema50_5m_value,
        vwap,
        resistance_level=bearish_resistance_anchor,
    )
    bearish_shallow = bearish_shallow_continuation_setup(bars_5m, ema20_5m_value, ema50_5m_value, vwap)
    bullish_entry_score = 0.0
    bearish_entry_score = 0.0
    if metadata["trend_follow_ready_bullish"]:
        bullish_entry_score += 1.5
    if bullish_pullback:
        bullish_entry_score += 1.5
    if bullish_shallow:
        bullish_entry_score += 1.0
    if bullish_vwap_hold:
        bullish_entry_score += 0.8
    if option_pressure["bullish_pressure"] >= 0.52:
        bullish_entry_score += 0.5
    if wall_migration["wall_migration_bias"] == "BULLISH":
        bullish_entry_score += 0.5
    if oi_flow["smart_money_bias"] == "BEARISH":
        bullish_entry_score -= 0.50
    if support_ref is not None and (spot - support_ref) / max(spot, 1.0) <= 0.004:
        bullish_entry_score += 0.5
    if metadata["trend_follow_ready_bearish"]:
        bearish_entry_score += 1.5
    if bearish_pullback:
        bearish_entry_score += 1.5
    if bearish_shallow:
        bearish_entry_score += 1.0
    if option_pressure["bearish_pressure"] >= 0.52:
        bearish_entry_score += 0.5
    if wall_migration["wall_migration_bias"] == "BEARISH":
        bearish_entry_score += 0.5
    if oi_flow["smart_money_bias"] == "BULLISH":
        bearish_entry_score -= 0.50
    if resistance_ref is not None and (resistance_ref - spot) / max(spot, 1.0) <= 0.004:
        bearish_entry_score += 0.5
    metadata["bullish_entry_score"] = bullish_entry_score
    metadata["bearish_entry_score"] = bearish_entry_score
    bullish_planner_alignment = (
        oi_flow["smart_money_bias"] == "BULLISH"
        or wall_migration["wall_migration_bias"] == "BULLISH"
        or (oi_flow["bullish_flow_score"] >= 0.55 and option_pressure["bullish_pressure"] >= 0.52)
    )
    bearish_planner_alignment = (
        oi_flow["smart_money_bias"] == "BEARISH"
        or wall_migration["wall_migration_bias"] == "BEARISH"
        or (oi_flow["bearish_flow_score"] >= 0.55 and option_pressure["bearish_pressure"] >= 0.52)
    )
    metadata["bullish_planner_alignment"] = bullish_planner_alignment
    metadata["bearish_planner_alignment"] = bearish_planner_alignment
    bullish_support_quality = 0.0
    if bullish_support_anchor is not None and spot > bullish_support_anchor:
        bullish_support_quality += 1.0
        if (spot - bullish_support_anchor) / max(spot, 1.0) <= 0.004:
            bullish_support_quality += 0.5
    if opening_range is not None and bullish_support_anchor is not None and abs(bullish_support_anchor - opening_range.high) <= 25.0:
        bullish_support_quality += 1.0
    if ema20_5m_value is not None and ema50_5m_value is not None and spot > ema20_5m_value > ema50_5m_value:
        bullish_support_quality += 0.5
    if bullish_planner_alignment:
        bullish_support_quality += 0.5
    if oi_flow["smart_money_bias"] == "BULLISH":
        bullish_support_quality += 0.5
    if wall_migration["wall_migration_bias"] == "BULLISH":
        bullish_support_quality += 0.5
    if bullish_pullback:
        bullish_support_quality += 0.5
    if bullish_vwap_hold:
        bullish_support_quality += 0.5
    metadata["bullish_support_quality_score"] = bullish_support_quality
    open_drive_bullish_reclaim = open_drive_bullish_reclaim_setup(
        bars_5m,
        opening_range,
        ema20_5m_value,
        ema50_5m_value,
        vwap,
    )
    gap_up_bullish_continuation = gap_up_continuation_setup(
        bars_5m,
        opening_range,
        ema20_5m_value,
        ema50_5m_value,
        vwap,
    )
    gap_down_bullish_recovery = gap_down_recovery_setup(
        bars_5m,
        opening_range,
        ema20_5m_value,
        ema50_5m_value,
        vwap,
    )
    gap_up_bearish_failure = gap_up_failure_reversal_setup(
        bars_5m,
        opening_range,
        ema20_5m_value,
        ema50_5m_value,
        vwap,
    )
    gap_down_bearish_continuation = gap_down_continuation_setup(
        bars_5m,
        opening_range,
        ema20_5m_value,
        ema50_5m_value,
        vwap,
    )
    sideways_bullish_reclaim = sideways_bullish_reclaim_setup(
        bars_5m,
        opening_range,
        ema20_5m_value,
        ema50_5m_value,
        vwap,
    )
    gap_bullish_alignment = bullish_planner_alignment or oi_flow["smart_money_bias"] == "BULLISH"
    gap_bearish_alignment = bearish_planner_alignment or oi_flow["smart_money_bias"] == "BEARISH"
    recent_gap_bars = bars_5m[-6:]
    recent_gap_close_location = close_location_value(
        recent_gap_bars[-1].close,
        min(bar.low for bar in recent_gap_bars),
        max(bar.high for bar in recent_gap_bars),
    ) if recent_gap_bars else 1.0
    gap_up_bullish_ready = (
        big_gap_up
        and trend_15m == "TREND_UP"
        and execution_5m == "UP_CONFIRMED"
        and gap_up_bullish_continuation
        and bullish_entry_score >= 3.3
        and bullish_trend_score >= 3.8
        and bullish_support_quality >= 4.0
        and gap_bullish_alignment
        and wall_migration["wall_migration_bias"] != "BEARISH"
    )
    gap_down_bullish_recovery_ready = (
        big_gap_down
        and trend_15m == "TREND_UP"
        and execution_5m == "UP_CONFIRMED"
        and gap_down_bullish_recovery
        and bullish_entry_score >= 3.4
        and bullish_trend_score >= 3.8
        and bullish_support_quality >= 4.0
        and oi_flow["smart_money_bias"] == "BULLISH"
        and wall_migration["wall_migration_bias"] != "BEARISH"
    )
    open_drive_bullish_ready = (
        open_drive_bullish
        and open_drive_bullish_reclaim
        and bullish_entry_score >= 3.0
        and bullish_trend_score >= 3.5
        and bullish_support_quality >= 4.0
        and (bullish_planner_alignment or bullish_support_quality >= 3.5)
    )
    sideways_bullish_reclaim_ready = (
        sideways_to_bullish
        and sideways_bullish_reclaim
        and bullish_entry_score >= 3.0
        and (bullish_planner_alignment or bullish_support_quality >= 3.5)
        and wall_migration["wall_migration_bias"] != "BEARISH"
    )
    sideways_bullish_shallow_ready = (
        sideways_to_bullish
        and bullish_shallow
        and bullish_entry_score >= 3.2
        and bullish_support_quality >= 4.0
        and (bullish_planner_alignment or option_pressure["bullish_pressure"] >= 0.55)
        and wall_migration["wall_migration_bias"] != "BEARISH"
        and (
            option_pressure["bullish_pressure"] >= 0.45
            or hour_change_pct >= 0.18
            or wall_migration["put_wall_shift"] > 0.0
        )
    )
    sideways_bullish_vwap_hold_ready = (
        sideways_to_bullish
        and bullish_vwap_hold
        and bullish_entry_score >= 3.1
        and bullish_support_quality >= 3.8
        and (
            bullish_planner_alignment
            or option_pressure["bullish_pressure"] >= 0.50
            or oi_flow["smart_money_bias"] == "BULLISH"
        )
        and wall_migration["wall_migration_bias"] != "BEARISH"
    )
    early_balance_bullish_reclaim_ready = (
        execution_5m == "UP_CONFIRMED"
        and trend_15m in {"TREND_UP", "NEUTRAL"}
        and time(10, 10) <= snapshot.timestamp.time() <= time(11, 15)
        and range_bias
        and not big_gap_up
        and not big_gap_down
        and bullish_pullback
        and bullish_entry_score >= 3.4
        and bullish_support_quality >= 4.0
        and bullish_planner_alignment
        and oi_flow["smart_money_bias"] == "BULLISH"
        and wall_migration["wall_migration_bias"] != "BEARISH"
        and current_structure_signal in {"BALANCED", "BULLISH_CHOCH", "BULLISH_BOS"}
    )
    recent_trend_bars = bars_5m[-6:]
    recent_trend_close_location = (
        close_location_value(
            recent_trend_bars[-1].close,
            min(bar.low for bar in recent_trend_bars),
            max(bar.high for bar in recent_trend_bars),
        )
        if recent_trend_bars
        else 0.0
    )
    afternoon_bullish_trend_hold_ready = (
        trend_15m == "TREND_UP"
        and execution_5m == "UP_CONFIRMED"
        and time(13, 0) <= snapshot.timestamp.time() <= time(14, 0)
        and metadata["trend_follow_ready_bullish"]
        and bullish_trend_score >= 4.5
        and bullish_entry_score >= 2.0
        and bullish_planner_alignment
        and option_pressure["bullish_pressure"] >= 0.55
        and wall_migration["wall_migration_bias"] != "BEARISH"
        and vwap is not None
        and ema20_5m_value is not None
        and ema50_5m_value is not None
        and len(recent_trend_bars) >= 6
        and all(bar.close > vwap for bar in recent_trend_bars)
        and sum(1 for bar in recent_trend_bars if bar.close > ema20_5m_value) >= 5
        and ema20_5m_value > ema50_5m_value
        and recent_trend_close_location >= 0.52
    )
    high_confluence_bullish_ready = (
        trend_15m == "NEUTRAL"
        and execution_5m == "UP_CONFIRMED"
        and time(10, 0) <= snapshot.timestamp.time() <= time(11, 15)
        and bullish_confluence_score >= 12.5
        and metadata["bullish_candle_pattern"] in strong_bullish_patterns
        and current_structure_signal in {"BULLISH_BOS", "BULLISH_CHOCH"}
        and bullish_planner_alignment
        and oi_flow["smart_money_bias"] == "BULLISH"
        and wall_migration["wall_migration_bias"] != "BEARISH"
        and (bullish_pullback or bullish_shallow or bullish_vwap_hold)
    )
    high_confluence_bearish_ready = (
        trend_15m == "NEUTRAL"
        and execution_5m == "DOWN_CONFIRMED"
        and time(10, 0) <= snapshot.timestamp.time() <= time(11, 15)
        and bearish_confluence_score >= 12.5
        and metadata["bearish_candle_pattern"] in strong_bearish_patterns
        and current_structure_signal in {"BEARISH_BOS", "BEARISH_CHOCH"}
        and bearish_planner_alignment
        and oi_flow["smart_money_bias"] == "BEARISH"
        and wall_migration["wall_migration_bias"] != "BULLISH"
        and (bearish_pullback or bearish_shallow)
    )
    wide_bullish_monetization_ready = (
        bullish_support_quality >= 4.5
        and bullish_planner_alignment
        and (
            option_pressure["bullish_pressure"] >= 0.50
            or hour_change_pct >= 0.28
            or wall_migration["bullish_wall_score"] >= 0.75
        )
    )
    metadata["gap_bullish_ready"] = bool(gap_up_bullish_ready or gap_down_bullish_recovery_ready)
    if high_confluence_bullish_ready:
        metadata["bullish_entry_ready"] = True
        metadata["bullish_setup"] = "HIGH_CONFLUENCE_CONTINUATION"
        metadata["playbook"] = "HIGH_CONFLUENCE_BULLISH_CONTINUATION"
        metadata["day_archetype"] = "HIGH_CONFLUENCE_BULLISH"
        metadata["preferred_width_points"] = 150.0 if wide_bullish_monetization_ready else 100.0
        metadata["allowed_width_points"] = (100.0, 150.0) if wide_bullish_monetization_ready else (100.0,)
        metadata["target_short_put_buffer_points"] = 25.0
        metadata["minimum_net_edge_rupees"] = 1200.0 if wide_bullish_monetization_ready else 1000.0
        reasons.append(
            "High-confluence bullish continuation confirmed: neutral 15m context gave way to a strong 5m upside expansion with aligned structure, location, and OI flow."
        )
    elif gap_up_bullish_ready:
        metadata["bullish_entry_ready"] = True
        metadata["bullish_setup"] = "GAP_CONTINUATION"
        metadata["playbook"] = "GAP_UP_BULLISH_CONTINUATION"
        metadata["preferred_width_points"] = 150.0 if wide_bullish_monetization_ready else 100.0
        metadata["allowed_width_points"] = (100.0, 150.0) if wide_bullish_monetization_ready else (100.0,)
        metadata["target_short_put_buffer_points"] = 35.0 if wide_bullish_monetization_ready else 30.0
        metadata["minimum_net_edge_rupees"] = 1250.0 if wide_bullish_monetization_ready else 950.0
    elif gap_down_bullish_recovery_ready:
        metadata["bullish_entry_ready"] = True
        metadata["bullish_setup"] = "GAP_RECOVERY"
        metadata["playbook"] = "GAP_DOWN_BULLISH_RECOVERY"
        metadata["preferred_width_points"] = 100.0
        metadata["allowed_width_points"] = (100.0,)
        metadata["target_short_put_buffer_points"] = 25.0
        metadata["minimum_net_edge_rupees"] = 900.0
    elif open_drive_bullish_ready:
        metadata["bullish_entry_ready"] = True
        metadata["bullish_setup"] = "OPEN_DRIVE_RECLAIM"
        metadata["playbook"] = "OPEN_DRIVE_BULLISH"
        metadata["preferred_width_points"] = 100.0
        metadata["allowed_width_points"] = (100.0, 150.0)
        metadata["target_short_put_buffer_points"] = 25.0
        metadata["minimum_net_edge_rupees"] = 950.0
    elif early_balance_bullish_reclaim_ready:
        metadata["bullish_entry_ready"] = True
        metadata["bullish_setup"] = "EARLY_BALANCE_RECLAIM"
        metadata["playbook"] = "EARLY_BALANCE_BULLISH_RECLAIM"
        metadata["preferred_width_points"] = 100.0
        metadata["allowed_width_points"] = (100.0,)
        metadata["target_short_put_buffer_points"] = 35.0
        metadata["minimum_net_edge_rupees"] = 950.0
    elif sideways_bullish_reclaim_ready:
        metadata["bullish_entry_ready"] = True
        metadata["bullish_setup"] = "PULLBACK_RECLAIM"
        metadata["playbook"] = "SIDEWAYS_TO_BULLISH_RECLAIM"
        metadata["preferred_width_points"] = 150.0 if wide_bullish_monetization_ready else 100.0
        metadata["allowed_width_points"] = (100.0, 150.0) if wide_bullish_monetization_ready else (100.0,)
        metadata["target_short_put_buffer_points"] = 50.0
        metadata["minimum_net_edge_rupees"] = 1100.0 if wide_bullish_monetization_ready else 800.0
    elif sideways_bullish_shallow_ready:
        metadata["bullish_entry_ready"] = True
        metadata["bullish_setup"] = "SHALLOW_CONTINUATION"
        metadata["playbook"] = "SIDEWAYS_TO_BULLISH_RECLAIM"
        metadata["preferred_width_points"] = 150.0 if wide_bullish_monetization_ready else 100.0
        metadata["allowed_width_points"] = (100.0, 150.0) if wide_bullish_monetization_ready else (100.0,)
        metadata["target_short_put_buffer_points"] = 45.0 if wide_bullish_monetization_ready else 40.0
        metadata["minimum_net_edge_rupees"] = 1200.0 if wide_bullish_monetization_ready else 900.0
    elif sideways_bullish_vwap_hold_ready:
        early_vwap_hold_high_flow_ready = bool(
            option_pressure["bullish_pressure"] >= 0.85
            and hour_change_pct >= 0.20
            and float(metadata.get("ema_distance_pct_5m") or 0.0) >= 0.10
            and bullish_support_quality >= 4.0
        )
        early_vwap_hold_flow_bias_ready = bool(
            oi_flow["smart_money_bias"] == "BULLISH"
            and oi_flow["bullish_flow_score"] >= 0.90
            and option_pressure["bullish_pressure"] >= 0.60
            and hour_change_pct >= 0.20
            and float(metadata.get("ema_distance_pct_5m") or 0.0) >= 0.10
            and bullish_support_quality >= 4.0
        )
        metadata["bullish_entry_ready"] = True
        metadata["bullish_setup"] = "VWAP_HOLD_HIGHER_LOW"
        metadata["playbook"] = "SIDEWAYS_TO_BULLISH_RECLAIM"
        metadata["preferred_width_points"] = 100.0
        metadata["allowed_width_points"] = (100.0,)
        metadata["target_short_put_buffer_points"] = 15.0 if early_vwap_hold_flow_bias_ready else 35.0
        metadata["minimum_net_edge_rupees"] = 850.0
        metadata["early_sideways_bullish_ready"] = bool(early_vwap_hold_high_flow_ready or early_vwap_hold_flow_bias_ready)
    elif afternoon_bullish_trend_hold_ready:
        metadata["bullish_entry_ready"] = True
        metadata["bullish_setup"] = "AFTERNOON_TREND_HOLD"
        metadata["playbook"] = "AFTERNOON_TREND_HOLD_BULLISH"
        metadata["preferred_width_points"] = 100.0
        metadata["allowed_width_points"] = (100.0,)
        metadata["target_short_put_buffer_points"] = 55.0
        metadata["minimum_net_edge_rupees"] = 1000.0
    if bearish_entry_score >= 3.0 and (bearish_pullback or bearish_shallow) and (bearish_planner_alignment or bearish_entry_score >= 4.0):
        metadata["bearish_entry_ready"] = True
        metadata["bearish_setup"] = "PULLBACK_REJECTION" if bearish_pullback else "SHALLOW_CONTINUATION"
        metadata["playbook"] = "BEARISH_CONTINUATION"
    if high_confluence_bearish_ready:
        metadata["bearish_entry_ready"] = True
        metadata["bearish_setup"] = "HIGH_CONFLUENCE_CONTINUATION"
        metadata["playbook"] = "HIGH_CONFLUENCE_BEARISH_CONTINUATION"
    gap_up_bearish_failure_ready = (
        big_gap_up
        and gap_pct <= 1.20
        and trend_15m == "TREND_DOWN"
        and execution_5m == "DOWN_CONFIRMED"
        and gap_up_bearish_failure
        and recent_gap_close_location <= 0.05
        and bearish_entry_score >= 3.4
        and bearish_trend_score >= 3.6
        and gap_bearish_alignment
    )
    gap_down_bearish_continuation_ready = (
        big_gap_down
        and gap_pct > -1.50
        and trend_15m == "TREND_DOWN"
        and execution_5m == "DOWN_CONFIRMED"
        and gap_down_bearish_continuation
        and bearish_entry_score >= 3.4
        and bearish_trend_score >= 3.8
        and gap_bearish_alignment
    )
    metadata["gap_bearish_ready"] = bool(gap_up_bearish_failure_ready or gap_down_bearish_continuation_ready)
    range_balance_score = 0.0
    if range_bias:
        range_balance_score += 1.0
    if opening_range is not None:
        range_mid = (opening_range.high + opening_range.low) / 2.0
        if abs(spot - range_mid) / max(spot, 1.0) <= 0.002:
            range_balance_score += 0.5
    if vwap is not None and abs(spot - vwap) / max(spot, 1.0) <= 0.0015:
        range_balance_score += 0.5
    if abs(option_pressure["bullish_pressure"] - option_pressure["bearish_pressure"]) <= 0.08:
        range_balance_score += 1.0
    if wall_migration["wall_migration_bias"] == "NEUTRAL":
        range_balance_score += 1.0
    if oi_flow["smart_money_bias"] == "NEUTRAL":
        range_balance_score += 1.0
    metadata["range_balance_score"] = range_balance_score
    metadata["range_entry_ready"] = bool(
        execution_5m == "RANGE_CONFIRMED"
        and snapshot.timestamp.time() >= RANGE_GATE_TIME
        and range_balance_score >= 4.0
        and abs(gap_pct) <= 0.15
        and balanced_range_condor_setup(
            bars_5m,
            opening_range,
            vwap,
            rv30_pct=rv30_pct,
        )
    )
    metadata["range_condor_credit_ratio"] = 0.16
    if metadata["range_entry_ready"]:
        metadata["preferred_width_points"] = 100.0
        metadata["allowed_width_points"] = (100.0,)
        metadata["minimum_net_edge_rupees"] = 500.0
    if (
        (trend_15m == "TREND_DOWN" and execution_5m == "DOWN_CONFIRMED")
        or (
            trend_15m == "NEUTRAL"
            and execution_5m == "DOWN_CONFIRMED"
            and (
                (
                    range_bias
                    and bearish_entry_score >= 3.4
                    and bearish_trend_score >= 3.4
                    and oi_flow["smart_money_bias"] == "BEARISH"
                    and current_structure_signal in {"BALANCED", "BEARISH_CHOCH", "BEARISH_BOS"}
                )
                or high_confluence_bearish_ready
            )
        )
    ):
        regime = RegimeLabel.DOWN_TREND
        metadata["day_archetype"] = (
            "EARLY_BALANCE_TO_BEARISH"
            if metadata.get("playbook") == "EARLY_BALANCE_BEARISH_FAILED_RECLAIM"
            else (
                "GAP_DOWN_CONTINUATION"
                if gap_down_bearish_continuation_ready
                else (
                    "GAP_UP_FAILURE"
                    if gap_up_bearish_failure_ready
                    else (
                        "HIGH_CONFLUENCE_BEARISH"
                        if high_confluence_bearish_ready
                        else ("OPEN_DRIVE_BEARISH" if open_drive_bearish else ("SIDEWAYS_TO_BEARISH" if sideways_to_bearish else "TREND_BEARISH"))
                    )
                )
            )
        )
        confidence += 0.15
        recent_bars = bars_5m[-6:]
        recent_high = max(bar.high for bar in recent_bars)
        session_high = max(bar.high for bar in bars_5m)
        session_low = min(bar.low for bar in bars_5m)
        last_close = recent_bars[-1].close
        metadata.update(
            {
                "recent_high_5m": recent_high,
                "session_close_location": close_location_value(last_close, session_low, session_high),
                "recent_close_location": close_location_value(
                    last_close,
                    min(bar.low for bar in recent_bars),
                    max(bar.high for bar in recent_bars),
                ),
                "below_vwap_last6": float(sum(1 for bar in recent_bars if vwap is not None and bar.close < vwap)),
                "below_ema20_last6": float(
                    sum(1 for bar in recent_bars if ema20_5m_value is not None and bar.close < ema20_5m_value)
                ),
            }
        )
        failed_reclaim = bearish_failed_reclaim_setup(recent_bars, ema20_5m_value, vwap)
        tight_breakdown = bearish_tight_breakdown_setup(
            recent_bars,
            ema20_5m_value,
            vwap,
            session_low=session_low,
            session_high=session_high,
        )
        early_balance_bearish_failed_reclaim_ready = (
            time(10, 10) <= snapshot.timestamp.time() <= time(11, 15)
            and range_bias
            and not big_gap_down
            and failed_reclaim
            and bearish_entry_score >= 3.4
            and bearish_trend_score >= 3.4
            and bearish_planner_alignment
            and oi_flow["smart_money_bias"] == "BEARISH"
            and wall_migration["wall_migration_bias"] != "BULLISH"
            and float(metadata.get("recent_close_location") or 1.0) <= 0.30
            and current_structure_signal in {"BALANCED", "BEARISH_CHOCH", "BEARISH_BOS"}
        )
        sideways_bearish_rejection_ready = (
            sideways_to_bearish
            and bearish_pullback
            and bearish_entry_score >= 3.4
            and bearish_trend_score >= 3.0
            and bearish_planner_alignment
            and option_pressure["bearish_pressure"] >= 0.55
            and oi_flow["smart_money_bias"] == "BEARISH"
            and wall_migration["wall_migration_bias"] != "BULLISH"
            and -0.35 <= hour_change_pct <= -0.10
            and float(metadata.get("ema_distance_pct_5m") or 0.0) <= -0.05
            and float(metadata.get("recent_close_location") or 0.0) <= 0.30
        )
        if gap_down_bearish_continuation_ready:
            metadata["bearish_entry_ready"] = True
            metadata["bearish_setup"] = "GAP_CONTINUATION"
            metadata["playbook"] = "GAP_DOWN_BEARISH_CONTINUATION"
            metadata["preferred_width_points"] = 100.0
            metadata["allowed_width_points"] = (100.0,)
            metadata["minimum_net_edge_rupees"] = 1100.0
            confidence += 0.11
            reasons.append(
                "Gap-down continuation confirmed: the downside gap stayed below VWAP/EMA20, the bounce failed into resistance, and bearish OI flow remained in control."
            )
        elif gap_up_bearish_failure_ready:
            metadata["bearish_entry_ready"] = True
            metadata["bearish_setup"] = "GAP_FAILURE"
            metadata["playbook"] = "GAP_UP_BEARISH_FAILURE"
            metadata["preferred_width_points"] = 100.0
            metadata["allowed_width_points"] = (100.0,)
            metadata["minimum_net_edge_rupees"] = 1000.0
            confidence += 0.10
            reasons.append(
                "Gap-up failure confirmed: the upside gap lost VWAP/opening-range support, reversal pressure built, and bearish OI alignment took over."
            )
        elif high_confluence_bearish_ready:
            metadata["bearish_entry_ready"] = True
            metadata["bearish_setup"] = "HIGH_CONFLUENCE_CONTINUATION"
            metadata["playbook"] = "HIGH_CONFLUENCE_BEARISH_CONTINUATION"
            metadata["day_archetype"] = "HIGH_CONFLUENCE_BEARISH"
            metadata["preferred_width_points"] = 100.0
            metadata["allowed_width_points"] = (100.0,)
            metadata["minimum_net_edge_rupees"] = 1200.0
            confidence += 0.12
            reasons.append(
                "High-confluence bearish continuation confirmed: neutral 15m context gave way to a strong 5m downside expansion with aligned structure, location, and OI flow."
            )
        elif early_balance_bearish_failed_reclaim_ready:
            metadata["bearish_entry_ready"] = True
            metadata["bearish_setup"] = "FAILED_RECLAIM"
            metadata["playbook"] = "EARLY_BALANCE_BEARISH_FAILED_RECLAIM"
            metadata["day_archetype"] = "EARLY_BALANCE_TO_BEARISH"
            metadata["preferred_width_points"] = 100.0
            metadata["allowed_width_points"] = (100.0,)
            metadata["minimum_net_edge_rupees"] = 950.0
            confidence += 0.10
            reasons.append(
                "Early balance-to-bearish failed reclaim confirmed: the first-hour balance failed under VWAP / EMA20, reclaim attempts were rejected, and bearish OI flow aligned early."
            )
        elif sideways_bearish_rejection_ready:
            metadata["bearish_entry_ready"] = True
            metadata["bearish_setup"] = "PULLBACK_REJECTION"
            metadata["playbook"] = "SIDEWAYS_TO_BEARISH_REJECTION"
            wide_bearish_rejection_ready = gap_pct >= 0.10
            metadata["preferred_width_points"] = 150.0 if wide_bearish_rejection_ready else 100.0
            metadata["allowed_width_points"] = (100.0, 150.0) if wide_bearish_rejection_ready else (100.0,)
            metadata["minimum_net_edge_rupees"] = 1200.0 if wide_bearish_rejection_ready else 1000.0
            confidence += 0.11
            reasons.append(
                "Sideways-to-bearish rejection confirmed: the session lost intraday balance, the pullback rejected below EMA20/VWAP, and bearish OI pressure stayed aligned."
            )
            if wide_bearish_rejection_ready:
                reasons.append("Positive opening gap increased upside premium, so the bearish rejection playbook can monetise through a wider call spread.")
        elif failed_reclaim:
            metadata["bearish_entry_ready"] = True
            metadata["bearish_setup"] = "FAILED_RECLAIM"
            metadata["playbook"] = "BEARISH_FAILED_RECLAIM"
            confidence += 0.10
            reasons.append("Separate bearish entry confirmed: failed reclaim below 5m EMA20/VWAP.")
        elif tight_breakdown:
            metadata["bearish_entry_ready"] = True
            metadata["bearish_setup"] = "TIGHT_BREAKDOWN"
            metadata["playbook"] = "BEARISH_CONTINUATION"
            confidence += 0.08
            reasons.append("Separate bearish entry confirmed: tight late-session breakdown remains pinned below 5m EMA20/VWAP.")
        elif bearish_pullback:
            confidence += 0.08
            reasons.append("Bearish pullback-rejection confirmed at 5m EMA20/VWAP/resistance.")
        elif bearish_shallow:
            confidence += 0.05
            reasons.append("Bearish shallow continuation confirmed after a weak bounce failed below 5m EMA20/VWAP.")
        else:
            reasons.append("Downtrend is present, but price has not yet given a proper failed reclaim, pullback rejection, or tight breakdown continuation.")
        if metadata["trend_follow_ready_bearish"]:
            confidence += 0.07
            reasons.append("Last-hour bearish trend score confirms downside continuation with EMA20/50, support/resistance, and option-chain pressure alignment.")
        if oi_flow["smart_money_bias"] == "BEARISH":
            confidence += 0.05
            reasons.append(
                f"Intraday OI flow supports the bearish setup: call-side resistance is building near {oi_flow['call_resistance_strike']} while put-side support is weakening."
            )
        if bearish_planner_alignment:
            confidence += 0.04
            reasons.append("Smart-money planner alignment supports the bearish playbook through OI flow or wall migration.")
        if ema20_5m_value is not None:
            if metadata.get("playbook") == "SIDEWAYS_TO_BEARISH_REJECTION":
                metadata["min_short_call_strike"] = max(recent_high + 25.0, last_close + 50.0, ema20_5m_value + 25.0)
            else:
                metadata["min_short_call_strike"] = max(recent_high + 50.0, last_close + 100.0, ema20_5m_value + 25.0)
    elif (
        (trend_15m == "TREND_UP" and execution_5m == "UP_CONFIRMED")
        or (
            trend_15m == "NEUTRAL"
            and execution_5m == "UP_CONFIRMED"
            and (
                (
                    range_bias
                    and bullish_entry_score >= 3.4
                    and bullish_support_quality >= 4.0
                    and oi_flow["smart_money_bias"] == "BULLISH"
                    and current_structure_signal in {"BALANCED", "BULLISH_CHOCH", "BULLISH_BOS"}
                    and time(10, 10) <= snapshot.timestamp.time() <= time(11, 15)
                )
                or high_confluence_bullish_ready
            )
        )
    ):
        regime = RegimeLabel.UP_TREND
        metadata["day_archetype"] = (
            "EARLY_BALANCE_TO_BULLISH"
            if metadata.get("playbook") == "EARLY_BALANCE_BULLISH_RECLAIM"
            else (
                "GAP_UP_CONTINUATION"
                if gap_up_bullish_ready
                else (
                    "GAP_DOWN_RECOVERY"
                    if gap_down_bullish_recovery_ready
                    else (
                        "HIGH_CONFLUENCE_BULLISH"
                        if high_confluence_bullish_ready
                        else ("OPEN_DRIVE_BULLISH" if open_drive_bullish else ("SIDEWAYS_TO_BULLISH" if sideways_to_bullish else "TREND_BULLISH"))
                    )
                )
            )
        )
        confidence += 0.15
        if metadata["playbook"] == "GAP_UP_BULLISH_CONTINUATION":
            confidence += 0.12
            reasons.append("Gap-up continuation playbook confirmed: the gap held above VWAP / OR-high support, pullback demand stayed firm, and smart-money flow remained supportive.")
        elif metadata["playbook"] == "GAP_DOWN_BULLISH_RECOVERY":
            confidence += 0.11
            reasons.append("Gap-down recovery playbook confirmed: the downside gap repaired through VWAP / OR-high, support flipped higher, and bullish flow took control.")
        elif metadata["playbook"] == "OPEN_DRIVE_BULLISH":
            confidence += 0.12
            reasons.append("Open-drive bullish playbook confirmed: trend-from-open held, the first pullback tagged EMA20 / OR-high support, and a reclaim candle printed.")
        elif metadata["playbook"] == "EARLY_BALANCE_BULLISH_RECLAIM":
            confidence += 0.10
            reasons.append("Early balance-to-bullish reclaim confirmed: the first-hour balance held, support reclaimed near VWAP / OR-high, and bullish OI flow aligned early.")
        elif metadata["playbook"] == "SIDEWAYS_TO_BULLISH_RECLAIM":
            confidence += 0.10
            reasons.append("Sideways-to-bullish reclaim confirmed: the morning balance resolved upward after 11:30 and support flipped higher with bullish OI wall migration.")
        elif bullish_pullback:
            confidence += 0.10
            reasons.append("Bullish pullback-reclaim confirmed at 5m EMA20/VWAP/support.")
        elif bullish_shallow:
            confidence += 0.07
            reasons.append("Bullish shallow continuation confirmed after a brief dip held above 5m EMA20/VWAP.")
        else:
            reasons.append("Uptrend exists, but price has not yet given a clean pullback-reclaim or shallow continuation entry.")
        if metadata["trend_follow_ready_bullish"]:
            confidence += 0.10
            reasons.append("Last-hour bullish trend score confirms continuation with EMA20/50, support/resistance, and option-chain pressure alignment.")
        if oi_flow["smart_money_bias"] == "BULLISH":
            confidence += 0.05
            reasons.append(
                f"Intraday OI flow supports the bullish setup: put-side support is building near {oi_flow['put_support_strike']} while call-side resistance is easing."
            )
        if bullish_planner_alignment:
            confidence += 0.04
            reasons.append("Smart-money planner alignment supports the bullish playbook through OI flow or wall migration.")
        if bullish_support_quality >= 3.5:
            confidence += 0.04
            reasons.append(f"Reclaimed support quality is strong ({bullish_support_quality:.2f}), improving strike safety and monetization quality.")
        if metadata["playbook"] == "GAP_UP_BULLISH_CONTINUATION":
            reclaim_support = max(level for level in [opening_range.high if opening_range else None, ema20_5m_value, vwap, bullish_support_anchor] if level is not None)
            metadata["max_short_put_strike"] = reclaim_support - 30.0
        elif metadata["playbook"] == "GAP_DOWN_BULLISH_RECOVERY":
            reclaim_support = max(level for level in [opening_range.high if opening_range else None, vwap, bullish_support_anchor] if level is not None)
            metadata["max_short_put_strike"] = reclaim_support - 25.0
        elif metadata["playbook"] == "OPEN_DRIVE_BULLISH":
            reclaim_support = max(level for level in [opening_range.high if opening_range else None, ema20_5m_value, bullish_support_anchor] if level is not None)
            metadata["max_short_put_strike"] = reclaim_support - 25.0
        elif metadata["playbook"] in {"SIDEWAYS_TO_BULLISH_RECLAIM", "EARLY_BALANCE_BULLISH_RECLAIM"}:
            reclaim_support = max(level for level in [opening_range.high if opening_range else None, bullish_support_anchor, vwap] if level is not None)
            if (
                metadata.get("bullish_setup") == "VWAP_HOLD_HIGHER_LOW"
                and bool(metadata.get("early_sideways_bullish_ready"))
                and oi_flow["smart_money_bias"] == "BULLISH"
            ):
                metadata["max_short_put_strike"] = reclaim_support - 15.0
            else:
                metadata["max_short_put_strike"] = reclaim_support - 50.0
        elif bullish_support_anchor is not None:
            metadata["max_short_put_strike"] = bullish_support_anchor - 25.0
    elif trend_15m == "NEUTRAL" and execution_5m == "RANGE_CONFIRMED":
        regime = RegimeLabel.RANGE
        confidence += 0.15
        metadata["day_archetype"] = "SIDEWAYS_RANGE"
        if metadata["range_entry_ready"]:
            metadata["playbook"] = "RANGE_BALANCED_CONDOR"
            confidence += 0.10
            reasons.append("Range playbook confirmed: price is balanced around VWAP, opening range is intact, and option-chain pressure is neutral.")
        else:
            metadata["playbook"] = "RANGE_NO_TRADE"
    else:
        reasons.append("Multi-timeframe regime remains unclear; defaulting to NO TRADE.")
        if any("event risk" in reason.lower() for reason in reasons):
            metadata["day_archetype"] = "EVENT_DAY"
            metadata["playbook"] = "EVENT_DAY_NO_TRADE"
        elif range_bias:
            metadata["day_archetype"] = "SIDEWAYS_RANGE"

    trade_plan = _build_trade_plan(
        metadata=metadata,
        regime=regime,
        opening_range_high=opening_range.high if opening_range else None,
        opening_range_low=opening_range.low if opening_range else None,
        vwap=vwap,
        support_ref=support_ref,
        resistance_ref=resistance_ref,
        bullish_support_anchor=bullish_support_anchor,
        bearish_resistance_anchor=bearish_resistance_anchor,
    )
    metadata["trade_plan"] = trade_plan
    metadata["structure_signal"] = trade_plan["structure_signal"]
    metadata["fvg_context"] = trade_plan["fvg_context"]
    metadata["order_block_context"] = trade_plan["order_block_context"]
    metadata["plan_execution"] = trade_plan["execution_plan"]
    metadata["plan_invalidation_level"] = trade_plan["invalidation_level"]
    metadata["plan_target_level"] = trade_plan["target_level"]
    metadata["plan_thesis"] = trade_plan["thesis"]

    return RegimeState(
        regime=regime,
        trend_15m=trend_15m,
        execution_5m=execution_5m,
        ema20_15m=ema20_value,
        ema20_slope_15m=ema20_slope_value,
        rv30_pct=rv30_pct,
        or_length_minutes=opening_range.length_minutes if opening_range else None,
        opening_range_high=opening_range.high if opening_range else None,
        opening_range_low=opening_range.low if opening_range else None,
        vwap=vwap,
        confidence=min(confidence, 1.0),
        reasons=reasons,
        metadata=metadata,
    )
