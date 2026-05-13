from __future__ import annotations

from datetime import time

from .data_models import AdaptiveParameters, MarketSnapshot, OhlcvBar, RegimeLabel, RegimeState, ValidationError
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
    compute_pcr_trend,
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
    put_call_ratio_by_oi,
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


def _market_state_classification(
    *,
    spot: float,
    now_time: time,
    vwap: float | None,
    ema_fast: float | None,
    ema_mid: float | None,
    ema_slow: float | None,
    current_structure_signal: str,
    execution_5m: str,
    bullish_pullback: bool,
    bullish_shallow: bool,
    bullish_vwap_hold: bool,
    bearish_pullback: bool,
    bearish_shallow: bool,
    range_entry_ready: bool,
    gap_up_bearish_failure_ready: bool,
    gap_down_bearish_continuation_ready: bool,
    gap_up_bullish_ready: bool,
    gap_down_bullish_recovery_ready: bool,
    open_drive_bullish_ready: bool,
    open_drive_bearish: bool,
    bullish_candle_pattern: str,
    bearish_candle_pattern: str,
    around_vwap: bool,
    opening_range_break_state: str = "NONE",
    candle_overlap_ratio: float = 1.0,
    trend_efficiency_ratio: float = 0.0,
    momentum_persistence_score: float = 0.0,
    higher_low_confirmed: bool = False,
    lower_high_confirmed: bool = False,
    failed_reclaim_candidate: bool = False,
    failed_bounce_candidate: bool = False,
) -> tuple[str, str, float, float, list[str]]:
    reasons: list[str] = []
    market_state = "TRUE_RANGE"
    market_state_bias = "NEUTRAL"
    state_confidence_score = 0.45
    state_quality_score = -1.25

    ema_alignment_bullish = bool(
        ema_fast is not None
        and ema_mid is not None
        and ema_slow is not None
        and ema_fast > ema_mid > ema_slow
    )
    ema_alignment_bearish = bool(
        ema_fast is not None
        and ema_mid is not None
        and ema_slow is not None
        and ema_fast < ema_mid < ema_slow
    )
    ema_compression = bool(
        ema_fast is not None
        and ema_mid is not None
        and ema_slow is not None
        and ((max(ema_fast, ema_mid, ema_slow) - min(ema_fast, ema_mid, ema_slow)) / max(spot, 1.0) <= 0.002)
    )
    sustained_above_vwap = bool(vwap is not None and spot > vwap and execution_5m == "UP_CONFIRMED")
    sustained_below_vwap = bool(vwap is not None and spot < vwap and execution_5m == "DOWN_CONFIRMED")

    bullish_failure_signal = bool(
        current_structure_signal == "BULLISH_CHOCH"
        or bullish_candle_pattern in {"BULLISH_BREAKDOWN_FAILURE", "BULLISH_MORNING_STAR", "BULLISH_HAMMER"}
        or gap_down_bullish_recovery_ready
    )
    bearish_failure_signal = bool(
        current_structure_signal == "BEARISH_CHOCH"
        or bearish_candle_pattern in {"BEARISH_BREAKOUT_FAILURE", "BEARISH_EVENING_STAR", "BEARISH_SHOOTING_STAR"}
        or gap_up_bearish_failure_ready
    )
    bearish_transition_signal = bool(bearish_failure_signal or failed_reclaim_candidate or failed_bounce_candidate)
    bullish_transition_signal = bool(bullish_failure_signal)
    directional_balance_bearish = bool(
        not ema_alignment_bullish
        and (
            lower_high_confirmed
            or bearish_pullback
            or bearish_shallow
            or execution_5m == "DOWN_CONFIRMED"
            or opening_range_break_state in {"DOWN", "FAILED_UP"}
            or failed_reclaim_candidate
            or failed_bounce_candidate
        )
        and (
            trend_efficiency_ratio >= 0.28
            or momentum_persistence_score >= 0.40
            or current_structure_signal in {"BEARISH_BOS", "BEARISH_CHOCH"}
        )
    )
    directional_balance_bullish = bool(
        not ema_alignment_bearish
        and (
            higher_low_confirmed
            or bullish_pullback
            or bullish_shallow
            or bullish_vwap_hold
            or execution_5m == "UP_CONFIRMED"
            or opening_range_break_state in {"UP", "FAILED_DOWN"}
            or gap_up_bullish_ready
            or open_drive_bullish_ready
        )
        and (
            trend_efficiency_ratio >= 0.28
            or momentum_persistence_score >= 0.40
            or current_structure_signal in {"BULLISH_BOS", "BULLISH_CHOCH"}
        )
    )
    true_range_like = bool(
        ema_compression
        or around_vwap
        or range_entry_ready
        or (candle_overlap_ratio >= 0.68 and trend_efficiency_ratio <= 0.45)
    )

    if bullish_transition_signal or bearish_transition_signal:
        market_state = "TRANSITION"
        if bearish_transition_signal and not bullish_transition_signal:
            market_state_bias = "BEARISH"
        elif bullish_transition_signal and not bearish_transition_signal:
            market_state_bias = "BULLISH"
        elif current_structure_signal.startswith("BEARISH"):
            market_state_bias = "BEARISH"
        elif current_structure_signal.startswith("BULLISH"):
            market_state_bias = "BULLISH"
        state_confidence_score = 0.75
        if current_structure_signal.endswith("CHOCH"):
            state_confidence_score += 0.10
        if bullish_transition_signal and bearish_transition_signal:
            state_confidence_score -= 0.05
        state_quality_score = 3.5 + (1.0 if current_structure_signal.endswith("CHOCH") else 0.5)
        if failed_bounce_candidate:
            state_quality_score += 0.4
        if failed_reclaim_candidate:
            state_quality_score += 0.25
        reasons.append("Market state is TRANSITION: structure shift or bearish/bullish failure signal has priority over pure trend continuation.")
    elif ema_alignment_bullish and sustained_above_vwap and current_structure_signal == "BULLISH_BOS":
        market_state = "TREND_UP"
        market_state_bias = "BULLISH"
        state_confidence_score = 0.72
        state_quality_score = 2.4
        if bullish_pullback or bullish_shallow or bullish_vwap_hold or open_drive_bullish_ready or gap_up_bullish_ready:
            state_quality_score += 0.85
        reasons.append("Market state is TREND_UP: fast/mid/slow EMA proxy is aligned and bullish structure is continuing above VWAP.")
    elif ema_alignment_bearish and sustained_below_vwap and current_structure_signal == "BEARISH_BOS":
        market_state = "TREND_DOWN"
        market_state_bias = "BEARISH"
        state_confidence_score = 0.72
        state_quality_score = 2.4
        if bearish_pullback or bearish_shallow or gap_down_bearish_continuation_ready or open_drive_bearish:
            state_quality_score += 0.85
        reasons.append("Market state is TREND_DOWN: fast/mid/slow EMA proxy is aligned and bearish structure is continuing below VWAP.")
    elif directional_balance_bearish or directional_balance_bullish:
        market_state = "DIRECTIONAL_BALANCE"
        if directional_balance_bearish and not directional_balance_bullish:
            market_state_bias = "BEARISH"
        elif directional_balance_bullish and not directional_balance_bearish:
            market_state_bias = "BULLISH"
        elif current_structure_signal.startswith("BEARISH") or opening_range_break_state in {"DOWN", "FAILED_UP"}:
            market_state_bias = "BEARISH"
        elif current_structure_signal.startswith("BULLISH") or opening_range_break_state in {"UP", "FAILED_DOWN"}:
            market_state_bias = "BULLISH"
        state_confidence_score = 0.64 + (0.08 if market_state_bias != "NEUTRAL" else 0.0)
        state_quality_score = 0.85
        if market_state_bias == "BEARISH":
            state_quality_score += 0.55
        if market_state_bias == "BULLISH":
            state_quality_score += 0.35
        if failed_bounce_candidate or failed_reclaim_candidate:
            state_quality_score += 0.65
        if opening_range_break_state in {"DOWN", "FAILED_UP", "UP", "FAILED_DOWN"}:
            state_quality_score += 0.25
        reasons.append("Market state is DIRECTIONAL_BALANCE: balance/compression persists, but directional bias and failure/acceptance context remain tradable.")
    else:
        market_state = "TRUE_RANGE"
        market_state_bias = "NEUTRAL"
        state_confidence_score = 0.70 if (true_range_like or ema_compression or around_vwap) else 0.52
        state_quality_score = -2.1 if not range_entry_ready else -1.1
        if candle_overlap_ratio >= 0.72:
            state_quality_score -= 0.35
        reasons.append("Market state is TRUE_RANGE: EMA compression, repeated VWAP rotation, and overlap dominate without directional acceptance.")

    if now_time >= time(14, 0):
        state_quality_score -= 0.75
        state_confidence_score = max(0.0, state_confidence_score - 0.08)
        reasons.append("Late-session state penalty applied after 14:00 IST.")

    return (
        market_state,
        market_state_bias,
        round(state_quality_score, 4),
        round(min(max(state_confidence_score, 0.0), 1.0), 4),
        reasons,
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _recent_path_quality(bars: list[OhlcvBar], *, count: int = 8) -> dict[str, float]:
    recent = bars[-count:]
    if len(recent) < 3:
        return {
            "candle_overlap_ratio": 1.0,
            "trend_efficiency_ratio": 0.0,
            "expansion_score": 0.0,
            "compression_score": 1.0,
            "wick_rejection_score": 0.0,
            "momentum_persistence_score": 0.0,
            "impulse_strength": 0.0,
            "pullback_depth_pct": 0.0,
            "pullback_duration_bars": 0.0,
        }

    overlap_samples: list[float] = []
    directional_matches = 0
    path_length = 0.0
    upper_rejection = 0.0
    lower_rejection = 0.0
    bullish_bodies = 0.0
    bearish_bodies = 0.0
    for previous, current in zip(recent, recent[1:]):
        overlap = max(0.0, min(previous.high, current.high) - max(previous.low, current.low))
        envelope = max(max(previous.high, current.high) - min(previous.low, current.low), 0.01)
        overlap_samples.append(overlap / envelope)
        delta = current.close - previous.close
        if delta != 0:
            directional_matches += 1 if delta > 0 else 0
        path_length += abs(delta)
        body = abs(current.close - current.open)
        range_points = max(current.high - current.low, 0.01)
        upper_wick = current.high - max(current.open, current.close)
        lower_wick = min(current.open, current.close) - current.low
        upper_rejection += upper_wick / range_points
        lower_rejection += lower_wick / range_points
        if current.close >= current.open:
            bullish_bodies += body / range_points
        else:
            bearish_bodies += body / range_points

    net_move = abs(recent[-1].close - recent[0].open)
    total_range = max(max(bar.high for bar in recent) - min(bar.low for bar in recent), 0.01)
    trend_efficiency_ratio = net_move / max(path_length, 0.01)
    candle_overlap_ratio = sum(overlap_samples) / len(overlap_samples)
    body_bias = abs(bullish_bodies - bearish_bodies) / max(len(recent) - 1, 1)
    expansion_score = _clamp((total_range / max(abs(recent[0].open), 1.0)) * 100.0 * 4.0 + (body_bias * 2.0), 0.0, 5.0)
    compression_score = _clamp(5.0 - (expansion_score * 0.8) - (trend_efficiency_ratio * 1.5), 0.0, 5.0)
    wick_rejection_score = _clamp(max(upper_rejection, lower_rejection) / max(len(recent) - 1, 1) * 5.0, 0.0, 5.0)
    momentum_persistence_score = _clamp((directional_matches / max(len(recent) - 1, 1)) * 5.0, 0.0, 5.0)

    pullback_depth_pct = 0.0
    pullback_duration_bars = 0.0
    closes_only = [bar.close for bar in recent]
    highest_close = max(closes_only)
    lowest_close = min(closes_only)
    if recent[-1].close >= recent[0].open:
        local_pullback_low = min(bar.low for bar in recent[-4:])
        pullback_depth_pct = max((highest_close - local_pullback_low) / max(highest_close, 1.0) * 100.0, 0.0)
        pullback_duration_bars = float(sum(1 for bar in recent[-4:] if bar.close <= bar.open))
    else:
        local_pullback_high = max(bar.high for bar in recent[-4:])
        pullback_depth_pct = max((local_pullback_high - lowest_close) / max(abs(lowest_close), 1.0) * 100.0, 0.0)
        pullback_duration_bars = float(sum(1 for bar in recent[-4:] if bar.close >= bar.open))

    return {
        "candle_overlap_ratio": round(candle_overlap_ratio, 4),
        "trend_efficiency_ratio": round(_clamp(trend_efficiency_ratio, 0.0, 1.0), 4),
        "expansion_score": round(expansion_score, 4),
        "compression_score": round(compression_score, 4),
        "wick_rejection_score": round(wick_rejection_score, 4),
        "momentum_persistence_score": round(momentum_persistence_score, 4),
        "impulse_strength": round(_clamp((net_move / total_range) * 5.0, 0.0, 5.0), 4),
        "pullback_depth_pct": round(pullback_depth_pct, 4),
        "pullback_duration_bars": round(pullback_duration_bars, 4),
    }


def _opening_range_break_state(
    bars: list[OhlcvBar],
    opening_range_high: float | None,
    opening_range_low: float | None,
    vwap: float | None,
) -> str:
    if not bars or opening_range_high is None or opening_range_low is None:
        return "NONE"
    recent = bars[-6:]
    last = recent[-1]
    above_count = sum(1 for bar in recent if bar.close > opening_range_high)
    below_count = sum(1 for bar in recent if bar.close < opening_range_low)
    broke_up = any(bar.high > opening_range_high for bar in recent)
    broke_down = any(bar.low < opening_range_low for bar in recent)
    if above_count >= 2 and (vwap is None or last.close > vwap):
        return "UP"
    if below_count >= 2 and (vwap is None or last.close < vwap):
        return "DOWN"
    if broke_up and last.close < opening_range_high and (vwap is None or last.close < vwap):
        return "FAILED_UP"
    if broke_down and last.close > opening_range_low and (vwap is None or last.close > vwap):
        return "FAILED_DOWN"
    return "NONE"


def _option_chain_context(
    *,
    spot: float,
    support_ref: float | None,
    resistance_ref: float | None,
    metadata: dict[str, float | str | bool | None],
    option_pressure: dict[str, float],
    oi_flow: dict[str, float | str | None],
    wall_migration: dict[str, float | str | None],
) -> dict[str, float | str | bool | None]:
    nearest_call_wall = metadata.get("current_call_wall")
    nearest_put_wall = metadata.get("current_put_wall")
    distance_to_call_wall = None if nearest_call_wall is None else float(nearest_call_wall) - spot
    distance_to_put_wall = None if nearest_put_wall is None else spot - float(nearest_put_wall)
    oi_pressure_imbalance = float(option_pressure.get("bearish_pressure", 0.0) - option_pressure.get("bullish_pressure", 0.0))
    overhead_call_pressure_score = 0.0
    downside_put_support_score = 0.0
    if distance_to_call_wall is not None:
        if distance_to_call_wall <= 50.0:
            overhead_call_pressure_score = 2.5
        elif distance_to_call_wall <= 100.0:
            overhead_call_pressure_score = 1.5
        elif distance_to_call_wall <= 150.0:
            overhead_call_pressure_score = 0.75
    if distance_to_put_wall is not None:
        if distance_to_put_wall <= 50.0:
            downside_put_support_score = 2.5
        elif distance_to_put_wall <= 100.0:
            downside_put_support_score = 1.5
        elif distance_to_put_wall <= 150.0:
            downside_put_support_score = 0.75

    if overhead_call_pressure_score >= 1.5 and downside_put_support_score >= 1.5:
        pressure_state = "BALANCED_WALLS"
    elif overhead_call_pressure_score >= 1.5:
        pressure_state = "OVERHEAD_CALL_PRESSURE"
    elif downside_put_support_score >= 1.5:
        pressure_state = "DOWNSIDE_PUT_SUPPORT"
    elif (wall_migration.get("wall_migration_bias") or "NEUTRAL") == "BEARISH":
        pressure_state = "BEARISH_WALL_SHIFT"
    elif (wall_migration.get("wall_migration_bias") or "NEUTRAL") == "BULLISH":
        pressure_state = "BULLISH_WALL_SHIFT"
    else:
        pressure_state = "NEUTRAL"

    return {
        "nearest_call_wall": nearest_call_wall,
        "nearest_put_wall": nearest_put_wall,
        "distance_to_call_wall": None if distance_to_call_wall is None else round(distance_to_call_wall, 2),
        "distance_to_put_wall": None if distance_to_put_wall is None else round(distance_to_put_wall, 2),
        "call_wall_migration": "UP" if float(wall_migration.get("call_wall_shift") or 0.0) > 0 else ("DOWN" if float(wall_migration.get("call_wall_shift") or 0.0) < 0 else "STABLE"),
        "put_wall_migration": "UP" if float(wall_migration.get("put_wall_shift") or 0.0) > 0 else ("DOWN" if float(wall_migration.get("put_wall_shift") or 0.0) < 0 else "STABLE"),
        "oi_pressure_imbalance": round(oi_pressure_imbalance, 4),
        "oi_pressure_bias": str(oi_flow.get("smart_money_bias") or "NEUTRAL"),
        "smart_money_bias": str(oi_flow.get("smart_money_bias") or "NEUTRAL"),
        "overhead_call_pressure_score": round(overhead_call_pressure_score, 4),
        "downside_put_support_score": round(downside_put_support_score, 4),
        "option_chain_pressure_state": pressure_state,
        "price_into_overhead_call_wall": bool(distance_to_call_wall is not None and distance_to_call_wall <= 75.0),
        "price_into_downside_put_wall": bool(distance_to_put_wall is not None and distance_to_put_wall <= 75.0),
        "nearest_resistance": _round_level(resistance_ref),
        "nearest_support": _round_level(support_ref),
    }

def _average_chain_iv(snapshot: MarketSnapshot) -> float | None:
    ivs = [
        float(quote.iv)
        for quote in snapshot.option_chain.quotes
        if quote.iv is not None and quote.ltp > 0
    ]
    if not ivs:
        return None
    return sum(ivs) / len(ivs)


def _bullish_shadow_subtype(
    *,
    playbook: str,
    bullish_setup: str | None,
    now_time: time,
    spot: float,
    ema20_5m_value: float | None,
    vwap: float | None,
    opening_range_high: float | None,
) -> str | None:
    if playbook not in {
        "SIDEWAYS_TO_BULLISH_RECLAIM",
        "EARLY_BALANCE_BULLISH_RECLAIM",
        "OPEN_DRIVE_BULLISH",
        "GAP_UP_BULLISH_CONTINUATION",
        "GAP_DOWN_BULLISH_RECOVERY",
        "AFTERNOON_TREND_HOLD_BULLISH",
        "HIGH_CONFLUENCE_BULLISH_CONTINUATION",
    }:
        return None
    if bullish_setup == "VWAP_HOLD_HIGHER_LOW" and vwap is not None and spot > vwap:
        return "VWAP_RECLAIM_HIGHER_LOW"
    if opening_range_high is not None and spot > opening_range_high and bullish_setup in {"SHALLOW_CONTINUATION", "HIGH_CONFLUENCE_CONTINUATION"}:
        return "OR_HOLD_BULLISH_CONTINUATION"
    if now_time >= time(13, 0) and bullish_setup in {"PULLBACK_RECLAIM", "VWAP_HOLD_HIGHER_LOW", "AFTERNOON_TREND_HOLD"}:
        return "LATE_SESSION_BULLISH_RECLAIM"
    if bullish_setup == "SHALLOW_CONTINUATION":
        return "SHALLOW_PULLBACK_CONTINUATION"
    if bullish_setup in {"PULLBACK_RECLAIM", "EARLY_BALANCE_RECLAIM", "OPEN_DRIVE_RECLAIM"} and ema20_5m_value is not None and vwap is not None and spot > ema20_5m_value and spot > vwap:
        return "EMA20_AND_VWAP_ALIGNED_BULLISH"
    return None


def _bearish_continuation_subtype(
    *,
    playbook: str,
    bearish_setup: str | None,
    now_time: time,
    big_gap_down: bool,
    spot: float,
    opening_range_low: float | None,
    vwap: float | None,
) -> str | None:
    if playbook not in {
        "BEARISH_CONTINUATION",
        "HIGH_CONFLUENCE_BEARISH_CONTINUATION",
        "GAP_DOWN_BEARISH_CONTINUATION",
    }:
        return None
    if big_gap_down and bearish_setup == "GAP_CONTINUATION":
        return "GAP_DOWN_FAILED_BOUNCE"
    if opening_range_low is not None and bearish_setup in {"PULLBACK_REJECTION", "TIGHT_BREAKDOWN"} and spot < opening_range_low:
        return "OR_LOW_RETEST_FAILURE"
    if vwap is not None and bearish_setup in {"PULLBACK_REJECTION", "HIGH_CONFLUENCE_CONTINUATION"} and spot < vwap:
        return "VWAP_REJECTION_CONTINUATION"
    if now_time >= time(13, 0):
        return "AFTERNOON_TREND_HOLD_BEARISH"
    if bearish_setup in {"SHALLOW_CONTINUATION", "PULLBACK_REJECTION"}:
        return "MIDDAY_WEAK_RECLAIM_CONTINUATION"
    return None


def _bearish_failed_reclaim_subtype(
    *,
    playbook: str,
    current_structure_signal: str,
    spot: float,
    ema20_5m_value: float | None,
    vwap: float | None,
    resistance_5m: float | None,
) -> str | None:
    if playbook not in {"BEARISH_FAILED_RECLAIM", "EARLY_BALANCE_BEARISH_FAILED_RECLAIM"}:
        return None
    if current_structure_signal == "BEARISH_CHOCH":
        return "CHOCH_FAILED_RECLAIM"
    if resistance_5m is not None and resistance_5m >= spot and (resistance_5m - spot) <= 35.0:
        return "SUPPLY_ZONE_FAILED_RECLAIM"
    if ema20_5m_value is not None and abs(spot - ema20_5m_value) <= 20.0 and spot < ema20_5m_value:
        return "EMA20_FAILED_RECLAIM"
    if vwap is not None and abs(spot - vwap) <= 20.0 and spot < vwap:
        return "VWAP_FAILED_RECLAIM"
    return "LOWER_HIGH_FAILED_RECLAIM"


def _bearish_rejection_transition_subtype(
    *,
    playbook: str,
    bearish_setup: str | None,
    market_state: str,
    opening_range_break_state: str,
    opening_range_low: float | None,
    spot: float,
    vwap: float | None,
    ema20_5m_value: float | None,
    ema50_5m_value: float | None,
    recent_rejection_high: float,
    recent_local_low: float,
    lower_high_confirmed: bool,
    bearish_pullback: bool,
    bearish_candle_quality_score: float,
    current_structure_signal: str,
    trend_efficiency_ratio: float,
    momentum_persistence_score: float,
) -> str | None:
    if playbook != "SIDEWAYS_TO_BEARISH_REJECTION":
        return None
    if bearish_setup != "PULLBACK_REJECTION":
        return None
    if market_state != "TRANSITION":
        return None
    if opening_range_break_state not in {"DOWN", "FAILED_UP"}:
        return None
    if not bearish_pullback or not lower_high_confirmed:
        return None
    if bearish_candle_quality_score < 3.25:
        return None
    if current_structure_signal not in {"BALANCED", "BEARISH_BOS", "BEARISH_CHOCH"}:
        return None
    if trend_efficiency_ratio < 0.22:
        return None
    bounce_distance = recent_rejection_high - recent_local_low
    if bounce_distance < 18.0:
        return None
    trigger_levels = [
        level
        for level in [opening_range_low, vwap, ema20_5m_value, ema50_5m_value]
        if level is not None
    ]
    if not trigger_levels:
        return None
    trigger_zone = min(float(level) for level in trigger_levels)
    rejection_under_trigger = recent_rejection_high <= trigger_zone - 8.0
    acceptance_lost = spot <= recent_rejection_high - 8.0 and spot < trigger_zone
    if not rejection_under_trigger or not acceptance_lost:
        return None
    if momentum_persistence_score < 1.2:
        return None
    return "BEARISH_REJECTION_TRANSITION"


def _bearish_failed_breakout_transition_shadow_subtype(
    *,
    market_state: str,
    failure_type: str,
    tradability_class: str,
    opening_range_break_state: str,
    execution_5m: str,
    bearish_trade_score: float,
    bearish_trade_score_threshold: float,
    threshold_buffer: float,
    setup_quality_score: float,
    strong_setup_quality_threshold: float,
    bearish_candle_quality_score: float,
    lower_high_confirmed: bool,
    current_structure_signal: str,
    price_vs_vwap: float | None,
    price_vs_ema20: float | None,
    price_vs_or_high: float | None,
    trend_efficiency_ratio: float,
    momentum_persistence_score: float,
) -> str | None:
    if market_state != "TRANSITION":
        return None
    if failure_type != "FAILED_BREAKOUT":
        return None
    if tradability_class != "TRADABLE":
        return None
    if opening_range_break_state != "FAILED_UP":
        return None
    if execution_5m in {"UP_CONFIRMED", "RANGE_CONFIRMED"}:
        return None
    if bearish_trade_score < bearish_trade_score_threshold - threshold_buffer:
        return None
    if setup_quality_score < strong_setup_quality_threshold:
        return None
    if current_structure_signal not in {"BALANCED", "BEARISH_BOS", "BEARISH_CHOCH"}:
        return None
    if not lower_high_confirmed and bearish_candle_quality_score < 3.0:
        return None
    if trend_efficiency_ratio < 0.28:
        return None
    if momentum_persistence_score < 0.5 or momentum_persistence_score > 3.8:
        return None
    rejection_points = 0
    for distance in (price_vs_vwap, price_vs_ema20, price_vs_or_high):
        if distance is not None and float(distance) <= -5.0:
            rejection_points += 1
    if rejection_points == 0:
        return None
    return "BEARISH_FAILED_BREAKOUT_TRANSITION"


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
    ema100_5m_value = ema_value(close_5m, period=100) or ema20_value
    ema20_5m_slope_value = ema_slope(close_5m, period=20, lookback=3)
    ema50_5m_slope_value = ema_slope(close_5m, period=50, lookback=3)
    ema100_proxy_slope_value = ema_slope(close_15m, period=20, lookback=5)
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
    previous_quotes = snapshot.previous_option_chain.quotes if snapshot.previous_option_chain is not None else None
    current_pcr = put_call_ratio_by_oi(snapshot.option_chain.quotes)
    previous_pcr = put_call_ratio_by_oi(previous_quotes or [])
    pcr_trend = compute_pcr_trend(snapshot.option_chain.quotes, previous_quotes)
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
        "ema100_5m": ema100_5m_value,
        "ema20_15m": ema20_value,
        "ema20_slope_5m": ema20_5m_slope_value,
        "ema50_slope_5m": ema50_5m_slope_value,
        "ema100_slope_proxy": ema100_proxy_slope_value,
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
        "pcr_by_oi": round(current_pcr, 4) if current_pcr is not None else None,
        "previous_pcr_by_oi": round(previous_pcr, 4) if previous_pcr is not None else None,
        "pcr_trend": pcr_trend,
        "bullish_entry_ready": False,
        "bullish_setup": None,
        "bullish_support_quality_score": 0.0,
        "bearish_entry_ready": False,
        "bearish_setup": None,
        "playbook": "NO_TRADE",
        "setup_subtype": None,
        "bullish_shadow_subtype": None,
        "bearish_shadow_subtype": None,
        "bearish_family": None,
        "bearish_subtype": None,
        "condor_profile": None,
        "condor_shape_bias": None,
        "condor_short_delta_band_bias": None,
        "avg_chain_iv": None,
        "range_compression_score": 0.0,
        "min_short_call_strike": None,
        "max_short_put_strike": None,
        "preferred_width_points": None,
        "allowed_width_points": None,
        "target_short_put_buffer_points": None,
        "minimum_net_edge_rupees": None,
        "setup_quality_score": 0.0,
        "setup_direction": "NONE",
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
        "market_state": "RANGE",
        "market_state_bias": "NEUTRAL",
        "state_quality_score": 0.0,
        "state_confidence_score": 0.0,
        "state_confidence": 0.0,
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
    metadata["bars_above_vwap"] = sum(1 for bar in bars_5m[-10:] if vwap is not None and bar.close > vwap)
    metadata["bars_below_vwap"] = sum(1 for bar in bars_5m[-10:] if vwap is not None and bar.close < vwap)
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
    metadata["bullish_upper_wick_fraction"] = float(bullish_candle_state.get("upper_wick_fraction") or 0.0)
    metadata["bullish_close_location"] = float(bullish_candle_state.get("close_location") or 0.5)
    metadata["bullish_body_fraction"] = float(bullish_candle_state.get("body_fraction") or 0.0)
    metadata["bearish_upper_wick_fraction"] = float(bearish_candle_state.get("upper_wick_fraction") or 0.0)
    metadata["bearish_close_location"] = float(bearish_candle_state.get("close_location") or 0.5)
    metadata["bearish_body_fraction"] = float(bearish_candle_state.get("body_fraction") or 0.0)
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
    failed_reclaim = False
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

    avg_chain_iv = _average_chain_iv(snapshot)
    range_compression_score = max(0.0, min(2.0, (params.rv_mid_cutoff - rv30_pct) / max(params.rv_mid_cutoff, 0.05))) if rv30_pct <= params.rv_mid_cutoff else 0.0
    metadata["avg_chain_iv"] = round(avg_chain_iv, 4) if avg_chain_iv is not None else None
    metadata["range_compression_score"] = round(range_compression_score, 4)

    bullish_shadow_subtype = _bullish_shadow_subtype(
        playbook=str(metadata.get("playbook") or "NO_TRADE"),
        bullish_setup=metadata.get("bullish_setup"),
        now_time=snapshot.timestamp.time(),
        spot=spot,
        ema20_5m_value=ema20_5m_value,
        vwap=vwap,
        opening_range_high=opening_range.high if opening_range else None,
    )
    if bullish_shadow_subtype is not None:
        metadata["bullish_shadow_subtype"] = bullish_shadow_subtype
        metadata["setup_subtype"] = bullish_shadow_subtype
        if bullish_shadow_subtype == "VWAP_RECLAIM_HIGHER_LOW":
            metadata["preferred_width_points"] = 75.0
            metadata["allowed_width_points"] = (75.0, 100.0)
            metadata["minimum_net_edge_rupees"] = 800.0
        elif bullish_shadow_subtype == "OR_HOLD_BULLISH_CONTINUATION":
            metadata["preferred_width_points"] = 100.0
            metadata["allowed_width_points"] = (100.0, 150.0)
            metadata["minimum_net_edge_rupees"] = 1100.0
        elif bullish_shadow_subtype == "LATE_SESSION_BULLISH_RECLAIM":
            metadata["preferred_width_points"] = 75.0
            metadata["allowed_width_points"] = (75.0, 100.0)
            metadata["minimum_net_edge_rupees"] = 700.0
        elif bullish_shadow_subtype == "SHALLOW_PULLBACK_CONTINUATION":
            metadata["preferred_width_points"] = 100.0
            metadata["allowed_width_points"] = (75.0, 100.0)
            metadata["minimum_net_edge_rupees"] = 850.0
        elif bullish_shadow_subtype == "EMA20_AND_VWAP_ALIGNED_BULLISH":
            metadata["preferred_width_points"] = 100.0
            metadata["allowed_width_points"] = (100.0, 150.0)
            metadata["minimum_net_edge_rupees"] = 900.0

    bearish_continuation_subtype = _bearish_continuation_subtype(
        playbook=str(metadata.get("playbook") or "NO_TRADE"),
        bearish_setup=metadata.get("bearish_setup"),
        now_time=snapshot.timestamp.time(),
        big_gap_down=big_gap_down,
        spot=spot,
        opening_range_low=opening_range.low if opening_range else None,
        vwap=vwap,
    )
    if bearish_continuation_subtype is not None:
        metadata["bearish_family"] = "BEARISH_CONTINUATION"
        metadata["bearish_subtype"] = bearish_continuation_subtype
        metadata["setup_subtype"] = bearish_continuation_subtype
        if bearish_continuation_subtype == "GAP_DOWN_FAILED_BOUNCE":
            metadata["preferred_width_points"] = 100.0
            metadata["allowed_width_points"] = (100.0, 150.0)
            metadata["minimum_net_edge_rupees"] = 1100.0
        elif bearish_continuation_subtype == "VWAP_REJECTION_CONTINUATION":
            metadata["preferred_width_points"] = 100.0
            metadata["allowed_width_points"] = (75.0, 100.0)
            metadata["minimum_net_edge_rupees"] = 950.0
        elif bearish_continuation_subtype == "OR_LOW_RETEST_FAILURE":
            metadata["preferred_width_points"] = 100.0
            metadata["allowed_width_points"] = (100.0,)
            metadata["minimum_net_edge_rupees"] = 1000.0
        elif bearish_continuation_subtype == "MIDDAY_WEAK_RECLAIM_CONTINUATION":
            metadata["preferred_width_points"] = 75.0
            metadata["allowed_width_points"] = (75.0, 100.0)
            metadata["minimum_net_edge_rupees"] = 850.0
        elif bearish_continuation_subtype == "AFTERNOON_TREND_HOLD_BEARISH":
            metadata["preferred_width_points"] = 75.0
            metadata["allowed_width_points"] = (75.0, 100.0)
            metadata["minimum_net_edge_rupees"] = 800.0

    bearish_failed_reclaim_subtype = _bearish_failed_reclaim_subtype(
        playbook=str(metadata.get("playbook") or "NO_TRADE"),
        current_structure_signal=current_structure_signal,
        spot=spot,
        ema20_5m_value=ema20_5m_value,
        vwap=vwap,
        resistance_5m=metadata.get("resistance_5m"),
    )
    if bearish_failed_reclaim_subtype is not None:
        metadata["bearish_family"] = "BEARISH_FAILED_RECLAIM"
        metadata["bearish_subtype"] = bearish_failed_reclaim_subtype
        metadata["setup_subtype"] = bearish_failed_reclaim_subtype
        if bearish_failed_reclaim_subtype in {"EMA20_FAILED_RECLAIM", "VWAP_FAILED_RECLAIM"}:
            metadata["preferred_width_points"] = 75.0
            metadata["allowed_width_points"] = (75.0, 100.0)
            metadata["minimum_net_edge_rupees"] = 850.0
        elif bearish_failed_reclaim_subtype == "LOWER_HIGH_FAILED_RECLAIM":
            metadata["preferred_width_points"] = 100.0
            metadata["allowed_width_points"] = (75.0, 100.0)
            metadata["minimum_net_edge_rupees"] = 900.0
        elif bearish_failed_reclaim_subtype == "SUPPLY_ZONE_FAILED_RECLAIM":
            metadata["preferred_width_points"] = 100.0
            metadata["allowed_width_points"] = (100.0,)
            metadata["minimum_net_edge_rupees"] = 1000.0
        elif bearish_failed_reclaim_subtype == "CHOCH_FAILED_RECLAIM":
            metadata["preferred_width_points"] = 100.0
            metadata["allowed_width_points"] = (100.0,)
            metadata["minimum_net_edge_rupees"] = 1050.0

    path_quality = _recent_path_quality(bars_5m)
    metadata.update(path_quality)
    opening_range_break_state = _opening_range_break_state(
        bars_5m,
        opening_range.high if opening_range else None,
        opening_range.low if opening_range else None,
        vwap,
    )
    higher_low_confirmed = bool(recent_higher_lows(hour_bars, needed=2))
    lower_high_confirmed = bool(recent_lower_highs(hour_bars, needed=2))
    failed_reclaim_candidate = bool(
        metadata.get("bearish_setup") == "FAILED_RECLAIM"
        or bearish_failed_reclaim_subtype is not None
        or failed_reclaim
    )
    pre_failed_bounce_candidate = bool(
        bearish_continuation_subtype == "GAP_DOWN_FAILED_BOUNCE"
        or (
            opening_range is not None
            and bearish_pullback
            and spot < opening_range.low
            and opening_range_break_state in {"DOWN", "FAILED_UP"}
        )
        or (
            vwap is not None
            and bearish_pullback
            and spot < vwap
            and opening_range_break_state in {"FAILED_UP", "DOWN"}
        )
        or (
            lower_high_confirmed
            and current_structure_signal in {"BEARISH_BOS", "BEARISH_CHOCH"}
            and bearish_pullback
        )
    )

    market_state, market_state_bias, state_quality_score, state_confidence_score, market_state_reasons = _market_state_classification(
        spot=spot,
        now_time=snapshot.timestamp.time(),
        vwap=vwap,
        ema_fast=ema20_5m_value,
        ema_mid=ema50_5m_value,
        ema_slow=ema20_value,
        current_structure_signal=current_structure_signal,
        execution_5m=execution_5m,
        bullish_pullback=bullish_pullback,
        bullish_shallow=bullish_shallow,
        bullish_vwap_hold=bullish_vwap_hold,
        bearish_pullback=bearish_pullback,
        bearish_shallow=bearish_shallow,
        range_entry_ready=bool(metadata.get("range_entry_ready")),
        gap_up_bearish_failure_ready=gap_up_bearish_failure_ready,
        gap_down_bearish_continuation_ready=gap_down_bearish_continuation_ready,
        gap_up_bullish_ready=gap_up_bullish_ready,
        gap_down_bullish_recovery_ready=gap_down_bullish_recovery_ready,
        open_drive_bullish_ready=open_drive_bullish_ready,
        open_drive_bearish=open_drive_bearish,
        bullish_candle_pattern=str(metadata.get("bullish_candle_pattern") or "NONE"),
        bearish_candle_pattern=str(metadata.get("bearish_candle_pattern") or "NONE"),
        around_vwap=bool(vwap is not None and oscillates_around_vwap(bars_5m, vwap, required_bars=6)),
        opening_range_break_state=opening_range_break_state,
        candle_overlap_ratio=float(path_quality["candle_overlap_ratio"]),
        trend_efficiency_ratio=float(path_quality["trend_efficiency_ratio"]),
        momentum_persistence_score=float(path_quality["momentum_persistence_score"]),
        higher_low_confirmed=higher_low_confirmed,
        lower_high_confirmed=lower_high_confirmed,
        failed_reclaim_candidate=failed_reclaim_candidate,
        failed_bounce_candidate=pre_failed_bounce_candidate,
    )
    metadata["market_state"] = market_state
    metadata["market_state_bias"] = market_state_bias
    metadata["state_quality_score"] = state_quality_score
    metadata["state_confidence_score"] = state_confidence_score
    reasons.extend(market_state_reasons)
    if snapshot.timestamp.time() >= time(14, 0):
        confidence = max(0.0, confidence - 0.05)

    if metadata.get("playbook") == "RANGE_BALANCED_CONDOR":
        if avg_chain_iv is not None and avg_chain_iv >= 22.0 and range_compression_score >= 0.75:
            metadata["condor_profile"] = "COMPRESSED_HIGH_IV"
            metadata["condor_short_delta_band_bias"] = "10_16"
            metadata["condor_shape_bias"] = "SYMMETRIC"
        elif metadata.get("smart_money_bias") == "BULLISH":
            metadata["condor_profile"] = "SLIGHT_BULLISH_RANGE"
            metadata["condor_short_delta_band_bias"] = "08_14"
            metadata["condor_shape_bias"] = "CALL_WIDER"
        elif metadata.get("smart_money_bias") == "BEARISH":
            metadata["condor_profile"] = "SLIGHT_BEARISH_RANGE"
            metadata["condor_short_delta_band_bias"] = "08_14"
            metadata["condor_shape_bias"] = "PUT_WIDER"
        else:
            metadata["condor_profile"] = "NEUTRAL_BALANCED_RANGE"
            metadata["condor_short_delta_band_bias"] = "12_20"
            metadata["condor_shape_bias"] = "SYMMETRIC"
        metadata["setup_subtype"] = metadata["condor_profile"]

    bullish_setup_quality = (
        float(metadata.get("bullish_confluence_score") or 0.0)
        + float(metadata.get("bullish_entry_score") or 0.0)
        + float(metadata.get("bullish_trend_score") or 0.0)
        + float(metadata.get("bullish_support_quality_score") or 0.0)
    )
    bearish_setup_quality = (
        float(metadata.get("bearish_confluence_score") or 0.0)
        + float(metadata.get("bearish_entry_score") or 0.0)
        + float(metadata.get("bearish_trend_score") or 0.0)
    )
    range_setup_quality = float(metadata.get("range_balance_score") or 0.0) + (
        1.5 if bool(metadata.get("range_entry_ready")) else 0.0
    )
    if market_state == "TREND_UP":
        bullish_setup_quality += state_quality_score
        bearish_setup_quality += min(state_quality_score, 0.0)
    elif market_state == "TREND_DOWN":
        bearish_setup_quality += state_quality_score
        bullish_setup_quality += min(state_quality_score, 0.0)
    elif market_state == "TRANSITION":
        if market_state_bias == "BULLISH":
            bullish_setup_quality += state_quality_score
            bearish_setup_quality -= 0.5
        elif market_state_bias == "BEARISH":
            bearish_setup_quality += state_quality_score
            bullish_setup_quality -= 0.5
        else:
            bullish_setup_quality += state_quality_score * 0.5
            bearish_setup_quality += state_quality_score * 0.5
    elif market_state == "DIRECTIONAL_BALANCE":
        if market_state_bias == "BEARISH":
            bearish_setup_quality += state_quality_score
            bullish_setup_quality += min(state_quality_score - 0.75, 0.0)
        elif market_state_bias == "BULLISH":
            bullish_setup_quality += state_quality_score * 0.75
            bearish_setup_quality += min(state_quality_score - 0.5, 0.0)
        else:
            bullish_setup_quality += state_quality_score * 0.35
            bearish_setup_quality += state_quality_score * 0.35
        range_setup_quality += max(state_quality_score * 0.25, 0.0)
    elif market_state == "TRUE_RANGE":
        bullish_setup_quality += state_quality_score
        bearish_setup_quality += state_quality_score
        range_setup_quality += max(state_quality_score, -0.25)
    if regime == RegimeLabel.UP_TREND and metadata.get("bullish_entry_ready"):
        metadata["setup_quality_score"] = round(bullish_setup_quality, 4)
        metadata["setup_direction"] = "BULLISH"
    elif regime == RegimeLabel.DOWN_TREND and metadata.get("bearish_entry_ready"):
        metadata["setup_quality_score"] = round(bearish_setup_quality, 4)
        metadata["setup_direction"] = "BEARISH"
    elif regime == RegimeLabel.RANGE and metadata.get("range_entry_ready"):
        metadata["setup_quality_score"] = round(range_setup_quality, 4)
        metadata["setup_direction"] = "RANGE"
    else:
        metadata["setup_quality_score"] = round(max(bullish_setup_quality, bearish_setup_quality, range_setup_quality), 4)
        metadata["setup_direction"] = "NEUTRAL"

    option_context = _option_chain_context(
        spot=spot,
        support_ref=support_ref,
        resistance_ref=resistance_ref,
        metadata=metadata,
        option_pressure=option_pressure,
        oi_flow=oi_flow,
        wall_migration=wall_migration,
    )
    metadata.update(option_context)

    time_remaining_minutes = max(0, ((15 * 60) + 15) - (snapshot.timestamp.hour * 60 + snapshot.timestamp.minute))
    ema_alignment = "MIXED"
    if ema20_5m_value is not None and ema50_5m_value is not None and ema100_5m_value is not None:
        if ema20_5m_value > ema50_5m_value > ema100_5m_value:
            ema_alignment = "BULLISH"
        elif ema20_5m_value < ema50_5m_value < ema100_5m_value:
            ema_alignment = "BEARISH"
    ema_spread_points = 0.0
    ema_compression_score = 0.0
    if ema20_5m_value is not None and ema50_5m_value is not None and ema100_5m_value is not None:
        ema_spread_points = ((max(ema20_5m_value, ema50_5m_value, ema100_5m_value) - min(ema20_5m_value, ema50_5m_value, ema100_5m_value)) / max(spot, 1.0)) * 1000.0
        ema_compression_score = _clamp(5.0 - (ema_spread_points * 2.0), 0.0, 5.0)
    metadata["ema_alignment"] = ema_alignment
    metadata["ema_spread_score"] = round(_clamp(ema_spread_points * 1.5, 0.0, 5.0), 4)
    metadata["ema_compression_score"] = round(ema_compression_score, 4)
    metadata["price_vs_ema20"] = None if ema20_5m_value is None else round(spot - ema20_5m_value, 2)
    metadata["price_vs_ema50"] = None if ema50_5m_value is None else round(spot - ema50_5m_value, 2)
    metadata["price_vs_ema100"] = None if ema100_5m_value is None else round(spot - ema100_5m_value, 2)
    metadata["ema20_slope"] = round(float(ema20_5m_slope_value or 0.0), 4)
    metadata["ema50_slope"] = round(float(ema50_5m_slope_value or 0.0), 4)
    metadata["ema100_slope"] = round(float(ema100_proxy_slope_value or 0.0), 4)
    metadata["price_vs_vwap"] = None if vwap is None else round(spot - vwap, 2)
    metadata["or_high"] = opening_range.high if opening_range else None
    metadata["or_low"] = opening_range.low if opening_range else None
    metadata["price_vs_or_high"] = None if opening_range is None else round(spot - opening_range.high, 2)
    metadata["price_vs_or_low"] = None if opening_range is None else round(spot - opening_range.low, 2)
    metadata["inside_opening_range"] = bool(opening_range is not None and opening_range.low <= spot <= opening_range.high)
    metadata["opening_range_break_state"] = opening_range_break_state
    metadata["higher_low_confirmed"] = higher_low_confirmed
    metadata["lower_high_confirmed"] = lower_high_confirmed
    if bool(metadata.get("higher_lows_5m")) and bool(metadata.get("higher_highs_5m")):
        metadata["swing_state"] = "HH_HL"
    elif bool(metadata.get("lower_highs_5m")) and bool(metadata.get("lower_lows_5m")):
        metadata["swing_state"] = "LH_LL"
    else:
        metadata["swing_state"] = "MIXED"

    resistance_candidates = [
        level
        for level in [
            resistance_ref,
            option_context.get("nearest_call_wall"),
        ]
        if isinstance(level, (int, float))
    ]
    support_candidates = [
        level
        for level in [
            support_ref,
            option_context.get("nearest_put_wall"),
        ]
        if isinstance(level, (int, float))
    ]
    nearest_resistance_level = min((float(level) for level in resistance_candidates if float(level) > spot), default=None)
    nearest_support_level = max((float(level) for level in support_candidates if float(level) < spot), default=None)
    distance_to_resistance = max((nearest_resistance_level - spot), 0.0) if nearest_resistance_level is not None else 999.0
    distance_to_support = max((spot - nearest_support_level), 0.0) if nearest_support_level is not None else 999.0
    metadata["distance_to_resistance"] = round(distance_to_resistance, 2)
    metadata["distance_to_support"] = round(distance_to_support, 2)
    metadata["open_space_up"] = round(min(distance_to_resistance, float(option_context.get("distance_to_call_wall") or distance_to_resistance)), 2)
    metadata["open_space_down"] = round(min(distance_to_support, float(option_context.get("distance_to_put_wall") or distance_to_support)), 2)
    metadata["inside_balance_zone"] = bool(metadata["inside_opening_range"] and abs(float(metadata.get("price_vs_vwap") or 0.0)) <= 20.0)

    failed_breakout = bool(
        opening_range_break_state == "FAILED_UP"
        or metadata.get("bearish_candle_pattern") == "BEARISH_BREAKOUT_FAILURE"
        or gap_up_bearish_failure_ready
    )
    failed_reclaim_flag = failed_reclaim_candidate
    recent_rejection_high = max((bar.high for bar in bars_5m[-4:]), default=spot)
    recent_local_low = min((bar.low for bar in bars_5m[-6:-1]), default=spot)
    ema_cluster_upper = max(
        value
        for value in [ema20_5m_value, ema50_5m_value]
        if value is not None
    ) if any(value is not None for value in [ema20_5m_value, ema50_5m_value]) else None
    ema_cluster_rejection = bool(
        ema_cluster_upper is not None
        and ema20_5m_value is not None
        and ema50_5m_value is not None
        and abs(ema20_5m_value - ema50_5m_value) <= 35.0
        and bearish_pullback
        and recent_rejection_high >= ema_cluster_upper - 10.0
        and spot < ema20_5m_value
        and lower_high_confirmed
    )
    vwap_failed_bounce = bool(
        vwap is not None
        and bearish_pullback
        and recent_rejection_high >= vwap - 10.0
        and spot < vwap
        and opening_range_break_state in {"DOWN", "FAILED_UP"}
    )
    lower_high_breakdown_retest = bool(
        lower_high_confirmed
        and current_structure_signal in {"BEARISH_BOS", "BEARISH_CHOCH"}
        and bearish_pullback
        and float(path_quality["trend_efficiency_ratio"]) >= 0.28
    )
    or_low_revisit_rejection = bool(
        opening_range is not None
        and opening_range.low is not None
        and recent_rejection_high >= opening_range.low - 8.0
        and spot < opening_range.low
        and bearish_pullback
    )
    weak_failed_bounce = bool(
        (bearish_pullback or bearish_shallow)
        and lower_high_confirmed
        and float(path_quality["momentum_persistence_score"]) <= 0.42
        and (vwap is None or spot < vwap)
    )
    bearish_rejection_transition_subtype = _bearish_rejection_transition_subtype(
        playbook=str(metadata.get("playbook") or "NO_TRADE"),
        bearish_setup=metadata.get("bearish_setup"),
        market_state=market_state,
        opening_range_break_state=opening_range_break_state,
        opening_range_low=opening_range.low if opening_range else None,
        spot=spot,
        vwap=vwap,
        ema20_5m_value=ema20_5m_value,
        ema50_5m_value=ema50_5m_value,
        recent_rejection_high=recent_rejection_high,
        recent_local_low=recent_local_low,
        lower_high_confirmed=lower_high_confirmed,
        bearish_pullback=bearish_pullback,
        bearish_candle_quality_score=float(metadata.get("bearish_candle_quality_score") or 0.0),
        current_structure_signal=current_structure_signal,
        trend_efficiency_ratio=float(path_quality["trend_efficiency_ratio"]),
        momentum_persistence_score=float(path_quality["momentum_persistence_score"]),
    )
    if bearish_rejection_transition_subtype is not None and metadata.get("bearish_subtype") is None:
        metadata["bearish_family"] = "BEARISH_REJECTION"
        metadata["bearish_subtype"] = bearish_rejection_transition_subtype
        metadata["setup_subtype"] = bearish_rejection_transition_subtype
    metadata["bearish_rejection_transition"] = bool(bearish_rejection_transition_subtype)
    failed_bounce = bool(
        bearish_continuation_subtype == "GAP_DOWN_FAILED_BOUNCE"
        or ema_cluster_rejection
        or vwap_failed_bounce
        or lower_high_breakdown_retest
        or or_low_revisit_rejection
        or weak_failed_bounce
        or (
            gap_down_bearish_continuation_ready
            and bearish_pullback
            and lower_high_confirmed
            and market_state in {"TREND_DOWN", "TRANSITION", "DIRECTIONAL_BALANCE"}
        )
    )
    accepted_breakout = bool(
        opening_range_break_state == "UP"
        and current_structure_signal == "BULLISH_BOS"
        and market_state == "TREND_UP"
    )
    accepted_breakdown = bool(
        opening_range_break_state == "DOWN"
        and current_structure_signal == "BEARISH_BOS"
        and market_state in {"TREND_DOWN", "TRANSITION"}
    )
    metadata["failed_breakout"] = failed_breakout
    metadata["failed_breakdown"] = bool(opening_range_break_state == "FAILED_DOWN")
    metadata["failed_reclaim"] = failed_reclaim_flag
    metadata["failed_bounce"] = failed_bounce
    metadata["accepted_breakout"] = accepted_breakout
    metadata["accepted_breakdown"] = accepted_breakdown
    if failed_breakout:
        failure_type = "FAILED_BREAKOUT"
    elif failed_reclaim_flag:
        failure_type = "FAILED_RECLAIM"
    elif failed_bounce:
        failure_type = "FAILED_BOUNCE"
    elif bearish_rejection_transition_subtype is not None:
        failure_type = "BEARISH_REJECTION_TRANSITION"
    elif accepted_breakout:
        failure_type = "ACCEPTED_BREAKOUT"
    elif accepted_breakdown:
        failure_type = "ACCEPTED_BREAKDOWN"
    else:
        failure_type = "NONE"
    metadata["failure_type"] = failure_type

    retest_success_score = 0.0
    retest_failure_score = 0.0
    if accepted_breakout or accepted_breakdown:
        retest_success_score += 2.0
    if failed_breakout or failed_reclaim_flag or failed_bounce:
        retest_failure_score += 2.5
    metadata["break_acceptance_score"] = round(2.0 if accepted_breakout or accepted_breakdown else 0.0, 4)
    metadata["retest_success_score"] = round(retest_success_score, 4)
    metadata["retest_failure_score"] = round(retest_failure_score, 4)

    first_test_buffer = 20.0
    first_test_hits = 0
    if nearest_resistance_level is not None:
        first_test_hits += sum(1 for bar in bars_5m[-8:-1] if abs(bar.high - nearest_resistance_level) <= first_test_buffer)
    if nearest_support_level is not None:
        first_test_hits += sum(1 for bar in bars_5m[-8:-1] if abs(bar.low - nearest_support_level) <= first_test_buffer)
    metadata["at_first_test_of_level"] = first_test_hits <= 1
    level_rejection_score = 0.0
    if failed_breakout or failed_reclaim_flag or failed_bounce:
        level_rejection_score += max(float(metadata.get("bearish_candle_quality_score") or 0.0), float(path_quality["wick_rejection_score"]) * 0.5)
    elif accepted_breakout:
        level_rejection_score += max(float(metadata.get("bullish_candle_quality_score") or 0.0), 1.0)
    metadata["level_rejection_score"] = round(_clamp(level_rejection_score, 0.0, 5.0), 4)

    bullish_trend_quality_score = _clamp(
        (bullish_trend_score * 1.55)
        + (1.0 if market_state == "TREND_UP" else 0.0)
        + (0.5 if current_structure_signal == "BULLISH_BOS" else 0.0)
        + (float(path_quality["momentum_persistence_score"]) * 0.4),
        0.0,
        10.0,
    )
    bearish_trend_quality_score = _clamp(
        (bearish_trend_score * 1.55)
        + (1.0 if market_state == "TREND_DOWN" else 0.0)
        + (0.75 if market_state == "TRANSITION" and market_state_bias == "BEARISH" else 0.0)
        + (0.5 if current_structure_signal in {"BEARISH_BOS", "BEARISH_CHOCH"} else 0.0)
        + (float(path_quality["momentum_persistence_score"]) * 0.4),
        0.0,
        10.0,
    )
    bearish_failure_score = 0.0
    bullish_failure_score = 0.0
    if failed_reclaim_flag:
        bearish_failure_score += 3.0
    if failed_bounce:
        bearish_failure_score += 3.0
    if bearish_rejection_transition_subtype is not None:
        bearish_failure_score += 2.5
    if failed_breakout:
        bearish_failure_score += 2.5
    if accepted_breakdown:
        bearish_failure_score += 1.5
    if current_structure_signal == "BEARISH_CHOCH":
        bearish_failure_score += 1.0
    bearish_failure_score += min(float(metadata.get("bearish_candle_quality_score") or 0.0), 2.0)
    if accepted_breakout:
        bullish_failure_score += 1.5
    if current_structure_signal == "BULLISH_CHOCH":
        bullish_failure_score += 1.0
    if opening_range_break_state == "FAILED_DOWN":
        bullish_failure_score += 2.5
    bullish_failure_score += min(float(metadata.get("bullish_candle_quality_score") or 0.0), 2.0)

    bearish_location_live_score = _clamp(
        (float(metadata.get("bearish_location_score") or 0.0) * 2.0)
        + (min(float(metadata.get("open_space_down") or 0.0), 250.0) / 80.0)
        + (0.75 if metadata.get("at_first_test_of_level") else 0.0)
        + float(metadata.get("level_rejection_score") or 0.0),
        0.0,
        10.0,
    )
    bullish_location_live_score = _clamp(
        (float(metadata.get("bullish_location_score") or 0.0) * 2.0)
        + (min(float(metadata.get("open_space_up") or 0.0), 250.0) / 80.0)
        + (0.75 if metadata.get("at_first_test_of_level") else 0.0)
        - float(metadata.get("overhead_call_pressure_score") or 0.0),
        0.0,
        10.0,
    )
    bearish_option_chain_pressure_score = _clamp(
        (max(float(metadata.get("oi_pressure_imbalance") or 0.0), 0.0) * 6.0)
        + (2.0 if metadata.get("smart_money_bias") == "BEARISH" else 0.0)
        + (1.5 if metadata.get("option_chain_pressure_state") in {"OVERHEAD_CALL_PRESSURE", "BEARISH_WALL_SHIFT"} else 0.0)
        + (0.75 if metadata.get("price_into_overhead_call_wall") else 0.0),
        0.0,
        10.0,
    )
    bullish_option_chain_pressure_score = _clamp(
        (max(-float(metadata.get("oi_pressure_imbalance") or 0.0), 0.0) * 6.0)
        + (2.0 if metadata.get("smart_money_bias") == "BULLISH" else 0.0)
        + (1.5 if metadata.get("option_chain_pressure_state") in {"DOWNSIDE_PUT_SUPPORT", "BULLISH_WALL_SHIFT"} else 0.0)
        - (float(metadata.get("overhead_call_pressure_score") or 0.0) * 0.75),
        0.0,
        10.0,
    )

    candidate_quotes = sorted(
        [quote for quote in snapshot.option_chain.quotes if quote.ltp > 0.0 and quote.bid >= 0.0 and quote.ask > 0.0],
        key=lambda quote: abs(quote.strike - spot),
    )[:8]
    avg_spread_ratio = (
        sum((quote.ask - quote.bid) / max(quote.ltp, 0.01) for quote in candidate_quotes) / len(candidate_quotes)
        if candidate_quotes
        else 0.25
    )
    liquidity_score = _clamp((0.20 - avg_spread_ratio) / 0.20 * 5.0, 0.0, 5.0)
    slippage_penalty = _clamp(snapshot.slippage_points / 1.5, 0.0, 2.5)
    bearish_expected_stop_pain = _clamp((120.0 - min(float(metadata.get("open_space_down") or 0.0), 120.0)) / 30.0, 0.0, 4.0)
    bullish_expected_stop_pain = _clamp((120.0 - min(float(metadata.get("open_space_up") or 0.0), 120.0)) / 30.0, 0.0, 4.0)
    time_score = _clamp(time_remaining_minutes / 45.0, 0.0, 4.0)
    bearish_monetization_score = _clamp(
        liquidity_score
        + time_score
        + min(float(metadata.get("open_space_down") or 0.0), 250.0) / 100.0
        - slippage_penalty
        - bearish_expected_stop_pain,
        0.0,
        10.0,
    )
    bullish_monetization_score = _clamp(
        liquidity_score
        + time_score
        + min(float(metadata.get("open_space_up") or 0.0), 250.0) / 100.0
        - slippage_penalty
        - bullish_expected_stop_pain
        - float(metadata.get("overhead_call_pressure_score") or 0.0),
        0.0,
        10.0,
    )

    if market_state == "TRUE_RANGE":
        range_penalty_score = 4.25
    elif market_state == "DIRECTIONAL_BALANCE":
        range_penalty_score = 1.1
    else:
        range_penalty_score = 0.0
    if market_state == "TRUE_RANGE" and float(path_quality["candle_overlap_ratio"]) > 0.65:
        range_penalty_score += 2.0
    elif market_state == "DIRECTIONAL_BALANCE" and float(path_quality["candle_overlap_ratio"]) > 0.74:
        range_penalty_score += 0.6
    low_liquidity_penalty = _clamp((0.12 - liquidity_score) * 0.0 + max(0.0, (avg_spread_ratio - 0.08) * 25.0), 0.0, 5.0)
    late_session_penalty = 1.5 if time_remaining_minutes < 75 else 0.0
    monetization_failure_penalty = 2.5 if max(bearish_monetization_score, bullish_monetization_score) < 2.5 else 0.0
    conflicting_signal_penalty = 0.0
    if market_state == "TRANSITION" and market_state_bias == "BULLISH" and failure_type in {"FAILED_BREAKOUT", "FAILED_RECLAIM", "FAILED_BOUNCE", "BEARISH_REJECTION_TRANSITION"}:
        conflicting_signal_penalty += 1.5
    if market_state == "TRANSITION" and market_state_bias == "BEARISH" and failure_type == "ACCEPTED_BREAKOUT":
        conflicting_signal_penalty += 1.5
    no_trade_score = (
        0.35 * range_penalty_score
        + 0.20 * low_liquidity_penalty
        + 0.15 * late_session_penalty
        + 0.15 * monetization_failure_penalty
        + 0.15 * conflicting_signal_penalty
    )
    market_state_score = state_quality_score
    bearish_trade_score = (
        0.22 * bearish_trend_quality_score
        + 0.24 * bearish_failure_score
        + 0.16 * bearish_location_live_score
        + 0.16 * bearish_option_chain_pressure_score
        + 0.12 * bearish_monetization_score
        + 0.10 * market_state_score
    )
    bullish_trade_score = (
        0.20 * bullish_trend_quality_score
        + 0.12 * bullish_failure_score
        + 0.20 * bullish_location_live_score
        + 0.14 * bullish_option_chain_pressure_score
        + 0.14 * bullish_monetization_score
        + 0.20 * market_state_score
    )

    stable_range = bool(
        market_state == "TRUE_RANGE"
        and float(path_quality["candle_overlap_ratio"]) <= 0.72
        and float(metadata.get("ema_compression_score") or 0.0) >= 1.0
        and abs(float(metadata.get("oi_pressure_imbalance") or 0.0)) <= 0.10
    )
    effective_bearish_threshold = params.bearish_trade_score_threshold
    effective_bearish_margin = params.bearish_trade_margin
    if market_state == "TREND_DOWN":
        effective_bearish_threshold = max(5.9, params.bearish_trade_score_threshold - 0.3)
        effective_bearish_margin = max(1.1, params.bearish_trade_margin - 0.3)
    elif market_state == "TRANSITION":
        effective_bearish_threshold = max(5.8, params.bearish_trade_score_threshold - 0.4)
        effective_bearish_margin = max(1.0, params.bearish_trade_margin - 0.4)
    elif market_state == "DIRECTIONAL_BALANCE":
        effective_bearish_threshold = max(5.45, params.bearish_trade_score_threshold - 0.9)
        effective_bearish_margin = max(0.8, params.bearish_trade_margin - 0.7)

    bearish_failure_context = failure_type in {"FAILED_BREAKOUT", "FAILED_RECLAIM", "FAILED_BOUNCE", "BEARISH_REJECTION_TRANSITION", "ACCEPTED_BREAKDOWN"}
    directional_balance_tradable = bool(
        market_state == "DIRECTIONAL_BALANCE"
        and market_state_bias == "BEARISH"
        and bearish_failure_context
        and bearish_failure_score >= 2.5
        and bearish_location_live_score >= 1.5
        and bearish_trade_score > no_trade_score + 0.35
    )
    if market_state in {"TREND_DOWN", "TRANSITION"} and bearish_failure_context and bearish_trade_score > no_trade_score + 0.05:
        tradability_class = "TRADABLE"
        tradability_reason = f"{market_state} with {failure_type} and aligned bearish structure/chain pressure is spread-tradable."
    elif directional_balance_tradable:
        tradability_class = "TRADABLE"
        tradability_reason = "Directional balance with bearish failure context is tradable even though the session is not a fully aligned trend yet."
    elif market_state == "TRUE_RANGE" and stable_range and bool(metadata.get("range_entry_ready")):
        tradability_class = "TRADABLE"
        tradability_reason = "Stable range with compressed EMAs and balanced option walls is tradable only through a strict condor."
    elif str(metadata.get("playbook") or "") == "LATE_SESSION_BULLISH_RECLAIM" or (
        market_state == "TREND_UP"
        and (
            float(metadata.get("open_space_up") or 0.0) < 100.0
            or bool(metadata.get("price_into_overhead_call_wall"))
            or time_remaining_minutes < 90
        )
    ):
        tradability_class = "LOW_EDGE"
        tradability_reason = "Bullish context is structurally valid but monetization edge is weak due to time, overhead pressure, or limited open space."
    else:
        tradability_class = "NOT_TRADABLE"
        tradability_reason = "Context does not offer a clean spread edge once state, location, path quality, and option-chain pressure are combined."

    bearish_shadow_subtype = _bearish_failed_breakout_transition_shadow_subtype(
        market_state=market_state,
        failure_type=failure_type,
        tradability_class=tradability_class,
        opening_range_break_state=opening_range_break_state,
        execution_5m=execution_5m,
        bearish_trade_score=bearish_trade_score,
        bearish_trade_score_threshold=effective_bearish_threshold,
        threshold_buffer=params.failed_breakout_transition_shadow_buffer,
        setup_quality_score=float(metadata.get("setup_quality_score") or 0.0),
        strong_setup_quality_threshold=params.strong_setup_quality_threshold,
        bearish_candle_quality_score=float(metadata.get("bearish_candle_quality_score") or 0.0),
        lower_high_confirmed=lower_high_confirmed,
        current_structure_signal=current_structure_signal,
        price_vs_vwap=metadata.get("price_vs_vwap"),  # type: ignore[arg-type]
        price_vs_ema20=metadata.get("price_vs_ema20"),  # type: ignore[arg-type]
        price_vs_or_high=metadata.get("price_vs_or_high"),  # type: ignore[arg-type]
        trend_efficiency_ratio=float(path_quality["trend_efficiency_ratio"]),
        momentum_persistence_score=float(path_quality["momentum_persistence_score"]),
    )
    metadata["bearish_shadow_subtype"] = bearish_shadow_subtype

    metadata["market_state_score"] = round(market_state_score, 4)
    metadata["trend_quality_score"] = round(max(bearish_trend_quality_score, bullish_trend_quality_score), 4)
    metadata["bearish_trend_quality_score"] = round(bearish_trend_quality_score, 4)
    metadata["bullish_trend_quality_score"] = round(bullish_trend_quality_score, 4)
    metadata["failure_score"] = round(max(bearish_failure_score, bullish_failure_score), 4)
    metadata["bearish_failure_score"] = round(bearish_failure_score, 4)
    metadata["bullish_failure_score"] = round(bullish_failure_score, 4)
    metadata["location_score"] = round(max(bearish_location_live_score, bullish_location_live_score), 4)
    metadata["bearish_location_live_score"] = round(bearish_location_live_score, 4)
    metadata["bullish_location_live_score"] = round(bullish_location_live_score, 4)
    metadata["option_chain_pressure_score"] = round(max(bearish_option_chain_pressure_score, bullish_option_chain_pressure_score), 4)
    metadata["bearish_option_chain_pressure_score"] = round(bearish_option_chain_pressure_score, 4)
    metadata["bullish_option_chain_pressure_score"] = round(bullish_option_chain_pressure_score, 4)
    metadata["monetization_score"] = round(max(bearish_monetization_score, bullish_monetization_score), 4)
    metadata["bearish_monetization_score"] = round(bearish_monetization_score, 4)
    metadata["bullish_monetization_score"] = round(bullish_monetization_score, 4)
    metadata["tradability_score"] = {"TRADABLE": 3.0, "LOW_EDGE": 1.0, "NOT_TRADABLE": -2.0}[tradability_class]
    metadata["tradability_class"] = tradability_class
    metadata["tradability_reason"] = tradability_reason
    metadata["bearish_trade_score"] = round(bearish_trade_score, 4)
    metadata["bullish_trade_score"] = round(bullish_trade_score, 4)
    metadata["no_trade_score"] = round(no_trade_score, 4)
    metadata["time_remaining_minutes"] = time_remaining_minutes
    metadata["liquidity_score"] = round(liquidity_score, 4)
    metadata["slippage_penalty"] = round(slippage_penalty, 4)
    metadata["range_penalty_score"] = round(range_penalty_score, 4)
    metadata["structure_is_monetizable"] = bool(max(bearish_monetization_score, bullish_monetization_score) >= 2.5)
    metadata["trade_score_margin_bearish"] = round(effective_bearish_margin, 4)
    metadata["trade_score_margin_bullish"] = params.bullish_trade_margin
    metadata["bearish_trade_score_threshold"] = round(effective_bearish_threshold, 4)
    metadata["bullish_trade_score_threshold"] = params.bullish_trade_score_threshold
    metadata["state_confidence"] = state_confidence_score
    if tradability_class == "TRADABLE":
        reasons.append(tradability_reason)
    else:
        reasons.append(f"Tradability classified as {tradability_class}: {tradability_reason}")

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
