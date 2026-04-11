from __future__ import annotations

from typing import Any, Dict, Optional


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, "", "-", "--"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def _round_strike(value: Optional[float], *, step: int = 50, direction: str = "nearest") -> Optional[int]:
    num = _safe_float(value)
    if num is None or step <= 0:
        return None
    scaled = float(num) / float(step)
    if direction == "up":
        rounded = int(-(-scaled // 1))
    elif direction == "down":
        rounded = int(scaled // 1)
    else:
        rounded = int(round(scaled))
    return int(rounded * step)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _bool_score(*values: Any) -> int:
    return sum(1 for value in values if bool(value))


def _pullback_state(
    *,
    spot: float,
    zone_low: float,
    zone_high: float,
    atr: float,
    bias_confidence: float,
) -> Dict[str, Any]:
    lower = min(float(zone_low), float(zone_high))
    upper = max(float(zone_low), float(zone_high))
    spot_val = float(spot)
    touched = lower <= spot_val <= upper
    distance = 0.0
    if not touched:
        distance = (lower - spot_val) if spot_val < lower else (spot_val - upper)
    tolerance = max(12.0, float(atr) * 0.45)
    if float(bias_confidence or 0.0) >= 0.80:
        tolerance *= 1.20
    elif float(bias_confidence or 0.0) >= 0.65:
        tolerance *= 1.10
    proximity_score = 1.0 if touched else max(0.0, 1.0 - (distance / max(tolerance, 1.0)))
    near = touched or distance <= tolerance
    ok = touched or (near and proximity_score >= 0.30)
    return {
        "touched": bool(touched),
        "near": bool(near),
        "ok": bool(ok),
        "distance_points": round(float(distance), 2),
        "tolerance_points": round(float(tolerance), 2),
        "score": round(float(_clamp(proximity_score, 0.0, 1.0)), 3),
    }


def _score_alignment(
    *,
    value: str,
    supportive: set[str],
    opposing: set[str],
    support_weight: float,
    opposing_weight: float,
    support_note: str,
    oppose_note: str,
    support_notes: list[str],
    oppose_notes: list[str],
) -> tuple[float, float]:
    normalized = str(value or "").upper()
    support_score = 0.0
    oppose_score = 0.0
    if normalized in supportive:
        support_score = float(support_weight)
        support_notes.append(support_note)
    elif normalized in opposing:
        oppose_score = float(opposing_weight)
        oppose_notes.append(oppose_note)
    return support_score, oppose_score


def _merge_pullback_zone(
    *,
    zone_low: float,
    zone_high: float,
    aux_low: Optional[float],
    aux_high: Optional[float],
    anchor: float,
    atr: float,
) -> tuple[float, float]:
    if aux_low is None or aux_high is None:
        return float(zone_low), float(zone_high)
    aux_low_f = min(float(aux_low), float(aux_high))
    aux_high_f = max(float(aux_low), float(aux_high))
    aux_mid = (aux_low_f + aux_high_f) / 2.0
    if abs(aux_mid - float(anchor)) > max(18.0, float(atr) * 1.35):
        return float(zone_low), float(zone_high)
    return min(float(zone_low), aux_low_f), max(float(zone_high), aux_high_f)


def _extract_fvg_context(
    *,
    bias: str,
    spot: float,
    atr: float,
    snap_5m: Dict[str, Any],
    snap_15m: Dict[str, Any],
) -> Dict[str, Any]:
    default = {
        "supportive": False,
        "bias": "NEUTRAL",
        "status": "NONE",
        "low": None,
        "high": None,
        "score": 0.0,
        "description": "No fair value gap context.",
    }
    tolerance = max(8.0, float(atr) * 0.40)
    for snap in (snap_5m, snap_15m):
        gap_bias = str(snap.get("fair_value_gap_bias") or "NEUTRAL").upper()
        if gap_bias != str(bias or "NEUTRAL").upper():
            continue
        low = _safe_float(snap.get("fair_value_gap_low"))
        high = _safe_float(snap.get("fair_value_gap_high"))
        if low is None or high is None:
            continue
        active = bool(snap.get("fair_value_gap_active")) or (float(low) - tolerance <= float(spot) <= float(high) + tolerance)
        return {
            "supportive": bool(active),
            "bias": gap_bias,
            "status": str(snap.get("fair_value_gap_status") or "NONE"),
            "low": round(float(low), 2),
            "high": round(float(high), 2),
            "score": round(float(_safe_float(snap.get("fair_value_gap_score")) or 0.0), 3),
            "description": str(snap.get("fair_value_gap_description") or "No fair value gap context."),
        }
    return default


def _extract_fib_context(
    *,
    bias: str,
    spot: float,
    atr: float,
    snap_5m: Dict[str, Any],
    snap_15m: Dict[str, Any],
) -> Dict[str, Any]:
    default = {
        "supportive": False,
        "low": None,
        "high": None,
        "description": "No Fibonacci pullback context.",
    }
    tolerance = max(8.0, float(atr) * 0.35)
    low_key = "fib_bullish_zone_low" if str(bias or "").upper() == "BULLISH" else "fib_bearish_zone_low"
    high_key = "fib_bullish_zone_high" if str(bias or "").upper() == "BULLISH" else "fib_bearish_zone_high"
    active_key = "fib_bullish_active" if str(bias or "").upper() == "BULLISH" else "fib_bearish_active"
    for snap in (snap_5m, snap_15m):
        low = _safe_float(snap.get(low_key))
        high = _safe_float(snap.get(high_key))
        if low is None or high is None:
            continue
        supportive = bool(snap.get(active_key)) or (float(low) - tolerance <= float(spot) <= float(high) + tolerance)
        return {
            "supportive": bool(supportive),
            "low": round(min(float(low), float(high)), 2),
            "high": round(max(float(low), float(high)), 2),
            "description": str(snap.get("fib_description") or "No Fibonacci pullback context."),
        }
    return default


def build_higher_tf_confluence(
    *,
    snap_15m: Dict[str, Any],
    snap_60m: Dict[str, Any],
    market_bias: str,
    rsi_divergence: Dict[str, Any],
) -> Dict[str, Any]:
    alignments: list[str] = []
    opposing_notes: list[str] = []
    support_score = 0.0
    oppose_score = 0.0
    patterns = [str(snap_15m.get("pattern") or "UNKNOWN"), str(snap_60m.get("pattern") or "UNKNOWN")]
    ema_alignments = [
        str(snap_15m.get("ema_alignment") or "NEUTRAL"),
        str(snap_60m.get("ema_alignment") or "NEUTRAL"),
    ]
    ema_trend_alignments = [
        str(snap_15m.get("ema_20_50_alignment") or "NEUTRAL"),
        str(snap_60m.get("ema_20_50_alignment") or "NEUTRAL"),
    ]
    price_action = [
        str(snap_15m.get("price_action_confirmation") or "NONE"),
        str(snap_60m.get("price_action_confirmation") or "NONE"),
    ]
    if market_bias == "BEARISH":
        scored = [
            _score_alignment(
                value=str(snap_15m.get("pattern") or ""),
                supportive={"DOWNTREND", "DOWN_BIAS"},
                opposing={"UPTREND", "UP_BIAS"},
                support_weight=0.32,
                opposing_weight=0.22,
                support_note="15m trend supports bearish idea",
                oppose_note="15m trend is still fighting the bearish idea",
                support_notes=alignments,
                oppose_notes=opposing_notes,
            ),
            _score_alignment(
                value=str(snap_60m.get("pattern") or ""),
                supportive={"DOWNTREND", "DOWN_BIAS"},
                opposing={"UPTREND", "UP_BIAS"},
                support_weight=0.38,
                opposing_weight=0.26,
                support_note="60m structure is also bearish",
                oppose_note="60m structure is still bullish against the setup",
                support_notes=alignments,
                oppose_notes=opposing_notes,
            ),
            _score_alignment(
                value=str(snap_15m.get("ema_alignment") or ""),
                supportive={"BEARISH"},
                opposing={"BULLISH"},
                support_weight=0.12,
                opposing_weight=0.08,
                support_note="15m EMA(5/20) is bearish",
                oppose_note="15m EMA alignment is still bullish",
                support_notes=alignments,
                oppose_notes=opposing_notes,
            ),
            _score_alignment(
                value=str(snap_60m.get("ema_alignment") or ""),
                supportive={"BEARISH"},
                opposing={"BULLISH"},
                support_weight=0.14,
                opposing_weight=0.10,
                support_note="60m EMA slope is bearish",
                oppose_note="60m EMA slope is still bullish",
                support_notes=alignments,
                oppose_notes=opposing_notes,
            ),
            _score_alignment(
                value=str(snap_15m.get("ema_20_50_alignment") or ""),
                supportive={"BEARISH"},
                opposing={"BULLISH"},
                support_weight=0.10,
                opposing_weight=0.08,
                support_note="15m EMA(20/50) trend is bearish",
                oppose_note="15m EMA(20/50) trend is still bullish",
                support_notes=alignments,
                oppose_notes=opposing_notes,
            ),
            _score_alignment(
                value=str(snap_60m.get("ema_20_50_alignment") or ""),
                supportive={"BEARISH"},
                opposing={"BULLISH"},
                support_weight=0.12,
                opposing_weight=0.09,
                support_note="60m EMA(20/50) trend is bearish",
                oppose_note="60m EMA(20/50) trend is still bullish",
                support_notes=alignments,
                oppose_notes=opposing_notes,
            ),
            _score_alignment(
                value=str(rsi_divergence.get("bias") or ""),
                supportive={"BEARISH"},
                opposing={"BULLISH"},
                support_weight=0.10,
                opposing_weight=0.08,
                support_note="RSI bearish divergence agrees with short idea",
                oppose_note="RSI divergence is bullish against the short idea",
                support_notes=alignments,
                oppose_notes=opposing_notes,
            ),
        ]
        support_score = sum(item[0] for item in scored)
        oppose_score = sum(item[1] for item in scored)
    elif market_bias == "BULLISH":
        scored = [
            _score_alignment(
                value=str(snap_15m.get("pattern") or ""),
                supportive={"UPTREND", "UP_BIAS"},
                opposing={"DOWNTREND", "DOWN_BIAS"},
                support_weight=0.32,
                opposing_weight=0.22,
                support_note="15m trend supports bullish idea",
                oppose_note="15m trend is still fighting the bullish idea",
                support_notes=alignments,
                oppose_notes=opposing_notes,
            ),
            _score_alignment(
                value=str(snap_60m.get("pattern") or ""),
                supportive={"UPTREND", "UP_BIAS"},
                opposing={"DOWNTREND", "DOWN_BIAS"},
                support_weight=0.38,
                opposing_weight=0.26,
                support_note="60m structure is also bullish",
                oppose_note="60m structure is still bearish against the setup",
                support_notes=alignments,
                oppose_notes=opposing_notes,
            ),
            _score_alignment(
                value=str(snap_15m.get("ema_alignment") or ""),
                supportive={"BULLISH"},
                opposing={"BEARISH"},
                support_weight=0.12,
                opposing_weight=0.08,
                support_note="15m EMA(5/20) is bullish",
                oppose_note="15m EMA alignment is still bearish",
                support_notes=alignments,
                oppose_notes=opposing_notes,
            ),
            _score_alignment(
                value=str(snap_60m.get("ema_alignment") or ""),
                supportive={"BULLISH"},
                opposing={"BEARISH"},
                support_weight=0.14,
                opposing_weight=0.10,
                support_note="60m EMA slope is bullish",
                oppose_note="60m EMA slope is still bearish",
                support_notes=alignments,
                oppose_notes=opposing_notes,
            ),
            _score_alignment(
                value=str(snap_15m.get("ema_20_50_alignment") or ""),
                supportive={"BULLISH"},
                opposing={"BEARISH"},
                support_weight=0.10,
                opposing_weight=0.08,
                support_note="15m EMA(20/50) trend is bullish",
                oppose_note="15m EMA(20/50) trend is still bearish",
                support_notes=alignments,
                oppose_notes=opposing_notes,
            ),
            _score_alignment(
                value=str(snap_60m.get("ema_20_50_alignment") or ""),
                supportive={"BULLISH"},
                opposing={"BEARISH"},
                support_weight=0.12,
                opposing_weight=0.09,
                support_note="60m EMA(20/50) trend is bullish",
                oppose_note="60m EMA(20/50) trend is still bearish",
                support_notes=alignments,
                oppose_notes=opposing_notes,
            ),
            _score_alignment(
                value=str(rsi_divergence.get("bias") or ""),
                supportive={"BULLISH"},
                opposing={"BEARISH"},
                support_weight=0.10,
                opposing_weight=0.08,
                support_note="RSI bullish divergence agrees with long-supportive idea",
                oppose_note="RSI divergence is bearish against the bullish idea",
                support_notes=alignments,
                oppose_notes=opposing_notes,
            ),
        ]
        support_score = sum(item[0] for item in scored)
        oppose_score = sum(item[1] for item in scored)
    bias_score = round(_clamp(float(support_score - oppose_score), -1.0, 1.0), 3)
    if not alignments and not opposing_notes:
        alignments.append("Higher timeframe structure is mixed.")
    notes = list(alignments[:3])
    if opposing_notes:
        notes.extend(opposing_notes[:1])
    normalized_score = round(_clamp((bias_score + 1.0) / 2.0, 0.0, 1.0), 3)
    return {
        "status": "ALIGNED" if bias_score >= 0.42 else ("PARTIAL" if bias_score >= 0.12 else "MIXED"),
        "score": normalized_score,
        "bias_score": bias_score,
        "support_score": round(float(support_score), 3),
        "opposing_score": round(float(oppose_score), 3),
        "weighted_ok": bool(bias_score >= 0.08),
        "patterns": patterns,
        "ema_alignments": ema_alignments,
        "ema_20_50_alignments": ema_trend_alignments,
        "price_action_confirmations": price_action,
        "notes": notes[:4],
    }


def _derive_orderflow_zones_from_runtime_context(
    *,
    spot: float,
    nearest_support: Optional[Dict[str, Any]],
    nearest_resistance: Optional[Dict[str, Any]],
    snap_5m: Dict[str, Any],
    structure_ctx: Dict[str, Any],
) -> Dict[str, Any]:
    default_zone = {"active": False, "level": None, "strength": 0.0, "summary": "No clear participation zone yet."}
    atr = max(12.0, float(_safe_float(snap_5m.get("atr_like_points")) or 0.0))
    price_action_bias = str(structure_ctx.get("price_action_bias") or "NEUTRAL").upper()
    price_action_confirmation = str(structure_ctx.get("price_action_confirmation") or "NONE").upper()
    retest_status = str(structure_ctx.get("retest_status") or "NONE").upper()
    primary_pattern = str(structure_ctx.get("primary_pattern") or snap_5m.get("primary_pattern") or "NONE").upper()
    pattern_set = {
        str(item or "").upper().split(":")[-1]
        for item in list(structure_ctx.get("price_action_patterns") or [])
        if str(item or "").strip()
    }

    support_level = _safe_float((nearest_support or {}).get("price"))
    resistance_level = _safe_float((nearest_resistance or {}).get("price"))
    buyers = dict(default_zone)
    sellers = dict(default_zone)

    if support_level is not None and abs(float(spot) - support_level) <= max(18.0, atr * 0.65):
        if (
            price_action_bias == "BULLISH"
            or price_action_confirmation in {"RETEST_CONFIRMED", "CANDLE_AND_RETEST_CONFIRMED", "REVERSAL_CONFIRMED"}
            or retest_status in {"SUPPORT_HOLD", "BREAKOUT_RETEST_HOLD", "FAILED_BREAKDOWN_RETEST", "DOUBLE_BOTTOM_NECKLINE_BREAK"}
            or primary_pattern in {"DOUBLE_BOTTOM_W_CONFIRMED", "FAKE_BREAKDOWN_DOWN", "LOWER_REJECTION", "EMA20_BREAKOUT_UP"}
            or bool(pattern_set.intersection({"DOUBLE_BOTTOM_W_CONFIRMED", "FAKE_BREAKDOWN_DOWN", "LOWER_REJECTION", "EMA20_BREAKOUT_UP"}))
        ):
            buyers = {
                "active": True,
                "level": round(float(support_level), 2),
                "strength": 0.72,
                "summary": f"Buyers are defending near {support_level:.0f}.",
            }

    if resistance_level is not None and abs(resistance_level - float(spot)) <= max(18.0, atr * 0.65):
        if (
            price_action_bias == "BEARISH"
            or price_action_confirmation in {"RETEST_CONFIRMED", "CANDLE_AND_RETEST_CONFIRMED", "REVERSAL_CONFIRMED"}
            or retest_status in {"RESISTANCE_HOLD", "BREAKDOWN_RETEST_HOLD", "FAILED_BREAKOUT_RETEST", "DOUBLE_TOP_NECKLINE_BREAK"}
            or primary_pattern in {"DOUBLE_TOP_M_CONFIRMED", "FAKE_BREAKOUT_UP", "UPPER_REJECTION", "EMA20_BREAKDOWN_DOWN"}
            or bool(pattern_set.intersection({"DOUBLE_TOP_M_CONFIRMED", "FAKE_BREAKOUT_UP", "UPPER_REJECTION", "EMA20_BREAKDOWN_DOWN"}))
        ):
            sellers = {
                "active": True,
                "level": round(float(resistance_level), 2),
                "strength": 0.72,
                "summary": f"Sellers are rejecting near {resistance_level:.0f}.",
            }
    return {"buyers": buyers, "sellers": sellers}


def _derive_rsi_divergence_from_runtime_context(
    *,
    market_bias: str,
    structure_ctx: Dict[str, Any],
    snap_5m: Dict[str, Any],
) -> Dict[str, Any]:
    explicit = structure_ctx.get("rsi_divergence") if isinstance(structure_ctx.get("rsi_divergence"), dict) else {}
    if explicit:
        return {
            "status": str(explicit.get("status") or "NONE"),
            "bias": str(explicit.get("bias") or "NEUTRAL"),
            "strength": float(_safe_float(explicit.get("strength")) or 0.0),
            "description": str(explicit.get("description") or "No RSI divergence."),
        }
    primary_pattern = str(structure_ctx.get("primary_pattern") or snap_5m.get("primary_pattern") or "NONE").upper()
    patterns = {
        str(item or "").upper().split(":")[-1]
        for item in list(structure_ctx.get("price_action_patterns") or [])
        if str(item or "").strip()
    }
    bullish_markers = {"DOUBLE_BOTTOM_W_CONFIRMED", "FAKE_BREAKDOWN_DOWN"}
    bearish_markers = {"DOUBLE_TOP_M_CONFIRMED", "FAKE_BREAKOUT_UP"}
    if primary_pattern in bullish_markers or bool(patterns.intersection(bullish_markers)):
        return {
            "status": "BULLISH_DIVERGENCE",
            "bias": "BULLISH",
            "strength": 0.45,
            "description": "Price action is showing a bullish reversal structure near support.",
        }
    if primary_pattern in bearish_markers or bool(patterns.intersection(bearish_markers)):
        return {
            "status": "BEARISH_DIVERGENCE",
            "bias": "BEARISH",
            "strength": 0.45,
            "description": "Price action is showing a bearish reversal structure near resistance.",
        }
    return {
        "status": "NONE",
        "bias": "NEUTRAL",
        "strength": 0.0,
        "description": "No RSI divergence.",
    }


def build_trade_plan(
    *,
    spot: float,
    snap_5m: Dict[str, Any],
    snap_15m: Dict[str, Any],
    snap_60m: Dict[str, Any],
    market_bias: str,
    bias_confidence: float,
    orb: Dict[str, Any],
    nearest_support: Optional[Dict[str, Any]],
    nearest_resistance: Optional[Dict[str, Any]],
    advisor: Dict[str, Any],
    buyer_seller_zones: Dict[str, Any],
    rsi_divergence: Dict[str, Any],
    higher_tf_confluence: Dict[str, Any],
) -> Dict[str, Any]:
    primary_pattern = str(snap_5m.get("primary_pattern") or snap_15m.get("primary_pattern") or "NONE").upper()
    retest_status = str(snap_5m.get("retest_status") or snap_15m.get("retest_status") or "NONE").upper()
    price_action_confirmation = str(snap_5m.get("price_action_confirmation") or "NONE").upper()
    ema20 = _safe_float(snap_5m.get("ema_slow"))
    ema50 = _safe_float(snap_5m.get("ema_base"))
    ema2050_5m = str(snap_5m.get("ema_20_50_alignment") or "NEUTRAL").upper()
    ema2050_15m = str(snap_15m.get("ema_20_50_alignment") or "NEUTRAL").upper()
    strong_candle_bias = str(snap_5m.get("strong_candle_bias") or "NEUTRAL").upper()
    strong_candle_ok = bool(snap_5m.get("strong_candle_confirmed"))
    strong_candle_status = str(snap_5m.get("strong_candle_status") or "NONE").upper()
    strong_candle_desc = str(snap_5m.get("strong_candle_description") or "No strong 5m confirmation candle.")
    atr = max(
        15.0,
        float(_safe_float(snap_5m.get("atr_like_points")) or 0.0),
        float(_safe_float(snap_15m.get("atr_like_points")) or 0.0) * 0.75,
    )
    pullback_buffer = max(10.0, atr * 0.35)
    option_context = (advisor.get("market_context") or {}).get("option_chain") if isinstance(advisor.get("market_context"), dict) else {}
    option_context = option_context if isinstance(option_context, dict) else {}
    trend_context = (advisor.get("market_context") or {}).get("trend") if isinstance(advisor.get("market_context"), dict) else {}
    trend_context = trend_context if isinstance(trend_context, dict) else {}
    pcr_bias = str(advisor.get("pcr_bias") or option_context.get("pcr_bias") or "NEUTRAL").upper()
    oi_build_bias = str(advisor.get("oi_build_bias") or (option_context.get("oi_build") or {}).get("bias") or "UNKNOWN").upper()
    breakout_confirmation = str(
        advisor.get("breakout_confirmation")
        or trend_context.get("breakout_confirmation")
        or orb.get("breakout_confirmation")
        or "NONE"
    ).upper()
    call_wall = option_context.get("call_wall_above") if isinstance(option_context.get("call_wall_above"), dict) else {}
    put_wall = option_context.get("put_wall_below") if isinstance(option_context.get("put_wall_below"), dict) else {}
    bearish_patterns = {
        "DOUBLE_TOP_M_CONFIRMED",
        "FAKE_BREAKOUT_UP",
        "BEARISH_ENGULFING",
        "UPPER_REJECTION",
        "EMA20_BREAKDOWN_DOWN",
        "BREAKOUT_CONTINUATION_DOWN",
    }
    bullish_patterns = {
        "DOUBLE_BOTTOM_W_CONFIRMED",
        "FAKE_BREAKDOWN_DOWN",
        "BULLISH_ENGULFING",
        "LOWER_REJECTION",
        "EMA20_BREAKOUT_UP",
        "BREAKOUT_CONTINUATION_UP",
    }

    plan: Dict[str, Any] = {
        "trade_idea": "No directional trade yet",
        "posture": "NO_TRADE",
        "setup_type": None,
        "setup_family": None,
        "entry_ready": False,
        "continuation_ready": False,
        "continuation_text": None,
        "reversal_ready": False,
        "reversal_text": None,
        "pullback_touched": False,
        "pullback_near": False,
        "pullback_ok": False,
        "pullback_score": 0.0,
        "pullback_distance_points": None,
        "pullback_tolerance_points": None,
        "confirmation_score": 0.0,
        "confirmation_support_count": 0,
        "confirmation_ready": False,
        "fake_breakout_conflict": False,
        "ema_2050_ok": False,
        "ema_2050_alignment_5m": ema2050_5m,
        "ema_2050_alignment_15m": ema2050_15m,
        "higher_tf_ok": False,
        "higher_tf_score": 0.0,
        "higher_tf_display_score": 0.5,
        "buyer_seller_ok": False,
        "strong_candle_ok": False,
        "strong_candle_bias": strong_candle_bias,
        "strong_candle_status": strong_candle_status,
        "strong_candle_text": strong_candle_desc,
        "fvg_supportive": False,
        "fair_value_gap_low": None,
        "fair_value_gap_high": None,
        "fib_supportive": False,
        "fib_zone_low": None,
        "fib_zone_high": None,
        "divergence_supportive": True,
        "structure_conflict_score": 0.0,
        "option_chain_conflict_score": 0.0,
        "reversal_conflict_score": 0.0,
        "total_conflict_score": 0.0,
        "trigger_window": "Wait for structure to align.",
        "entry_trigger_text": "Wait for pullback plus confirmation before selling premium.",
        "confirmation": "Need pullback plus confirmation before selling premium.",
        "invalidation": "Stand aside while price is between decision zones.",
        "invalidation_spot": None,
        "strike_hint": "No strike hint while structure is unclear.",
        "reason_summary": [],
        "entry_block_reasons": [],
        "pullback_zone_low": None,
        "pullback_zone_high": None,
        "entry_zone_mid": None,
        "short_strike_hint": None,
        "hedge_strike_hint": None,
        "required_confluences": [],
    }

    if market_bias == "BEARISH":
        fib_context = _extract_fib_context(
            bias="BEARISH",
            spot=spot,
            atr=atr,
            snap_5m=snap_5m,
            snap_15m=snap_15m,
        )
        fvg_context = _extract_fvg_context(
            bias="BEARISH",
            spot=spot,
            atr=atr,
            snap_5m=snap_5m,
            snap_15m=snap_15m,
        )
        candidates = [
            _safe_float((nearest_resistance or {}).get("price")),
            _safe_float(snap_15m.get("resistance")),
            _safe_float(orb.get("orb_high")),
            ema20,
        ]
        candidates = [value for value in candidates if value is not None]
        resistance_anchor = max(candidates) if candidates else float(spot) + atr
        zone_low = round(float(resistance_anchor - pullback_buffer), 2)
        zone_high = round(float(resistance_anchor + pullback_buffer), 2)
        zone_low, zone_high = _merge_pullback_zone(
            zone_low=zone_low,
            zone_high=zone_high,
            aux_low=_safe_float(fib_context.get("low")),
            aux_high=_safe_float(fib_context.get("high")),
            anchor=resistance_anchor,
            atr=atr,
        )
        pullback = _pullback_state(
            spot=float(spot),
            zone_low=zone_low,
            zone_high=zone_high,
            atr=atr,
            bias_confidence=bias_confidence,
        )
        confirmed = (
            primary_pattern in bearish_patterns
            or price_action_confirmation in {"REVERSAL_CONFIRMED", "BREAKOUT_CONFIRMED", "CANDLE_AND_RETEST_CONFIRMED"}
            or retest_status in {"RESISTANCE_HOLD", "BREAKDOWN_RETEST_HOLD", "FAILED_BREAKOUT_RETEST", "DOUBLE_TOP_NECKLINE_BREAK"}
            or bool(fvg_context.get("supportive"))
        )
        fake_break_conflict = primary_pattern in {"FAKE_BREAKDOWN_DOWN", "DOUBLE_BOTTOM_W_CONFIRMED"}
        seller_active = bool((buyer_seller_zones.get("sellers") or {}).get("active")) or bool(fvg_context.get("supportive"))
        chain_seller_supportive = pcr_bias == "BEARISH" or oi_build_bias in {"BEARISH", "BEARISH_RESISTANCE"}
        htf_score = float(_safe_float(higher_tf_confluence.get("bias_score")) or 0.0)
        htf_aligned = bool(higher_tf_confluence.get("weighted_ok")) or htf_score >= 0.08
        divergence_supportive = str(rsi_divergence.get("bias") or "NEUTRAL").upper() in {"BEARISH", "NEUTRAL"}
        ema2050_price_ok = bool(
            ema20 is not None
            and ema50 is not None
            and float(ema20) <= float(ema50) + max(6.0, atr * 0.08)
        )
        ema_2050_ok = (ema2050_5m == "BEARISH" or ema2050_price_ok) and ema2050_15m in {"BEARISH", "NEUTRAL"}
        strong_candle_match = strong_candle_ok and strong_candle_bias == "BEARISH"
        continuation_evidence = bool(
            breakout_confirmation == "DOWN_CONFIRMED"
            or primary_pattern in {"BREAKOUT_CONTINUATION_DOWN", "EMA20_BREAKDOWN_DOWN"}
            or price_action_confirmation in {"BREAKOUT_CONFIRMED", "CANDLE_AND_RETEST_CONFIRMED"}
            or retest_status in {"BREAKDOWN_RETEST_HOLD"}
        )
        reversal_evidence = bool(
            primary_pattern in {"DOUBLE_TOP_M_CONFIRMED", "FAKE_BREAKOUT_UP", "BEARISH_ENGULFING", "UPPER_REJECTION"}
            or price_action_confirmation == "REVERSAL_CONFIRMED"
            or retest_status in {"RESISTANCE_HOLD", "FAILED_BREAKOUT_RETEST", "DOUBLE_TOP_NECKLINE_BREAK"}
        )
        htf_supportive = bool(higher_tf_confluence.get("weighted_ok")) or htf_score >= 0.0
        confirmation_support_count = _bool_score(
            confirmed or strong_candle_match,
            seller_active or chain_seller_supportive,
            htf_supportive,
        )
        confirmation_score = round(float(confirmation_support_count) / 3.0, 3)
        structure_conflict_score = 0.0
        if breakout_confirmation == "UP_CONFIRMED":
            structure_conflict_score += 0.42
        elif breakout_confirmation == "NONE":
            structure_conflict_score += 0.08
        if ema2050_15m == "BULLISH":
            structure_conflict_score += 0.18
        if str(snap_15m.get("pattern") or "").upper() in {"UPTREND", "UP_BIAS"}:
            structure_conflict_score += 0.18
        if str(snap_60m.get("pattern") or "").upper() in {"UPTREND", "UP_BIAS"}:
            structure_conflict_score += 0.18
        option_chain_conflict_score = 0.0
        if pcr_bias == "BULLISH":
            option_chain_conflict_score += 0.22
        if oi_build_bias in {"BULLISH", "BULLISH_SUPPORT"}:
            option_chain_conflict_score += 0.22
        try:
            call_wall_dist = float(_safe_float(call_wall.get("distance_from_spot")) or 0.0)
            if call_wall_dist <= 0:
                option_chain_conflict_score += 0.08
        except Exception:
            pass
        reversal_conflict_score = 0.0
        if fake_break_conflict:
            reversal_conflict_score += 0.45
        if not divergence_supportive:
            reversal_conflict_score += 0.18
        if reversal_evidence and breakout_confirmation == "UP_CONFIRMED" and retest_status not in {"FAILED_BREAKOUT_RETEST", "DOUBLE_TOP_NECKLINE_BREAK"}:
            reversal_conflict_score += 0.20
        total_conflict_score = round(
            _clamp(structure_conflict_score * 0.45 + option_chain_conflict_score * 0.30 + reversal_conflict_score * 0.25, 0.0, 1.0),
            3,
        )
        setup_family = "CONTINUATION" if continuation_evidence and breakout_confirmation == "DOWN_CONFIRMED" else ("REVERSAL" if reversal_evidence else "PULLBACK")
        continuation_context_ready = bool(
            continuation_evidence
            and bias_confidence >= 0.72
            and ema_2050_ok
            and (seller_active or chain_seller_supportive)
            and confirmation_support_count >= 2
            and not fake_break_conflict
        )
        continuation_trigger_ready = bool(
            primary_pattern in {"BREAKOUT_CONTINUATION_DOWN", "EMA20_BREAKDOWN_DOWN"}
            or price_action_confirmation in {"BREAKOUT_CONFIRMED", "CANDLE_AND_RETEST_CONFIRMED"}
            or retest_status in {"BREAKDOWN_RETEST_HOLD", "RESISTANCE_HOLD", "FAILED_BREAKOUT_RETEST"}
            or strong_candle_match
        )
        continuation_far_ready = bool(
            not bool(pullback.get("ok"))
            and float(pullback.get("distance_points") or 0.0) >= max(20.0, float(pullback.get("tolerance_points") or 0.0) * 1.20)
            and (htf_supportive or htf_score >= -0.02)
            and continuation_trigger_ready
        )
        continuation_near_ready = bool(
            (bool(pullback.get("near")) or float(pullback.get("score") or 0.0) >= 0.22)
            and not bool(pullback.get("touched"))
            and (htf_supportive or htf_score >= 0.08 or bool(fvg_context.get("supportive")) or bool(fib_context.get("supportive")))
            and (
                strong_candle_match
                or (confirmed and (bool(fvg_context.get("supportive")) or bool(fib_context.get("supportive"))))
                or ((seller_active and chain_seller_supportive) and htf_score >= 0.08)
            )
            and continuation_trigger_ready
        )
        continuation_ready = bool(continuation_context_ready and (continuation_far_ready or continuation_near_ready))
        reversal_ready = bool(
            reversal_evidence
            and bool(pullback.get("ok"))
            and strong_candle_match
            and ema_2050_ok
            and confirmation_support_count >= 2
            and divergence_supportive
            and not fake_break_conflict
            and (
                breakout_confirmation != "UP_CONFIRMED"
                or retest_status in {"FAILED_BREAKOUT_RETEST", "DOUBLE_TOP_NECKLINE_BREAK"}
                or primary_pattern in {"DOUBLE_TOP_M_CONFIRMED", "FAKE_BREAKOUT_UP"}
            )
        )
        entry_block_reasons = []
        if not bool(pullback.get("ok")) and not continuation_ready:
            entry_block_reasons.append("PULLBACK_ZONE_NOT_REACHED")
        if not confirmed and not reversal_ready:
            entry_block_reasons.append("CONFIRMATION_MISSING")
        if not ema_2050_ok:
            entry_block_reasons.append("EMA2050_CONFIRMATION_MISSING")
        if bool(pullback.get("ok")) and not continuation_ready and not strong_candle_match:
            entry_block_reasons.append("STRONG_5M_CANDLE_MISSING")
        if fake_break_conflict:
            entry_block_reasons.append("FAKE_BREAKOUT_CONFLICT")
        if not seller_active and not continuation_ready and not reversal_ready:
            entry_block_reasons.append("SELLER_ZONE_UNCONFIRMED")
        if confirmation_support_count < 2:
            entry_block_reasons.append("INSUFFICIENT_CONFIRMATION_SCORE")
        if not htf_aligned and confirmation_support_count < 2:
            entry_block_reasons.append("HTF_CONFLUENCE_MISSING")
        if not divergence_supportive:
            entry_block_reasons.append("RSI_DIVERGENCE_AGAINST_SETUP")
        if setup_family == "REVERSAL" and breakout_confirmation == "UP_CONFIRMED" and not reversal_ready:
            entry_block_reasons.append("REVERSAL_NOT_CONFIRMED")
        posture = "WAIT_PULLBACK"
        if bool(pullback.get("near")):
            posture = "WAIT_CONFIRMATION"
        if continuation_ready:
            posture = "READY_TREND_CONTINUATION"
        elif reversal_ready:
            posture = "READY_REVERSAL"
        if not entry_block_reasons:
            posture = "READY_TREND_CONTINUATION" if continuation_ready else ("READY_REVERSAL" if reversal_ready else "READY_AFTER_CONFIRMATION")
        short_hint = _round_strike(
            max(
                zone_high,
                float(_safe_float(call_wall.get("strike")) or zone_high),
            ),
            direction="up",
        )
        hedge_gap = 150 if str(snap_15m.get("volatility_regime") or "").upper() == "HIGH" else 100
        hedge_hint = None if short_hint is None else short_hint + hedge_gap
        invalidation_spot = round(zone_high + max(8.0, atr * 0.2), 2)
        confluences = [
            "Pullback into resistance / EMA20 zone",
            "5m and 15m EMA20/50 trend stays bearish",
            "Strong red 5m confirmation candle after pullback",
            "Bearish rejection or fake breakout failure",
            "Seller presence visible at resistance or bearish FVG",
            "15m / 60m structure stays bearish",
        ]
        if bool(fib_context.get("supportive")):
            confluences.append("Pullback is sitting inside the bearish Fibonacci retracement pocket")
        if bool(fvg_context.get("supportive")):
            confluences.append("Bearish fair value gap is active near the decision zone")
        if str(rsi_divergence.get("status") or "NONE").upper() != "NONE":
            confluences.append(str(rsi_divergence.get("description") or "RSI divergence confirmation"))
        plan.update(
            {
                "trade_idea": "Bear call spread",
                "posture": posture,
                "setup_type": "CALL_CREDIT_SPREAD",
                "setup_family": setup_family,
                "entry_ready": not entry_block_reasons,
                "continuation_ready": continuation_ready,
                "continuation_text": (
                    "Strong bearish continuation is confirmed; a perfect pullback touch is no longer required."
                    if continuation_ready
                    else None
                ),
                "reversal_ready": reversal_ready,
                "reversal_text": (
                    "Bearish reversal is confirmed after pullback into resistance; sell only after the rejection holds."
                    if reversal_ready
                    else None
                ),
                "pullback_touched": bool(pullback.get("touched")),
                "pullback_near": bool(pullback.get("near")),
                "pullback_ok": bool(pullback.get("ok")),
                "pullback_score": float(pullback.get("score") or 0.0),
                "pullback_distance_points": _safe_float(pullback.get("distance_points")),
                "pullback_tolerance_points": _safe_float(pullback.get("tolerance_points")),
                "confirmation_score": confirmation_score,
                "confirmation_support_count": confirmation_support_count,
                "confirmation_ready": bool((continuation_ready or reversal_ready) or (confirmed and confirmation_support_count >= 2 and strong_candle_match)),
                "fake_breakout_conflict": fake_break_conflict,
                "ema_2050_ok": bool(ema_2050_ok),
                "ema_2050_alignment_5m": ema2050_5m,
                "ema_2050_alignment_15m": ema2050_15m,
                "higher_tf_ok": bool(htf_supportive),
                "higher_tf_score": round(float(htf_score), 3),
                "higher_tf_display_score": _safe_float(higher_tf_confluence.get("score")) if _safe_float(higher_tf_confluence.get("score")) is not None else 0.5,
                "buyer_seller_ok": bool(seller_active or continuation_ready or reversal_ready),
                "strong_candle_ok": bool(strong_candle_match),
                "strong_candle_bias": strong_candle_bias,
                "strong_candle_status": strong_candle_status,
                "strong_candle_text": strong_candle_desc,
                "fvg_supportive": bool(fvg_context.get("supportive")),
                "fair_value_gap_low": _safe_float(fvg_context.get("low")),
                "fair_value_gap_high": _safe_float(fvg_context.get("high")),
                "fib_supportive": bool(fib_context.get("supportive")),
                "fib_zone_low": _safe_float(fib_context.get("low")),
                "fib_zone_high": _safe_float(fib_context.get("high")),
                "divergence_supportive": divergence_supportive,
                "structure_conflict_score": round(float(_clamp(structure_conflict_score, 0.0, 1.0)), 3),
                "option_chain_conflict_score": round(float(_clamp(option_chain_conflict_score, 0.0, 1.0)), 3),
                "reversal_conflict_score": round(float(_clamp(reversal_conflict_score, 0.0, 1.0)), 3),
                "total_conflict_score": total_conflict_score,
                "trigger_window": (
                    f"Trend is strong enough for continuation below resistance; preferred pullback zone remains {zone_low:.0f}-{zone_high:.0f}."
                    if continuation_ready
                    else (
                        f"Bearish reversal is ready near {zone_low:.0f}-{zone_high:.0f}; sell only if the rejection holds."
                        if reversal_ready
                        else f"Wait for pullback into {zone_low:.0f}-{zone_high:.0f}, then sell only after bearish rejection or failed upside break."
                    )
                ),
                "entry_trigger_text": (
                    "Bearish continuation is confirmed; a fresh lower-high / breakdown-hold is enough to sell the call spread."
                    if continuation_ready
                    else (
                        f"Bearish reversal is confirmed near {zone_low:.0f}-{zone_high:.0f}; sell only after the neckline/rejection holds."
                        if reversal_ready
                        else f"Enter only after NIFTY pulls back into {zone_low:.0f}-{zone_high:.0f} and prints bearish confirmation."
                    )
                ),
                "confirmation": "Need lower high, rejection candle, fake breakout failure, or M-pattern confirmation near resistance/EMA20.",
                "invalidation": f"Avoid entry if a 5m close reclaims {zone_high:.0f} with acceptance above the pullback zone.",
                "invalidation_spot": invalidation_spot,
                "strike_hint": f"Prefer short call above {short_hint or '—'} and hedge near {hedge_hint or '—'} once confirmation prints.",
                "reason_summary": [
                    f"Bias {market_bias.lower()} with confidence {round(bias_confidence * 100)}%",
                    f"5m/15m pattern: {snap_5m.get('pattern') or '—'} / {snap_15m.get('pattern') or '—'}",
                    f"Pattern confirmation: {primary_pattern if primary_pattern != 'NONE' else price_action_confirmation}",
                    f"EMA20/50 trend: 5m={ema2050_5m}, 15m={ema2050_15m}, EMA20={ema20 if ema20 is not None else '—'}, EMA50={ema50 if ema50 is not None else '—'}",
                    f"PCR/OI: {advisor.get('pcr_bias') or 'NEUTRAL'} / {advisor.get('oi_build_bias') or 'UNKNOWN'}",
                    f"Pullback quality {int(round(float(pullback.get('score') or 0.0) * 100))}% within ATR tolerance {float(pullback.get('tolerance_points') or 0.0):.0f} pts.",
                    strong_candle_desc,
                    (buyer_seller_zones.get("sellers") or {}).get("summary") or "No clear seller zone yet.",
                    str(fvg_context.get("description") or "No fair value gap context."),
                    str(fib_context.get("description") or "No Fibonacci pullback context."),
                    str(rsi_divergence.get("description") or "No RSI divergence."),
                    f"HTF score {float(htf_score):+.2f}: " + "; ".join(higher_tf_confluence.get("notes") or ["Higher timeframe structure is mixed."]),
                    f"Confirmation score {confirmation_support_count}/3, conflicts structure={structure_conflict_score:.2f}, chain={option_chain_conflict_score:.2f}, reversal={reversal_conflict_score:.2f}.",
                    (
                        "Trend continuation path is active; exact pullback touch is waived because sellers are defending momentum and confluence remains supportive."
                        if continuation_ready
                        else (
                            "Reversal path is active; continuation is not required because resistance rejection is already visible."
                            if reversal_ready
                            else "Continuation path is not active; a pullback into resistance plus a strong bearish 5m candle is required."
                        )
                    ),
                ],
                "entry_block_reasons": entry_block_reasons,
                "pullback_zone_low": zone_low,
                "pullback_zone_high": zone_high,
                "entry_zone_mid": round((zone_low + zone_high) / 2.0, 2),
                "short_strike_hint": short_hint,
                "hedge_strike_hint": hedge_hint,
                "required_confluences": confluences,
            }
        )
        return plan

    if market_bias == "BULLISH":
        fib_context = _extract_fib_context(
            bias="BULLISH",
            spot=spot,
            atr=atr,
            snap_5m=snap_5m,
            snap_15m=snap_15m,
        )
        fvg_context = _extract_fvg_context(
            bias="BULLISH",
            spot=spot,
            atr=atr,
            snap_5m=snap_5m,
            snap_15m=snap_15m,
        )
        candidates = [
            _safe_float((nearest_support or {}).get("price")),
            _safe_float(snap_15m.get("support")),
            _safe_float(orb.get("orb_low")),
            ema20,
        ]
        candidates = [value for value in candidates if value is not None]
        support_anchor = min(candidates) if candidates else float(spot) - atr
        zone_low = round(float(support_anchor - pullback_buffer), 2)
        zone_high = round(float(support_anchor + pullback_buffer), 2)
        zone_low, zone_high = _merge_pullback_zone(
            zone_low=zone_low,
            zone_high=zone_high,
            aux_low=_safe_float(fib_context.get("low")),
            aux_high=_safe_float(fib_context.get("high")),
            anchor=support_anchor,
            atr=atr,
        )
        pullback = _pullback_state(
            spot=float(spot),
            zone_low=zone_low,
            zone_high=zone_high,
            atr=atr,
            bias_confidence=bias_confidence,
        )
        confirmed = (
            primary_pattern in bullish_patterns
            or price_action_confirmation in {"REVERSAL_CONFIRMED", "BREAKOUT_CONFIRMED", "CANDLE_AND_RETEST_CONFIRMED"}
            or retest_status in {"SUPPORT_HOLD", "BREAKOUT_RETEST_HOLD", "FAILED_BREAKDOWN_RETEST", "DOUBLE_BOTTOM_NECKLINE_BREAK"}
            or bool(fvg_context.get("supportive"))
        )
        fake_break_conflict = primary_pattern in {"FAKE_BREAKOUT_UP", "DOUBLE_TOP_M_CONFIRMED"}
        buyer_active = bool((buyer_seller_zones.get("buyers") or {}).get("active")) or bool(fvg_context.get("supportive"))
        chain_buyer_supportive = pcr_bias == "BULLISH" or oi_build_bias in {"BULLISH", "BULLISH_SUPPORT"}
        htf_score = float(_safe_float(higher_tf_confluence.get("bias_score")) or 0.0)
        htf_aligned = bool(higher_tf_confluence.get("weighted_ok")) or htf_score >= 0.08
        divergence_supportive = str(rsi_divergence.get("bias") or "NEUTRAL").upper() in {"BULLISH", "NEUTRAL"}
        ema2050_price_ok = bool(
            ema20 is not None
            and ema50 is not None
            and float(ema20) >= float(ema50) - max(6.0, atr * 0.08)
        )
        ema_2050_ok = (ema2050_5m == "BULLISH" or ema2050_price_ok) and ema2050_15m in {"BULLISH", "NEUTRAL"}
        strong_candle_match = strong_candle_ok and strong_candle_bias == "BULLISH"
        continuation_evidence = bool(
            breakout_confirmation == "UP_CONFIRMED"
            or primary_pattern in {"BREAKOUT_CONTINUATION_UP", "EMA20_BREAKOUT_UP"}
            or price_action_confirmation in {"BREAKOUT_CONFIRMED", "CANDLE_AND_RETEST_CONFIRMED"}
            or retest_status in {"BREAKOUT_RETEST_HOLD"}
        )
        reversal_evidence = bool(
            primary_pattern in {"DOUBLE_BOTTOM_W_CONFIRMED", "FAKE_BREAKDOWN_DOWN", "BULLISH_ENGULFING", "LOWER_REJECTION"}
            or price_action_confirmation == "REVERSAL_CONFIRMED"
            or retest_status in {"SUPPORT_HOLD", "FAILED_BREAKDOWN_RETEST", "DOUBLE_BOTTOM_NECKLINE_BREAK"}
        )
        htf_supportive = bool(higher_tf_confluence.get("weighted_ok")) or htf_score >= 0.0
        confirmation_support_count = _bool_score(
            confirmed or strong_candle_match,
            buyer_active or chain_buyer_supportive,
            htf_supportive,
        )
        confirmation_score = round(float(confirmation_support_count) / 3.0, 3)
        structure_conflict_score = 0.0
        if breakout_confirmation == "DOWN_CONFIRMED":
            structure_conflict_score += 0.42
        elif breakout_confirmation == "NONE":
            structure_conflict_score += 0.08
        if ema2050_15m == "BEARISH":
            structure_conflict_score += 0.18
        if str(snap_15m.get("pattern") or "").upper() in {"DOWNTREND", "DOWN_BIAS"}:
            structure_conflict_score += 0.18
        if str(snap_60m.get("pattern") or "").upper() in {"DOWNTREND", "DOWN_BIAS"}:
            structure_conflict_score += 0.18
        option_chain_conflict_score = 0.0
        if pcr_bias == "BEARISH":
            option_chain_conflict_score += 0.22
        if oi_build_bias in {"BEARISH", "BEARISH_RESISTANCE"}:
            option_chain_conflict_score += 0.22
        try:
            put_wall_dist = float(_safe_float(put_wall.get("distance_from_spot")) or 0.0)
            if put_wall_dist <= 0:
                option_chain_conflict_score += 0.08
        except Exception:
            pass
        reversal_conflict_score = 0.0
        if fake_break_conflict:
            reversal_conflict_score += 0.45
        if not divergence_supportive:
            reversal_conflict_score += 0.18
        if reversal_evidence and breakout_confirmation == "DOWN_CONFIRMED" and retest_status not in {"FAILED_BREAKDOWN_RETEST", "DOUBLE_BOTTOM_NECKLINE_BREAK"}:
            reversal_conflict_score += 0.20
        total_conflict_score = round(
            _clamp(structure_conflict_score * 0.45 + option_chain_conflict_score * 0.30 + reversal_conflict_score * 0.25, 0.0, 1.0),
            3,
        )
        setup_family = "CONTINUATION" if continuation_evidence and breakout_confirmation == "UP_CONFIRMED" else ("REVERSAL" if reversal_evidence else "PULLBACK")
        continuation_context_ready = bool(
            continuation_evidence
            and bias_confidence >= 0.72
            and ema_2050_ok
            and (buyer_active or chain_buyer_supportive)
            and confirmation_support_count >= 2
            and not fake_break_conflict
        )
        continuation_trigger_ready = bool(
            primary_pattern in {"BREAKOUT_CONTINUATION_UP", "EMA20_BREAKOUT_UP"}
            or price_action_confirmation in {"BREAKOUT_CONFIRMED", "CANDLE_AND_RETEST_CONFIRMED"}
            or retest_status in {"BREAKOUT_RETEST_HOLD", "SUPPORT_HOLD", "FAILED_BREAKDOWN_RETEST"}
            or strong_candle_match
        )
        continuation_far_ready = bool(
            not bool(pullback.get("ok"))
            and float(pullback.get("distance_points") or 0.0) >= max(20.0, float(pullback.get("tolerance_points") or 0.0) * 1.20)
            and (htf_supportive or htf_score >= -0.02)
            and continuation_trigger_ready
        )
        continuation_near_ready = bool(
            (bool(pullback.get("near")) or float(pullback.get("score") or 0.0) >= 0.22)
            and not bool(pullback.get("touched"))
            and (htf_supportive or htf_score >= 0.08 or bool(fvg_context.get("supportive")) or bool(fib_context.get("supportive")))
            and (
                strong_candle_match
                or (confirmed and (bool(fvg_context.get("supportive")) or bool(fib_context.get("supportive"))))
                or ((buyer_active and chain_buyer_supportive) and htf_score >= 0.08)
            )
            and continuation_trigger_ready
        )
        continuation_ready = bool(continuation_context_ready and (continuation_far_ready or continuation_near_ready))
        reversal_ready = bool(
            reversal_evidence
            and bool(pullback.get("ok"))
            and strong_candle_match
            and ema_2050_ok
            and confirmation_support_count >= 2
            and divergence_supportive
            and not fake_break_conflict
            and (
                breakout_confirmation != "DOWN_CONFIRMED"
                or retest_status in {"FAILED_BREAKDOWN_RETEST", "DOUBLE_BOTTOM_NECKLINE_BREAK"}
                or primary_pattern in {"DOUBLE_BOTTOM_W_CONFIRMED", "FAKE_BREAKDOWN_DOWN"}
            )
        )
        entry_block_reasons = []
        if not bool(pullback.get("ok")) and not continuation_ready:
            entry_block_reasons.append("PULLBACK_ZONE_NOT_REACHED")
        if not confirmed and not reversal_ready:
            entry_block_reasons.append("CONFIRMATION_MISSING")
        if not ema_2050_ok:
            entry_block_reasons.append("EMA2050_CONFIRMATION_MISSING")
        if bool(pullback.get("ok")) and not continuation_ready and not strong_candle_match:
            entry_block_reasons.append("STRONG_5M_CANDLE_MISSING")
        if fake_break_conflict:
            entry_block_reasons.append("FAKE_BREAKDOWN_CONFLICT")
        if not buyer_active and not continuation_ready and not reversal_ready:
            entry_block_reasons.append("BUYER_ZONE_UNCONFIRMED")
        if confirmation_support_count < 2:
            entry_block_reasons.append("INSUFFICIENT_CONFIRMATION_SCORE")
        if not htf_aligned and confirmation_support_count < 2:
            entry_block_reasons.append("HTF_CONFLUENCE_MISSING")
        if not divergence_supportive:
            entry_block_reasons.append("RSI_DIVERGENCE_AGAINST_SETUP")
        if setup_family == "REVERSAL" and breakout_confirmation == "DOWN_CONFIRMED" and not reversal_ready:
            entry_block_reasons.append("REVERSAL_NOT_CONFIRMED")
        posture = "WAIT_PULLBACK"
        if bool(pullback.get("near")):
            posture = "WAIT_CONFIRMATION"
        if continuation_ready:
            posture = "READY_TREND_CONTINUATION"
        elif reversal_ready:
            posture = "READY_REVERSAL"
        if not entry_block_reasons:
            posture = "READY_TREND_CONTINUATION" if continuation_ready else ("READY_REVERSAL" if reversal_ready else "READY_AFTER_CONFIRMATION")
        short_hint = _round_strike(
            min(
                zone_low,
                float(_safe_float(put_wall.get("strike")) or zone_low),
            ),
            direction="down",
        )
        hedge_gap = 150 if str(snap_15m.get("volatility_regime") or "").upper() == "HIGH" else 100
        hedge_hint = None if short_hint is None else short_hint - hedge_gap
        invalidation_spot = round(zone_low - max(8.0, atr * 0.2), 2)
        confluences = [
            "Pullback into support / EMA20 zone",
            "5m and 15m EMA20/50 trend stays bullish",
            "Strong green 5m confirmation candle after pullback",
            "Bullish rejection or fake breakdown failure",
            "Buyer presence visible at support or bullish FVG",
            "15m / 60m structure stays bullish",
        ]
        if bool(fib_context.get("supportive")):
            confluences.append("Pullback is sitting inside the bullish Fibonacci retracement pocket")
        if bool(fvg_context.get("supportive")):
            confluences.append("Bullish fair value gap is active near the decision zone")
        if str(rsi_divergence.get("status") or "NONE").upper() != "NONE":
            confluences.append(str(rsi_divergence.get("description") or "RSI divergence confirmation"))
        plan.update(
            {
                "trade_idea": "Bull put spread",
                "posture": posture,
                "setup_type": "PUT_CREDIT_SPREAD",
                "setup_family": setup_family,
                "entry_ready": not entry_block_reasons,
                "continuation_ready": continuation_ready,
                "continuation_text": (
                    "Strong bullish continuation is confirmed; a perfect pullback touch is no longer required."
                    if continuation_ready
                    else None
                ),
                "reversal_ready": reversal_ready,
                "reversal_text": (
                    "Bullish reversal is confirmed after pullback into support; sell only after the hold survives."
                    if reversal_ready
                    else None
                ),
                "pullback_touched": bool(pullback.get("touched")),
                "pullback_near": bool(pullback.get("near")),
                "pullback_ok": bool(pullback.get("ok")),
                "pullback_score": float(pullback.get("score") or 0.0),
                "pullback_distance_points": _safe_float(pullback.get("distance_points")),
                "pullback_tolerance_points": _safe_float(pullback.get("tolerance_points")),
                "confirmation_score": confirmation_score,
                "confirmation_support_count": confirmation_support_count,
                "confirmation_ready": bool((continuation_ready or reversal_ready) or (confirmed and confirmation_support_count >= 2 and strong_candle_match)),
                "fake_breakout_conflict": fake_break_conflict,
                "ema_2050_ok": bool(ema_2050_ok),
                "ema_2050_alignment_5m": ema2050_5m,
                "ema_2050_alignment_15m": ema2050_15m,
                "higher_tf_ok": bool(htf_supportive),
                "higher_tf_score": round(float(htf_score), 3),
                "higher_tf_display_score": _safe_float(higher_tf_confluence.get("score")) if _safe_float(higher_tf_confluence.get("score")) is not None else 0.5,
                "buyer_seller_ok": bool(buyer_active or continuation_ready or reversal_ready),
                "strong_candle_ok": bool(strong_candle_match),
                "strong_candle_bias": strong_candle_bias,
                "strong_candle_status": strong_candle_status,
                "strong_candle_text": strong_candle_desc,
                "fvg_supportive": bool(fvg_context.get("supportive")),
                "fair_value_gap_low": _safe_float(fvg_context.get("low")),
                "fair_value_gap_high": _safe_float(fvg_context.get("high")),
                "fib_supportive": bool(fib_context.get("supportive")),
                "fib_zone_low": _safe_float(fib_context.get("low")),
                "fib_zone_high": _safe_float(fib_context.get("high")),
                "divergence_supportive": divergence_supportive,
                "structure_conflict_score": round(float(_clamp(structure_conflict_score, 0.0, 1.0)), 3),
                "option_chain_conflict_score": round(float(_clamp(option_chain_conflict_score, 0.0, 1.0)), 3),
                "reversal_conflict_score": round(float(_clamp(reversal_conflict_score, 0.0, 1.0)), 3),
                "total_conflict_score": total_conflict_score,
                "trigger_window": (
                    f"Trend is strong enough for continuation above support; preferred pullback zone remains {zone_low:.0f}-{zone_high:.0f}."
                    if continuation_ready
                    else (
                        f"Bullish reversal is ready near {zone_low:.0f}-{zone_high:.0f}; sell only if the hold survives."
                        if reversal_ready
                        else f"Wait for pullback into {zone_low:.0f}-{zone_high:.0f}, then sell only after bullish hold or failed downside break."
                    )
                ),
                "entry_trigger_text": (
                    "Bullish continuation is confirmed; a fresh higher-low / breakout-hold is enough to sell the put spread."
                    if continuation_ready
                    else (
                        f"Bullish reversal is confirmed near {zone_low:.0f}-{zone_high:.0f}; sell only after the neckline/hold survives."
                        if reversal_ready
                        else f"Enter only after NIFTY pulls back into {zone_low:.0f}-{zone_high:.0f} and prints bullish confirmation."
                    )
                ),
                "confirmation": "Need higher low, bullish rejection candle, W-pattern confirmation, or bullish retest hold near support/EMA20.",
                "invalidation": f"Avoid entry if a 5m close loses {zone_low:.0f} with acceptance below the pullback zone.",
                "invalidation_spot": invalidation_spot,
                "strike_hint": f"Prefer short put near {short_hint or '—'} and hedge near {hedge_hint or '—'} once confirmation prints.",
                "reason_summary": [
                    f"Bias {market_bias.lower()} with confidence {round(bias_confidence * 100)}%",
                    f"5m/15m pattern: {snap_5m.get('pattern') or '—'} / {snap_15m.get('pattern') or '—'}",
                    f"Pattern confirmation: {primary_pattern if primary_pattern != 'NONE' else price_action_confirmation}",
                    f"EMA20/50 trend: 5m={ema2050_5m}, 15m={ema2050_15m}, EMA20={ema20 if ema20 is not None else '—'}, EMA50={ema50 if ema50 is not None else '—'}",
                    f"PCR/OI: {advisor.get('pcr_bias') or 'NEUTRAL'} / {advisor.get('oi_build_bias') or 'UNKNOWN'}",
                    f"Pullback quality {int(round(float(pullback.get('score') or 0.0) * 100))}% within ATR tolerance {float(pullback.get('tolerance_points') or 0.0):.0f} pts.",
                    strong_candle_desc,
                    (buyer_seller_zones.get("buyers") or {}).get("summary") or "No clear buyer zone yet.",
                    str(fvg_context.get("description") or "No fair value gap context."),
                    str(fib_context.get("description") or "No Fibonacci pullback context."),
                    str(rsi_divergence.get("description") or "No RSI divergence."),
                    f"HTF score {float(htf_score):+.2f}: " + "; ".join(higher_tf_confluence.get("notes") or ["Higher timeframe structure is mixed."]),
                    f"Confirmation score {confirmation_support_count}/3, conflicts structure={structure_conflict_score:.2f}, chain={option_chain_conflict_score:.2f}, reversal={reversal_conflict_score:.2f}.",
                    (
                        "Trend continuation path is active; exact pullback touch is waived because buyers are defending momentum and confluence remains supportive."
                        if continuation_ready
                        else (
                            "Reversal path is active; continuation is not required because support hold is already visible."
                            if reversal_ready
                            else "Continuation path is not active; a pullback into support plus a strong bullish 5m candle is required."
                        )
                    ),
                ],
                "entry_block_reasons": entry_block_reasons,
                "pullback_zone_low": zone_low,
                "pullback_zone_high": zone_high,
                "entry_zone_mid": round((zone_low + zone_high) / 2.0, 2),
                "short_strike_hint": short_hint,
                "hedge_strike_hint": hedge_hint,
                "required_confluences": confluences,
            }
        )
        return plan

    support_px = _safe_float((nearest_support or {}).get("price"))
    resistance_px = _safe_float((nearest_resistance or {}).get("price"))
    range_text = "No clean edge yet."
    if support_px is not None and resistance_px is not None:
        range_text = f"Range is roughly {support_px:.0f}-{resistance_px:.0f}; wait for edge test and rejection."
    plan.update(
        {
            "trade_idea": "Stand aside / range read",
            "posture": "WAIT_EDGE_TEST",
            "trigger_window": range_text,
            "entry_trigger_text": range_text,
            "confirmation": "Need price to reach a range edge, reject, and confirm with candle + retest.",
            "invalidation": "Avoid mid-range entries where reward-to-risk is poor.",
            "reason_summary": [
                f"Bias {market_bias.lower()} is not decisive enough.",
                f"5m/15m pattern: {snap_5m.get('pattern') or '—'} / {snap_15m.get('pattern') or '—'}",
                f"Primary pattern: {primary_pattern}",
                f"PCR/OI: {advisor.get('pcr_bias') or 'NEUTRAL'} / {advisor.get('oi_build_bias') or 'UNKNOWN'}",
                (buyer_seller_zones.get("buyers") or {}).get("summary")
                or (buyer_seller_zones.get("sellers") or {}).get("summary")
                or "Buyer/seller participation is not obvious yet.",
                str(rsi_divergence.get("description") or "No RSI divergence."),
                "; ".join(higher_tf_confluence.get("notes") or ["Higher timeframe structure is mixed."]),
            ],
            "entry_block_reasons": ["BIAS_NOT_DECISIVE"],
            "required_confluences": [
                "Range edge touch",
                "Clear rejection candle",
                "Support/resistance retest hold",
                "Higher timeframe agreement",
            ],
        }
    )
    return plan


def build_trade_plan_from_runtime_context(
    *,
    spot: float,
    trend_ctx: Dict[str, Any],
    oc_ctx: Dict[str, Any],
    structure_ctx: Dict[str, Any],
    market_bias: str,
    bias_confidence: float,
) -> Dict[str, Any]:
    tf_map = trend_ctx.get("timeframes") if isinstance(trend_ctx.get("timeframes"), dict) else {}
    snap_5m = dict(tf_map.get("5")) if isinstance(tf_map.get("5"), dict) else {}
    snap_15m = dict(tf_map.get("15")) if isinstance(tf_map.get("15"), dict) else {}
    snap_60m = dict(tf_map.get("60")) if isinstance(tf_map.get("60"), dict) else {}
    for snap in (snap_5m, snap_15m):
        if "primary_pattern" not in snap:
            snap["primary_pattern"] = structure_ctx.get("primary_pattern")
        if "price_action_confirmation" not in snap:
            snap["price_action_confirmation"] = structure_ctx.get("price_action_confirmation")
        if "retest_status" not in snap:
            snap["retest_status"] = structure_ctx.get("retest_status")
    nearest_support = None
    if isinstance(structure_ctx.get("intraday_support"), dict):
        nearest_support = {
            "price": _safe_float((structure_ctx.get("intraday_support") or {}).get("level")),
            "side": "support",
            "timeframe": "intraday",
        }
    elif isinstance(structure_ctx.get("swing_support"), dict):
        nearest_support = {
            "price": _safe_float((structure_ctx.get("swing_support") or {}).get("level")),
            "side": "support",
            "timeframe": "swing",
        }
    nearest_resistance = None
    if isinstance(structure_ctx.get("intraday_resistance"), dict):
        nearest_resistance = {
            "price": _safe_float((structure_ctx.get("intraday_resistance") or {}).get("level")),
            "side": "resistance",
            "timeframe": "intraday",
        }
    elif isinstance(structure_ctx.get("swing_resistance"), dict):
        nearest_resistance = {
            "price": _safe_float((structure_ctx.get("swing_resistance") or {}).get("level")),
            "side": "resistance",
            "timeframe": "swing",
        }
    rsi_divergence = _derive_rsi_divergence_from_runtime_context(
        market_bias=market_bias,
        structure_ctx=structure_ctx,
        snap_5m=snap_5m,
    )
    buyer_seller_zones = _derive_orderflow_zones_from_runtime_context(
        spot=spot,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        snap_5m=snap_5m,
        structure_ctx=structure_ctx,
    )
    higher_tf_confluence = build_higher_tf_confluence(
        snap_15m=snap_15m,
        snap_60m=snap_60m,
        market_bias=market_bias,
        rsi_divergence=rsi_divergence,
    )
    advisor = {
        "pcr_bias": str(oc_ctx.get("pcr_bias") or "NEUTRAL"),
        "oi_build_bias": str(((oc_ctx.get("oi_build") or {}) if isinstance(oc_ctx.get("oi_build"), dict) else {}).get("bias") or "UNKNOWN"),
        "market_context": {"option_chain": oc_ctx},
    }
    return build_trade_plan(
        spot=spot,
        snap_5m=snap_5m,
        snap_15m=snap_15m,
        snap_60m=snap_60m,
        market_bias=market_bias,
        bias_confidence=bias_confidence,
        orb=trend_ctx.get("orb") if isinstance(trend_ctx.get("orb"), dict) else {},
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        advisor=advisor,
        buyer_seller_zones=buyer_seller_zones,
        rsi_divergence=rsi_divergence,
        higher_tf_confluence=higher_tf_confluence,
    )
