from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def _safe_float(value: Any) -> Optional[float]:
    if value in (None, "", "-", "--"):
        return None
    try:
        return float(value)
    except Exception:
        return None


def detect_recent_price_action(
    candles: List[dict],
    *,
    support: Optional[float],
    resistance: Optional[float],
    atr_like_points: float,
    ema_slow: Optional[float] = None,
    ema_base: Optional[float] = None,
) -> Dict[str, Any]:
    default = {
        "price_action_bias": "NEUTRAL",
        "price_action_confirmation": "NONE",
        "price_action_score": 0.0,
        "primary_pattern": "NONE",
        "price_action_patterns": [],
        "retest_status": "NONE",
        "retest_bias": "NEUTRAL",
        "retest_level": None,
        "retest_score": 0.0,
        "fair_value_gap_bias": "NEUTRAL",
        "fair_value_gap_status": "NONE",
        "fair_value_gap_low": None,
        "fair_value_gap_high": None,
        "fair_value_gap_active": False,
        "fair_value_gap_score": 0.0,
        "fair_value_gap_description": "No fair value gap context.",
        "fib_swing_high": None,
        "fib_swing_low": None,
        "fib_382": None,
        "fib_500": None,
        "fib_618": None,
        "fib_bullish_zone_low": None,
        "fib_bullish_zone_high": None,
        "fib_bullish_active": False,
        "fib_bearish_zone_low": None,
        "fib_bearish_zone_high": None,
        "fib_bearish_active": False,
        "fib_description": "No Fibonacci pullback context.",
        "strong_candle_bias": "NEUTRAL",
        "strong_candle_status": "NONE",
        "strong_candle_confirmed": False,
        "strong_candle_score": 0.0,
        "strong_candle_body_points": None,
        "strong_candle_description": "No strong 5m confirmation candle.",
    }
    parsed: List[Dict[str, float]] = []
    for candle in (candles or [])[-6:]:
        try:
            open_px = float(candle.get("open") or candle.get("close") or 0.0)
            high_px = float(candle.get("high") or candle.get("close") or open_px)
            low_px = float(candle.get("low") or candle.get("close") or open_px)
            close_px = float(candle.get("close") or open_px)
        except Exception:
            continue
        rng = max(0.01, high_px - low_px)
        body = abs(close_px - open_px)
        upper_wick = max(0.0, high_px - max(open_px, close_px))
        lower_wick = max(0.0, min(open_px, close_px) - low_px)
        close_pos = (close_px - low_px) / rng
        parsed.append(
            {
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
                "range": rng,
                "body": body,
                "upper_wick": upper_wick,
                "lower_wick": lower_wick,
                "close_pos": close_pos,
            }
        )
    if len(parsed) < 3:
        return default

    last = parsed[-1]
    prev = parsed[-2]
    prev2 = parsed[-3]
    prior_rows = parsed[:-1]
    prior_high = max(row["high"] for row in prior_rows)
    prior_low = min(row["low"] for row in prior_rows)
    avg_range = max(0.01, float(atr_like_points or 0.0), sum(row["range"] for row in parsed) / len(parsed))
    level_buffer = max(8.0, avg_range * 0.25)

    bullish_score = 0.0
    bearish_score = 0.0
    bullish_patterns: List[str] = []
    bearish_patterns: List[str] = []
    retest_status = "NONE"
    retest_bias = "NEUTRAL"
    retest_level: Optional[float] = None
    retest_score = 0.0

    def _record_retest(*, status: str, bias: str, level: Optional[float], score: float) -> None:
        nonlocal retest_status, retest_bias, retest_level, retest_score
        if float(score) >= float(retest_score):
            retest_status = str(status)
            retest_bias = str(bias)
            retest_level = None if level is None else float(level)
            retest_score = float(score)

    def _append_once(target: List[str], value: str) -> None:
        text = str(value or "").upper()
        if text and text not in target:
            target.append(text)

    def _derive_recent_fvg(rows: List[Dict[str, float]]) -> Dict[str, Any]:
        out = {
            "fair_value_gap_bias": "NEUTRAL",
            "fair_value_gap_status": "NONE",
            "fair_value_gap_low": None,
            "fair_value_gap_high": None,
            "fair_value_gap_active": False,
            "fair_value_gap_score": 0.0,
            "fair_value_gap_description": "No fair value gap context.",
        }
        gap_threshold = max(6.0, level_buffer * 0.35, avg_range * 0.18)
        search_rows = rows[-6:] if len(rows) > 6 else rows
        for idx in range(len(search_rows) - 3, -1, -1):
            left = search_rows[idx]
            right = search_rows[idx + 2]
            bullish_gap = float(right["low"]) - float(left["high"])
            bearish_gap = float(left["low"]) - float(right["high"])
            if bullish_gap >= gap_threshold:
                zone_low = float(left["high"])
                zone_high = float(right["low"])
                active = float(last["low"]) <= zone_high + (level_buffer * 0.40) and float(last["close"]) >= zone_low - (level_buffer * 0.15)
                out.update(
                    {
                        "fair_value_gap_bias": "BULLISH",
                        "fair_value_gap_status": "BULLISH_FVG_ACTIVE" if active else "BULLISH_FVG_PRESENT",
                        "fair_value_gap_low": round(zone_low, 2),
                        "fair_value_gap_high": round(zone_high, 2),
                        "fair_value_gap_active": bool(active),
                        "fair_value_gap_score": round(min(1.0, bullish_gap / max(avg_range, 1.0)), 3),
                        "fair_value_gap_description": f"Bullish FVG sits around {zone_low:.0f}-{zone_high:.0f}.",
                    }
                )
                return out
            if bearish_gap >= gap_threshold:
                zone_low = float(right["high"])
                zone_high = float(left["low"])
                active = float(last["high"]) >= zone_low - (level_buffer * 0.40) and float(last["close"]) <= zone_high + (level_buffer * 0.15)
                out.update(
                    {
                        "fair_value_gap_bias": "BEARISH",
                        "fair_value_gap_status": "BEARISH_FVG_ACTIVE" if active else "BEARISH_FVG_PRESENT",
                        "fair_value_gap_low": round(zone_low, 2),
                        "fair_value_gap_high": round(zone_high, 2),
                        "fair_value_gap_active": bool(active),
                        "fair_value_gap_score": round(min(1.0, bearish_gap / max(avg_range, 1.0)), 3),
                        "fair_value_gap_description": f"Bearish FVG sits around {zone_low:.0f}-{zone_high:.0f}.",
                    }
                )
                return out
        return out

    def _derive_fib_context(rows: List[Dict[str, float]]) -> Dict[str, Any]:
        out = {
            "fib_swing_high": None,
            "fib_swing_low": None,
            "fib_382": None,
            "fib_500": None,
            "fib_618": None,
            "fib_bullish_zone_low": None,
            "fib_bullish_zone_high": None,
            "fib_bullish_active": False,
            "fib_bearish_zone_low": None,
            "fib_bearish_zone_high": None,
            "fib_bearish_active": False,
            "fib_description": "No Fibonacci pullback context.",
        }
        if not rows:
            return out
        swing_high = max(float(row["high"]) for row in rows)
        swing_low = min(float(row["low"]) for row in rows)
        swing_range = max(0.0, swing_high - swing_low)
        min_retracement_range = max(avg_range * 1.6, max(6.0, abs(float(last["close"])) * 0.0015))
        if swing_range < min_retracement_range:
            return out
        fib_382 = swing_high - (0.382 * swing_range)
        fib_500 = swing_high - (0.500 * swing_range)
        fib_618 = swing_high - (0.618 * swing_range)
        bull_zone_low = min(fib_618, fib_500)
        bull_zone_high = max(fib_382, fib_500)
        bear_zone_low = min(swing_low + (0.382 * swing_range), swing_low + (0.500 * swing_range))
        bear_zone_high = max(swing_low + (0.500 * swing_range), swing_low + (0.618 * swing_range))
        zone_tol = max(6.0, level_buffer * 0.45)
        bull_active = (bull_zone_low - zone_tol) <= float(last["close"]) <= (bull_zone_high + zone_tol)
        bear_active = (bear_zone_low - zone_tol) <= float(last["close"]) <= (bear_zone_high + zone_tol)
        description = (
            f"Fib pullback zone is {bull_zone_low:.0f}-{bull_zone_high:.0f} for bullish continuation "
            f"and {bear_zone_low:.0f}-{bear_zone_high:.0f} for bearish continuation."
        )
        out.update(
            {
                "fib_swing_high": round(swing_high, 2),
                "fib_swing_low": round(swing_low, 2),
                "fib_382": round(fib_382, 2),
                "fib_500": round(fib_500, 2),
                "fib_618": round(fib_618, 2),
                "fib_bullish_zone_low": round(bull_zone_low, 2),
                "fib_bullish_zone_high": round(bull_zone_high, 2),
                "fib_bullish_active": bool(bull_active),
                "fib_bearish_zone_low": round(bear_zone_low, 2),
                "fib_bearish_zone_high": round(bear_zone_high, 2),
                "fib_bearish_active": bool(bear_active),
                "fib_description": description,
            }
        )
        return out

    def _local_pivot_lows(rows: List[Dict[str, float]]) -> List[Tuple[int, float]]:
        out: List[Tuple[int, float]] = []
        for idx in range(1, len(rows) - 1):
            cur = rows[idx]
            if cur["low"] <= rows[idx - 1]["low"] and cur["low"] <= rows[idx + 1]["low"]:
                out.append((idx, float(cur["low"])))
        return out

    def _local_pivot_highs(rows: List[Dict[str, float]]) -> List[Tuple[int, float]]:
        out: List[Tuple[int, float]] = []
        for idx in range(1, len(rows) - 1):
            cur = rows[idx]
            if cur["high"] >= rows[idx - 1]["high"] and cur["high"] >= rows[idx + 1]["high"]:
                out.append((idx, float(cur["high"])))
        return out

    if (
        last["close"] > last["open"]
        and prev["close"] < prev["open"]
        and last["open"] <= prev["close"]
        and last["close"] >= prev["open"]
        and last["body"] >= (prev["body"] * 0.9)
    ):
        bullish_score += 0.85
        _append_once(bullish_patterns, "BULLISH_ENGULFING")
    if (
        last["close"] < last["open"]
        and prev["close"] > prev["open"]
        and last["open"] >= prev["close"]
        and last["close"] <= prev["open"]
        and last["body"] >= (prev["body"] * 0.9)
    ):
        bearish_score += 0.85
        _append_once(bearish_patterns, "BEARISH_ENGULFING")
    if last["lower_wick"] >= max(last["body"] * 1.6, avg_range * 0.18) and last["close_pos"] >= 0.60:
        bullish_score += 0.55
        _append_once(bullish_patterns, "LOWER_REJECTION")
    if last["upper_wick"] >= max(last["body"] * 1.6, avg_range * 0.18) and last["close_pos"] <= 0.40:
        bearish_score += 0.55
        _append_once(bearish_patterns, "UPPER_REJECTION")

    strong_body_threshold = max(6.0, avg_range * 0.48, float(atr_like_points or 0.0) * 0.22)
    strong_range_threshold = max(strong_body_threshold * 1.15, avg_range * 0.82)
    strong_candle_bias = "NEUTRAL"
    strong_candle_status = "NONE"
    strong_candle_score = 0.0
    strong_candle_description = "No strong 5m confirmation candle."
    if (
        last["close"] > last["open"]
        and last["body"] >= strong_body_threshold
        and last["range"] >= strong_range_threshold
        and last["close_pos"] >= 0.72
    ):
        bullish_score += 0.42
        strong_candle_bias = "BULLISH"
        strong_candle_status = "STRONG_GREEN_5M"
        strong_candle_score = min(1.0, last["body"] / max(strong_body_threshold, 1.0))
        strong_candle_description = "Strong bullish 5m candle closed near the high after the pullback."
        _append_once(bullish_patterns, "STRONG_BULLISH_CANDLE")
    elif (
        last["close"] < last["open"]
        and last["body"] >= strong_body_threshold
        and last["range"] >= strong_range_threshold
        and last["close_pos"] <= 0.28
    ):
        bearish_score += 0.42
        strong_candle_bias = "BEARISH"
        strong_candle_status = "STRONG_RED_5M"
        strong_candle_score = min(1.0, last["body"] / max(strong_body_threshold, 1.0))
        strong_candle_description = "Strong bearish 5m candle closed near the low after the pullback."
        _append_once(bearish_patterns, "STRONG_BEARISH_CANDLE")
    if last["close"] > prior_high and last["close_pos"] >= 0.62:
        bullish_score += 0.70
        _append_once(bullish_patterns, "BREAKOUT_CONTINUATION_UP")
    if last["close"] < prior_low and last["close_pos"] <= 0.38:
        bearish_score += 0.70
        _append_once(bearish_patterns, "BREAKOUT_CONTINUATION_DOWN")
    if (
        prev["high"] <= prev2["high"]
        and prev["low"] >= prev2["low"]
        and last["close"] > prev["high"] + (level_buffer * 0.05)
        and last["close"] > last["open"]
    ):
        bullish_score += 0.50
        _append_once(bullish_patterns, "INSIDE_BAR_BREAK_UP")
    if (
        prev["high"] <= prev2["high"]
        and prev["low"] >= prev2["low"]
        and last["close"] < prev["low"] - (level_buffer * 0.05)
        and last["close"] < last["open"]
    ):
        bearish_score += 0.50
        _append_once(bearish_patterns, "INSIDE_BAR_BREAK_DOWN")

    ema20 = _safe_float(ema_slow)
    if ema20 is not None:
        ema_buffer = max(4.0, level_buffer * 0.08)
        if (
            prev["close"] <= float(ema20) - ema_buffer
            and last["close"] >= float(ema20) + ema_buffer
            and last["close_pos"] >= 0.58
        ):
            bullish_score += 0.55
            _append_once(bullish_patterns, "EMA20_BREAKOUT_UP")
        if (
            prev["close"] >= float(ema20) + ema_buffer
            and last["close"] <= float(ema20) - ema_buffer
            and last["close_pos"] <= 0.42
        ):
            bearish_score += 0.55
            _append_once(bearish_patterns, "EMA20_BREAKDOWN_DOWN")

    pivot_lows = _local_pivot_lows(parsed)
    pivot_highs = _local_pivot_highs(parsed)
    neckline_buffer = max(5.0, level_buffer * 0.08)
    twin_extrema_tolerance = max(avg_range * 0.75, level_buffer * 1.2)

    if len(pivot_lows) >= 2:
        (low_idx_1, low_1), (low_idx_2, low_2) = pivot_lows[-2], pivot_lows[-1]
        if low_idx_2 - low_idx_1 >= 2:
            neckline = max(parsed[idx]["high"] for idx in range(low_idx_1 + 1, low_idx_2))
            lows_clustered = abs(float(low_1) - float(low_2)) <= twin_extrema_tolerance
            second_low_held = float(low_2) >= float(low_1) - twin_extrema_tolerance
            neckline_broken = last["close"] >= float(neckline) + neckline_buffer and last["close"] > last["open"]
            if lows_clustered and second_low_held and neckline_broken:
                bullish_score += 0.95
                _append_once(bullish_patterns, "DOUBLE_BOTTOM_W_CONFIRMED")
                _record_retest(
                    status="DOUBLE_BOTTOM_NECKLINE_BREAK",
                    bias="BULLISH",
                    level=float(neckline),
                    score=0.91,
                )

    if len(pivot_highs) >= 2:
        (high_idx_1, high_1), (high_idx_2, high_2) = pivot_highs[-2], pivot_highs[-1]
        if high_idx_2 - high_idx_1 >= 2:
            neckline = min(parsed[idx]["low"] for idx in range(high_idx_1 + 1, high_idx_2))
            highs_clustered = abs(float(high_1) - float(high_2)) <= twin_extrema_tolerance
            second_high_failed = float(high_2) <= float(high_1) + twin_extrema_tolerance
            neckline_broken = last["close"] <= float(neckline) - neckline_buffer and last["close"] < last["open"]
            if highs_clustered and second_high_failed and neckline_broken:
                bearish_score += 0.95
                _append_once(bearish_patterns, "DOUBLE_TOP_M_CONFIRMED")
                _record_retest(
                    status="DOUBLE_TOP_NECKLINE_BREAK",
                    bias="BEARISH",
                    level=float(neckline),
                    score=0.91,
                )

    if support is not None:
        support_f = float(support)
        if (
            last["low"] <= support_f + level_buffer
            and last["close"] >= support_f + (level_buffer * 0.10)
            and last["close_pos"] >= 0.55
        ):
            _record_retest(status="SUPPORT_HOLD", bias="BULLISH", level=support_f, score=0.72)
        if (
            prev["close"] < support_f - (level_buffer * 0.05)
            and last["close"] > support_f + (level_buffer * 0.05)
            and (last["lower_wick"] >= max(last["body"], avg_range * 0.15) or last["close_pos"] >= 0.58)
        ):
            bullish_score += 0.88
            _append_once(bullish_patterns, "FAKE_BREAKDOWN_DOWN")
            _record_retest(status="FAILED_BREAKDOWN_RETEST", bias="BULLISH", level=support_f, score=0.90)
    if resistance is not None:
        resistance_f = float(resistance)
        if (
            last["high"] >= resistance_f - level_buffer
            and last["close"] <= resistance_f - (level_buffer * 0.10)
            and last["close_pos"] <= 0.45
        ):
            _record_retest(status="RESISTANCE_HOLD", bias="BEARISH", level=resistance_f, score=0.72)
        if (
            prev["close"] > resistance_f + (level_buffer * 0.05)
            and last["close"] < resistance_f - (level_buffer * 0.05)
            and (last["upper_wick"] >= max(last["body"], avg_range * 0.15) or last["close_pos"] <= 0.42)
        ):
            bearish_score += 0.88
            _append_once(bearish_patterns, "FAKE_BREAKOUT_UP")
            _record_retest(status="FAILED_BREAKOUT_RETEST", bias="BEARISH", level=resistance_f, score=0.90)
    if resistance is not None:
        resistance_f = float(resistance)
        if prev["close"] > resistance_f and last["low"] <= resistance_f + level_buffer and last["close"] > resistance_f:
            _record_retest(status="BREAKOUT_RETEST_HOLD", bias="BULLISH", level=resistance_f, score=0.88)
        elif prev["close"] > resistance_f and last["close"] < resistance_f:
            _record_retest(status="RETEST_FAILED", bias="BEARISH", level=resistance_f, score=0.82)
    if support is not None:
        support_f = float(support)
        if prev["close"] < support_f and last["high"] >= support_f - level_buffer and last["close"] < support_f:
            _record_retest(status="BREAKDOWN_RETEST_HOLD", bias="BEARISH", level=support_f, score=0.88)
        elif prev["close"] < support_f and last["close"] > support_f:
            _record_retest(status="RETEST_FAILED", bias="BULLISH", level=support_f, score=0.82)

    price_action_bias = "NEUTRAL"
    price_action_patterns: List[str] = []
    price_action_score = 0.0
    if bullish_score >= max(0.55, bearish_score + 0.20):
        price_action_bias = "BULLISH"
        price_action_patterns = bullish_patterns
        price_action_score = bullish_score
    elif bearish_score >= max(0.55, bullish_score + 0.20):
        price_action_bias = "BEARISH"
        price_action_patterns = bearish_patterns
        price_action_score = bearish_score

    unique_patterns: List[str] = []
    for pattern in price_action_patterns:
        text = str(pattern or "").upper()
        if text and text not in unique_patterns:
            unique_patterns.append(text)

    confirmation = "NONE"
    fail_statuses = {"NONE", "RETEST_FAILED", "FAILED_BREAKOUT_RETEST", "FAILED_BREAKDOWN_RETEST"}
    bullish_reversal_patterns = {"DOUBLE_BOTTOM_W_CONFIRMED", "FAKE_BREAKDOWN_DOWN"}
    bearish_reversal_patterns = {"DOUBLE_TOP_M_CONFIRMED", "FAKE_BREAKOUT_UP"}
    bullish_breakout_patterns = {"BREAKOUT_CONTINUATION_UP", "INSIDE_BAR_BREAK_UP", "EMA20_BREAKOUT_UP"}
    bearish_breakout_patterns = {"BREAKOUT_CONTINUATION_DOWN", "INSIDE_BAR_BREAK_DOWN", "EMA20_BREAKDOWN_DOWN"}
    if price_action_bias != "NEUTRAL" and retest_bias == price_action_bias and retest_status not in fail_statuses:
        confirmation = "CANDLE_AND_RETEST_CONFIRMED"
    elif price_action_bias == "BULLISH" and bullish_reversal_patterns.intersection(unique_patterns):
        confirmation = "REVERSAL_CONFIRMED"
    elif price_action_bias == "BEARISH" and bearish_reversal_patterns.intersection(unique_patterns):
        confirmation = "REVERSAL_CONFIRMED"
    elif price_action_bias == "BULLISH" and bullish_breakout_patterns.intersection(unique_patterns):
        confirmation = "BREAKOUT_CONFIRMED"
    elif price_action_bias == "BEARISH" and bearish_breakout_patterns.intersection(unique_patterns):
        confirmation = "BREAKOUT_CONFIRMED"
    elif price_action_bias != "NEUTRAL":
        confirmation = "CANDLE_CONFIRMED"
    elif retest_bias != "NEUTRAL" and retest_status not in fail_statuses:
        price_action_bias = retest_bias
        confirmation = "RETEST_CONFIRMED"
        unique_patterns = [retest_status]
        price_action_score = retest_score

    fvg_context = _derive_recent_fvg(parsed)
    fib_context = _derive_fib_context(parsed)

    return {
        "price_action_bias": price_action_bias,
        "price_action_confirmation": confirmation,
        "price_action_score": round(float(max(price_action_score, retest_score)), 3),
        "primary_pattern": unique_patterns[0] if unique_patterns else "NONE",
        "price_action_patterns": unique_patterns[:4],
        "retest_status": retest_status,
        "retest_bias": retest_bias,
        "retest_level": None if retest_level is None else round(float(retest_level), 2),
        "retest_score": round(float(retest_score), 3),
        **fvg_context,
        **fib_context,
        "strong_candle_bias": strong_candle_bias,
        "strong_candle_status": strong_candle_status,
        "strong_candle_confirmed": bool(strong_candle_bias in {"BULLISH", "BEARISH"}),
        "strong_candle_score": round(float(strong_candle_score), 3),
        "strong_candle_body_points": round(float(last["body"]), 2),
        "strong_candle_description": strong_candle_description,
    }
