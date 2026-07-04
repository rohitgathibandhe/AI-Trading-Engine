from __future__ import annotations

import json
from datetime import datetime, time
from pathlib import Path
from statistics import fmean
from typing import Iterable

from .data_models import OhlcvBar, OhlcvSeries, ORLevels, OptionType, OptionsContractQuote

_MARKET_CONTEXT_PATH = Path(__file__).resolve().parents[1] / "state" / "market_context.json"


MARKET_OPEN = time(9, 15)
RANGE_GATE_TIME = time(10, 0)
FORCE_EXIT_TIME = time(15, 15)


def session_bars(series: OhlcvSeries) -> list[OhlcvBar]:
    return [bar for bar in series.bars if bar.timestamp.date() == series.session_date]


def closes(series: Iterable[OhlcvBar]) -> list[float]:
    return [bar.close for bar in series]


def highs(series: Iterable[OhlcvBar]) -> list[float]:
    return [bar.high for bar in series]


def lows(series: Iterable[OhlcvBar]) -> list[float]:
    return [bar.low for bar in series]


def price_change_pct(start: float, end: float) -> float:
    if start <= 0:
        return 0.0
    return ((end - start) / start) * 100.0


def opening_gap_pct(previous_close: float | None, session_open: float | None) -> float:
    if previous_close is None or session_open is None or previous_close <= 0 or session_open <= 0:
        return 0.0
    return price_change_pct(previous_close, session_open)


def ema(values: list[float], period: int = 20) -> list[float]:
    if not values:
        return []
    alpha = 2.0 / (period + 1.0)
    ema_values = [values[0]]
    for value in values[1:]:
        ema_values.append((value * alpha) + (ema_values[-1] * (1.0 - alpha)))
    return ema_values


def ema_value(values: list[float], period: int = 20) -> float | None:
    ema_values = ema(values, period=period)
    return ema_values[-1] if ema_values else None


def ema_slope(values: list[float], period: int = 20, lookback: int = 3) -> float:
    ema_values = ema(values, period=period)
    if len(ema_values) <= lookback:
        return 0.0
    return ema_values[-1] - ema_values[-1 - lookback]


def ema_distance_pct(fast: float | None, slow: float | None, reference: float) -> float:
    if fast is None or slow is None or reference <= 0:
        return 0.0
    return ((fast - slow) / reference) * 100.0


def compute_vwap(bars: list[OhlcvBar]) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for bar in bars:
        typical = (bar.high + bar.low + bar.close) / 3.0
        numerator += typical * bar.volume
        denominator += bar.volume
    if denominator <= 0:
        return None
    return numerator / denominator


def realized_volatility_pct(bars_5m: list[OhlcvBar], window_bars: int = 6) -> float:
    if len(bars_5m) < window_bars:
        return 0.0
    window = bars_5m[-window_bars:]
    price_range = max(bar.high for bar in window) - min(bar.low for bar in window)
    reference = window[-1].close
    if reference <= 0:
        return 0.0
    return (price_range / reference) * 100.0


def rolling_window_bars(bars: list[OhlcvBar], lookback: int) -> list[OhlcvBar]:
    if lookback <= 0:
        return []
    return bars[-lookback:]


def adaptive_or_length_minutes(rv30_pct: float, rv_high_cutoff: float, rv_mid_cutoff: float) -> int:
    if rv30_pct > rv_high_cutoff:
        return 45
    if rv_mid_cutoff <= rv30_pct <= rv_high_cutoff:
        return 30
    return 15


def compute_opening_range(bars_5m: list[OhlcvBar], length_minutes: int) -> ORLevels | None:
    session = [bar for bar in bars_5m if bar.timestamp.time() >= MARKET_OPEN]
    required_bars = length_minutes // 5
    if len(session) < required_bars:
        return None
    opening_window = session[:required_bars]
    return ORLevels(
        length_minutes=length_minutes,
        high=max(bar.high for bar in opening_window),
        low=min(bar.low for bar in opening_window),
    )


def last_n_closes_below(level: float, bars_5m: list[OhlcvBar], n: int = 2) -> bool:
    if len(bars_5m) < n:
        return False
    return all(bar.close < level for bar in bars_5m[-n:])


def last_n_closes_above(level: float, bars_5m: list[OhlcvBar], n: int = 2) -> bool:
    if len(bars_5m) < n:
        return False
    return all(bar.close > level for bar in bars_5m[-n:])


def _pivot_highs(bars: list[OhlcvBar]) -> list[float]:
    pivots: list[float] = []
    for idx in range(1, len(bars) - 1):
        if bars[idx].high > bars[idx - 1].high and bars[idx].high > bars[idx + 1].high:
            pivots.append(bars[idx].high)
    return pivots


def _pivot_lows(bars: list[OhlcvBar]) -> list[float]:
    pivots: list[float] = []
    for idx in range(1, len(bars) - 1):
        if bars[idx].low < bars[idx - 1].low and bars[idx].low < bars[idx + 1].low:
            pivots.append(bars[idx].low)
    return pivots


def recent_lower_highs(bars: list[OhlcvBar], needed: int = 3) -> bool:
    pivots = _pivot_highs(bars)
    if len(pivots) < needed:
        return False
    recent = pivots[-needed:]
    return all(left > right for left, right in zip(recent, recent[1:]))


def recent_higher_lows(bars: list[OhlcvBar], needed: int = 3) -> bool:
    pivots = _pivot_lows(bars)
    if len(pivots) < needed:
        return False
    recent = pivots[-needed:]
    return all(left < right for left, right in zip(recent, recent[1:]))


def bullish_reversal_structure(bars: list[OhlcvBar]) -> bool:
    if len(bars) < 4:
        return False
    recent = bars[-6:]
    closes_up = recent[-1].close > recent[-2].close > recent[-3].close
    return closes_up and recent_higher_lows(recent, needed=2)


def bearish_reversal_structure(bars: list[OhlcvBar]) -> bool:
    if len(bars) < 4:
        return False
    recent = bars[-6:]
    closes_down = recent[-1].close < recent[-2].close < recent[-3].close
    return closes_down and recent_lower_highs(recent, needed=2)


def close_location_value(close: float, low: float, high: float) -> float:
    if high <= low:
        return 0.5
    return max(0.0, min(1.0, (close - low) / (high - low)))


def candle_body_fraction(bar: OhlcvBar) -> float:
    candle_range = max(bar.high - bar.low, 1e-6)
    return abs(bar.close - bar.open) / candle_range


def upper_wick_fraction(bar: OhlcvBar) -> float:
    candle_range = max(bar.high - bar.low, 1e-6)
    return max(bar.high - max(bar.open, bar.close), 0.0) / candle_range


def lower_wick_fraction(bar: OhlcvBar) -> float:
    candle_range = max(bar.high - bar.low, 1e-6)
    return max(min(bar.open, bar.close) - bar.low, 0.0) / candle_range


def bearish_candle_context(bars: list[OhlcvBar]) -> dict[str, float | str | bool]:
    if len(bars) < 2:
        return {
            "pattern": "NONE",
            "quality_score": 0.0,
            "body_fraction": 0.0,
            "upper_wick_fraction": 0.0,
            "lower_wick_fraction": 0.0,
            "close_location": 0.5,
            "expansion": False,
            "engulfing": False,
            "rejection_wick": False,
        }
    recent = bars[-6:]
    last = recent[-1]
    prev = recent[-2]
    prior = recent[-3] if len(recent) >= 3 else prev
    body_fraction = candle_body_fraction(last)
    upper_wick = upper_wick_fraction(last)
    lower_wick = lower_wick_fraction(last)
    close_location = close_location_value(last.close, last.low, last.high)
    prior_high = max(bar.high for bar in recent[:-1])
    bearish_expansion = (
        last.close < last.open
        and last.close < prev.low
        and body_fraction >= 0.55
        and close_location <= 0.25
    )
    bearish_engulfing = (
        prev.close > prev.open
        and last.open >= min(prev.close, prev.high)
        and last.close <= prev.open
        and last.close < last.open
        and body_fraction >= 0.45
    )
    bearish_rejection = (
        last.close < last.open
        and upper_wick >= 0.30
        and close_location <= 0.35
    )
    bearish_breakout_failure = (
        last.high > prior_high
        and last.close < last.open
        and last.close < max(prev.high, prev.close)
        and upper_wick >= 0.30
        and close_location <= 0.35
    )
    bearish_shooting_star = (
        last.close < last.open
        and body_fraction <= 0.35
        and upper_wick >= 0.45
        and close_location <= 0.30
    )
    bearish_evening_star = (
        len(recent) >= 3
        and prior.close > prior.open
        and candle_body_fraction(prior) >= 0.45
        and candle_body_fraction(prev) <= 0.35
        and last.close < last.open
        and last.close <= ((prior.open + prior.close) / 2.0)
        and close_location <= 0.35
    )
    bearish_inside_bar_failure = (
        len(recent) >= 3
        and prev.high <= prior.high
        and prev.low >= prior.low
        and last.close < prev.low
        and last.close < last.open
        and close_location <= 0.35
    )
    quality_score = 0.0
    if bearish_breakout_failure:
        quality_score += 2.7
    elif bearish_expansion:
        quality_score += 2.5
    elif bearish_engulfing:
        quality_score += 2.0
    elif bearish_evening_star:
        quality_score += 2.1
    elif bearish_inside_bar_failure:
        quality_score += 1.8
    elif bearish_shooting_star:
        quality_score += 1.7
    elif bearish_rejection:
        quality_score += 1.5
    if body_fraction >= 0.60:
        quality_score += 0.75
    elif body_fraction >= 0.45:
        quality_score += 0.40
    if close_location <= 0.20:
        quality_score += 0.75
    elif close_location <= 0.35:
        quality_score += 0.40
    if upper_wick >= 0.25:
        quality_score += 0.35
    if recent_lower_highs(recent, needed=2):
        quality_score += 0.40
    pattern = "NONE"
    if bearish_breakout_failure:
        pattern = "BEARISH_BREAKOUT_FAILURE"
    elif bearish_expansion:
        pattern = "BEARISH_EXPANSION"
    elif bearish_engulfing:
        pattern = "BEARISH_ENGULFING"
    elif bearish_evening_star:
        pattern = "BEARISH_EVENING_STAR"
    elif bearish_inside_bar_failure:
        pattern = "BEARISH_INSIDE_BAR_FAILURE"
    elif bearish_shooting_star:
        pattern = "BEARISH_SHOOTING_STAR"
    elif bearish_rejection:
        pattern = "BEARISH_REJECTION_WICK"
    return {
        "pattern": pattern,
        "quality_score": round(quality_score, 4),
        "body_fraction": round(body_fraction, 4),
        "upper_wick_fraction": round(upper_wick, 4),
        "lower_wick_fraction": round(lower_wick, 4),
        "close_location": round(close_location, 4),
        "expansion": bearish_expansion,
        "engulfing": bearish_engulfing,
        "rejection_wick": bearish_rejection,
        "breakout_failure": bearish_breakout_failure,
        "shooting_star": bearish_shooting_star,
        "evening_star": bearish_evening_star,
        "inside_bar_failure": bearish_inside_bar_failure,
    }


def bullish_candle_context(bars: list[OhlcvBar]) -> dict[str, float | str | bool]:
    if len(bars) < 2:
        return {
            "pattern": "NONE",
            "quality_score": 0.0,
            "body_fraction": 0.0,
            "upper_wick_fraction": 0.0,
            "lower_wick_fraction": 0.0,
            "close_location": 0.5,
            "expansion": False,
            "engulfing": False,
            "rejection_wick": False,
        }
    recent = bars[-6:]
    last = recent[-1]
    prev = recent[-2]
    prior = recent[-3] if len(recent) >= 3 else prev
    body_fraction = candle_body_fraction(last)
    upper_wick = upper_wick_fraction(last)
    lower_wick = lower_wick_fraction(last)
    close_location = close_location_value(last.close, last.low, last.high)
    prior_low = min(bar.low for bar in recent[:-1])
    bullish_expansion = (
        last.close > last.open
        and last.close > prev.high
        and body_fraction >= 0.55
        and close_location >= 0.75
    )
    bullish_engulfing = (
        prev.close < prev.open
        and last.open <= max(prev.close, prev.low)
        and last.close >= prev.open
        and last.close > last.open
        and body_fraction >= 0.45
    )
    bullish_rejection = (
        last.close > last.open
        and lower_wick >= 0.30
        and close_location >= 0.65
    )
    bullish_breakdown_failure = (
        last.low < prior_low
        and last.close > last.open
        and last.close > min(prev.low, prev.close)
        and lower_wick >= 0.30
        and close_location >= 0.65
    )
    bullish_hammer = (
        last.close > last.open
        and body_fraction <= 0.35
        and lower_wick >= 0.45
        and close_location >= 0.70
    )
    bullish_morning_star = (
        len(recent) >= 3
        and prior.close < prior.open
        and candle_body_fraction(prior) >= 0.45
        and candle_body_fraction(prev) <= 0.35
        and last.close > last.open
        and last.close >= ((prior.open + prior.close) / 2.0)
        and close_location >= 0.65
    )
    bullish_inside_bar_failure = (
        len(recent) >= 3
        and prev.high <= prior.high
        and prev.low >= prior.low
        and last.close > prev.high
        and last.close > last.open
        and close_location >= 0.65
    )
    quality_score = 0.0
    if bullish_breakdown_failure:
        quality_score += 2.7
    elif bullish_expansion:
        quality_score += 2.5
    elif bullish_engulfing:
        quality_score += 2.0
    elif bullish_morning_star:
        quality_score += 2.1
    elif bullish_inside_bar_failure:
        quality_score += 1.8
    elif bullish_hammer:
        quality_score += 1.7
    elif bullish_rejection:
        quality_score += 1.5
    if body_fraction >= 0.60:
        quality_score += 0.75
    elif body_fraction >= 0.45:
        quality_score += 0.40
    if close_location >= 0.80:
        quality_score += 0.75
    elif close_location >= 0.65:
        quality_score += 0.40
    if lower_wick >= 0.25:
        quality_score += 0.35
    if recent_higher_lows(recent, needed=2):
        quality_score += 0.40
    pattern = "NONE"
    if bullish_breakdown_failure:
        pattern = "BULLISH_BREAKDOWN_FAILURE"
    elif bullish_expansion:
        pattern = "BULLISH_EXPANSION"
    elif bullish_engulfing:
        pattern = "BULLISH_ENGULFING"
    elif bullish_morning_star:
        pattern = "BULLISH_MORNING_STAR"
    elif bullish_inside_bar_failure:
        pattern = "BULLISH_INSIDE_BAR_FAILURE"
    elif bullish_hammer:
        pattern = "BULLISH_HAMMER"
    elif bullish_rejection:
        pattern = "BULLISH_REJECTION_WICK"
    return {
        "pattern": pattern,
        "quality_score": round(quality_score, 4),
        "body_fraction": round(body_fraction, 4),
        "upper_wick_fraction": round(upper_wick, 4),
        "lower_wick_fraction": round(lower_wick, 4),
        "close_location": round(close_location, 4),
        "expansion": bullish_expansion,
        "engulfing": bullish_engulfing,
        "rejection_wick": bullish_rejection,
        "breakdown_failure": bullish_breakdown_failure,
        "hammer": bullish_hammer,
        "morning_star": bullish_morning_star,
        "inside_bar_failure": bullish_inside_bar_failure,
    }


def bearish_failed_reclaim_setup(bars: list[OhlcvBar], ema20_5m: float | None, vwap: float | None) -> bool:
    if len(bars) < 6 or ema20_5m is None or vwap is None:
        return False
    recent = bars[-6:]
    last = recent[-1]
    prev = recent[-2]
    recent_high = max(bar.high for bar in recent)
    recent_low = min(bar.low for bar in recent)
    touch_buffer = max((recent_high - recent_low) * 0.20, 5.0)
    touched_ema = any(bar.high >= (ema20_5m - touch_buffer) for bar in recent[-4:-1])
    touched_vwap = any(bar.high >= (vwap - touch_buffer) for bar in recent[-4:-1])
    return (
        (touched_ema or touched_vwap)
        and last.close < last.open
        and last.close < prev.close
        and last.close < ema20_5m
        and last.close < vwap
        and close_location_value(last.close, recent_low, recent_high) <= 0.35
    )


def bearish_tight_breakdown_setup(
    bars: list[OhlcvBar],
    ema20_5m: float | None,
    vwap: float | None,
    *,
    session_low: float,
    session_high: float,
) -> bool:
    if len(bars) < 6 or ema20_5m is None or vwap is None:
        return False
    recent = bars[-6:]
    recent_high = max(bar.high for bar in recent)
    recent_low = min(bar.low for bar in recent)
    last = recent[-1]
    prev = recent[-2]
    below_ema = sum(1 for bar in recent if bar.close < ema20_5m)
    below_vwap = sum(1 for bar in recent if bar.close < vwap)
    gap_to_ema = ema20_5m - last.close
    return (
        below_vwap == len(recent)
        and below_ema >= len(recent) - 1
        and close_location_value(last.close, session_low, session_high) <= 0.12
        and close_location_value(last.close, recent_low, recent_high) <= 0.20
        and 10.0 <= gap_to_ema <= 30.0
        and (last.close < prev.close or last.close < last.open)
        and (recent_high - recent_low) <= 25.0
    )


def bullish_pullback_reclaim_setup(
    bars: list[OhlcvBar],
    ema20_5m: float | None,
    ema50_5m: float | None,
    vwap: float | None,
    *,
    support_level: float | None,
) -> bool:
    if len(bars) < 6 or ema20_5m is None or ema50_5m is None or vwap is None:
        return False
    recent = bars[-6:]
    last = recent[-1]
    prev = recent[-2]
    recent_high = max(bar.high for bar in recent)
    recent_low = min(bar.low for bar in recent)
    reference_support = max(level for level in [support_level, ema20_5m, vwap] if level is not None)
    touch_buffer = max((recent_high - recent_low) * 0.28, 10.0)
    touched_support = any(bar.low <= (reference_support + touch_buffer) for bar in recent[-4:-1])
    reclaim_strength = max(prev.close, ema20_5m, vwap)
    return (
        touched_support
        and last.close > last.open
        and (last.close > prev.high or last.high > prev.high)
        and last.close > reclaim_strength
        and ema20_5m > ema50_5m
        and last.low >= (reference_support - touch_buffer)
        and close_location_value(last.close, recent_low, recent_high) >= 0.58
    )


def bullish_shallow_continuation_setup(
    bars: list[OhlcvBar],
    ema20_5m: float | None,
    ema50_5m: float | None,
    vwap: float | None,
) -> bool:
    if len(bars) < 6 or ema20_5m is None or ema50_5m is None or vwap is None:
        return False
    recent = bars[-6:]
    last = recent[-1]
    prev = recent[-2]
    recent_high = max(bar.high for bar in recent)
    recent_low = min(bar.low for bar in recent)
    closes_above_ema20 = sum(1 for bar in recent if bar.close > ema20_5m)
    closes_above_vwap = sum(1 for bar in recent if bar.close > vwap)
    pullback_seen = any(bar.close < bar.open for bar in recent[-3:-1])
    return (
        closes_above_ema20 >= 4
        and closes_above_vwap >= 4
        and (last.close > prev.high or last.high > prev.high)
        and last.close > ema20_5m > ema50_5m
        and pullback_seen
        and close_location_value(last.close, recent_low, recent_high) >= 0.62
    )


def bullish_vwap_hold_higher_low_setup(
    bars: list[OhlcvBar],
    ema20_5m: float | None,
    ema50_5m: float | None,
    vwap: float | None,
) -> bool:
    if len(bars) < 8 or ema20_5m is None or ema50_5m is None or vwap is None:
        return False
    recent = bars[-8:]
    last = recent[-1]
    prev = recent[-2]
    recent_high = max(bar.high for bar in recent)
    recent_low = min(bar.low for bar in recent)
    closes_above_vwap = sum(1 for bar in recent if bar.close > vwap)
    higher_low_seen = any(
        recent[idx].low > recent[idx - 1].low
        and recent[idx].close >= recent[idx].open
        for idx in range(2, len(recent) - 1)
    )
    dip_to_vwap = any(bar.low <= (vwap + 8.0) for bar in recent[-5:-1])
    return (
        closes_above_vwap >= 6
        and last.close > prev.close
        and last.close > vwap
        and last.close > ema20_5m > ema50_5m
        and higher_low_seen
        and dip_to_vwap
        and close_location_value(last.close, recent_low, recent_high) >= 0.56
    )


def open_drive_bullish_reclaim_setup(
    bars: list[OhlcvBar],
    opening_range: ORLevels | None,
    ema20_5m: float | None,
    ema50_5m: float | None,
    vwap: float | None,
) -> bool:
    if len(bars) < 8 or opening_range is None or ema20_5m is None or ema50_5m is None or vwap is None:
        return False
    if bars[-1].timestamp.time() > time(11, 15):
        return False
    breakout_idx: int | None = None
    for idx, bar in enumerate(bars):
        if bar.timestamp.time() > time(10, 45):
            break
        if (
            bar.close > opening_range.high
            and bar.close > vwap
            and bar.close > ema20_5m > ema50_5m
            and close_location_value(bar.close, bar.low, bar.high) >= 0.55
        ):
            breakout_idx = idx
            break
    if breakout_idx is None or breakout_idx >= len(bars) - 2:
        return False
    last = bars[-1]
    prev = bars[-2]
    post_breakout = bars[breakout_idx + 1 : -1]
    if len(post_breakout) < 1:
        return False
    recent_high = max(bar.high for bar in bars[-6:])
    recent_low = min(bar.low for bar in bars[-6:])
    support_zone = max(opening_range.high, ema20_5m)
    touch_buffer = max((recent_high - recent_low) * 0.35, 18.0)
    touched_support = any(
        (support_zone - (touch_buffer * 1.5)) <= bar.low <= (support_zone + touch_buffer)
        for bar in post_breakout
    )
    constructive_pullback = any(bar.close <= bar.open or bar.low <= (support_zone + touch_buffer) for bar in post_breakout)
    no_deep_failure = all(bar.low >= (opening_range.low - (touch_buffer * 1.25)) and bar.close >= (vwap - (touch_buffer * 1.25)) for bar in post_breakout)
    return (
        touched_support
        and constructive_pullback
        and no_deep_failure
        and last.close > max(prev.close, support_zone)
        and last.close > ema20_5m > ema50_5m
        and last.close > vwap
        and close_location_value(last.close, recent_low, recent_high) >= 0.56
    )


def gap_up_continuation_setup(
    bars: list[OhlcvBar],
    opening_range: ORLevels | None,
    ema20_5m: float | None,
    ema50_5m: float | None,
    vwap: float | None,
) -> bool:
    if len(bars) < 8 or opening_range is None or ema20_5m is None or ema50_5m is None or vwap is None:
        return False
    if bars[-1].timestamp.time() > time(11, 30):
        return False
    recent = bars[-8:]
    last = recent[-1]
    prev = recent[-2]
    recent_high = max(bar.high for bar in recent)
    recent_low = min(bar.low for bar in recent)
    support_zone = max(opening_range.high, ema20_5m, vwap)
    touch_buffer = max((recent_high - recent_low) * 0.28, 12.0)
    pullback_seen = any(
        (support_zone - (touch_buffer * 1.25)) <= bar.low <= (support_zone + touch_buffer)
        and bar.close >= vwap
        for bar in recent[-5:-1]
    )
    held_above_support = all(bar.close >= (support_zone - touch_buffer) for bar in recent[-4:])
    return (
        pullback_seen
        and held_above_support
        and last.close > max(prev.high, support_zone)
        and last.close > ema20_5m > ema50_5m
        and close_location_value(last.close, recent_low, recent_high) >= 0.60
    )


def gap_up_failure_reversal_setup(
    bars: list[OhlcvBar],
    opening_range: ORLevels | None,
    ema20_5m: float | None,
    ema50_5m: float | None,
    vwap: float | None,
) -> bool:
    if len(bars) < 8 or opening_range is None or ema20_5m is None or ema50_5m is None or vwap is None:
        return False
    recent = bars[-8:]
    last = recent[-1]
    prev = recent[-2]
    recent_high = max(bar.high for bar in recent)
    recent_low = min(bar.low for bar in recent)
    rejection_seen = any(bar.high >= (ema20_5m - 8.0) for bar in recent[-4:-1])
    return (
        rejection_seen
        and last.close < min(prev.close, opening_range.low, vwap)
        and last.close < ema20_5m < ema50_5m
        and close_location_value(last.close, recent_low, recent_high) <= 0.28
    )


def gap_down_continuation_setup(
    bars: list[OhlcvBar],
    opening_range: ORLevels | None,
    ema20_5m: float | None,
    ema50_5m: float | None,
    vwap: float | None,
) -> bool:
    if len(bars) < 8 or opening_range is None or ema20_5m is None or ema50_5m is None or vwap is None:
        return False
    recent = bars[-8:]
    last = recent[-1]
    prev = recent[-2]
    recent_high = max(bar.high for bar in recent)
    recent_low = min(bar.low for bar in recent)
    resistance_zone = min(opening_range.low, ema20_5m, vwap)
    touch_buffer = max((recent_high - recent_low) * 0.24, 10.0)
    rejection_seen = any(
        (resistance_zone - touch_buffer) <= bar.high <= (resistance_zone + (touch_buffer * 1.25))
        for bar in recent[-5:-1]
    )
    return (
        rejection_seen
        and last.close < min(prev.close, opening_range.low, vwap)
        and last.close < ema20_5m < ema50_5m
        and close_location_value(last.close, recent_low, recent_high) <= 0.25
    )


def gap_down_recovery_setup(
    bars: list[OhlcvBar],
    opening_range: ORLevels | None,
    ema20_5m: float | None,
    ema50_5m: float | None,
    vwap: float | None,
) -> bool:
    if len(bars) < 10 or opening_range is None or ema20_5m is None or ema50_5m is None or vwap is None:
        return False
    if bars[-1].timestamp.time() < time(10, 15):
        return False
    recent = bars[-8:]
    last = recent[-1]
    prev = recent[-2]
    recent_high = max(bar.high for bar in recent)
    recent_low = min(bar.low for bar in recent)
    recovery_zone = max(opening_range.high, vwap, ema20_5m)
    touch_buffer = max((recent_high - recent_low) * 0.30, 12.0)
    support_flip = any(
        (recovery_zone - (touch_buffer * 1.25)) <= bar.low <= (recovery_zone + touch_buffer)
        for bar in recent[-4:-1]
    )
    held_vwap = sum(1 for bar in recent[-5:] if bar.close > vwap) >= 4
    return (
        support_flip
        and held_vwap
        and last.close > max(prev.close, recovery_zone)
        and last.close > ema20_5m > ema50_5m
        and close_location_value(last.close, recent_low, recent_high) >= 0.58
    )


def sideways_bullish_reclaim_setup(
    bars: list[OhlcvBar],
    opening_range: ORLevels | None,
    ema20_5m: float | None,
    ema50_5m: float | None,
    vwap: float | None,
) -> bool:
    if len(bars) < 18 or opening_range is None or ema20_5m is None or ema50_5m is None or vwap is None:
        return False
    if bars[-1].timestamp.time() < time(11, 30):
        return False
    morning_balance = [bar for bar in bars if time(9, 30) <= bar.timestamp.time() < time(11, 30)]
    if len(morning_balance) < 12:
        return False
    inside_count = sum(1 for bar in morning_balance if bar.high <= opening_range.high and bar.low >= opening_range.low)
    if inside_count / len(morning_balance) < 0.60:
        return False
    if not oscillates_around_vwap(morning_balance, vwap, required_bars=min(10, len(morning_balance))):
        return False
    afternoon = [bar for bar in bars if bar.timestamp.time() >= time(11, 30)]
    if len(afternoon) < 3:
        return False
    breakout_seen = any(bar.close > opening_range.high for bar in afternoon[:-1])
    if not breakout_seen:
        return False
    last = bars[-1]
    prev = bars[-2]
    recent_high = max(bar.high for bar in bars[-6:])
    recent_low = min(bar.low for bar in bars[-6:])
    support_zone = max(opening_range.high, min(ema20_5m, last.close))
    touch_buffer = max((recent_high - recent_low) * 0.24, 12.0)
    reclaim_window = afternoon[-4:-1] if len(afternoon) >= 4 else afternoon[:-1]
    support_flip = any((support_zone - (touch_buffer * 1.5)) <= bar.low <= (support_zone + touch_buffer) for bar in reclaim_window)
    return (
        support_flip
        and last.close > max(prev.close, support_zone)
        and last.close > ema20_5m > ema50_5m
        and last.close > vwap
        and close_location_value(last.close, recent_low, recent_high) >= 0.58
    )


def balanced_range_condor_setup(
    bars: list[OhlcvBar],
    opening_range: ORLevels | None,
    vwap: float | None,
    *,
    rv30_pct: float,
) -> bool:
    if len(bars) < 18 or opening_range is None or vwap is None:
        return False
    if bars[-1].timestamp.time() < time(11, 30):
        return False
    if rv30_pct >= 0.30:
        return False
    recent = bars[-12:]
    inside_count = sum(1 for bar in recent if bar.high <= opening_range.high and bar.low >= opening_range.low)
    if inside_count / len(recent) < 0.67:
        return False
    if not oscillates_around_vwap(recent, vwap, required_bars=min(6, len(recent))):
        return False
    recent_range_pct = price_change_pct(min(bar.low for bar in recent), max(bar.high for bar in recent))
    if recent_range_pct >= 0.55:
        return False
    range_mid = (opening_range.high + opening_range.low) / 2.0
    return abs(bars[-1].close - range_mid) / max(bars[-1].close, 1.0) <= 0.0025


def bearish_pullback_rejection_setup(
    bars: list[OhlcvBar],
    ema20_5m: float | None,
    ema50_5m: float | None,
    vwap: float | None,
    *,
    resistance_level: float | None,
) -> bool:
    if len(bars) < 6 or ema20_5m is None or ema50_5m is None or vwap is None:
        return False
    recent = bars[-6:]
    last = recent[-1]
    prev = recent[-2]
    recent_high = max(bar.high for bar in recent)
    recent_low = min(bar.low for bar in recent)
    reference_resistance = min(level for level in [resistance_level, ema20_5m, vwap] if level is not None)
    touch_buffer = max((recent_high - recent_low) * 0.22, 8.0)
    touched_resistance = any(bar.high >= (reference_resistance - touch_buffer) for bar in recent[-4:-1])
    return (
        touched_resistance
        and last.close < last.open
        and last.close < prev.low
        and last.close < ema20_5m
        and last.close < ema50_5m
        and last.close < vwap
        and last.high <= (reference_resistance + touch_buffer)
        and close_location_value(last.close, recent_low, recent_high) <= 0.35
    )


def bearish_shallow_continuation_setup(
    bars: list[OhlcvBar],
    ema20_5m: float | None,
    ema50_5m: float | None,
    vwap: float | None,
) -> bool:
    if len(bars) < 6 or ema20_5m is None or ema50_5m is None or vwap is None:
        return False
    recent = bars[-6:]
    last = recent[-1]
    prev = recent[-2]
    recent_high = max(bar.high for bar in recent)
    recent_low = min(bar.low for bar in recent)
    closes_below_ema20 = sum(1 for bar in recent if bar.close < ema20_5m)
    closes_below_vwap = sum(1 for bar in recent if bar.close < vwap)
    bounce_seen = any(bar.close > bar.open for bar in recent[-3:-1])
    return (
        closes_below_ema20 >= 5
        and closes_below_vwap >= 5
        and last.close < prev.low
        and last.close < ema20_5m < ema50_5m
        and bounce_seen
        and close_location_value(last.close, recent_low, recent_high) <= 0.28
    )


def remains_inside_opening_range(bars_5m: list[OhlcvBar], opening_range: ORLevels, required_bars: int = 6) -> bool:
    if len(bars_5m) < required_bars:
        return False
    recent = bars_5m[-required_bars:]
    return all(bar.high <= opening_range.high and bar.low >= opening_range.low for bar in recent)


def oscillates_around_vwap(bars_5m: list[OhlcvBar], vwap: float, required_bars: int = 6) -> bool:
    if len(bars_5m) < required_bars or vwap <= 0:
        return False
    recent = bars_5m[-required_bars:]
    closes_above = sum(1 for bar in recent if bar.close > vwap)
    closes_below = sum(1 for bar in recent if bar.close < vwap)
    mean_deviation_pct = fmean(abs(bar.close - vwap) / vwap * 100.0 for bar in recent)
    return closes_above >= 2 and closes_below >= 2 and mean_deviation_pct <= 0.20


def opening_range_whipsaw(
    bars_5m: list[OhlcvBar],
    opening_range: ORLevels,
    *,
    cutoff_time: time = RANGE_GATE_TIME,
) -> bool:
    early = [bar for bar in bars_5m if bar.timestamp.time() <= cutoff_time]
    if not early:
        return False
    breached_up = any(bar.close > opening_range.high for bar in early)
    breached_down = any(bar.close < opening_range.low for bar in early)
    return breached_up and breached_down


def nearest_support(bars: list[OhlcvBar], spot: float, count: int = 1) -> list[float]:
    pivots = [pivot for pivot in _pivot_lows(bars) if pivot < spot]
    return sorted(pivots, reverse=True)[:count]


def nearest_resistance(bars: list[OhlcvBar], spot: float, count: int = 1) -> list[float]:
    pivots = [pivot for pivot in _pivot_highs(bars) if pivot > spot]
    return sorted(pivots)[:count]


def latest_pivot_low(bars: list[OhlcvBar], *, lookback: int = 12) -> float | None:
    recent = bars[-lookback:] if lookback > 0 else bars
    pivots = _pivot_lows(recent)
    return pivots[-1] if pivots else None


def latest_pivot_high(bars: list[OhlcvBar], *, lookback: int = 12) -> float | None:
    recent = bars[-lookback:] if lookback > 0 else bars
    pivots = _pivot_highs(recent)
    return pivots[-1] if pivots else None


def latest_fair_value_gap(
    bars: list[OhlcvBar],
    direction: str,
    *,
    lookback: int = 16,
    min_gap_points: float = 5.0,
) -> dict[str, float | bool | None]:
    if len(bars) < 3:
        return {"active": False, "low": None, "high": None, "size": 0.0}
    recent = bars[-lookback:] if lookback > 0 else bars
    last = recent[-1]
    for idx in range(len(recent) - 1, 1, -1):
        left = recent[idx - 2]
        middle = recent[idx - 1]
        right = recent[idx]
        if direction == "bullish" and right.low > left.high and right.close >= middle.high:
            gap_low = left.high
            gap_high = right.low
            gap_size = gap_high - gap_low
            if gap_size < min_gap_points:
                continue
            active = last.close >= gap_high and ((last.close - gap_high) / max(last.close, 1.0) * 100.0) <= 0.45
            return {"active": active, "low": gap_low, "high": gap_high, "size": gap_size}
        if direction == "bearish" and right.high < left.low and right.close <= middle.low:
            gap_low = right.high
            gap_high = left.low
            gap_size = gap_high - gap_low
            if gap_size < min_gap_points:
                continue
            active = last.close <= gap_low and ((gap_low - last.close) / max(last.close, 1.0) * 100.0) <= 0.45
            return {"active": active, "low": gap_low, "high": gap_high, "size": gap_size}
    return {"active": False, "low": None, "high": None, "size": 0.0}


def latest_order_block_zone(
    bars: list[OhlcvBar],
    direction: str,
    *,
    lookback: int = 20,
    min_impulse_points: float = 15.0,
) -> dict[str, float | bool | None]:
    if len(bars) < 5:
        return {"active": False, "low": None, "high": None, "impulse": 0.0}
    recent = bars[-lookback:] if lookback > 0 else bars
    last = recent[-1]
    for idx in range(len(recent) - 4, -1, -1):
        bar = recent[idx]
        future = recent[idx + 1 : idx + 4]
        if len(future) < 2:
            continue
        if direction == "bullish" and bar.close < bar.open:
            impulse = max(candidate.high for candidate in future) - bar.high
            if impulse < min_impulse_points or future[-1].close <= bar.high:
                continue
            zone_low = bar.low
            zone_high = max(bar.open, bar.close)
            active = last.close >= zone_high and ((last.close - zone_high) / max(last.close, 1.0) * 100.0) <= 0.55
            return {"active": active, "low": zone_low, "high": zone_high, "impulse": impulse}
        if direction == "bearish" and bar.close > bar.open:
            impulse = bar.low - min(candidate.low for candidate in future)
            if impulse < min_impulse_points or future[-1].close >= bar.low:
                continue
            zone_low = min(bar.open, bar.close)
            zone_high = bar.high
            active = last.close <= zone_low and ((zone_low - last.close) / max(last.close, 1.0) * 100.0) <= 0.55
            return {"active": active, "low": zone_low, "high": zone_high, "impulse": impulse}
    return {"active": False, "low": None, "high": None, "impulse": 0.0}


def market_structure_state(
    bars: list[OhlcvBar],
    ema20_5m: float | None,
    ema50_5m: float | None,
    vwap: float | None,
) -> dict[str, float | bool | None]:
    default_state: dict[str, float | bool | None] = {
        "bullish_bos": False,
        "bearish_bos": False,
        "bullish_choch": False,
        "bearish_choch": False,
        "bullish_fvg_active": False,
        "bullish_fvg_low": None,
        "bullish_fvg_high": None,
        "bearish_fvg_active": False,
        "bearish_fvg_low": None,
        "bearish_fvg_high": None,
        "bullish_order_block_active": False,
        "bullish_order_block_low": None,
        "bullish_order_block_high": None,
        "bearish_order_block_active": False,
        "bearish_order_block_low": None,
        "bearish_order_block_high": None,
        "bullish_smc_alignment": False,
        "bearish_smc_alignment": False,
    }
    if len(bars) < 8:
        return default_state

    recent = bars[-8:]
    last = recent[-1]
    pivot_high = latest_pivot_high(bars, lookback=min(20, len(bars)))
    pivot_low = latest_pivot_low(bars, lookback=min(20, len(bars)))
    bullish_bos = pivot_high is not None and last.close > pivot_high
    bearish_bos = pivot_low is not None and last.close < pivot_low
    recent_pre_break = recent[:-2]
    bullish_pre_bias = (
        ema20_5m is not None
        and vwap is not None
        and sum(1 for bar in recent_pre_break if bar.close < min(ema20_5m, vwap)) >= max(2, len(recent_pre_break) // 2)
    )
    bearish_pre_bias = (
        ema20_5m is not None
        and vwap is not None
        and sum(1 for bar in recent_pre_break if bar.close > max(ema20_5m, vwap)) >= max(2, len(recent_pre_break) // 2)
    )
    bullish_choch = bullish_bos and bullish_pre_bias and ema20_5m is not None and vwap is not None and last.close > ema20_5m and last.close > vwap
    bearish_choch = bearish_bos and bearish_pre_bias and ema20_5m is not None and vwap is not None and last.close < ema20_5m and last.close < vwap
    bullish_fvg = latest_fair_value_gap(bars, "bullish")
    bearish_fvg = latest_fair_value_gap(bars, "bearish")
    bullish_order_block = latest_order_block_zone(bars, "bullish")
    bearish_order_block = latest_order_block_zone(bars, "bearish")
    bullish_smc_alignment = bool((bullish_bos or bullish_choch or recent_higher_lows(recent, needed=2)) and (bullish_fvg["active"] or bullish_order_block["active"]))
    bearish_smc_alignment = bool((bearish_bos or bearish_choch or recent_lower_highs(recent, needed=2)) and (bearish_fvg["active"] or bearish_order_block["active"]))

    return {
        "bullish_bos": bullish_bos,
        "bearish_bos": bearish_bos,
        "bullish_choch": bullish_choch,
        "bearish_choch": bearish_choch,
        "bullish_fvg_active": bool(bullish_fvg["active"]),
        "bullish_fvg_low": bullish_fvg["low"],
        "bullish_fvg_high": bullish_fvg["high"],
        "bearish_fvg_active": bool(bearish_fvg["active"]),
        "bearish_fvg_low": bearish_fvg["low"],
        "bearish_fvg_high": bearish_fvg["high"],
        "bullish_order_block_active": bool(bullish_order_block["active"]),
        "bullish_order_block_low": bullish_order_block["low"],
        "bullish_order_block_high": bullish_order_block["high"],
        "bearish_order_block_active": bool(bearish_order_block["active"]),
        "bearish_order_block_low": bearish_order_block["low"],
        "bearish_order_block_high": bearish_order_block["high"],
        "bullish_smc_alignment": bullish_smc_alignment,
        "bearish_smc_alignment": bearish_smc_alignment,
    }


def early_structure_intent_bearish(
    bars_5m: list[OhlcvBar],
    ema20_5m: float | None,
    vwap: float | None,
    opening_range_high: float | None,
    opening_range_low: float | None,
    bearish_chain_pressure: float,
) -> dict:
    """
    Detects bearish intent in the first 2–5 bars of the session,
    before trend_15m and execution_5m labels confirm.

    Uses EMA20 position/crossover, LH/LL structure, candle quality,
    OR location, and chain pressure as early confluence factors.
    Fires when strength >= 2.5 AND at least one signal from each of
    the three required categories (EMA, structure, pressure) is present.
    """
    result: dict = {"triggered": False, "strength": 0.0, "signals": {}}
    if len(bars_5m) < 2:
        return result

    last = bars_5m[-1]
    spot = last.close

    # EMA signals
    below_ema20 = ema20_5m is not None and spot < ema20_5m
    ema20_cross_bearish = (
        ema20_5m is not None
        and len(bars_5m) >= 3
        and bars_5m[-1].close < ema20_5m
        and bars_5m[-2].close >= ema20_5m
    )
    ema20_holding_below = (
        ema20_5m is not None
        and len(bars_5m) >= 3
        and all(b.close < ema20_5m for b in bars_5m[-3:])
    )

    # Candle signals
    candle = bearish_candle_context(bars_5m)
    strong_bearish_candle = float(candle.get("quality_score") or 0.0) >= 2.0
    body_and_close = (
        float(candle.get("body_fraction") or 0.0) >= 0.55
        and float(candle.get("close_location") or 1.0) <= 0.30
    )

    # Market structure signals: LH/LL sequence within first few bars
    lh_ll_sequence = len(bars_5m) >= 4 and recent_lower_highs(bars_5m[-4:], needed=2)
    lower_high_from_open = (
        len(bars_5m) >= 3
        and all(b.high < bars_5m[0].high for b in bars_5m[1:])
        and (len(bars_5m) < 3 or bars_5m[-1].high <= bars_5m[-2].high)
    )

    # OR location signals
    or_mid = (
        (opening_range_high + opening_range_low) / 2.0
        if opening_range_high is not None and opening_range_low is not None
        else None
    )
    below_or_mid = or_mid is not None and spot < or_mid
    testing_or_low = opening_range_low is not None and spot <= opening_range_low * 1.001

    # Pressure signals
    strong_pressure = bearish_chain_pressure >= 0.62
    moderate_pressure = bearish_chain_pressure >= 0.55

    # VWAP
    below_vwap = vwap is not None and spot < vwap

    strength = 0.0
    signals: dict = {}

    if ema20_holding_below:
        strength += 1.0
        signals["ema20_holding_below"] = True
    elif ema20_cross_bearish:
        strength += 0.7
        signals["ema20_cross_bearish"] = True
    elif below_ema20:
        strength += 0.5
        signals["below_ema20"] = True

    if strong_bearish_candle:
        strength += 1.0
        signals["strong_bearish_candle"] = True
    elif body_and_close:
        strength += 0.6
        signals["bearish_body_and_close"] = True

    if lh_ll_sequence:
        strength += 0.8
        signals["lh_ll_sequence"] = True
    elif lower_high_from_open:
        strength += 0.4
        signals["lower_high_from_open"] = True

    if strong_pressure:
        strength += 0.8
        signals["strong_chain_pressure"] = True
    elif moderate_pressure:
        strength += 0.4
        signals["moderate_chain_pressure"] = True

    if below_or_mid:
        strength += 0.4
        signals["below_or_mid"] = True
    if testing_or_low:
        strength += 0.3
        signals["testing_or_low"] = True
    if below_vwap:
        strength += 0.4
        signals["below_vwap"] = True

    has_ema = bool(below_ema20 or ema20_cross_bearish or ema20_holding_below)
    has_structure = bool(lh_ll_sequence or lower_high_from_open or strong_bearish_candle)
    has_pressure = bool(strong_pressure or moderate_pressure)

    result["triggered"] = bool(strength >= 2.5 and has_ema and has_structure and has_pressure)
    result["strength"] = round(strength, 3)
    result["signals"] = signals
    return result


def early_structure_intent_bullish(
    bars_5m: list[OhlcvBar],
    ema20_5m: float | None,
    vwap: float | None,
    opening_range_high: float | None,
    opening_range_low: float | None,
    bullish_chain_pressure: float,
) -> dict:
    """
    Symmetric bullish counterpart to early_structure_intent_bearish.
    Detects bullish intent in the first 2–5 bars using EMA20 position,
    HL/HH structure, candle quality, OR location, and chain pressure.
    """
    result: dict = {"triggered": False, "strength": 0.0, "signals": {}}
    if len(bars_5m) < 2:
        return result

    last = bars_5m[-1]
    spot = last.close

    # EMA signals
    above_ema20 = ema20_5m is not None and spot > ema20_5m
    ema20_cross_bullish = (
        ema20_5m is not None
        and len(bars_5m) >= 3
        and bars_5m[-1].close > ema20_5m
        and bars_5m[-2].close <= ema20_5m
    )
    ema20_holding_above = (
        ema20_5m is not None
        and len(bars_5m) >= 3
        and all(b.close > ema20_5m for b in bars_5m[-3:])
    )

    # Candle signals
    candle = bullish_candle_context(bars_5m)
    strong_bullish_candle = float(candle.get("quality_score") or 0.0) >= 2.0
    body_and_close = (
        float(candle.get("body_fraction") or 0.0) >= 0.55
        and float(candle.get("close_location") or 0.0) >= 0.70
    )

    # Market structure: HL/HH sequence
    hl_hh_sequence = len(bars_5m) >= 4 and recent_higher_lows(bars_5m[-4:], needed=2)
    higher_low_from_open = (
        len(bars_5m) >= 3
        and all(b.low > bars_5m[0].low for b in bars_5m[1:])
        and (len(bars_5m) < 3 or bars_5m[-1].low >= bars_5m[-2].low)
    )

    # OR location
    or_mid = (
        (opening_range_high + opening_range_low) / 2.0
        if opening_range_high is not None and opening_range_low is not None
        else None
    )
    above_or_mid = or_mid is not None and spot > or_mid
    testing_or_high = opening_range_high is not None and spot >= opening_range_high * 0.999

    # Pressure
    strong_pressure = bullish_chain_pressure >= 0.62
    moderate_pressure = bullish_chain_pressure >= 0.55

    # VWAP
    above_vwap = vwap is not None and spot > vwap

    strength = 0.0
    signals: dict = {}

    if ema20_holding_above:
        strength += 1.0
        signals["ema20_holding_above"] = True
    elif ema20_cross_bullish:
        strength += 0.7
        signals["ema20_cross_bullish"] = True
    elif above_ema20:
        strength += 0.5
        signals["above_ema20"] = True

    if strong_bullish_candle:
        strength += 1.0
        signals["strong_bullish_candle"] = True
    elif body_and_close:
        strength += 0.6
        signals["bullish_body_and_close"] = True

    if hl_hh_sequence:
        strength += 0.8
        signals["hl_hh_sequence"] = True
    elif higher_low_from_open:
        strength += 0.4
        signals["higher_low_from_open"] = True

    if strong_pressure:
        strength += 0.8
        signals["strong_chain_pressure"] = True
    elif moderate_pressure:
        strength += 0.4
        signals["moderate_chain_pressure"] = True

    if above_or_mid:
        strength += 0.4
        signals["above_or_mid"] = True
    if testing_or_high:
        strength += 0.3
        signals["testing_or_high"] = True
    if above_vwap:
        strength += 0.4
        signals["above_vwap"] = True

    has_ema = bool(above_ema20 or ema20_cross_bullish or ema20_holding_above)
    has_structure = bool(hl_hh_sequence or higher_low_from_open or strong_bullish_candle)
    has_pressure = bool(strong_pressure or moderate_pressure)

    result["triggered"] = bool(strength >= 2.5 and has_ema and has_structure and has_pressure)
    result["strength"] = round(strength, 3)
    result["signals"] = signals
    return result


def in_market_hours(ts: datetime) -> bool:
    return MARKET_OPEN <= ts.time() <= FORCE_EXIT_TIME


def option_chain_pressure(spot: float, quotes: list[OptionsContractQuote], depth: int = 3) -> dict[str, float]:
    call_quotes = sorted(
        [quote for quote in quotes if quote.option_type == OptionType.CALL and quote.strike > spot and quote.oi is not None],
        key=lambda quote: quote.strike,
    )[:depth]
    put_quotes = sorted(
        [quote for quote in quotes if quote.option_type == OptionType.PUT and quote.strike < spot and quote.oi is not None],
        key=lambda quote: quote.strike,
        reverse=True,
    )[:depth]
    call_oi = float(sum(quote.oi or 0 for quote in call_quotes))
    put_oi = float(sum(quote.oi or 0 for quote in put_quotes))
    total_oi = max(call_oi + put_oi, 1.0)
    return {
        "call_oi_nearby": call_oi,
        "put_oi_nearby": put_oi,
        "bullish_pressure": put_oi / total_oi,
        "bearish_pressure": call_oi / total_oi,
    }


def put_call_ratio_by_oi(quotes: list[OptionsContractQuote]) -> float | None:
    total_put_oi = sum(float(quote.oi or 0.0) for quote in quotes if quote.option_type == OptionType.PUT)
    total_call_oi = sum(float(quote.oi or 0.0) for quote in quotes if quote.option_type == OptionType.CALL)
    if total_put_oi <= 0 or total_call_oi <= 0:
        return None
    return total_put_oi / total_call_oi


def compute_pcr_trend(
    current_quotes: list[OptionsContractQuote],
    previous_quotes: list[OptionsContractQuote] | None,
    threshold: float = 0.05,
) -> str:
    if not previous_quotes:
        return "UNKNOWN"
    current_pcr = put_call_ratio_by_oi(current_quotes)
    previous_pcr = put_call_ratio_by_oi(previous_quotes)
    if current_pcr is None or previous_pcr is None:
        return "UNKNOWN"
    delta = current_pcr - previous_pcr
    if delta > threshold:
        return "RISING"
    if delta < -threshold:
        return "FALLING"
    return "STABLE"


def option_chain_oi_flow(
    spot: float,
    current_quotes: list[OptionsContractQuote],
    previous_quotes: list[OptionsContractQuote] | None,
    depth: int = 4,
) -> dict[str, float | str | None]:
    if not previous_quotes:
        return {
            "bullish_flow_score": 0.0,
            "bearish_flow_score": 0.0,
            "put_support_strike": None,
            "call_resistance_strike": None,
            "put_support_oi_change": 0.0,
            "call_resistance_oi_change": 0.0,
            "smart_money_bias": "UNKNOWN",
        }

    previous_map = {
        (quote.strike, quote.option_type): quote
        for quote in previous_quotes
    }

    def _nearby(side: OptionType) -> list[tuple[OptionsContractQuote, OptionsContractQuote]]:
        eligible: list[tuple[OptionsContractQuote, OptionsContractQuote]] = []
        for quote in current_quotes:
            if quote.option_type != side or quote.oi is None:
                continue
            previous = previous_map.get((quote.strike, quote.option_type))
            if previous is None or previous.oi is None:
                continue
            if side == OptionType.CALL and quote.strike <= spot:
                continue
            if side == OptionType.PUT and quote.strike >= spot:
                continue
            eligible.append((quote, previous))
        key_fn = (lambda item: item[0].strike - spot) if side == OptionType.CALL else (lambda item: spot - item[0].strike)
        return sorted(eligible, key=key_fn)[:depth]

    nearby_calls = _nearby(OptionType.CALL)
    nearby_puts = _nearby(OptionType.PUT)

    call_write = 0.0
    call_unwind = 0.0
    put_write = 0.0
    put_unwind = 0.0
    call_wall_strike: float | None = None
    put_wall_strike: float | None = None
    call_wall_change = 0.0
    put_wall_change = 0.0

    for current, previous in nearby_calls:
        oi_change = float((current.oi or 0) - (previous.oi or 0))
        premium_change = current.ltp - previous.ltp
        if oi_change > call_wall_change:
            call_wall_change = oi_change
            call_wall_strike = current.strike
        if oi_change > 0 and premium_change <= 0:
            call_write += oi_change
        if oi_change < 0 and premium_change >= 0:
            call_unwind += abs(oi_change)

    for current, previous in nearby_puts:
        oi_change = float((current.oi or 0) - (previous.oi or 0))
        premium_change = current.ltp - previous.ltp
        if oi_change > put_wall_change:
            put_wall_change = oi_change
            put_wall_strike = current.strike
        if oi_change > 0 and premium_change <= 0:
            put_write += oi_change
        if oi_change < 0 and premium_change >= 0:
            put_unwind += abs(oi_change)

    bullish_flow = put_write + call_unwind
    bearish_flow = call_write + put_unwind
    total_flow = max(bullish_flow + bearish_flow, 1.0)
    bullish_score = bullish_flow / total_flow
    bearish_score = bearish_flow / total_flow
    if bullish_score >= 0.60 and bullish_score > bearish_score:
        smart_money_bias = "BULLISH"
    elif bearish_score >= 0.60 and bearish_score > bullish_score:
        smart_money_bias = "BEARISH"
    else:
        smart_money_bias = "NEUTRAL"

    # OI velocity: rate of buildup at dominant wall strikes (pct of prior OI per poll cycle).
    # High velocity at call wall = resistance hardening fast → reinforces bearish bias.
    # High velocity at put wall = support flooring fast → reinforces bullish bias.
    call_wall_prev_oi = next(
        (float(prev.oi or 0) for cur, prev in nearby_calls if cur.strike == call_wall_strike),
        0.0,
    )
    put_wall_prev_oi = next(
        (float(prev.oi or 0) for cur, prev in nearby_puts if cur.strike == put_wall_strike),
        0.0,
    )
    call_wall_velocity = round(call_wall_change / max(call_wall_prev_oi, 1.0) * 100, 2)
    put_wall_velocity = round(put_wall_change / max(put_wall_prev_oi, 1.0) * 100, 2)

    return {
        "bullish_flow_score": bullish_score,
        "bearish_flow_score": bearish_score,
        "put_support_strike": put_wall_strike,
        "call_resistance_strike": call_wall_strike,
        "put_support_oi_change": put_wall_change,
        "call_resistance_oi_change": call_wall_change,
        "smart_money_bias": smart_money_bias,
        "call_wall_oi_velocity": call_wall_velocity,
        "put_wall_oi_velocity": put_wall_velocity,
    }


def option_chain_wall_migration(
    spot: float,
    current_quotes: list[OptionsContractQuote],
    previous_quotes: list[OptionsContractQuote] | None,
    depth: int = 4,
) -> dict[str, float | str | None]:
    if not previous_quotes:
        return {
            "current_put_wall": None,
            "previous_put_wall": None,
            "put_wall_shift": 0.0,
            "current_call_wall": None,
            "previous_call_wall": None,
            "call_wall_shift": 0.0,
            "bullish_wall_score": 0.0,
            "bearish_wall_score": 0.0,
            "wall_migration_bias": "UNKNOWN",
        }

    def _dominant_wall(quotes: list[OptionsContractQuote], option_type: OptionType) -> float | None:
        eligible = [
            quote
            for quote in quotes
            if quote.option_type == option_type
            and quote.oi is not None
            and ((option_type == OptionType.CALL and quote.strike > spot) or (option_type == OptionType.PUT and quote.strike < spot))
        ]
        if not eligible:
            return None
        if option_type == OptionType.CALL:
            nearby = sorted(eligible, key=lambda quote: quote.strike - spot)[:depth]
        else:
            nearby = sorted(eligible, key=lambda quote: spot - quote.strike)[:depth]
        if not nearby:
            return None
        return max(nearby, key=lambda quote: quote.oi or 0).strike

    current_put_wall = _dominant_wall(current_quotes, OptionType.PUT)
    previous_put_wall = _dominant_wall(previous_quotes, OptionType.PUT)
    current_call_wall = _dominant_wall(current_quotes, OptionType.CALL)
    previous_call_wall = _dominant_wall(previous_quotes, OptionType.CALL)

    put_wall_shift = float((current_put_wall or 0.0) - (previous_put_wall or current_put_wall or 0.0)) if current_put_wall is not None and previous_put_wall is not None else 0.0
    call_wall_shift = float((current_call_wall or 0.0) - (previous_call_wall or current_call_wall or 0.0)) if current_call_wall is not None and previous_call_wall is not None else 0.0

    bullish_score = 0.0
    bearish_score = 0.0
    if put_wall_shift > 0:
        bullish_score += 1.0
    elif put_wall_shift < 0:
        bearish_score += 0.75
    if call_wall_shift > 0:
        bullish_score += 0.75
    elif call_wall_shift < 0:
        bearish_score += 1.0

    if bullish_score >= 1.25 and bullish_score > bearish_score:
        bias = "BULLISH"
    elif bearish_score >= 1.25 and bearish_score > bullish_score:
        bias = "BEARISH"
    else:
        bias = "NEUTRAL"

    return {
        "current_put_wall": current_put_wall,
        "previous_put_wall": previous_put_wall,
        "put_wall_shift": put_wall_shift,
        "current_call_wall": current_call_wall,
        "previous_call_wall": previous_call_wall,
        "call_wall_shift": call_wall_shift,
        "bullish_wall_score": bullish_score,
        "bearish_wall_score": bearish_score,
        "wall_migration_bias": bias,
    }


def _ema(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    k = 2.0 / (period + 1)
    ema = sum(values[:period]) / period
    for v in values[period:]:
        ema = v * k + ema * (1 - k)
    return ema


def compute_daily_trend_features(daily_series: OhlcvSeries | None) -> dict[str, object]:
    """Multi-timeframe daily trend analysis for strategy bias.

    Returns a dict with:
    - daily_trend: STRONGLY_BULLISH / BULLISH / NEUTRAL / BEARISH / STRONGLY_BEARISH
    - daily_bias_score: -3 (strongly bearish) to +3 (strongly bullish)
    - daily_ema20, daily_ema50, daily_ema200
    - weekly_high, weekly_low (last 5 trading days)
    - daily_atr: average true range over last 14 days
    - price_vs_ema20_pct, price_vs_ema50_pct
    - higher_tf_available: bool
    """
    empty: dict[str, object] = {
        "daily_trend": "NEUTRAL",
        "daily_bias_score": 0,
        "daily_ema20": None,
        "daily_ema50": None,
        "daily_ema200": None,
        "weekly_high": None,
        "weekly_low": None,
        "daily_atr": None,
        "price_vs_ema20_pct": None,
        "price_vs_ema50_pct": None,
        "higher_tf_available": False,
    }
    if daily_series is None or not daily_series.bars:
        return empty

    bars = daily_series.bars
    closes = [b.close for b in bars]
    highs = [b.high for b in bars]
    lows = [b.low for b in bars]

    last_close = closes[-1]
    ema20 = _ema(closes, 20)
    ema50 = _ema(closes, 50)
    ema200 = _ema(closes, 200)

    # Weekly levels: last 5 trading days
    recent = bars[-5:] if len(bars) >= 5 else bars
    weekly_high = max(b.high for b in recent)
    weekly_low = min(b.low for b in recent)

    # ATR (14-day)
    atr: float | None = None
    if len(bars) >= 15:
        true_ranges: list[float] = []
        for i in range(1, min(15, len(bars))):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            true_ranges.append(tr)
        atr = fmean(true_ranges) if true_ranges else None

    # Distance from EMAs
    price_vs_ema20_pct = ((last_close - ema20) / ema20 * 100) if ema20 else None
    price_vs_ema50_pct = ((last_close - ema50) / ema50 * 100) if ema50 else None

    # Trend scoring: each alignment adds/subtracts from bias_score
    bias_score = 0
    if ema20 and ema50:
        if last_close > ema20 > ema50:
            bias_score += 2  # price above both short and medium EMAs
        elif last_close > ema20:
            bias_score += 1
        elif last_close < ema20 < ema50:
            bias_score -= 2
        elif last_close < ema20:
            bias_score -= 1
    if ema200:
        if last_close > ema200:
            bias_score += 1
        else:
            bias_score -= 1

    # Classify trend
    if bias_score >= 3:
        trend = "STRONGLY_BULLISH"
    elif bias_score >= 1:
        trend = "BULLISH"
    elif bias_score <= -3:
        trend = "STRONGLY_BEARISH"
    elif bias_score <= -1:
        trend = "BEARISH"
    else:
        trend = "NEUTRAL"

    return {
        "daily_trend": trend,
        "daily_bias_score": bias_score,
        "daily_ema20": round(ema20, 2) if ema20 else None,
        "daily_ema50": round(ema50, 2) if ema50 else None,
        "daily_ema200": round(ema200, 2) if ema200 else None,
        "weekly_high": round(weekly_high, 2),
        "weekly_low": round(weekly_low, 2),
        "daily_atr": round(atr, 2) if atr else None,
        "price_vs_ema20_pct": round(price_vs_ema20_pct, 3) if price_vs_ema20_pct is not None else None,
        "price_vs_ema50_pct": round(price_vs_ema50_pct, 3) if price_vs_ema50_pct is not None else None,
        "higher_tf_available": True,
    }


def compute_vwap_bands(bars: list[OhlcvBar]) -> dict[str, float | None]:
    """VWAP with 1-sigma and 2-sigma volume-weighted standard deviation bands.

    High-volume bars receive more influence on band width — mirroring how
    institutional activity defines the expected intraday price envelope.
    band_compression_pct < 1.0% signals a coiling market; > 2.5% = expansion.
    """
    denom = sum(bar.volume for bar in bars)
    if denom <= 0:
        return {
            "vwap": None, "upper_1": None, "lower_1": None,
            "upper_2": None, "lower_2": None,
            "band_width": None, "band_compression_pct": None,
        }
    typicals = [(bar.high + bar.low + bar.close) / 3.0 for bar in bars]
    vwap = sum(t * bar.volume for t, bar in zip(typicals, bars)) / denom
    variance = sum(bar.volume * (t - vwap) ** 2 for t, bar in zip(typicals, bars)) / denom
    std = variance ** 0.5
    band_width = 4.0 * std  # full envelope: upper_2 to lower_2
    band_compression_pct = round(band_width / vwap * 100, 3) if vwap > 0 else None
    return {
        "vwap": round(vwap, 2),
        "upper_1": round(vwap + std, 2),
        "lower_1": round(vwap - std, 2),
        "upper_2": round(vwap + 2 * std, 2),
        "lower_2": round(vwap - 2 * std, 2),
        "band_width": round(band_width, 2),
        "band_compression_pct": band_compression_pct,
    }


def compute_volume_spike(bars: list[OhlcvBar], window: int = 20) -> dict[str, float | bool]:
    """Volume anomaly: compares the latest bar to the rolling mean of prior N bars.

    ratio >= 1.5 → volume spike (institutional or news-driven surge, often precedes a move).
    ratio < 0.5 → volume drought (watch for range-bound chop or liquidity thin-out).
    """
    if not bars:
        return {"volume_ratio": 1.0, "is_spike": False}
    lookback = bars[-min(window + 1, len(bars)):-1]
    if not lookback:
        return {"volume_ratio": 1.0, "is_spike": False}
    avg_vol = sum(bar.volume for bar in lookback) / len(lookback)
    if avg_vol <= 0:
        return {"volume_ratio": 1.0, "is_spike": False}
    ratio = bars[-1].volume / avg_vol
    return {"volume_ratio": round(ratio, 2), "is_spike": ratio >= 1.5}


def compute_max_pain(quotes: list[OptionsContractQuote]) -> dict[str, float | None]:
    """Strike where total option-buyer notional pain is minimised.

    For each candidate settlement S:
      call_pain = Σ max(0, S - strike) × call_OI
      put_pain  = Σ max(0, strike - S) × put_OI
    max_pain_strike = argmin(call_pain + put_pain).

    Market makers are net-short options so they tend to pin expiry near this level.
    Use as a gravitational reference for where the chain may settle, not a hard gate.
    """
    if not quotes:
        return {"max_pain_strike": None}
    oi_by_strike: dict[int, dict[str, int]] = {}
    for q in quotes:
        k = int(q.strike)
        if k not in oi_by_strike:
            oi_by_strike[k] = {"ce": 0, "pe": 0}
        if q.option_type == OptionType.CALL and q.oi:
            oi_by_strike[k]["ce"] += int(q.oi)
        elif q.option_type == OptionType.PUT and q.oi:
            oi_by_strike[k]["pe"] += int(q.oi)
    strikes = sorted(oi_by_strike)
    if not strikes:
        return {"max_pain_strike": None}
    min_pain = float("inf")
    max_pain_strike = strikes[len(strikes) // 2]
    for candidate in strikes:
        call_pain = sum(max(0, candidate - k) * oi_by_strike[k]["ce"] for k in strikes)
        put_pain = sum(max(0, k - candidate) * oi_by_strike[k]["pe"] for k in strikes)
        total = call_pain + put_pain
        if total < min_pain:
            min_pain = total
            max_pain_strike = candidate
    return {"max_pain_strike": float(max_pain_strike)}


def compute_atm_straddle(spot: float, quotes: list[OptionsContractQuote]) -> dict[str, float | None]:
    """ATM straddle price = CE_ltp + PE_ltp at the nearest-to-spot strike.

    For 0-1 DTE Nifty weekly options the straddle price directly equals the
    market's consensus 1-sigma expected daily range — no scaling needed.
    Short strikes inside this range are inside the market's own forecast,
    dramatically increasing assignment risk on trending days.
    """
    by_strike: dict[int, dict[str, float]] = {}
    for q in quotes:
        k = int(q.strike)
        if k not in by_strike:
            by_strike[k] = {}
        if q.option_type == OptionType.CALL and q.ltp > 0:
            by_strike[k]["ce"] = q.ltp
        elif q.option_type == OptionType.PUT and q.ltp > 0:
            by_strike[k]["pe"] = q.ltp
    atm_strike = None
    atm_dist = float("inf")
    for k, sides in by_strike.items():
        if "ce" in sides and "pe" in sides:
            d = abs(k - spot)
            if d < atm_dist:
                atm_dist = d
                atm_strike = k
    if atm_strike is None:
        return {"atm_strike": None, "atm_straddle_price": None, "expected_move_pts": None}
    sides = by_strike[atm_strike]
    straddle = sides["ce"] + sides["pe"]
    return {
        "atm_strike": float(atm_strike),
        "atm_straddle_price": round(straddle, 2),
        "expected_move_pts": round(straddle, 2),
    }


def compute_vwap_reclaim(bars: list[OhlcvBar], vwap: float) -> dict[str, bool | float]:
    """VWAP reclaim: price dipped below VWAP for ≥2 prior bars, now closed above with volume.

    Signals institutional buyers defending the mean — high-quality bullish continuation
    when paired with positive OI flow.  Returns reclaim=False when VWAP is unknown or bars
    are insufficient.
    """
    if not bars or vwap <= 0 or len(bars) < 3:
        return {"vwap_reclaim": False, "vwap_reclaim_strength": 0.0}
    if bars[-1].close <= vwap:
        return {"vwap_reclaim": False, "vwap_reclaim_strength": 0.0}
    prior = bars[-4:-1] if len(bars) >= 4 else bars[:-1]
    below_count = sum(1 for bar in prior if bar.close < vwap)
    if below_count < min(2, len(prior)):
        return {"vwap_reclaim": False, "vwap_reclaim_strength": 0.0}
    avg_prior_vol = sum(b.volume for b in prior) / max(len(prior), 1)
    vol_ratio = bars[-1].volume / max(avg_prior_vol, 1.0)
    dist_pct = (bars[-1].close - vwap) / max(vwap, 1.0)
    strength = round(min(dist_pct * 100 + max(vol_ratio - 1.0, 0.0) * 0.3, 2.0), 3)
    return {"vwap_reclaim": True, "vwap_reclaim_strength": max(0.0, strength)}


def compute_momentum_persistence(bars: list[OhlcvBar], n: int = 6) -> dict[str, float]:
    """Volume-weighted fraction of last N bars agreeing on direction (green vs red).

    Persistence ≥ 0.70 = institutional trend participation.
    Persistence ≤ 0.35 = choppy, unreliable candle signals.
    """
    if not bars or len(bars) < 2:
        return {"bullish_persistence": 0.5, "bearish_persistence": 0.5}
    recent = bars[-min(n, len(bars)):]
    total_vol = sum(bar.volume for bar in recent)
    if total_vol <= 0:
        return {"bullish_persistence": 0.5, "bearish_persistence": 0.5}
    bull_vol = sum(bar.volume for bar in recent if bar.close >= bar.open)
    bear_vol = sum(bar.volume for bar in recent if bar.close < bar.open)
    return {
        "bullish_persistence": round(bull_vol / total_vol, 3),
        "bearish_persistence": round(bear_vol / total_vol, 3),
    }


def read_market_context() -> dict:
    """Read auxiliary market context (VIX, BankNifty) written by DhanLiveMarketDataProvider.

    Returns an empty dict when the file is absent or unreadable — callers must handle missing keys.
    """
    try:
        if _MARKET_CONTEXT_PATH.exists():
            return json.loads(_MARKET_CONTEXT_PATH.read_text())
    except Exception:
        pass
    return {}
