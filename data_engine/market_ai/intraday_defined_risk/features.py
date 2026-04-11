from __future__ import annotations

from datetime import datetime, time
from statistics import fmean
from typing import Iterable

from .data_models import OhlcvBar, OhlcvSeries, ORLevels, OptionType, OptionsContractQuote


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
    if rv30_pct >= 0.22:
        return False
    recent = bars[-12:]
    inside_count = sum(1 for bar in recent if bar.high <= opening_range.high and bar.low >= opening_range.low)
    if inside_count / len(recent) < 0.80:
        return False
    if not oscillates_around_vwap(recent, vwap, required_bars=min(8, len(recent))):
        return False
    recent_range_pct = price_change_pct(min(bar.low for bar in recent), max(bar.high for bar in recent))
    if recent_range_pct >= 0.45:
        return False
    range_mid = (opening_range.high + opening_range.low) / 2.0
    return abs(bars[-1].close - range_mid) / max(bars[-1].close, 1.0) <= 0.0018


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

    return {
        "bullish_flow_score": bullish_score,
        "bearish_flow_score": bearish_score,
        "put_support_strike": put_wall_strike,
        "call_resistance_strike": call_wall_strike,
        "put_support_oi_change": put_wall_change,
        "call_resistance_oi_change": call_wall_change,
        "smart_money_bias": smart_money_bias,
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
