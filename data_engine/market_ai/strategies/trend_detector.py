from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .models import MarketSnapshot, TrendContext, TrendSide

GAP_THRESHOLD = 0.5
CANDLE_STRONG_BODY = 0.6
VOLUME_SPIKE_MULTIPLIER = 1.5
CONFIDENCE_THRESHOLD = 2


def _candle_body_ratio(candle: Dict[str, float]) -> float:
    high = candle.get("high", 0.0)
    low = candle.get("low", 0.0)
    open_ = candle.get("open", 0.0)
    close = candle.get("close", 0.0)
    rng = max(high - low, 1e-6)
    body = abs(close - open_)
    return body / rng


def _is_green(c: Dict[str, float]) -> bool:
    return c.get("close", 0.0) >= c.get("open", 0.0)


def detect_trend_from_open(market: MarketSnapshot, avg_5m_volume: float, vwap: float = 0.0, pivot: float = 0.0) -> TrendContext:
    candles = market.candles_5m or []
    bull = 0
    bear = 0

    gap_pct = ((market.spot - market.yesterday_close) / market.yesterday_close) * 100 if market.yesterday_close else 0
    if gap_pct >= GAP_THRESHOLD:
        bull += 2
    elif gap_pct <= -GAP_THRESHOLD:
        bear += 2

    if market.spot > market.yesterday_high:
        bull += 1
    elif market.spot < market.yesterday_low:
        bear += 1

    vol_spike = False
    if candles:
        c1 = candles[0]
        if c1.get("volume") and avg_5m_volume:
            if c1["volume"] >= avg_5m_volume * VOLUME_SPIKE_MULTIPLIER:
                vol_spike = True
        body_ratio = _candle_body_ratio(c1)
        if body_ratio >= CANDLE_STRONG_BODY:
            bull += 1 if _is_green(c1) else 0
            bear += 1 if not _is_green(c1) else 0
        if vwap and pivot:
            if c1.get("close", 0) > vwap and c1.get("close", 0) > pivot:
                bull += 1
            elif c1.get("close", 0) < vwap and c1.get("close", 0) < pivot:
                bear += 1
        if vol_spike:
            bull += 1 if _is_green(c1) else 0
            bear += 1 if not _is_green(c1) else 0

    if len(candles) >= 2:
        c2 = candles[1]
        higher_high = c2.get("high", 0) > candles[0].get("high", 0)
        higher_low = c2.get("low", 0) > candles[0].get("low", 0)
        lower_high = c2.get("high", 0) < candles[0].get("high", 0)
        lower_low = c2.get("low", 0) < candles[0].get("low", 0)
        if higher_high and higher_low and _is_green(c2):
            bull += 2
        if lower_high and lower_low and not _is_green(c2):
            bear += 2
        if pivot and abs(c2.get("close", 0) - pivot) <= (abs(market.yesterday_high - market.yesterday_low) * 0.05):
            bull -= 1
            bear -= 1

    bull = max(bull, 0)
    bear = max(bear, 0)
    confidence = abs(bull - bear)

    if bull >= 3 and (bull - bear) >= CONFIDENCE_THRESHOLD:
        side = TrendSide.BULL
    elif bear >= 3 and (bear - bull) >= CONFIDENCE_THRESHOLD:
        side = TrendSide.BEAR
    else:
        side = TrendSide.SIDEWAYS

    return TrendContext(trend_side=side, bull_score=bull, bear_score=bear, confidence=confidence)
