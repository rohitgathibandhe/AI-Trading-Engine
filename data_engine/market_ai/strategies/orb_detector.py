from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

from .models import MarketSnapshot, TrendSide


@dataclass
class ORBConfig:
    enabled: bool = True
    timeframe_minutes: int = 15  # allowed: 15 or 30
    min_body_points: float = 40.0
    min_range_points: float = 60.0
    breakout_buffer_points: float = 5.0
    min_volume_factor: float = 1.2
    reentry_buffer_points: float = 10.0


@dataclass
class ORBLevels:
    orb_high: float
    orb_low: float
    orb_mid: float
    timeframe_minutes: int
    completed_at: datetime


@dataclass
class ORBBreakoutSignal:
    active: bool
    direction: TrendSide
    breakout_price: float
    breakout_candle_time: datetime
    reason: str


def _as_datetime(value) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except Exception:
            return None
    return None


def _avg_volume(candles: List[dict]) -> float:
    vols = [float(c.get("volume") or 0.0) for c in candles if c.get("volume") is not None]
    return sum(vols) / len(vols) if vols else 0.0


def compute_orb_levels(
    candles_5m: List[dict],
    open_time: datetime,
    config: ORBConfig,
) -> Optional[ORBLevels]:
    if not config.enabled:
        return None
    frame_minutes = 15 if config.timeframe_minutes not in (15, 30) else config.timeframe_minutes
    required_candles = frame_minutes // 5
    if not candles_5m or len(candles_5m) < required_candles:
        return None
    # Assume candles sorted oldest -> newest
    orb_candles = candles_5m[:required_candles]
    highs = [float(c.get("high") or 0.0) for c in orb_candles]
    lows = [float(c.get("low") or 0.0) for c in orb_candles]
    if not highs or not lows:
        return None
    orb_high = max(highs)
    orb_low = min(lows)
    if orb_high - orb_low < config.min_range_points:
        return None
    completed_at = open_time + timedelta(minutes=frame_minutes)
    return ORBLevels(
        orb_high=orb_high,
        orb_low=orb_low,
        orb_mid=(orb_high + orb_low) / 2.0,
        timeframe_minutes=frame_minutes,
        completed_at=completed_at,
    )


def detect_orb_breakout(
    market: MarketSnapshot,
    orb_levels: ORBLevels,
    config: ORBConfig,
) -> ORBBreakoutSignal:
    default_signal = ORBBreakoutSignal(False, TrendSide.SIDEWAYS, 0.0, market.now, "")
    if not config.enabled or not orb_levels:
        return default_signal
    candles = market.candles_5m or []
    if not candles:
        return default_signal
    latest = candles[-1]
    candle_time = _as_datetime(latest.get("timestamp") or latest.get("time") or market.now.isoformat())
    if candle_time and candle_time <= orb_levels.completed_at:
        return default_signal

    open_price = float(latest.get("open") or market.spot)
    close_price = float(latest.get("close") or market.spot)
    high_price = float(latest.get("high") or max(open_price, close_price))
    low_price = float(latest.get("low") or min(open_price, close_price))
    volume = float(latest.get("volume") or 0.0)
    body = abs(close_price - open_price)
    rng = high_price - low_price
    avg_vol = _avg_volume(candles[:-1]) or volume

    if (
        close_price > orb_levels.orb_high + config.breakout_buffer_points
        and body >= config.min_body_points
        and rng >= config.min_range_points
        and volume >= config.min_volume_factor * avg_vol
    ):
        return ORBBreakoutSignal(
            True,
            TrendSide.BULL,
            close_price,
            candle_time or market.now,
            f"{orb_levels.timeframe_minutes}m ORB high broken with strong candle",
        )

    if (
        close_price < orb_levels.orb_low - config.breakout_buffer_points
        and body >= config.min_body_points
        and rng >= config.min_range_points
        and volume >= config.min_volume_factor * avg_vol
    ):
        return ORBBreakoutSignal(
            True,
            TrendSide.BEAR,
            close_price,
            candle_time or market.now,
            f"{orb_levels.timeframe_minutes}m ORB low broken with strong candle",
        )

    return default_signal
