from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from ..strategies.models import MarketSnapshot, TrendSide


@dataclass
class ORBConfig:
    timeframe_minutes: int = 15
    buffer_points: float = 5.0
    min_body: float = 40.0
    min_range: float = 60.0
    min_volume_factor: float = 1.2


@dataclass
class ORBLevels:
    orb_high: float
    orb_low: float
    completed_at: datetime


@dataclass
class ORBBreakoutSignal:
    active: bool
    direction: TrendSide
    breakout_price: float
    candle_time: datetime
    reason: str


class ORBEngine:
    def __init__(self, config: Optional[ORBConfig] = None):
        self.config = config or ORBConfig()

    def compute_levels(self, market: MarketSnapshot) -> Optional[ORBLevels]:
        candles = market.candles_5m or []
        needed = self.config.timeframe_minutes // 5
        if len(candles) < needed:
            return None
        orb_slice = candles[:needed]
        highs = [float(c.get("high") or 0.0) for c in orb_slice]
        lows = [float(c.get("low") or 0.0) for c in orb_slice]
        if not highs or not lows:
            return None
        return ORBLevels(
            orb_high=max(highs),
            orb_low=min(lows),
            completed_at=orb_slice[-1].get("timestamp", market.now),
        )

    def detect_breakout(
        self,
        market: MarketSnapshot,
        orb_levels: Optional[ORBLevels],
    ) -> ORBBreakoutSignal:
        latest = market.candles_5m[-1] if (market.candles_5m) else {}
        candle_time = latest.get("timestamp", market.now)
        default = ORBBreakoutSignal(False, TrendSide.SIDEWAYS, market.spot, candle_time, "")
        if not orb_levels:
            return default
        if candle_time <= orb_levels.completed_at:
            return default
        open_price = float(latest.get("open") or market.spot)
        close_price = float(latest.get("close") or market.spot)
        high_price = float(latest.get("high") or max(open_price, close_price))
        low_price = float(latest.get("low") or min(open_price, close_price))
        volume = float(latest.get("volume") or 0.0)
        body = abs(close_price - open_price)
        rng = high_price - low_price
        avg_volume = self._avg_volume(market.candles_5m[:-1])

        if (
            close_price > orb_levels.orb_high + self.config.buffer_points
            and self._validate_candle(body, rng, volume, avg_volume)
        ):
            return ORBBreakoutSignal(True, TrendSide.BULL, close_price, candle_time, "ORB high breakout")
        if (
            close_price < orb_levels.orb_low - self.config.buffer_points
            and self._validate_candle(body, rng, volume, avg_volume)
        ):
            return ORBBreakoutSignal(True, TrendSide.BEAR, close_price, candle_time, "ORB low breakout")
        return default

    def _avg_volume(self, candles):
        vols = [float(c.get("volume") or 0.0) for c in candles if c.get("volume") is not None]
        return sum(vols) / len(vols) if vols else 1.0

    def _validate_candle(self, body, rng, volume, avg_volume):
        return (
            body >= self.config.min_body
            and rng >= self.config.min_range
            and volume >= self.config.min_volume_factor * avg_volume
        )
