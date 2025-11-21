from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..strategies.models import MarketSnapshot


@dataclass
class SRLevels:
    supports: List[float]
    resistances: List[float]


class SREngine:
    def __init__(self, lookback_candles: int = 30):
        self.lookback_candles = lookback_candles

    def _swing_points(self, candles: List[dict], window: int = 2):
        supports: List[float] = []
        resistances: List[float] = []
        for i in range(window, len(candles) - window):
            lows = [float(candles[j].get("low") or 0.0) for j in range(i - window, i + window + 1)]
            highs = [float(candles[j].get("high") or 0.0) for j in range(i - window, i + window + 1)]
            center_low = float(candles[i].get("low") or 0.0)
            center_high = float(candles[i].get("high") or 0.0)
            if center_low == min(lows):
                supports.append(center_low)
            if center_high == max(highs):
                resistances.append(center_high)
        return supports, resistances

    def compute(self, market: MarketSnapshot) -> SRLevels:
        candles = (market.candles_5m or [])[-self.lookback_candles :]
        supports = [market.yesterday_low] if market.yesterday_low else []
        resistances = [market.yesterday_high] if market.yesterday_high else []
        swing_supports, swing_resistances = self._swing_points(candles)
        supports.extend(swing_supports)
        resistances.extend(swing_resistances)
        supports = sorted({round(level, 2) for level in supports if level})
        resistances = sorted({round(level, 2) for level in resistances if level})
        return SRLevels(supports=supports, resistances=resistances)
