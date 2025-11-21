from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .models import MarketSnapshot
from .orb_detector import ORBLevels


@dataclass
class SRLevels:
    major_supports: List[float]
    major_resistances: List[float]


def _swing_points(candles: List[dict], window: int = 3) -> tuple[List[float], List[float]]:
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


def compute_sr_levels(market: MarketSnapshot, orb_levels: ORBLevels | None) -> SRLevels:
    supports: List[float] = []
    resistances: List[float] = []
    candles = market.candles_5m or []

    if orb_levels:
        supports.append(orb_levels.orb_low)
        resistances.append(orb_levels.orb_high)

    if market.yesterday_low:
        supports.append(market.yesterday_low)
    if market.yesterday_high:
        resistances.append(market.yesterday_high)

    swing_supports, swing_resistances = _swing_points(candles[-30:] if len(candles) > 30 else candles)
    supports.extend(swing_supports)
    resistances.extend(swing_resistances)

    # keep unique sorted levels
    supports = sorted({round(level, 2) for level in supports if level})
    resistances = sorted({round(level, 2) for level in resistances if level})
    return SRLevels(major_supports=supports, major_resistances=resistances)
