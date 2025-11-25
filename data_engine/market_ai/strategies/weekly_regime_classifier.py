"""
Simple weekly regime classifier for NIFTY weekly option strategies.

Classification is intentionally rule-based and lightweight so it can run both
in backtests and live without extra data dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict, Optional


RegimeLiteral = Literal["SIDEWAYS", "BULL", "BEAR", "NO_TRADE"]


class RegimeDecision(TypedDict):
    regime: RegimeLiteral
    score: float
    reason: str


@dataclass
class RegimeConfig:
    sideways_trend_thresh: float = 2.5   # % absolute 20D move to still call it flat
    small_range_thresh: float = 1.0      # % intraday range to call “small”
    medium_range_thresh: float = 1.5     # % intraday range still acceptable
    trend_strong_thresh: float = 4.5     # % absolute 20D move considered strong
    max_gap_for_trade: float = 1.2       # % absolute gap allowed
    min_score: float = 0.6


def classify_week(
    gap_pct: float,
    intraday_range_pct: float,
    trend_20: float,
    cfg: Optional[RegimeConfig] = None,
) -> RegimeDecision:
    """
    Classify the week into SIDEWAYS / BULL / BEAR / NO_TRADE.
    """
    c = cfg or RegimeConfig()
    gap = gap_pct or 0.0
    rng = intraday_range_pct or 0.0
    tr20 = trend_20 or 0.0

    strong_trend = abs(tr20) > c.trend_strong_thresh
    mild_trend = abs(tr20) > c.sideways_trend_thresh
    small_range = rng <= c.small_range_thresh
    medium_range = rng <= c.medium_range_thresh
    small_gap = abs(gap) <= c.max_gap_for_trade

    # SIDEWAYS
    if (abs(tr20) <= c.sideways_trend_thresh) and small_range and small_gap:
        return {"regime": "SIDEWAYS", "score": 0.7, "reason": "Flat 20D trend, small range, small gap"}

    # BULL
    if tr20 > c.sideways_trend_thresh and gap >= -0.5 and medium_range and not strong_trend:
        return {"regime": "BULL", "score": 0.7, "reason": "Up 20D trend, no big downside gap"}

    # BEAR
    if tr20 < -c.sideways_trend_thresh and gap <= 0.5 and medium_range and not strong_trend:
        return {"regime": "BEAR", "score": 0.7, "reason": "Down 20D trend, no big upside gap"}

    # Anything else → NO_TRADE
    return {"regime": "NO_TRADE", "score": 0.0, "reason": "Messy or high volatility week"}

