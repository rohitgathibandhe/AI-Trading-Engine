from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from math import fabs


@dataclass
class RegimeProbabilities:
    prob_bull: float
    prob_bear: float
    prob_sideways: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "prob_bull": self.prob_bull,
            "prob_bear": self.prob_bear,
            "prob_sideways": self.prob_sideways,
        }


@dataclass
class RegimeInputs:
    gap_pct: float = 0.0
    trend_strength: float = 0.0
    orb_break_dir: int = 0   # +1 bull, -1 bear, 0 none
    bullish_vwap: float = 0.0
    bearish_vwap: float = 0.0
    within_sr_band: float = 0.0  # 0-1 measure of being within support/resistance band


class RegimeScorer:
    """Weighted, interpretable regime scorer."""

    def __init__(self):
        # Default weights; can be tuned later
        self.weights = {
            "gap": 0.3,
            "trend": 0.3,
            "orb": 0.2,
            "vwap": 0.2,
        }

    def score(self, features) -> RegimeProbabilities:
        f = features
        # Normalize signals 0-1
        gap_score = min(1.0, max(0.0, f.gap_pct / 0.7 if f.gap_pct >= 0 else -f.gap_pct / 0.7))
        gap_dir = 1.0 if f.gap_pct > 0 else -1.0 if f.gap_pct < 0 else 0.0

        bull_trend = min(1.0, max(0.0, f.trend_strength if gap_dir >= 0 else 0.0))
        bear_trend = min(1.0, max(0.0, f.trend_strength if gap_dir < 0 else 0.0))

        bull_orb = 1.0 if f.orb_break_dir > 0 else 0.0
        bear_orb = 1.0 if f.orb_break_dir < 0 else 0.0
        no_orb = 1.0 if f.orb_break_dir == 0 else 0.0

        bull_vwap = f.bullish_vwap
        bear_vwap = f.bearish_vwap

        # Bull score
        bull = (
            self.weights["gap"] * max(0.0, gap_dir) * gap_score
            + self.weights["trend"] * bull_trend
            + self.weights["orb"] * bull_orb
            + self.weights["vwap"] * bull_vwap
        )
        # Bear score
        bear = (
            self.weights["gap"] * max(0.0, -gap_dir) * gap_score
            + self.weights["trend"] * bear_trend
            + self.weights["orb"] * bear_orb
            + self.weights["vwap"] * bear_vwap
        )
        # Sideways score
        low_gap = 1.0 - min(1.0, fabs(f.gap_pct) / 0.7)
        low_trend = 1.0 - min(1.0, f.trend_strength)
        sideways = (
            0.25 * low_gap
            + 0.25 * low_trend
            + 0.25 * no_orb
            + 0.25 * f.within_sr_band
        )

        total = bull + bear + sideways
        if total <= 1e-6:
            return RegimeProbabilities(1 / 3, 1 / 3, 1 / 3)
        return RegimeProbabilities(bull / total, bear / total, sideways / total)
