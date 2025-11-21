from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
from math import fabs


@dataclass
class FeatureSet:
    gap_pct: float
    prev_day_range_pct: float
    orb_range_pct: float
    orb_break_dir: float
    vwap_distance: float
    vix_value: float
    vix_percentile: float
    atm_iv_percentile: float
    trend_strength: float
    mean_reversion_likelihood: float
    within_sr_band: float
    bullish_vwap: float
    bearish_vwap: float

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


class FeatureExtractor:
    """
    Lightweight, hand-crafted feature builder. Uses only locally available snapshot fields
    so it can run both live and in backtests without extra network calls.
    """

    def __init__(self):
        pass

    def compute(self, market, vwap: float, pivot: float) -> FeatureSet:
        # Gap metrics
        gap_pct = 0.0
        if market.yesterday_close:
            gap_pct = (market.spot - market.yesterday_close) / market.yesterday_close * 100.0

        prev_range_pct = 0.0
        if market.yesterday_close:
            prev_range_pct = (market.yesterday_high - market.yesterday_low) / market.yesterday_close * 100.0

        # ORB approximation: first candle range/direction
        orb_range_pct = 0.0
        orb_break_dir = 0.0
        if market.candles_5m:
            c1 = market.candles_5m[0]
            rng = float(c1.get("high", 0.0) - c1.get("low", 0.0))
            if market.yesterday_close:
                orb_range_pct = rng / market.yesterday_close * 100.0
            body = float(c1.get("close", 0.0) - c1.get("open", 0.0))
            if body > 0 and c1.get("close", 0.0) >= c1.get("high", 0.0) - 1e-6:
                orb_break_dir = 1.0
            elif body < 0 and c1.get("close", 0.0) <= c1.get("low", 0.0) + 1e-6:
                orb_break_dir = -1.0

        # VWAP / pivot positioning
        vwap_distance = 0.0
        if vwap:
            vwap_distance = (market.spot - vwap) / vwap
        bullish_vwap = 1.0 if vwap_distance > 0 else 0.0
        bearish_vwap = 1.0 if vwap_distance < 0 else 0.0

        # Trend proxy using day range compression
        trend_strength = min(1.0, max(0.0, fabs(vwap_distance) * 2 + fabs(gap_pct) / 1.0))
        mean_reversion_likelihood = 1.0 - min(1.0, trend_strength + orb_range_pct / 2.0)

        vix_value = float(getattr(market, "india_vix", 0.0) or 0.0)
        vix_percentile = float(getattr(market, "vix_percentile", 0.5) or 0.5)
        atm_iv_percentile = float(getattr(market, "atm_iv_percentile", 0.5) or 0.5)

        # Within SR band proxy: use distance to pivot
        within_sr_band = 1.0 if fabs(market.spot - pivot) <= max(1.0, market.spot * 0.002) else 0.0

        return FeatureSet(
            gap_pct=gap_pct,
            prev_day_range_pct=prev_range_pct,
            orb_range_pct=orb_range_pct,
            orb_break_dir=orb_break_dir,
            vwap_distance=vwap_distance,
            vix_value=vix_value,
            vix_percentile=vix_percentile,
            atm_iv_percentile=atm_iv_percentile,
            trend_strength=trend_strength,
            mean_reversion_likelihood=mean_reversion_likelihood,
            within_sr_band=within_sr_band,
            bullish_vwap=bullish_vwap,
            bearish_vwap=bearish_vwap,
        )
