"""Market regime heuristics for the strategy recommender."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, cast

import numpy as np
import pandas as pd


@dataclass
class MarketState:
    as_of: datetime
    trend: str
    volatility: str
    iv_rank: float
    realized_vol: float
    atr_pct: float
    drift_pct: float
    vix: Optional[float]
    breadth: Optional[float]
    features: Dict[str, Any]


class MarketRegimeAnalyzer:
    """Compute market state from recent feature history and option data."""

    def __init__(
        self,
        lookback: timedelta = timedelta(hours=3),
        min_points: int = 30,
    ) -> None:
        self.lookback = lookback
        self.min_points = min_points

    def analyze_feature_history(
        self,
        df: pd.DataFrame,
        *,
        vix: Optional[float] = None,
        fallback_state: Optional[MarketState] = None,
    ) -> Optional[MarketState]:
        if df.empty:
            return fallback_state
        frame = df.copy()
        if "timestamp" not in frame.columns:
            return fallback_state
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame = frame.dropna(subset=["timestamp"])  # type: ignore[arg-type]
        frame = frame.sort_values("timestamp")
        cutoff = frame["timestamp"].max() - self.lookback
        frame = frame[frame["timestamp"] >= cutoff]
        frame = frame.dropna(subset=["spot"])  # type: ignore[arg-type]
        if len(frame) < max(self.min_points, 5):
            return fallback_state

        # Derived metrics
        prices = cast(pd.Series, frame["spot"].astype(float))
        returns = prices.pct_change().dropna()
        realized_vol = float(np.sqrt(returns.pow(2).mean()) * np.sqrt(252)) if not returns.empty else 0.0
        atr_pct = self._compute_atr_pct(frame)
        drift_pct = float((prices.iloc[-1] / prices.iloc[0]) - 1.0)
        iv_rank = self._approx_iv_rank(frame)
        breadth = self._estimate_breadth(frame)

        trend = self._classify_trend(prices)
        volatility = self._classify_volatility(realized_vol, atr_pct, vix)

        state = MarketState(
            as_of=frame["timestamp"].iloc[-1].to_pydatetime(),
            trend=trend,
            volatility=volatility,
            iv_rank=iv_rank,
            realized_vol=realized_vol,
            atr_pct=atr_pct,
            drift_pct=drift_pct,
            vix=vix,
            breadth=breadth,
            features={
                "spot": float(prices.iloc[-1]),
                "lookback_points": len(frame),
                "returns_std": float(returns.std()) if not returns.empty else 0.0,
            },
        )
        return state

    @staticmethod
    def _compute_atr_pct(frame: pd.DataFrame) -> float:
        high = frame.get("high")
        low = frame.get("low")
        close = frame.get("spot")
        if high is None or low is None or close is None:
            return 0.0
        try:
            atr = (high.astype(float) - low.astype(float)).ewm(span=14, adjust=False).mean().iloc[-1]
            spot = float(close.iloc[-1])
            if spot <= 0:
                return 0.0
            return float(atr / spot)
        except Exception:
            return 0.0

    @staticmethod
    def _approx_iv_rank(frame: pd.DataFrame) -> float:
        implied = frame.get("iv")
        if implied is None or implied.isna().all():
            return 0.5
        series = implied.dropna().astype(float)
        if len(series) < 5:
            return 0.5
        recent = series.iloc[-1]
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return 0.5
        return float((recent - min_val) / (max_val - min_val))

    @staticmethod
    def _estimate_breadth(frame: pd.DataFrame) -> Optional[float]:
        adv = frame.get("advancing")
        dec = frame.get("declining")
        if adv is None or dec is None:
            return None
        try:
            adv_latest = float(adv.dropna().iloc[-1])
            dec_latest = float(dec.dropna().iloc[-1])
            total = adv_latest + dec_latest
            if total == 0:
                return None
            return adv_latest / total
        except Exception:
            return None

    @staticmethod
    def _classify_trend(prices: pd.Series) -> str:
        ema_fast = prices.ewm(span=10, adjust=False).mean()
        ema_slow = prices.ewm(span=30, adjust=False).mean()
        slope = (ema_fast.iloc[-1] - ema_fast.iloc[0]) / ema_fast.iloc[0]
        spread = float(ema_fast.iloc[-1] - ema_slow.iloc[-1])
        pct_spread = spread / prices.iloc[-1]
        if slope > 0.01 and pct_spread > 0.002:
            return "trending_up"
        if slope < -0.01 and pct_spread < -0.002:
            return "trending_down"
        return "range_bound"

    @staticmethod
    def _classify_volatility(realized_vol: float, atr_pct: float, vix: Optional[float]) -> str:
        vol_metric = realized_vol
        if vix is not None:
            vol_metric = max(vol_metric, vix / 100.0)
        vol_metric = max(vol_metric, atr_pct * np.sqrt(252))
        if vol_metric < 0.12:
            return "low_vol"
        if vol_metric < 0.22:
            return "medium_vol"
        return "high_vol"


def summarize_market_state(state: Optional[MarketState]) -> Dict[str, Any]:
    if state is None:
        return {}
    return {
        "trend": state.trend,
        "volatility": state.volatility,
        "iv_rank": round(state.iv_rank, 3),
        "realized_vol": round(state.realized_vol, 3),
        "atr_pct": round(state.atr_pct, 3),
        "drift_pct": round(state.drift_pct, 3),
        "vix": state.vix,
        "breadth": state.breadth,
        "as_of": state.as_of.isoformat(),
    }
