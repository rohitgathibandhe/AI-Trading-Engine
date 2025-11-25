from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict, Optional, Dict, Any

from market_ai.strategies.weekly_regime_classifier import RegimeConfig, classify_week


StrategyLiteral = Literal["BULL_PUT_SPREAD", "BEAR_CALL_SPREAD", "NO_TRADE"]


class RouterDecision(TypedDict):
    regime: Literal["SIDEWAYS", "BULL", "BEAR", "NO_TRADE"]
    strategy: StrategyLiteral
    score: float
    reason: str


@dataclass
class RouterConfig:
    regime: RegimeConfig = RegimeConfig()
    min_score: float = 0.6


def decide_weekly_strategy(features: Dict[str, Any], cfg: Optional[RouterConfig] = None) -> RouterDecision:
    """
    Route to weekly spread strategies based on regime.

    Expected features keys:
        gap_pct, spot_intraday_range_pct, spot_trend_20
    """
    c = cfg or RouterConfig()
    gap = float(features.get("gap_pct") or 0.0)
    rng = float(features.get("spot_intraday_range_pct") or 0.0)
    tr20 = float(features.get("spot_trend_20") or 0.0)
    dec = classify_week(gap, rng, tr20, c.regime)
    strategy: StrategyLiteral = "NO_TRADE"
    if dec["score"] >= c.min_score:
        if dec["regime"] == "BULL":
            strategy = "BULL_PUT_SPREAD"
        elif dec["regime"] == "BEAR":
            strategy = "BEAR_CALL_SPREAD"
        else:
            strategy = "NO_TRADE"
    return {
        "regime": dec["regime"],
        "strategy": strategy,
        "score": dec["score"],
        "reason": dec["reason"],
    }
