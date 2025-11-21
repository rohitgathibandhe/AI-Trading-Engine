from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from market_ai.strategies import StrategyType, TrendSide


@dataclass
class PolicyDecision:
    should_trade: bool
    regime: TrendSide
    confidence: float
    strategy: StrategyType
    size_multiplier: float
    notes: str

    def as_dict(self) -> Dict[str, str]:
        return {
            "should_trade": str(self.should_trade),
            "regime": self.regime.name,
            "confidence": f"{self.confidence:.2f}",
            "strategy": self.strategy.name,
            "size_multiplier": f"{self.size_multiplier:.2f}",
            "notes": self.notes,
        }


class PolicyEngine:
    """Maps regime probabilities + learning feedback into an action."""

    def __init__(self, learning_manager=None):
        self.learning = learning_manager

    def decide(self, probs: Dict[str, float]) -> PolicyDecision:
        prob_bull = probs.get("prob_bull", 0.0)
        prob_bear = probs.get("prob_bear", 0.0)
        prob_side = probs.get("prob_sideways", 0.0)
        best = max(prob_bull, prob_bear, prob_side)
        if best < 0.5:
            return PolicyDecision(False, TrendSide.NO_TRADE, best, StrategyType.NONE, 0.0, "Low confidence")

        if best == prob_bull:
            regime = TrendSide.BULL
            strategy = StrategyType.BULL_PUT_SPREAD
            base_conf = prob_bull
        elif best == prob_bear:
            regime = TrendSide.BEAR
            strategy = StrategyType.BEAR_CALL_SPREAD
            base_conf = prob_bear
        else:
            regime = TrendSide.SIDEWAYS
            strategy = StrategyType.STRANGLE
            base_conf = prob_side

        size_mult = 1.0
        if self.learning:
            if regime == TrendSide.BEAR and self.learning.should_downweight_bear():
                return PolicyDecision(False, TrendSide.NO_TRADE, base_conf, StrategyType.NONE, 0.0, "Bear disabled by learning")
            size_mult = self.learning.size_multiplier(regime.name, strategy.name, base_conf)
        else:
            if base_conf < 0.6:
                size_mult = 0.5
            elif base_conf >= 0.75:
                size_mult = 1.25

        return PolicyDecision(True, regime, base_conf, strategy, size_mult, "Policy decision")
