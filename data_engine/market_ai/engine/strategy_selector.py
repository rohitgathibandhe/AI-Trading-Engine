from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, List

from ..strategies.models import OptionType, TrendSide


@dataclass
class StrategyPlan:
    regime: str
    strategy_name: str
    notes: str
    legs_to_open: List[Dict[str, float]]
    deferred_side: OptionType | None = None
    defer_until: time | None = None


class StrategySelector:
    def __init__(self):
        self.regime_map = {
            "RANGE": "SHORT_STRANGLE",
            "BULL": "SKEWED_PUT_STRANGLE",
            "BEAR": "SKEWED_CALL_STRANGLE",
            "MIXED": "STAGGERED_STRANGLE",
        }

    def decide(self, scores: Dict[str, float]) -> StrategyPlan:
        regime = self._select_regime(scores)
        strategy = self.regime_map.get(regime, "NO_TRADE")
        notes = f"Regime={regime} bull={scores['bull_score']} bear={scores['bear_score']} range={scores['range_score']}"
        return StrategyPlan(regime, strategy, notes, legs_to_open=[])

    def apply_gap_logic(
        self,
        plan: StrategyPlan,
        scores: Dict[str, float],
        sr_levels,
    ) -> StrategyPlan:
        if scores["big_gap_score"] < 0.6:
            return plan
        if scores["bull_score"] > scores["bear_score"]:
            plan.deferred_side = OptionType.CALL
            plan.notes += " | Gap-up: sell PE first"
        else:
            plan.deferred_side = OptionType.PUT
            plan.notes += " | Gap-down: sell CE first"
        plan.defer_until = time(11, 0)
        return plan

    def _select_regime(self, scores: Dict[str, float]) -> str:
        if scores["range_score"] >= 0.6:
            return "RANGE"
        if scores["bull_score"] >= 0.6 and scores["bull_score"] > scores["bear_score"]:
            return "BULL"
        if scores["bear_score"] >= 0.6 and scores["bear_score"] > scores["bull_score"]:
            return "BEAR"
        return "MIXED"
