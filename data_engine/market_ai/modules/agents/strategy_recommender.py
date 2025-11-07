"""Environment-aware strategy recommendation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

from market_ai.modules.analytics import MarketState
from market_ai.modules.strategies.strategy_selector import StrategyScore, StrategySelector, select_preferred_strategy


@dataclass
class StrategyCandidate:
    name: str
    score: float
    confidence: float
    rationale: str
    sizing_hint: float = 1.0


class StrategyRecommender:
    """Blend heuristic rules with selector model outputs."""

    def __init__(
        self,
        selector_model_path: Optional[Path] = None,
        feature_log_path: Optional[Path] = None,
        bias: Optional[Dict[str, float]] = None,
        whitelist: Optional[Iterable[str]] = None,
    ) -> None:
        self.selector_model_path = selector_model_path
        self.feature_log_path = feature_log_path
        self._selector: Optional[StrategySelector] = None
        if selector_model_path and selector_model_path.exists():
            self._selector = StrategySelector(selector_model_path)
        self._bias: Dict[str, float] = {}
        for name, value in (bias or {}).items():
            try:
                self._bias[str(name)] = float(value)
            except Exception:
                continue
        self._whitelist: Set[str] = {str(name) for name in (whitelist or []) if name}

    def recommend(self, state: MarketState) -> List[StrategyCandidate]:
        heuristics = self._heuristic_scores(state)
        blended: Dict[str, StrategyCandidate] = {c.name: c for c in heuristics}

        selector_score: Optional[StrategyScore] = None
        if self._selector and self.feature_log_path:
            selector_score = select_preferred_strategy(self.selector_model_path, self.feature_log_path) if self.selector_model_path else None

        if selector_score:
            bonus = selector_score.expected_credit
            existing = blended.get(selector_score.name)
            rationale = f"Selector expected_credit={selector_score.expected_credit:.2f} confidence={selector_score.confidence:.2f}"
            if existing:
                combined_score = 0.6 * existing.score + 0.4 * bonus
                combined_conf = max(existing.confidence, selector_score.confidence)
                blended[selector_score.name] = StrategyCandidate(
                    name=selector_score.name,
                    score=combined_score,
                    confidence=combined_conf,
                    rationale=f"{existing.rationale}; {rationale}",
                    sizing_hint=existing.sizing_hint,
                )
            else:
                blended[selector_score.name] = StrategyCandidate(
                    name=selector_score.name,
                    score=bonus,
                    confidence=selector_score.confidence,
                    rationale=rationale,
                    sizing_hint=1.0,
                )

        if self._whitelist:
            for name in self._whitelist:
                if name not in blended:
                    blended[name] = StrategyCandidate(
                        name=name,
                        score=self._bias.get(name, 0.0),
                        confidence=0.4,
                        rationale="Whitelist override",
                        sizing_hint=1.0,
                    )

        if self._bias:
            for name, boost in self._bias.items():
                if name not in blended:
                    continue
                if boost == 0:
                    continue
                existing = blended[name]
                blended[name] = StrategyCandidate(
                    name=name,
                    score=existing.score + boost,
                    confidence=existing.confidence,
                    rationale=f"{existing.rationale}; bias+{boost:.2f}",
                    sizing_hint=existing.sizing_hint,
                )

        ranked = sorted(blended.values(), key=lambda c: c.score, reverse=True)
        return ranked

    @staticmethod
    def _heuristic_scores(state: MarketState) -> List[StrategyCandidate]:
        scores: Dict[str, StrategyCandidate] = {}

        def add(name: str, score: float, confidence: float, rationale: str, sizing: float = 1.0) -> None:
            prev = scores.get(name)
            if prev and prev.score >= score:
                return
            scores[name] = StrategyCandidate(name, score, confidence, rationale, sizing)

        trend = state.trend
        vol = state.volatility
        iv_rank = state.iv_rank
        drift = state.drift_pct

        if trend == "trending_up":
            add("credit_spread", 0.75, 0.6, "Trend up; sell put spreads", sizing=1.1)
            if vol != "low_vol":
                add("hedged_strangle", 0.65, 0.5, "Uptrend but vol medium/high; hedge wings")
        elif trend == "trending_down":
            add("monthly_strangle_with_weekly_hedge", 0.7, 0.6, "Downtrend; prefer wider strangles with hedge")
            add("iron_condor", 0.6, 0.5, "Down move; use balanced condor")
        else:
            add("iron_condor", 0.72, 0.55, "Range-bound; condor benefits")
            add("hedged_strangle", 0.65, 0.5, "Range-bound; hedge for surprise moves")

        if vol == "high_vol":
            add("monthly_strangle_with_weekly_hedge", 0.8, 0.65, "High vol; sell strangle with hedges", sizing=min(1.0, 0.9))
        elif vol == "low_vol":
            add("credit_spread", 0.7, 0.55, "Low vol; directional spreads")
            add("iron_condor", 0.6, 0.45, "Low vol; condor but size smaller")

        if iv_rank > 0.7:
            add("monthly_strangle_with_weekly_hedge", 0.82, 0.65, "IV rank elevated; collect premium")
        elif iv_rank < 0.3:
            add("credit_spread", 0.6, 0.5, "IV rank low; prefer defined risk spreads")

        if abs(drift) < 0.01 and vol != "high_vol":
            add("iron_condor", 0.75, 0.55, "Flat drift; condor top pick")

        return list(scores.values())


def serialize_recommendations(candidates: Iterable[StrategyCandidate]) -> List[Dict[str, Any]]:
    return [
        {
            "name": c.name,
            "score": round(c.score, 3),
            "confidence": round(c.confidence, 3),
            "rationale": c.rationale,
            "sizing_hint": round(c.sizing_hint, 3),
        }
        for c in candidates
    ]
