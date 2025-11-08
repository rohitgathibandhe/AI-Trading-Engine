"""
Lightweight export surface for the legacy ``market_ai.modules.agents`` package.

Most code now imports helpers straight from their source modules
(``market_ai.agents``, ``market_ai.modules.analytics`` etc.), but we keep these
aliases so older notebooks/scripts keep working without modification.
"""

from .adaptive_agent import AdaptiveAgent  # compat shim -> market_ai.agents.adaptive_agent
from .strategy_selector import StrategySelector
from .risk_manager import RiskManager
from .strategy_recommender import StrategyRecommender, StrategyCandidate

__all__ = [
    "AdaptiveAgent",
    "StrategySelector",
    "RiskManager",
    "StrategyRecommender",
    "StrategyCandidate",
]
