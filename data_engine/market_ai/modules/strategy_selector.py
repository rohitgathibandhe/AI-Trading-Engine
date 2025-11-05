# market_ai/modules/strategy_selector.py
"""
Dynamic strategy selector based on VIX regimes or other market conditions.
"""
from __future__ import annotations
from typing import Dict, Optional

def select_strategy_by_vix(vix: float) -> str:
    """
    Return the name of the strategy module/config key based on VIX value.
    """
    if vix < 10:
        return "iron_condor"
    elif vix < 12:
        return "credit_spread"
    elif vix < 18:
        return "hedged_strangle"
    else:
        return "monthly_strangle_with_weekly_hedge"

def select_strategy(market_context: Dict[str, float]) -> Optional[str]:
    """
    Generic selector: accepts market_context with keys like {"vix": 12.4, "iv_rank": 30, ...}
    For now only VIX is used.
    """
    vix = market_context.get("vix")
    if vix is None:
        return None
    return select_strategy_by_vix(vix)
