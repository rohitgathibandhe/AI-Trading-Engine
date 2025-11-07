from datetime import datetime

import pytest

from market_ai.modules.analytics.market_state import MarketState
from market_ai.modules.agents.strategy_recommender import StrategyRecommender

pytestmark = pytest.mark.integration


def test_recommender_prefers_strangle_when_whitelisted():
    state = MarketState(
        as_of=datetime(2025, 11, 5, 12, 0, 0),
        trend="range_bound",
        volatility="high_vol",
        iv_rank=0.82,
        realized_vol=0.30,
        atr_pct=0.05,
        drift_pct=0.001,
        vix=12.0,
        breadth=0.55,
        features={"spot": 19500.0},
    )

    recommender = StrategyRecommender(
        selector_model_path=None,
        feature_log_path=None,
        bias={"monthly_strangle_with_weekly_hedge": 0.25},
        whitelist=["monthly_strangle_with_weekly_hedge"],
    )

    ranked = recommender.recommend(state)
    assert ranked, "expected at least one recommendation"
    assert ranked[0].name == "monthly_strangle_with_weekly_hedge"
