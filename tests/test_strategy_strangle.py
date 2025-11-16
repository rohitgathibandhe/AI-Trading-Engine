from datetime import date

import pytest

from market_ai.modules.strategies.monthly_strangle_with_weekly_hedge import (
    MonthlyStrangleWithWeeklyHedge,
)

pytestmark = pytest.mark.unit


from typing import Optional


class StubMarket:
    def __init__(self, option_chain: dict):
        self._chain = option_chain

    def get_option_chain(self, underlying_id: int, expiry: str, as_of_date: Optional[str] = None) -> dict:
        return self._chain


def _build_chain():
    return {
        "24800": {
            "ce": {"last_price": 6.0, "greeks": {"delta": 0.08}},
            "pe": {"last_price": 36.0, "greeks": {"delta": -0.32}},
        },
        "24950": {
            "ce": {"last_price": 10.0, "greeks": {"delta": 0.12}},
            "pe": {"last_price": 25.0, "greeks": {"delta": -0.22}},
        },
        "25200": {
            "ce": {"last_price": 14.0, "greeks": {"delta": 0.18}},
            "pe": {"last_price": 10.0, "greeks": {"delta": -0.08}},
        },
        "25350": {
            "ce": {"last_price": 20.0, "greeks": {"delta": 0.26}},
            "pe": {"last_price": 4.0, "greeks": {"delta": -0.04}},
        },
    }


def test_strangle_selects_otm_strikes_near_target_delta(monkeypatch):
    chain = _build_chain()
    market = StubMarket(chain)
    cfg = {
        "lot_size": 50,
        "min_distance_pct": 0.0,
        "entry_relax_plan": [{"label": "base", "min_distance_pct": 0.0, "delta_band": {"call": (0.05, 0.4), "put": (-0.4, -0.05)}}],
    }
    strategy = MonthlyStrangleWithWeeklyHedge(cfg, market)
    strategy.chain_ingestor = None  # force usage of stub market
    # Skip any augmentation from rolling cache so the stub chain is the only source.
    import market_ai.modules.strategies.monthly_strangle_with_weekly_hedge as strat_mod
    monkeypatch.setattr(strat_mod, "_load_local_rolling_option_chain", lambda *args, **kwargs: {})

    # Use a recent date so the strategy does not force local-only mode.
    entry_day = date.today()
    expiry = date(entry_day.year, entry_day.month, 25)
    strategy._enter_on_expiry_for_next_month(entry_day, expiry, spot_hint=25000.0)

    key = f"{expiry.year}-{expiry.month:02d}"
    assert key in strategy.positions
    pos = strategy.positions[key]
    # Both strikes should be chosen from the provided chain and stay OTM relative to spot_hint.
    assert pos.ce_strike in (25200.0, 25350.0)
    assert pos.ce_strike > 25000.0
    assert pos.pe_strike in (24800.0, 24950.0)
    assert pos.pe_strike < 25000.0
    assert pos.net_credit > 0
