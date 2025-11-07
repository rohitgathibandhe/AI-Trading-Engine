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
    strategy = MonthlyStrangleWithWeeklyHedge({"lot_size": 50}, market)
    strategy.chain_ingestor = None  # force usage of stub market

    entry_day = date(2025, 10, 28)
    expiry = date(2025, 11, 25)
    strategy._enter_on_expiry_for_next_month(entry_day, expiry, spot_hint=25000.0)

    key = f"{expiry.year}-{expiry.month:02d}"
    assert key in strategy.positions
    pos = strategy.positions[key]
    assert pos.ce_strike == pytest.approx(25200.0)
    # nearest OTM put with |delta| >= target is 24950 (delta -0.22)
    assert pos.pe_strike == pytest.approx(24950.0)
    assert pos.net_credit > 0
