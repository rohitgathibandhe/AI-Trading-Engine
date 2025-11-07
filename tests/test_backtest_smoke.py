from datetime import datetime
from typing import Optional

import pandas as pd
import pytest

from market_ai.modules.strategies import monthly_strangle_with_weekly_hedge as strat_mod
from market_ai.modules.strategies.monthly_strangle_with_weekly_hedge import (
    OpenPosition,
    TradeAction,
    run_backtest,
)

pytestmark = pytest.mark.backtest


class TinyIngestor:
    def __init__(self):
        self._df = pd.DataFrame(
            [
                {
                    "strike": 24800.0,
                    "ltp_ce": 8.0,
                    "ltp_pe": 28.0,
                    "delta_ce": 0.12,
                    "delta_pe": -0.32,
                    "raw_call": {"securityId": 110},
                    "raw_put": {"securityId": 210},
                    "underlying_ltp": 25010.0,
                },
                {
                    "strike": 25200.0,
                    "ltp_ce": 18.0,
                    "ltp_pe": 15.0,
                    "delta_ce": 0.20,
                    "delta_pe": -0.20,
                    "raw_call": {"securityId": 120},
                    "raw_put": {"securityId": 220},
                    "underlying_ltp": 25010.0,
                },
            ]
        )

    def fetch(self, expiry: str) -> pd.DataFrame:
        return self._df.copy()


class StubMarket:
    def __init__(self):
        self._chain = {
            "25000": {
                "ce": {"last_price": 20.0, "greeks": {"delta": 0.18}},
                "pe": {"last_price": 24.0, "greeks": {"delta": -0.18}},
            }
        }

    def get_option_chain(self, underlying_id: int, expiry: str, as_of_date: Optional[str] = None) -> dict:
        return self._chain


def test_backtest_one_day_smoke(monkeypatch):
    df = pd.DataFrame(
        [
            {"date": datetime(2025, 10, 28).date(), "close": 25000.0},
            {"date": datetime(2025, 11, 25).date(), "close": 25200.0},
        ]
    )

    def _factory(*args, **kwargs):
        return TinyIngestor()

    monkeypatch.setattr(strat_mod, "OptionChainIngestor", _factory)

    def fake_enter(self, entry_day, sell_expiry, spot_hint):
        key = f"{sell_expiry.year}-{sell_expiry.month:02d}"
        self.positions[key] = OpenPosition(
            month_key=key,
            entry_date=entry_day,
            expiry=sell_expiry,
            ce_strike=25200.0,
            pe_strike=24800.0,
            ce_prem=13.5,
            pe_prem=33.5,
            qty=50,
            net_credit=2350.0,
            is_closed=True,
        )
        self.trades.append(
            TradeAction(entry_day.isoformat(), "SELL_CE", 25200.0, sell_expiry.isoformat(), 13.5, 50, {"tag": "test"})
        )
        self.trades.append(
            TradeAction(entry_day.isoformat(), "SELL_PE", 24800.0, sell_expiry.isoformat(), 33.5, 50, {"tag": "test"})
        )

    monkeypatch.setattr(
        strat_mod.MonthlyStrangleWithWeeklyHedge,
        "_enter_on_expiry_for_next_month",
        fake_enter,
    )

    monkeypatch.setattr(
        strat_mod.MonthlyStrangleWithWeeklyHedge,
        "_monitor_and_adjust",
        lambda *args, **kwargs: None,
    )

    trades_df, _, summary = run_backtest(df, {"lot_size": 50, "market": StubMarket()})
    assert summary["n_trades"] >= 2
    pnl = summary["net_credit"] / 2.0  # harness double-counts for paired sells
    assert pnl == pytest.approx(2350.0)
