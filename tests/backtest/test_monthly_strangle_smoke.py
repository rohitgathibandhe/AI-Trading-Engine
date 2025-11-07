import json
from pathlib import Path

import pandas as pd
import pytest

from market_ai.modules.strategies import monthly_strangle_with_weekly_hedge as strat_mod
from market_ai.modules.strategies.monthly_strangle_with_weekly_hedge import run_backtest

pytestmark = pytest.mark.backtest


class StubChainIngestor:
    """Replaces OptionChainIngestor.fetch with deterministic data for tests."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def fetch(self, expiry: str) -> pd.DataFrame:
        return self._df.copy()


def _patch_ingestor(monkeypatch, option_chain_df: pd.DataFrame):
    def _factory(*args, **kwargs):
        return StubChainIngestor(option_chain_df)

    monkeypatch.setattr(strat_mod, "OptionChainIngestor", _factory)


def test_monthly_strangle_backtest_smoke(monkeypatch, backtest_prices_df, option_chain_df, mock_market_adapter):
    _patch_ingestor(monkeypatch, option_chain_df)

    cfg = {
        "lot_size": 50,
        "round_multiple": 50,
        "market": mock_market_adapter,
        "option_chain": {"underlying_id": 13, "underlying_seg": "NSE_FNO"},
    }

    trades_df, _, summary = run_backtest(backtest_prices_df[["date", "close"]], cfg)

    assert isinstance(summary, dict)
    assert summary["monthly_cycles"] >= 1
    assert summary["n_trades"] >= 2
    assert not trades_df.empty, "expected at least one trade in smoke backtest"
