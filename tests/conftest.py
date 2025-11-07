from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import pytest

from tests.mocks.dhan_mock import (
    FakeDhanClient,
    MockMarketAdapter,
    build_chain_dataframe,
)


@pytest.fixture(scope="session")
def tests_root() -> Path:
    return Path(__file__).resolve().parent


@pytest.fixture(scope="session")
def sample_positions(tests_root: Path) -> list[dict]:
    path = tests_root / "data" / "sample_positions.json"
    return json.loads(path.read_text())


@pytest.fixture(scope="session")
def sample_ltp_map() -> Dict[Tuple[str, int], float]:
    return {
        ("NSE_FNO", 53023): 2.15,
        ("NSE_FNO", 52848): 4.70,
        ("NSE_FNO", 53214): 81.60,
        ("NSE_FNO", 52802): 73.80,
    }


@pytest.fixture(scope="session")
def backtest_prices_df(tests_root: Path) -> pd.DataFrame:
    return pd.read_csv(tests_root / "data" / "backtest_prices.csv")


@pytest.fixture(scope="session")
def mock_market_adapter(tests_root: Path) -> MockMarketAdapter:
    return MockMarketAdapter(option_chain_path=tests_root / "data" / "mock_option_chain.json")


@pytest.fixture(scope="session")
def option_chain_df(tests_root: Path):
    return build_chain_dataframe(tests_root / "data" / "mock_option_chain.json")


@pytest.fixture
def fake_dhan_client(sample_positions) -> FakeDhanClient:
    return FakeDhanClient(open_positions=sample_positions.copy())
