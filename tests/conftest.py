from __future__ import annotations

import json
import sys
import importlib
from pathlib import Path
from typing import Dict, Tuple

# Normalize sys.path before any project imports so we always take this checkout.
sys.path = [p for p in sys.path if "Algotrade_ai" not in p]
REPO_ROOT = Path(__file__).resolve().parents[1]
MARKET_AI_SRC = REPO_ROOT / "data_engine"
if str(MARKET_AI_SRC) not in sys.path:
    sys.path.insert(0, str(MARKET_AI_SRC))
if "market_ai" in sys.modules:
    loaded = Path(getattr(sys.modules["market_ai"], "__file__", "") or "")
    if loaded and MARKET_AI_SRC not in loaded.parents:
        sys.modules.pop("market_ai")
importlib.invalidate_caches()
importlib.import_module("market_ai")

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
