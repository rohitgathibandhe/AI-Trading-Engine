from __future__ import annotations

from datetime import date
import json
from pathlib import Path

import pandas as pd

from market_ai.intraday_defined_risk.backtest import run_backtest
from market_ai.intraday_defined_risk.backtest_dataset_tools import (
    build_dense_backtest_dataset,
    inspect_backtest_dataset,
)
from market_ai.utils.dates import last_tuesday_of_month


def _write_minimal_backtest_dataset(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"timestamp": "2026-03-02T09:15:00", "open": 22000.0, "high": 22010.0, "low": 21990.0, "close": 22005.0, "volume": 1000.0},
            {"timestamp": "2026-03-02T09:20:00", "open": 22005.0, "high": 22012.0, "low": 22000.0, "close": 22008.0, "volume": 1000.0},
            {"timestamp": "2026-03-02T09:25:00", "open": 22008.0, "high": 22015.0, "low": 22004.0, "close": 22012.0, "volume": 1000.0},
            {"timestamp": "2026-03-02T09:30:00", "open": 22012.0, "high": 22018.0, "low": 22006.0, "close": 22010.0, "volume": 1000.0},
            {"timestamp": "2026-03-02T09:35:00", "open": 22010.0, "high": 22016.0, "low": 22002.0, "close": 22006.0, "volume": 1000.0},
            {"timestamp": "2026-03-02T09:40:00", "open": 22006.0, "high": 22009.0, "low": 21998.0, "close": 22000.0, "volume": 1000.0},
        ]
    ).to_csv(root / "nifty_5m.csv", index=False)
    pd.DataFrame(
        [
            {"timestamp": "2026-03-02T09:25:00", "open": 22000.0, "high": 22015.0, "low": 21990.0, "close": 22012.0, "volume": 3000.0},
            {"timestamp": "2026-03-02T09:40:00", "open": 22012.0, "high": 22018.0, "low": 21998.0, "close": 22000.0, "volume": 3000.0},
        ]
    ).to_csv(root / "nifty_15m.csv", index=False)
    pd.DataFrame(
        [
            {"timestamp": "2026-03-02T09:30:00", "expiry": "2026-03-03", "spot": 22010.0, "strike": 22100, "option_type": "CALL", "bid": 40.0, "ask": 42.0, "ltp": 41.0, "delta": 0.22, "iv": 14.0, "oi": 100000},
            {"timestamp": "2026-03-02T09:30:00", "expiry": "2026-03-03", "spot": 22010.0, "strike": 21900, "option_type": "PUT", "bid": 38.0, "ask": 40.0, "ltp": 39.0, "delta": -0.22, "iv": 14.0, "oi": 100000},
            {"timestamp": "2026-03-02T10:00:00", "expiry": "2026-03-03", "spot": 22005.0, "strike": 22100, "option_type": "CALL", "bid": 39.0, "ask": 41.0, "ltp": 40.0, "delta": 0.21, "iv": 14.0, "oi": 100100},
            {"timestamp": "2026-03-02T10:00:00", "expiry": "2026-03-03", "spot": 22005.0, "strike": 21900, "option_type": "PUT", "bid": 39.0, "ask": 41.0, "ltp": 40.0, "delta": -0.23, "iv": 14.0, "oi": 100100},
            {"timestamp": "2026-03-02T13:00:00", "expiry": "2026-03-03", "spot": 21990.0, "strike": 22100, "option_type": "CALL", "bid": 34.0, "ask": 36.0, "ltp": 35.0, "delta": 0.18, "iv": 13.5, "oi": 100200},
            {"timestamp": "2026-03-02T13:00:00", "expiry": "2026-03-03", "spot": 21990.0, "strike": 21900, "option_type": "PUT", "bid": 45.0, "ask": 47.0, "ltp": 46.0, "delta": -0.28, "iv": 13.5, "oi": 100200},
        ]
    ).to_csv(root / "options_chain.csv", index=False)


def test_inspect_dataset_detects_sparse_dataset(tmp_path: Path) -> None:
    data_root = tmp_path / "dataset"
    _write_minimal_backtest_dataset(data_root)

    report = inspect_backtest_dataset(data_root)

    assert report.trading_days == 1
    assert report.total_unique_decision_timestamps == 3
    assert report.days_with_fewer_than_40_timestamps == ["2026-03-02"]
    assert report.days_ending_before_1445 == ["2026-03-02"]
    assert report.warnings


def test_dense_builder_uses_no_lookahead_snapshot_selection(tmp_path: Path) -> None:
    rolling_root = tmp_path / "rolling_option"
    day_dir = rolling_root / "2026-03-02"
    day_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "tradeDate": "2026-03-02",
                "tradeTime": "09:36:00",
                "expiryDate": "2026-03-03",
                "strikePrice": 22100,
                "optionType": "CE",
                "ltp": 40.0,
                "open": 40.0,
                "high": 40.0,
                "low": 40.0,
                "close": 40.0,
                "volume": 1000,
                "oi": 100000,
                "iv": 14.0,
                "spot": 22010.0,
                "delta": 0.22,
            },
            {
                "tradeDate": "2026-03-02",
                "tradeTime": "09:36:00",
                "expiryDate": "2026-03-03",
                "strikePrice": 21900,
                "optionType": "PE",
                "ltp": 39.0,
                "open": 39.0,
                "high": 39.0,
                "low": 39.0,
                "close": 39.0,
                "volume": 1000,
                "oi": 100000,
                "iv": 14.0,
                "spot": 22010.0,
                "delta": -0.22,
            },
            {
                "tradeDate": "2026-03-02",
                "tradeTime": "09:40:00",
                "expiryDate": "2026-03-03",
                "strikePrice": 22100,
                "optionType": "CE",
                "ltp": 39.0,
                "open": 39.0,
                "high": 39.0,
                "low": 39.0,
                "close": 39.0,
                "volume": 1000,
                "oi": 100100,
                "iv": 13.8,
                "spot": 22005.0,
                "delta": 0.21,
            },
            {
                "tradeDate": "2026-03-02",
                "tradeTime": "09:40:00",
                "expiryDate": "2026-03-03",
                "strikePrice": 21900,
                "optionType": "PE",
                "ltp": 40.0,
                "open": 40.0,
                "high": 40.0,
                "low": 40.0,
                "close": 40.0,
                "volume": 1000,
                "oi": 100100,
                "iv": 13.8,
                "spot": 22005.0,
                "delta": -0.23,
            },
        ]
    ).to_parquet(day_dir / "rolling_option.parquet", index=False)

    output_root = tmp_path / "dense"
    report = build_dense_backtest_dataset(
        rolling_root,
        output_root,
        start_date="2026-03-02",
        end_date="2026-03-02",
        interval="5m",
        monitor_start="09:35",
        monitor_end="09:40",
    )

    chain = pd.read_csv(output_root / "options_chain.csv")
    missing = pd.read_csv(output_root / "missing_timestamps.csv")

    assert report.populated_timestamps == 1
    assert sorted(chain["timestamp"].unique().tolist()) == ["2026-03-02T09:40:00"]
    assert "NO_SNAPSHOT_AT_OR_BEFORE_TIMESTAMP" in set(missing["reason"].tolist())


def test_last_tuesday_of_month_helper_returns_last_tuesday() -> None:
    assert last_tuesday_of_month(2026, 3) == date(2026, 3, 31)


def test_run_backtest_emits_sparse_dataset_warning(tmp_path: Path, capsys) -> None:
    data_root = tmp_path / "dataset"
    _write_minimal_backtest_dataset(data_root)

    result = run_backtest(
        data_root,
        start="2026-03-02T09:30:00",
        end="2026-03-02T13:00:00",
    )
    captured = capsys.readouterr()

    assert "DATASET_DENSITY_LOW" in captured.out
    assert "dataset_density" in result["summary"]
