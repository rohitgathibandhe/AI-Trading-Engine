from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_ai.intraday_defined_risk.dataset import (
    _filter_session_spot_outliers,
    _resolve_expiry_text,
    assess_structured_dataset,
    build_structured_dataset_from_rolling,
    enrich_structured_chain_deltas,
    refresh_structured_training_dataset_from_rolling,
    scaffold_dataset_layout,
)


def test_scaffold_dataset_layout_creates_expected_files(tmp_path: Path) -> None:
    paths = scaffold_dataset_layout(tmp_path)
    assert paths.ohlcv_5m.exists()
    assert paths.option_chain_decision_times.exists()
    assert paths.execution_fills.exists()
    assert paths.session_labels.exists()


def test_assess_structured_dataset_reports_insufficient_on_empty_scaffold(tmp_path: Path) -> None:
    scaffold_dataset_layout(tmp_path)
    report = assess_structured_dataset(tmp_path)
    assert report.status == "INSUFFICIENT"
    assert report.has_5m_ohlcv is True
    assert report.has_chain_snapshots is True
    assert report.has_execution_fills is True
    assert report.has_session_labels is True
    assert report.available_trading_days == 0
    assert report.issues


def test_build_structured_dataset_from_rolling_creates_outputs(tmp_path: Path) -> None:
    rolling_root = tmp_path / "rolling_option"
    day_dir = rolling_root / "2025-11-10"
    day_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "tradeDate": "2025-11-10",
                "tradeTime": "09:30:00",
                "expiryDate": "2025-11-13",
                "strikePrice": 24500,
                "optionType": "CE",
                "ltp": 120.0,
                "open": 121.0,
                "high": 123.0,
                "low": 118.0,
                "close": 120.0,
                "volume": 1000,
                "oi": 2000,
                "iv": 12.0,
                "spot": 24480.0,
                "delta": 0.22,
                "gamma": 0.0,
                "theta": 0.0,
                "vega": 0.0,
                "selector": "ATM",
                "raw_option_type": "CALL",
            },
            {
                "tradeDate": "2025-11-10",
                "tradeTime": "09:30:00",
                "expiryDate": "2025-11-13",
                "strikePrice": 24300,
                "optionType": "PE",
                "ltp": 110.0,
                "open": 112.0,
                "high": 113.0,
                "low": 108.0,
                "close": 110.0,
                "volume": 900,
                "oi": 1800,
                "iv": 11.5,
                "spot": 24480.0,
                "delta": -0.21,
                "gamma": 0.0,
                "theta": 0.0,
                "vega": 0.0,
                "selector": "ATM",
                "raw_option_type": "PUT",
            },
            {
                "tradeDate": "2025-11-10",
                "tradeTime": "10:00:00",
                "expiryDate": "2025-11-13",
                "strikePrice": 24500,
                "optionType": "CE",
                "ltp": 115.0,
                "open": 116.0,
                "high": 117.0,
                "low": 114.0,
                "close": 115.0,
                "volume": 950,
                "oi": 2050,
                "iv": 12.1,
                "spot": 24460.0,
                "delta": 0.20,
                "gamma": 0.0,
                "theta": 0.0,
                "vega": 0.0,
                "selector": "ATM",
                "raw_option_type": "CALL",
            },
            {
                "tradeDate": "2025-11-10",
                "tradeTime": "10:00:00",
                "expiryDate": "2025-11-13",
                "strikePrice": 24300,
                "optionType": "PE",
                "ltp": 118.0,
                "open": 119.0,
                "high": 120.0,
                "low": 117.0,
                "close": 118.0,
                "volume": 930,
                "oi": 1810,
                "iv": 11.6,
                "spot": 24460.0,
                "delta": -0.23,
                "gamma": 0.0,
                "theta": 0.0,
                "vega": 0.0,
                "selector": "ATM",
                "raw_option_type": "PUT",
            },
            {
                "tradeDate": "2025-11-10",
                "tradeTime": "13:00:00",
                "expiryDate": "2025-11-13",
                "strikePrice": 24500,
                "optionType": "CE",
                "ltp": 90.0,
                "open": 92.0,
                "high": 93.0,
                "low": 89.0,
                "close": 90.0,
                "volume": 880,
                "oi": 2100,
                "iv": 11.0,
                "spot": 24390.0,
                "delta": 0.16,
                "gamma": 0.0,
                "theta": 0.0,
                "vega": 0.0,
                "selector": "ATM",
                "raw_option_type": "CALL",
            },
            {
                "tradeDate": "2025-11-10",
                "tradeTime": "13:00:00",
                "expiryDate": "2025-11-13",
                "strikePrice": 24300,
                "optionType": "PE",
                "ltp": 145.0,
                "open": 144.0,
                "high": 146.0,
                "low": 143.0,
                "close": 145.0,
                "volume": 910,
                "oi": 1850,
                "iv": 12.2,
                "spot": 24390.0,
                "delta": -0.28,
                "gamma": 0.0,
                "theta": 0.0,
                "vega": 0.0,
                "selector": "ATM",
                "raw_option_type": "PUT",
            },
        ]
    ).to_parquet(day_dir / "rolling_option.parquet", index=False)

    data_root = tmp_path / "structured"
    report = build_structured_dataset_from_rolling(rolling_root, data_root)

    assert report.processed_sessions == 1
    assert report.ohlcv_rows >= 3
    assert report.chain_rows >= 6
    assert (data_root / "nifty_5m.csv").exists()
    assert (data_root / "option_chain_decision_times.csv").exists()
    assert (data_root / "session_labels.csv").exists()


def test_enrich_structured_chain_deltas_derives_missing_values(tmp_path: Path) -> None:
    paths = scaffold_dataset_layout(tmp_path)
    pd.DataFrame(
        [
            {
                "session_date": "2026-03-28",
                "timestamp": "2026-03-28T09:30:00",
                "decision_time": "09:30",
                "expiry": "2026-04-02",
                "spot": 23500.0,
                "strike": 23600,
                "option_type": "CALL",
                "bid": "",
                "ask": "",
                "ltp": 48.0,
                "delta": "",
                "delta_raw": "",
                "delta_source": "",
                "iv": 14.5,
                "oi": 120000,
            },
            {
                "session_date": "2026-03-28",
                "timestamp": "2026-03-28T09:30:00",
                "decision_time": "09:30",
                "expiry": "2026-04-02",
                "spot": 23500.0,
                "strike": 23400,
                "option_type": "PUT",
                "bid": "",
                "ask": "",
                "ltp": 52.0,
                "delta": -0.41,
                "delta_raw": -0.41,
                "delta_source": "PROVIDED",
                "iv": 14.5,
                "oi": 110000,
            },
        ]
    ).to_csv(paths.option_chain_decision_times, index=False)

    report = enrich_structured_chain_deltas(tmp_path)

    assert report.total_rows == 2
    assert report.rows_missing_delta == 1
    assert report.rows_enriched == 1
    enriched = pd.read_csv(paths.option_chain_decision_times)
    assert pd.notna(enriched.loc[0, "delta"])
    assert enriched.loc[0, "delta_source"] == "BS_IV_DERIVED"
    assert enriched.loc[1, "delta_source"] == "PROVIDED"


def test_enrich_structured_chain_deltas_infers_weekly_expiry_when_missing(tmp_path: Path) -> None:
    paths = scaffold_dataset_layout(tmp_path)
    pd.DataFrame(
        [
            {
                "session_date": "2026-03-30",
                "timestamp": "2026-03-30T10:00:00",
                "decision_time": "10:00",
                "expiry": "",
                "spot": 23480.0,
                "strike": 23600,
                "option_type": "CALL",
                "bid": "",
                "ask": "",
                "ltp": 36.0,
                "delta": "",
                "delta_raw": "",
                "delta_source": "",
                "iv": 13.2,
                "oi": 98000,
            }
        ]
    ).to_csv(paths.option_chain_decision_times, index=False)

    report = enrich_structured_chain_deltas(tmp_path)
    enriched = pd.read_csv(paths.option_chain_decision_times)

    assert report.rows_enriched == 1
    assert pd.notna(enriched.loc[0, "delta"])
    assert enriched.loc[0, "delta_source"] == "BS_IV_DERIVED_INFERRED_EXPIRY"


def test_resolve_expiry_text_overrides_far_dated_dummy_expiry_for_weekly_intraday() -> None:
    expiry, inferred = _resolve_expiry_text("2026-04-28", session_date="2026-01-06")
    assert expiry == "2026-01-08"
    assert inferred is True


def test_filter_session_spot_outliers_removes_bad_prints() -> None:
    frame = pd.DataFrame(
        [
            {"timestamp": "2026-01-06T09:15:00", "spot": 26235.75, "volume": 100},
            {"timestamp": "2026-01-06T09:15:01", "spot": 26210.0, "volume": 100},
            {"timestamp": "2026-01-06T09:15:02", "spot": 13910.8, "volume": 100},
        ]
    )
    filtered = _filter_session_spot_outliers(frame)
    assert len(filtered) == 2
    assert filtered["spot"].min() > 20000


def test_refresh_structured_training_dataset_preserves_chain_rows(tmp_path: Path) -> None:
    rolling_root = tmp_path / "rolling_option"
    day_dir = rolling_root / "2025-11-10"
    day_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "tradeDate": "2025-11-10",
                "tradeTime": "09:30:00",
                "expiryDate": "2025-11-13",
                "strikePrice": 24500,
                "optionType": "CE",
                "ltp": 120.0,
                "open": 121.0,
                "high": 123.0,
                "low": 118.0,
                "close": 120.0,
                "volume": 1000,
                "oi": 2000,
                "iv": 12.0,
                "spot": 24480.0,
                "delta": 0.22,
            },
            {
                "tradeDate": "2025-11-10",
                "tradeTime": "10:00:00",
                "expiryDate": "2025-11-13",
                "strikePrice": 24300,
                "optionType": "PE",
                "ltp": 118.0,
                "open": 119.0,
                "high": 120.0,
                "low": 117.0,
                "close": 118.0,
                "volume": 930,
                "oi": 1810,
                "iv": 11.6,
                "spot": 24460.0,
                "delta": -0.23,
            },
        ]
    ).to_parquet(day_dir / "rolling_option.parquet", index=False)

    data_root = tmp_path / "structured"
    paths = scaffold_dataset_layout(data_root)
    pd.DataFrame(
        [
            {
                "session_date": "2025-11-10",
                "timestamp": "2025-11-10T09:30:00",
                "decision_time": "09:30",
                "expiry": "2025-11-13",
                "spot": 24480.0,
                "strike": 24500,
                "option_type": "CALL",
                "bid": 10.0,
                "ask": 11.0,
                "ltp": 10.5,
                "delta": 0.22,
                "delta_raw": 0.22,
                "delta_source": "PROVIDED",
                "iv": 12.0,
                "oi": 2000,
            }
        ]
    ).to_csv(paths.option_chain_decision_times, index=False)

    report = refresh_structured_training_dataset_from_rolling(rolling_root, data_root, from_date="2025-11-10", to_date="2025-11-10")

    assert report.preserved_chain_rows == 1
    assert report.ohlcv_rows >= 2
    assert report.label_rows == 1
    preserved = pd.read_csv(paths.option_chain_decision_times)
    assert len(preserved) == 1
