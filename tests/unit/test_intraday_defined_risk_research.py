from __future__ import annotations

from pathlib import Path

import pandas as pd

from market_ai.intraday_defined_risk.research import (
    append_research_backtest_dataset_from_rolling,
    build_monitor_times,
    build_research_backtest_dataset,
)


def test_build_monitor_times_generates_5_minute_grid() -> None:
    values = build_monitor_times(start="09:30", end="10:00", step_minutes=5)
    assert values == ("09:30", "09:35", "09:40", "09:45", "09:50", "09:55", "10:00")


def test_build_research_backtest_dataset_derives_missing_chain_fields(tmp_path: Path) -> None:
    structured = tmp_path / "structured"
    structured.mkdir()
    pd.DataFrame(
        [
            {"timestamp": "2025-11-13T09:15:00", "open": 24500.0, "high": 24520.0, "low": 24490.0, "close": 24510.0, "volume": 1000.0},
            {"timestamp": "2025-11-13T09:20:00", "open": 24510.0, "high": 24530.0, "low": 24500.0, "close": 24520.0, "volume": 1100.0},
            {"timestamp": "2025-11-13T09:25:00", "open": 24520.0, "high": 24540.0, "low": 24510.0, "close": 24530.0, "volume": 1200.0},
        ]
    ).to_csv(structured / "nifty_5m.csv", index=False)
    pd.DataFrame(
        [
            {
                "session_date": "2025-11-13",
                "timestamp": "2025-11-13T09:30:00",
                "decision_time": "09:30",
                "expiry": "2025-11-20",
                "spot": 24530.0,
                "strike": 24700,
                "option_type": "CALL",
                "bid": "",
                "ask": "",
                "ltp": 32.5,
                "delta": "",
                "iv": 14.0,
                "oi": 250000,
            },
            {
                "session_date": "2025-11-13",
                "timestamp": "2025-11-13T09:30:00",
                "decision_time": "09:30",
                "expiry": "2025-11-20",
                "spot": 24530.0,
                "strike": 24300,
                "option_type": "PUT",
                "bid": "",
                "ask": "",
                "ltp": 28.0,
                "delta": "",
                "iv": 15.0,
                "oi": 275000,
            },
        ]
    ).to_csv(structured / "option_chain_decision_times.csv", index=False)
    pd.DataFrame([], columns=["session_date", "timestamp", "strategy", "side", "strike", "option_type", "quoted_mid", "fill_price", "quantity", "slippage_points", "pnl_rupees"]).to_csv(structured / "execution_fills.csv", index=False)
    pd.DataFrame([{"session_date": "2025-11-13", "label": "trend_up"}]).to_csv(structured / "session_labels.csv", index=False)

    output = tmp_path / "research"
    report = build_research_backtest_dataset(structured, output, from_date="2025-11-13", to_date="2025-11-13")

    chain = pd.read_csv(output / "options_chain.csv")
    assert report.trading_days == 1
    assert report.derived_delta_rows == 2
    assert report.derived_bid_ask_rows == 2
    assert chain["delta"].notna().all()
    assert (chain["bid"] > 0).all()
    assert (chain["ask"] > chain["bid"]).all()
    assert (output / "nifty_15m.csv").exists()


def test_append_research_backtest_dataset_from_rolling_appends_tail(tmp_path: Path) -> None:
    rolling_root = tmp_path / "rolling"
    for session_date, spot in [("2025-11-13", 24500.0), ("2025-11-14", 24600.0)]:
        day_dir = rolling_root / session_date
        day_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "tradeDate": session_date,
                    "tradeTime": "09:30:00",
                    "expiryDate": "",
                    "strikePrice": int(spot + 100),
                    "optionType": "CE",
                    "ltp": 120.0,
                    "open": 121.0,
                    "high": 123.0,
                    "low": 118.0,
                    "close": 120.0,
                    "volume": 1000,
                    "oi": 2000,
                    "iv": 12.0,
                    "spot": spot,
                    "delta": None,
                },
                {
                    "tradeDate": session_date,
                    "tradeTime": "09:35:00",
                    "expiryDate": "",
                    "strikePrice": int(spot - 100),
                    "optionType": "PE",
                    "ltp": 118.0,
                    "open": 119.0,
                    "high": 120.0,
                    "low": 117.0,
                    "close": 118.0,
                    "volume": 930,
                    "oi": 1810,
                    "iv": 11.6,
                    "spot": spot,
                    "delta": None,
                },
            ]
        ).to_parquet(day_dir / "rolling_option.parquet", index=False)

    output_root = tmp_path / "research"
    first = append_research_backtest_dataset_from_rolling(
        rolling_root,
        output_root,
        from_date="2025-11-13",
        to_date="2025-11-13",
        monitor_times=("09:30", "09:35"),
    )
    second = append_research_backtest_dataset_from_rolling(
        rolling_root,
        output_root,
        from_date="2025-11-13",
        to_date="2025-11-14",
        monitor_times=("09:30", "09:35"),
    )

    chain = pd.read_csv(output_root / "options_chain.csv")
    assert first.appended_trading_days == 1
    assert second.appended_trading_days == 1
    assert second.total_trading_days == 2
    assert pd.to_datetime(chain["timestamp"]).dt.date.nunique() == 2
