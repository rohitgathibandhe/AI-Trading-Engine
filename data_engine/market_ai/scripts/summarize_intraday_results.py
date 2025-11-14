#!/usr/bin/env python3
"""
Summarize intraday backtest output (trades CSV) into daily P&L stats.

Usage:
    python3 data_engine/market_ai/scripts/summarize_intraday_results.py \
        --trades reports/intraday_theta_trades.csv \
        --output reports/intraday_theta_daily_summary.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize intraday trades into daily statistics.")
    parser.add_argument("--trades", type=Path, required=True, help="Path to intraday_theta_trades.csv")
    parser.add_argument("--output", type=Path, help="Optional CSV path for daily summary")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trades_df = pd.read_csv(args.trades)
    if trades_df.empty:
        print("[summarize] trades file is empty")
        return

    trades_df["timestamp"] = pd.to_datetime(trades_df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
    trades_df["date"] = trades_df["timestamp"].dt.date

    closes = trades_df.loc[trades_df["action"] == "CLOSE"].copy()
    if closes.empty:
        print("[summarize] no CLOSE events found; cannot compute P&L")
        return

    closes["realized"] = pd.to_numeric(closes["realized"], errors="coerce").fillna(0.0)
    daily = closes.groupby("date").agg(
        trades=("realized", "size"),
        daily_pnl=("realized", "sum"),
        max_win=("realized", "max"),
        max_loss=("realized", "min"),
    )
    daily["running_pnl"] = daily["daily_pnl"].cumsum()

    print(daily)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(args.output)
        print(f"[summarize] wrote daily stats to {args.output}")


if __name__ == "__main__":
    main()

