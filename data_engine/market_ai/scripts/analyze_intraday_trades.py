#!/usr/bin/env python3
"""
Analyze intraday trade log (reports/intraday_theta_trades.csv) and print
summary stats by structure, regime, reason, etc.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze intraday trade log.")
    parser.add_argument("--trades", type=Path, required=True, help="Path to intraday_theta_trades.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.trades)
    if df.empty:
        print("No trades found.")
        return
    closes = df.loc[df["action"] == "CLOSE"].copy()
    if closes.empty:
        print("No CLOSE rows; nothing to analyze.")
        return

    closes["date"] = pd.to_datetime(closes["timestamp"]).dt.date
    closes["realized"] = pd.to_numeric(closes["realized"], errors="coerce").fillna(0.0)
    closes["structure"] = closes.get("structure", "STRANGLE")
    closes["reason"] = closes.get("reason", "").fillna("N/A")
    closes["regime"] = closes["reason"].str.extract(r"regime=([A-Za-z_]+)")

    metrics = {
        "trades": closes["realized"].count(),
        "avg_pnl": closes["realized"].mean(),
        "median_pnl": closes["realized"].median(),
        "win_rate": (closes["realized"] > 0).mean(),
        "total_pnl": closes["realized"].sum(),
    }
    print("Overall metrics:", metrics)

    by_structure = closes.groupby("structure")["realized"].agg(["count", "mean", "sum"])
    print("\nP&L by structure:\n", by_structure)

    by_reason = closes.groupby("reason")["realized"].agg(["count", "mean", "sum"])
    print("\nP&L by exit reason:\n", by_reason)

    by_date = closes.groupby("date")["realized"].sum()
    print("\nDaily P&L head:\n", by_date.head())


if __name__ == "__main__":
    main()

