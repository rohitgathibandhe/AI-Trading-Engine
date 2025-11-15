#!/usr/bin/env python3
"""
Aggregate weekly theta daily snapshots into a model-ready dataset.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ML dataset from weekly theta daily logs.")
    parser.add_argument("--daily-log", type=Path, required=True, help="Path to weekly_theta_daily_log.csv")
    parser.add_argument("--trades", type=Path, help="Optional weekly_theta_trades.csv for additional labels")
    parser.add_argument("--output", type=Path, default=Path("reports/weekly_theta_learning_dataset.csv"))
    return parser.parse_args()


def _parse_events(raw: Any) -> list[Dict[str, Any]]:
    if pd.isna(raw):
        return []
    if isinstance(raw, (list, tuple)):
        return list(raw)
    text = str(raw).strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except Exception:
        try:
            # fall back to python literal-style
            import ast

            return ast.literal_eval(text)
        except Exception:
            return []


def main() -> None:
    args = parse_args()
    if not args.daily_log.exists():
        raise SystemExit(f"Daily log not found: {args.daily_log}")
    daily_df = pd.read_csv(args.daily_log)
    if daily_df.empty:
        print("[learning-dataset] daily log empty; nothing to write.")
        return
    daily_df["date"] = pd.to_datetime(daily_df["date"])
    daily_df["events_list"] = daily_df["events"].apply(_parse_events)
    daily_df["has_event_risk"] = daily_df["event_count"].fillna(0).astype(int) > 0
    daily_df["blocked_today"] = daily_df["entry_block_reason"].notna()
    daily_df["close_flag"] = daily_df["close_reason"].notna()
    daily_df["drawdown_from_best"] = (daily_df["best_pnl"] - daily_df["current_pnl"]).fillna(0.0)
    daily_df["entry_dt"] = pd.to_datetime(daily_df["entry_date"], errors="coerce")
    daily_df = daily_df.sort_values("date").reset_index(drop=True)
    cycle_ids, _ = pd.factorize(daily_df["entry_dt"])
    daily_df["cycle_id"] = cycle_ids
    daily_df.loc[daily_df["entry_dt"].isna(), "cycle_id"] = -1
    closes = daily_df.loc[(daily_df["close_flag"]) & daily_df["entry_dt"].notna(), ["entry_dt", "close_pnl", "date"]].copy()
    closes = closes.rename(columns={"date": "close_date"})
    cycle_info = (
        closes.groupby("entry_dt")
        .agg(final_realized=("close_pnl", "last"), cycle_close_date=("close_date", "last"))
        .reset_index()
    )
    daily_df = daily_df.merge(cycle_info, on="entry_dt", how="left")
    entry_dt_local = daily_df["entry_dt"].dt.tz_localize(None)
    entry_days = entry_dt_local.dt.normalize()
    daily_dates = daily_df["date"].dt.normalize()
    daily_df["days_since_entry"] = (daily_dates - entry_days).dt.days
    cycle_close = pd.to_datetime(daily_df["cycle_close_date"], errors="coerce")
    if cycle_close.notna().any():
        cycle_close = cycle_close.dt.normalize()
    daily_df["days_until_close"] = (cycle_close - daily_dates).dt.days
    daily_df.loc[daily_df["entry_dt"].isna(), ["days_since_entry", "days_until_close"]] = None
    if args.trades and args.trades.exists():
        trades_df = pd.read_csv(args.trades)
        closes = trades_df.loc[trades_df["action"] == "CLOSE"][["timestamp", "realized", "reason"]].copy()
        closes["timestamp"] = pd.to_datetime(closes["timestamp"])
        closes["close_date"] = closes["timestamp"].dt.date
        agg = closes.groupby("close_date").agg({"realized": "sum"}).rename(columns={"realized": "realized_on_close"}).reset_index()
        daily_df["date_key"] = daily_df["date"].dt.date
        daily_df = daily_df.merge(agg, how="left", left_on="date_key", right_on="close_date")
        daily_df.drop(columns=["date_key", "close_date"], inplace=True, errors="ignore")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    daily_df.to_csv(args.output, index=False)
    print(f"[learning-dataset] wrote {len(daily_df)} rows to {args.output}")


if __name__ == "__main__":
    main()
