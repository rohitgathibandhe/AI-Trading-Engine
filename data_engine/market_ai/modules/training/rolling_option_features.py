# -*- coding: utf-8 -*-
"""
Utilities to transform rolling option parquet snapshots into selector features.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd


@dataclass
class RollingOptionDataset:
    root_dir: Path

    def iter_parquet(self) -> Iterable[Path]:
        yield from sorted(self.root_dir.glob("**/rolling_option.parquet"))


def _standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "tradeDate": "trade_date",
        "tradeTime": "trade_time",
        "expiryDate": "expiry",
        "strikePrice": "strike",
        "optionType": "option_type",
        "ltp": "ltp",
        "last_traded_price": "ltp",
        "delta": "delta",
        "underlyingPrice": "spot",
        "underlyingLtp": "spot",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def build_features_from_parquet(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = _standardise_columns(df)
    required = {"trade_date", "trade_time", "expiry", "strike", "option_type", "ltp"}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    for col in ("strike", "ltp", "delta", "spot"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["option_type"] = df["option_type"].str.upper()

    grouped = []
    for (trade_date, trade_time, expiry, strike), chunk in df.groupby(["trade_date", "trade_time", "expiry", "strike"], dropna=True):
        if chunk.empty:
            continue
        call_row = chunk[chunk["option_type"].str.startswith("C")].iloc[:1]
        put_row = chunk[chunk["option_type"].str.startswith("P")].iloc[:1]
        if call_row.empty or put_row.empty:
            continue
        call = call_row.iloc[0]
        put = put_row.iloc[0]
        record = {
            "timestamp": f"{trade_date} {trade_time}",
            "strategy": "strangle",
            "expiry": expiry,
            "exchange_seg": "NSE_FNO",
            "spot": call.get("spot") or put.get("spot"),
            "ce_strike": strike,
            "pe_strike": strike,
            "ce_ltp": call.get("ltp"),
            "pe_ltp": put.get("ltp"),
            "ce_delta": call.get("delta"),
            "pe_delta": put.get("delta"),
            "ce_entry": call.get("ltp"),
            "pe_entry": put.get("ltp"),
            "net_credit": (call.get("ltp") or 0.0) + (put.get("ltp") or 0.0),
            "warn_only": 0,
            "trade_mode": "historical",
            "context": "rollingoption",
        }
        grouped.append(record)
    return pd.DataFrame(grouped)


def build_dataset(root_dir: Path) -> pd.DataFrame:
    dataset = RollingOptionDataset(root_dir)
    frames: List[pd.DataFrame] = []
    for parquet_path in dataset.iter_parquet():
        df = build_features_from_parquet(parquet_path)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
