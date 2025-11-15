#!/usr/bin/env python3
"""
Convert cached rolling-option parquet dumps into the intraday CSV format required
by IntradayThetaScalp.

Usage:
    python3 data_engine/market_ai/scripts/make_intraday_dataset.py \
        --root data_engine/market_ai/state/rolling_option \
        --dates 2024-11-13,2024-11-14 \
        --selector ATM \
        --output reports/intraday_from_rolling.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

TIMEZONE = "Asia/Kolkata"
CALL_ALIASES = {"CALL", "CE"}
PUT_ALIASES = {"PUT", "PE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build intraday dataset from rolling option cache.")
    parser.add_argument("--root", type=Path, default=Path("data_engine/market_ai/state/rolling_option"))
    parser.add_argument(
        "--dates",
        required=True,
        help="Comma separated list of trade dates (YYYY-MM-DD) to include.",
    )
    parser.add_argument("--selector", default="ATM", help="Selector tag to filter on (default: ATM).")
    parser.add_argument("--output", type=Path, default=Path("reports/intraday_from_rolling.csv"))
    parser.add_argument("--tz", default=TIMEZONE, help="Timezone for timestamps.")
    return parser.parse_args()


def _clean_option_type(value: str) -> Optional[str]:
    if not isinstance(value, str):
        return None
    upper = value.strip().upper()
    if upper in CALL_ALIASES:
        return "CALL"
    if upper in PUT_ALIASES:
        return "PUT"
    return None


def _iv_rank(series: pd.Series) -> pd.Series:
    clean = series.astype(float).replace([np.inf, -np.inf], np.nan)
    min_val = clean.min()
    max_val = clean.max()
    if pd.isna(min_val) or pd.isna(max_val) or min_val == max_val:
        return pd.Series([np.nan] * len(series), index=series.index)
    return (clean - min_val) / (max_val - min_val)


def _skew(a: pd.Series, b: pd.Series) -> pd.Series:
    denom = (a.abs() + b.abs()).replace(0, np.nan)
    return (a - b) / denom


def process_day(day_dir: Path, selector: str, tz: str) -> pd.DataFrame:
    parquet_path = day_dir / "rolling_option.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"{parquet_path} not found")
    df = pd.read_parquet(parquet_path)
    df = df.copy()
    df["selector"] = df["selector"].astype(str).str.upper()
    selector_upper = selector.upper()
    df = df.loc[df["selector"] == selector_upper]
    df["optionTypeNorm"] = df["optionType"].apply(_clean_option_type)
    df = df.loc[df["optionTypeNorm"].isin(["CALL", "PUT"])]
    df["timestamp"] = pd.to_datetime(df["tradeDate"] + " " + df["tradeTime"]).dt.tz_localize(tz)

    call_cols = {
        "strikePrice": "atm_call_strike",
        "ltp": "atm_call_ltp",
        "delta": "atm_call_delta",
        "iv": "atm_call_iv",
        "oi": "atm_call_oi",
        "volume": "atm_call_volume",
    }
    put_cols = {
        "strikePrice": "atm_put_strike",
        "ltp": "atm_put_ltp",
        "delta": "atm_put_delta",
        "iv": "atm_put_iv",
        "oi": "atm_put_oi",
        "volume": "atm_put_volume",
    }

    calls = df.loc[df["optionTypeNorm"] == "CALL", ["timestamp", "spot", *call_cols.keys()]].rename(columns=call_cols)
    puts = df.loc[df["optionTypeNorm"] == "PUT", ["timestamp", *put_cols.keys()]].rename(columns=put_cols)

    merged = pd.merge(calls, puts, on="timestamp", how="inner")
    merged = merged.sort_values("timestamp")
    merged["combined_premium_pct"] = (merged["atm_call_ltp"] + merged["atm_put_ltp"]) / merged["spot"].replace(0, np.nan)
    merged["iv_rank"] = _iv_rank(merged["atm_call_iv"])
    merged["iv_skew"] = _skew(merged["atm_call_iv"], merged["atm_put_iv"])
    merged["oi_skew"] = _skew(merged["atm_call_oi"], merged["atm_put_oi"])
    merged["volume_skew"] = _skew(
        merged["atm_call_volume"].replace(0, np.nan),
        merged["atm_put_volume"].replace(0, np.nan),
    )
    merged["source_date"] = day_dir.name
    return merged


def main() -> None:
    args = parse_args()
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    all_rows: List[pd.DataFrame] = []
    for day in dates:
        day_dir = args.root / day
        if not day_dir.exists():
            print(f"[make_intraday_dataset] skipping {day} (directory missing)")
            continue
        try:
            df = process_day(day_dir, args.selector, args.tz)
        except Exception as exc:
            print(f"[make_intraday_dataset] failed for {day}: {exc}")
            continue
        all_rows.append(df)
        print(f"[make_intraday_dataset] processed {day}: {len(df)} rows")

    if not all_rows:
        print("[make_intraday_dataset] no data processed")
        return

    final = pd.concat(all_rows, ignore_index=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output, index=False)
    print(f"[make_intraday_dataset] wrote {len(final)} rows to {args.output}")


if __name__ == "__main__":
    main()
