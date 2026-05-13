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
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import os
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from market_ai.utils.nifty_expiry_calendar import infer_nifty_weekly_expiry, next_weekday_on_or_after

TIMEZONE = "Asia/Kolkata"
WEEKLY_EXPIRY_WEEKDAY = os.environ.get("MARKET_AI_WEEKLY_EXPIRY_WEEKDAY")
CALL_ALIASES = {"CALL", "CE"}
PUT_ALIASES = {"PUT", "PE"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build intraday dataset from rolling option cache.")
    parser.add_argument("--root", type=Path, default=Path("data_engine/market_ai/state/rolling_option"))
    parser.add_argument(
        "--dates",
        required=True,
        help="Comma separated list of trade dates (YYYY-MM-DD) to include, or 'ALL' to include all subdirs.",
    )
    parser.add_argument("--selector", default=None, help="(Deprecated) single selector tag to filter on.")
    parser.add_argument(
        "--selectors",
        default="ATM,ATM+2,ATM-2,ATM+4,ATM-4,ATM+6,ATM-6,ATM+8,ATM-8,ATM+10,ATM-10,ATM+12,ATM-12",
        help="Comma separated list of selectors; first entry becomes the ATM baseline.",
    )
    parser.add_argument("--output", type=Path, default=Path("reports/intraday_from_rolling.csv"))
    parser.add_argument("--tz", default=TIMEZONE, help="Timezone for timestamps.")
    return parser.parse_args()


def _weekly_expiry_for(trade_day: datetime, trading_days: set[date] | None = None) -> datetime:
    if WEEKLY_EXPIRY_WEEKDAY is not None:
        target = next_weekday_on_or_after(trade_day.date(), int(WEEKLY_EXPIRY_WEEKDAY))
    else:
        target = infer_nifty_weekly_expiry(trade_day.date(), trading_days=trading_days)
    if target is None:
        raise ValueError(f"Could not infer weekly expiry for {trade_day.date().isoformat()}")
    return datetime.combine(target, datetime.min.time())


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


FEATURE_MAP = {
    "strikePrice": "strike",
    "ltp": "ltp",
    "delta": "delta",
    "iv": "iv",
    "oi": "oi",
    "volume": "volume",
}


def _selector_slug(selector: str) -> str:
    slug = selector.strip().lower()
    slug = slug.replace("+", "_plus").replace("-", "_minus")
    slug = slug.replace(" ", "_").replace("/", "_")
    return slug or "atm"


def _top_oi_levels(df: pd.DataFrame, option_type: str) -> pd.DataFrame:
    subset = df.loc[df["optionTypeNorm"] == option_type, ["timestamp", "strikePrice", "oi"]].copy()
    if subset.empty:
        return pd.DataFrame(columns=["timestamp", "strikePrice", "oi"])
    subset["oi_for_rank"] = subset["oi"].fillna(-1)
    subset = subset.sort_values(["timestamp", "oi_for_rank"], ascending=[True, False])
    subset = subset.drop_duplicates("timestamp", keep="first")
    return subset.drop(columns=["oi_for_rank"])


def _build_selector_block(df: pd.DataFrame, selector: str, include_spot: bool) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    call_cols = ["timestamp", "spot", *FEATURE_MAP.keys()] if include_spot else ["timestamp", *FEATURE_MAP.keys()]
    calls = df.loc[df["optionTypeNorm"] == "CALL", call_cols].copy()
    puts = df.loc[df["optionTypeNorm"] == "PUT", ["timestamp", *FEATURE_MAP.keys()]].copy()
    if calls.empty or puts.empty:
        return pd.DataFrame()
    slug = _selector_slug(selector)
    call_prefix = "atm_call" if slug == "atm" else f"call_{slug}"
    put_prefix = "atm_put" if slug == "atm" else f"put_{slug}"
    call_rename = {src: f"{call_prefix}_{dst}" for src, dst in FEATURE_MAP.items()}
    put_rename = {src: f"{put_prefix}_{dst}" for src, dst in FEATURE_MAP.items()}
    if include_spot and "spot" in calls.columns:
        spot_col = calls.pop("spot")
        calls = calls.rename(columns=call_rename)
        calls.insert(1, "spot", spot_col)
    else:
        calls = calls.rename(columns=call_rename)
    puts = puts.rename(columns=put_rename)
    merged = pd.merge(calls, puts, on="timestamp", how="inner")
    return merged


def _pick_by_distance(
    df: pd.DataFrame,
    spot_map: pd.Series,
    option_type: str,
    offsets: List[int],
) -> pd.DataFrame:
    """
    For each timestamp, pick the contract whose strike is closest to spot±offset.
    Offsets are in absolute points.
    """
    records = []
    grouped = df.groupby("timestamp")
    for ts, g in grouped:
        spot = spot_map.get(ts)
        if pd.isna(spot):
            continue
        for off in offsets:
            if option_type == "CALL":
                target = spot + off
                candidates = g[g["strikePrice"] >= target]
                if candidates.empty:
                    candidates = g[g["strikePrice"] <= target]
            else:
                target = spot - off
                candidates = g[g["strikePrice"] <= target]
                if candidates.empty:
                    candidates = g[g["strikePrice"] >= target]
            if candidates.empty:
                continue
            picked = candidates.loc[(candidates["strikePrice"] - target).abs().idxmin()]
            rec = {
                "timestamp": ts,
                "offset": off,
                "strike": picked.get("strikePrice"),
                "ltp": picked.get("ltp"),
                "delta": picked.get("delta"),
                "iv": picked.get("iv"),
                "oi": picked.get("oi"),
                "volume": picked.get("volume"),
            }
            records.append(rec)
    return pd.DataFrame.from_records(records)


def process_day(day_dir: Path, selectors: List[str], tz: str, trading_days: set[date] | None = None) -> Optional[pd.DataFrame]:
    parquet_path = day_dir / "rolling_option.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"{parquet_path} not found")
    df = pd.read_parquet(parquet_path)
    df = df.copy()
    df["selector"] = df["selector"].astype(str).str.upper()
    selector_order = [sel.upper() for sel in selectors if sel]
    if not selector_order:
        selector_order = ["ATM"]
    df["optionTypeNorm"] = df["optionType"].apply(_clean_option_type)
    df = df.loc[df["optionTypeNorm"].isin(["CALL", "PUT"])]
    df["timestamp"] = pd.to_datetime(df["tradeDate"] + " " + df["tradeTime"]).dt.tz_localize(tz)

    selector_blocks = {}
    for idx, selector in enumerate(selector_order):
        subset = df.loc[df["selector"] == selector]
        if subset.empty:
            continue
        block = _build_selector_block(subset, selector, include_spot=(idx == 0))
        if block.empty:
            continue
        selector_blocks[selector] = block
    if not selector_blocks:
        return None
    base_selector = selector_order[0]
    base_block = selector_blocks.get(base_selector)
    if base_block is None:
        return None
    merged = base_block
    for selector, block in selector_blocks.items():
        if selector == base_selector:
            continue
        merged = pd.merge(merged, block, on="timestamp", how="left")

    call_top = _top_oi_levels(df, "CALL").rename(
        columns={"strikePrice": "call_oi_max_strike", "oi": "call_oi_max"}
    )
    put_top = _top_oi_levels(df, "PUT").rename(
        columns={"strikePrice": "put_oi_max_strike", "oi": "put_oi_max"}
    )

    merged = pd.merge(merged, call_top, on="timestamp", how="left")
    merged = pd.merge(merged, put_top, on="timestamp", how="left")
    merged = merged.sort_values("timestamp")

    # Distance-based selectors (strangle-friendly)
    try:
        spot_map = merged.set_index("timestamp")["spot"]
        calls = df.loc[df["optionTypeNorm"] == "CALL", ["timestamp", "strikePrice", "ltp", "delta", "iv", "oi", "volume"]]
        puts = df.loc[df["optionTypeNorm"] == "PUT", ["timestamp", "strikePrice", "ltp", "delta", "iv", "oi", "volume"]]
        offsets_pts = [250, 300, 350, 400]
        for off in offsets_pts:
            cblock = _pick_by_distance(calls, spot_map, "CALL", [off])
            pblock = _pick_by_distance(puts, spot_map, "PUT", [off])
            if not cblock.empty:
                merged = merged.merge(
                    cblock.add_prefix(f"call_off{off}_"),
                    left_on="timestamp",
                    right_on=f"call_off{off}_timestamp",
                    how="left",
                ).drop(columns=[f"call_off{off}_timestamp"])
            if not pblock.empty:
                merged = merged.merge(
                    pblock.add_prefix(f"put_off{off}_"),
                    left_on="timestamp",
                    right_on=f"put_off{off}_timestamp",
                    how="left",
                ).drop(columns=[f"put_off{off}_timestamp"])
    except Exception:
        pass

    expiry_map = (
        df[["timestamp", "expiryDate"]]
        .dropna(subset=["expiryDate"])
        .drop_duplicates("timestamp", keep="last")
    )
    if expiry_map.empty:
        expiry_hint = df["expiryDate"].dropna().head(1)
        if not expiry_hint.empty:
            expiry_map = pd.DataFrame(
                {"timestamp": merged["timestamp"], "expiryDate": expiry_hint.iloc[0]}
            )
    if not expiry_map.empty:
        expiry_map["expiryDate"] = pd.to_datetime(expiry_map["expiryDate"], errors="coerce")
        merged = pd.merge(merged, expiry_map, on="timestamp", how="left")
    else:
        merged["expiryDate"] = pd.NaT
    merged["expiryDate"] = pd.to_datetime(merged["expiryDate"], errors="coerce")
    if merged["expiryDate"].isna().all():
        trade_day = datetime.strptime(day_dir.name, "%Y-%m-%d")
        inferred = _weekly_expiry_for(trade_day, trading_days=trading_days)
        merged["expiryDate"] = pd.Timestamp(inferred)
    ts_naive = merged["timestamp"].dt.tz_localize(None)
    expiry_naive = merged["expiryDate"]
    merged["days_to_expiry"] = (expiry_naive.dt.normalize() - ts_naive.dt.normalize()).dt.days
    merged["next_week_expiry"] = merged["expiryDate"] + pd.Timedelta(days=7)
    merged["days_to_next_expiry"] = (
        (merged["next_week_expiry"].dt.normalize() - ts_naive.dt.normalize()).dt.days
    )
    merged["combined_premium_pct"] = (merged["atm_call_ltp"] + merged["atm_put_ltp"]) / merged["spot"].replace(0, np.nan)
    merged["iv_rank"] = _iv_rank(merged["atm_call_iv"])
    merged["iv_skew"] = _skew(merged["atm_call_iv"], merged["atm_put_iv"])
    merged["oi_skew"] = _skew(merged["atm_call_oi"], merged["atm_put_oi"])
    merged["volume_skew"] = _skew(
        merged["atm_call_volume"].replace(0, np.nan),
        merged["atm_put_volume"].replace(0, np.nan),
    )
    merged["source_date"] = day_dir.name

    merged = _augment_spot_features(merged)
    return merged


def _augment_spot_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    grouped = df.groupby("source_date")
    for periods in (1, 3, 5):
        df[f"spot_return_{periods}"] = grouped["spot"].transform(lambda s: s.pct_change(periods=periods))
    for span in (20, 50, 100):
        df[f"spot_ema_{span}"] = grouped["spot"].transform(lambda s: s.ewm(span=span, adjust=False).mean())
        df[f"spot_trend_{span}"] = (df["spot"] - df[f"spot_ema_{span}"]) / df[f"spot_ema_{span}"]
    df["spot_volatility"] = grouped["spot"].transform(lambda s: s.pct_change().rolling(20, min_periods=5).std())
    df["spot_intraday_range_pct"] = grouped["spot"].transform(lambda s: (s.cummax() - s.cummin()) / s.cummax().replace(0, np.nan))
    df["call_oi_change"] = grouped["atm_call_oi"].transform(lambda s: s.diff())
    df["put_oi_change"] = grouped["atm_put_oi"].transform(lambda s: s.diff())
    df["call_iv_change"] = grouped["atm_call_iv"].transform(lambda s: s.diff())
    df["put_iv_change"] = grouped["atm_put_iv"].transform(lambda s: s.diff())
    return df


def main() -> None:
    args = parse_args()
    if args.selectors:
        selectors = [s.strip() for s in args.selectors.split(",") if s.strip()]
    elif args.selector:
        selectors = [args.selector.strip()]
    else:
        selectors = ["ATM"]
    if not selectors:
        selectors = ["ATM"]
    if args.dates.strip().upper() == "ALL":
        dates = sorted([p.name for p in args.root.iterdir() if p.is_dir()])
    else:
        dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    trading_days = {
        datetime.strptime(day, "%Y-%m-%d").date()
        for day in dates
        if len(day) == 10
    }
    all_rows: List[pd.DataFrame] = []
    for day in dates:
        day_dir = args.root / day
        if not day_dir.exists():
            print(f"[make_intraday_dataset] skipping {day} (directory missing)")
            continue
        try:
            df = process_day(day_dir, selectors, args.tz, trading_days=trading_days)
        except Exception as exc:
            print(f"[make_intraday_dataset] failed for {day}: {exc}")
            continue
        if df is None or df.empty:
            print(f"[make_intraday_dataset] skipping {day} (no selector data)")
            continue
        all_rows.append(df)
        print(f"[make_intraday_dataset] processed {day}: {len(df)} rows")

    if not all_rows:
        print("[make_intraday_dataset] no data processed")
        return

    final = pd.concat(all_rows, ignore_index=True)
    final = final.sort_values("timestamp").reset_index(drop=True)
    day_levels = (
        final.groupby("source_date")["spot"]
        .agg(day_high="max", day_low="min", day_open="first", day_close="last")
        .sort_index()
    )
    shifted = day_levels.shift(1)
    prev_high = shifted["day_high"].to_dict()
    prev_low = shifted["day_low"].to_dict()
    prev_open = shifted["day_open"].to_dict()
    prev_close = shifted["day_close"].to_dict()
    range_series = (day_levels["day_high"] - day_levels["day_low"]).shift(1)
    prev_range = range_series.to_dict()
    final["prev_day_high"] = final["source_date"].map(prev_high)
    final["prev_day_low"] = final["source_date"].map(prev_low)
    final["prev_day_open"] = final["source_date"].map(prev_open)
    final["prev_day_close"] = final["source_date"].map(prev_close)
    final["prev_day_range"] = final["source_date"].map(prev_range)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(args.output, index=False)
    print(f"[make_intraday_dataset] wrote {len(final)} rows to {args.output}")


if __name__ == "__main__":
    main()
