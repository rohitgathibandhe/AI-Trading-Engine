#!/usr/bin/env python3
"""
Snapshot the Dhan option chain at a fixed cadence and export intraday-ready
rows for the IntradayThetaScalp backtests.

Usage:
    python3 data_engine/market_ai/scripts/collect_intraday_chain.py \
        --expiry 2024-11-28 --duration-min 30 --interval-sec 60 \
        --output reports/intraday_chain_sample.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional
from zoneinfo import ZoneInfo

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DATA_ENGINE_ROOT = REPO_ROOT / "data_engine"
for path in (REPO_ROOT, DATA_ENGINE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from market_ai.modules.data_fetch.option_chain_ingestor import OptionChainIngestor, ChainConfig  # noqa: E402


def _safe_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    if val in (None, "", [], {}, "nan"):
        return default
    try:
        return float(val)
    except Exception:
        return default


def _skew_value(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    denom = abs(a) + abs(b)
    if denom == 0:
        return None
    return (a - b) / denom


def _iv_rank(value: Optional[float], series: pd.Series) -> Optional[float]:
    clean = series.dropna()
    if value is None or clean.empty:
        return None
    lo = clean.min()
    hi = clean.max()
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect option-chain snapshots for intraday backtests.")
    parser.add_argument("--underlying-id", type=int, default=13, help="DHAN security ID (NIFTY=13).")
    parser.add_argument("--underlying-seg", default="NSE_FNO", help="Segment code (NSE_FNO, IDX_I, etc.).")
    parser.add_argument("--expiry", required=True, help="Expiry date (YYYY-MM-DD) or tag (e.g., weekly).")
    parser.add_argument("--interval-sec", type=float, default=60.0, help="Seconds between snapshots.")
    parser.add_argument("--duration-min", type=float, default=30.0, help="Duration to collect data (minutes).")
    parser.add_argument("--output", type=Path, default=Path("reports/intraday_chain_snapshot.csv"))
    parser.add_argument("--timezone", default="Asia/Kolkata")
    return parser.parse_args()


def compute_snapshot(df: pd.DataFrame, now: datetime) -> Optional[Dict[str, Any]]:
    if df is None or df.empty:
        return None
    spot_series = df["underlying_ltp"].dropna()
    approx_spot = _safe_float(spot_series.iloc[0]) if not spot_series.empty else None
    if approx_spot is None:
        approx_spot = _safe_float(df["strike"].median())

    if approx_spot is None:
        return None

    idx = (df["strike"] - approx_spot).abs().idxmin()
    row = df.loc[idx]
    call_ltp = _safe_float(row.get("ltp_ce"), 0.0) or 0.0
    put_ltp = _safe_float(row.get("ltp_pe"), 0.0) or 0.0
    call_iv = _safe_float(row.get("IV_CE"))
    call_delta = _safe_float(row.get("delta_ce"))
    put_delta = _safe_float(row.get("delta_pe"))
    call_oi = _safe_float(row.get("oi_ce"))
    put_oi = _safe_float(row.get("oi_pe"))
    call_vol = _safe_float(row.get("volume_ce"))
    put_vol = _safe_float(row.get("volume_pe"))

    iv_rank = _iv_rank(call_iv, df["IV_CE"])
    combined_premium_pct = None
    if approx_spot:
        combined_premium_pct = (call_ltp + put_ltp) / approx_spot

    payload = {
        "timestamp": now.isoformat(),
        "spot": approx_spot,
        "atm_call_strike": row.get("strike"),
        "atm_put_strike": row.get("strike"),
        "atm_call_ltp": call_ltp,
        "atm_put_ltp": put_ltp,
        "atm_call_delta": call_delta,
        "atm_put_delta": put_delta,
        "atm_call_iv": call_iv,
        "atm_put_iv": _safe_float(row.get("IV_PE")),
        "atm_call_oi": call_oi,
        "atm_put_oi": put_oi,
        "atm_call_volume": call_vol,
        "atm_put_volume": put_vol,
        "iv_skew": _skew_value(_safe_float(row.get("IV_CE")), _safe_float(row.get("IV_PE"))),
        "oi_skew": _skew_value(call_oi, put_oi),
        "volume_skew": _skew_value(call_vol, put_vol),
        "iv_rank": iv_rank,
        "combined_premium_pct": combined_premium_pct,
        "raw_call": json.dumps(row.get("raw_call", {}), default=str),
        "raw_put": json.dumps(row.get("raw_put", {}), default=str),
    }
    return payload


def main() -> None:
    args = parse_args()
    tz = ZoneInfo(args.timezone)
    ingestor = OptionChainIngestor(ChainConfig(args.underlying_id, args.underlying_seg))
    end_time = datetime.now(tz) + timedelta(minutes=args.duration_min)
    rows: List[Dict[str, Any]] = []
    print(f"[collector] Starting capture until {end_time.isoformat()}; interval={args.interval_sec}s")
    try:
        while datetime.now(tz) <= end_time:
            ts = datetime.now(tz)
            try:
                chain_df = ingestor.fetch(args.expiry)
            except Exception as exc:
                print(f"[collector] fetch failed: {exc}")
                time.sleep(args.interval_sec)
                continue
            snapshot = compute_snapshot(chain_df, ts)
            if snapshot:
                rows.append(snapshot)
                print(
                    f"[collector] {ts.strftime('%H:%M:%S')} spot={snapshot['spot']:.2f} "
                    f"premium={snapshot['combined_premium_pct'] or 0:.4f}"
                )
            else:
                print(f"[collector] {ts.strftime('%H:%M:%S')} snapshot unavailable (insufficient data)")
            time.sleep(args.interval_sec)
    except KeyboardInterrupt:
        print("[collector] interrupted; writing partial data...")

    if not rows:
        print("[collector] no data captured")
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    file_exists = args.output.exists()
    fieldnames = list(rows[0].keys())
    with args.output.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)
    print(f"[collector] wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
