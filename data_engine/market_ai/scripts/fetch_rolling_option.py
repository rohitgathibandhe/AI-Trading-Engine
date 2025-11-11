#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI to backfill Dhan rolling option data into state/rolling_option/ partitions.
"""
from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, List

if __package__ is None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


import pandas as pd

from market_ai.modules.data_fetch.option_chain_ingestor import ChainConfig, OptionChainIngestor
from market_ai.modules.data_fetch.optionchain_repo import OptionChainRepo
from market_ai.modules.data_fetch.dhan_rolling_option import RollingOptionConfig, RollingOptionIngestor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Dhan rolling option history")
    parser.add_argument("underlying", help="Underlying symbol (e.g. NIFTY)")
    parser.add_argument("--mode", choices=["chain", "rolling"], default="chain", help="Source: live option-chain snapshot (chain) or charts/rollingoption (rolling)")
    parser.add_argument("--seg", default="NSE_FNO", help="Underlying segment (default NSE_FNO)")
    parser.add_argument("--security-id", type=int, required=True, help="Numeric securityId from Dhan")
    parser.add_argument("--start", help="Start date YYYY-MM-DD", required=True)
    parser.add_argument("--end", help="End date YYYY-MM-DD", required=True)
    parser.add_argument("--expiry", help="Target expiry date YYYY-MM-DD (defaults to inferred next monthly)")
    parser.add_argument("--out", type=Path, help="Output directory (default state/rolling_option_ladder)")
    parser.add_argument("--interval", default="5", help="Rolling option interval (default 5 minutes or '1' for rollingoption)")
    parser.add_argument("--throttle", type=float, default=0.3, help="Throttle seconds between DHAN calls (default 0.3, unused in fallback)")
    parser.add_argument("--limit", type=int, default=3, help="Maximum expiries to fetch per day (default 3)")
    parser.add_argument("--db", type=Path, help="SQLite repo output (default state/option_chain_history.sqlite)")
    parser.add_argument("--page-limit", type=int, default=500, help="Page size for rollingoption API (default 500)")
    parser.add_argument(
        "--strike-selector",
        dest="strike_selectors",
        action="append",
        help="Strike selector to request (repeat for multiple, e.g., ATM, ATM+2, ATM-2).",
    )
    parser.add_argument(
        "--option-type",
        dest="option_types",
        action="append",
        choices=["CALL", "PUT"],
        help="Option side to request (repeat to fetch both CALL and PUT).",
    )
    parser.add_argument(
        "--required-data",
        dest="required_data",
        action="append",
        help="Optional list of requiredData fields for DHAN rollingoption (repeat or comma-separate).",
    )
    return parser.parse_args()


def _collect_list(values: List[str] | None) -> List[str] | None:
    if not values:
        return None
    collected: List[str] = []
    for val in values:
        for piece in str(val).split(","):
            piece = piece.strip()
            if piece:
                collected.append(piece)
    return collected or None


def _sanitize_expiry_list(expiries: Iterable[str], limit: int | None) -> List[str]:
    today = date.today()
    clean: List[str] = []
    for exp in expiries:
        try:
            d = datetime.strptime(str(exp), "%Y-%m-%d").date()
        except Exception:
            continue
        if d >= today:
            clean.append(d.isoformat())
    clean.sort()
    if limit is not None and limit > 0:
        clean = clean[:limit]
    return clean


def run_rolling_option(args: argparse.Namespace, out_dir: Path) -> None:
    """Fetch rollingoption history using Dhan's /v2/charts/rollingoption."""
    start_ts = datetime.strptime(args.start, "%Y-%m-%d")
    end_ts = datetime.strptime(args.end, "%Y-%m-%d")
    ingestor = RollingOptionIngestor(out_dir=out_dir)
    strike_selectors = _collect_list(getattr(args, "strike_selectors", None))
    option_types = _collect_list(getattr(args, "option_types", None))
    required_data = _collect_list(getattr(args, "required_data", None))
    cfg = RollingOptionConfig(
        underlying=args.underlying,
        segment=args.seg,
        security_id=args.security_id,
        start=start_ts,
        end=end_ts,
        expiry=args.expiry,
        interval=args.interval,
        limit_per_page=args.page_limit,
        strike_selectors=strike_selectors or None,
        option_types=option_types or None,
        required_data=required_data or None,
    )
    parquet_paths = ingestor.fetch_range(cfg)
    if parquet_paths:
        print(f"[rollingoption] wrote {len(parquet_paths)} parquet files under {ingestor.out_dir}")
    else:
        print("[rollingoption] no data returned for the requested window")


def main() -> None:
    args = parse_args()
    out_dir = args.out or (Path(__file__).resolve().parents[2] / "state" / "rolling_option_ladder")
    out_dir.mkdir(parents=True, exist_ok=True)
    start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end, "%Y-%m-%d").date()

    if args.mode == "rolling":
        run_rolling_option(args, out_dir)
        return

    cfg = ChainConfig(underlying_id=int(args.security_id), underlying_seg=str(args.seg))
    ingestor = OptionChainIngestor(cfg)
    repo_path = args.db or (Path(__file__).resolve().parents[2] / "state" / "option_chain_history.sqlite")
    repo = OptionChainRepo(str(repo_path))

    if args.expiry:
        target_expiries = [args.expiry]
    else:
        try:
            listed = ingestor.list_expiries().dropna().tolist()
        except Exception as exc:
            raise SystemExit(f"Could not list expiries from Dhan: {exc}") from exc
        target_expiries = _sanitize_expiry_list(listed, args.limit)
        if not target_expiries:
            raise SystemExit("No upcoming expiries resolved from Dhan; aborting.")

    written_files = 0
    repo_rows: list[dict] = []
    day = start_date
    while day <= end_date:
        if day != date.today():
            print(f"[warn] Fallback option-chain cannot time-travel; storing live snapshot under {day}")
        for expiry in target_expiries:
            try:
                df = ingestor.fetch(expiry)
            except Exception as exc:
                print(f"Failed to fetch chain for expiry {expiry} on {day}: {exc}")
                continue
            if df.empty:
                print(f"No data returned for expiry {expiry}; skipping.")
                continue

            df = df.assign(as_of_date=day.isoformat(), expiry=expiry)
            parquet_df = df.drop(columns=["raw_call", "raw_put"], errors="ignore").copy()
            out_path = out_dir / f"{day.isoformat()}__{expiry}.parquet"
            parquet_df.to_parquet(out_path, index=False)
            written_files += 1

            for row in df.to_dict(orient="records"):
                repo_rows.append(
                    {
                        "underlying_id": args.security_id,
                        "underlying_seg": args.seg,
                        "as_of_date": row["as_of_date"],
                        "expiry": row["expiry"],
                        "strike": row["strike"],
                        "ce": row.get("raw_call") or {},
                        "pe": row.get("raw_put") or {},
                    }
                )
        day += timedelta(days=1)

    if repo_rows:
        repo.upsert_many(repo_rows)
    print(f"Saved {written_files} parquet snapshots under {out_dir}")
    print(f"Wrote {len(repo_rows)} rows to {repo_path}")


if __name__ == "__main__":
    main()
