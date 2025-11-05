#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI to backfill Dhan rolling option data into state/rolling_option/ partitions.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

if __package__ is None:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from market_ai.modules.data_fetch.dhan_api import make_client
from market_ai.modules.data_fetch.dhan_rolling_option import RollingOptionIngestor, RollingOptionConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Dhan rolling option history")
    parser.add_argument("underlying", help="Underlying symbol (e.g. NIFTY)")
    parser.add_argument("--seg", default="NSE_FNO", help="Underlying segment (default NSE_FNO)")
    parser.add_argument("--security-id", type=int, required=True, help="Numeric securityId from Dhan")
    parser.add_argument("--start", help="Start date YYYY-MM-DD", required=True)
    parser.add_argument("--end", help="End date YYYY-MM-DD", required=True)
    parser.add_argument("--expiry", help="Filter by expiry date YYYY-MM-DD")
    parser.add_argument("--out", type=Path, help="Output directory (default state/rolling_option)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = RollingOptionConfig(
        underlying=args.underlying,
        segment=args.seg,
        security_id=args.security_id,
        start=datetime.strptime(args.start, "%Y-%m-%d"),
        end=datetime.strptime(args.end, "%Y-%m-%d"),
        expiry=args.expiry,
    )
    ingestor = RollingOptionIngestor(client=make_client(), out_dir=args.out)
    paths = ingestor.fetch_range(cfg)
    print(f"Written {len(paths)} parquet files", paths)


if __name__ == "__main__":
    main()
