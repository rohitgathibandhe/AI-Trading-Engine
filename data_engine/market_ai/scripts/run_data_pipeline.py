#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily data pipeline:
  1) Pull latest rolling-option slabs from Dhan (lookback window configurable)
  2) Rebuild feature_history + engineered datasets for training
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import List

SCRIPT_PATH = Path(__file__).resolve()
SCRIPTS_DIR = SCRIPT_PATH.parent
ENGINE_DIR = SCRIPTS_DIR.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling-option ingest + feature refresh pipeline")
    parser.add_argument("underlying", help="Underlying symbol (e.g. NIFTY)")
    parser.add_argument("--security-id", type=int, required=True, help="Numeric securityId from Dhan")
    parser.add_argument("--seg", default="NSE_FNO", help="Exchange segment (default NSE_FNO)")
    parser.add_argument("--lookback-days", type=int, default=30, help="Days to backfill per run (default 30)")
    parser.add_argument("--interval", default="1", help="Rolling option interval (default 1 minute)")
    parser.add_argument(
        "--strike-selector",
        dest="strike_selectors",
        action="append",
        help="Strike selector(s) to fetch (repeat or comma-separate)",
    )
    parser.add_argument(
        "--option-type",
        dest="option_types",
        action="append",
        choices=["CALL", "PUT"],
        help="Option sides to request (repeat to include both)",
    )
    parser.add_argument(
        "--required-data",
        dest="required_data",
        action="append",
        help="Rollingoption requiredData fields (default uses fetch_rolling_option defaults)",
    )
    parser.add_argument(
        "--rolling-root",
        type=Path,
        default=ENGINE_DIR / "state" / "rolling_option",
        help="Where rolling-option parquet folders live",
    )
    parser.add_argument(
        "--feature-history",
        type=Path,
        default=ENGINE_DIR / "state" / "feature_history.csv",
        help="Feature history CSV path",
    )
    parser.add_argument(
        "--engineered-out",
        type=Path,
        default=ENGINE_DIR / "state" / "feature_history_engineered.parquet",
        help="Engineered feature parquet path",
    )
    parser.add_argument(
        "--target-out",
        type=Path,
        default=ENGINE_DIR / "state" / "feature_targets.parquet",
        help="Target parquet path",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=ENGINE_DIR / "state" / "feature_history_summary.json",
        help="Feature summary JSON",
    )
    parser.add_argument("--horizon", type=int, default=5, help="Target horizon rows (default 5)")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip DHAN fetch step")
    parser.add_argument("--skip-refresh", action="store_true", help="Skip feature refresh step")
    parser.add_argument("--skip-training", action="store_true", help="Skip selector retraining step")
    parser.add_argument("--selector-model", type=Path, default=ENGINE_DIR / "state" / "strategy_selector_model.json", help="Selector model output")
    parser.add_argument("--selector-report", type=Path, default=ENGINE_DIR / "state" / "strategy_selector_training_summary.json", help="Selector training summary output")
    parser.add_argument("--selector-min-samples", type=int, default=50, help="Minimum samples per strategy")
    parser.add_argument("--selector-val-fraction", type=float, default=0.2, help="Validation split fraction")
    return parser.parse_args()


def _collect(values: List[str] | None) -> List[str] | None:
    if not values:
        return None
    out: List[str] = []
    for val in values:
        for piece in str(val).split(","):
            piece = piece.strip()
            if piece:
                out.append(piece)
    return out or None


def _run(cmd: List[str]) -> None:
    print(f"[pipeline] running: {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    today = date.today()
    lookback = max(1, int(args.lookback_days))
    start_day = today - timedelta(days=lookback - 1)
    selectors = _collect(args.strike_selectors)
    option_types = _collect(args.option_types)
    required = _collect(args.required_data)

    fetch_script = SCRIPTS_DIR / "fetch_rolling_option.py"
    refresh_script = SCRIPTS_DIR / "refresh_feature_history.py"
    selector_script = SCRIPTS_DIR / "strategy_selector_scaffold.py"

    if not args.skip_fetch:
        cmd = [
            sys.executable,
            str(fetch_script),
            args.underlying,
            "--mode",
            "rolling",
            "--seg",
            args.seg,
            "--security-id",
            str(args.security_id),
            "--start",
            start_day.isoformat(),
            "--end",
            today.isoformat(),
            "--interval",
            args.interval,
            "--out",
            str(args.rolling_root),
        ]
        if selectors:
            for sel in selectors:
                cmd += ["--strike-selector", sel]
        if option_types:
            for side in option_types:
                cmd += ["--option-type", side]
        if required:
            for field in required:
                cmd += ["--required-data", field]
        _run(cmd)
    else:
        print("[pipeline] skipping rolling-option fetch (per flag)")

    if not args.skip_refresh:
        cmd = [
            sys.executable,
            str(refresh_script),
            "--rolling-root",
            str(args.rolling_root),
            "--feature-history",
            str(args.feature_history),
            "--engineered-out",
            str(args.engineered_out),
            "--target-out",
            str(args.target_out),
            "--summary-out",
            str(args.summary_out),
            "--horizon",
            str(args.horizon),
        ]
        _run(cmd)
    else:
        print("[pipeline] skipping feature refresh (per flag)")

    if not args.skip_training:
        cmd = [
            sys.executable,
            str(selector_script),
            "--history",
            str(args.feature_history),
            "--model-out",
            str(args.selector_model),
            "--report-out",
            str(args.selector_report),
            "--horizon",
            str(args.horizon),
            "--min-samples",
            str(args.selector_min_samples),
            "--val-fraction",
            str(args.selector_val_fraction),
        ]
        _run(cmd)
    else:
        print("[pipeline] skipping selector training (per flag)")


if __name__ == "__main__":
    main()
