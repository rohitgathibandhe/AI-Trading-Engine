#!/usr/bin/env python3
"""
Utility to backtest the IntradayThetaScalp prototype on a CSV file.

Expected CSV columns (tz-aware timestamps preferred):
    timestamp, atm_call_ltp, atm_put_ltp[, atm_call_strike, atm_put_strike, ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DATA_ENGINE_ROOT = REPO_ROOT / "data_engine"
for path in (REPO_ROOT, DATA_ENGINE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from market_ai.modules.strategies.intraday_theta_scalp import run_backtest, IntradayConfig  # noqa: E402


def _parse_time(value: str):
    from datetime import datetime

    return datetime.strptime(value, "%H:%M").time()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run intraday theta scalp backtest on CSV data.")
    parser.add_argument("--input", type=Path, required=True, help="CSV with intraday bars (timestamp, atm_call_ltp, atm_put_ltp).")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"), help="Folder to write trades/metrics summaries.")
    parser.add_argument("--flatten-time", default="15:10", help="Flatten cut-off (HH:MM).")
    parser.add_argument("--min-profit", type=float, default=1_000.0, help="Daily profit target before trail.")
    parser.add_argument("--trail-pct", type=float, default=0.25, help="Give-back percent once target hit.")
    parser.add_argument("--hard-loss", type=float, default=5_000.0, help="Hard daily loss limit.")
    parser.add_argument("--timezone", default="Asia/Kolkata", help="Timezone for timestamps and trade window.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    if "timestamp" not in df.columns:
        raise ValueError("CSV must contain a 'timestamp' column.")
    daily_cfg: Dict[str, Any] = {
        "min_profit": args.min_profit,
        "trail_pct": args.trail_pct,
        "hard_loss_per_day": args.hard_loss,
        "flatten_time": _parse_time(args.flatten_time),
    }
    cfg: Dict[str, Any] = {
        "daily_target": daily_cfg,
        "timezone": args.timezone,
    }
    trades_df, metrics_df, summary = run_backtest(df, cfg)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trades_path = args.output_dir / "intraday_theta_trades.csv"
    metrics_path = args.output_dir / "intraday_theta_metrics.csv"
    summary_path = args.output_dir / "intraday_theta_summary.json"

    trades_df.to_csv(trades_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"Wrote trades to {trades_path}")
    print(f"Wrote metrics to {metrics_path}")
    print(f"Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
