#!/usr/bin/env python3
"""
Run weekly spreads (bull put / bear call) backtest on intraday_from_rolling_latest.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[3]
for p in (ROOT, ROOT / "data_engine"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from market_ai.strategies.weekly_spreads import WeeklySpreadsEngine, SpreadConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Weekly spreads backtest (bull put / bear call).")
    parser.add_argument("--input", type=Path, default=ROOT / "reports" / "intraday_from_rolling_latest.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "weekly_spreads_trades.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    engine = WeeklySpreadsEngine(SpreadConfig())
    trades = engine.run_backtest(df)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trades.to_csv(args.output, index=False)
    summary = {
        "entries": len(trades),
        "total_pnl": float(trades["pnl"].sum()) if not trades.empty else 0.0,
        "avg_pnl": float(trades["pnl"].mean()) if not trades.empty else 0.0,
    }
    print("Summary:", json.dumps(summary, indent=2))
    print(f"Wrote trades to {args.output}")


if __name__ == "__main__":
    main()
