#!/usr/bin/env python3
"""
Run the weekly theta strangle backtest on a prepared intraday CSV.

Example:
python3 data_engine/market_ai/scripts/run_weekly_theta_backtest.py \
    --input reports/intraday_from_rolling_12m.csv \
    --qty 2 \
    --min-prev-range 0.003 \
    --pnl-target 6000 \
    --pnl-stop 4000 \
    --no-demand-prev-break
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

from market_ai.modules.strategies.weekly_theta_strangle import (  # noqa: E402
    run_backtest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WeeklyThetaStrangle backtest.")
    parser.add_argument("--input", type=Path, required=True, help="Intraday CSV (timestamp, atm_call_ltp, ...).")
    parser.add_argument("--output-dir", type=Path, default=Path("reports"), help="Directory to write trades/summary.")
    parser.add_argument("--qty", type=int, default=2, help="Number of short strangle lots.")
    parser.add_argument("--lot-size", type=int, default=50, help="Lot size per contract.")
    parser.add_argument("--entry-day", type=int, default=0, help="Entry weekday (0=Mon).")
    parser.add_argument("--exit-day", type=int, default=3, help="Preferred exit weekday (0=Mon).")
    parser.add_argument("--min-premium", type=float, default=0.016, help="Combined premium pct threshold.")
    parser.add_argument("--min-prev-range", type=float, default=0.003, help="Minimum prior day range pct.")
    parser.add_argument("--max-prev-range", type=float, default=0.05, help="Maximum prior day range pct.")
    parser.add_argument("--pnl-target", type=float, default=6_000.0, help="Weekly profit target (INR).")
    parser.add_argument("--pnl-stop", type=float, default=4_000.0, help="Weekly loss stop (INR).")
    parser.add_argument("--trailing-lock", type=float, default=0.25, help="Trailing give-back percentage.")
    parser.add_argument("--hard-exit-day", type=int, default=3, help="Force exit on/after this weekday.")
    parser.add_argument("--max-hold-days", type=int, default=5, help="Maximum holding days.")
    parser.add_argument("--demand-prev-break", action="store_true", help="Require prior-day breakout to enter.")
    parser.add_argument("--structure", default="STRANGLE", help="Structure type (STRANGLE or IRON_CONDOR).")
    parser.add_argument("--wing-offset", type=int, default=2, help="Wing offset (for iron condor).")
    parser.add_argument("--hybrid", action="store_true", help="Enable hybrid auto-selection (strangle vs condor).")
    parser.add_argument("--trend-threshold", type=float, default=0.02, help="Prev-range pct to treat as trend.")
    parser.add_argument("--condor-threshold", type=float, default=0.01, help="Prev-range pct to favor condor.")
    parser.add_argument("--oi-distance", type=float, default=0.01, help="Min distance (pct) from spot to OI walls for condor.")
    parser.add_argument("--event-calendar", type=Path, default=Path("data_engine/market_ai/state/events.json"), help="Event calendar JSON.")
    parser.add_argument("--skip-severities", default="high", help="Comma separated event severities to skip.")
    parser.add_argument("--expiry-offset-weeks", type=int, default=0, help="0 = front weekly, 1 = next weekly, etc.")
    parser.add_argument("--min-days-to-expiry", type=int, default=0, help="Require at least N days to target expiry.")
    parser.add_argument("--ml-exit-model", type=Path, help="Optional ML exit model path.")
    parser.add_argument("--ml-exit-threshold", type=float, default=0.6, help="Probability cutoff for ML exit.")
    parser.add_argument("--emit-plan", type=Path, help="Optional JSON file to emit plan summary.")
    parser.add_argument("--max-gap-pct", type=float, default=0.01, help="Skip entry if Monday gap > pct.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    entry_rules: Dict[str, Any] = {
        "min_premium_pct": args.min_premium,
        "min_prev_range_pct": args.min_prev_range,
        "max_prev_range_pct": args.max_prev_range,
        "demand_prev_break": args.demand_prev_break,
        "min_days_to_target_expiry": args.min_days_to_expiry,
        "max_gap_pct": args.max_gap_pct,
    }
    targets: Dict[str, Any] = {
        "pnl_target": args.pnl_target,
        "pnl_stop": args.pnl_stop,
        "trailing_lock_pct": args.trailing_lock,
        "hard_exit_day": args.hard_exit_day,
        "max_hold_days": args.max_hold_days,
    }
    cfg: Dict[str, Any] = {
        "lot_size": args.lot_size,
        "qty": args.qty,
        "entry_day": args.entry_day,
        "exit_day": args.exit_day,
        "entry_rules": entry_rules,
        "targets": targets,
        "structure": args.structure,
        "wing_offset": args.wing_offset,
        "hybrid_enabled": args.hybrid,
        "trend_range_threshold": args.trend_threshold,
        "condor_range_threshold": args.condor_threshold,
        "oi_distance_pct": args.oi_distance,
        "event_calendar_path": str(args.event_calendar) if args.event_calendar else None,
        "skip_event_severities": tuple(s.strip() for s in args.skip_severities.split(",") if s.strip()),
        "expiry_offset_weeks": args.expiry_offset_weeks,
        "ml_exit_model_path": str(args.ml_exit_model) if args.ml_exit_model else None,
        "ml_exit_min_prob": args.ml_exit_threshold,
    }

    trades_df, daily_df, summary = run_backtest(df, cfg)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trades_path = args.output_dir / "weekly_theta_trades.csv"
    summary_path = args.output_dir / "weekly_theta_summary.json"
    daily_log_path = args.output_dir / "weekly_theta_daily_log.csv"
    trades_df.to_csv(trades_path, index=False)
    if isinstance(daily_df, pd.DataFrame):
        daily_df.to_csv(daily_log_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"Wrote trades to {trades_path}")
    if isinstance(daily_df, pd.DataFrame):
        print(f"Wrote daily log to {daily_log_path}")
    print(f"Summary: {json.dumps(summary, indent=2)}")
    if args.emit_plan:
        plan = {
            "config": cfg,
            "summary": summary,
        }
        args.emit_plan.parent.mkdir(parents=True, exist_ok=True)
        args.emit_plan.write_text(json.dumps(plan, indent=2, default=str))
        print(f"Wrote plan to {args.emit_plan}")


if __name__ == "__main__":
    main()
