#!/usr/bin/env python3
"""
Generate the next-week plan using the WeeklyThetaStrangle backtest.

Creates a JSON plan (structure, strikes, qty) and optionally appends an intent
into state/order_intents.jsonl for paper/live hand-off.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DATA_ENGINE_ROOT = REPO_ROOT / "data_engine"
for path in (REPO_ROOT, DATA_ENGINE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from market_ai.modules.strategies.weekly_theta_strangle import run_backtest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate next weekly plan/intents.")
    parser.add_argument("--input", type=Path, required=True, help="Intraday CSV (e.g., reports/intraday_from_rolling_latest.csv).")
    parser.add_argument("--plan-path", type=Path, default=Path("state/weekly_plan.json"), help="Where to write the plan JSON.")
    parser.add_argument("--intent-path", type=Path, default=Path("data_engine/market_ai/state/order_intents.jsonl"), help="Intent JSONL file.")
    parser.add_argument("--qty", type=int, default=2)
    parser.add_argument("--min-prev-range", type=float, default=0.003)
    parser.add_argument("--max-prev-range", type=float, default=0.05)
    parser.add_argument("--pnl-target", type=float, default=6_000.0)
    parser.add_argument("--pnl-stop", type=float, default=4_000.0)
    parser.add_argument("--hybrid", action="store_true", help="Enable hybrid strangle/condor mode.")
    parser.add_argument("--structure", default="STRANGLE")
    parser.add_argument("--wing-offset", type=int, default=4)
    parser.add_argument("--trend-threshold", type=float, default=0.02)
    parser.add_argument("--condor-threshold", type=float, default=0.012)
    parser.add_argument("--oi-distance", type=float, default=0.012)
    parser.add_argument("--event-calendar", type=Path, default=Path("data_engine/market_ai/state/events.json"))
    parser.add_argument("--expiry-offset", type=int, default=0, help="0 = front weekly, 1 = next weekly, etc.")
    parser.add_argument("--min-days-to-expiry", type=int, default=0, help="Require at least N days to target expiry.")
    parser.add_argument("--ml-exit-model", type=Path, help="Optional path to trained ML exit model.")
    parser.add_argument("--ml-exit-threshold", type=float, default=0.6, help="Probability to trigger ML exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    cfg: Dict[str, Any] = {
        "qty": args.qty,
        "entry_rules": {
            "min_prev_range_pct": args.min_prev_range,
            "max_prev_range_pct": args.max_prev_range,
            "min_days_to_target_expiry": args.min_days_to_expiry,
            "max_gap_pct": args.max_gap_pct,
        },
        "targets": {
            "pnl_target": args.pnl_target,
            "pnl_stop": args.pnl_stop,
        },
        "structure": args.structure,
        "wing_offset": args.wing_offset,
        "hybrid_enabled": args.hybrid,
        "trend_range_threshold": args.trend_threshold,
        "condor_range_threshold": args.condor_threshold,
        "oi_distance_pct": args.oi_distance,
        "event_calendar_path": str(args.event_calendar) if args.event_calendar else None,
        "expiry_offset_weeks": args.expiry_offset,
        "ml_exit_model_path": str(args.ml_exit_model) if args.ml_exit_model else None,
        "ml_exit_min_prob": args.ml_exit_threshold,
    }
    trades_df, daily_df, summary = run_backtest(df, cfg)
    plan = {
        "generated_at": datetime.now().isoformat(),
        "dataset": str(args.input),
        "config": cfg,
        "summary": summary,
    }
    if isinstance(daily_df, pd.DataFrame) and not daily_df.empty:
        latest_daily = daily_df.iloc[-1].to_dict()
        plan["latest_daily_snapshot"] = {k: (None if pd.isna(v) else v) for k, v in latest_daily.items()}
    openings = trades_df[trades_df["action"] == "OPEN"].copy()
    if not openings.empty:
        latest_open = openings.iloc[-1].to_dict()
        latest_open = {k: (None if pd.isna(v) else v) for k, v in latest_open.items()}
        plan["latest_signal"] = latest_open
        append_intent(latest_open, args.intent_path, cfg.get("qty", 1))
    args.plan_path.parent.mkdir(parents=True, exist_ok=True)
    args.plan_path.write_text(json.dumps(plan, indent=2, default=str))
    print(f"Wrote weekly plan to {args.plan_path}")
    if not openings.empty:
        print("Latest signal:", json.dumps(plan["latest_signal"], default=str, indent=2))


def append_intent(signal: Dict[str, Any], intent_path: Path, qty: int) -> None:
    intent = {
        "timestamp": datetime.now().isoformat(),
        "strategy": "weekly_theta",
        "structure": signal.get("structure", "STRANGLE"),
        "call_strike": signal.get("call_strike"),
        "put_strike": signal.get("put_strike"),
        "long_call_strike": signal.get("long_call_strike"),
        "long_put_strike": signal.get("long_put_strike"),
        "call_ltp": signal.get("call_ltp"),
        "put_ltp": signal.get("put_ltp"),
        "qty": qty,
        "raw": signal,
    }
    intent_path.parent.mkdir(parents=True, exist_ok=True)
    with intent_path.open("a") as fh:
        fh.write(json.dumps(intent, default=str) + "\n")


if __name__ == "__main__":
    main()
    parser.add_argument("--max-gap-pct", type=float, default=0.01)
