#!/usr/bin/env python3
"""
Parameter sweep harness for IntradayThetaScalp.

Example:
python3 data_engine/market_ai/scripts/sweep_intraday_params.py \
    --input reports/intraday_from_rolling_2m.csv \
    --min-premiums 0.016,0.018,0.02 \
    --trail-pcts 0.2,0.3 \
    --risk-stop-pcts 0.3,0.4 \
    --per-leg-stop-pcts 0.35,0.45 \
    --output reports/intraday_param_sweep.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable, List, Dict, Any

import pandas as pd

from market_ai.modules.strategies.intraday_theta_scalp import run_backtest


def _parse_float_list(value: str, fallback: Iterable[float]) -> List[float]:
    if not value:
        return list(fallback)
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep intraday strategy parameters.")
    parser.add_argument("--input", type=Path, required=True, help="Intraday CSV (timestamp, atm_call_ltp, ...).")
    parser.add_argument("--output", type=Path, default=Path("reports/intraday_param_sweep.json"))
    parser.add_argument("--min-premiums", help="Comma separated combined premium pct thresholds.")
    parser.add_argument("--trail-pcts", help="Comma separated trailing give-back percents.")
    parser.add_argument("--risk-stop-pcts", help="Comma separated risk stop percentages.")
    parser.add_argument("--per-leg-stop-pcts", help="Comma separated per-leg stop percentages.")
    parser.add_argument("--min-profit", type=float, default=2_000.0, help="Daily profit target.")
    parser.add_argument("--hard-loss", type=float, default=5_000.0, help="Daily hard loss.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    min_prem_values = _parse_float_list(args.min_premiums or "", [0.016, 0.018, 0.020])
    trail_values = _parse_float_list(args.trail_pcts or "", [0.2, 0.3])
    risk_stop_values = _parse_float_list(args.risk_stop_pcts or "", [0.3, 0.4])
    per_leg_values = _parse_float_list(args.per_leg_stop_pcts or "", [0.35, 0.45])

    combos = list(itertools.product(min_prem_values, trail_values, risk_stop_values, per_leg_values))
    results: List[Dict[str, Any]] = []
    for idx, (min_prem, trail_pct, risk_stop, per_leg_stop) in enumerate(combos, 1):
        cfg = {
            "entry_filter": {"min_premium_pct": min_prem},
            "daily_target": {"min_profit": args.min_profit, "trail_pct": trail_pct, "hard_loss_per_day": args.hard_loss},
            "risk_stop_pct": risk_stop,
            "per_leg_stop_pct": per_leg_stop,
        }
        trades_df, metrics_df, summary = run_backtest(df, cfg)
        summary.update(
            {
                "min_premium_pct": min_prem,
                "trail_pct": trail_pct,
                "risk_stop_pct": risk_stop,
                "per_leg_stop_pct": per_leg_stop,
                "entries": summary.get("entries", len(trades_df)),
                "closes": summary.get("closes", len(trades_df)),
            }
        )
        results.append(summary)
        print(
            f"[sweep] {idx}/{len(combos)} minPrem={min_prem:.3f} trail={trail_pct:.2f} "
            f"riskStop={risk_stop:.2f} legStop={per_leg_stop:.2f} -> PnL={summary.get('total_realized')}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"[sweep] wrote {len(results)} rows to {args.output}")


if __name__ == "__main__":
    main()

