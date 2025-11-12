#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter sweep harness for MonthlyStrangleWithWeeklyHedge.

Examples:
python3 data_engine/market_ai/scripts/sweep_strangle_params.py \
  --input data_engine/predictions_latest.csv \
  --underlying-id 13 --underlying-seg NSE_FNO \
  --profit-targets 0.3,0.4 \
  --stop-losses 1.2,1.4 \
  --hedge-deltas 0.10,0.15
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable, List, Dict, Any

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
ENGINE_DIR = SCRIPT_PATH.parents[1]
PACKAGE_ROOT = ENGINE_DIR.parents[0]

if __package__ is None:
    import sys
    sys.path.insert(0, str(PACKAGE_ROOT))

from market_ai.ai_strategy_backtest import _build_market  # type: ignore  # noqa: E402
from market_ai.modules.backtest_engine_adapter import run_backtest as adapter_run  # noqa: E402


def _parse_float_list(value: str, fallback: Iterable[float]) -> List[float]:
    if not value:
        return list(fallback)
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return list(fallback)
    return [float(p) for p in parts]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep exit/hedge parameters for the strangle strategy")
    parser.add_argument("--input", type=Path, default=Path("data_engine/predictions_latest.csv"), help="Predictions CSV (must have 'date')")
    parser.add_argument("--underlying-id", type=int, default=13)
    parser.add_argument("--underlying-seg", default="NSE_FNO")
    parser.add_argument("--lot-size", type=int, default=50)
    parser.add_argument("--capital", type=float, default=500000)
    parser.add_argument("--market", choices=["rolling", "live", "historical"], default="rolling")
    parser.add_argument("--profit-targets", help="Comma separated list of profit target (e.g., 0.3,0.4)")
    parser.add_argument("--stop-losses", help="Comma separated list of combined SL (e.g., 1.2,1.4)")
    parser.add_argument("--hedge-deltas", help="Comma separated list of hedge delta triggers (e.g., 0.10,0.12)")
    parser.add_argument("--hedge-prices", help="Comma separated list of max hedge price (defaults 35)")
    parser.add_argument("--hedge-distances", help="Comma separated list of roll distance (points)")
    parser.add_argument("--min-entry-credit", type=float, default=0.0, help="Minimum per-leg LTP to consider at entry")
    parser.add_argument("--min-delta-abs", type=float, default=0.05, help="Minimum |delta| to consider at entry")
    parser.add_argument("--output", type=Path, default=Path("reports/strangle_param_sweep.json"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    market = _build_market(args.market, args.underlying_seg)

    pt_vals = _parse_float_list(args.profit_targets or "", [0.30, 0.35, 0.40])
    sl_vals = _parse_float_list(args.stop_losses or "", [1.2, 1.3, 1.4])
    hedge_delta_vals = _parse_float_list(args.hedge_deltas or "", [0.10, 0.12, 0.15])
    hedge_price_vals = _parse_float_list(args.hedge_prices or "", [35.0])
    hedge_distance_vals = _parse_float_list(args.hedge_distances or "", [200.0])

    combos = list(itertools.product(pt_vals, sl_vals, hedge_delta_vals, hedge_price_vals, hedge_distance_vals))
    results: List[Dict[str, Any]] = []

    for idx, (pt, sl, hedge_delta, hedge_price, hedge_dist) in enumerate(combos, 1):
        cfg = {
            "strategy": "monthly_strangle_with_weekly_hedge",
            "capital": args.capital,
            "underlying_id": args.underlying_id,
            "underlying_seg": args.underlying_seg,
            "lot_size": args.lot_size,
            "profit_target_pct": pt,
            "combined_stop_loss_pct": sl,
            "hedge_delta_max": hedge_delta,
            "hedge_price_max": hedge_price,
            "hedge_roll_distance": hedge_dist,
            "min_entry_credit": args.min_entry_credit,
            "min_delta_abs": args.min_delta_abs,
            "market": market,
        }
        print(f"[sweep] {idx}/{len(combos)} PT={pt} SL={sl} HedgeΔ={hedge_delta} Price≤{hedge_price} Dist={hedge_dist}")
        trades_df, metrics_df, summary = adapter_run(df, cfg)
        summary = dict(summary)
        summary.update({
            "profit_target_pct": pt,
            "combined_stop_loss_pct": sl,
            "hedge_delta": hedge_delta,
            "hedge_price_max": hedge_price,
            "hedge_roll_distance": hedge_dist,
            "n_trades": int(summary.get("n_trades", len(trades_df))),
        })
        results.append(summary)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"[sweep] wrote {len(results)} results to {args.output}")


if __name__ == "__main__":
    main()
