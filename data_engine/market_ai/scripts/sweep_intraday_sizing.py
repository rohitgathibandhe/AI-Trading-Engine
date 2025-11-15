#!/usr/bin/env python3
"""
Sweep different position sizing presets for IntradayThetaScalp.

Example:
python3 data_engine/market_ai/scripts/sweep_intraday_sizing.py \
    --input reports/intraday_from_rolling_3m.csv \
    --strangle-qtys 1,2,3 \
    --spread-qtys 1,2 \
    --aggressive-triggers 900,1200 \
    --loss-pauses 0,-1200 \
    --output reports/intraday_sizing_sweep.json
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Iterable, List, Dict, Any

import pandas as pd

import sys

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DATA_ENGINE_ROOT = REPO_ROOT / "data_engine"
for path in (REPO_ROOT, DATA_ENGINE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from market_ai.modules.strategies.intraday_theta_scalp import run_backtest  # noqa: E402


def _parse_int_list(value: str, fallback: Iterable[int]) -> List[int]:
    if not value:
        return list(fallback)
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_float_list(value: str, fallback: Iterable[float]) -> List[float]:
    if not value:
        return list(fallback)
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep sizing presets for IntradayThetaScalp.")
    parser.add_argument("--input", type=Path, required=True, help="Intraday CSV file for replay.")
    parser.add_argument("--output", type=Path, default=Path("reports/intraday_sizing_sweep.json"))
    parser.add_argument("--strangle-qtys", help="Comma separated strangle qty values (default: 1,2,3).")
    parser.add_argument("--spread-qtys", help="Comma separated spread qty values (default: 1,2).")
    parser.add_argument("--max-qtys", help="Comma separated maximum qty caps (default: 3,4).")
    parser.add_argument("--aggressive-triggers", help="Comma separated triggers to add extra lot (default: 900,1200).")
    parser.add_argument("--aggressive-bonuses", help="Comma separated lot bonuses once trigger hit (default: 1,2).")
    parser.add_argument("--loss-pauses", help="Comma separated daily loss pause thresholds (default: 0,-1200).")
    parser.add_argument("--capital-pcts", help="Comma separated capital pct limits (default: 0.6,0.75).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)

    strangle_qtys = _parse_int_list(args.strangle_qtys or "", [1, 2, 3])
    spread_qtys = _parse_int_list(args.spread_qtys or "", [1, 2])
    max_qtys = _parse_int_list(args.max_qtys or "", [3, 4])
    aggressive_triggers = _parse_float_list(args.aggressive_triggers or "", [900.0, 1200.0])
    aggressive_bonuses = _parse_int_list(args.aggressive_bonuses or "", [1, 2])
    loss_pauses = _parse_float_list(args.loss_pauses or "", [0.0, -1200.0])
    capital_limits = _parse_float_list(args.capital_pcts or "", [0.6, 0.75])

    combos = list(
        itertools.product(
            strangle_qtys,
            spread_qtys,
            max_qtys,
            aggressive_triggers,
            aggressive_bonuses,
            loss_pauses,
            capital_limits,
        )
    )
    results: List[Dict[str, Any]] = []
    for idx, (sq, spread_q, max_q, trigger, bonus, loss_pause, capital_pct) in enumerate(combos, 1):
        cfg = {
            "position_sizing": {
                "strangle_qty": sq,
                "spread_qty": spread_q,
                "iron_qty": max(1, spread_q - 1),
                "base_qty": max(1, min(sq, spread_q)),
                "max_qty": max_q,
                "aggressive_trigger": trigger,
                "aggressive_bonus": bonus,
                "loss_pause_trigger": loss_pause,
                "capital_pct_limit": capital_pct,
            }
        }
        trades_df, metrics_df, summary = run_backtest(df, cfg)
        total_realized = summary.get("total_realized", 0.0)
        max_dd = summary.get("max_drawdown", 0.0)
        avg_pnl = total_realized / max(1, len(trades_df))
        results.append(
            {
                "strangle_qty": sq,
                "spread_qty": spread_q,
                "max_qty": max_q,
                "aggressive_trigger": trigger,
                "aggressive_bonus": bonus,
                "loss_pause_trigger": loss_pause,
                "capital_pct_limit": capital_pct,
                "total_realized": total_realized,
                "max_drawdown": max_dd,
                "avg_pnl_per_trade": avg_pnl,
                "entries": summary.get("entries", len(trades_df)),
                "closes": summary.get("closes", len(trades_df)),
            }
        )
        print(
            f"[sizing] {idx}/{len(combos)} sq={sq} spread={spread_q} max={max_q} "
            f"trigger={trigger} bonus={bonus} lossPause={loss_pause} cap={capital_pct} "
            f"-> PnL={total_realized:.1f} DD={max_dd:.1f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(f"[sizing] wrote {len(results)} combos to {args.output}")


if __name__ == "__main__":
    main()
