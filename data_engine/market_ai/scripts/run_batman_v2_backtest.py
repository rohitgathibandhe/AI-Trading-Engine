"""
Quick Batman V2 (double-ratio) backtest scaffold.

This uses the rolling option cache under data_engine/state/rolling_option/.
Greeks are not present in the cache, so deltas are synthesized from moneyness.
The simulation is coarse and paper-only:
  - Pick 20–25 delta legs (1 long + 2 shorts per side) if available.
  - Net credit check.
  - MTM across available timestamps; exits on:
        * hard loss (default -3% of capital)
        * both-side delta stress
        * DTE <= 3
        * crude BE breach (short strikes +/- 300)
  - Writes trades to reports/batman_v2_backtest/batman_v2_trades.csv
    and summary json.

This is intentionally minimal; it does NOT implement rolls/adjustments yet.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from market_ai.strategies.batman_double_ratio import (
    BatmanConfig,
    BatmanStrategy,
)

ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = ROOT / "state"
ROLLING_DIR = STATE_DIR / "rolling_option"
REPORT_DIR = ROOT / "reports" / "batman_v2_backtest"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def synth_delta(row: Dict) -> Optional[float]:
    try:
        spot = float(row.get("spot") or 0)
        strike = float(row.get("strike") or 0)
        if spot <= 0 or strike <= 0:
            return None
        diff_pts = strike - spot
        delta_step = diff_pts / 600.0  # flatten slope for synthetic delta
        if (row.get("option_type") or "").upper() == "CE":
            return max(0.05, min(0.95, 0.25 - delta_step))
        else:
            return -max(0.05, min(0.95, 0.25 + delta_step))
    except Exception:
        return None


def main():
    windows = sorted(ROLLING_DIR.glob("rolling_*_CALL.json"))
    trades = []
    cfg = BatmanConfig()
    strategy = BatmanStrategy(cfg)

    def _to_rows(block: Dict, opt_type: str) -> List[Dict]:
        if not block:
            return []
        keys = [k for k in block.keys() if isinstance(block[k], list)]
        if not keys:
            return []
        n = min(len(block[k]) for k in keys)
        rows = []
        for i in range(n):
            row = {k: block[k][i] for k in keys}
            row["option_type"] = opt_type
            rows.append(row)
        return rows

    for call_file in windows:
        base = call_file.name.replace("_CALL.json", "")
        put_file = ROLLING_DIR / f"{base}_PUT.json"
        if not put_file.exists():
            continue
        with call_file.open() as f:
            call_block = json.load(f).get("data", {}).get("ce", {})
        with put_file.open() as f:
            put_block = json.load(f).get("data", {}).get("pe", {})
        # Normalize records into chain list
        chain = _to_rows(call_block, "CE") + _to_rows(put_block, "PE")
        if not chain:
            continue

        # Entry on first timestamp
        entries = sorted(set(r.get("timestamp") for r in chain if r.get("timestamp")))
        if not entries:
            continue
        entry_ts = min(entries)
        entry_date = datetime.fromtimestamp(entry_ts).date()
        expiry_guess = entry_date + timedelta(days=7)
        # inject expiry into rows for selection
        for r in chain:
            r.setdefault("expiry", expiry_guess)

        # derive spot
        entry_spot = next((r.get("spot") for r in chain if r.get("timestamp") == entry_ts and r.get("spot")), None)
        if entry_spot is None:
            # fallback to first nonzero spot anywhere
            entry_spot = next((r.get("spot") for r in chain if r.get("spot")), 0.0)

        # run entry via strategy
        entered = strategy.maybe_enter(
            as_of=datetime.fromtimestamp(entry_ts),
            spot=entry_spot,
            expiry=expiry_guess,
            vix=cfg.min_vix + 1,  # mock VIX high enough
            gap_pct=0.0,
            monthly_range_pos=0.5,
            chain=chain,
        )
        if not entered:
            continue

        # Simulate over days
        sorted_ts = sorted(entries)
        exit_reason = None
        exit_pnl = 0.0
        for ts in sorted_ts:
            as_of = datetime.fromtimestamp(ts)
            dte = (expiry_guess - as_of.date()).days
            # build chain snapshot for this ts
            ts_chain = []
            for r in chain:
                if r.get("timestamp") == ts:
                    r["delta"] = r.get("delta") if r.get("delta") is not None else synth_delta(r)
                    ts_chain.append(r)
            spot_vals = [r.get("spot") for r in ts_chain if r.get("spot")]
            spot = spot_vals[0] if spot_vals else 0.0
            reason = strategy.on_tick(
                as_of=as_of,
                spot=spot,
                vix=cfg.min_vix + 1,
                adx=0.0,
                chain=ts_chain,
            )
            if reason is not None or (strategy.state and not strategy.state.is_active()):
                exit_reason = reason or "CLOSED"
                exit_pnl = strategy.state.pnl_unrealized if strategy.state else 0.0
                break

        if exit_reason is None:
            exit_reason = "TIME_EXIT"
            exit_pnl = strategy.state.pnl_unrealized if strategy.state else 0.0
        legs = strategy.state.all_legs() if strategy.state else []
        short_call = legs[4].strike if len(legs) >= 5 else None
        short_put = legs[1].strike if len(legs) >= 2 else None
        credit = 0.0
        if legs:
            credit = sum(
                l.entry_price for l in legs if not l.is_long
            ) - sum(l.entry_price for l in legs if l.is_long)
        trades.append(
            {
                "entry_date": entry_date.isoformat(),
                "exit_reason": exit_reason,
                "expiry": expiry_guess.isoformat(),
                "short_call": short_call,
                "short_put": short_put,
                "entry_credit": credit,
                "pnl": exit_pnl,
            }
        )
        # reset state for next window
        strategy.state = None

    trades_df = pd.DataFrame(trades)
    trades_path = REPORT_DIR / "batman_v2_trades.csv"
    trades_df.to_csv(trades_path, index=False)
    summary = {
        "trades": len(trades_df),
        "pnl_sum": trades_df["pnl"].sum() if not trades_df.empty else 0.0,
        "pnl_mean": trades_df["pnl"].mean() if not trades_df.empty else 0.0,
    }
    (REPORT_DIR / "batman_v2_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote trades to {trades_path}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()
