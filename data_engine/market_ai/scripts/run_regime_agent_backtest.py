#!/usr/bin/env python3
"""
Offline backtest for the regime-aware agent (StrategySelector + hedged strangle).

It replays a historical feature CSV (default: reports/intraday_from_rolling_latest.csv),
derives a lightweight option-chain snapshot from the provided strikes (ATM +/- offsets),
and feeds those into the same StrategySelector used in production. Trades are opened/closed
exactly as the live agent would, with MTM and realized P&L tracked over time.

Outputs:
  - reports/regime_agent_trades.csv (chronological trade log)
  - reports/regime_agent_summary.json (aggregate metrics)
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, date as dt_date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import sys

SCRIPT_ROOT = Path(__file__).resolve()
REPO_ROOT = SCRIPT_ROOT.parents[3]
for candidate in (REPO_ROOT, REPO_ROOT / "data_engine"):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from market_ai.strategies import (
    StrategySelector,
    MarketSnapshot,
    RiskConfig,
    OptionLeg,
)

DATA_ROOT = REPO_ROOT
DEFAULT_CSV = DATA_ROOT / "reports" / "intraday_from_rolling_latest.csv"
DEFAULT_TRADES = DATA_ROOT / "reports" / "regime_agent_trades.csv"
DEFAULT_SUMMARY = DATA_ROOT / "reports" / "regime_agent_summary.json"


CALL_PREFIXES = [
    ("atm_call", "CE"),
    ("call_atm_plus2", "CE"),
    ("call_atm_minus2", "CE"),
    ("call_atm_plus4", "CE"),
    ("call_atm_minus4", "CE"),
    ("call_atm_plus6", "CE"),
    ("call_atm_minus6", "CE"),
]

PUT_PREFIXES = [
    ("atm_put", "PE"),
    ("put_atm_plus2", "PE"),
    ("put_atm_minus2", "PE"),
    ("put_atm_plus4", "PE"),
    ("put_atm_minus4", "PE"),
    ("put_atm_plus6", "PE"),
    ("put_atm_minus6", "PE"),
]


def _coerce_float(val) -> Optional[float]:
    try:
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None


def _row_to_chain(row: pd.Series, expiry: dt_date) -> List[Dict[str, float]]:
    """Build a minimal option-chain list from the available prefixes."""
    rows: List[Dict[str, float]] = []

    def add(prefix: str, opt_type: str) -> None:
        strike = _coerce_float(row.get(f"{prefix}_strike"))
        ltp = _coerce_float(row.get(f"{prefix}_ltp"))
        delta = _coerce_float(row.get(f"{prefix}_delta"))
        if strike is None or ltp is None:
            return
        rows.append(
            {
                "expiry": expiry,
                "option_type": "CE" if opt_type == "CE" else "PE",
                "strike": strike,
                "ltp": ltp,
                "delta": delta,
                "security_id": f"{opt_type}_{int(round(strike))}",
            }
        )

    for prefix, opt in CALL_PREFIXES:
        add(prefix, opt)
    for prefix, opt in PUT_PREFIXES:
        add(prefix, opt)
    return rows


def _as_snapshot(row: pd.Series) -> MarketSnapshot:
    now = pd.to_datetime(row["timestamp"]).to_pydatetime()
    spot = float(row["spot"])
    prev_high = _coerce_float(row.get("prev_day_high")) or spot
    prev_low = _coerce_float(row.get("prev_day_low")) or spot
    vix = _coerce_float(row.get("spot_volatility")) or 0.0
    return MarketSnapshot(
        symbol="NIFTY",
        spot=spot,
        candles_15m=[],
        yesterday_high=prev_high,
        yesterday_low=prev_low,
        india_vix=vix,
        now=now,
    )


def _leg_key(leg: OptionLeg) -> Tuple[str, float]:
    opt = "CE" if leg.option_type.value.startswith("CALL") else "PE"
    return opt, float(leg.strike)


def _update_leg_ltps(legs: List[OptionLeg], chain_map: Dict[Tuple[str, float], float]) -> None:
    for leg in legs:
        ltp = chain_map.get(_leg_key(leg))
        if ltp is not None:
            leg.current_ltp = ltp


def _lookup_ltp(chain_map: Dict[Tuple[str, float], float], leg: OptionLeg) -> Optional[float]:
    return chain_map.get(_leg_key(leg))


def _compute_mtm(legs: List[OptionLeg]) -> float:
    mtm = 0.0
    for leg in legs:
        if leg.current_ltp is None:
            continue
        if leg.side == "SELL":
            pnl = (leg.entry_price - leg.current_ltp) * leg.quantity
        else:
            pnl = (leg.current_ltp - leg.entry_price) * leg.quantity
        mtm += pnl
    return mtm


def run_backtest(csv_path: Path, trades_out: Path, summary_out: Path) -> None:
    df = pd.read_csv(csv_path)
    df = df.sort_values("timestamp")
    selector = StrategySelector(symbol="NIFTY", lot_size=75)
    risk = RiskConfig(
        max_intraday_loss=-3000,
        intraday_target=4000,
        allow_carry_forward=False,
        max_carry_days=0,
        vix_carry_threshold=12.0,
        last_entry_time=datetime.strptime("14:45", "%H:%M").time(),
        force_exit_time=datetime.strptime("15:15", "%H:%M").time(),
    )

    open_legs: List[OptionLeg] = []
    realized = 0.0
    trades: List[Dict[str, Any]] = []
    equity_curve: List[Tuple[datetime, float]] = []

    for _, row in df.iterrows():
        try:
            expiry = pd.to_datetime(row["expiryDate"]).date()
        except Exception:
            continue
        chain_rows = _row_to_chain(row, expiry)
        if not chain_rows:
            continue
        chain_map = {(r["option_type"], r["strike"]): r["ltp"] for r in chain_rows}
        _update_leg_ltps(open_legs, chain_map)
        market = _as_snapshot(row)
        basket_mtm = _compute_mtm(open_legs)
        decision = selector.decide(
            market=market,
            option_chain=chain_rows,
            expiry=expiry,
            risk=risk,
            current_positions=open_legs,
            basket_mtm=basket_mtm,
        )
        timestamp = market.now
        action = decision.action_type
        if action.startswith("OPEN"):
            for leg in decision.legs_to_open:
                ltp = _lookup_ltp(chain_map, leg)
                if ltp is None:
                    continue
                new_leg = OptionLeg(
                    symbol=leg.symbol,
                    expiry=leg.expiry,
                    strike=leg.strike,
                    option_type=leg.option_type,
                    side=leg.side,
                    quantity=leg.quantity,
                    entry_price=ltp,
                    security_id=leg.security_id,
                    current_ltp=ltp,
                )
                open_legs.append(new_leg)
                trades.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "event": "OPEN",
                        "action": action,
                        "strike": leg.strike,
                        "option_type": leg.option_type.value,
                        "side": leg.side.value,
                        "quantity": leg.quantity,
                        "price": ltp,
                    }
                )
        elif action.startswith("CLOSE"):
            closing = decision.legs_to_close if decision.legs_to_close else open_legs.copy()
            for leg in closing:
                match = next(
                    (
                        existing
                        for existing in open_legs
                        if existing.strike == leg.strike
                        and existing.option_type == leg.option_type
                        and existing.side == leg.side
                    ),
                    None,
                )
                if not match:
                    continue
                ltp = _lookup_ltp(chain_map, match)
                if ltp is None:
                    continue
                if match.side == "SELL":
                    pnl = (match.entry_price - ltp) * match.quantity
                else:
                    pnl = (ltp - match.entry_price) * match.quantity
                realized += pnl
                trades.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "event": "CLOSE",
                        "action": action,
                        "strike": match.strike,
                        "option_type": match.option_type.value,
                        "side": match.side.value,
                        "quantity": match.quantity,
                        "price": ltp,
                        "pnl": pnl,
                    }
                )
                open_legs.remove(match)
        equity_curve.append((timestamp, realized + _compute_mtm(open_legs)))

    pd.DataFrame(trades).to_csv(trades_out, index=False)
    summary = {
        "trades": len(trades),
        "realized_pnl": realized,
        "open_positions": len(open_legs),
        "final_equity": equity_curve[-1][1] if equity_curve else realized,
    }
    summary_out.write_text(json.dumps(summary, indent=2))
    print(f"Backtest completed: {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regime-aware agent backtest")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input historical features CSV")
    parser.add_argument("--trades-out", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    run_backtest(args.csv, args.trades_out, args.summary_out)


if __name__ == "__main__":
    main()
