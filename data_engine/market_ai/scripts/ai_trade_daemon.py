#!/usr/bin/env python3
"""
Minimal daemon runner that wires StrategySelector in warn-only mode.
Fetches market snapshot, option chain, positions; decides actions; logs decisions.
Place actual orders only when warn_only=False.
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path
from typing import List

SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[3]
for p in (ROOT, ROOT / "data_engine"):
    p = str(p)
    if p not in sys.path:
        sys.path.insert(0, p)

from market_ai.dhan_wrapper import DhanWrapper  # noqa: E402
from market_ai.strategies import (
    MarketSnapshot,
    OptionLeg,
    RiskConfig,
    StrategySelector,
    TrendSide,
)  # noqa: E402


def map_positions_to_legs(rows: list) -> List[OptionLeg]:
    legs = []
    for r in rows or []:
        try:
            exch = str(r.get("exchangeSegment") or r.get("exchange_seg") or "").upper()
            if not exch.startswith("NSE_FNO"):
                continue
            symbol = r.get("tradingSymbol") or r.get("symbol") or ""
            if "NIFTY" not in symbol.upper():
                continue
            net = float(r.get("netQty") or r.get("netqty") or 0)
            if net == 0:
                continue
            side = "SELL" if net < 0 else "BUY"
            expiry = r.get("expiryDate") or r.get("expiry")
            legs.append(
                OptionLeg(
                    symbol="NIFTY",
                    expiry=datetime.fromisoformat(str(expiry)).date() if expiry else datetime.now().date(),
                    strike=float(r.get("strikePrice") or r.get("strike") or 0),
                    option_type=("CALL" if "C" in str(r.get("optionType") or r.get("option_type") or "C").upper() else "PUT"),
                    side=side,
                    quantity=abs(int(net)),
                    entry_price=float(r.get("avgPrice") or r.get("avg_price") or 0.0),
                    security_id=str(r.get("securityId") or r.get("security_id") or ""),
                )
            )
        except Exception:
            continue
    return legs


def map_chain(raw_chain: dict, expiry) -> List[dict]:
    out = []
    data = raw_chain or {}
    for strike, legs in data.items():
        for opt_key, opt_val in legs.items():
            if not isinstance(opt_val, dict):
                continue
            out.append(
                {
                    "expiry": expiry,
                    "option_type": "CE" if opt_key.lower().startswith("ce") else "PE",
                    "strike": float(strike),
                    "ltp": opt_val.get("last_price") or opt_val.get("ltp") or opt_val.get("close"),
                    "delta": opt_val.get("delta"),
                    "security_id": opt_val.get("securityId") or opt_val.get("security_id"),
                }
            )
    return out


def main() -> None:
    warn_only = True
    dw = DhanWrapper()
    selector = StrategySelector(symbol="NIFTY", lot_size=75)
    risk = RiskConfig(
        max_intraday_loss=-3000,
        intraday_target=4000,
        allow_carry_forward=False,
        max_carry_days=0,
        vix_carry_threshold=12.0,
        last_entry_time=dtime(14, 45),
        force_exit_time=dtime(15, 15),
    )
    while True:
        try:
            now = datetime.now()
            # market snapshot: requires you to populate candles/yhigh/ylow/vix from your data source
            candles = []  # TODO: fill with dicts
            y_high = 0.0
            y_low = 0.0
            vix = 0.0
            spot = float(dw.get_ltp_once("IDX_I", 13) or 0.0)
            market = MarketSnapshot(symbol="NIFTY", spot=spot, candles_15m=candles, yesterday_high=y_high, yesterday_low=y_low, india_vix=vix, now=now)

            positions_raw = dw.get_positions_live()
            legs = map_positions_to_legs(positions_raw)
            basket_mtm = 0.0  # TODO: compute from positions/ltps

            # expiry resolution: take next Tuesday via Dhan optionchain list
            expiry_list = dw.get_optionchain_expirylist("IDX_I", 13)
            expiry = expiry_list[0] if expiry_list else now.date()

            chain_raw = dw.get_option_chain(13, "IDX_I", expiry)
            chain = map_chain(chain_raw, expiry)

            decision = selector.decide(
                market=market,
                option_chain=chain,
                expiry=expiry,
                risk=risk,
                current_positions=legs,
                basket_mtm=basket_mtm,
            )
            print(f"[{now.isoformat()}] decision={decision.action_type} reason={decision.reason}")
            if not warn_only:
                # Place orders according to decision.legs_to_open/close
                pass
        except Exception as exc:
            print(f"[daemon] error: {exc}")
        time.sleep(10)


if __name__ == "__main__":
    main()
