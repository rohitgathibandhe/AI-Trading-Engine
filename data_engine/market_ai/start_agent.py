#!/usr/bin/env python3
"""
Refactored agent entrypoint.

Uses the new StrategySelector framework (regime-aware) instead of the legacy
weekly_theta_strangle loop. Warn-only by default; flip WARN_ONLY=false to allow
order placement once verified.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, time as dtime
from pathlib import Path
from typing import List, Optional

from logging.handlers import RotatingFileHandler

# ── Path/setup ────────────────────────────────────────────────────────────────
THIS_FILE = Path(__file__).resolve()
ENGINE_DIR = THIS_FILE.parent
DATA_ENGINE_DIR = ENGINE_DIR.parent
PROJECT_ROOT = DATA_ENGINE_DIR.parent
STATE_DIR = ENGINE_DIR / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
AGENT_LOG = STATE_DIR / "agent.log"

for p in (ENGINE_DIR, DATA_ENGINE_DIR, PROJECT_ROOT, ENGINE_DIR / "strategies"):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

from market_ai.dhan_wrapper import DhanWrapper  # noqa: E402
from market_ai.strategies import (  # noqa: E402
    StrategySelector,
    MarketSnapshot,
    OptionLeg,
    RiskConfig,
    StrategyType,
)

# ── Logging ──────────────────────────────────────────────────────────────────
formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
handler = RotatingFileHandler(AGENT_LOG, maxBytes=5 * 1024 * 1024, backupCount=5, encoding="utf-8")
handler.setFormatter(formatter)
console = logging.StreamHandler(sys.stdout)
console.setFormatter(formatter)
root = logging.getLogger()
root.setLevel(logging.INFO)
root.handlers = [handler, console]
log = logging.getLogger("start_agent")

# ── Helpers ─────────────────────────────────────────────────────────────────
def _as_bool(val: Optional[str], default: bool) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def _map_positions(rows: list) -> List[OptionLeg]:
    legs: List[OptionLeg] = []
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
            qty = abs(int(net))
            side = "SELL" if net < 0 else "BUY"
            expiry = r.get("expiryDate") or r.get("expiry")
            expiry_date = datetime.fromisoformat(str(expiry)).date() if expiry else datetime.now().date()
            legs.append(
                OptionLeg(
                    symbol="NIFTY",
                    expiry=expiry_date,
                    strike=float(r.get("strikePrice") or r.get("strike") or 0),
                    option_type="CALL" if "C" in str(r.get("optionType") or r.get("option_type") or "C").upper() else "PUT",
                    side=side,
                    quantity=qty,
                    entry_price=float(r.get("avgPrice") or r.get("avg_price") or 0.0),
                    security_id=str(r.get("securityId") or r.get("security_id") or ""),
                )
            )
        except Exception:
            continue
    return legs


def _map_chain(chain_raw: dict, expiry) -> List[dict]:
    rows: List[dict] = []
    for strike, legs in (chain_raw or {}).items():
        for opt_name, opt_data in (legs or {}).items():
            if not isinstance(opt_data, dict):
                continue
            rows.append(
                {
                    "expiry": expiry,
                    "option_type": "CE" if opt_name.lower().startswith("ce") else "PE",
                    "strike": float(strike),
                    "ltp": opt_data.get("last_price") or opt_data.get("ltp") or opt_data.get("close"),
                    "delta": opt_data.get("delta"),
                    "security_id": opt_data.get("securityId") or opt_data.get("security_id"),
                }
            )
    return rows


def _compute_mtm(legs: List[OptionLeg]) -> float:
    mtm = 0.0
    for leg in legs:
        if leg.current_ltp is None:
            continue
        pnl = (leg.entry_price - leg.current_ltp) if leg.side == "SELL" else (leg.current_ltp - leg.entry_price)
        mtm += pnl * leg.quantity
    return mtm


def _fetch_expiry(dw: DhanWrapper):
    try:
        expiries = dw.get_optionchain_expirylist("IDX_I", 13)
        return expiries[0] if expiries else datetime.now().date()
    except Exception:
        return datetime.now().date()


def build_market_snapshot(dw: DhanWrapper) -> MarketSnapshot:
    spot = float(dw.get_ltp_once("IDX_I", 13) or 0.0)
    now = datetime.now()
    return MarketSnapshot(
        symbol="NIFTY",
        spot=spot,
        candles_15m=[],  # TODO: wire real candle data
        yesterday_high=spot,
        yesterday_low=spot,
        india_vix=0.0,
        now=now,
    )


def main() -> None:
    warn_only = _as_bool(os.getenv("WARN_ONLY"), True)
    poll_sec = float(os.getenv("POLL_SEC", "10"))
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
    log.info("Regime-aware agent started (warn_only=%s)", warn_only)

    while True:
        try:
            market = build_market_snapshot(dw)
            expiry = _fetch_expiry(dw)
            positions_raw = dw.get_positions_live()
            legs = _map_positions(positions_raw)
            chain_raw = dw.get_option_chain(13, "IDX_I", expiry)
            chain = _map_chain(chain_raw, expiry)
            basket_mtm = _compute_mtm(legs)

            decision = selector.decide(
                market=market,
                option_chain=chain,
                expiry=expiry,
                risk=risk,
                current_positions=legs,
                basket_mtm=basket_mtm,
            )
            log.info("Decision=%s strategy=%s reason=%s", decision.action_type, decision.strategy_type.value, decision.reason)
            if not warn_only and decision.action_type.startswith("OPEN"):
                for leg in decision.legs_to_open:
                    side = "BUY" if leg.side == "BUY" else "SELL"
                    resp = dw.place_order(
                        side=side,
                        exchange_seg="NSE_FNO",
                        security_id=leg.security_id or 0,
                        quantity=leg.quantity,
                        product_type="MIS",
                        order_type="MARKET",
                    )
                    log.info("order placed resp=%s", resp)
            if not warn_only and decision.action_type.startswith("CLOSE"):
                for leg in decision.legs_to_close:
                    side = "BUY" if leg.side == "SELL" else "SELL"
                    resp = dw.place_order(
                        side=side,
                        exchange_seg="NSE_FNO",
                        security_id=leg.security_id or 0,
                        quantity=leg.quantity,
                        product_type="MIS",
                        order_type="MARKET",
                    )
                    log.info("close order resp=%s", resp)
        except Exception as exc:
            log.exception("Agent loop error: %s", exc)
        time.sleep(poll_sec)


if __name__ == "__main__":
    main()
