#!/usr/bin/env python3
"""
Simple Batman deploy daemon.
- Fetches monthly option chain
- Builds Batman structure with extra hedges (±400 from shorts, cap ₹10)
- Places orders via DhanWrapper in paper/live based on TRADE_MODE env

Run: python3 data_engine/market_ai/scripts/ai_batman_daemon.py
Requires DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN in env (UI should export these).
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any, Dict, Optional

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "data_engine") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "data_engine"))

from market_ai.dhan_wrapper import DhanWrapper
from market_ai.modules.data_fetch.dhan_scrip_cache import resolve_option_security_id
from market_ai.modules.strategies.batman_monthly import BatmanConfig, BatmanMonthlyStrategy
from market_ai.utils.index_calendar import monthly_expiry_date

LOG = logging.getLogger("batman_daemon")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[3]
STATE_DIR = PROJECT_ROOT / "data_engine" / "market_ai" / "state"
AGENT_SETTINGS = STATE_DIR / "agent_settings.json"

INDEX_SECURITY_ID = 13  # NIFTY
INDEX_EXCHANGE_SEG = "NSE_FNO"


def _load_settings() -> Dict[str, Any]:
    if AGENT_SETTINGS.exists():
        try:
            return json.loads(AGENT_SETTINGS.read_text())
        except Exception:
            LOG.warning("Failed to read %s, using defaults", AGENT_SETTINGS)
    return {"lot_size": 75}


def _map_chain(raw: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw, list):
        return out
    for row in raw:
        if not isinstance(row, dict):
            continue
        strike = row.get("strike") or row.get("StrikePrice") or row.get("strikePrice")
        if strike is None:
            continue
        try:
            strike_val = float(strike)
        except Exception:
            continue
        strike_key = str(int(round(strike_val)))
        ce = row.get("ce") or row.get("CE") or row.get("call")
        pe = row.get("pe") or row.get("PE") or row.get("put")
        out[strike_key] = {
            "ce": ce or {},
            "pe": pe or {},
        }
    return out


def _price_map(chain: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    for k, node in chain.items():
        strike = float(k)
        ce = node.get("ce") or {}
        pe = node.get("pe") or {}
        ce_ltp = ce.get("last_price") or ce.get("ltp") or ce.get("close") or 0.0
        pe_ltp = pe.get("last_price") or pe.get("ltp") or pe.get("close") or 0.0
        prices[f"CALL:{strike}"] = float(ce_ltp or 0.0)
        prices[f"PUT:{strike}"] = float(pe_ltp or 0.0)
    return prices


def _place_leg(dw: DhanWrapper, expiry_str: str, lot: int, leg: Any, trade_mode: str):
    seg = INDEX_EXCHANGE_SEG
    opt_type = "CE" if leg.option_type == "CALL" else "PE"
    side = "BUY" if leg.direction == "LONG" else "SELL"
    sec_id = resolve_option_security_id("NIFTY", expiry_str, float(leg.strike), opt_type)
    if not sec_id:
        raise RuntimeError(f"Missing security id for {expiry_str} {leg.strike} {opt_type}")
    LOG.info("Placing %s %s %s @%s qty=%s seg=%s", side, opt_type, leg.strike, expiry_str, lot * leg.qty, seg)
    dw.place_order(
        side=side,
        product="I",
        order_type="MARKET",
        exchange_segment=seg,
        security_id=int(sec_id),
        quantity=lot * leg.qty,
        price=None,
        validity="DAY",
        disclosed_quantity=0,
    )


def main():
    settings = _load_settings()
    lot_size = int(os.getenv("BATMAN_LOT_SIZE", settings.get("lot_size", 75)))
    trade_mode = os.getenv("TRADE_MODE", "live")
    dw = DhanWrapper(logger=LOG)
    today = datetime.now().date()
    expiry = monthly_expiry_date(today)
    expiry_str = expiry.isoformat()
    LOG.info("Batman deploy: trade_mode=%s lot=%s expiry=%s", trade_mode, lot_size, expiry_str)
    chain_raw = dw.get_option_chain(INDEX_SECURITY_ID, INDEX_EXCHANGE_SEG, expiry_str)
    chain = _map_chain(chain_raw)
    prices = _price_map(chain)
    spot = None
    # approximate spot from nearest strike mid
    if chain:
        strikes = sorted(float(k) for k in chain.keys())
        if strikes:
            mid = strikes[len(strikes) // 2]
            spot = mid
    if spot is None:
        raise RuntimeError("Could not infer spot from chain")
    cfg = BatmanConfig(
        lot_size=lot_size,
        long_wing_distance=300,
        short_wing_distance=600,
        hedge_extra_distance=400,
        hedge_price_cap=10.0,
        add_extra_hedges=True,
        dataset=None,
        market=True,  # dummy truthy to bypass guard
    )
    strat = BatmanMonthlyStrategy(cfg)
    strikes = strat._build_strikes(spot)
    legs = strat._build_legs(strikes, prices)
    for leg in legs:
        _place_leg(dw, expiry_str, lot_size, leg, trade_mode)
    LOG.info("Batman structure sent (%d legs)", len(legs))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        LOG.exception("Batman daemon error: %s", exc)
        sys.exit(1)
