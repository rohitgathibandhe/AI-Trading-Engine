#!/usr/bin/env python3
"""
Batman deploy daemon (IDX_I option chain, paper-safe).
- Targets ±300/±500 strikes (rounded to nearest 50) from spot.
- Logs per-leg LTP and net credit; skips live orders in TRADE_MODE=paper.
"""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "data_engine") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "data_engine"))

from market_ai.dhan_wrapper import DhanWrapper
from market_ai.modules.data_fetch.dhan_scrip_cache import resolve_option_security_id
from market_ai.utils.index_calendar import monthly_expiry_date

LOG = logging.getLogger("batman_daemon")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

INDEX_SECURITY_ID = 13
INDEX_EXCHANGE_SEG = "IDX_I"
STATE_DIR = PROJECT_ROOT / "data_engine" / "market_ai" / "state"
AGENT_SETTINGS = STATE_DIR / "agent_settings.json"


def _load_settings() -> Dict[str, Any]:
    if AGENT_SETTINGS.exists():
        try:
            return json.loads(AGENT_SETTINGS.read_text())
        except Exception:
            pass
    return {
        "lot_size": 75,
        "nifty_lot_size": 75,
        "batman_long_offset": 300.0,
        "batman_short_offset": 500.0,
        "batman_qty_long": 1,
        "batman_qty_short": 3,
        "nifty_expiry_weekday": "Tuesday",
        "expiry_shift_if_holiday": True,
        "holiday_list": [],
        "batman_weekly_hedge_enabled": True,
        "batman_weekly_hedge_price_cap": 12.0,
        "batman_weekly_hedge_min_distance": 300.0,
    }


def _coerce_chain(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:
            return {}
    return raw if isinstance(raw, dict) else {}


def _map_chain(raw: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw, (dict, list)):
        return out
    if isinstance(raw, dict):
        oc = raw.get("data", {}).get("oc") or raw.get("oc")
        if isinstance(oc, dict):
            for k, node in oc.items():
                out[str(k)] = {"ce": node.get("ce") or {}, "pe": node.get("pe") or {}}
            return out
    if isinstance(raw, list):
        for row in raw:
            if not isinstance(row, dict):
                continue
            strike = row.get("strike") or row.get("StrikePrice") or row.get("strikePrice")
            if strike is None:
                continue
            try:
                strike_key = str(int(round(float(strike))))
            except Exception:
                continue
            out[strike_key] = {"ce": row.get("ce") or {}, "pe": row.get("pe") or {}}
    return out


def _price_map(chain: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    for k, node in chain.items():
        strike = float(k)
        ce = node.get("ce") or {}
        pe = node.get("pe") or {}
        prices[f"CALL:{strike}"] = float(ce.get("last_price") or ce.get("ltp") or ce.get("close") or 0.0)
        prices[f"PUT:{strike}"] = float(pe.get("last_price") or pe.get("ltp") or pe.get("close") or 0.0)
    return prices


def _nearest(strikes: Dict[float, float], target: float) -> float:
    if not strikes:
        return target
    best = None
    for s, p in strikes.items():
        if p > 0 and (best is None or abs(s - target) < abs(best - target)):
            best = s
    if best is None:
        best = min(strikes.keys(), key=lambda x: abs(x - target))
    return best


def _round_strike(x: float, step: float = 100.0) -> float:
    return round(x / step) * step


def _place_leg(dw: DhanWrapper, expiry_str: str, lot: int, leg: Any, trade_mode: str):
    seg = INDEX_EXCHANGE_SEG
    opt_type = "CE" if leg.option_type == "CALL" else "PE"
    side = "BUY" if leg.direction == "LONG" else "SELL"
    if trade_mode == "paper":
        LOG.info("[PAPER] Skip order %s %s %s", side, opt_type, leg.strike)
        return
    sec_id = resolve_option_security_id("NIFTY", expiry_str, float(leg.strike), opt_type)
    if not sec_id:
        raise RuntimeError(f"Missing security id for {expiry_str} {leg.strike} {opt_type}")
    dw.place_order(
        side=side,
        product_type="INTRADAY",
        order_type="MARKET",
        exchange_seg=seg,
        security_id=int(sec_id),
        quantity=lot * leg.qty,
        validity="DAY",
    )


def _weekday_to_int(value) -> int | None:
    if isinstance(value, int):
        return value if 0 <= value <= 6 else None
    if isinstance(value, str):
        name = value.strip().lower()
        mapping = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }
        if name in mapping:
            return mapping[name]
        try:
            num = int(name)
            if 0 <= num <= 6:
                return num
        except Exception:
            return None
    return None


def _parse_holidays(raw) -> set[date]:
    if raw is None:
        return set()
    if isinstance(raw, str):
        raw = [h.strip() for h in raw.replace(",", "\n").splitlines() if h.strip()]
    out = set()
    if isinstance(raw, list):
        for item in raw:
            try:
                clean = str(item).split("T", 1)[0].split(" ", 1)[0]
                out.add(datetime.fromisoformat(clean).date())
            except Exception:
                continue
    return out


def _select_expiry(dw: DhanWrapper, settings: Dict[str, Any]) -> date:
    today = date.today()
    target_weekday = _weekday_to_int(settings.get("nifty_expiry_weekday", "Tuesday"))
    holidays = _parse_holidays(settings.get("holiday_list"))
    shift_if_holiday = bool(settings.get("expiry_shift_if_holiday", True))
    try:
        expiries = dw.get_optionchain_expirylist(INDEX_EXCHANGE_SEG, INDEX_SECURITY_ID)
    except Exception:
        expiries = []
    parsed: list[date] = []
    if isinstance(expiries, list):
        source = expiries
    elif isinstance(expiries, dict):
        source = expiries.get("data") or expiries.get("Expiry") or expiries.get("expiry") or []
    else:
        source = []
    for raw in source:
        try:
            clean = str(raw).split("T", 1)[0].split(" ", 1)[0]
            parsed.append(datetime.fromisoformat(clean).date())
        except Exception:
            continue
    parsed.sort()
    if not parsed:
        return monthly_expiry_date(today)
    holidays |= _derive_shifted_holidays(target_weekday, parsed)

    def ok(cand: date) -> bool:
        if cand < today:
            return False
        if target_weekday is not None and cand.weekday() != target_weekday:
            return False
        if shift_if_holiday and cand in holidays:
            return False
        return True

    filtered = [c for c in parsed if ok(c)]
    if filtered:
        return filtered[0]
    filtered_no_weekday = [c for c in parsed if c >= today and (not shift_if_holiday or c not in holidays)]
    if filtered_no_weekday:
        return filtered_no_weekday[0]
    return parsed[-1]


def _first_future_expiry(dw: DhanWrapper, settings: Dict[str, Any]) -> date:
    """Pick the earliest future expiry with weekday/holiday rules (for weekly hedges)."""
    today = date.today()
    target_weekday = _weekday_to_int(settings.get("nifty_expiry_weekday", "Tuesday"))
    holidays = _parse_holidays(settings.get("holiday_list"))
    shift_if_holiday = bool(settings.get("expiry_shift_if_holiday", True))
    try:
        expiries = dw.get_optionchain_expirylist(INDEX_EXCHANGE_SEG, INDEX_SECURITY_ID)
    except Exception:
        expiries = []
    parsed: list[date] = []
    source = expiries if isinstance(expiries, list) else expiries.get("data") if isinstance(expiries, dict) else []
    for raw in source:
        try:
            clean = str(raw).split("T", 1)[0].split(" ", 1)[0]
            parsed.append(datetime.fromisoformat(clean).date())
        except Exception:
            continue
    parsed = [p for p in parsed if p >= today]
    parsed.sort()
    if not parsed:
        return today
    def ok(c: date) -> bool:
        if target_weekday is not None and c.weekday() != target_weekday:
            return False
        if shift_if_holiday and c in holidays:
            return False
        return True
    for cand in parsed:
        if ok(cand):
            return cand
    return parsed[0]


def main():
    settings = _load_settings()
    lot_size = int(os.getenv("BATMAN_LOT_SIZE", settings.get("nifty_lot_size", settings.get("lot_size", 75))))
    long_offset = float(os.getenv("BATMAN_LONG_OFFSET", settings.get("batman_long_offset", 300.0)))
    short_offset = float(os.getenv("BATMAN_SHORT_OFFSET", settings.get("batman_short_offset", 500.0)))
    qty_long = int(os.getenv("BATMAN_QTY_LONG", settings.get("batman_qty_long", 1)))
    qty_short = int(os.getenv("BATMAN_QTY_SHORT", settings.get("batman_qty_short", 3)))
    weekly_hedge_enabled = os.getenv("BATMAN_WEEKLY_HEDGE_ENABLED", "1") == "1" and bool(
        settings.get("batman_weekly_hedge_enabled", True)
    )
    weekly_hedge_cap = float(os.getenv("BATMAN_WEEKLY_HEDGE_PRICE_CAP", settings.get("batman_weekly_hedge_price_cap", 12.0)))
    weekly_hedge_min_dist = float(
        os.getenv("BATMAN_WEEKLY_HEDGE_MIN_DISTANCE", settings.get("batman_weekly_hedge_min_distance", 300.0))
    )
    trade_mode = os.getenv("TRADE_MODE", "paper")
    dw = DhanWrapper(logger=LOG)
    expiry_override = os.getenv("BATMAN_EXPIRY")
    if expiry_override:
        try:
            expiry = datetime.strptime(expiry_override, "%Y-%m-%d").date()
        except Exception:
            LOG.warning("Invalid BATMAN_EXPIRY %s; using monthly expiry", expiry_override)
            expiry = monthly_expiry_date(date.today())
    else:
        # use configured weekday/holidays if available; fallback to monthly helper
        try:
            expiry = _select_expiry(dw, settings)
        except Exception:
            expiry = monthly_expiry_date(date.today())
    expiry_str = expiry.isoformat()
    LOG.info("Batman deploy: trade_mode=%s lot=%s expiry=%s", trade_mode, lot_size, expiry_str)

    chain_raw = dw.get_option_chain(INDEX_SECURITY_ID, INDEX_EXCHANGE_SEG, expiry_str)
    LOG.info("Raw option chain response (truncated): %s", str(chain_raw)[:400])
    chain_data = _coerce_chain(chain_raw)
    chain = _map_chain(chain_data)
    prices = _price_map(chain)
    spot = None
    if isinstance(chain_data, dict):
        spot = chain_data.get("data", {}).get("last_price") or chain_data.get("data", {}).get("data", {}).get("last_price")
    if spot is None and chain:
        strikes = sorted(float(k) for k in chain.keys())
        if strikes:
            spot = strikes[len(strikes) // 2]
    if spot is None:
        spot = dw.get_ltp_once(INDEX_EXCHANGE_SEG, INDEX_SECURITY_ID)
    if spot is None:
        raise RuntimeError("Could not infer spot from chain/LTP")

    ce_prices = {float(k.split(":")[1]): v for k, v in prices.items() if k.startswith("CALL:")}
    pe_prices = {float(k.split(":")[1]): v for k, v in prices.items() if k.startswith("PUT:")}

    long_ce = _nearest(ce_prices, _round_strike(spot + long_offset))
    short_ce = _nearest(ce_prices, _round_strike(spot + short_offset))
    long_pe = _nearest(pe_prices, _round_strike(spot - long_offset))
    short_pe = _nearest(pe_prices, _round_strike(spot - short_offset))

    # Optional weekly hedges (far OTM, cheap)
    weekly_hedges = []
    if weekly_hedge_enabled:
        try:
            weekly_expiry = _first_future_expiry(dw, settings).isoformat()
            weekly_chain_raw = dw.get_option_chain(INDEX_SECURITY_ID, INDEX_EXCHANGE_SEG, weekly_expiry)
            weekly_prices = _price_map(_map_chain(_coerce_chain(weekly_chain_raw)))
            weekly_ce = {float(k.split(":")[1]): v for k, v in weekly_prices.items() if k.startswith("CALL:") and v > 0}
            weekly_pe = {float(k.split(":")[1]): v for k, v in weekly_prices.items() if k.startswith("PUT:") and v > 0}
            def pick_weekly(target, book, above=True):
                candidates = []
                for s,p in book.items():
                    if above and s <= target + weekly_hedge_min_dist:
                        continue
                    if (not above) and s >= target - weekly_hedge_min_dist:
                        continue
                    if 0 < p <= weekly_hedge_cap:
                        candidates.append((s,p))
                if not candidates:
                    return None
                # pick farthest OTM within cap
                if above:
                    return max(candidates, key=lambda x: x[0])
                return min(candidates, key=lambda x: x[0])
            wh_ce = pick_weekly(spot, weekly_ce, above=True)
            wh_pe = pick_weekly(spot, weekly_pe, above=False)
            if wh_ce:
                weekly_hedges.append(("CALL", wh_ce[0], wh_ce[1]))
            if wh_pe:
                weekly_hedges.append(("PUT", wh_pe[0], wh_pe[1]))
            if weekly_hedges:
                LOG.info("Weekly hedges selected (exp=%s): %s", weekly_expiry, weekly_hedges)
        except Exception as exc:
            LOG.warning("Weekly hedge selection failed: %s", exc)

    class Leg:
        def __init__(self, option_type, strike, qty, direction):
            self.option_type = option_type
            self.strike = strike
            self.qty = qty
            self.direction = direction

    legs = [
        Leg("CALL", long_ce, qty_long, "LONG"),
        Leg("CALL", short_ce, qty_short, "SHORT"),
        Leg("PUT", long_pe, qty_long, "LONG"),
        Leg("PUT", short_pe, qty_short, "SHORT"),
    ]
    for opt_type, strike, _ltp in weekly_hedges:
        legs.append(Leg(opt_type, strike, 1, "LONG"))

    net_premium = 0.0
    for leg in legs:
        key = f"{'CALL' if leg.option_type=='CALL' else 'PUT'}:{leg.strike}"
        ltp = prices.get(key, 0.0)
        # if weekly hedge, try weekly price map fallback
        if ltp == 0.0 and weekly_hedges:
            try:
                weekly_expiry = _first_future_expiry(dw, settings).isoformat()
                weekly_chain_raw = dw.get_option_chain(INDEX_SECURITY_ID, INDEX_EXCHANGE_SEG, weekly_expiry)
                weekly_prices = _price_map(_map_chain(_coerce_chain(weekly_chain_raw)))
                ltp = weekly_prices.get(key, ltp)
            except Exception:
                pass
        sign = -1 if leg.direction == "LONG" else 1
        prem = sign * ltp * leg.qty * lot_size
        net_premium += prem
        LOG.info("%s %s %s qty=%s ltp=%.2f prem=%.2f", leg.direction, leg.option_type, leg.strike, leg.qty * lot_size, ltp, prem)
    LOG.info("Net premium (credit +): %.2f", net_premium)

    for leg in legs:
        _place_leg(dw, expiry_str, lot_size, leg, trade_mode)
    LOG.info("Batman structure sent (%d legs)", len(legs))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        LOG.exception("Batman daemon error: %s", exc)
        sys.exit(1)
