#!/usr/bin/env python3
"""
Lightweight HTTP server for a flicker-free Paper P&L frontend.

Serves:
  - /api/paper_positions : JSON snapshot of paper positions (aggregated from trade_blotter.csv)
  - static files under web/paper_pnl/ (index.html uses the API above)

Usage:
  python scripts/paper_pnl_server.py --port 8000
Then open http://localhost:8000 in your browser.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import subprocess
import signal

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "data_engine" / "market_ai" / "state"
AGENT_LOG = STATE_DIR / "agent.log"
BLOTTER_CSV = STATE_DIR / "trade_blotter.csv"
STRATEGY_STATE = STATE_DIR / "strategy_state.json"
SETTINGS_JSON = STATE_DIR / "agent_settings.json"
LIVE_GATE_STATUS_JSON = STATE_DIR / "live_gate_status.json"
LIVE_GATE_SESSIONS_JSONL = STATE_DIR / "live_gate_sessions.jsonl"
POSITION_RECONCILE_STATUS_JSON = STATE_DIR / "position_reconcile_status.json"
EXECUTION_RECOVERY_STATUS_JSON = STATE_DIR / "execution_recovery_status.json"
EXECUTION_JOURNAL_JSONL = STATE_DIR / "execution_journal.jsonl"
AGENT_HEARTBEAT_JSON = STATE_DIR / "agent_heartbeat.json"
AGENT_ALERTS_JSONL = STATE_DIR / "agent_alerts.jsonl"
TELEGRAM_ALERT_STATUS_JSON = STATE_DIR / "telegram_alert_status.json"
BATMAN_BKM_TUNING_ADVICE_JSON = STATE_DIR / "batman_bkm_tuning_advice.json"
BATMAN_BKM_TUNING_HISTORY_JSONL = STATE_DIR / "batman_bkm_tuning_history.jsonl"
CREDS_FILE = STATE_DIR / "creds.json"
LAST_STRATEGY_FILE = STATE_DIR / "last_strategy.json"
PID_FILE = STATE_DIR / "agent.pid"
AGENT_ENTRY = ROOT / "data_engine" / "market_ai" / "start_agent.py"
# New unified frontend location
STATIC_DIR = ROOT / "web" / "app"
INDEX_SECURITY_ID = int(os.getenv("MARKET_AI_INDEX_SECURITY_ID", "13"))
INDEX_EXCHANGE_SEG = os.getenv("MARKET_AI_INDEX_EXCHANGE_SEG", "IDX_I")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from data_engine.market_ai.modules.agents.batman_bkm_tuning_advisor import (  # type: ignore
    TuningPaths as BatmanBKMTuningPaths,
    apply_proposal as apply_batman_bkm_tuning_proposal,
    load_or_refresh_advice as load_or_refresh_batman_bkm_tuning_advice,
    refresh_advice as refresh_batman_bkm_tuning_advice,
)
from data_engine.market_ai.modules.agents.position_reconciler import (  # type: ignore
    PositionReconciler,
    PositionReconcilerConfig,
)
from data_engine.market_ai.modules.agents.execution_recovery_guard import (  # type: ignore
    ExecutionRecoveryGuard,
    ExecutionRecoveryConfig,
    ExecutionJournal,
)
from data_engine.market_ai.modules.agents.ops_monitor import (  # type: ignore
    compute_watchdog_status,
    AlertJournal,
    AlertConfig,
)
from data_engine.market_ai.modules.agents.telegram_alerts import (  # type: ignore
    TelegramAlertForwarder,
    TelegramAlertConfig,
)


def _parse_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _parse_int(val: Any, default: int = 0) -> int:
    try:
        return int(float(val))
    except Exception:
        return default


def _build_chain_map(expiry: str) -> Dict[str, Any]:
    """
    Fetch option chain for the index and return:
      - map: sec_id -> {option_type, symbol, strike, ltp}
      - spot: underlying spot from chain response
    Cached per expiry to reduce load.
    """
    if not expiry:
        return {"map": {}, "spot": None}
    now = time.time()
    cached = _CHAIN_CACHE.get(expiry)
    if cached and now - cached.get("ts", 0) < 60:
        return cached
    try:
        from data_engine.market_ai.dhan_wrapper import DhanWrapper  # type: ignore
        creds = _json_read(CREDS_FILE)
        cid = (creds.get("client_id") or "").strip()
        tok = (creds.get("access_token") or "").strip()
        if cid and tok:
            os.environ["DHAN_CLIENT_ID"] = cid
            os.environ["DHAN_ACCESS_TOKEN"] = tok
        dw = DhanWrapper(logger=None)
        expiry_date = expiry
        resp = dw.get_option_chain(INDEX_SECURITY_ID, INDEX_EXCHANGE_SEG, expiry_date)
        if resp is None or not resp:
            # try to at least get spot
            spot_only = _parse_float(dw.get_ltp_once(INDEX_EXCHANGE_SEG, INDEX_SECURITY_ID), None) if hasattr(dw, "get_ltp_once") else None
            return {"map": {}, "spot": spot_only, "ts": time.time()}
    except Exception as exc:
        print(f"[chain_map] fetch failed: {exc}")
        return {"map": {}, "spot": None}

    def _first(d, *keys):
        for k in keys:
            if isinstance(d, dict) and k in d:
                return d[k]
        return None

    data = resp.get("data") if isinstance(resp, dict) else {}
    oc = data.get("oc") if isinstance(data, dict) else None
    if oc is None and isinstance(resp, dict) and isinstance(resp.get("oc"), dict):
        oc = resp["oc"]
    spot = None
    if isinstance(data, dict):
        spot = _parse_float(_first(data, "last_price", "underlying_value", "underlyingValue", "underlying_price"), None)
    if spot is None:
        try:
            from data_engine.market_ai.dhan_wrapper import DhanWrapper  # type: ignore
            dw = DhanWrapper(logger=None)
            spot = _parse_float(dw.get_ltp_once(INDEX_EXCHANGE_SEG, INDEX_SECURITY_ID), None) if hasattr(dw, "get_ltp_once") else None
        except Exception:
            pass

    chain_map: Dict[str, Dict[str, Any]] = {}
    if isinstance(oc, dict):
        for strike_key, node in oc.items():
            for field, opt_type in (("ce", "CE"), ("pe", "PE")):
                leg = node.get(field) or {}
                sec_id = _first(leg, "securityId", "security_id", "instrumentId", "instrument_id")
                if not sec_id:
                    continue
                sec_id_str = str(int(float(sec_id)))
                symbol = _first(leg, "trading_symbol", "tradingSymbol", "symbol", "instrument")
                last_price = _parse_float(_first(leg, "last_price", "lastPrice", "ltp", "LTP"), None)
                chain_map[sec_id_str] = {
                    "option_type": opt_type,
                    "symbol": symbol,
                    "strike": _parse_float(strike_key, None),
                    "ltp": last_price,
                }
    else:
        print("[chain_map] oc missing; using spot only")
    out = {"map": chain_map, "spot": spot, "ts": now}
    _CHAIN_CACHE[expiry] = out
    return out

# Simple in-memory cache to avoid hammering marketfeed
_LTP_CACHE: Dict[Tuple[str, str], Tuple[float, float]] = {}  # (seg,sid) -> (ltp, ts)
_LAST_LTP_FETCH_TS: float = 0.0
_NEXT_LTP_ALLOWED_TS: float = 0.0
_CHAIN_CACHE: Dict[str, Dict[str, Any]] = {}  # expiry -> {"map": {sec_id: {...}}, "spot": float, "ts": float}


def load_positions(blotter_path: Path, mode: str = "paper") -> Dict[str, Any]:
    """
    Aggregate OPEN + latest MTM rows into per-leg positions and total P&L.
    Also derives closed legs (realized) when notes == CLOSE.
    When mode==live, enrich missing LTPs via Dhan marketfeed for sec_ids present.
    """
    legs: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    closed: List[Dict[str, Any]] = []
    latest_expiry: Optional[str] = None
    mode = (mode or "").lower()

    if not blotter_path.exists():
        return {
            "positions": [],
            "closed": [],
            "total_pnl": 0.0,
            "realized_pnl": 0.0,
            "as_of": datetime.now().isoformat(),
            "blotter_tail": [],
            "margin_available": None,
            "margin_used": None,
        }

    tail_rows: List[Dict[str, Any]] = []
    with blotter_path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows = sorted(list(reader), key=lambda r: r.get("timestamp", ""))
        tail_rows = rows[-30:] if len(rows) > 30 else rows

    # determine latest expiry in selected mode rows
    for row in rows:
        if (row.get("trade_mode") or "").lower() != mode:
            continue
        exp = row.get("expiry") or ""
        if exp and (latest_expiry is None or exp > latest_expiry):
            latest_expiry = exp

    chain_map: Dict[str, Any] = {}
    spot_val: Optional[float] = None
    if latest_expiry:
        cm = _build_chain_map(latest_expiry)
        chain_map = cm.get("map") or {}
        spot_val = cm.get("spot")

    # Collect sec_ids to bulk fetch LTPs
    ltp_lookup: Dict[Tuple[str, str], float] = {}
    ltp_status = "skip"
    pairs: Dict[str, list[int]] = {}
    for row in rows:
        if (row.get("trade_mode") or "").lower() != mode:
            continue
        seg = str(row.get("exchange_seg") or "NSE_FNO")
        try:
            sid = int(float(row.get("security_id") or 0))
        except Exception:
            continue
        if sid:
            pairs.setdefault(seg, []).append(sid)
    if pairs:
        # dedupe sec_ids per segment to avoid API 400 on duplicates
        for seg, ids in pairs.items():
            pairs[seg] = sorted(list({int(i) for i in ids}))
        try:
            import requests
            creds = _json_read(CREDS_FILE)
            headers = {
                "client-id": (creds.get("client_id") or "").strip(),
                "access-token": (creds.get("access_token") or "").strip(),
            }
            if not headers["client-id"] or not headers["access-token"]:
                ltp_status = "no-creds"
                raise RuntimeError("client_id/access_token missing")

            # Rate-limit with backoff to avoid 429
            global _LAST_LTP_FETCH_TS, _NEXT_LTP_ALLOWED_TS
            now_ts = time.time()
            if now_ts < _NEXT_LTP_ALLOWED_TS or (now_ts - _LAST_LTP_FETCH_TS) < 4:
                ltp_status = "cached"
            else:
                base = os.getenv("DHAN_API_BASE", "https://api.dhan.co").rstrip("/")
                resp = requests.post(f"{base}/v2/marketfeed/ltp", headers=headers, json=pairs, timeout=8)
                if resp.status_code == 429:
                    # back off 10s on throttle
                    _NEXT_LTP_ALLOWED_TS = now_ts + 10
                    ltp_status = "throttled:429 (cache)"
                elif resp.status_code >= 400:
                    ltp_status = f"http{resp.status_code}: {resp.text[:120]}"
                    raise RuntimeError(f"marketfeed {resp.status_code}")
                else:
                    data = resp.json().get("data", {})
                    for seg, sec_map in data.items():
                        if isinstance(sec_map, dict):
                            for sid, payload in sec_map.items():
                                ltp = payload.get("last_price") if isinstance(payload, dict) else None
                                if ltp is not None:
                                    key = (seg, str(sid))
                                    ltp_lookup[key] = float(ltp)
                                    _LTP_CACHE[key] = (float(ltp), now_ts)
                    _LAST_LTP_FETCH_TS = now_ts
                    ltp_status = "ok"
        except Exception as exc:
            # If creds missing or HTTP errors, mark status but continue with stale/cache prices.
            ltp_status = f"error: {exc}"
            print(f"[ltp_enrich] failed: {exc}")

    for row in rows:
        if (row.get("trade_mode") or "").lower() != mode:
            continue
        exp = row.get("expiry") or ""
        if latest_expiry and exp != latest_expiry:
            continue
        sec_id = str(row.get("security_id") or "")
        strike = row.get("strike") or ""
        side = str(row.get("side") or "").upper()
        qty = _parse_int(row.get("quantity"), 0)
        price = _parse_float(row.get("price"), 0.0)
        notes = str(row.get("notes") or "").upper()
        key = (sec_id, str(strike), str(exp))
        leg = legs.setdefault(
            key,
            {
                "entry": None,
                "entry_qty": 0,
                "ltp": None,
                "qty": 0,
                "side": side,
                "strike": strike,
                "expiry": exp,
                "sec_id": sec_id,
                "option_type": None,
                "symbol": None,
            },
        )

        if notes == "OPEN":
            prev_qty = leg.get("entry_qty", 0)
            signed = qty if side == "SELL" else -qty
            new_qty = prev_qty + signed
            if new_qty != 0:
                prev_entry = leg.get("entry") or 0.0
                leg["entry"] = ((prev_entry * prev_qty) + (price * signed)) / new_qty
                leg["entry_qty"] = new_qty
            leg["qty"] = new_qty
            leg["side"] = "SELL" if new_qty > 0 else "BUY"
        elif notes == "MTM":
            leg["ltp"] = price
            leg["qty"] = qty if side == "SELL" else -qty
            leg["side"] = "SELL" if leg["qty"] > 0 else "BUY"
        elif notes == "CLOSE":
            # Realized P&L for the quantity closed
            entry_px = leg.get("entry")
            if entry_px is None:
                continue
            close_side = "SELL" if side == "SELL" else "BUY"
            close_qty = qty
            pnl = (entry_px - price) * close_qty if close_side == "SELL" else (price - entry_px) * close_qty
            closed.append(
                {
                    "side": close_side,
                    "strike": strike,
                    "expiry": exp,
                    "sec_id": sec_id,
                    "qty": close_qty,
                    "entry": entry_px,
                    "exit": price,
                    "pnl": pnl,
                    "timestamp": row.get("timestamp"),
                }
            )

    # Enrich LTP where blotter lacks MTM rows (both paper and live)
    if ltp_lookup or _LTP_CACHE:
        for key, leg in legs.items():
            seg = next((r.get("exchange_seg") for r in rows if (r.get("security_id") or "") == leg.get("sec_id")), "NSE_FNO")
            cache_key = (seg, str(leg.get("sec_id")))
            ltp = ltp_lookup.get(cache_key)
            if ltp is None:
                cached = _LTP_CACHE.get(cache_key)
                if cached:
                    ltp = cached[0]
            if ltp is not None:
                leg["ltp"] = ltp
            # enrich option_type/symbol from chain map
            if not leg.get("option_type"):
                cm = chain_map.get(str(leg.get("sec_id")))
                if cm:
                    leg["option_type"] = cm.get("option_type")
                    leg["symbol"] = cm.get("symbol")
        # If still missing option_type, infer by comparing to median strike
    strikes_f = []
    for leg in legs.values():
        try:
            strikes_f.append(float(leg.get("strike")))
        except Exception:
            continue
    atm_guess = None
    if strikes_f:
        strikes_f.sort()
        atm_guess = strikes_f[len(strikes_f)//2]
    for leg in legs.values():
        if not leg.get("option_type"):
            try:
                ref = spot_val if spot_val is not None else atm_guess
                if ref is not None and leg.get("strike") is not None:
                    leg["option_type"] = "CE" if float(leg["strike"]) >= float(ref) else "PE"
            except Exception:
                continue

    positions: List[Dict[str, Any]] = []
    total_pnl = 0.0
    for leg in legs.values():
        entry_px = leg.get("entry")
        ltp_px = leg.get("ltp") if leg.get("ltp") not in (None, "") else entry_px
        if entry_px is None:
            continue
        qty = abs(int(leg.get("qty") or leg.get("entry_qty") or 0))
        side = leg.get("side")
        pnl = None
        if ltp_px is not None:
            if side == "SELL":
                pnl = (entry_px - ltp_px) * qty
            else:
                pnl = (ltp_px - entry_px) * qty
            total_pnl += pnl
        positions.append(
            {
                "side": side,
                "strike": leg.get("strike"),
                "expiry": leg.get("expiry"),
                "sec_id": leg.get("sec_id"),
                "qty": qty,
                "entry": entry_px,
                "ltp": ltp_px,
                "pnl": pnl,
            }
        )

    return {
        "positions": positions,
        "total_pnl": total_pnl,
        "closed": closed,
        "realized_pnl": sum(c.get("pnl", 0.0) for c in closed),
        "as_of": datetime.now().isoformat(),
        "blotter_tail": tail_rows,
        "margin_available": None,
        "margin_used": None,
        "ltp_status": ltp_status,
        "spot": spot_val,
    }


def _load_broker_live_positions() -> Dict[str, Any]:
    """Fetch positions from Dhan with LTP and P&L and funds. Derive closed legs from netted zero qty."""
    try:
        from data_engine.market_ai.dhan_wrapper import DhanWrapper  # type: ignore
    except Exception as exc:
        print(f"[live_fallback] unable to import DhanWrapper: {exc}")
        return {"positions": [], "total_pnl": 0.0, "source": "none"}

    creds = _json_read(CREDS_FILE)
    cid = (creds.get("client_id") or "").strip()
    tok = (creds.get("access_token") or "").strip()
    if cid and tok:
        os.environ["DHAN_CLIENT_ID"] = cid
        os.environ["DHAN_ACCESS_TOKEN"] = tok
    positions_full: List[Dict[str, Any]] = []
    try:
        dw = DhanWrapper(logger=None)
        # Prefer live positions with LTP if available
        try:
            raw = dw.get_positions_live_with_ltp()  # type: ignore[attr-defined]
        except Exception:
            raw = dw.get_positions_raw()  # type: ignore[attr-defined]
        funds = dw.get_funds()
        # Also pull full positions via REST to get netQty==0 legs
        try:
            import requests
            headers = {"client-id": cid, "access-token": tok}
            base = os.getenv("DHAN_API_BASE", "https://api.dhan.co").rstrip("/")
            resp = requests.get(f"{base}/v2/positions", headers=headers, timeout=10)
            resp.raise_for_status()
            positions_full = resp.json()
        except Exception as exc:
            print(f"[live_positions_full] fetch failed: {exc}")
    except Exception as exc:
        print(f"[live_fallback] get_positions_raw failed: {exc}")
        return {"positions": [], "total_pnl": 0.0, "source": "none", "margin_available": None, "margin_used": None}

    positions: List[Dict[str, Any]] = []
    closed: List[Dict[str, Any]] = []
    seg_map: Dict[str, list[int]] = {}

    for row in raw or []:
        seg = str(row.get("exchangeSegment") or row.get("exchange_seg") or "NSE_FNO")
        buy_qty = _parse_int(row.get("buyQty") or row.get("buy_qty") or row.get("buy_quantity"), 0)
        sell_qty = _parse_int(row.get("sellQty") or row.get("sell_qty") or row.get("sell_quantity"), 0)
        net_calc = sell_qty - buy_qty  # positive => net short
        net_qty = _parse_int(row.get("netQty") or row.get("net_qty"), net_calc)
        # prefer computed net_calc to avoid sign ambiguity
        net = net_calc if net_calc != 0 else net_qty
        qty = abs(net)
        side = "SELL" if net > 0 else "BUY"
        buy_avg = _parse_float(
            row.get("buyAvg")
            or row.get("buy_avg")
            or row.get("averageBuyPrice")
            or row.get("avgBuyPrice"),
            0.0,
        )
        sell_avg = _parse_float(
            row.get("sellAvg")
            or row.get("sell_avg")
            or row.get("averageSellPrice")
            or row.get("avgSellPrice"),
            0.0,
        )
        entry = sell_avg if side == "SELL" else buy_avg
        ltp = _parse_float(row.get("ltp") or row.get("LTP") or row.get("last_price") or row.get("lastPrice"), entry)
        expiry = row.get("expiryDate") or row.get("expiry") or ""
        strike = row.get("tradingSymbol") or row.get("symbol") or row.get("strikePrice") or row.get("strike") or ""
        sec_id = row.get("securityId") or row.get("security_id") or row.get("instrumentId") or ""
        if sec_id:
            seg_map.setdefault(seg, []).append(int(sec_id))
        if net != 0:
            positions.append(
                {
                    "side": side,
                    "strike": strike,
                    "expiry": expiry,
                    "sec_id": sec_id,
                    "qty": qty,
                    "entry": entry,
                    "ltp": ltp,
                }
            )
        else:
            # closed leg today or still in book; compute realized from buy/sell avgs
            if buy_qty > 0 and sell_qty > 0:
                pnl_real = (sell_avg - buy_avg) * min(buy_qty, sell_qty)
                closed.append(
                    {
                        "side": "FLAT",
                        "strike": strike,
                        "expiry": expiry,
                        "sec_id": sec_id,
                        "qty": min(buy_qty, sell_qty),
                        "entry": buy_avg,
                        "exit": sell_avg,
                        "pnl": pnl_real,
                        "timestamp": row.get("updateTime") or row.get("createTime") or "",
                    }
                )

    # Enrich LTP via marketfeed and recompute P&L
    try:
        import requests
        if seg_map:
            headers = {"client-id": cid, "access-token": tok}
            base = os.getenv("DHAN_API_BASE", "https://api.dhan.co").rstrip("/")
            resp = requests.post(f"{base}/v2/marketfeed/ltp", headers=headers, json=seg_map, timeout=8)
            resp.raise_for_status()
            data = resp.json().get("data", {})
            for pos in positions:
                seg = next(iter(seg_map.keys())) if len(seg_map) == 1 else "NSE_FNO"
                sid = pos.get("sec_id")
                ltp = data.get(seg, {}).get(str(sid), {}).get("last_price") if isinstance(data.get(seg, {}), dict) else None
                if ltp is not None:
                    pos["ltp"] = float(ltp)
    except Exception as exc:
        print(f"[live_ltp_fallback] failed: {exc}")

    total_pnl = 0.0
    for pos in positions:
        entry = pos.get("entry") or 0.0
        ltp = pos.get("ltp") or entry
        qty = pos.get("qty") or 0
        if pos.get("side") == "SELL":
            pos["pnl"] = (entry - ltp) * qty
        else:
            pos["pnl"] = (ltp - entry) * qty
        total_pnl += pos["pnl"]

    # Add closed legs from positions_full (netQty == 0) – only if there was activity today (day buys/sells)
    for row in positions_full or []:
        net_qty = _parse_int(row.get("netQty") or row.get("net_quantity") or 0, 0)
        if net_qty != 0:
            continue
        day_activity = _parse_int(row.get("dayBuyQty") or row.get("dayBuyQty") or 0, 0) + _parse_int(row.get("daySellQty") or row.get("daySellQty") or 0, 0)
        if day_activity == 0:
            continue
        exch = str(row.get("exchangeSegment") or "").upper()
        if "FNO" not in exch and "IDX" not in exch and "DER" not in exch:
            continue
        sec_id = row.get("securityId") or row.get("security_id") or ""
        symbol = row.get("tradingSymbol") or row.get("symbol") or ""
        expiry = row.get("drvExpiryDate") or row.get("expiryDate") or ""
        buy_avg = _parse_float(row.get("buyAvg") or row.get("buy_avg"), None)
        sell_avg = _parse_float(row.get("sellAvg") or row.get("sell_avg"), None)
        buy_qty = _parse_int(row.get("buyQty") or row.get("buy_qty") or 0, 0)
        sell_qty = _parse_int(row.get("sellQty") or row.get("sell_qty") or 0, 0)
        realized = _parse_float(row.get("realizedProfit") or row.get("realized_profit") or 0.0, 0.0)
        closed.append(
            {
                "side": "FLAT",
                "strike": symbol,
                "expiry": expiry,
                "sec_id": sec_id,
                "qty": min(buy_qty, sell_qty) if buy_qty and sell_qty else (sell_qty or buy_qty),
                "entry": buy_avg,
                "exit": sell_avg,
                "pnl": realized,
                "timestamp": "",
            }
        )

    # Deduplicate closed rows (sec_id, strike); prefer FLAT over SELL, prefer non-empty expiry and entry/exit
    # Sort closed to prefer rows with expiry, entry, FLAT
    closed.sort(key=lambda c: (
        c.get("expiry") not in (None, "", "0001-01-01"),
        c.get("entry") is not None and c.get("exit") is not None,
        c.get("side") == "FLAT"
    ), reverse=True)
    dedup = {}
    for c in closed:
        key = (str(c.get("sec_id")), c.get("strike"))
        if key in dedup:
            continue
        else:
            dedup[key] = c
    closed = list(dedup.values())
    # Prefer FLAT rows; drop SELL placeholders
    closed = [c for c in closed if c.get("side") == "FLAT"]
    # Prefer entries with non-empty expiry
    non_empty_exp = [c for c in closed if c.get("expiry") not in (None, "", "0001-01-01")]
    if non_empty_exp:
        closed = non_empty_exp

    return {
        "positions": positions,
        "closed": closed,
        "total_pnl": total_pnl,
        "as_of": datetime.now().isoformat(),
        "blotter_tail": [],
        "source": "broker",
        "margin_available": funds.get("available") if 'funds' in locals() else None,
        "margin_used": funds.get("utilized") if 'funds' in locals() else None,
    }


def _load_broker_trades_today() -> List[Dict[str, Any]]:
    """
    Pull today's executed orders directly from Dhan /v2/orders.
    Compute realized P&L by netting buys vs sells per security; emit a row when the position is flattened, using full history.
    Filter to F&O-style (derivative) trades so cash-equity intraday noise doesn't appear in closed options.
    """
    creds = _json_read(CREDS_FILE)
    cid = (creds.get("client_id") or "").strip()
    tok = (creds.get("access_token") or "").strip()
    if not cid or not tok:
        return []
    headers = {"client-id": cid, "access-token": tok}
    try:
        import requests
        base = os.getenv("DHAN_API_BASE", "https://api.dhan.co").rstrip("/")
        resp = requests.get(f"{base}/v2/orders", headers=headers, timeout=10)
        resp.raise_for_status()
        rows = resp.json()
    except Exception as exc:
        print(f"[live_orders] failed to fetch orders: {exc}")
        return []
    today = datetime.now().date().isoformat()
    # sort by timestamp for FIFO
    try:
        rows = sorted(rows, key=lambda r: r.get("createTime") or r.get("updateTime") or "")
    except Exception:
        pass

    per_sec_state = {}  # sec_id -> {"qty": signed, "buy_qty": int, "sell_qty": int, "buy_cost": float, "sell_cost": float, "symbol": str, "expiry": str, "ts": str}
    closed_rows: List[Dict[str, Any]] = []

    for row in rows or []:
        if str(row.get("orderStatus") or "").upper() != "TRADED":
            continue
        qty = _parse_int(row.get("filledQty") or row.get("quantity"), 0)
        if qty == 0:
            continue
        exch = str(row.get("exchangeSegment") or row.get("exchange_segment") or "").upper()
        # keep only derivatives / F&O style; skip pure EQ
        if not (exch.endswith("FNO") or exch.endswith("DER") or "IDX" in exch or row.get("drvExpiryDate", "") not in ("", "0001-01-01")):
            continue
        side = "SELL" if str(row.get("transactionType") or "").upper() == "SELL" else "BUY"
        price = _parse_float(row.get("averageTradedPrice") or row.get("price"), 0.0)
        sec_id = row.get("securityId") or ""
        symbol = row.get("tradingSymbol") or row.get("securityId") or ""
        expiry = row.get("drvExpiryDate") or ""
        ts = row.get("createTime") or row.get("updateTime") or ""
        state = per_sec_state.setdefault(sec_id, {"qty": 0, "buy_qty": 0, "sell_qty": 0, "buy_cost": 0.0, "sell_cost": 0.0, "symbol": symbol, "expiry": expiry, "ts": ts})
        multiplier = 100.0 if exch == "MCX_COMM" else 1.0

        if side == "BUY":
            state["qty"] += qty
            state["buy_qty"] += qty
            state["buy_cost"] += price * qty
        else:
            state["qty"] -= qty
            state["sell_qty"] += qty
            state["sell_cost"] += price * qty

        # Emit realized P&L when position flat AFTER this trade; show only if flatten date is today
        if state["qty"] == 0 and state["buy_qty"] and state["sell_qty"]:
            buy_avg = state["buy_cost"] / state["buy_qty"]
            sell_avg = state["sell_cost"] / state["sell_qty"]
            realized = (sell_avg - buy_avg) * min(state["buy_qty"], state["sell_qty"]) * multiplier
            close_date = str(ts).split(" ")[0]
            if close_date == today:
                closed_rows.append(
                    {
                        "side": "FLAT",
                        "strike": state["symbol"],
                        "expiry": state["expiry"],
                        "sec_id": sec_id,
                        "qty": min(state["buy_qty"], state["sell_qty"]),
                        "entry": buy_avg,
                        "exit": sell_avg,
                        "pnl": realized,
                        "timestamp": ts,
                    }
                )
            state["buy_qty"] = state["sell_qty"] = 0
            state["buy_cost"] = state["sell_cost"] = 0.0

    # Also surface exits that have sells today but no corresponding buys in the feed (older entries).
    for sec_id, st in per_sec_state.items():
        if st["sell_qty"] > 0 and st["buy_qty"] == 0:
            closed_rows.append(
                {
                    "side": "SELL",
                    "strike": st["symbol"],
                    "expiry": st["expiry"],
                    "sec_id": sec_id,
                    "qty": st["sell_qty"],
                    "entry": None,
                    "exit": st["sell_cost"] / st["sell_qty"] if st["sell_qty"] else None,
                    "pnl": None,
                    "timestamp": st["ts"],
                }
            )

    return closed_rows

    return closed_rows

    return closed_rows


def _json_read(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _json_write(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2))


def _ist_now() -> datetime:
    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    return datetime.now(tz) if tz else datetime.now()


def _default_live_gate_status(settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cfg = settings or _json_read(SETTINGS_JSON)
    stage1_lot = max(1, _parse_int(cfg.get("live_stage1_lot_multiplier"), 1))
    now = _ist_now()
    return {
        "status": "PROBATION",
        "stage": "S1",
        "lot_multiplier_active": stage1_lot,
        "sessions_total": 0,
        "sessions_pass": 0,
        "sessions_fail": 0,
        "cum_mtm": 0.0,
        "current_session_date": now.date().isoformat(),
        "last_fail_reason": None,
        "locked_for_date": None,
        "updated_at": now.isoformat(timespec="seconds"),
        "hard_lock": False,
        "consecutive_failures": 0,
    }


def _load_live_gate_status() -> Dict[str, Any]:
    payload = _json_read(LIVE_GATE_STATUS_JSON)
    if not payload or not isinstance(payload, dict):
        return _default_live_gate_status()
    return payload


def _reset_live_gate_status() -> Dict[str, Any]:
    status = _default_live_gate_status()
    LIVE_GATE_STATUS_JSON.parent.mkdir(parents=True, exist_ok=True)
    _json_write(LIVE_GATE_STATUS_JSON, status)
    try:
        LIVE_GATE_SESSIONS_JSONL.write_text("")
    except Exception:
        pass
    return status


def _default_reconcile_status() -> Dict[str, Any]:
    rec = PositionReconciler(
        config=PositionReconcilerConfig(),
        status_path=POSITION_RECONCILE_STATUS_JSON,
        logger=None,
    )
    return rec.snapshot()


def _load_reconcile_status() -> Dict[str, Any]:
    if not POSITION_RECONCILE_STATUS_JSON.exists():
        return _default_reconcile_status()
    payload = _json_read(POSITION_RECONCILE_STATUS_JSON)
    return payload if isinstance(payload, dict) and payload else _default_reconcile_status()


def _reset_reconcile_status() -> Dict[str, Any]:
    rec = PositionReconciler(
        config=PositionReconcilerConfig(),
        status_path=POSITION_RECONCILE_STATUS_JSON,
        logger=None,
    )
    return rec.reset()


def _default_execution_recovery_status() -> Dict[str, Any]:
    guard = ExecutionRecoveryGuard(
        config=ExecutionRecoveryConfig(),
        status_path=EXECUTION_RECOVERY_STATUS_JSON,
        logger=None,
    )
    return guard.snapshot()


def _load_execution_recovery_status() -> Dict[str, Any]:
    if not EXECUTION_RECOVERY_STATUS_JSON.exists():
        return _default_execution_recovery_status()
    payload = _json_read(EXECUTION_RECOVERY_STATUS_JSON)
    return payload if isinstance(payload, dict) and payload else _default_execution_recovery_status()


def _reset_execution_recovery_status() -> Dict[str, Any]:
    guard = ExecutionRecoveryGuard(
        config=ExecutionRecoveryConfig(),
        status_path=EXECUTION_RECOVERY_STATUS_JSON,
        logger=None,
    )
    return guard.reset()


def _safe_autocleanup_bkm_after_stop() -> Dict[str, Any]:
    """
    Safely clean stale Batman BKM local/journal state after operator stop.

    Cleanup runs only when broker positions are confirmed flat. This prevents
    orphaning live positions by blindly clearing local state.
    """
    out: Dict[str, Any] = {
        "attempted": True,
        "performed": False,
        "reason": None,
        "local_open_expiries": [],
        "journal_active_expiries": [],
        "broker_open_count": None,
    }
    try:
        local_state = _json_read(STRATEGY_STATE)
        bkm_state = local_state.get("BATMAN_BKM") if isinstance(local_state, dict) else {}
        local_open_expiries = sorted(
            str(exp)
            for exp, meta in (bkm_state or {}).items()
            if isinstance(meta, dict) and str(meta.get("status") or "").upper() == "OPEN"
        )
        out["local_open_expiries"] = local_open_expiries

        settings = _json_read(SETTINGS_JSON)
        lookback_days = max(1, _parse_int(settings.get("live_exec_recovery_lookback_days"), 45)) if isinstance(settings, dict) else 45
        journal = ExecutionJournal(journal_path=EXECUTION_JOURNAL_JSONL, logger=None)
        journal_summary = journal.analyze_bkm(lookback_days=lookback_days)
        active_baskets = journal_summary.get("active_baskets") if isinstance(journal_summary, dict) else {}
        journal_active_expiries = sorted(str(k) for k in (active_baskets or {}).keys())
        out["journal_active_expiries"] = journal_active_expiries

        if not local_open_expiries and not journal_active_expiries:
            out["reason"] = "NO_OPEN_BKM_STATE"
            return out

        broker_payload = _load_broker_live_positions()
        if str(broker_payload.get("source") or "") != "broker":
            out["reason"] = "BROKER_CHECK_UNAVAILABLE"
            return out
        broker_positions = broker_payload.get("positions") if isinstance(broker_payload, dict) else []
        broker_positions = broker_positions if isinstance(broker_positions, list) else []
        out["broker_open_count"] = len(broker_positions)
        if broker_positions:
            out["reason"] = "BROKER_NOT_FLAT"
            return out

        ts = _ist_now()
        union_expiries = sorted(set(local_open_expiries) | set(journal_active_expiries))
        for expiry in union_expiries:
            active_meta = (active_baskets or {}).get(expiry, {}) if isinstance(active_baskets, dict) else {}
            payload = {
                "strategy": "BATMAN_BKM",
                "expiry": expiry,
                "op_id": f"AUTOCLEAN-{int(ts.timestamp())}",
                "details": {
                    "manual_reconcile": True,
                    "reason": "auto_cleanup_on_control_stop",
                },
            }
            if isinstance(active_meta, dict):
                legs = active_meta.get("legs")
                meta = active_meta.get("meta")
                if isinstance(legs, list):
                    payload["legs"] = legs
                if isinstance(meta, dict):
                    payload["meta"] = meta
            journal.record("BKM_CLOSE_SUCCESS", payload=payload, when=ts)

        if isinstance(local_state, dict):
            local_state["BATMAN_BKM"] = {}
            _json_write(STRATEGY_STATE, local_state)
        else:
            _json_write(STRATEGY_STATE, {"BATMAN_V2": {}, "BATMAN_BKM": {}})

        rec_status = _reset_reconcile_status()
        exec_status = _reset_execution_recovery_status()
        out.update(
            {
                "performed": True,
                "reason": "CLEANED_BKM_STATE",
                "cleaned_expiries": union_expiries,
                "reconcile_status": rec_status.get("status"),
                "execution_recovery_status": exec_status.get("status"),
            }
        )
        return out
    except Exception as exc:
        out["reason"] = "AUTO_CLEANUP_ERROR"
        out["error"] = str(exc)
        return out


def _load_agent_heartbeat() -> Dict[str, Any]:
    payload = _json_read(AGENT_HEARTBEAT_JSON)
    return payload if isinstance(payload, dict) else {}


def _watchdog_health_status() -> Dict[str, Any]:
    settings = _json_read(SETTINGS_JSON)
    stale_after = _parse_float(settings.get("ops_watchdog_stale_after_sec"), 45.0) if isinstance(settings, dict) else 45.0
    hb = _load_agent_heartbeat()
    return compute_watchdog_status(heartbeat_payload=hb, stale_after_sec=stale_after)


def _load_alerts_tail(limit: int = 100) -> List[Dict[str, Any]]:
    journal = AlertJournal(path=AGENT_ALERTS_JSONL, config=AlertConfig(), logger=None)
    return journal.tail(limit=max(0, int(limit)))


def _clear_alerts() -> None:
    journal = AlertJournal(path=AGENT_ALERTS_JSONL, config=AlertConfig(), logger=None)
    journal.clear()


def _telegram_alert_forwarder() -> TelegramAlertForwarder:
    settings = _json_read(SETTINGS_JSON)
    cfg = TelegramAlertConfig.from_settings(settings if isinstance(settings, dict) else {})
    return TelegramAlertForwarder(
        config=cfg,
        alerts_path=AGENT_ALERTS_JSONL,
        status_path=TELEGRAM_ALERT_STATUS_JSON,
        logger=None,
    )


def _default_telegram_alert_status() -> Dict[str, Any]:
    return _telegram_alert_forwarder().snapshot()


def _load_telegram_alert_status() -> Dict[str, Any]:
    if not TELEGRAM_ALERT_STATUS_JSON.exists():
        return _default_telegram_alert_status()
    payload = _json_read(TELEGRAM_ALERT_STATUS_JSON)
    return payload if isinstance(payload, dict) and payload else _default_telegram_alert_status()


def _send_telegram_test_message(text: Optional[str] = None) -> Dict[str, Any]:
    forwarder = _telegram_alert_forwarder()
    creds = _json_read(CREDS_FILE)
    return forwarder.send_test_message(creds=creds if isinstance(creds, dict) else {}, text=text)


def _is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def _start_agent_process(trade_mode: str) -> Dict[str, Any]:
    trade_mode = str(trade_mode or "paper").strip().lower() or "paper"
    if PID_FILE.exists():
        try:
            pid_data = json.loads(PID_FILE.read_text())
            pid = int(pid_data.get("pid", 0))
            if pid and _is_process_alive(pid):
                return {"ok": False, "error": "agent already running", "pid": pid}
        except Exception:
            pass
    env = os.environ.copy()
    env["TRADE_MODE"] = trade_mode
    pybin = sys.executable or "python3"
    proc = subprocess.Popen(
        [pybin, str(AGENT_ENTRY)],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
    )
    pid_payload = {"pid": proc.pid, "trade_mode": trade_mode, "started_at": datetime.now().isoformat()}
    _json_write(PID_FILE, pid_payload)
    return {"ok": True, "pid": proc.pid, "trade_mode": trade_mode}


def _stop_agent_process(*, auto_cleanup_bkm: bool = True, graceful_wait_sec: float = 5.0) -> Dict[str, Any]:
    pid_data = _json_read(PID_FILE)
    pid = int(pid_data.get("pid", 0)) if pid_data else 0
    still_running = False
    if pid and _is_process_alive(pid):
        os.kill(pid, signal.SIGTERM)
        deadline = time.time() + max(0.0, float(graceful_wait_sec))
        while time.time() < deadline:
            if not _is_process_alive(pid):
                break
            time.sleep(0.2)
        still_running = _is_process_alive(pid)
    _json_write(PID_FILE, {})
    cleanup: Dict[str, Any] = {"attempted": False, "performed": False, "reason": "AUTO_CLEANUP_DISABLED_OR_SKIPPED"}
    if auto_cleanup_bkm and not still_running:
        cleanup = _safe_autocleanup_bkm_after_stop()
    elif auto_cleanup_bkm and still_running:
        cleanup = {"attempted": False, "performed": False, "reason": "AGENT_STILL_RUNNING"}
    return {"ok": True, "pid": pid, "agent_still_running": still_running, "cleanup": cleanup}


def _batman_bkm_tuning_paths() -> BatmanBKMTuningPaths:
    return BatmanBKMTuningPaths(
        settings_path=SETTINGS_JSON,
        strategy_state_path=STRATEGY_STATE,
        live_gate_status_path=LIVE_GATE_STATUS_JSON,
        live_gate_sessions_path=LIVE_GATE_SESSIONS_JSONL,
        advice_path=BATMAN_BKM_TUNING_ADVICE_JSON,
        history_path=BATMAN_BKM_TUNING_HISTORY_JSONL,
    )


def _load_batman_bkm_tuning_advice() -> Dict[str, Any]:
    return load_or_refresh_batman_bkm_tuning_advice(_batman_bkm_tuning_paths())


def _refresh_batman_bkm_tuning_advice() -> Dict[str, Any]:
    return refresh_batman_bkm_tuning_advice(_batman_bkm_tuning_paths())


def _apply_batman_bkm_tuning_proposal_with_safety(proposal_id: str) -> Dict[str, Any]:
    pid_data = _json_read(PID_FILE)
    pid = int(pid_data.get("pid", 0)) if pid_data else 0
    trade_mode = str(pid_data.get("trade_mode") or "").lower()
    if pid and trade_mode == "live":
        # Prefer safety over liveness detection ambiguity: a live-mode pid file is enough to block.
        raise RuntimeError("Refusing to apply tuning while agent PID file indicates live mode. Stop agent first.")
    return apply_batman_bkm_tuning_proposal(paths=_batman_bkm_tuning_paths(), proposal_id=proposal_id)


class PaperHandler(SimpleHTTPRequestHandler):
    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # type: ignore[override]
        if self.path.startswith("/api/settings"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(raw.decode("utf-8"))
                current = {}
                if SETTINGS_JSON.exists():
                    try:
                        current = json.loads(SETTINGS_JSON.read_text())
                    except Exception:
                        current = {}
                current.update(data or {})
                SETTINGS_JSON.write_text(json.dumps(current, indent=2))
                self._send_json({"ok": True, "settings": current})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/creds"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                try:
                    data = json.loads(raw.decode("utf-8"))
                except Exception:
                    data = {}
                current = _json_read(CREDS_FILE)
                current.update(data or {})
                CREDS_FILE.parent.mkdir(parents=True, exist_ok=True)
                _json_write(CREDS_FILE, current)
                self._send_json({"ok": True, "creds": current})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/strategy"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(raw.decode("utf-8"))
                if data.get("strategy_file"):
                    _json_write(LAST_STRATEGY_FILE, {"strategy_file": data["strategy_file"]})
                settings = _json_read(SETTINGS_JSON)
                settings.update(data.get("settings", {}))
                _json_write(SETTINGS_JSON, settings)
                self._send_json({"ok": True, "strategy": data})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/tuning/batman_bkm/refresh"):
            try:
                report = _refresh_batman_bkm_tuning_advice()
                self._send_json({"ok": True, "advice": report})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/tuning/batman_bkm/apply"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(raw.decode("utf-8")) if raw else {}
                proposal_id = str(data.get("proposal_id") or "").strip()
                approve = bool(data.get("approve", False))
                if not approve:
                    raise RuntimeError("Manual approval required: set approve=true")
                if not proposal_id:
                    raise RuntimeError("proposal_id is required")
                result = _apply_batman_bkm_tuning_proposal_with_safety(proposal_id)
                self._send_json(result)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/control/restart"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(raw.decode("utf-8")) if raw else {}
                trade_mode = str(data.get("trade_mode", "paper"))
                auto_cleanup = bool(data.get("auto_cleanup_bkm", True))
                stop_res = _stop_agent_process(auto_cleanup_bkm=auto_cleanup)
                if bool(stop_res.get("agent_still_running")):
                    self._send_json({"ok": False, "error": "agent still running after stop request", "stop": stop_res}, status=409)
                    return
                start_res = _start_agent_process(trade_mode=trade_mode)
                if not bool(start_res.get("ok")):
                    self._send_json({"ok": False, "error": start_res.get("error") or "restart start failed", "stop": stop_res, "start": start_res}, status=500)
                    return
                self._send_json({"ok": True, "pid": start_res.get("pid"), "trade_mode": trade_mode, "stop": stop_res})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/control/start"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(raw.decode("utf-8"))
                trade_mode = str(data.get("trade_mode", "paper"))
                res = _start_agent_process(trade_mode=trade_mode)
                status = 200 if bool(res.get("ok")) else 409
                self._send_json(res, status=status)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/control/stop"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(raw.decode("utf-8")) if raw else {}
                auto_cleanup = bool(data.get("auto_cleanup_bkm", True))
                res = _stop_agent_process(auto_cleanup_bkm=auto_cleanup)
                self._send_json(res)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/live_gate/reset"):
            try:
                status = _reset_live_gate_status()
                self._send_json({"ok": True, "live_gate": status})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/reconcile/reset"):
            try:
                status = _reset_reconcile_status()
                self._send_json({"ok": True, "reconcile": status})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/execution_recovery/reset"):
            try:
                status = _reset_execution_recovery_status()
                self._send_json({"ok": True, "execution_recovery": status})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/alerts/clear"):
            try:
                _clear_alerts()
                self._send_json({"ok": True})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/alerts/telegram/test"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(raw.decode("utf-8")) if raw else {}
                text = str(data.get("text") or "").strip() or None
                out = _send_telegram_test_message(text=text)
                self._send_json({"ok": True, "telegram": out})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/paper/close_all"):
            try:
                # Close all open paper legs by writing CLOSE rows at current LTP
                payload = load_positions(BLOTTER_CSV, mode="paper")
                legs = payload.get("positions", [])
                if not legs:
                    self._send_json({"ok": True, "closed": 0})
                    return
                now_ts = datetime.now().isoformat()
                rows_to_append = []
                for leg in legs:
                    side_close = "BUY" if leg.get("side") == "SELL" else "SELL"
                    qty = leg.get("qty") or 0
                    if qty == 0:
                        continue
                    rows_to_append.append({
                        "timestamp": now_ts,
                        "trade_mode": "paper",
                        "warn_only": "0",
                        "executed": "1",
                        "side": side_close,
                        "order_type": "MARKET",
                        "exchange_seg": "NSE_FNO",
                        "product_type": "MARGIN",
                        "security_id": leg.get("sec_id"),
                        "quantity": qty,
                        "price": leg.get("ltp") if leg.get("ltp") is not None else leg.get("entry") or 0.0,
                        "delta": "",
                        "expiry": leg.get("expiry"),
                        "strike": leg.get("strike"),
                        "tag": "MANUAL_CLOSE",
                        "notes": "CLOSE",
                    })
                # append to blotter
                header = ["timestamp","trade_mode","warn_only","executed","side","order_type","exchange_seg","product_type","security_id","quantity","price","delta","expiry","strike","tag","notes"]
                exists = BLOTTER_CSV.exists()
                with BLOTTER_CSV.open("a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=header)
                    if not exists:
                        writer.writeheader()
                    for r in rows_to_append:
                        writer.writerow(r)
                self._send_json({"ok": True, "closed": len(rows_to_append)})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        return super().do_POST()

    def do_GET(self) -> None:  # type: ignore[override]
        if self.path.startswith("/api/paper_positions"):
            try:
                payload = load_positions(BLOTTER_CSV, mode="paper")
                # also include strategy status for context
                try:
                    state = json.loads(STRATEGY_STATE.read_text())
                except Exception:
                    state = {}
                payload["strategy_state"] = state
                self._send_json(payload)
            except Exception as exc:  # pragma: no cover - defensive
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/live_positions"):
            try:
                payload = load_positions(BLOTTER_CSV, mode="live")
                broker_payload = _load_broker_live_positions()
                if not payload.get("positions") and broker_payload:
                    payload = broker_payload
                else:
                    # merge closed from broker into existing payload
                    if broker_payload and broker_payload.get("closed"):
                        closed = payload.get("closed", []) or []
                        closed.extend(broker_payload.get("closed", []))
                        payload["closed"] = closed
    # Also surface today's broker executions as closed rows (legacy)
                try:
                    trades = _load_broker_trades_today()
                    if trades:
                        closed = payload.get("closed", []) or []
                        closed.extend(trades)
                        # de-dupe prefer rows with expiry and entry
                        closed.sort(key=lambda c: (
                            c.get("expiry") not in (None, "", "0001-01-01"),
                            c.get("entry") is not None and c.get("exit") is not None,
                            c.get("side") == "FLAT"
                        ), reverse=True)
                        dedup = {}
                        for c in closed:
                            key = (str(c.get("sec_id")), c.get("strike"))
                            if key in dedup:
                                continue
                            dedup[key] = c
                        cleaned = list(dedup.values())
                        cleaned = [c for c in cleaned if c.get("side") == "FLAT"]
                        non_empty = [c for c in cleaned if c.get("expiry") not in (None, "", "0001-01-01")]
                        if non_empty:
                            cleaned = non_empty
                        payload["closed"] = cleaned
                except Exception:
                    pass
                self._send_json(payload)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/strategy_state"):
            try:
                state = json.loads(STRATEGY_STATE.read_text()) if STRATEGY_STATE.exists() else {}
                self._send_json({"strategy_state": state})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/settings"):
            try:
                settings = json.loads(SETTINGS_JSON.read_text()) if SETTINGS_JSON.exists() else {}
                self._send_json({"settings": settings})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/creds"):
            try:
                creds = _json_read(CREDS_FILE)
                self._send_json({"creds": creds})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/strategy"):
            try:
                last = _json_read(LAST_STRATEGY_FILE)
                settings = _json_read(SETTINGS_JSON)
                self._send_json({"strategy": last, "settings": settings})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/tuning/batman_bkm/advice"):
            try:
                self._send_json(_load_batman_bkm_tuning_advice())
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/live_gate/status"):
            try:
                self._send_json(_load_live_gate_status())
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/reconcile/status"):
            try:
                self._send_json(_load_reconcile_status())
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/execution_recovery/status"):
            try:
                self._send_json(_load_execution_recovery_status())
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/health"):
            try:
                self._send_json(_watchdog_health_status())
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/alerts/telegram/status"):
            try:
                self._send_json(_load_telegram_alert_status())
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/alerts"):
            try:
                tail = 100
                try:
                    from urllib.parse import urlparse, parse_qs

                    qs = parse_qs(urlparse(self.path).query)
                    if "tail" in qs:
                        tail = int(qs["tail"][0])
                except Exception:
                    pass
                alerts = _load_alerts_tail(limit=tail)
                self._send_json({"alerts": alerts, "count": len(alerts)})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/control/status"):
            try:
                pid_data = _json_read(PID_FILE)
                pid = int(pid_data.get("pid", 0)) if pid_data else 0
                running = pid and _is_process_alive(pid)
                gate = _load_live_gate_status()
                reconcile = _load_reconcile_status()
                exec_recovery = _load_execution_recovery_status()
                watchdog = _watchdog_health_status()
                alerts = _load_alerts_tail(limit=50)
                telegram_status = _load_telegram_alert_status()
                advice = _json_read(BATMAN_BKM_TUNING_ADVICE_JSON)
                proposal_count = 0
                if isinstance(advice, dict):
                    try:
                        proposal_count = len(advice.get("proposals") or [])
                    except Exception:
                        proposal_count = 0
                last_alert = alerts[-1] if alerts else {}
                self._send_json(
                    {
                        "running": bool(running),
                        "pid": pid,
                        "watchdog_status": watchdog.get("status"),
                        "heartbeat_age_sec": watchdog.get("age_sec"),
                        "last_heartbeat_at": watchdog.get("last_heartbeat_at"),
                        "live_gate_status": gate.get("status"),
                        "live_gate_stage": gate.get("stage"),
                        "locked_for_date": gate.get("locked_for_date"),
                        "sessions_total": _parse_int(gate.get("sessions_total"), 0),
                        "sessions_pass": _parse_int(gate.get("sessions_pass"), 0),
                        "sessions_fail": _parse_int(gate.get("sessions_fail"), 0),
                        "reconcile_status": reconcile.get("status"),
                        "reconcile_hard_lock": bool(reconcile.get("hard_lock", False)),
                        "reconcile_mismatch_streak": _parse_int(reconcile.get("mismatch_streak"), 0),
                        "reconcile_last_mismatch_reason": reconcile.get("last_mismatch_reason"),
                        "execution_recovery_status": exec_recovery.get("status"),
                        "execution_recovery_hard_lock": bool(exec_recovery.get("hard_lock", False)),
                        "execution_recovery_last_reason": exec_recovery.get("last_reason"),
                        "alerts_recent_count": len(alerts),
                        "last_alert_severity": last_alert.get("severity"),
                        "last_alert_code": last_alert.get("code"),
                        "telegram_alerts_enabled": bool(telegram_status.get("enabled", False)),
                        "telegram_alerts_configured": bool(telegram_status.get("configured", False)),
                        "telegram_alerts_last_sent_at": telegram_status.get("last_sent_at"),
                        "telegram_alerts_last_error": telegram_status.get("last_error"),
                        "batman_bkm_tuning_proposals_pending": proposal_count,
                        "batman_bkm_tuning_last_generated_at": advice.get("generated_at") if isinstance(advice, dict) else None,
                    }
                )
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/logs"):
            try:
                tail = 200
                try:
                    from urllib.parse import urlparse, parse_qs
                    qs = parse_qs(urlparse(self.path).query)
                    if "tail" in qs:
                        tail = int(qs["tail"][0])
                except Exception:
                    pass
                lines = []
                if AGENT_LOG.exists():
                    with AGENT_LOG.open("r") as f:
                        lines = f.readlines()[-tail:]
                self._send_json({"lines": lines})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return

        # Serve static frontend
        if self.path in ("/", "/index.html"):
            self.path = "/index.html"
        return super().do_GET()

    def translate_path(self, path: str) -> str:
        # Serve files from STATIC_DIR by default
        root = STATIC_DIR if STATIC_DIR.exists() else Path.cwd()
        # Adapted from SimpleHTTPRequestHandler: map URL to local file under root
        # while preventing path traversal.
        import posixpath
        path = path.split("?",1)[0].split("#",1)[0]
        path = posixpath.normpath(path)
        words = path.split("/")
        words = [_f for _f in words if _f]
        resolved = root
        for word in words:
            resolved = resolved / word
        return str(resolved)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper P&L mini server")
    parser.add_argument("--port", type=int, default=int(os.getenv("PAPER_PNL_PORT", "8000")))
    args = parser.parse_args()
    os.chdir(STATIC_DIR)
    httpd = HTTPServer(("", args.port), PaperHandler)
    print(f"Serving Paper P&L frontend on http://localhost:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
