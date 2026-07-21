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
import math
import os
import re
import sys
import traceback
from datetime import datetime
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import subprocess
import signal
import requests
import pandas as pd

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


class ReusableHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


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
AGENT_HEARTBEAT_JSON = STATE_DIR / "intraday_v83_heartbeat.json"
AGENT_ALERTS_JSONL = STATE_DIR / "agent_alerts.jsonl"
TELEGRAM_ALERT_STATUS_JSON = STATE_DIR / "telegram_alert_status.json"
BATMAN_BKM_AI_STATUS_JSON = STATE_DIR / "batman_bkm_ai_status.json"
BATMAN_BKM_AI_EVENTS_JSONL = STATE_DIR / "batman_bkm_ai_events.jsonl"
BATMAN_BKM_LEARNING_STATUS_JSON = STATE_DIR / "batman_bkm_learning_status.json"
BATMAN_BKM_LEARNING_OUTCOMES_JSONL = STATE_DIR / "batman_bkm_learning_outcomes.jsonl"
BATMAN_BKM_AI_PROTECT_STATUS_JSON = STATE_DIR / "batman_bkm_ai_protect_status.json"
BATMAN_BKM_AI_PROTECT_CLEAR_REQUEST_JSON = STATE_DIR / "batman_bkm_ai_protect_clear_request.json"
DECISION_COMMITTEE_STATUS_JSON = STATE_DIR / "decision_committee_status.json"
DECISION_COMMITTEE_HISTORY_JSONL = STATE_DIR / "decision_committee_history.jsonl"
DECISION_COMMITTEE_OUTCOMES_JSONL = STATE_DIR / "decision_committee_outcomes.jsonl"
INTRADAY_AI_LEARNING_STATUS_JSON = STATE_DIR / "intraday_ai_learning_status.json"
INTRADAY_AI_ADVISOR_STATUS_JSON = STATE_DIR / "intraday_ai_advisor_status.json"
INTRADAY_AI_DEPLOY_REQUEST_JSON = STATE_DIR / "intraday_ai_deploy_request.json"
INTRADAY_AI_DEPLOY_STATUS_JSON = STATE_DIR / "intraday_ai_deploy_status.json"
INTRADAY_AI_POSITION_STATUS_JSON = STATE_DIR / "intraday_ai_position_status.json"
INTRADAY_AI_TRADE_HISTORY_JSONL = STATE_DIR / "intraday_ai_trade_history.jsonl"
TELEGRAM_COMMAND_STATUS_JSON = STATE_DIR / "telegram_command_status.json"
BATMAN_BKM_TUNING_ADVICE_JSON = STATE_DIR / "batman_bkm_tuning_advice.json"
BATMAN_BKM_TUNING_HISTORY_JSONL = STATE_DIR / "batman_bkm_tuning_history.jsonl"
CREDS_FILE = STATE_DIR / "creds.json"
WEEKLY_IC_PENDING_FILE  = STATE_DIR / "weekly_ic_pending.json"
WEEKLY_IC_POSITION_FILE = STATE_DIR / "weekly_ic_position.json"
LAST_STRATEGY_FILE = STATE_DIR / "last_strategy.json"
PID_FILE = STATE_DIR / "agent.pid"
AGENT_ENTRY = ROOT / "data_engine" / "market_ai" / "start_agent.py"
V83_RUN_CONFIG_JSON = STATE_DIR / "intraday_v83_run_live_config.json"
V83_RUNNER_LOG = STATE_DIR / "intraday_v83_runner.log"
V83_PAPER_STATE_JSON = STATE_DIR / "intraday_v83_paper_state.json"
V83_RUNTIME_STATE_JSON = STATE_DIR / "intraday_v83_runtime_state.json"
V83_PAPER_LIVE_TRADES_JSONL = STATE_DIR / "intraday_v83_paper_live_trades.jsonl"
V83_RUNTIME_CONFIG_JSON = STATE_DIR / "intraday_v83_runtime_config.json"
V83_PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / "com.algoagent.intraday_v83.plist"
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
from data_engine.market_ai.modules.agents.execution_policy import (  # type: ignore
    build_execution_policy as _shared_build_execution_policy,
    intraday_ai_mode_name as _shared_intraday_ai_mode_name,
)
from data_engine.market_ai.modules.analytics.live_trade_monitor import (  # type: ignore
    find_multi_tf_zones,
    nearest_zones,
)
from data_engine.market_ai.modules.analytics.price_action_patterns import (  # type: ignore
    detect_recent_price_action,
)
from data_engine.market_ai.modules.analytics.intraday_trade_planner import (  # type: ignore
    build_trade_plan as _shared_build_trade_plan,
)
from data_engine.market_ai.intraday_defined_risk.ops_runtime import (  # type: ignore
    RuntimeMode as V83RuntimeMode,
    build_operator_status_report as _v83_operator_status_report,
    flatten_all_emergency as _v83_flatten_all_emergency,
    load_runtime_config as _v83_load_runtime_config,
    set_runtime_mode as _v83_set_runtime_mode,
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


def _intraday_ai_mode_name(settings: Dict[str, Any]) -> str:
    return _shared_intraday_ai_mode_name(settings or {})


def _intraday_ai_execution_enabled(settings: Dict[str, Any], trade_mode: str) -> bool:
    return bool(_shared_build_execution_policy(settings or {}, trade_mode).get("intraday_new_entries_allowed", False))


def _execution_policy(settings: Dict[str, Any], trade_mode: str) -> Dict[str, Any]:
    return _shared_build_execution_policy(settings or {}, trade_mode)


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
_INDEX_CHART_CACHE: Dict[Tuple[str, int], Dict[str, Any]] = {}  # (session_date, interval) -> {"ts": float, "payload": {...}}


def _fetch_ltp_lookup(pairs: Dict[str, List[int]]) -> Tuple[Dict[Tuple[str, str], float], str]:
    if not pairs:
        return {}, "skip"
    creds = _json_read(CREDS_FILE)
    client_id = str(creds.get("client_id") or "").strip()
    access_token = str(creds.get("access_token") or "").strip()
    if not client_id or not access_token:
        return {}, "no-creds"

    global _LAST_LTP_FETCH_TS, _NEXT_LTP_ALLOWED_TS
    now_ts = time.time()
    if now_ts < _NEXT_LTP_ALLOWED_TS or (now_ts - _LAST_LTP_FETCH_TS) < 4:
        return {}, "cached"

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "client-id": client_id,
        "access-token": access_token,
    }
    payload: Dict[str, Any] = {seg: sorted({int(i) for i in ids}) for seg, ids in pairs.items() if ids}
    payload["dhanClientId"] = client_id
    base = os.getenv("DHAN_API_BASE", "https://api.dhan.co").rstrip("/")

    try:
        resp = requests.post(f"{base}/v2/marketfeed/ltp", headers=headers, json=payload, timeout=8)
        if resp.status_code == 429:
            _NEXT_LTP_ALLOWED_TS = now_ts + 10
            return {}, "throttled:429 (cache)"
        if resp.status_code == 401:
            return {}, "auth-error: refresh dhan access token"
        if resp.status_code == 403:
            return {}, "auth-error: check static IP / token"
        if resp.status_code >= 400:
            return {}, f"http{resp.status_code}"

        lookup: Dict[Tuple[str, str], float] = {}
        data = (resp.json() or {}).get("data", {})
        if isinstance(data, dict):
            for seg, sec_map in data.items():
                if not isinstance(sec_map, dict):
                    continue
                for sid, node in sec_map.items():
                    if not isinstance(node, dict):
                        continue
                    ltp = node.get("last_price")
                    if ltp is None:
                        continue
                    try:
                        key = (str(seg), str(sid))
                        lookup[key] = float(ltp)
                        _LTP_CACHE[key] = (float(ltp), now_ts)
                    except Exception:
                        continue
        _LAST_LTP_FETCH_TS = now_ts
        return lookup, ("ok" if lookup else "empty")
    except Exception as exc:
        return {}, f"error: {exc}"


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
    if latest_expiry and mode != "paper":
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
    if pairs and mode != "paper":
        # dedupe sec_ids per segment to avoid API 400 on duplicates
        for seg, ids in pairs.items():
            pairs[seg] = sorted(list({int(i) for i in ids}))
        ltp_lookup, ltp_status = _fetch_ltp_lookup(pairs)
        if ltp_status.startswith("error:"):
            print(f"[ltp_enrich] failed: {ltp_status}")

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
            pnl = (price - entry_px) * close_qty if close_side == "SELL" else (entry_px - price) * close_qty
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
            signed_close = close_qty if side == "SELL" else -close_qty
            remaining_qty = int(leg.get("entry_qty", 0) or 0) + int(signed_close)
            leg["entry_qty"] = remaining_qty
            leg["qty"] = remaining_qty
            if remaining_qty == 0:
                leg["side"] = None
            else:
                leg["side"] = "SELL" if remaining_qty > 0 else "BUY"

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
        if qty <= 0:
            continue
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
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        print(f"[live_fallback] get_positions_raw failed: {exc}")
        return {"positions": [], "total_pnl": 0.0, "source": "none", "margin_available": None, "margin_used": None}

    positions: List[Dict[str, Any]] = []
    closed: List[Dict[str, Any]] = []
    seg_map: Dict[str, list[int]] = {}
    full_by_sid: Dict[str, Dict[str, Any]] = {}
    for fr in positions_full or []:
        if not isinstance(fr, dict):
            continue
        sid = fr.get("securityId") or fr.get("security_id") or fr.get("instrumentId") or ""
        sid_s = str(sid).strip()
        if sid_s:
            full_by_sid[sid_s] = fr

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
        sec_id = row.get("securityId") or row.get("security_id") or row.get("instrumentId") or ""
        full_row = full_by_sid.get(str(sec_id).strip(), {})
        expiry = (
            row.get("drvExpiryDate")
            or row.get("expiryDate")
            or row.get("expiry")
            or row.get("drv_expiry_date")
            or (full_row.get("drvExpiryDate") if isinstance(full_row, dict) else None)
            or (full_row.get("expiryDate") if isinstance(full_row, dict) else None)
            or (full_row.get("expiry") if isinstance(full_row, dict) else None)
            or ""
        )
        strike = (
            row.get("tradingSymbol")
            or row.get("trading_symbol")
            or row.get("displayName")
            or row.get("display_name")
            or row.get("symbol")
            or row.get("strikePrice")
            or row.get("strike")
            or (full_row.get("tradingSymbol") if isinstance(full_row, dict) else None)
            or (full_row.get("symbol") if isinstance(full_row, dict) else None)
            or (full_row.get("displayName") if isinstance(full_row, dict) else None)
            or (full_row.get("strikePrice") if isinstance(full_row, dict) else None)
            or ""
        )
        if sec_id:
            try:
                seg_map.setdefault(seg, []).append(int(sec_id))
            except Exception:
                pass
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
        if seg_map:
            ltp_lookup, ltp_status = _fetch_ltp_lookup(seg_map)
            for pos in positions:
                seg = next(iter(seg_map.keys())) if len(seg_map) == 1 else "NSE_FNO"
                sid = pos.get("sec_id")
                ltp = ltp_lookup.get((seg, str(sid)))
                if ltp is not None:
                    pos["ltp"] = float(ltp)
            if ltp_status.startswith("auth-error"):
                print(f"[live_ltp_fallback] {ltp_status}")
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


def _default_bkm_ai_status() -> Dict[str, Any]:
    now = _ist_now()
    return {
        "status": "IDLE",
        "mode": "ADVISOR",
        "action": "HOLD",
        "severity": "INFO",
        "score": 0.0,
        "confidence": 0.0,
        "reasons": [],
        "market_context": {},
        "plan": {},
        "current_session_date": now.date().isoformat(),
        "updated_at": now.isoformat(timespec="seconds"),
    }


def _load_bkm_ai_status() -> Dict[str, Any]:
    if not BATMAN_BKM_AI_STATUS_JSON.exists():
        return _default_bkm_ai_status()
    payload = _json_read(BATMAN_BKM_AI_STATUS_JSON)
    return payload if isinstance(payload, dict) and payload else _default_bkm_ai_status()


def _default_bkm_learning_status() -> Dict[str, Any]:
    now = _ist_now()
    return {
        "status": "IDLE",
        "total_outcomes": 0,
        "wins": 0,
        "losses": 0,
        "flats": 0,
        "realized_pnl_rs": 0.0,
        "last_outcome_at": None,
        "last_outcome_id": None,
        "last_entry_assessment": {},
        "top_negative_buckets": [],
        "top_positive_buckets": [],
        "updated_at": now.isoformat(timespec="seconds"),
    }


def _load_bkm_learning_status() -> Dict[str, Any]:
    payload = _json_read(BATMAN_BKM_LEARNING_STATUS_JSON)
    if not payload or not isinstance(payload, dict):
        return _default_bkm_learning_status()
    out = _default_bkm_learning_status()
    out.update(payload)
    out["last_entry_assessment"] = out.get("last_entry_assessment") if isinstance(out.get("last_entry_assessment"), dict) else {}
    out["top_negative_buckets"] = out.get("top_negative_buckets") if isinstance(out.get("top_negative_buckets"), list) else []
    out["top_positive_buckets"] = out.get("top_positive_buckets") if isinstance(out.get("top_positive_buckets"), list) else []
    return out


def _default_bkm_ai_protect_status() -> Dict[str, Any]:
    now = _ist_now()
    return {
        "status": "UNLOCKED",
        "active": False,
        "protection_enabled": True,
        "session_date": now.date().isoformat(),
        "reason": None,
        "source_action": None,
        "score": None,
        "expiry": None,
        "updated_at": now.isoformat(timespec="seconds"),
        "last_transition_at": now.isoformat(timespec="seconds"),
    }


def _load_bkm_ai_protect_status() -> Dict[str, Any]:
    payload = _json_read(BATMAN_BKM_AI_PROTECT_STATUS_JSON)
    if not payload or not isinstance(payload, dict):
        return _default_bkm_ai_protect_status()
    out = _default_bkm_ai_protect_status()
    out.update(payload)
    out["active"] = bool(out.get("active", False))
    out["protection_enabled"] = bool(out.get("protection_enabled", True))
    out["status"] = "LOCKED" if out["active"] else "UNLOCKED"
    return out


def _default_decision_committee_status() -> Dict[str, Any]:
    now = _ist_now()
    return {
        "status": "IDLE",
        "focus": "INTRADAY",
        "verdict": "IDLE",
        "consensus_bias": "NEUTRAL",
        "ensemble_confidence": 0.0,
        "advisor_signal": "NO_TRADE",
        "advisor_strategy": None,
        "expiry": None,
        "spot": None,
        "has_open_bkm": False,
        "current_session_date": now.date().isoformat(),
        "updated_at": now.isoformat(timespec="seconds"),
        "components": {},
        "reasons": [],
    }


def _load_decision_committee_status() -> Dict[str, Any]:
    payload = _json_read(DECISION_COMMITTEE_STATUS_JSON)
    if not payload or not isinstance(payload, dict):
        return _default_decision_committee_status()
    out = _default_decision_committee_status()
    out.update(payload)
    out["components"] = out.get("components") if isinstance(out.get("components"), dict) else {}
    out["reasons"] = out.get("reasons") if isinstance(out.get("reasons"), list) else []
    return out


def _default_intraday_learning_status() -> Dict[str, Any]:
    now = _ist_now()
    return {
        "status": "IDLE",
        "total_outcomes": 0,
        "wins": 0,
        "losses": 0,
        "flats": 0,
        "realized_pnl_rs": 0.0,
        "last_outcome_at": None,
        "last_assessment": {},
        "top_negative_buckets": [],
        "top_positive_buckets": [],
        "updated_at": now.isoformat(timespec="seconds"),
    }


def _load_intraday_learning_status() -> Dict[str, Any]:
    payload = _json_read(INTRADAY_AI_LEARNING_STATUS_JSON)
    if not payload or not isinstance(payload, dict):
        return _default_intraday_learning_status()
    out = _default_intraday_learning_status()
    out.update(payload)
    out["last_assessment"] = out.get("last_assessment") if isinstance(out.get("last_assessment"), dict) else {}
    out["top_negative_buckets"] = out.get("top_negative_buckets") if isinstance(out.get("top_negative_buckets"), list) else []
    out["top_positive_buckets"] = out.get("top_positive_buckets") if isinstance(out.get("top_positive_buckets"), list) else []
    return out


def _load_jsonl_tail(path: Path, limit: int = 50) -> List[Dict[str, Any]]:
    if not path.exists() or limit <= 0:
        return []
    rows: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    if len(rows) <= limit:
        return rows
    return rows[-limit:]


def _default_intraday_ai_status() -> Dict[str, Any]:
    now = _ist_now()
    return {
        "status": "IDLE",
        "signal": "NO_TRADE",
        "priority": "INFO",
        "strategy": None,
        "expiry": None,
        "spot": None,
        "market_bias": "NEUTRAL",
        "trend_confidence": 0.0,
        "signal_conflict_score": 0.0,
        "pcr_unbalanced": False,
        "pcr_unbalanced_side": "NEUTRAL",
        "volatility_regime": "UNKNOWN",
        "breakout_confirmation": "NONE",
        "has_open_bkm": False,
        "recommendation": {},
        "reasons": [],
        "current_session_date": now.date().isoformat(),
        "updated_at": now.isoformat(timespec="seconds"),
    }


def _load_intraday_ai_status() -> Dict[str, Any]:
    if not INTRADAY_AI_ADVISOR_STATUS_JSON.exists():
        return _default_intraday_ai_status()
    payload = _json_read(INTRADAY_AI_ADVISOR_STATUS_JSON)
    return payload if isinstance(payload, dict) and payload else _default_intraday_ai_status()


def _default_intraday_ai_deploy_status() -> Dict[str, Any]:
    now = _ist_now()
    return {
        "status": "IDLE",
        "pending": False,
        "request_id": None,
        "requested_at": None,
        "request_source": None,
        "request_note": None,
        "processed_at": None,
        "result_code": None,
        "message": None,
        "details": {},
        "current_session_date": now.date().isoformat(),
        "updated_at": now.isoformat(timespec="seconds"),
    }


def _load_intraday_ai_deploy_status() -> Dict[str, Any]:
    payload = _json_read(INTRADAY_AI_DEPLOY_STATUS_JSON)
    if not payload or not isinstance(payload, dict):
        return _default_intraday_ai_deploy_status()
    out = _default_intraday_ai_deploy_status()
    out.update(payload)
    out["pending"] = bool(out.get("pending", False))
    out["details"] = out.get("details") if isinstance(out.get("details"), dict) else {}
    return out


def _load_intraday_ai_position_status() -> Dict[str, Any]:
    payload = _json_read(INTRADAY_AI_POSITION_STATUS_JSON)
    if isinstance(payload, dict) and payload:
        return payload
    now = _ist_now()
    return {
        "status": "IDLE",
        "position_id": None,
        "trade_mode": None,
        "strategy_type": None,
        "strategy_label": None,
        "expiry": None,
        "opened_at": None,
        "closed_at": None,
        "current_session_date": now.date().isoformat(),
        "last_reason": None,
        "entry_spot": None,
        "current_spot": None,
        "current_pnl_rs": 0.0,
        "peak_pnl_rs": 0.0,
        "trail_floor_rs": None,
        "trailing_enabled": True,
        "trailing_active": False,
        "sl_total_rs": 0.0,
        "tp_total_rs": 0.0,
        "invalidation_spot_level": None,
        "max_hold_till": "15:05",
        "lots_multiplier": 1,
        "session_trade_count": 0,
        "entry_features": {},
        "legs": [],
        "last_evaluated_at": None,
        "updated_at": now.isoformat(timespec="seconds"),
    }


def _init_dhan_wrapper() -> Any:
    creds = _json_read(CREDS_FILE)
    client_id = str(creds.get("client_id") or "").strip()
    access_token = str(creds.get("access_token") or "").strip()
    if not client_id or not access_token:
        raise RuntimeError("DHAN_CREDS_MISSING")
    os.environ["DHAN_CLIENT_ID"] = client_id
    os.environ["DHAN_ACCESS_TOKEN"] = access_token
    from data_engine.market_ai.dhan_wrapper import DhanWrapper  # type: ignore

    return DhanWrapper(logger=None)


def _parse_candle_timestamp(raw: Any) -> Optional[datetime]:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, (int, float)):
        try:
            return datetime.fromtimestamp(float(raw))
        except Exception:
            return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except Exception:
        return None


def _normalize_intraday_candles(raw: List[dict], *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for candle in raw or []:
        if not isinstance(candle, dict):
            continue
        ts = _parse_candle_timestamp(candle.get("timestamp") or candle.get("time"))
        if ts is None:
            continue
        open_px = _parse_float(candle.get("open"), None)
        high_px = _parse_float(candle.get("high"), None)
        low_px = _parse_float(candle.get("low"), None)
        close_px = _parse_float(candle.get("close"), None)
        if None in (open_px, high_px, low_px, close_px):
            continue
        rows.append(
            {
                "timestamp": ts,
                "open": float(open_px),
                "high": float(high_px),
                "low": float(low_px),
                "close": float(close_px),
                "volume": _parse_float(candle.get("volume"), 0.0),
            }
        )
    rows.sort(key=lambda row: row["timestamp"])
    if limit is not None and limit > 0:
        rows = rows[-limit:]
    return rows


def _ema_series(values: List[float], period: int) -> List[Optional[float]]:
    if not values:
        return []
    alpha = 2.0 / (float(period) + 1.0)
    out: List[Optional[float]] = []
    ema_val: Optional[float] = None
    for value in values:
        try:
            numeric = float(value)
        except Exception:
            out.append(None)
            continue
        ema_val = numeric if ema_val is None else ((numeric * alpha) + (ema_val * (1.0 - alpha)))
        out.append(round(float(ema_val), 2))
    return out


def _round_strike(value: Optional[float], *, direction: str) -> Optional[int]:
    if value is None:
        return None
    step = 50.0
    numeric = float(value)
    if direction == "up":
        return int(math.ceil(numeric / step) * step)
    if direction == "down":
        return int(math.floor(numeric / step) * step)
    return int(round(numeric / step) * step)


def _price_action_snapshot(candles: List[dict], *, interval_min: int) -> Dict[str, Any]:
    rows = [c for c in candles if c]
    closes = [float(c.get("close") or 0.0) for c in rows if c.get("close") not in (None, "")]
    highs = [float(c.get("high") or c.get("close") or 0.0) for c in rows if c.get("high") is not None or c.get("close") is not None]
    lows = [float(c.get("low") or c.get("close") or 0.0) for c in rows if c.get("low") is not None or c.get("close") is not None]
    vols = [float(c.get("volume") or 0.0) for c in rows if c.get("volume") not in (None, "")]
    if len(closes) < 2:
        return {
            "interval_min": interval_min,
            "trend": "UNKNOWN",
            "pattern": "UNKNOWN",
            "bars": len(closes),
            "change_points": None,
            "change_pct": None,
            "range_points": None,
            "close_position_in_range": None,
            "dir_score": 0.0,
            "close": closes[-1] if closes else None,
            "support": None,
            "resistance": None,
            "distance_to_support": None,
            "distance_to_resistance": None,
            "atr_like_points": None,
            "atr_like_pct": None,
            "ema_fast": None,
            "ema_slow": None,
            "ema_alignment": "NEUTRAL",
            "volatility_regime": "UNKNOWN",
            "breakout_dir": "NONE",
            "breakout_confirmed": False,
            "price_action_bias": "NEUTRAL",
            "price_action_confirmation": "NONE",
            "price_action_score": 0.0,
            "primary_pattern": "NONE",
            "price_action_patterns": [],
            "retest_status": "NONE",
            "retest_bias": "NEUTRAL",
            "retest_level": None,
            "retest_score": 0.0,
        }

    lookback = min(len(closes), 12 if interval_min <= 5 else (10 if interval_min <= 15 else 8))
    window = closes[-lookback:]
    window_vols = vols[-lookback:] if vols else []
    first_close = float(window[0])
    last_close = float(window[-1])
    change_points = last_close - first_close
    change_pct = (change_points / max(1.0, abs(first_close))) * 100.0
    window_high = max(highs[-lookback:]) if highs else max(window)
    window_low = min(lows[-lookback:]) if lows else min(window)
    range_points = max(0.0, float(window_high - window_low))
    close_pos = ((last_close - window_low) / range_points) if range_points > 0 else None
    support = round(float(window_low), 2)
    resistance = round(float(window_high), 2)
    dist_support = max(0.0, float(last_close - window_low))
    dist_resistance = max(0.0, float(window_high - last_close))

    bar_ranges: List[float] = []
    for candle in rows[-lookback:]:
        try:
            high_px = float(candle.get("high") or candle.get("close") or 0.0)
            low_px = float(candle.get("low") or candle.get("close") or 0.0)
            bar_ranges.append(max(0.0, high_px - low_px))
        except Exception:
            continue
    atr_like_points = (sum(bar_ranges) / len(bar_ranges)) if bar_ranges else 0.0
    atr_like_pct = ((atr_like_points / max(1.0, abs(last_close))) * 100.0) if last_close else 0.0
    if atr_like_pct >= (0.32 if interval_min <= 5 else (0.45 if interval_min <= 15 else 0.65)):
        vol_regime = "HIGH"
    elif atr_like_pct <= (0.14 if interval_min <= 5 else (0.20 if interval_min <= 15 else 0.30)):
        vol_regime = "LOW"
    else:
        vol_regime = "NORMAL"

    prior_high = max(window[:-1]) if len(window) >= 2 else last_close
    prior_low = min(window[:-1]) if len(window) >= 2 else last_close
    breakout_dir = "NONE"
    breakout_confirmed = False
    if len(window) >= 3:
        vol_avg = (sum(window_vols[:-1]) / len(window_vols[:-1])) if len(window_vols) >= 2 else 0.0
        vol_last = float(window_vols[-1]) if window_vols else 0.0
        vol_confirm = True if vol_avg <= 0 else (vol_last >= (1.15 * vol_avg))
        if last_close > prior_high:
            breakout_dir = "UP"
            breakout_confirmed = bool(vol_confirm)
        elif last_close < prior_low:
            breakout_dir = "DOWN"
            breakout_confirmed = bool(vol_confirm)

    ema_fast_series = _ema_series(closes, 5)
    ema_slow_series = _ema_series(closes, 20)
    ema_fast = ema_fast_series[-1] if ema_fast_series else None
    ema_slow = ema_slow_series[-1] if ema_slow_series else None
    ema_fast_prev = ema_fast_series[-2] if len(ema_fast_series) >= 2 else ema_fast
    ema_slow_prev = ema_slow_series[-2] if len(ema_slow_series) >= 2 else ema_slow
    ema_gap_pct = None
    ema_alignment = "NEUTRAL"
    if ema_fast is not None and ema_slow is not None and ema_fast_prev is not None and ema_slow_prev is not None:
        ema_gap = float(ema_fast - ema_slow)
        ema_gap_pct = ema_gap / max(1.0, abs(last_close))
        ema_fast_slope = float(ema_fast - ema_fast_prev)
        ema_slow_slope = float(ema_slow - ema_slow_prev)
        if ema_gap_pct >= 0.0008 and ema_fast_slope > 0 and ema_slow_slope >= 0:
            ema_alignment = "BULLISH"
        elif ema_gap_pct <= -0.0008 and ema_fast_slope < 0 and ema_slow_slope <= 0:
            ema_alignment = "BEARISH"

    trend_threshold_pct = 0.10 if interval_min <= 5 else (0.18 if interval_min <= 15 else 0.28)
    trend = "RANGE"
    if change_pct >= trend_threshold_pct:
        trend = "UP"
    elif change_pct <= -trend_threshold_pct:
        trend = "DOWN"
    pattern = "RANGE"
    if trend == "UP":
        pattern = "UPTREND" if (close_pos is not None and close_pos >= 0.65) else "UP_BIAS"
    elif trend == "DOWN":
        pattern = "DOWNTREND" if (close_pos is not None and close_pos <= 0.35) else "DOWN_BIAS"

    dir_score = 1.0 if trend == "UP" else (-1.0 if trend == "DOWN" else 0.0)
    if trend == "RANGE" and close_pos is not None:
        dir_score = round((close_pos - 0.5) * 0.6, 3)

    price_action = detect_recent_price_action(
        rows[-max(lookback, 6):],
        support=support,
        resistance=resistance,
        atr_like_points=atr_like_points,
        ema_slow=ema_slow,
    )
    return {
        "interval_min": interval_min,
        "trend": trend,
        "pattern": pattern,
        "bars": len(window),
        "change_points": round(float(change_points), 2),
        "change_pct": round(float(change_pct), 3),
        "range_points": round(float(range_points), 2),
        "close_position_in_range": None if close_pos is None else round(float(close_pos), 3),
        "dir_score": round(float(dir_score), 3),
        "close": round(float(last_close), 2),
        "support": support,
        "resistance": resistance,
        "distance_to_support": round(float(dist_support), 2),
        "distance_to_resistance": round(float(dist_resistance), 2),
        "atr_like_points": round(float(atr_like_points), 2),
        "atr_like_pct": round(float(atr_like_pct), 3),
        "ema_fast": None if ema_fast is None else round(float(ema_fast), 2),
        "ema_slow": None if ema_slow is None else round(float(ema_slow), 2),
        "ema_alignment": ema_alignment,
        "volatility_regime": vol_regime,
        "breakout_dir": breakout_dir,
        "breakout_confirmed": bool(breakout_confirmed),
        "price_action_bias": str(price_action.get("price_action_bias") or "NEUTRAL"),
        "price_action_confirmation": str(price_action.get("price_action_confirmation") or "NONE"),
        "price_action_score": round(float(price_action.get("price_action_score") or 0.0), 3),
        "primary_pattern": str(price_action.get("primary_pattern") or "NONE"),
        "price_action_patterns": list(price_action.get("price_action_patterns") or []),
        "retest_status": str(price_action.get("retest_status") or "NONE"),
        "retest_bias": str(price_action.get("retest_bias") or "NEUTRAL"),
        "retest_level": price_action.get("retest_level"),
        "retest_score": round(float(price_action.get("retest_score") or 0.0), 3),
    }


def _candles_to_zone_frame(candles: List[dict], *, timeframe: str) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for candle in candles or []:
        ts = _parse_candle_timestamp(candle.get("timestamp"))
        if ts is None:
            continue
        rows.append(
            {
                "timestamp": ts,
                "open": float(candle.get("open") or 0.0),
                "high": float(candle.get("high") or candle.get("open") or 0.0),
                "low": float(candle.get("low") or candle.get("open") or 0.0),
                "close": float(candle.get("close") or candle.get("open") or 0.0),
            }
        )
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])
    frame.attrs["timeframe"] = timeframe
    return frame


def _zone_to_payload(zone: Any) -> Optional[Dict[str, Any]]:
    if zone is None:
        return None
    timestamp = getattr(zone, "timestamp", None)
    return {
        "side": getattr(zone, "side", None),
        "price": round(float(getattr(zone, "price", 0.0)), 2),
        "timeframe": getattr(zone, "timeframe", None),
        "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else None,
    }


def _compute_orb_state(candles_5m: List[dict], *, orb_minutes: int = 15) -> Dict[str, Any]:
    if not candles_5m:
        return {
            "timeframe_minutes": orb_minutes,
            "orb_high": None,
            "orb_low": None,
            "orb_mid": None,
            "completed_at": None,
            "breakout_confirmation": "NONE",
            "breakout_active": False,
            "breakout_reason": None,
        }
    first_ts = candles_5m[0]["timestamp"]
    session_start = first_ts.replace(hour=9, minute=15, second=0, microsecond=0)
    opening_bars = [
        candle
        for candle in candles_5m
        if isinstance(candle.get("timestamp"), datetime)
        and session_start <= candle["timestamp"] < (session_start + pd.Timedelta(minutes=orb_minutes))
    ]
    if not opening_bars:
        opening_bars = candles_5m[: max(1, orb_minutes // 5)]
    orb_high = max(float(candle.get("high") or 0.0) for candle in opening_bars) if opening_bars else None
    orb_low = min(float(candle.get("low") or 0.0) for candle in opening_bars) if opening_bars else None
    orb_mid = None if orb_high is None or orb_low is None else round((orb_high + orb_low) / 2.0, 2)
    completed_at = opening_bars[-1]["timestamp"].isoformat() if opening_bars else None
    last_close = float(candles_5m[-1].get("close") or 0.0) if candles_5m else None
    avg_range = sum(max(0.0, float(c.get("high") or 0.0) - float(c.get("low") or 0.0)) for c in candles_5m[-6:]) / max(1, len(candles_5m[-6:]))
    buffer = max(5.0, avg_range * 0.15)
    breakout_confirmation = "NONE"
    breakout_reason = None
    if last_close is not None and orb_high is not None and last_close >= orb_high + buffer:
        breakout_confirmation = "UP_CONFIRMED"
        breakout_reason = "price accepted above opening range high"
    elif last_close is not None and orb_low is not None and last_close <= orb_low - buffer:
        breakout_confirmation = "DOWN_CONFIRMED"
        breakout_reason = "price accepted below opening range low"
    return {
        "timeframe_minutes": orb_minutes,
        "orb_high": None if orb_high is None else round(float(orb_high), 2),
        "orb_low": None if orb_low is None else round(float(orb_low), 2),
        "orb_mid": orb_mid,
        "completed_at": completed_at,
        "breakout_confirmation": breakout_confirmation,
        "breakout_active": breakout_confirmation != "NONE",
        "breakout_reason": breakout_reason,
    }


def _rsi_series(values: List[float], period: int = 14) -> List[Optional[float]]:
    if not values:
        return []
    out: List[Optional[float]] = [None] * len(values)
    if len(values) <= period:
        return out
    gains: List[float] = []
    losses: List[float] = []
    for idx in range(1, len(values)):
        change = float(values[idx]) - float(values[idx - 1])
        gains.append(max(change, 0.0))
        losses.append(max(-change, 0.0))
    avg_gain = sum(gains[:period]) / float(period)
    avg_loss = sum(losses[:period]) / float(period)
    rs = avg_gain / avg_loss if avg_loss > 0 else float("inf")
    out[period] = round(100.0 - (100.0 / (1.0 + rs)), 2)
    for idx in range(period + 1, len(values)):
        gain = gains[idx - 1]
        loss = losses[idx - 1]
        avg_gain = ((avg_gain * (period - 1)) + gain) / float(period)
        avg_loss = ((avg_loss * (period - 1)) + loss) / float(period)
        rs = avg_gain / avg_loss if avg_loss > 0 else float("inf")
        out[idx] = round(100.0 - (100.0 / (1.0 + rs)), 2)
    return out


def _last_two_price_pivots(candles: List[dict], *, side: str) -> List[Tuple[int, float]]:
    if len(candles) < 5:
        return []
    out: List[Tuple[int, float]] = []
    for idx in range(1, len(candles) - 1):
        prev_row = candles[idx - 1]
        row = candles[idx]
        next_row = candles[idx + 1]
        if side == "low":
            if float(row["low"]) <= float(prev_row["low"]) and float(row["low"]) <= float(next_row["low"]):
                out.append((idx, float(row["low"])))
        else:
            if float(row["high"]) >= float(prev_row["high"]) and float(row["high"]) >= float(next_row["high"]):
                out.append((idx, float(row["high"])))
    return out[-2:]


def _detect_rsi_divergence(candles: List[dict], rsi_values: List[Optional[float]]) -> Dict[str, Any]:
    default = {
        "status": "NONE",
        "bias": "NEUTRAL",
        "strength": 0.0,
        "description": "No RSI divergence.",
    }
    if len(candles) < 5 or len(rsi_values) != len(candles):
        return default

    lows = _last_two_price_pivots(candles, side="low")
    if len(lows) == 2:
        (idx1, low1), (idx2, low2) = lows
        rsi1 = rsi_values[idx1]
        rsi2 = rsi_values[idx2]
        if rsi1 is not None and rsi2 is not None and low2 < low1 and rsi2 > rsi1 + 1.0:
            strength = min(1.0, max(0.0, ((rsi2 - rsi1) / 12.0)))
            return {
                "status": "BULLISH_DIVERGENCE",
                "bias": "BULLISH",
                "strength": round(strength, 3),
                "description": "Price made a lower low while RSI made a higher low.",
            }

    highs = _last_two_price_pivots(candles, side="high")
    if len(highs) == 2:
        (idx1, high1), (idx2, high2) = highs
        rsi1 = rsi_values[idx1]
        rsi2 = rsi_values[idx2]
        if rsi1 is not None and rsi2 is not None and high2 > high1 and rsi2 < rsi1 - 1.0:
            strength = min(1.0, max(0.0, ((rsi1 - rsi2) / 12.0)))
            return {
                "status": "BEARISH_DIVERGENCE",
                "bias": "BEARISH",
                "strength": round(strength, 3),
                "description": "Price made a higher high while RSI made a lower high.",
            }
    return default


def _detect_orderflow_zones(
    *,
    candles_5m: List[dict],
    spot: float,
    nearest_support: Optional[Dict[str, Any]],
    nearest_resistance: Optional[Dict[str, Any]],
    snap_5m: Dict[str, Any],
) -> Dict[str, Any]:
    default_zone = {"active": False, "level": None, "strength": 0.0, "summary": "No clear participation zone yet."}
    if not candles_5m:
        return {"buyers": dict(default_zone), "sellers": dict(default_zone)}
    last = candles_5m[-1]
    avg_volume = sum(float(c.get("volume") or 0.0) for c in candles_5m[-6:]) / max(1, len(candles_5m[-6:]))
    volume_ratio = (float(last.get("volume") or 0.0) / avg_volume) if avg_volume > 0 else 1.0
    body = abs(float(last["close"]) - float(last["open"]))
    lower_wick = max(0.0, min(float(last["open"]), float(last["close"])) - float(last["low"]))
    upper_wick = max(0.0, float(last["high"]) - max(float(last["open"]), float(last["close"])))
    atr = max(10.0, float(snap_5m.get("atr_like_points") or 0.0))
    support_level = _parse_float(nearest_support.get("price"), None) if nearest_support else None
    resistance_level = _parse_float(nearest_resistance.get("price"), None) if nearest_resistance else None
    buyers = dict(default_zone)
    sellers = dict(default_zone)

    if support_level is not None:
        near_support = abs(spot - support_level) <= max(15.0, atr * 0.55)
        strong_lower_rejection = lower_wick >= max(body * 1.2, atr * 0.22)
        if near_support and strong_lower_rejection:
            strength = min(1.0, 0.45 + max(0.0, min(0.35, (volume_ratio - 1.0) * 0.4)) + 0.15)
            buyers = {
                "active": True,
                "level": round(float(support_level), 2),
                "strength": round(strength, 3),
                "summary": f"Buyers defended near {support_level:.0f} with lower-wick rejection.",
            }

    if resistance_level is not None:
        near_resistance = abs(resistance_level - spot) <= max(15.0, atr * 0.55)
        strong_upper_rejection = upper_wick >= max(body * 1.2, atr * 0.22)
        if near_resistance and strong_upper_rejection:
            strength = min(1.0, 0.45 + max(0.0, min(0.35, (volume_ratio - 1.0) * 0.4)) + 0.15)
            sellers = {
                "active": True,
                "level": round(float(resistance_level), 2),
                "strength": round(strength, 3),
                "summary": f"Sellers rejected price near {resistance_level:.0f} with upper-wick supply.",
            }
    return {"buyers": buyers, "sellers": sellers}


def _build_higher_tf_confluence(
    *,
    snap_15m: Dict[str, Any],
    snap_60m: Dict[str, Any],
    market_bias: str,
    rsi_divergence: Dict[str, Any],
) -> Dict[str, Any]:
    alignments: List[str] = []
    score = 0.0
    patterns = [str(snap_15m.get("pattern") or "UNKNOWN"), str(snap_60m.get("pattern") or "UNKNOWN")]
    ema_alignments = [str(snap_15m.get("ema_alignment") or "NEUTRAL"), str(snap_60m.get("ema_alignment") or "NEUTRAL")]
    price_action = [str(snap_15m.get("price_action_confirmation") or "NONE"), str(snap_60m.get("price_action_confirmation") or "NONE")]
    if market_bias == "BEARISH":
        if str(snap_15m.get("pattern") or "").upper() in {"DOWNTREND", "DOWN_BIAS"}:
            score += 0.35
            alignments.append("15m trend supports bearish idea")
        if str(snap_60m.get("pattern") or "").upper() in {"DOWNTREND", "DOWN_BIAS"}:
            score += 0.4
            alignments.append("60m structure is also bearish")
        if str(snap_15m.get("ema_alignment") or "").upper() == "BEARISH":
            score += 0.2
            alignments.append("15m EMA(5/20) is bearish")
        if str(snap_60m.get("ema_alignment") or "").upper() == "BEARISH":
            score += 0.2
            alignments.append("60m EMA slope is bearish")
        if str(rsi_divergence.get("bias") or "").upper() == "BEARISH":
            score += 0.2
            alignments.append("RSI bearish divergence agrees with short idea")
    elif market_bias == "BULLISH":
        if str(snap_15m.get("pattern") or "").upper() in {"UPTREND", "UP_BIAS"}:
            score += 0.35
            alignments.append("15m trend supports bullish idea")
        if str(snap_60m.get("pattern") or "").upper() in {"UPTREND", "UP_BIAS"}:
            score += 0.4
            alignments.append("60m structure is also bullish")
        if str(snap_15m.get("ema_alignment") or "").upper() == "BULLISH":
            score += 0.2
            alignments.append("15m EMA(5/20) is bullish")
        if str(snap_60m.get("ema_alignment") or "").upper() == "BULLISH":
            score += 0.2
            alignments.append("60m EMA slope is bullish")
        if str(rsi_divergence.get("bias") or "").upper() == "BULLISH":
            score += 0.2
            alignments.append("RSI bullish divergence agrees with long-supportive idea")
    if not alignments:
        alignments.append("Higher timeframe structure is mixed.")
    return {
        "status": "ALIGNED" if score >= 0.75 else ("PARTIAL" if score >= 0.35 else "MIXED"),
        "score": round(min(1.0, score), 3),
        "patterns": patterns,
        "ema_alignments": ema_alignments,
        "price_action_confirmations": price_action,
        "notes": alignments[:4],
    }


def _market_bias_from_snapshot(snap_5m: Dict[str, Any], snap_15m: Dict[str, Any], snap_60m: Dict[str, Any], advisor: Dict[str, Any]) -> Tuple[str, float]:
    bias_score = 0.0
    bias_score += float(snap_5m.get("dir_score") or 0.0) * 1.0
    bias_score += float(snap_15m.get("dir_score") or 0.0) * 2.0
    bias_score += float(snap_60m.get("dir_score") or 0.0) * 3.0
    pcr_bias = str(advisor.get("pcr_bias") or "").upper()
    oi_bias = str(advisor.get("oi_build_bias") or "").upper()
    if pcr_bias == "BULLISH":
        bias_score += 0.8
    elif pcr_bias == "BEARISH":
        bias_score -= 0.8
    if oi_bias == "BULLISH":
        bias_score += 0.4
    elif oi_bias == "BEARISH":
        bias_score -= 0.4
    primary_pattern = str(snap_5m.get("primary_pattern") or snap_15m.get("primary_pattern") or "NONE").upper()
    if primary_pattern in {"DOUBLE_BOTTOM_W_CONFIRMED", "FAKE_BREAKDOWN_DOWN"}:
        bias_score += 0.6
    elif primary_pattern in {"DOUBLE_TOP_M_CONFIRMED", "FAKE_BREAKOUT_UP"}:
        bias_score -= 0.6
    if bias_score >= 1.0:
        return "BULLISH", round(min(1.0, abs(bias_score) / 4.5), 3)
    if bias_score <= -1.0:
        return "BEARISH", round(min(1.0, abs(bias_score) / 4.5), 3)
    return "NEUTRAL", round(min(1.0, abs(bias_score) / 4.5), 3)


def _build_structure_regime(
    *,
    spot: float,
    snap_5m: Dict[str, Any],
    snap_15m: Dict[str, Any],
    snap_60m: Dict[str, Any],
    orb: Dict[str, Any],
    market_bias: str,
) -> str:
    breakout = str(orb.get("breakout_confirmation") or "NONE").upper()
    if market_bias == "BEARISH" and breakout == "DOWN_CONFIRMED":
        return "SWING_DOWN"
    if market_bias == "BULLISH" and breakout == "UP_CONFIRMED":
        return "SWING_UP"
    close_pos = snap_15m.get("close_position_in_range")
    range_points = float(snap_15m.get("range_points") or 0.0)
    atr = float(snap_15m.get("atr_like_points") or 0.0)
    if range_points > 0 and atr > 0 and range_points <= (atr * 4.0) and close_pos is not None and 0.35 <= float(close_pos) <= 0.65:
        return "RANGE"
    if market_bias == "BEARISH":
        return "BEARISH_DRIFT"
    if market_bias == "BULLISH":
        return "BULLISH_DRIFT"
    return "TRANSITION"


def _build_trade_plan(
    *,
    spot: float,
    snap_5m: Dict[str, Any],
    snap_15m: Dict[str, Any],
    snap_60m: Dict[str, Any],
    market_bias: str,
    bias_confidence: float,
    orb: Dict[str, Any],
    nearest_support: Optional[Dict[str, Any]],
    nearest_resistance: Optional[Dict[str, Any]],
    advisor: Dict[str, Any],
    buyer_seller_zones: Dict[str, Any],
    rsi_divergence: Dict[str, Any],
    higher_tf_confluence: Dict[str, Any],
) -> Dict[str, Any]:
    return _shared_build_trade_plan(
        spot=spot,
        snap_5m=snap_5m,
        snap_15m=snap_15m,
        snap_60m=snap_60m,
        market_bias=market_bias,
        bias_confidence=bias_confidence,
        orb=orb,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        advisor=advisor,
        buyer_seller_zones=buyer_seller_zones,
        rsi_divergence=rsi_divergence,
        higher_tf_confluence=higher_tf_confluence,
    )


def _build_intraday_index_chart_payload_from_candles(
    *,
    candles_5m: List[dict],
    candles_15m: List[dict],
    candles_60m: List[dict],
    advisor_status: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    advisor = advisor_status if isinstance(advisor_status, dict) else _load_intraday_ai_status()
    candles_5m = _normalize_intraday_candles(candles_5m, limit=96)
    candles_15m = _normalize_intraday_candles(candles_15m, limit=64)
    candles_60m = _normalize_intraday_candles(candles_60m, limit=48)
    now = _ist_now()
    if not candles_5m:
        return {
            "status": "NO_DATA",
            "symbol": "NIFTY",
            "interval_min": 5,
            "current_session_date": now.date().isoformat(),
            "updated_at": now.isoformat(timespec="seconds"),
            "headline": "No NIFTY intraday chart data available.",
            "candles": [],
            "ema20": [],
            "levels": {},
            "structure": {"regime": "UNKNOWN", "market_bias": "NEUTRAL"},
            "plan": {
                "trade_idea": "No trade plan",
                "posture": "NO_DATA",
                "trigger_window": "Waiting for intraday data.",
                "confirmation": "No confirmation possible without candles.",
                "invalidation": "No data.",
                "strike_hint": "No strike hint.",
                "reason_summary": ["Live chart data is unavailable."],
            },
        }

    closes = [float(candle["close"]) for candle in candles_5m]
    ema20 = _ema_series(closes, 20)
    snap_5m = _price_action_snapshot(candles_5m, interval_min=5)
    snap_15m = _price_action_snapshot(candles_15m or candles_5m, interval_min=15)
    snap_60m = _price_action_snapshot(candles_60m or candles_15m or candles_5m, interval_min=60)
    spot = float(candles_5m[-1]["close"])
    orb = _compute_orb_state(candles_5m)

    frame_5m = _candles_to_zone_frame(candles_5m, timeframe="5m")
    frame_15m = _candles_to_zone_frame(candles_15m or candles_5m, timeframe="15m")
    frame_60m = _candles_to_zone_frame(candles_60m or candles_15m or candles_5m, timeframe="1h")
    zones = find_multi_tf_zones({"5m": frame_5m, "15m": frame_15m, "1h": frame_60m})
    nearest_support_zone, nearest_resistance_zone = nearest_zones(zones, spot)
    supports = sorted([zone for zone in zones if zone.side == "support"], key=lambda zone: abs(zone.price - spot))[:4]
    resistances = sorted([zone for zone in zones if zone.side == "resistance"], key=lambda zone: abs(zone.price - spot))[:4]
    nearest_support = _zone_to_payload(nearest_support_zone)
    nearest_resistance = _zone_to_payload(nearest_resistance_zone)
    rsi_5m_series = _rsi_series(closes, 14)
    rsi_divergence = _detect_rsi_divergence(candles_5m, rsi_5m_series)
    buyer_seller_zones = _detect_orderflow_zones(
        candles_5m=candles_5m,
        spot=spot,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        snap_5m=snap_5m,
    )

    market_bias, bias_confidence = _market_bias_from_snapshot(snap_5m, snap_15m, snap_60m, advisor)
    higher_tf_confluence = _build_higher_tf_confluence(
        snap_15m=snap_15m,
        snap_60m=snap_60m,
        market_bias=market_bias,
        rsi_divergence=rsi_divergence,
    )
    regime = _build_structure_regime(
        spot=spot,
        snap_5m=snap_5m,
        snap_15m=snap_15m,
        snap_60m=snap_60m,
        orb=orb,
        market_bias=market_bias,
    )
    plan = _build_trade_plan(
        spot=spot,
        snap_5m=snap_5m,
        snap_15m=snap_15m,
        snap_60m=snap_60m,
        market_bias=market_bias,
        bias_confidence=bias_confidence,
        orb=orb,
        nearest_support=nearest_support,
        nearest_resistance=nearest_resistance,
        advisor=advisor,
        buyer_seller_zones=buyer_seller_zones,
        rsi_divergence=rsi_divergence,
        higher_tf_confluence=higher_tf_confluence,
    )

    weekly_structure = (advisor.get("market_context") or {}).get("structure") if isinstance(advisor.get("market_context"), dict) else {}
    weekly_structure = weekly_structure if isinstance(weekly_structure, dict) else {}
    labels = [candle["timestamp"].strftime("%H:%M") for candle in candles_5m]
    candle_payload = [
        {
            "timestamp": candle["timestamp"].isoformat(),
            "open": round(float(candle["open"]), 2),
            "high": round(float(candle["high"]), 2),
            "low": round(float(candle["low"]), 2),
            "close": round(float(candle["close"]), 2),
            "volume": round(float(candle.get("volume") or 0.0), 2),
        }
        for candle in candles_5m
    ]
    latest_primary_pattern = str(snap_5m.get("primary_pattern") or snap_15m.get("primary_pattern") or "NONE")
    headline = f"{market_bias.title()} {regime.replace('_', ' ').title()} | {plan.get('posture', 'NO_TRADE').replace('_', ' ').title()}"
    return {
        "status": "OK",
        "symbol": "NIFTY",
        "interval_min": 5,
        "current_session_date": candles_5m[-1]["timestamp"].date().isoformat(),
        "updated_at": now.isoformat(timespec="seconds"),
        "headline": headline,
        "labels": labels,
        "candles": candle_payload,
        "ema20": ema20,
        "rsi14": rsi_5m_series,
        "orb": orb,
        "levels": {
            "nearest_support": nearest_support,
            "nearest_resistance": nearest_resistance,
            "supports": [_zone_to_payload(zone) for zone in supports],
            "resistances": [_zone_to_payload(zone) for zone in resistances],
            "weekly_support": _parse_float(weekly_structure.get("weekly_support"), None),
            "weekly_resistance": _parse_float(weekly_structure.get("weekly_resistance"), None),
            "buyers": buyer_seller_zones.get("buyers") or {},
            "sellers": buyer_seller_zones.get("sellers") or {},
        },
        "structure": {
            "regime": regime,
            "market_bias": market_bias,
            "bias_confidence": round(float(bias_confidence), 3),
            "spot": round(float(spot), 2),
            "session_high": round(max(float(candle["high"]) for candle in candles_5m), 2),
            "session_low": round(min(float(candle["low"]) for candle in candles_5m), 2),
            "range_points": round(max(float(candle["high"]) for candle in candles_5m) - min(float(candle["low"]) for candle in candles_5m), 2),
            "ema20": _parse_float(ema20[-1], None),
            "ema_alignment_5m": snap_5m.get("ema_alignment"),
            "ema_alignment_15m": snap_15m.get("ema_alignment"),
            "price_action_confirmation": snap_5m.get("price_action_confirmation") or "NONE",
            "primary_pattern": latest_primary_pattern,
            "price_action_patterns": list(snap_5m.get("price_action_patterns") or []),
            "retest_status": snap_5m.get("retest_status") or "NONE",
            "breakout_confirmation": orb.get("breakout_confirmation") or "NONE",
            "rsi_divergence": rsi_divergence,
            "higher_tf_confluence": higher_tf_confluence,
            "timeframes": {"5": snap_5m, "15": snap_15m, "60": snap_60m},
        },
        "plan": plan,
        "advisor_overlay": {
            "signal": advisor.get("signal"),
            "strategy": advisor.get("strategy"),
            "market_bias": advisor.get("market_bias"),
            "trend_confidence": advisor.get("trend_confidence"),
            "signal_conflict_score": advisor.get("signal_conflict_score"),
            "pcr_bias": advisor.get("pcr_bias"),
            "oi_build_bias": advisor.get("oi_build_bias"),
            "breakout_confirmation": advisor.get("breakout_confirmation"),
            "orb_confirmation": advisor.get("orb_confirmation"),
            "updated_at": advisor.get("updated_at"),
        },
    }


def _load_intraday_index_chart_payload(interval_min: int = 5) -> Dict[str, Any]:
    now = _ist_now()
    cache_key = (now.date().isoformat(), interval_min)
    cached = _INDEX_CHART_CACHE.get(cache_key)
    if cached and (time.time() - cached.get("ts", 0.0)) < 45:
        return cached["payload"]
    advisor = _load_intraday_ai_status()
    try:
        dw = _init_dhan_wrapper()
        candles_5m = dw.get_intraday_candles(INDEX_SECURITY_ID, INDEX_EXCHANGE_SEG, interval=5, from_date=now.date().isoformat(), to_date=now.date().isoformat())
        candles_15m = dw.get_intraday_candles(INDEX_SECURITY_ID, INDEX_EXCHANGE_SEG, interval=15, from_date=now.date().isoformat(), to_date=now.date().isoformat())
        candles_60m = dw.get_intraday_candles(INDEX_SECURITY_ID, INDEX_EXCHANGE_SEG, interval=60, from_date=now.date().isoformat(), to_date=now.date().isoformat())
    except Exception as exc:
        return {
            "status": "ERROR",
            "symbol": "NIFTY",
            "interval_min": interval_min,
            "current_session_date": now.date().isoformat(),
            "updated_at": now.isoformat(timespec="seconds"),
            "headline": "Unable to build NIFTY chart context.",
            "error": str(exc),
            "candles": [],
            "ema20": [],
            "levels": {},
            "structure": {"regime": "UNKNOWN", "market_bias": advisor.get("market_bias") or "NEUTRAL"},
            "plan": {
                "trade_idea": "No trade plan",
                "posture": "NO_DATA",
                "trigger_window": "Chart fetch failed.",
                "confirmation": "Resolve feed/credential issue first.",
                "invalidation": "No data.",
                "strike_hint": "No strike hint.",
                "reason_summary": [str(exc)],
            },
        }
    payload = _build_intraday_index_chart_payload_from_candles(
        candles_5m=candles_5m,
        candles_15m=candles_15m,
        candles_60m=candles_60m,
        advisor_status=advisor,
    )
    _INDEX_CHART_CACHE[cache_key] = {"ts": time.time(), "payload": payload}
    return payload


def _load_v83_paper_trades() -> List[Dict[str, Any]]:
    """Parse intraday_v83_paper_live_trades.jsonl into flat completed-trade rows.

    The file alternates PAPER_ENTRY / PAPER_EXIT events keyed by entry_timestamp.
    We pair them and emit one row per closed trade.
    """
    if not V83_PAPER_LIVE_TRADES_JSONL.exists():
        return []
    entries: Dict[str, Any] = {}
    completed: List[Dict[str, Any]] = []
    try:
        for line in V83_PAPER_LIVE_TRADES_JSONL.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except Exception:
                continue
            event = str(ev.get("event") or "")
            ets = str(ev.get("entry_timestamp") or ev.get("timestamp") or "")
            if event == "PAPER_ENTRY":
                entries[ets] = ev
            elif event == "PAPER_EXIT" and ets in entries:
                entry = entries.pop(ets)
                pnl = float(ev.get("realized_paper_pnl") or 0.0)
                entry_ts = entry.get("entry_timestamp") or entry.get("timestamp") or ""
                exit_ts  = ev.get("exit_timestamp") or ev.get("timestamp") or ""
                hold_min: Optional[float] = None
                try:
                    from datetime import datetime as _dt
                    t0 = _dt.fromisoformat(entry_ts)
                    t1 = _dt.fromisoformat(exit_ts)
                    hold_min = (t1 - t0).total_seconds() / 60.0
                except Exception:
                    pass
                legs = entry.get("legs") or []
                short_leg = next((l for l in legs if l.get("action") == "SELL"), {})
                completed.append({
                    "trade_mode": "paper",
                    "pnl_rs": pnl,
                    "reason": str(ev.get("exit_reason") or "UNKNOWN").upper(),
                    "hold_minutes": hold_min,
                    "strategy_type": str(entry.get("strategy") or "").upper(),
                    "market_bias": str((entry.get("gate") or {}).get("market_state") or "").upper(),
                    "price_action_bias": None,
                    "retest_status": None,
                    "entry_time": entry_ts,
                    "exit_time": exit_ts,
                    "short_strike": short_leg.get("strike"),
                    "short_delta": short_leg.get("delta"),
                    "exit_mode": str(ev.get("exit_mode") or ""),
                    "mae_rupees": float(ev.get("mae_rupees") or 0.0),
                    "mfe_rupees": float(ev.get("mfe_rupees") or 0.0),
                })
    except Exception:
        pass
    return completed


def _load_intraday_trade_history(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not INTRADAY_AI_TRADE_HISTORY_JSONL.exists():
        return rows
    try:
        with INTRADAY_AI_TRADE_HISTORY_JSONL.open("r", encoding="utf-8") as handle:
            for line in handle:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except Exception:
                    continue
                if isinstance(payload, dict):
                    rows.append(payload)
    except Exception:
        return []
    rows.sort(key=lambda row: str(row.get("closed_at") or row.get("opened_at") or ""))
    if limit is not None and limit > 0:
        rows = rows[-limit:]
    return rows


import threading  # noqa: E402  (local: used only by the live-price refresher below)

_open_leg_ltp_cache: Dict[float, Dict[str, Any]] = {}
_open_leg_ltp_cache_ts: float = 0.0
_open_leg_ltp_cache_exp: Optional[str] = None  # which expiry the cache is for (invalidate on change)
_OPEN_LEG_LTP_TTL = 3.0   # Dhan chain is ~1 req/3s; 3s is the floor for the fast poller

# Background live-price refresher. The Dhan chain fetch can BLOCK for seconds (the 1-req/3s
# throttle, or up to 90s on a 429 cooldown), so doing it inline made /api/open_position_live stall
# ~2.5s every time the 3s cache expired — the UI lag the user reported. Instead a single daemon
# thread refreshes the cache continuously and the endpoint just reads it (always instant).
_ltp_refresher_thread: Optional[threading.Thread] = None
_ltp_refresher_lock = threading.Lock()
_ltp_wanted_exp: Optional[str] = None   # requested by the endpoint; None => resolve front tradeable
_ltp_wanted_at: float = 0.0             # last time the endpoint asked (thread idles when stale)
_open_leg_spot: Optional[float] = None  # underlying spot from the same chain fetch (cheap for the UI)


def _resolve_front_tradeable_expiry(dw) -> Optional[str]:
    """First expiry AFTER today — the front weekly the agent trades (skips a 0-DTE series on roll day)."""
    from datetime import date as _date
    try:
        exps = dw.get_optionchain_expirylist("IDX_I", INDEX_SECURITY_ID) or []
    except Exception:
        return None
    today = _date.today()
    for e in exps:
        try:
            if _date.fromisoformat(str(e)) > today:
                return str(e)
        except (TypeError, ValueError):
            continue
    return str(exps[0]) if exps else None


def _ltp_refresher_loop() -> None:
    """Daemon: keep _open_leg_ltp_cache fresh for the currently-wanted expiry. ALL the slow/blocking
    Dhan work lives here, off the request path, so the endpoint never waits."""
    global _open_leg_ltp_cache, _open_leg_ltp_cache_ts, _open_leg_ltp_cache_exp, _open_leg_spot
    dw = None
    while True:
        try:
            if (time.time() - _ltp_wanted_at) > 60:  # nobody viewed the position in 60s → idle
                time.sleep(2)
                continue
            if dw is None:
                from data_engine.market_ai.dhan_wrapper import DhanWrapper  # type: ignore
                creds = _json_read(CREDS_FILE)
                if isinstance(creds, dict):
                    cid = (creds.get("client_id") or "").strip()
                    tok = (creds.get("access_token") or "").strip()
                    if cid and tok:
                        os.environ["DHAN_CLIENT_ID"] = cid
                        os.environ["DHAN_ACCESS_TOKEN"] = tok
                dw = DhanWrapper(logger=None)
            exp = _ltp_wanted_exp or _resolve_front_tradeable_expiry(dw)
            if exp:
                raw = dw.get_option_chain(INDEX_SECURITY_ID, "IDX_I", exp)
                _dd = ((((raw or {}).get("data") or {}).get("data")) or {})
                oc = _dd.get("oc") or {}
                out: Dict[float, Dict[str, Any]] = {}
                for k, node in oc.items():
                    try:
                        sk = round(float(k), 2)
                    except (TypeError, ValueError):
                        continue
                    ce = (node.get("ce") or {}).get("last_price")
                    pe = (node.get("pe") or {}).get("last_price")
                    out[sk] = {"CALL": ce, "CE": ce, "PUT": pe, "PE": pe}
                if out:  # rebind (atomic); readers never see a half-built dict
                    _open_leg_ltp_cache = out
                    _open_leg_ltp_cache_ts = time.time()
                    _open_leg_ltp_cache_exp = exp
                    try:
                        _sp = _dd.get("last_price")
                        if _sp:
                            _open_leg_spot = float(_sp)
                    except (TypeError, ValueError):
                        pass
        except Exception:
            pass
        time.sleep(_OPEN_LEG_LTP_TTL)


def _live_leg_ltps(position_expiry: Optional[str] = None) -> Dict[float, Dict[str, Any]]:
    """Current LTP per strike from the live Nifty chain: {strike: {"CALL":x, "PUT":y}}.

    The agent stores only ENTRY-time leg quotes in paper_state (frozen), so the UI showed
    static LTP and 0 per-leg P&L. Fetching live prices here makes both update each poll.
    Cached (per expiry) so UI polling doesn't hammer Dhan's 1-req/3s chain limit.

    `position_expiry` (YYYY-MM-DD) marks the open position against the EXACT series it was booked
    on — so entry and LTP always agree. When absent (older positions with no stamped expiry) it
    falls back to the front TRADEABLE weekly (skipping a series expiring today on roll days).
    """
    from datetime import date as _date
    global _ltp_refresher_thread, _ltp_wanted_exp, _ltp_wanted_at

    exp: Optional[str] = None
    if position_expiry:
        try:
            _date.fromisoformat(str(position_expiry))
            exp = str(position_expiry)
        except (TypeError, ValueError):
            exp = None

    # Tell the background refresher which expiry to keep warm, and note that someone's watching.
    _ltp_wanted_exp = exp
    _ltp_wanted_at = time.time()

    # Lazily start the single daemon refresher (double-checked under a lock).
    if _ltp_refresher_thread is None or not _ltp_refresher_thread.is_alive():
        with _ltp_refresher_lock:
            if _ltp_refresher_thread is None or not _ltp_refresher_thread.is_alive():
                _ltp_refresher_thread = threading.Thread(
                    target=_ltp_refresher_loop, daemon=True, name="ltp-refresher"
                )
                _ltp_refresher_thread.start()

    # Return the refresher's latest cache INSTANTLY — never block the request on Dhan. When a
    # position stamped a specific expiry, only serve the cache if it's for that same series; for
    # the heuristic (unstamped) path serve whatever the refresher last fetched. Empty only until
    # the first refresh lands (a few seconds on first load), during which the caller shows the
    # frozen entry quote.
    if _open_leg_ltp_cache and (exp is None or _open_leg_ltp_cache_exp == exp):
        return _open_leg_ltp_cache
    return {}


def _build_v83_open_position_payload() -> Dict[str, Any]:
    """Return the current v83 paper open position in a UI-ready format.

    Reads from ``intraday_v83_paper_state.json`` which the agent keeps updated.
    Returns an empty dict when no position is open.
    """
    try:
        state = _json_read(V83_PAPER_STATE_JSON) or {}
    except Exception:
        state = {}
    # The agent nests the live position under "active_position" (older schema kept the
    # structure at top level). Read from active_position when present so the UI actually
    # shows the open trade — without this, a real open paper position renders as "flat".
    pos = state.get("active_position") if isinstance(state.get("active_position"), dict) else state
    if not pos or not pos.get("structure"):
        return {"open": False, "legs": [], "open_mtm": None, "expiry": None, "playbook": None}

    structure = pos.get("structure") or {}
    legs_raw = structure.get("legs") or []
    # MFE/MAE/mark P&L are top-level in the agent's state; metadata lives under the position.
    mfe = _parse_float(state.get("mfe_rupees"), 0.0)
    mae = _parse_float(state.get("mae_rupees"), 0.0)
    mark_pnl = _parse_float(state.get("last_mark_pnl_rupees"), None)
    playbook = (pos.get("metadata") or structure.get("metadata") or {}).get("playbook")
    expiry = None

    # Leg qty lives at the position level (lots x lot_size), not on each leg.
    _lots = _parse_int(pos.get("lots"), 0)
    _lot_size = _parse_int(pos.get("lot_size"), 0)
    _pos_qty = _lots * _lot_size if (_lots and _lot_size) else 0

    ui_legs: List[Dict[str, Any]] = []
    for leg in legs_raw:
        strike = leg.get("strike")
        option_type = leg.get("option_type") or leg.get("instrument_type") or ""
        action = str(leg.get("action") or "").upper()
        qty = abs(int(leg.get("quantity") or leg.get("qty") or _pos_qty or 0))
        entry_price = _parse_float(leg.get("entry_price") or leg.get("ltp"), None)
        ltp = _parse_float(leg.get("ltp"), entry_price)
        exp = leg.get("expiry")
        if exp and expiry is None:
            expiry = exp
        pnl = None
        if entry_price is not None and ltp is not None and qty > 0:
            pnl = (entry_price - ltp) * qty if action == "SELL" else (ltp - entry_price) * qty
        ui_legs.append({
            "side": action,
            "option_type": option_type,
            "strike": strike,
            "expiry": exp,
            "qty": qty,
            "entry": entry_price,
            "ltp": ltp,
            "pnl": round(pnl, 2) if pnl is not None else None,
        })

    # Overlay LIVE leg prices (agent stores only frozen entry quotes). Each leg's stored
    # "entry" is the entry price; fetch the current LTP by strike and recompute per-leg P&L
    # so both the LTP column and the legs total update every poll instead of showing 0.
    # Pass the position's own expiry so live LTP is marked against the SAME series the legs
    # were booked on (exact); falls back to the front tradeable weekly when unstamped.
    live = _live_leg_ltps(expiry)
    live_total = 0.0
    have_live = False
    for leg in ui_legs:
        entry = leg.get("entry")
        strike = leg.get("strike")
        otype = str(leg.get("option_type") or "").upper()
        qty = leg.get("qty") or 0
        live_ltp = None
        if strike is not None:
            live_ltp = (live.get(round(float(strike), 2)) or {}).get(otype)
        if live_ltp is not None:
            have_live = True
            leg["ltp"] = live_ltp
            if entry is not None and qty:
                pnl = (entry - live_ltp) * qty if leg.get("side") == "SELL" else (live_ltp - entry) * qty
                leg["pnl"] = round(pnl, 2)
                live_total += pnl

    # Prefer the live-computed total (fresh each poll); fall back to the agent's mark.
    open_mtm = round(live_total, 2) if have_live else mark_pnl

    return {
        "open": True,
        "legs": ui_legs,
        "open_mtm": open_mtm,
        "mark_pnl_rupees": mark_pnl,
        "mfe_rupees": mfe,
        "mae_rupees": mae,
        "expiry": expiry,
        "playbook": playbook,
    }


def _get_v83_market_state() -> str:
    """Return the latest v83 market state label from runtime state."""
    try:
        rs = _json_read(V83_RUNTIME_STATE_JSON) or {}
        candidate = rs.get("last_candidate") or {}
        mkt = candidate.get("market_state")
        if mkt:
            return str(mkt)
        # Also check last simulated order
        sim = rs.get("last_simulated_order") or {}
        gate = sim.get("gate") or {}
        mkt2 = gate.get("market_state")
        if mkt2:
            return str(mkt2)
    except Exception:
        pass
    return "—"


def _build_intraday_performance_payload(*, trade_mode: str) -> Dict[str, Any]:
    mode = str(trade_mode or "").strip().lower()
    # For paper mode: use v83 trades as primary source (old intraday_ai_trade_history is Batman BKM)
    if mode == "paper":
        rows = _load_v83_paper_trades()
    else:
        rows = [row for row in _load_intraday_trade_history() if str(row.get("trade_mode") or "").strip().lower() == mode]
    total_trades = len(rows)
    wins = 0
    losses = 0
    flats = 0
    realized_pnl = 0.0
    hold_values: List[float] = []
    exit_reason_counts: Dict[str, int] = {}
    setup_stats: Dict[str, Dict[str, Dict[str, float]]] = {
        "strategy_type": {},
        "market_bias": {},
        "price_action_bias": {},
        "retest_status": {},
    }
    recent_trades: List[Dict[str, Any]] = []

    def _add_setup_stat(group: str, key: Any, pnl: float) -> None:
        label = str(key or "UNKNOWN").strip().upper() or "UNKNOWN"
        bucket = setup_stats[group].setdefault(label, {"count": 0.0, "pnl_rs": 0.0})
        bucket["count"] += 1.0
        bucket["pnl_rs"] += float(pnl)

    for row in rows:
        pnl = _parse_float(row.get("pnl_rs"), 0.0)
        realized_pnl += pnl
        if pnl > 0:
            wins += 1
        elif pnl < 0:
            losses += 1
        else:
            flats += 1
        hold_min = row.get("hold_minutes")
        if hold_min not in (None, "", "-", "--"):
            hold_values.append(_parse_float(hold_min, 0.0))
        reason = str(row.get("reason") or "UNKNOWN").strip().upper() or "UNKNOWN"
        exit_reason_counts[reason] = int(exit_reason_counts.get(reason, 0)) + 1
        entry_features = row.get("entry_features") if isinstance(row.get("entry_features"), dict) else {}
        _add_setup_stat("strategy_type", row.get("strategy_type"), pnl)
        _add_setup_stat("market_bias", row.get("market_bias") or entry_features.get("market_bias"), pnl)
        _add_setup_stat("price_action_bias", entry_features.get("price_action_bias"), pnl)
        _add_setup_stat("retest_status", entry_features.get("retest_status"), pnl)

    rows_desc = list(reversed(rows))
    for row in rows_desc[:10]:
        pnl_val = round(_parse_float(row.get("pnl_rs"), 0.0), 2)
        recent_trades.append(
            {
                "position_id": row.get("position_id"),
                "strategy_label": row.get("strategy_type") or row.get("strategy_label"),
                "closed_at": row.get("exit_time") or row.get("closed_at"),
                "entry_time": row.get("entry_time"),
                "pnl_rs": pnl_val,
                "hold_minutes": row.get("hold_minutes"),
                "reason": row.get("reason"),
                "exit_mode": row.get("exit_mode"),
                "result": "WIN" if pnl_val > 0 else ("LOSS" if pnl_val < 0 else "FLAT"),
                "short_strike": row.get("short_strike"),
                "short_delta": row.get("short_delta"),
                "mae_rupees": row.get("mae_rupees"),
                "mfe_rupees": row.get("mfe_rupees"),
            }
        )

    exit_reasons = [
        {"reason": reason, "count": count}
        for reason, count in sorted(exit_reason_counts.items(), key=lambda item: (-item[1], item[0]))
    ]
    setup_breakdown = {
        group: [
            {"label": label, "count": int(round(values["count"])), "pnl_rs": round(float(values["pnl_rs"]), 2)}
            for label, values in sorted(stats.items(), key=lambda item: (-item[1]["pnl_rs"], -item[1]["count"], item[0]))
        ]
        for group, stats in setup_stats.items()
    }
    avg_hold_minutes = round(sum(hold_values) / len(hold_values), 2) if hold_values else None
    win_rate_pct = round((wins / total_trades) * 100.0, 1) if total_trades > 0 else None
    return {
        "trade_mode": mode,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "flats": flats,
        "win_rate_pct": win_rate_pct,
        "avg_hold_minutes": avg_hold_minutes,
        "realized_pnl_rs": round(realized_pnl, 2),
        "exit_reasons": exit_reasons,
        "setup_breakdown": setup_breakdown,
        "recent_trades": recent_trades,
    }


def _build_committee_review_payload(*, trade_mode: str) -> Dict[str, Any]:
    mode = str(trade_mode or "").strip().lower()
    rows = [
        row
        for row in _load_jsonl_tail(DECISION_COMMITTEE_OUTCOMES_JSONL, limit=10000)
        if str(row.get("trade_mode") or "").strip().lower() == mode
    ]
    total = len(rows)
    wins = sum(1 for row in rows if str(row.get("result") or "").upper() == "WIN")
    losses = sum(1 for row in rows if str(row.get("result") or "").upper() == "LOSS")
    flats = sum(1 for row in rows if str(row.get("result") or "").upper() == "FLAT")
    pnl = round(sum(_parse_float(row.get("pnl_rs"), 0.0) for row in rows), 2)
    hold_values = [
        _parse_float(row.get("hold_minutes"), 0.0)
        for row in rows
        if row.get("hold_minutes") not in (None, "", "-", "--")
    ]
    avg_hold = round(sum(hold_values) / len(hold_values), 2) if hold_values else None

    def _bucket(rows_in: List[Dict[str, Any]], key_name: str) -> List[Dict[str, Any]]:
        buckets: Dict[str, Dict[str, Any]] = {}
        for row in rows_in:
            label = str(row.get(key_name) or "UNKNOWN").strip().upper() or "UNKNOWN"
            bucket = buckets.setdefault(label, {"label": label, "count": 0, "pnl_rs": 0.0})
            bucket["count"] += 1
            bucket["pnl_rs"] = round(float(bucket["pnl_rs"]) + _parse_float(row.get("pnl_rs"), 0.0), 2)
        return sorted(buckets.values(), key=lambda item: (-int(item["count"]), str(item["label"])))

    by_verdict = _bucket(rows, "committee_verdict_at_entry")
    by_bias = _bucket(rows, "committee_bias_at_entry")
    by_exit = _bucket(rows, "exit_reason")

    recent = []
    for row in sorted(rows, key=lambda item: str(item.get("closed_at") or item.get("timestamp") or ""), reverse=True)[:10]:
        recent.append(
            {
                "closed_at": row.get("closed_at"),
                "strategy_label": row.get("strategy_label"),
                "committee_verdict_at_entry": row.get("committee_verdict_at_entry"),
                "committee_bias_at_entry": row.get("committee_bias_at_entry"),
                "committee_confidence_at_entry": _parse_float(row.get("committee_confidence_at_entry"), None),
                "exit_reason": row.get("exit_reason"),
                "result": row.get("result"),
                "pnl_rs": _parse_float(row.get("pnl_rs"), 0.0),
                "hold_minutes": _parse_float(row.get("hold_minutes"), None),
            }
        )

    best_verdict = by_verdict[0]["label"] if by_verdict else "—"
    best_bias = "—"
    if by_bias:
        best_bias = sorted(by_bias, key=lambda item: (-float(item["pnl_rs"]), -int(item["count"]), str(item["label"])))[0]["label"]

    return {
        "trade_mode": mode,
        "total_outcomes": total,
        "wins": wins,
        "losses": losses,
        "flats": flats,
        "win_rate_pct": round((wins / total) * 100.0, 2) if total else None,
        "realized_pnl_rs": pnl,
        "avg_hold_minutes": avg_hold,
        "best_verdict_label": best_verdict,
        "best_bias_label": best_bias,
        "by_verdict": by_verdict,
        "by_bias": by_bias,
        "by_exit_reason": by_exit,
        "recent_outcomes": recent,
    }


def _parse_iso_dt(value: Any) -> Optional[datetime]:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def _load_batman_bkm_event_history(limit: int = 500) -> List[Dict[str, Any]]:
    rows = _load_jsonl_tail(BATMAN_BKM_AI_EVENTS_JSONL, limit=limit)
    return [row for row in rows if isinstance(row, dict)]


def _build_batman_bkm_review_payload(*, trade_mode: str) -> Dict[str, Any]:
    state = _json_read(STRATEGY_STATE)
    bucket = (state.get("BATMAN_BKM") if isinstance(state, dict) else {}) or {}
    if not isinstance(bucket, dict):
        bucket = {}

    event_rows = _load_batman_bkm_event_history(limit=1000)
    event_rows.sort(key=lambda row: str(row.get("timestamp") or row.get("updated_at") or ""))

    open_baskets: List[Dict[str, Any]] = []
    closed_baskets: List[Dict[str, Any]] = []

    market_open_hour = 9
    market_open_minute = 15

    for expiry, payload in bucket.items():
        if not isinstance(payload, dict):
            continue
        status = str(payload.get("status") or "").upper() or "UNKNOWN"
        opened_at = str(payload.get("opened_at") or "") or None
        closed_at = str(payload.get("closed_at") or "") or None
        opened_dt = _parse_iso_dt(opened_at)
        closed_dt = _parse_iso_dt(closed_at)
        held_minutes = None
        if opened_dt and closed_dt:
            held_minutes = round(max((closed_dt - opened_dt).total_seconds(), 0.0) / 60.0, 2)

        metrics = payload.get("quality_metrics") if isinstance(payload.get("quality_metrics"), dict) else {}
        latest_event = None
        for row in reversed(event_rows):
            if str(row.get("basket_expiry") or "") != str(expiry):
                continue
            latest_event = row
            break

        review = {
            "expiry": str(expiry),
            "status": status,
            "opened_at": opened_at,
            "closed_at": closed_at,
            "held_minutes": held_minutes,
            "reason": str(payload.get("reason") or ""),
            "pnl_rs": round(_parse_float(payload.get("pnl"), 0.0), 2),
            "net_credit_rs": round(_parse_float(payload.get("net_credit"), 0.0), 2),
            "credit_pct": round(_parse_float(payload.get("credit_pct"), 0.0), 4),
            "quality_status": str(payload.get("quality_status") or "UNKNOWN"),
            "quality_score": round(_parse_float(payload.get("quality_score"), 0.0), 2),
            "opened_pre_market": bool(
                opened_dt
                and ((opened_dt.hour, opened_dt.minute) < (market_open_hour, market_open_minute))
            ),
            "daily_lock_triggered": str(payload.get("reason") or "").upper() == "DAILY_LOCK_RED",
            "center_offset_pts": round(_parse_float(metrics.get("center_offset"), 0.0), 2),
            "short_distance_min_pts": round(_parse_float(metrics.get("short_distance_min"), 0.0), 2),
            "short_width_pts": round(_parse_float(metrics.get("short_width"), 0.0), 2),
            "outer_distance_ratio": round(_parse_float(metrics.get("outer_distance_ratio"), 0.0), 2),
            "loss_to_credit_ratio": None,
            "lessons": [],
            "improvements": [],
            "close_context": {
                "market_bias": str((latest_event or {}).get("market_bias") or ""),
                "position_risk_side": str((latest_event or {}).get("position_risk_side") or ""),
                "ai_reasons": list((latest_event or {}).get("reasons") or []),
                "ai_confidence": _parse_float((latest_event or {}).get("confidence"), None),
                "ai_score": _parse_float((latest_event or {}).get("score"), None),
            },
        }

        pnl_rs = float(review["pnl_rs"])
        net_credit = float(review["net_credit_rs"])
        if net_credit > 0 and pnl_rs < 0:
            review["loss_to_credit_ratio"] = round(abs(pnl_rs) / net_credit, 2)

        lessons: List[str] = []
        improvements: List[str] = []

        if review["quality_status"] == "PASS" and review["quality_score"] >= 90:
            lessons.append("STRUCTURE_SHAPE_WAS_GOOD")
        if review["opened_pre_market"]:
            lessons.append("ENTRY_HAPPENED_PRE_MARKET")
            improvements.append("BLOCK_BATMAN_ENTRY_BEFORE_09_15")
        if review["daily_lock_triggered"]:
            lessons.append("BASKET_HIT_DAILY_LOSS_LOCK")
            improvements.append("REQUIRE_MARKET_OPEN_CONFIRMATION_BEFORE_FIRST_DEPLOY")
        if held_minutes is not None and held_minutes <= 30:
            lessons.append("LOSS_CAME_EARLY_AFTER_ENTRY")
            improvements.append("ADD_OPENING_RANGE_STABILIZATION_5_TO_15_MIN")
        if review["close_context"]["market_bias"]:
            lessons.append(f"CLOSE_BIAS_{review['close_context']['market_bias']}")
        ai_reasons = review["close_context"]["ai_reasons"]
        if ai_reasons:
            if any(str(reason).upper() == "LOSS_BUILDING" for reason in ai_reasons):
                improvements.append("TIGHTEN_EARLY_LOSS_ACCELERATION_CHECK")
            if any("DRAWDOWN" in str(reason).upper() for reason in ai_reasons):
                improvements.append("USE_OI_PCR_RECHECK_BEFORE_HOLDING_OPENING_DRIFT")

        # De-duplicate while keeping order.
        dedup_lessons: List[str] = []
        for item in lessons:
            if item not in dedup_lessons:
                dedup_lessons.append(item)
        dedup_improvements: List[str] = []
        for item in improvements:
            if item not in dedup_improvements:
                dedup_improvements.append(item)
        review["lessons"] = dedup_lessons
        review["improvements"] = dedup_improvements

        if status == "OPEN":
            open_baskets.append(review)
        else:
            closed_baskets.append(review)

    open_baskets.sort(key=lambda row: str(row.get("opened_at") or row.get("expiry") or ""))
    closed_baskets.sort(key=lambda row: str(row.get("closed_at") or row.get("opened_at") or row.get("expiry") or ""))
    latest_closed = closed_baskets[-1] if closed_baskets else None

    return {
        "trade_mode": str(trade_mode or "").strip().lower(),
        "open_baskets": open_baskets,
        "closed_baskets": list(reversed(closed_baskets[-10:])),
        "latest_closed": latest_closed,
        "summary": {
            "open_count": len(open_baskets),
            "closed_count": len(closed_baskets),
            "realized_pnl_rs": round(sum(_parse_float(item.get("pnl_rs"), 0.0) for item in closed_baskets), 2),
        },
    }


def _build_batman_bkm_learning_payload(*, trade_mode: str) -> Dict[str, Any]:
    status = _load_bkm_learning_status()
    last_assessment = status.get("last_entry_assessment") if isinstance(status.get("last_entry_assessment"), dict) else {}
    return {
        "trade_mode": str(trade_mode or "").strip().lower(),
        "status": str(status.get("status") or "IDLE"),
        "summary": {
            "total_outcomes": _parse_int(status.get("total_outcomes"), 0),
            "wins": _parse_int(status.get("wins"), 0),
            "losses": _parse_int(status.get("losses"), 0),
            "flats": _parse_int(status.get("flats"), 0),
            "realized_pnl_rs": round(_parse_float(status.get("realized_pnl_rs"), 0.0), 2),
            "last_outcome_at": status.get("last_outcome_at"),
            "last_outcome_id": status.get("last_outcome_id"),
            "updated_at": status.get("updated_at"),
        },
        "last_entry_assessment": {
            "mode": last_assessment.get("mode"),
            "status": last_assessment.get("status"),
            "block_entry": bool(last_assessment.get("block_entry", False)),
            "risk_score_adjust": round(_parse_float(last_assessment.get("risk_score_adjust"), 0.0), 2),
            "reasons": list(last_assessment.get("reasons") or []),
            "supportive_reasons": list(last_assessment.get("supportive_reasons") or []),
            "timestamp": last_assessment.get("timestamp"),
        },
        "top_negative_buckets": list(status.get("top_negative_buckets") or []),
        "top_positive_buckets": list(status.get("top_positive_buckets") or []),
    }


def _load_telegram_command_status() -> Dict[str, Any]:
    payload = _json_read(TELEGRAM_COMMAND_STATUS_JSON)
    if isinstance(payload, dict) and payload:
        return payload
    now = _ist_now()
    return {
        "enabled": False,
        "configured": False,
        "last_poll_at": None,
        "last_update_id": 0,
        "last_command": None,
        "last_command_at": None,
        "last_result": None,
        "last_error": None,
        "next_poll_after": None,
        "updated_at": now.isoformat(timespec="seconds"),
    }


def _queue_intraday_ai_deploy_request(*, source: str, note: str) -> Dict[str, Any]:
    now = _ist_now()
    if INTRADAY_AI_DEPLOY_REQUEST_JSON.exists():
        pending = _json_read(INTRADAY_AI_DEPLOY_REQUEST_JSON)
        if isinstance(pending, dict) and pending:
            return {
                "ok": False,
                "queued": False,
                "error": "DEPLOY_REQUEST_PENDING",
                "pending_request": pending,
            }
    request_id = f"INTRADAYDEP-{int(time.time() * 1000)}"
    payload = {
        "request_id": request_id,
        "requested_at": now.isoformat(timespec="seconds"),
        "source": str(source or "manual_ui"),
        "note": str(note or ""),
    }
    INTRADAY_AI_DEPLOY_REQUEST_JSON.parent.mkdir(parents=True, exist_ok=True)
    _json_write(INTRADAY_AI_DEPLOY_REQUEST_JSON, payload)
    status = _load_intraday_ai_deploy_status()
    status.update(
        {
            "status": "REQUESTED",
            "pending": True,
            "request_id": request_id,
            "requested_at": payload["requested_at"],
            "request_source": payload["source"],
            "request_note": payload["note"],
            "processed_at": None,
            "result_code": None,
            "message": "Deployment request accepted and queued.",
            "details": {},
            "current_session_date": now.date().isoformat(),
            "updated_at": now.isoformat(timespec="seconds"),
        }
    )
    _json_write(INTRADAY_AI_DEPLOY_STATUS_JSON, status)
    return {
        "ok": True,
        "queued": True,
        "request_id": request_id,
        "request": payload,
        "deploy_status": status,
    }


def _clear_bkm_ai_protect_lock(*, source: str = "manual_ui", note: str = "") -> Dict[str, Any]:
    now = _ist_now()
    before = _load_bkm_ai_protect_status()
    cleared = _default_bkm_ai_protect_status()
    cleared["protection_enabled"] = bool(before.get("protection_enabled", True))
    cleared["session_date"] = now.date().isoformat()
    cleared["updated_at"] = now.isoformat(timespec="seconds")
    cleared["last_transition_at"] = now.isoformat(timespec="seconds")
    BATMAN_BKM_AI_PROTECT_STATUS_JSON.parent.mkdir(parents=True, exist_ok=True)
    _json_write(BATMAN_BKM_AI_PROTECT_STATUS_JSON, cleared)
    _json_write(
        BATMAN_BKM_AI_PROTECT_CLEAR_REQUEST_JSON,
        {
            "requested_at": now.isoformat(timespec="seconds"),
            "source": str(source or "manual_ui"),
            "note": str(note or ""),
        },
    )
    return {"ok": True, "before": before, "after": _load_bkm_ai_protect_status(), "signal_sent": True}


def _set_bkm_ai_protection_enabled(*, enabled: bool, source: str = "manual_ui", note: str = "") -> Dict[str, Any]:
    now = _ist_now()
    before = _load_bkm_ai_protect_status()
    after = dict(before)
    after["session_date"] = now.date().isoformat()
    after["protection_enabled"] = bool(enabled)
    if not enabled:
        after["active"] = False
        after["status"] = "UNLOCKED"
        after["reason"] = None
        after["source_action"] = None
        after["score"] = None
        after["expiry"] = None
    after["updated_at"] = now.isoformat(timespec="seconds")
    after["last_transition_at"] = now.isoformat(timespec="seconds")
    BATMAN_BKM_AI_PROTECT_STATUS_JSON.parent.mkdir(parents=True, exist_ok=True)
    _json_write(BATMAN_BKM_AI_PROTECT_STATUS_JSON, after)
    _json_write(
        BATMAN_BKM_AI_PROTECT_CLEAR_REQUEST_JSON,
        {
            "requested_at": now.isoformat(timespec="seconds"),
            "source": str(source or "manual_ui"),
            "note": str(note or ""),
            "operation": "set_protection_enabled",
            "protection_enabled": bool(enabled),
        },
    )
    return {"ok": True, "before": before, "after": _load_bkm_ai_protect_status(), "signal_sent": True}


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


def _parse_bkm_pos_identity(pos: Dict[str, Any]) -> Tuple[Optional[str], Optional[float], Optional[str]]:
    symbol = str(pos.get("strike") or "").strip()
    # Dhan symbols can arrive as NIFTY-Mar2026-24500-PE; normalize separators for parsing.
    symbol = (
        symbol.replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("_", "-")
    )
    expiry_raw = str(pos.get("expiry") or "").strip()

    expiry_iso: Optional[str] = None
    for fmt in (None, "%Y-%m-%d", "%d-%m-%Y", "%d-%b-%Y", "%d %b %Y", "%d %B %Y"):
        try:
            if fmt is None:
                expiry_iso = datetime.fromisoformat(expiry_raw).date().isoformat()
            else:
                expiry_iso = datetime.strptime(expiry_raw, fmt).date().isoformat()
            break
        except Exception:
            continue

    strike_val: Optional[float] = None
    opt_val: Optional[str] = None
    try:
        strike_val = float(symbol)
    except Exception:
        strike_val = None
    m = re.search(r"(?P<strike>\d+(?:\.\d+)?)[\s\-_/]*(?P<opt>CE|PE|CALL|PUT)\b", symbol.upper())
    if m:
        try:
            strike_val = float(m.group("strike"))
        except Exception:
            pass
        opt_raw = str(m.group("opt") or "").upper()
        if opt_raw == "CALL":
            opt_raw = "CE"
        if opt_raw == "PUT":
            opt_raw = "PE"
        opt_val = opt_raw

    if not opt_val:
        sym_u = symbol.upper()
        if " CALL" in sym_u:
            opt_val = "CE"
        elif " PUT" in sym_u:
            opt_val = "PE"
        elif sym_u.endswith(" CE"):
            opt_val = "CE"
        elif sym_u.endswith(" PE"):
            opt_val = "PE"

    return expiry_iso, strike_val, opt_val


def _assess_imported_bkm_quality(
    *,
    legs: List[Dict[str, Any]],
    expiry: str,
    net_credit: float,
    margin_required: float,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        from data_engine.market_ai.modules.strategies.batman_bkm_monthly import (  # type: ignore
            BatmanBKMConfig,
            BatmanBKMStrategy,
            Leg as BKMLeg,
        )
    except Exception as exc:
        return {
            "ok": False,
            "status": "UNKNOWN",
            "reason": "QUALITY_EVAL_IMPORT_ERROR",
            "reasons": ["QUALITY_EVAL_IMPORT_ERROR"],
            "score": 0.0,
            "metrics": {"error": str(exc)},
        }

    cfg = BatmanBKMConfig(
        base_distance_points=_parse_int((settings or {}).get("batman_bkm_base_distance"), 400),
        inner_step_points=_parse_int((settings or {}).get("batman_bkm_inner_step"), 200),
        outer_step_points=_parse_int((settings or {}).get("batman_bkm_outer_step"), 800),
        strike_rounding=_parse_int((settings or {}).get("batman_bkm_strike_rounding"), 50),
        lot_size=_parse_int((settings or {}).get("batman_v2_lot_size") or (settings or {}).get("lot_size"), 65),
        lot_multiplier=_parse_int((settings or {}).get("batman_bkm_lot_multiplier"), 1),
        max_credit_pct=_parse_float((settings or {}).get("batman_bkm_max_credit_pct"), 6.0),
        credit_step_points=_parse_int((settings or {}).get("batman_bkm_credit_step"), 100),
        max_widen_iterations=_parse_int((settings or {}).get("batman_bkm_max_widen_iterations"), 10),
        balance_tolerance=_parse_float((settings or {}).get("batman_bkm_balance_tolerance"), 5000.0),
        max_hedge_lots=_parse_int((settings or {}).get("batman_bkm_max_hedge_lots"), 6),
        tp_pct=_parse_float((settings or {}).get("batman_bkm_tp_pct"), 0.02),
        sl_pct=_parse_float((settings or {}).get("batman_bkm_sl_pct"), 0.025),
        enable_balance=bool((settings or {}).get("batman_bkm_enable_balance", True)),
        estimated_margin=_parse_float((settings or {}).get("batman_bkm_estimated_margin"), 1_000_000.0),
        min_credit_pct=_parse_float((settings or {}).get("batman_bkm_min_credit_pct"), 0.75),
        min_short_distance_points=_parse_float((settings or {}).get("batman_bkm_min_short_distance_points"), 450.0),
        min_short_width_points=_parse_float((settings or {}).get("batman_bkm_min_short_width_points"), 1000.0),
        max_center_offset_points=_parse_float((settings or {}).get("batman_bkm_max_center_offset_points"), 250.0),
        max_outer_distance_ratio=_parse_float((settings or {}).get("batman_bkm_max_outer_distance_ratio"), 1.80),
        max_tail_loss_imbalance_abs=_parse_float((settings or {}).get("batman_bkm_max_tail_loss_imbalance_abs"), 35000.0),
        max_tail_loss_imbalance_ratio=_parse_float((settings or {}).get("batman_bkm_max_tail_loss_imbalance_ratio"), 0.35),
        max_worst_loss_to_credit_ratio=_parse_float((settings or {}).get("batman_bkm_max_worst_loss_to_credit_ratio"), 25.0),
        quality_block_on_fail=bool((settings or {}).get("batman_bkm_quality_block_on_fail", True)),
    )
    strat = BatmanBKMStrategy(cfg)
    built_legs: List[BKMLeg] = []
    for leg in legs:
        try:
            built_legs.append(
                BKMLeg(
                    option_type=str(leg.get("option_type") or "").upper(),
                    side=str(leg.get("side") or "").upper(),
                    strike=_parse_float(leg.get("strike"), 0.0),
                    qty=_parse_int(leg.get("quantity"), 0),
                    entry=_parse_float(leg.get("entry_price"), 0.0),
                    ltp=_parse_float(leg.get("entry_price"), 0.0),
                    security_id=str(leg.get("security_id") or ""),
                    expiry=str(expiry or ""),
                )
            )
        except Exception:
            continue
    cm = _build_chain_map(str(expiry or ""))
    spot = cm.get("spot") if isinstance(cm, dict) else None
    quality = strat.assess_legs_quality(
        legs=built_legs,
        spot=_parse_float(spot, 0.0) if spot is not None else None,
        margin_required=float(margin_required or 0.0),
        net_credit=float(net_credit or 0.0),
    )
    return quality if isinstance(quality, dict) else {}


def _build_importable_bkm_from_broker_positions(positions: List[Dict[str, Any]], settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Validate and normalize a live broker basket into BKM journal/local-state payload.
    Returns: {"ok": bool, "imported": bool, ...}
    """
    if not isinstance(positions, list):
        return {"ok": False, "imported": False, "error": "invalid_broker_positions_payload"}
    if not positions:
        return {"ok": True, "imported": False, "reason": "NO_BROKER_OPEN_POSITIONS"}

    legs: List[Dict[str, Any]] = []
    parse_errors: List[str] = []
    for i, pos in enumerate(positions):
        if not isinstance(pos, dict):
            continue
        expiry_iso, strike_val, opt_val = _parse_bkm_pos_identity(pos)
        side = str(pos.get("side") or "").upper()
        qty = _parse_int(pos.get("qty"), 0)
        entry = _parse_float(pos.get("entry"), 0.0)
        sec_id = str(pos.get("sec_id") or "").strip()
        if side not in {"BUY", "SELL"} or qty <= 0 or strike_val is None or not opt_val or not expiry_iso:
            parse_errors.append(
                f"idx={i}: side={side} qty={qty} strike={strike_val} opt={opt_val} expiry={expiry_iso} raw={pos.get('strike')}"
            )
            continue
        legs.append(
            {
                "expiry": expiry_iso,
                "strike": float(strike_val),
                "option_type": opt_val,
                "side": side,
                "quantity": int(qty),
                "entry_price": float(entry or 0.0),
                "security_id": sec_id,
            }
        )

    if len(legs) != len(positions):
        return {"ok": False, "imported": False, "error": "BROKER_LEG_PARSE_FAILED", "details": {"parse_errors": parse_errors[:10]}}

    expiries = sorted({str(l["expiry"]) for l in legs})
    if len(expiries) != 1:
        return {"ok": False, "imported": False, "error": "MULTI_EXPIRY_OPEN_POSITIONS", "details": {"expiries": expiries}}
    expiry = expiries[0]

    # Focus on an exact 6-leg BKM basket.
    if len(legs) != 6:
        return {
            "ok": False,
            "imported": False,
            "error": "NOT_BKM_LEG_COUNT",
            "details": {"open_positions": len(legs), "expected": 6},
        }

    by_opt: Dict[str, List[Dict[str, Any]]] = {"CE": [], "PE": []}
    for leg in legs:
        opt = str(leg.get("option_type") or "").upper()
        if opt not in by_opt:
            return {"ok": False, "imported": False, "error": "UNSUPPORTED_OPTION_TYPE", "details": {"option_type": opt}}
        by_opt[opt].append(leg)
    if len(by_opt["CE"]) != 3 or len(by_opt["PE"]) != 3:
        return {
            "ok": False,
            "imported": False,
            "error": "NOT_BKM_CE_PE_SPLIT",
            "details": {"ce_count": len(by_opt["CE"]), "pe_count": len(by_opt["PE"])},
        }

    def _validate_side(opt: str, side_legs: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        legs_sorted = sorted(side_legs, key=lambda x: float(x.get("strike") or 0.0))
        strikes = [float(l["strike"]) for l in legs_sorted]
        if len(set(round(s, 4) for s in strikes)) != 3:
            return False, {"reason": "duplicate_strikes", "opt": opt, "strikes": strikes}
        side_pattern = [str(l.get("side") or "").upper() for l in legs_sorted]
        if side_pattern != ["BUY", "SELL", "BUY"]:
            return False, {"reason": "invalid_side_pattern", "opt": opt, "pattern": side_pattern, "strikes": strikes}
        buy_qtys = [int(l.get("quantity") or 0) for l in legs_sorted if str(l.get("side") or "").upper() == "BUY"]
        sell_qtys = [int(l.get("quantity") or 0) for l in legs_sorted if str(l.get("side") or "").upper() == "SELL"]
        if len(buy_qtys) != 2 or len(sell_qtys) != 1:
            return False, {"reason": "invalid_qty_split", "opt": opt}
        q_long = min(buy_qtys)
        q_hedge = max(buy_qtys)
        q_short = sell_qtys[0]
        if q_long <= 0 or (q_short % q_long) != 0:
            return False, {"reason": "invalid_short_ratio", "opt": opt, "q_long": q_long, "q_short": q_short}
        short_ratio = q_short // q_long
        if short_ratio not in {2, 3}:
            return False, {
                "reason": "unsupported_short_ratio",
                "opt": opt,
                "q_long": q_long,
                "q_short": q_short,
                "short_ratio": short_ratio,
            }
        if q_hedge < q_long or (q_hedge % q_long) != 0:
            return False, {"reason": "invalid_hedge_ratio", "opt": opt, "q_long": q_long, "q_hedge": q_hedge}
        return True, {
            "opt": opt,
            "sorted_legs": legs_sorted,
            "q_long": q_long,
            "q_short": q_short,
            "q_hedge": q_hedge,
            "short_ratio": short_ratio,
        }

    ok_ce, ce_info = _validate_side("CE", by_opt["CE"])
    ok_pe, pe_info = _validate_side("PE", by_opt["PE"])
    if not ok_ce or not ok_pe:
        return {"ok": False, "imported": False, "error": "NOT_BKM_PATTERN", "details": {"ce": ce_info, "pe": pe_info}}
    if int(ce_info.get("q_long", 0)) != int(pe_info.get("q_long", 0)):
        return {
            "ok": False,
            "imported": False,
            "error": "BKM_PRIMARY_QTY_MISMATCH",
            "details": {"ce_q_long": ce_info.get("q_long"), "pe_q_long": pe_info.get("q_long")},
        }

    # Preserve canonical BKM order expected by logs/tools: PE buy/sell/buy, then CE buy/sell/buy.
    pe_sorted = ce_sorted = []  # type: ignore[assignment]
    pe_sorted = pe_info["sorted_legs"]  # low->high => BUY,SELL,BUY (PE hedge,sell,buy)
    ce_sorted = ce_info["sorted_legs"]  # low->high => BUY,SELL,BUY
    # Reorder PE to buy near ATM first, sell middle, hedge far (high->mid->low).
    pe_ordered = [pe_sorted[2], pe_sorted[1], pe_sorted[0]]
    ce_ordered = [ce_sorted[0], ce_sorted[1], ce_sorted[2]]
    ordered_legs = pe_ordered + ce_ordered

    net_credit = 0.0
    for leg in ordered_legs:
        sign = 1.0 if str(leg.get("side") or "").upper() == "SELL" else -1.0
        net_credit += sign * float(leg.get("entry_price") or 0.0) * int(leg.get("quantity") or 0)

    settings = settings if isinstance(settings, dict) else _json_read(SETTINGS_JSON)
    est_margin = _parse_float((settings or {}).get("batman_bkm_estimated_margin"), 1_000_000.0)
    credit_pct = (net_credit / est_margin) * 100.0 if est_margin > 0 else 0.0
    quality = _assess_imported_bkm_quality(
        legs=ordered_legs,
        expiry=expiry,
        net_credit=net_credit,
        margin_required=est_margin,
        settings=settings or {},
    )
    quality_ok = bool(quality.get("ok")) if isinstance(quality, dict) else False
    import_quality_enforce = bool((settings or {}).get("batman_bkm_import_enforce_quality", False))
    if import_quality_enforce and not quality_ok:
        return {
            "ok": False,
            "imported": False,
            "error": "BKM_QUALITY_CHECK_FAILED",
            "details": {
                "quality": quality,
                "message": "Imported basket failed Batman quality checks.",
            },
        }

    meta = {
        "net_credit": net_credit,
        "margin_required": est_margin,
        "credit_pct": credit_pct,
        "imported_from_broker": True,
        "quality_status": str((quality or {}).get("status") or ("PASS" if quality_ok else "BLOCKED")),
        "quality_score": float((quality or {}).get("score") or 0.0),
        "quality_reasons": list((quality or {}).get("reasons") or []),
        "quality_metrics": dict((quality or {}).get("metrics") or {}),
    }
    return {
        "ok": True,
        "imported": True,
        "expiry": expiry,
        "legs": ordered_legs,
        "meta": meta,
        "quality": quality,
        "quality_warning": bool(not quality_ok),
        "q_long": int(ce_info["q_long"]),
        "short_ratio": int(ce_info.get("short_ratio") or 0),
        "ce_hedge_qty": int(ce_info["q_hedge"]),
        "pe_hedge_qty": int(pe_info["q_hedge"]),
    }


def _import_live_bkm_from_broker() -> Dict[str, Any]:
    # Refuse while agent is running to avoid in-memory/runtime divergence.
    pid_data = _json_read(PID_FILE)
    pid = int(pid_data.get("pid", 0)) if pid_data else 0
    if pid and _is_process_alive(pid):
        return {"ok": False, "imported": False, "error": "AGENT_RUNNING_STOP_FIRST", "pid": pid}

    broker_payload = _load_broker_live_positions()
    if str(broker_payload.get("source") or "") != "broker":
        return {"ok": False, "imported": False, "error": "BROKER_CHECK_UNAVAILABLE"}
    positions = broker_payload.get("positions") if isinstance(broker_payload, dict) else []
    settings = _json_read(SETTINGS_JSON)
    normalized = _build_importable_bkm_from_broker_positions(
        positions if isinstance(positions, list) else [],
        settings=settings if isinstance(settings, dict) else {},
    )
    if not bool(normalized.get("ok")) or not bool(normalized.get("imported")):
        return normalized

    expiry = str(normalized.get("expiry") or "")
    legs = normalized.get("legs") if isinstance(normalized.get("legs"), list) else []
    meta = normalized.get("meta") if isinstance(normalized.get("meta"), dict) else {}
    now = _ist_now().isoformat(timespec="seconds")

    # Close any previously active BKM basket in journal so analyze_bkm() sees exactly one active basket.
    lookback_days = max(1, _parse_int((settings or {}).get("live_exec_recovery_lookback_days"), 45))
    journal = ExecutionJournal(journal_path=EXECUTION_JOURNAL_JSONL, logger=None)
    summary = journal.analyze_bkm(lookback_days=lookback_days)
    active_baskets = summary.get("active_baskets") if isinstance(summary, dict) else {}
    for active_expiry, active_meta in (active_baskets or {}).items():
        payload: Dict[str, Any] = {
            "strategy": "BATMAN_BKM",
            "expiry": str(active_expiry),
            "op_id": f"IMPORTCLOSE-{int(time.time())}",
            "details": {"manual_reconcile": True, "reason": "import_live_bkm_replace_active"},
        }
        if isinstance(active_meta, dict):
            if isinstance(active_meta.get("legs"), list):
                payload["legs"] = active_meta["legs"]
            if isinstance(active_meta.get("meta"), dict):
                payload["meta"] = active_meta["meta"]
        journal.record("BKM_CLOSE_SUCCESS", payload, when=_ist_now())

    journal.record(
        "BKM_OPEN_SUCCESS",
        {
            "strategy": "BATMAN_BKM",
            "expiry": expiry,
            "op_id": f"BKMIMPORT-{int(time.time() * 1000)}",
            "legs": legs,
            "meta": meta,
            "details": {"opened_legs": len(legs), "planned_legs": len(legs), "imported_from_broker": True},
        },
        when=_ist_now(),
    )

    local_state = _json_read(STRATEGY_STATE)
    if not isinstance(local_state, dict):
        local_state = {}
    if not isinstance(local_state.get("BATMAN_V2"), dict):
        local_state["BATMAN_V2"] = {}
    local_state["BATMAN_BKM"] = {
        expiry: {
            "status": "OPEN",
            "opened_at": now,
            "net_credit": float(meta.get("net_credit") or 0.0),
            "credit_pct": float(meta.get("credit_pct") or 0.0),
            "quality_status": str(meta.get("quality_status") or "UNKNOWN"),
            "quality_score": float(meta.get("quality_score") or 0.0),
            "quality_reasons": list(meta.get("quality_reasons") or []),
            "imported_from_broker": True,
        }
    }
    _json_write(STRATEGY_STATE, local_state)

    rec_status = _reset_reconcile_status()
    exec_status = _reset_execution_recovery_status()
    return {
        "ok": True,
        "imported": True,
        "expiry": expiry,
        "legs": len(legs),
        "meta": meta,
        "quality": normalized.get("quality"),
        "quality_warning": bool(normalized.get("quality_warning", False)),
        "reconcile_status": rec_status.get("status"),
        "execution_recovery_status": exec_status.get("status"),
    }


def _load_agent_heartbeat() -> Dict[str, Any]:
    payload = _json_read(AGENT_HEARTBEAT_JSON)
    return payload if isinstance(payload, dict) else {}


def _write_agent_heartbeat_stub(*, status: str, phase: str, pid: Optional[int], trade_mode: Optional[str]) -> None:
    payload: Dict[str, Any] = {
        "status": str(status or "UNKNOWN").upper(),
        "phase": str(phase or "unknown"),
        "pid": None if not pid else int(pid),
        "trade_mode": str(trade_mode or "").lower() or None,
        "timestamp": _ist_now().isoformat(timespec="seconds"),
    }
    _json_write(AGENT_HEARTBEAT_JSON, payload)


def _clear_agent_heartbeat() -> None:
    try:
        if AGENT_HEARTBEAT_JSON.exists():
            AGENT_HEARTBEAT_JSON.unlink()
    except Exception:
        pass


def _watchdog_health_status() -> Dict[str, Any]:
    settings = _json_read(SETTINGS_JSON)
    stale_after = _parse_float(settings.get("ops_watchdog_stale_after_sec"), 45.0) if isinstance(settings, dict) else 45.0
    hb = _load_agent_heartbeat()
    watchdog = compute_watchdog_status(heartbeat_payload=hb, stale_after_sec=stale_after)
    pid_data = _json_read(PID_FILE)
    active_pid = int(pid_data.get("pid", 0)) if isinstance(pid_data, dict) else 0
    active_running = bool(active_pid and _is_process_alive(active_pid))
    hb_pid = _parse_int(hb.get("pid"), 0) if isinstance(hb, dict) else 0
    hb_status = str(hb.get("status") or "").upper() if isinstance(hb, dict) else ""
    if hb_status == "STOPPED":
        watchdog["status"] = "STOPPED"
    elif hb_pid and not _is_process_alive(hb_pid) and not active_running:
        watchdog["status"] = "STOPPED"
    elif hb_status == "STARTING" and active_running and hb_pid == active_pid:
        watchdog["status"] = "STARTING"
    watchdog["pid_alive"] = bool(hb_pid and _is_process_alive(hb_pid))
    watchdog["active_pid"] = active_pid or None
    watchdog["active_pid_running"] = active_running
    return watchdog


_preflight_cache: Dict[str, Any] = {}
_preflight_cache_ts: float = 0.0
_PREFLIGHT_TTL = 30.0          # whole rollup: cheap file reads, refresh often
_preflight_token_cache: Dict[str, Any] = {}
_preflight_token_cache_ts: float = 0.0
_PREFLIGHT_TOKEN_TTL = 300.0   # broker calls: throttle to once / 5 min


def _read_v83_plist_env() -> Dict[str, str]:
    """Env toggles the launchd job is configured with (USE_SELECTOR, SEL_* …)."""
    try:
        import plistlib
        with open(V83_PLIST_PATH, "rb") as fh:
            data = plistlib.load(fh)
        env = data.get("EnvironmentVariables") or {}
        return {str(k): str(v) for k, v in env.items()}
    except Exception:
        return {}


def _preflight_token_check() -> Dict[str, Any]:
    """Live broker reachability — cached 5 min so dashboard polls don't hammer Dhan."""
    global _preflight_token_cache, _preflight_token_cache_ts
    now = time.time()
    if _preflight_token_cache and (now - _preflight_token_cache_ts) < _PREFLIGHT_TOKEN_TTL:
        return _preflight_token_cache
    out: Dict[str, Any] = {"ok": False, "detail": "", "chain_ok": False, "chain_detail": ""}
    try:
        from data_engine.market_ai.dhan_wrapper import DhanWrapper  # type: ignore
        creds = _json_read(CREDS_FILE)
        cid = (creds.get("client_id") or "").strip() if isinstance(creds, dict) else ""
        tok = (creds.get("access_token") or "").strip() if isinstance(creds, dict) else ""
        if cid and tok:
            os.environ["DHAN_CLIENT_ID"] = cid
            os.environ["DHAN_ACCESS_TOKEN"] = tok
        dw = DhanWrapper(logger=None)
        funds = dw.get_funds()
        avail = funds.get("available") if isinstance(funds, dict) else None
        out["ok"] = bool(funds) and avail is not None
        out["detail"] = (f"funds available Rs {avail:,.0f}" if out["ok"]
                         else "get_funds empty — token may be expired")
        try:
            el = dw.get_optionchain_expirylist("IDX_I", INDEX_SECURITY_ID)
            out["chain_ok"] = bool(el)
            out["chain_detail"] = (f"{len(el)} expiries (next {el[0]})" if el else "no expiries returned")
        except Exception as exc:
            out["chain_detail"] = f"error: {str(exc)[:80]}"
    except Exception as exc:
        out["detail"] = f"error: {str(exc)[:100]}"
    _preflight_token_cache = out
    _preflight_token_cache_ts = time.time()
    return out


def _preflight_status() -> Dict[str, Any]:
    """One-glance 'is the agent ready to trade today' rollup: process, broker token,
    config armed, and the active strategy toggles. GREEN = clear to trade."""
    global _preflight_cache, _preflight_cache_ts
    now = time.time()
    if _preflight_cache and (now - _preflight_cache_ts) < _PREFLIGHT_TTL:
        return _preflight_cache

    # Market-session context (IST) — computed first so checks can be session-aware.
    try:
        from zoneinfo import ZoneInfo
        ist = datetime.now(ZoneInfo("Asia/Kolkata"))
        hhmm = ist.strftime("%H:%M")
        weekday = ist.weekday() < 5
        market_open = weekday and ("09:15" <= hhmm <= "15:30")
        session = ("OPEN" if market_open else ("PRE_OPEN" if (weekday and hhmm < "09:15")
                   else ("CLOSED" if weekday else "WEEKEND")))
        gen_at = ist.strftime("%Y-%m-%d %H:%M:%S IST")
    except Exception:
        session, market_open, gen_at = "UNKNOWN", False, ""

    checks: List[Dict[str, Any]] = []

    def add(key: str, label: str, ok: bool, detail: str, critical: bool = True) -> None:
        checks.append({
            "key": key, "label": label, "ok": bool(ok), "critical": critical,
            "status": "PASS" if ok else ("FAIL" if critical else "WARN"),
            "detail": detail,
        })

    # 1) Process / heartbeat — STALE heartbeat only matters once the market is open.
    try:
        wd = _watchdog_health_status()
        st = str(wd.get("status") or "").upper()
        alive = bool(wd.get("pid_alive") or wd.get("active_pid_running"))
        stopped = (st == "STOPPED") or (not alive)
        fresh_ok = st not in ("STALE", "STOPPED")
        if stopped:
            add("process", "Agent running", False, f"status={st or 'UNKNOWN'} — process not alive")
        elif market_open and not fresh_ok:
            add("process", "Agent running", False, f"status={st} during market hours — heartbeat stalled")
        else:
            note = st if fresh_ok else f"{st} (idle pre-open — normal)"
            add("process", "Agent running", True, f"status={note}, pid={wd.get('active_pid') or '—'}")
    except Exception as exc:
        add("process", "Agent running", False, f"error: {str(exc)[:80]}")

    # 2) Broker token + option chain (cached)
    tok = _preflight_token_check()
    add("token", "Broker token valid", bool(tok.get("ok")), tok.get("detail", ""))
    add("chain", "Option chain reachable", bool(tok.get("chain_ok")), tok.get("chain_detail", ""))

    # 3) Config armed (from runtime config file)
    cfg = _json_read(V83_RUNTIME_CONFIG_JSON)
    cfg = cfg if isinstance(cfg, dict) else {}
    mode = cfg.get("mode")
    live_arm = cfg.get("live_arm")
    frozen = cfg.get("v83_frozen")
    strategies = cfg.get("live_enabled_strategies") or []
    add("mode", "Paper mode", mode == "PAPER_LIVE", f"mode={mode}")
    add("safety", "Live-arm OFF (safe)", live_arm is False, f"live_arm={live_arm}")
    add("frozen", "Config not frozen", frozen is False, f"v83_frozen={frozen}", critical=False)
    add("engine", "Put-debit engine enabled", "PUT_DEBIT_SPREAD" in strategies,
        f"{len(strategies)} strategies enabled")

    # 4) Strategy toggles the launchd job runs with
    env = _read_v83_plist_env()
    add("selector", "Strategy selector ON", env.get("USE_SELECTOR") == "1",
        f"USE_SELECTOR={env.get('USE_SELECTOR') or 'unset'}")
    add("condor", "Condor-off (validated)", env.get("SEL_NO_CONDOR") == "1",
        f"SEL_NO_CONDOR={env.get('SEL_NO_CONDOR') or 'unset'}", critical=False)
    add("blackout", "10:00-10:30 blackout", bool(env.get("SEL_DEBIT_BLACKOUT")),
        f"SEL_DEBIT_BLACKOUT={env.get('SEL_DEBIT_BLACKOUT') or 'unset'}", critical=False)

    crit_fail = any((not c["ok"]) and c["critical"] for c in checks)
    warn = any((not c["ok"]) and (not c["critical"]) for c in checks)
    overall = "RED" if crit_fail else ("AMBER" if warn else "GREEN")

    result = {
        "overall": overall,
        "checks": checks,
        "pass_count": sum(1 for c in checks if c["ok"]),
        "total": len(checks),
        "market_session": session,
        "generated_at": gen_at,
    }
    _preflight_cache = result
    _preflight_cache_ts = now
    return result


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


_v83_ops_cache: Dict[str, Any] = {}
_v83_ops_cache_ts: float = 0.0
_V83_OPS_CACHE_TTL = 12.0

# Caches for secondary payloads not shown in the primary UI (60s TTL is fine).
_secondary_pnl_cache: Dict[str, Any] = {}
_secondary_pnl_cache_ts: float = 0.0
_SECONDARY_PNL_CACHE_TTL = 60.0

# /api/paper_positions rebuilds a HEAVY payload (incl. load_positions -> synchronous Dhan fetches
# that block up to a 429 cooldown → 6-22s observed). Serve-stale-while-rebuilding cache: rapid
# polls return instantly from cache; only ONE thread rebuilds at a time while the rest serve the
# last payload, so slow rebuilds never pile up or block the UI. Live P&L still comes fresh from
# the fast /api/open_position_live.
_paper_positions_cache: Optional[Dict[str, Any]] = None
_paper_positions_cache_ts: float = 0.0
_paper_positions_rebuild_lock = threading.Lock()
_PAPER_POSITIONS_TTL = 3.0

# /api/live_positions makes THREE synchronous broker fetches (blotter LTP enrichment, broker open
# positions, broker executions). In paper mode there are no broker positions, so these just time
# out/retry — measured 20-37s — and the Live Trades tab's Promise.all waits for the whole thing.
# Stale-while-revalidate: the endpoint returns the cached payload INSTANTLY and kicks the (slow)
# rebuild onto a background thread, so the tab never blocks. First load shows "loading" until the
# first rebuild lands.
_live_positions_cache: Optional[Dict[str, Any]] = None
_live_positions_cache_ts: float = 0.0
_live_positions_rebuilding = False
_live_positions_lock = threading.Lock()
_LIVE_POSITIONS_TTL = 10.0


def _build_live_positions_payload() -> Dict[str, Any]:
    """Fetch the user's REAL Dhan account — open positions (with Dhan's native LTP + P&L) and today's
    closed trades — so the Live Trades tab mirrors the Dhan app. Reads the real broker via creds.json
    even in PAPER mode (the agent's paper/live setting is irrelevant to what's in the Dhan account).

    Runs only on the background refresh thread (stale-while-revalidate), so its latency never blocks
    the tab. Deliberately does NOT call load_positions(BLOTTER) — that did a throttled option-chain
    LTP enrichment on a STALE local blotter and was the ~37s culprit; Dhan's positions API already
    carries LTP and unrealised P&L per leg."""
    broker_payload = _load_broker_live_positions()
    if isinstance(broker_payload, dict):
        payload: Dict[str, Any] = {
            "positions": broker_payload.get("positions", []) or [],
            "closed": broker_payload.get("closed", []) or [],
            "total_pnl": broker_payload.get("total_pnl", 0.0) or 0.0,
            "as_of": broker_payload.get("as_of"),
            "source": broker_payload.get("source") or "broker",
            "margin_available": broker_payload.get("margin_available"),
            "margin_used": broker_payload.get("margin_used"),
        }
    else:
        payload = {"positions": [], "closed": [], "total_pnl": 0.0, "source": "none"}
    # Merge today's broker executions (fills) as additional closed rows.
    try:
        trades = _load_broker_trades_today()
        if trades:
            closed = payload.get("closed", []) or []
            closed.extend(trades)
            closed.sort(key=lambda c: (
                c.get("expiry") not in (None, "", "0001-01-01"),
                c.get("entry") is not None and c.get("exit") is not None,
                c.get("side") == "FLAT"
            ), reverse=True)
            dedup = {}
            for c in closed:
                key = (str(c.get("sec_id")), c.get("strike"))
                if key not in dedup:  # sort above puts the FLAT row first, so it wins the key
                    dedup[key] = c
            # Only netted-out (side == FLAT) legs are genuinely CLOSED. The fills feed also carries
            # the open-side SELL/BUY executions of still-OPEN positions; those must not show as
            # closed (that inflated the count, e.g. 8 -> 11). Keep FLAT only, matching the Dhan app.
            payload["closed"] = [c for c in dedup.values() if str(c.get("side") or "").upper() == "FLAT"]
    except Exception:
        pass
    return payload


def _live_positions_bg_rebuild() -> None:
    global _live_positions_cache, _live_positions_cache_ts, _live_positions_rebuilding
    try:
        payload = _build_live_positions_payload()
        _live_positions_cache = payload
        _live_positions_cache_ts = time.time()
    except Exception:
        pass
    finally:
        _live_positions_rebuilding = False


def _load_v83_ops_status() -> Dict[str, Any]:
    global _v83_ops_cache, _v83_ops_cache_ts
    now = time.time()
    if _v83_ops_cache and (now - _v83_ops_cache_ts) < _V83_OPS_CACHE_TTL:
        return _v83_ops_cache
    try:
        cfg = _v83_load_runtime_config()
        broker_positions = None
        if cfg.mode == V83RuntimeMode.MICRO_LIVE:
            broker_payload = _load_broker_live_positions()
            positions = broker_payload.get("positions") if isinstance(broker_payload, dict) else None
            broker_positions = positions if isinstance(positions, list) else None
        result = _v83_operator_status_report(write=True, broker_positions=broker_positions)
        _v83_ops_cache.update(result)
        _v83_ops_cache_ts = time.time()
        return _v83_ops_cache
    except Exception as exc:
        return {
            "generated_at": _ist_now().isoformat(timespec="seconds"),
            "runtime_config": {"mode": "UNKNOWN", "live_arm": False},
            "runtime_state": {"primary_block_reason": "OPS_STATUS_ERROR"},
            "health": {"status": "BLOCKED", "block_reasons": [str(exc)], "components": {}},
            "broker_sync_status": {"status": "UNKNOWN", "reason": "OPS_STATUS_ERROR"},
            "recovery_status": {"active": True, "reason": "OPS_STATUS_ERROR"},
            "error": str(exc),
        }


def _is_process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except Exception:
        return False
    # os.kill(pid, 0) succeeds for zombie processes too — exclude them.
    try:
        result = subprocess.run(
            ["ps", "-o", "stat=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2,
        )
        stat = result.stdout.strip()
        return bool(stat) and not stat.startswith("Z")
    except Exception:
        return True


def _api_error_payload(exc: BaseException, *, code: str) -> Dict[str, Any]:
    tb = traceback.format_exc(limit=12)
    return {
        "ok": False,
        "error": code,
        "detail": f"{type(exc).__name__}: {exc}",
        "traceback": tb.splitlines()[-12:],
    }


def _write_v83_run_config(*, trade_mode: str) -> Path:
    payload = {
        "provider_class": "market_ai.intraday_defined_risk.live_runtime:DhanLiveMarketDataProvider",
        "executor_class": "market_ai.intraday_defined_risk.live_runtime:PaperOnlyExecutor",
        "learning_db_path": str(STATE_DIR / "intraday_v83_paper_live_learning.sqlite3"),
        "poll_seconds": 30,
        "underlying_id": INDEX_SECURITY_ID,
        "underlying_seg": INDEX_EXCHANGE_SEG,
        "lot_size": 65,
        "slippage_points": 0.5,
        "margin_estimate_per_lot": 20_000.0,
        "max_risk_rupees_per_trade": 18_000.0,
        "max_margin_rupees": 810_000.0,
        "max_daily_loss_rupees": 18_000.0,
        "dhan_http_timeout_sec": 8,
        "dhan_http_retries": 1,
        "dhan_option_chain_attempts": 1,
        "trade_mode": trade_mode,
    }
    V83_RUN_CONFIG_JSON.parent.mkdir(parents=True, exist_ok=True)
    _json_write(V83_RUN_CONFIG_JSON, payload)
    return V83_RUN_CONFIG_JSON


_V83_LAUNCHD_LABEL = "com.algoagent.intraday_v83"


def _kickstart_v83_via_launchd() -> subprocess.CompletedProcess:
    """Use launchctl kickstart to restart the canonical launchd-managed v83 job."""
    # Kill any running instances first so kickstart -k doesn't hang waiting for
    # a slow graceful shutdown (agent can take >15s to exit during market hours).
    for p in _find_all_v83_pids():
        try:
            os.kill(p, 9)
        except OSError:
            pass
    time.sleep(0.5)
    try:
        return subprocess.run(
            ["launchctl", "kickstart", "-k", f"gui/{os.getuid()}/{_V83_LAUNCHD_LABEL}"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired:
        # Return a fake failed result so _start_v83_process falls through to the
        # subprocess fallback path.
        return subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="launchctl kickstart timed out"
        )


def _start_v83_process(*, trade_mode: str) -> Dict[str, Any]:
    trade_mode = str(trade_mode or "paper").strip().lower() or "paper"
    if trade_mode != "paper":
        return {
            "ok": False,
            "error": "V83_LIVE_START_BLOCKED",
            "message": "v83 can be started from this UI only in PAPER_LIVE. MICRO_LIVE requires explicit runtime arming.",
        }
    _write_v83_run_config(trade_mode=trade_mode)
    runtime_config = _v83_set_runtime_mode(V83RuntimeMode.PAPER_LIVE, live_arm=False, source="ui_start_agent")

    # Prefer launchctl kickstart so the launchd-managed job is the sole running
    # instance. Spawning via subprocess alongside a KeepAlive launchd job creates
    # duplicate processes that race on shared state files.
    kickstart_result = _kickstart_v83_via_launchd()
    if kickstart_result.returncode == 0:
        # Give launchd a moment to start the process, then find its PID.
        time.sleep(1.5)
        new_pids = _find_all_v83_pids()
        new_pid = new_pids[0] if new_pids else None
        pid_payload = {
            "pid": new_pid,
            "trade_mode": trade_mode,
            "strategy_file": "intraday_defined_risk_v83",
            "v83_runtime_mode": runtime_config.mode.value,
            "started_at": datetime.now().isoformat(),
        }
        _json_write(PID_FILE, pid_payload)
        _write_agent_heartbeat_stub(status="STARTING", phase="v83_launch_requested", pid=new_pid, trade_mode=trade_mode)
        return {
            "ok": True,
            "pid": new_pid,
            "trade_mode": trade_mode,
            "strategy_file": "intraday_defined_risk_v83",
            "v83_runtime_mode": runtime_config.mode.value,
            "log_path": str(V83_RUNNER_LOG),
            "start_method": "launchctl_kickstart",
        }

    # Fallback: launchctl not available or job not loaded — spawn directly.
    env = os.environ.copy()
    env["TRADE_MODE"] = trade_mode
    data_engine_path = str(ROOT / "data_engine")
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = data_engine_path if not existing_pythonpath else f"{data_engine_path}:{existing_pythonpath}"
    pybin = sys.executable or "python3"
    cmd = [
        pybin,
        "-u",
        "-m",
        "market_ai.intraday_defined_risk.cli",
        "run_live",
        "--config",
        str(ROOT / "data_engine" / "market_ai" / "state" / "intraday_v83_run_live_config.json"),
    ]
    V83_RUNNER_LOG.parent.mkdir(parents=True, exist_ok=True)
    with V83_RUNNER_LOG.open("ab") as log_handle:
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log_handle,
            stderr=log_handle,
            env=env,
        )
    pid_payload = {
        "pid": proc.pid,
        "trade_mode": trade_mode,
        "strategy_file": "intraday_defined_risk_v83",
        "v83_runtime_mode": runtime_config.mode.value,
        "started_at": datetime.now().isoformat(),
    }
    _json_write(PID_FILE, pid_payload)
    _write_agent_heartbeat_stub(status="STARTING", phase="v83_launch_requested", pid=proc.pid, trade_mode=trade_mode)
    return {
        "ok": True,
        "pid": proc.pid,
        "trade_mode": trade_mode,
        "strategy_file": "intraday_defined_risk_v83",
        "v83_runtime_mode": runtime_config.mode.value,
        "log_path": str(V83_RUNNER_LOG),
        "start_method": "subprocess_fallback",
        "launchctl_error": kickstart_result.stderr,
    }


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
    try:
        selected_strategy = str((_json_read(LAST_STRATEGY_FILE) or {}).get("strategy_file") or "batman_bkm_monthly").strip()
    except Exception:
        selected_strategy = "batman_bkm_monthly"
    if selected_strategy == "intraday_defined_risk_v83":
        return _start_v83_process(trade_mode=trade_mode)
    auto_import_result: Optional[Dict[str, Any]] = None
    if trade_mode == "live":
        try:
            strategy_file = selected_strategy
        except Exception:
            strategy_file = "batman_bkm_monthly"
        if strategy_file == "batman_bkm_monthly":
            try:
                auto_import_result = _import_live_bkm_from_broker()
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise
                auto_import_result = _api_error_payload(exc, code="LIVE_BKM_AUTO_IMPORT_CRASH")
            # Fail-safe: if we cannot verify/import live broker state, do not start and risk locking/orphan behavior.
            if not bool(auto_import_result.get("ok")):
                nonfatal_auto_import_errors = {
                    "NOT_BKM_LEG_COUNT",
                    "NOT_BKM_CE_PE_SPLIT",
                    "NOT_BKM_PATTERN",
                }
                auto_import_error = str(auto_import_result.get("error") or "").upper()
                if auto_import_error in nonfatal_auto_import_errors:
                    auto_import_result = {
                        **auto_import_result,
                        "ok": True,
                        "imported": False,
                        "warning": "BROKER_POSITIONS_NOT_BKM_IGNORED",
                    }
                else:
                    return {
                        "ok": False,
                        "error": str(auto_import_result.get("error") or "LIVE_BKM_AUTO_IMPORT_FAILED"),
                        "auto_import": auto_import_result,
                    }
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
    _write_agent_heartbeat_stub(status="STARTING", phase="launch_requested", pid=proc.pid, trade_mode=trade_mode)
    out = {"ok": True, "pid": proc.pid, "trade_mode": trade_mode}
    if auto_import_result is not None:
        out["auto_import"] = auto_import_result
    return out


def _find_all_v83_pids() -> list:
    """Return PIDs of all running run_live v83 agent processes."""
    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "market_ai.intraday_defined_risk.cli"],
            text=True,
        ).strip()
        return [int(p) for p in out.splitlines() if p.strip().isdigit()]
    except subprocess.CalledProcessError:
        return []


def _stop_agent_process(*, auto_cleanup_bkm: bool = True, graceful_wait_sec: float = 5.0) -> Dict[str, Any]:
    pid_data = _json_read(PID_FILE)
    pid = int(pid_data.get("pid", 0)) if pid_data else 0
    trade_mode = str(pid_data.get("trade_mode") or "").lower() if isinstance(pid_data, dict) else None
    # Kill the PID-file process AND any launchd-managed or orphan duplicates.
    pids_to_kill = set(_find_all_v83_pids())
    if pid:
        pids_to_kill.add(pid)
    for p in pids_to_kill:
        try:
            if _is_process_alive(p):
                os.kill(p, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.time() + max(0.0, float(graceful_wait_sec))
    while time.time() < deadline:
        if not any(_is_process_alive(p) for p in pids_to_kill):
            break
        time.sleep(0.2)
    still_running = any(_is_process_alive(p) for p in pids_to_kill)
    _json_write(PID_FILE, {})
    if still_running:
        _write_agent_heartbeat_stub(status="STOPPING", phase="stop_requested", pid=pid or None, trade_mode=trade_mode)
    else:
        _write_agent_heartbeat_stub(status="STOPPED", phase="api_stop", pid=pid or None, trade_mode=trade_mode)
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
        learning_status_path=BATMAN_BKM_LEARNING_STATUS_JSON,
        learning_outcomes_path=BATMAN_BKM_LEARNING_OUTCOMES_JSONL,
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


# ── Weekly Iron Condor deploy / close helpers ─────────────────────────────────

def _dhan_place_order(
    creds: dict,
    side: str,
    security_id: int,
    quantity: int,
    price: float,
    exchange_seg: str = "NSE_FNO",
    product_type: str = "MARGIN",
    order_type: str = "LIMIT",
) -> Dict[str, Any]:
    """Place a single order via Dhan /v2/orders. Returns the JSON response."""
    import requests as _req
    base = os.getenv("DHAN_API_BASE", "https://api.dhan.co").rstrip("/")
    headers = {
        "client-id": str(creds.get("client_id") or ""),
        "access-token": str(creds.get("access_token") or ""),
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "transactionType": side.upper(),
        "exchangeSegment": exchange_seg,
        "productType": product_type.upper(),
        "orderType": order_type,
        "validity": "DAY",
        "securityId": str(security_id),
        "quantity": int(quantity),
        "price": round(float(price), 2) if order_type == "LIMIT" else 0,
        "afterMarketOrder": False,
        "amoTime": None,
        "dhanClientId": str(creds.get("client_id") or ""),
    }
    resp = _req.post(f"{base}/v2/orders", headers=headers, json=payload, timeout=12)
    try:
        js = resp.json()
    except Exception:
        js = {"raw": resp.text[:300]}
    js["_http_status"] = resp.status_code
    return js


def _deploy_weekly_ic(pending: Dict[str, Any]) -> Dict[str, Any]:
    """Place all 4 IC legs and return a result dict with per-leg status."""
    creds = _json_read(CREDS_FILE)
    qty = int(pending.get("quantity") or 0)
    legs = [
        ("SELL", pending.get("sc_sid"), pending.get("sc_ltp", 0), f"SELL {pending.get('short_call')} CE"),
        ("BUY",  pending.get("lc_sid"), pending.get("lc_ltp", 0), f"BUY  {pending.get('long_call')}  CE"),
        ("SELL", pending.get("sp_sid"), pending.get("sp_ltp", 0), f"SELL {pending.get('short_put')}  PE"),
        ("BUY",  pending.get("lp_sid"), pending.get("lp_ltp", 0), f"BUY  {pending.get('long_put')}   PE"),
    ]
    results = []
    for side, sid, ltp, label in legs:
        if not sid:
            results.append({"label": label, "ok": False, "error": "no security_id"})
            continue
        try:
            r = _dhan_place_order(creds, side, int(sid), qty, float(ltp or 0))
            ok = r.get("_http_status", 0) < 400 and r.get("status") != "failed"
            results.append({"label": label, "ok": ok, "order_id": r.get("orderId") or r.get("order_id"), "resp": r})
        except Exception as exc:
            results.append({"label": label, "ok": False, "error": str(exc)})
    all_ok = all(r["ok"] for r in results)
    return {"ok": all_ok, "legs": results}


def _close_weekly_ic(position: Dict[str, Any]) -> Dict[str, Any]:
    """Reverse all 4 IC legs (BUY back shorts, SELL longs)."""
    creds = _json_read(CREDS_FILE)
    qty = int(position.get("quantity") or 0)
    legs = [
        ("BUY",  position.get("sc_sid"), position.get("sc_ltp", 0), f"BUY  {position.get('short_call')} CE"),
        ("SELL", position.get("lc_sid"), position.get("lc_ltp", 0), f"SELL {position.get('long_call')}  CE"),
        ("BUY",  position.get("sp_sid"), position.get("sp_ltp", 0), f"BUY  {position.get('short_put')}  PE"),
        ("SELL", position.get("lp_sid"), position.get("lp_ltp", 0), f"SELL {position.get('long_put')}   PE"),
    ]
    results = []
    for side, sid, ltp, label in legs:
        if not sid:
            results.append({"label": label, "ok": False, "error": "no security_id"})
            continue
        try:
            r = _dhan_place_order(creds, side, int(sid), qty, float(ltp or 0))
            ok = r.get("_http_status", 0) < 400 and r.get("status") != "failed"
            results.append({"label": label, "ok": ok, "order_id": r.get("orderId") or r.get("order_id"), "resp": r})
        except Exception as exc:
            results.append({"label": label, "ok": False, "error": str(exc)})
    all_ok = all(r["ok"] for r in results)
    return {"ok": all_ok, "legs": results}


_WEEKLY_IC_HTML_STYLE = """
<style>
  body { font-family: -apple-system, Arial, sans-serif; background: #0d1117; color: #c9d1d9;
         max-width: 600px; margin: 60px auto; padding: 24px; }
  h2 { color: #58a6ff; }
  .ok   { color: #3fb950; }
  .fail { color: #f85149; }
  .leg  { font-family: monospace; padding: 6px 0; border-bottom: 1px solid #21262d; }
  .box  { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin: 16px 0; }
  a     { color: #58a6ff; }
</style>
"""


def _weekly_ic_deploy_page(pending: Dict[str, Any], result: Dict[str, Any]) -> str:
    expiry = pending.get("expiry", "?")
    spot   = pending.get("spot_at_entry", 0)
    legs_html = "".join(
        f'<div class="leg {"ok" if r["ok"] else "fail"}">'
        f'{"✅" if r["ok"] else "❌"} {r["label"]} — '
        f'{"Order " + str(r.get("order_id") or "sent") if r["ok"] else r.get("error", "failed")}'
        f"</div>"
        for r in result["legs"]
    )
    status_cls = "ok" if result["ok"] else "fail"
    status_txt = "✅ All 4 legs placed" if result["ok"] else "⚠️ Some legs failed — check Dhan order book"
    return f"""<!DOCTYPE html><html><head><title>Weekly IC Deploy</title>{_WEEKLY_IC_HTML_STYLE}</head>
<body>
<h2>📊 Weekly IC Deploy — {expiry}</h2>
<div class="box">
  <b>NIFTY spot at plan:</b> {spot:,.0f}&nbsp;&nbsp;
  <b>Expiry:</b> {expiry}<br><br>
  {legs_html}
</div>
<p class="{status_cls}"><b>{status_txt}</b></p>
<p>Check your <a href="https://web.dhan.co" target="_blank">Dhan order book</a> to verify fills.
Prices were set at plan LTP — adjust manually if needed.</p>
<p style="color:#8b949e;font-size:0.85em">Exit by next Tuesday before 3:20 PM IST.<br>
Use the close link from your Telegram alert when ready to exit.</p>
</body></html>"""


def _weekly_ic_close_page(position: Dict[str, Any], result: Dict[str, Any]) -> str:
    expiry = position.get("expiry", "?")
    legs_html = "".join(
        f'<div class="leg {"ok" if r["ok"] else "fail"}">'
        f'{"✅" if r["ok"] else "❌"} {r["label"]} — '
        f'{"Order " + str(r.get("order_id") or "sent") if r["ok"] else r.get("error", "failed")}'
        f"</div>"
        for r in result["legs"]
    )
    status_cls = "ok" if result["ok"] else "fail"
    status_txt = "✅ All 4 close orders placed" if result["ok"] else "⚠️ Some close orders failed — check Dhan"
    return f"""<!DOCTYPE html><html><head><title>Weekly IC Close</title>{_WEEKLY_IC_HTML_STYLE}</head>
<body>
<h2>🔴 Weekly IC Close — {expiry}</h2>
<div class="box">
  {legs_html}
</div>
<p class="{status_cls}"><b>{status_txt}</b></p>
<p>Check your <a href="https://web.dhan.co" target="_blank">Dhan order book</a> to verify fills.</p>
</body></html>"""


def _send_weekly_ic_telegram(text: str) -> None:
    try:
        creds = _json_read(CREDS_FILE)
        bot_token = str(creds.get("telegram_bot_token") or "").strip()
        chat_id   = str(creds.get("telegram_chat_id") or "").strip()
        if not bot_token or not chat_id:
            return
        import requests as _req
        _req.post(
            f"https://api.telegram.org/bot{bot_token}/sendMessage",
            json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        pass


class PaperHandler(SimpleHTTPRequestHandler):
    def _send_json(self, payload: Dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
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
        if self.path.startswith("/api/bkm/import_live"):
            try:
                result = _import_live_bkm_from_broker()
                status = 200 if bool(result.get("ok")) else 409
                self._send_json(result, status=status)
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise
                self._send_json(_api_error_payload(exc, code="BKM_IMPORT_LIVE_HANDLER_CRASH"), status=500)
            return
        if self.path.startswith("/api/bkm/ai_protect/clear"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(raw.decode("utf-8")) if raw else {}
                result = _clear_bkm_ai_protect_lock(
                    source=str((data or {}).get("source") or "manual_ui"),
                    note=str((data or {}).get("note") or ""),
                )
                self._send_json(result)
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise
                self._send_json(_api_error_payload(exc, code="BKM_AI_PROTECT_CLEAR_HANDLER_CRASH"), status=500)
            return
        if self.path.startswith("/api/bkm/ai_protect/set"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(raw.decode("utf-8")) if raw else {}
                enabled = bool((data or {}).get("enabled", True))
                result = _set_bkm_ai_protection_enabled(
                    enabled=enabled,
                    source=str((data or {}).get("source") or "manual_ui"),
                    note=str((data or {}).get("note") or ""),
                )
                self._send_json(result)
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise
                self._send_json(_api_error_payload(exc, code="BKM_AI_PROTECT_SET_HANDLER_CRASH"), status=500)
            return
        if self.path.startswith("/api/intraday_ai/deploy"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(raw.decode("utf-8")) if raw else {}
                source = str((data or {}).get("source") or "manual_ui")
                note = str((data or {}).get("note") or "")

                settings = _json_read(SETTINGS_JSON)
                pid_data = _json_read(PID_FILE)
                pid = int(pid_data.get("pid", 0)) if isinstance(pid_data, dict) else 0
                running = bool(pid and _is_process_alive(pid))
                require_running = bool((settings or {}).get("intraday_ai_deploy_require_agent_running", True))
                if require_running and not running:
                    self._send_json(
                        {
                            "ok": False,
                            "queued": False,
                            "error": "AGENT_NOT_RUNNING",
                            "message": "Start the agent before deploying.",
                        },
                        status=409,
                    )
                    return
                require_live_mode = bool((settings or {}).get("intraday_ai_deploy_require_live_mode", True))
                active_mode = str((pid_data or {}).get("trade_mode") or "").strip().lower()
                execution_policy = _execution_policy(
                    settings,
                    active_mode or str((settings or {}).get("trade_mode") or "").strip().lower(),
                )
                intraday_mode = str(execution_policy.get("intraday_ai_mode") or _intraday_ai_mode_name(settings))
                if not bool((settings or {}).get("intraday_ai_enabled", True)):
                    self._send_json(
                        {
                            "ok": False,
                            "queued": False,
                            "error": "INTRADAY_AI_DISABLED",
                            "message": "Intraday AI is disabled.",
                        },
                        status=409,
                    )
                    return
                if not bool(execution_policy.get("new_entries_allowed", False)):
                    self._send_json(
                        {
                            "ok": False,
                            "queued": False,
                            "error": "AGENT_EXECUTION_MODE_BLOCKS_NEW_ENTRIES",
                            "message": f"Agent execution mode is {execution_policy.get('mode')}; new entries are blocked.",
                            "agent_execution_mode": execution_policy.get("mode"),
                        },
                        status=409,
                    )
                    return
                if intraday_mode != "EXECUTION":
                    self._send_json(
                        {
                            "ok": False,
                            "queued": False,
                            "error": "INTRADAY_AI_NOT_EXECUTION_MODE",
                            "message": f"Intraday AI mode is {intraday_mode}. Switch it to EXECUTION before deploying.",
                            "intraday_ai_mode": intraday_mode,
                        },
                        status=409,
                    )
                    return
                if not bool(execution_policy.get("intraday_new_entries_allowed", False)):
                    self._send_json(
                        {
                            "ok": False,
                            "queued": False,
                            "error": "INTRADAY_AI_EXECUTION_DISABLED",
                            "message": f"Intraday execution is disabled for {active_mode or str((settings or {}).get('trade_mode') or '').strip().lower() or 'this'} mode.",
                            "intraday_ai_mode": intraday_mode,
                            "trade_mode": active_mode or None,
                            "agent_execution_mode": execution_policy.get("mode"),
                        },
                        status=409,
                    )
                    return
                if require_live_mode and active_mode != "live":
                    self._send_json(
                        {
                            "ok": False,
                            "queued": False,
                            "error": "AGENT_NOT_LIVE_MODE",
                            "message": "Agent is not in live mode.",
                            "trade_mode": active_mode or None,
                        },
                        status=409,
                    )
                    return
                strategy = str((_json_read(LAST_STRATEGY_FILE) or {}).get("strategy_file") or "batman_bkm_monthly").strip()
                if strategy != "batman_bkm_monthly":
                    self._send_json(
                        {
                            "ok": False,
                            "queued": False,
                            "error": "UNSUPPORTED_STRATEGY",
                            "message": "Intraday AI deploy is currently supported only with batman_bkm_monthly runtime.",
                            "strategy_file": strategy,
                        },
                        status=409,
                    )
                    return
                result = _queue_intraday_ai_deploy_request(source=source, note=note)
                status = 200 if bool(result.get("ok")) else 409
                self._send_json(result, status=status)
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise
                self._send_json(_api_error_payload(exc, code="INTRADAY_AI_DEPLOY_HANDLER_CRASH"), status=500)
            return
        if self.path.startswith("/api/intraday_defined_risk/runtime_mode"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(raw.decode("utf-8")) if raw else {}
                mode = str((data or {}).get("mode") or "").strip().upper()
                if mode not in {item.value for item in V83RuntimeMode}:
                    self._send_json({"ok": False, "error": "INVALID_RUNTIME_MODE", "mode": mode}, status=400)
                    return
                live_arm = bool((data or {}).get("live_arm", False))
                config = _v83_set_runtime_mode(mode, live_arm=live_arm, source=str((data or {}).get("source") or "ui"))
                self._send_json({"ok": True, "runtime_config": config.to_dict(), "ops": _load_v83_ops_status()})
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise
                self._send_json(_api_error_payload(exc, code="V83_RUNTIME_MODE_HANDLER_CRASH"), status=500)
            return
        if self.path.startswith("/api/intraday_defined_risk/flatten_all"):
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length) if length > 0 else b"{}"
                data = json.loads(raw.decode("utf-8")) if raw else {}
                result = _v83_flatten_all_emergency(source=str((data or {}).get("source") or "ui"))
                self._send_json({"ok": True, "result": result, "ops": _load_v83_ops_status()})
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise
                self._send_json(_api_error_payload(exc, code="V83_FLATTEN_ALL_HANDLER_CRASH"), status=500)
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
                self._send_json({
                    "ok": True,
                    "pid": start_res.get("pid"),
                    "trade_mode": trade_mode,
                    "strategy_file": start_res.get("strategy_file"),
                    "v83_runtime_mode": start_res.get("v83_runtime_mode"),
                    "stop": stop_res,
                    "start": start_res,
                    "auto_import": start_res.get("auto_import"),
                })
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise
                self._send_json(_api_error_payload(exc, code="CONTROL_RESTART_HANDLER_CRASH"), status=500)
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
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise
                self._send_json(_api_error_payload(exc, code="CONTROL_START_HANDLER_CRASH"), status=500)
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
        if self.path.startswith("/api/preflight"):
            try:
                self._send_json(_preflight_status())
            except Exception as exc:  # pragma: no cover - defensive
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/open_position_live"):
            # Lightweight endpoint for FAST (~3s) polling of just the open position's live
            # legs + total — avoids the heavy /api/paper_positions payload so the UI can
            # refresh the live P&L far more often than the 15s full loop.
            try:
                self._send_json({"v83_open_position": _build_v83_open_position_payload()})
            except Exception as exc:  # pragma: no cover - defensive
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/paper_positions"):
            try:
                global _paper_positions_cache, _paper_positions_cache_ts
                _now = time.time()
                # Fresh cache → instant.
                if _paper_positions_cache is not None and (_now - _paper_positions_cache_ts) < _PAPER_POSITIONS_TTL:
                    self._send_json(_paper_positions_cache)
                    return
                # Stale: only ONE thread rebuilds; everyone else serves the last payload so a slow
                # rebuild never blocks the UI or piles up.
                if not _paper_positions_rebuild_lock.acquire(blocking=False):
                    if _paper_positions_cache is not None:
                        self._send_json(_paper_positions_cache)
                        return
                    _paper_positions_rebuild_lock.acquire()  # first-ever load: wait for the build
                try:
                    # ── V83-specific data (primary source of truth for PnL tab) ──
                    intraday_perf = _build_intraday_performance_payload(trade_mode="paper")
                    v83_open = _build_v83_open_position_payload()
                    v83_market_state = _get_v83_market_state()

                    # Spot for the payoff chart now comes from the background LTP refresher's cache
                    # (free) instead of load_positions(BLOTTER) — a SYNCHRONOUS Dhan fetch that
                    # dominated this endpoint's latency (6-22s) and enriched a blotter v83 doesn't
                    # even use. _build_v83_open_position_payload already started the refresher.
                    _spot_cached = _open_leg_spot

                    payload: Dict[str, Any] = {
                        # V83 open position — authoritative source for Open Legs / MTM
                        "v83_open_position": v83_open,
                        "positions": v83_open["legs"],          # UI legs-table uses this
                        "total_pnl": v83_open["open_mtm"] or 0.0,
                        # V83 realized P&L — authoritative source (intraday_performance)
                        "realized_pnl": intraday_perf["realized_pnl_rs"],
                        # Market state for the "Market Status" card
                        "v83_market_state": v83_market_state,
                        # Pass-through fields the UI also uses
                        "spot": _spot_cached,
                        "ltp_status": "LIVE" if _spot_cached else "STALE",
                        "as_of": None,
                        # Full performance payload for Trade History section
                        "intraday_performance": intraday_perf,
                        # Strategy state kept for any legacy widget that still reads it
                        "strategy_state": {},
                    }

                    # Secondary payloads (batman/committee/learning) cached 60 s —
                    # no longer rendered in main UI but kept for completeness.
                    global _secondary_pnl_cache, _secondary_pnl_cache_ts
                    _now_ts = time.time()
                    if not _secondary_pnl_cache or (_now_ts - _secondary_pnl_cache_ts) > _SECONDARY_PNL_CACHE_TTL:
                        _secondary_pnl_cache = {
                            "committee_review": _build_committee_review_payload(trade_mode="paper"),
                            "intraday_learning": _load_intraday_learning_status(),
                            "batman_bkm_review": _build_batman_bkm_review_payload(trade_mode="paper"),
                            "batman_bkm_learning": _build_batman_bkm_learning_payload(trade_mode="paper"),
                        }
                        _secondary_pnl_cache_ts = _now_ts
                    payload.update(_secondary_pnl_cache)
                    _paper_positions_cache = payload
                    _paper_positions_cache_ts = time.time()
                finally:
                    _paper_positions_rebuild_lock.release()
                self._send_json(payload)
            except Exception as exc:  # pragma: no cover - defensive
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/live_positions"):
            try:
                global _live_positions_rebuilding
                _now = time.time()
                fresh = (
                    _live_positions_cache is not None
                    and (_now - _live_positions_cache_ts) < _LIVE_POSITIONS_TTL
                )
                # Stale (or never built) → kick a background rebuild, but DON'T wait for it.
                if not fresh and not _live_positions_rebuilding:
                    with _live_positions_lock:
                        if not _live_positions_rebuilding:
                            _live_positions_rebuilding = True
                            threading.Thread(
                                target=_live_positions_bg_rebuild, daemon=True, name="live-positions"
                            ).start()
                # Serve whatever we have RIGHT NOW — never block the Live Trades tab on the broker.
                self._send_json(
                    _live_positions_cache
                    or {"positions": [], "closed": [], "total_pnl": 0.0, "source": "loading"}
                )
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
        if self.path.startswith("/api/intraday_defined_risk/ops_status"):
            try:
                self._send_json({"ok": True, "ops": _load_v83_ops_status()})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/control/status"):
            try:
                settings = _json_read(SETTINGS_JSON)
                pid_data = _json_read(PID_FILE)
                pid = int(pid_data.get("pid", 0)) if pid_data else 0
                running = pid and _is_process_alive(pid)
                gate = _load_live_gate_status()
                reconcile = _load_reconcile_status()
                exec_recovery = _load_execution_recovery_status()
                bkm_ai = _load_bkm_ai_status()
                bkm_learning = _load_bkm_learning_status()
                committee = _load_decision_committee_status()
                intraday_learning = _load_intraday_learning_status()
                intraday_ai = _load_intraday_ai_status()
                intraday_deploy = _load_intraday_ai_deploy_status()
                intraday_position = _load_intraday_ai_position_status()
                bkm_ai_plan = bkm_ai.get("plan") if isinstance(bkm_ai.get("plan"), dict) else {}
                bkm_ai_ctx = bkm_ai.get("market_context") if isinstance(bkm_ai.get("market_context"), dict) else {}
                bkm_ai_structure = bkm_ai_ctx.get("structure") if isinstance(bkm_ai_ctx.get("structure"), dict) else {}
                bkm_ai_protect = _load_bkm_ai_protect_status()
                intraday_ai_rec = intraday_ai.get("recommendation") if isinstance(intraday_ai.get("recommendation"), dict) else {}
                runtime_trade_mode = (
                    str(pid_data.get("trade_mode") or "").strip().lower()
                    if isinstance(pid_data, dict)
                    else ""
                ) or str((settings or {}).get("trade_mode") or "").strip().lower()
                execution_policy = _execution_policy(settings, runtime_trade_mode)
                intraday_ai_mode = str(
                    intraday_ai.get("mode") or execution_policy.get("intraday_ai_mode") or _intraday_ai_mode_name(settings)
                ).upper()
                intraday_ai_execution_enabled = bool(execution_policy.get("intraday_new_entries_allowed", False))
                today_ist = _ist_now().date().isoformat()
                ai_mode_name = str(bkm_ai.get("mode") or "ADVISOR").upper()
                ai_mode_protect = ai_mode_name in {"AUTO_PROTECT", "PROTECTIVE"}
                protect_active = bool(
                    ai_mode_protect
                    and
                    bkm_ai_protect.get("protection_enabled", True)
                    and
                    bkm_ai_protect.get("active", False)
                    and str(bkm_ai_protect.get("session_date") or "") == today_ist
                )
                protect_enabled = (
                    bool(bkm_ai_protect.get("protection_enabled", True))
                    if str(bkm_ai_protect.get("session_date") or "") == today_ist
                    else True
                )
                watchdog = _watchdog_health_status()
                alerts = _load_alerts_tail(limit=50)
                telegram_status = _load_telegram_alert_status()
                telegram_cmd_status = _load_telegram_command_status()
                advice = _json_read(BATMAN_BKM_TUNING_ADVICE_JSON)
                strategy_state = _json_read(STRATEGY_STATE)
                bkm_open_meta: Dict[str, Any] = {}
                try:
                    bkm_bucket = (strategy_state or {}).get("BATMAN_BKM") if isinstance(strategy_state, dict) else {}
                    if isinstance(bkm_bucket, dict):
                        for exp_key, info in bkm_bucket.items():
                            if isinstance(info, dict) and str(info.get("status") or "").upper() == "OPEN":
                                bkm_open_meta = dict(info)
                                bkm_open_meta.setdefault("expiry", exp_key)
                                break
                except Exception:
                    bkm_open_meta = {}
                proposal_count = 0
                if isinstance(advice, dict):
                    try:
                        proposal_count = len(advice.get("proposals") or [])
                    except Exception:
                        proposal_count = 0
                last_alert = alerts[-1] if alerts else {}
                v83_ops = _load_v83_ops_status()
                active_strategy = (
                    str(pid_data.get("strategy_file") or "").strip()
                    if isinstance(pid_data, dict)
                    else ""
                ) or str((_json_read(LAST_STRATEGY_FILE) or {}).get("strategy_file") or "batman_bkm_monthly").strip()
                v83_runtime_config = v83_ops.get("runtime_config") if isinstance(v83_ops, dict) else {}
                v83_runtime_state = v83_ops.get("runtime_state") if isinstance(v83_ops, dict) else {}
                v83_candidate = (
                    v83_runtime_state.get("last_candidate")
                    if isinstance(v83_runtime_state, dict) and isinstance(v83_runtime_state.get("last_candidate"), dict)
                    else {}
                )
                v83_candidate_meta = (
                    v83_candidate.get("metadata")
                    if isinstance(v83_candidate, dict) and isinstance(v83_candidate.get("metadata"), dict)
                    else {}
                )
                v83_trade_funnel = (
                    v83_candidate_meta.get("trade_funnel")
                    if isinstance(v83_candidate_meta.get("trade_funnel"), dict)
                    else {}
                )
                self._send_json(
                    {
                        "running": bool(running),
                        "pid": pid,
                        "trade_mode": (str(pid_data.get("trade_mode") or "").lower() if isinstance(pid_data, dict) else None),
                        "strategy_file": active_strategy,
                        "agent_execution_mode": execution_policy.get("mode"),
                        "execution_new_entries_allowed": bool(execution_policy.get("new_entries_allowed", False)),
                        "execution_manage_existing_allowed": bool(execution_policy.get("manage_existing_allowed", False)),
                        "paper_execution_allowed": bool(execution_policy.get("paper_execution_allowed", False)),
                        "live_execution_allowed": bool(execution_policy.get("live_execution_allowed", False)),
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
                        "bkm_ai_status": bkm_ai.get("status"),
                        "bkm_ai_mode": bkm_ai.get("mode"),
                        "bkm_ai_action": bkm_ai.get("action"),
                        "bkm_ai_severity": bkm_ai.get("severity"),
                        "bkm_ai_score": _parse_float(bkm_ai.get("score"), 0.0),
                        "bkm_ai_confidence": _parse_float(bkm_ai.get("confidence"), 0.0),
                        "bkm_ai_reasons": bkm_ai.get("reasons") if isinstance(bkm_ai.get("reasons"), list) else [],
                        "bkm_ai_basket_expiry": bkm_ai.get("basket_expiry"),
                        "bkm_ai_market_bias": bkm_ai.get("market_bias"),
                        "bkm_ai_position_risk_side": bkm_ai.get("position_risk_side"),
                        "bkm_learning_status": bkm_learning.get("status"),
                        "bkm_learning_total_outcomes": _parse_int(bkm_learning.get("total_outcomes"), 0),
                        "bkm_learning_realized_pnl_rs": _parse_float(bkm_learning.get("realized_pnl_rs"), 0.0),
                        "bkm_learning_last_outcome_at": bkm_learning.get("last_outcome_at"),
                        "bkm_learning_last_entry_assessment": (
                            bkm_learning.get("last_entry_assessment")
                            if isinstance(bkm_learning.get("last_entry_assessment"), dict)
                            else {}
                        ),
                        "bkm_ai_trend_confidence": _parse_float(
                            (bkm_ai_structure.get("trend_confidence") if isinstance(bkm_ai_structure, dict) else None),
                            None,
                        ),
                        "bkm_ai_signal_conflict_score": _parse_float(
                            (bkm_ai_structure.get("signal_conflict_score") if isinstance(bkm_ai_structure, dict) else None),
                            None,
                        ),
                        "bkm_ai_intraday_support": (
                            (bkm_ai_structure.get("intraday_support") or {}).get("level")
                            if isinstance(bkm_ai_structure.get("intraday_support"), dict)
                            else None
                        ),
                        "bkm_ai_intraday_resistance": (
                            (bkm_ai_structure.get("intraday_resistance") or {}).get("level")
                            if isinstance(bkm_ai_structure.get("intraday_resistance"), dict)
                            else None
                        ),
                        "bkm_ai_weekly_support": bkm_ai_structure.get("weekly_support") if isinstance(bkm_ai_structure, dict) else None,
                        "bkm_ai_weekly_resistance": bkm_ai_structure.get("weekly_resistance") if isinstance(bkm_ai_structure, dict) else None,
                        "bkm_ai_entry_lock_active": (bool(bkm_ai_plan.get("entry_lock_active", False)) and protect_enabled) or protect_active,
                        "bkm_ai_protection_enabled": protect_enabled,
                        "bkm_ai_protect_lock_active": protect_active,
                        "bkm_ai_protect_lock_reason": bkm_ai_protect.get("reason") if protect_active else None,
                        "bkm_ai_protect_lock_updated_at": bkm_ai_protect.get("updated_at") if protect_active else None,
                        "batman_entry_execution_enabled": bool(execution_policy.get("batman_new_entries_allowed", False)),
                        "batman_manage_existing_enabled": bool(execution_policy.get("batman_manage_existing_allowed", False)),
                        "bkm_ai_plan_next_course": bkm_ai_plan.get("next_course"),
                        "bkm_ai_updated_at": bkm_ai.get("updated_at"),
                        "intraday_ai_status": intraday_ai.get("status"),
                        "intraday_ai_mode": intraday_ai_mode,
                        "intraday_ai_execution_enabled": bool(intraday_ai_execution_enabled),
                        "intraday_manage_existing_enabled": bool(execution_policy.get("intraday_manage_existing_allowed", False)),
                        "intraday_ai_signal": intraday_ai.get("signal"),
                        "intraday_ai_priority": intraday_ai.get("priority"),
                        "intraday_ai_strategy": intraday_ai.get("strategy"),
                        "intraday_ai_market_bias": intraday_ai.get("market_bias"),
                        "intraday_ai_trend_confidence": _parse_float(intraday_ai.get("trend_confidence"), None),
                        "intraday_ai_signal_conflict_score": _parse_float(intraday_ai.get("signal_conflict_score"), None),
                        "intraday_ai_expiry": intraday_ai.get("expiry"),
                        "intraday_ai_updated_at": intraday_ai.get("updated_at"),
                        "intraday_ai_headline": intraday_ai_rec.get("headline"),
                        "intraday_ai_what_to_enter": intraday_ai_rec.get("what_to_enter"),
                        "intraday_ai_deploy_status": intraday_deploy.get("status"),
                        "intraday_ai_deploy_pending": bool(intraday_deploy.get("pending", False)),
                        "intraday_ai_deploy_result_code": intraday_deploy.get("result_code"),
                        "intraday_ai_deploy_message": intraday_deploy.get("message"),
                        "intraday_ai_deploy_updated_at": intraday_deploy.get("updated_at"),
                        "intraday_ai_position_status": intraday_position.get("status"),
                        "intraday_ai_position_id": intraday_position.get("position_id"),
                        "intraday_ai_position_strategy": intraday_position.get("strategy_label"),
                        "intraday_ai_position_pnl_rs": _parse_float(intraday_position.get("current_pnl_rs"), 0.0),
                        "intraday_ai_position_opened_at": intraday_position.get("opened_at"),
                        "intraday_ai_position_reason": intraday_position.get("last_reason"),
                        "intraday_ai_session_trade_count": _parse_int(intraday_position.get("session_trade_count"), 0),
                        "intraday_committee": committee,
                        "intraday_committee_status": committee.get("status"),
                        "intraday_committee_verdict": committee.get("verdict"),
                        "intraday_committee_bias": committee.get("consensus_bias"),
                        "intraday_committee_confidence": _parse_float(committee.get("ensemble_confidence"), None),
                        "intraday_committee_signal": committee.get("advisor_signal"),
                        "intraday_committee_strategy": committee.get("advisor_strategy"),
                        "intraday_committee_updated_at": committee.get("updated_at"),
                        "intraday_committee_reasons": committee.get("reasons") if isinstance(committee.get("reasons"), list) else [],
                        "intraday_learning": intraday_learning,
                        "intraday_learning_status": intraday_learning.get("status"),
                        "intraday_learning_total_outcomes": _parse_int(intraday_learning.get("total_outcomes"), 0),
                        "intraday_learning_realized_pnl_rs": _parse_float(intraday_learning.get("realized_pnl_rs"), 0.0),
                        "intraday_learning_last_outcome_at": intraday_learning.get("last_outcome_at"),
                        "intraday_learning_last_assessment": (
                            intraday_learning.get("last_assessment")
                            if isinstance(intraday_learning.get("last_assessment"), dict)
                            else {}
                        ),
                        "bkm_quality_status": bkm_open_meta.get("quality_status"),
                        "bkm_quality_score": _parse_float(bkm_open_meta.get("quality_score"), None),
                        "bkm_quality_reasons": (
                            bkm_open_meta.get("quality_reasons")
                            if isinstance(bkm_open_meta.get("quality_reasons"), list)
                            else []
                        ),
                        "bkm_open_expiry": bkm_open_meta.get("expiry"),
                        "telegram_commands_enabled": bool(telegram_cmd_status.get("enabled", False)),
                        "telegram_commands_configured": bool(telegram_cmd_status.get("configured", False)),
                        "telegram_commands_last_command": telegram_cmd_status.get("last_command"),
                        "telegram_commands_last_result": telegram_cmd_status.get("last_result"),
                        "telegram_commands_last_error": telegram_cmd_status.get("last_error"),
                        "telegram_commands_last_poll_at": telegram_cmd_status.get("last_poll_at"),
                        "alerts_recent_count": len(alerts),
                        "last_alert_severity": last_alert.get("severity"),
                        "last_alert_code": last_alert.get("code"),
                        "telegram_alerts_enabled": bool(telegram_status.get("enabled", False)),
                        "telegram_alerts_configured": bool(telegram_status.get("configured", False)),
                        "telegram_alerts_last_sent_at": telegram_status.get("last_sent_at"),
                        "telegram_alerts_last_error": telegram_status.get("last_error"),
                        "batman_bkm_tuning_proposals_pending": proposal_count,
                        "batman_bkm_tuning_last_generated_at": advice.get("generated_at") if isinstance(advice, dict) else None,
                        "intraday_defined_risk_ops": v83_ops,
                        "v83_execution_mode": (v83_runtime_config.get("mode") if isinstance(v83_runtime_config, dict) else None),
                        "v83_runtime_mode": (v83_runtime_config.get("mode") if isinstance(v83_runtime_config, dict) else None),
                        "v83_live_arm": bool((v83_runtime_config.get("live_arm") if isinstance(v83_runtime_config, dict) else False)),
                        "v83_health_status": ((v83_ops.get("health") or {}).get("status") if isinstance(v83_ops, dict) else None),
                        "v83_primary_block_reason": (
                            v83_runtime_state.get("primary_block_reason") if isinstance(v83_runtime_state, dict) else None
                        ),
                        "v83_market_state": v83_candidate_meta.get("market_state") or v83_trade_funnel.get("market_state"),
                        "v83_tradability_class": v83_candidate_meta.get("tradability_class") or v83_trade_funnel.get("tradability_class"),
                        "v83_candidate_playbook": v83_candidate_meta.get("playbook") or v83_trade_funnel.get("playbook"),
                        "v83_failure_type": v83_candidate_meta.get("failure_type") or v83_trade_funnel.get("failure_type"),
                    }
                )
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/bkm/ai_status"):
            try:
                self._send_json({"ok": True, "ai": _load_bkm_ai_status(), "protective_lock": _load_bkm_ai_protect_status()})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/intraday_ai/status"):
            try:
                self._send_json({"ok": True, "advisor": _load_intraday_ai_status()})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/intraday_ai/committee_review"):
            try:
                self._send_json({"ok": True, "review": _build_committee_review_payload(trade_mode="paper")})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/intraday_ai/committee"):
            try:
                self._send_json(
                    {
                        "ok": True,
                        "committee": _load_decision_committee_status(),
                        "history": _load_jsonl_tail(DECISION_COMMITTEE_HISTORY_JSONL, limit=25),
                    }
                )
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/intraday_ai/index_chart"):
            try:
                self._send_json({"ok": True, "chart": _load_intraday_index_chart_payload()})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/intraday_ai/learning"):
            try:
                self._send_json({"ok": True, "learning": _load_intraday_learning_status()})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/intraday_ai/deploy_status"):
            try:
                self._send_json({"ok": True, "deploy": _load_intraday_ai_deploy_status()})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/intraday_ai/position"):
            try:
                self._send_json({"ok": True, "position": _load_intraday_ai_position_status()})
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)
            return
        if self.path.startswith("/api/telegram/commands/status"):
            try:
                self._send_json({"ok": True, "telegram_commands": _load_telegram_command_status()})
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

        # ── Weekly IC deploy ──────────────────────────────────────────────
        if self.path.startswith("/api/weekly_ic/deploy"):
            try:
                from urllib.parse import urlparse, parse_qs
                qs = parse_qs(urlparse(self.path).query)
                token = (qs.get("token") or [""])[0].strip()
                pending: Dict[str, Any] = {}
                if WEEKLY_IC_PENDING_FILE.exists():
                    try:
                        pending = json.loads(WEEKLY_IC_PENDING_FILE.read_text())
                    except Exception:
                        pass

                if not pending:
                    self._send_html("<h2>No pending IC plan found.</h2><p>Run the weekly IC executor first.</p>", status=404)
                    return
                if pending.get("status") == "DEPLOYED":
                    self._send_html("<h2>Already deployed.</h2><p>This IC plan has already been executed.</p>")
                    return
                if pending.get("token") != token:
                    self._send_html("<h2>Invalid or expired token.</h2>", status=403)
                    return

                result = _deploy_weekly_ic(pending)

                # Update state
                pending["status"] = "DEPLOYED"
                pending["deployed_at"] = datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat()
                pending["deploy_result"] = result
                WEEKLY_IC_PENDING_FILE.write_text(json.dumps(pending, indent=2, default=str))
                WEEKLY_IC_POSITION_FILE.write_text(json.dumps({**pending, "status": "OPEN"}, indent=2, default=str))

                # Telegram notification
                legs_txt = "\n".join(
                    f'  {"✅" if r["ok"] else "❌"} {r["label"]}'
                    for r in result["legs"]
                )
                status_txt = "All 4 legs placed ✅" if result["ok"] else "⚠️ Some legs failed"
                _send_weekly_ic_telegram(
                    f"📊 <b>Weekly IC Deployed — {pending.get('expiry')}</b>\n\n"
                    f"{legs_txt}\n\n"
                    f"<b>{status_txt}</b>\n"
                    f"Net credit: ~Rs {pending.get('gross_credit', 0):,.0f}\n"
                    f"Max loss: Rs {pending.get('max_loss', 0):,.0f}\n"
                    f"Exit: Next Tuesday before 3:20 PM IST"
                )

                html = _weekly_ic_deploy_page(pending, result)
                self._send_html(html)
            except Exception as exc:
                self._send_html(f"<h2>Error</h2><pre>{exc}</pre>", status=500)
            return

        # ── Weekly IC close ───────────────────────────────────────────────
        if self.path.startswith("/api/weekly_ic/close"):
            try:
                from urllib.parse import urlparse, parse_qs
                qs = parse_qs(urlparse(self.path).query)
                token = (qs.get("token") or [""])[0].strip()

                position: Dict[str, Any] = {}
                for fpath in (WEEKLY_IC_POSITION_FILE, WEEKLY_IC_PENDING_FILE):
                    if fpath.exists():
                        try:
                            data = json.loads(fpath.read_text())
                            if data.get("status") in ("OPEN", "DEPLOYED"):
                                position = data
                                break
                        except Exception:
                            pass

                if not position:
                    self._send_html("<h2>No open IC position found.</h2>", status=404)
                    return
                if position.get("close_token") != token:
                    self._send_html("<h2>Invalid or expired close token.</h2>", status=403)
                    return

                result = _close_weekly_ic(position)

                # Update state
                position["status"] = "CLOSED"
                position["closed_at"] = datetime.now(timezone(timedelta(hours=5, minutes=30))).isoformat()
                position["close_result"] = result
                WEEKLY_IC_POSITION_FILE.write_text(json.dumps(position, indent=2, default=str))
                WEEKLY_IC_PENDING_FILE.write_text(json.dumps(position, indent=2, default=str))

                legs_txt = "\n".join(
                    f'  {"✅" if r["ok"] else "❌"} {r["label"]}'
                    for r in result["legs"]
                )
                status_txt = "All 4 legs closed ✅" if result["ok"] else "⚠️ Some close orders failed"
                _send_weekly_ic_telegram(
                    f"🔴 <b>Weekly IC Closed — {position.get('expiry')}</b>\n\n"
                    f"{legs_txt}\n\n"
                    f"<b>{status_txt}</b>"
                )

                html = _weekly_ic_close_page(position, result)
                self._send_html(html)
            except Exception as exc:
                self._send_html(f"<h2>Error</h2><pre>{exc}</pre>", status=500)
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
    httpd = ReusableHTTPServer(("", args.port), PaperHandler)
    print(f"Serving Paper P&L frontend on http://localhost:{args.port}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
