#!/usr/bin/env python3
"""
Refactored agent entrypoint.

Uses the new StrategySelector framework (regime-aware) instead of the legacy
weekly_theta_strangle loop. Placement always goes to whichever mode (live vs paper)
is selected via credentials; there is no warn-only mode.
"""
from __future__ import annotations

import atexit
import json
import csv
import logging
import os
import sys
import time
from datetime import datetime, date as dt_date, time as dtime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # Python 3.9+
    from zoneinfo import ZoneInfo  # type: ignore
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from logging.handlers import RotatingFileHandler

# ── Path/setup ────────────────────────────────────────────────────────────────
THIS_FILE = Path(__file__).resolve()
ENGINE_DIR = THIS_FILE.parent
DATA_ENGINE_DIR = ENGINE_DIR.parent
PROJECT_ROOT = DATA_ENGINE_DIR.parent
STATE_DIR = ENGINE_DIR / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
AGENT_LOG = STATE_DIR / "agent.log"
PID_FILE = STATE_DIR / "agent.pid"
FEATURE_LOG_PATH = STATE_DIR / "feature_history.csv"
INTEL_LOG_PATH = STATE_DIR / "intel_log.csv"
TRADE_BLOTTER_PATH = STATE_DIR / "trade_blotter.csv"
TRADE_BLOTTER_SUMMARY = STATE_DIR / "trade_blotter_summary.json"
BLOTTER_FIELDS = [
    "timestamp",
    "trade_mode",
    "warn_only",
    "executed",
    "side",
    "order_type",
    "exchange_seg",
    "product_type",
    "security_id",
    "quantity",
    "price",
    "strike",
    "tag",
    "notes",
]
FEATURE_FIELDS = [
    "timestamp",
    "strategy",
    "expiry",
    "exchange_seg",
    "spot",
    "ce_strike",
    "pe_strike",
    "ce_ltp",
    "pe_ltp",
    "ce_delta",
    "pe_delta",
    "ce_entry",
    "pe_entry",
    "net_credit",
    "warn_only",
    "trade_mode",
    "context",
    "block_reason",
    "net_delta",
    "total_notional",
    "positions_summary",
]
_LAST_ENV_LOG: Optional[datetime] = None
DEFAULT_SETTINGS: Dict[str, Any] = {
    "lot_size": 75,
    "nifty_lot_size": 75,
    "nifty_expiry_weekday": "Tuesday",
    "expiry_shift_if_holiday": True,
    "holiday_list": [],
    "smart_selector_enabled": True,
    "max_daily_trades": 5,
    "max_daily_loss_pct": 0.03,
    "partial_target_pct": 0.65,
    "per_leg_sl_mult": 1.6,
    "per_leg_tp_mult": 0.5,
    "vix_adaptive_low": 12.0,
    "vix_adaptive_high": 20.0,
    "strangle_delta_low": 0.15,
    "strangle_delta_high": 0.15,
    "strangle_offset_low": 150.0,
    "strangle_offset_high": 150.0,
    "spread_short_delta_low": 0.25,
    "spread_short_delta_high": 0.25,
    "gap_entry_threshold": 0.004,
    "iv_floor_percentile": 0.2,
}
CREDS_FILE = STATE_DIR / "creds.json"
INDEX_SECURITY_ID = int(os.getenv("MARKET_AI_INDEX_SECURITY_ID", "13"))
INDEX_EXCHANGE_SEG = os.getenv("MARKET_AI_INDEX_SEGMENT", "IDX_I")
INDEX_INSTRUMENT = os.getenv("MARKET_AI_INDEX_INSTRUMENT", "INDEX")
INTRADAY_INTERVAL_MIN = int(os.getenv("MARKET_AI_INTRADAY_INTERVAL", "5"))

_INTRADAY_CACHE: Dict[Tuple[str, int], Dict[str, Any]] = {}

for p in (ENGINE_DIR, DATA_ENGINE_DIR, PROJECT_ROOT, ENGINE_DIR / "strategies"):
    ps = str(p)
    if ps not in sys.path:
        sys.path.insert(0, ps)

from market_ai.dhan_wrapper import DhanWrapper  # noqa: E402
from market_ai.modules.data_fetch.dhan_scrip_cache import (  # noqa: E402
    resolve_option_security_id,
    refresh_scrip_master,
)
from market_ai.strategies import (  # noqa: E402
    StrategySelector,
    MarketSnapshot,
    OptionLeg,
    RiskConfig,
    StrategyType,
    OptionType,
    LegSide,
    TrendContext,
    TrendSide,
    ORBState,
    TradeAction,
)
from market_ai.strategies.trend_detector import detect_trend_from_open
from market_ai.strategies.orb_detector import ORBConfig, compute_orb_levels, detect_orb_breakout
from market_ai.strategies.sr_levels import compute_sr_levels
from market_ai.engine.feature_extractor import FeatureExtractor
from market_ai.engine.regime_scorer import RegimeScorer
from market_ai.engine.policy_engine import PolicyEngine
from market_ai.engine.learning_manager import LearningManager

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


def _load_saved_creds() -> Dict[str, str]:
    if not CREDS_FILE.exists():
        return {}
    try:
        data = json.loads(CREDS_FILE.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _ensure_dhan_credentials() -> None:
    saved = _load_saved_creds()
    cid_file = str(saved.get("client_id") or "").strip()
    tok_file = str(saved.get("access_token") or "").strip()
    cid_env = os.getenv("DHAN_CLIENT_ID", "").strip()
    tok_env = os.getenv("DHAN_ACCESS_TOKEN", "").strip()
    updated = False
    if cid_file and cid_file != cid_env:
        os.environ["DHAN_CLIENT_ID"] = cid_file
        updated = True
        log.info("DHAN_CLIENT_ID loaded from %s", CREDS_FILE)
    if tok_file and tok_file != tok_env:
        os.environ["DHAN_ACCESS_TOKEN"] = tok_file
        updated = True
        log.info("DHAN_ACCESS_TOKEN loaded from %s", CREDS_FILE)
    cid_final = os.getenv("DHAN_CLIENT_ID", "").strip()
    tok_final = os.getenv("DHAN_ACCESS_TOKEN", "").strip()
    if not cid_final or not tok_final:
        log.warning("DHAN credentials not found in env or %s; agent will fail to authenticate.", CREDS_FILE)
    elif not updated:
        log.info("DHAN credentials already present in environment.")


def _load_agent_settings() -> Dict[str, Any]:
    path = os.getenv("AGENT_SETTINGS_JSON")
    if not path:
        return dict(DEFAULT_SETTINGS)
    try:
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict):
            raise TypeError("settings json is not a dict")
        merged = dict(DEFAULT_SETTINGS)
        merged.update(data)
        # keep lot_size and nifty_lot_size in sync if only one is provided
        if "nifty_lot_size" not in merged:
            merged["nifty_lot_size"] = merged.get("lot_size", DEFAULT_SETTINGS["lot_size"])
        if "lot_size" not in merged:
            merged["lot_size"] = merged.get("nifty_lot_size", DEFAULT_SETTINGS["nifty_lot_size"])
        return merged
    except Exception as exc:
        log.warning("Failed to load agent settings from %s (%s); falling back to defaults", path, exc)
        return dict(DEFAULT_SETTINGS)


def _build_risk_config(settings: Dict[str, Any]) -> RiskConfig:
    partial = float(settings.get("partial_target_pct", DEFAULT_SETTINGS["partial_target_pct"]))
    partial = max(0.0, min(0.95, partial))
    per_leg_sl = float(settings.get("per_leg_sl_mult", DEFAULT_SETTINGS["per_leg_sl_mult"]))
    per_leg_tp = float(settings.get("per_leg_tp_mult", DEFAULT_SETTINGS["per_leg_tp_mult"]))
    per_leg_sl = max(0.0, per_leg_sl)
    per_leg_tp = max(0.0, per_leg_tp)
    vix_low = float(settings.get("vix_adaptive_low", DEFAULT_SETTINGS["vix_adaptive_low"]))
    vix_high = float(settings.get("vix_adaptive_high", DEFAULT_SETTINGS["vix_adaptive_high"]))
    strangle_delta_low = float(settings.get("strangle_delta_low", DEFAULT_SETTINGS["strangle_delta_low"]))
    strangle_delta_high = float(settings.get("strangle_delta_high", DEFAULT_SETTINGS["strangle_delta_high"]))
    strangle_offset_low = float(settings.get("strangle_offset_low", DEFAULT_SETTINGS["strangle_offset_low"]))
    strangle_offset_high = float(settings.get("strangle_offset_high", DEFAULT_SETTINGS["strangle_offset_high"]))
    spread_delta_low = float(settings.get("spread_short_delta_low", DEFAULT_SETTINGS["spread_short_delta_low"]))
    spread_delta_high = float(settings.get("spread_short_delta_high", DEFAULT_SETTINGS["spread_short_delta_high"]))
    return RiskConfig(
        max_intraday_loss=float(settings.get("max_intraday_loss", -3000)),
        intraday_target=float(settings.get("intraday_target", 4000)),
        allow_carry_forward=bool(settings.get("allow_carry_forward", False)),
        max_carry_days=int(settings.get("max_carry_days", 0)),
        vix_carry_threshold=float(settings.get("vix_carry_threshold", 12.0)),
        last_entry_time=dtime(14, 45),
        force_exit_time=dtime(15, 15),
        max_daily_loss_pct=float(settings.get("max_daily_loss_pct", DEFAULT_SETTINGS["max_daily_loss_pct"])),
        max_daily_trades=int(settings.get("max_daily_trades", DEFAULT_SETTINGS["max_daily_trades"])),
        per_leg_sl_mult=per_leg_sl,
        per_leg_tp_mult=per_leg_tp,
        partial_target_pct=partial,
        vix_adaptive_low=vix_low,
        vix_adaptive_high=vix_high,
        strangle_delta_low=strangle_delta_low,
        strangle_delta_high=strangle_delta_high,
        strangle_offset_low=strangle_offset_low,
        strangle_offset_high=strangle_offset_high,
        spread_short_delta_low=spread_delta_low,
        spread_short_delta_high=spread_delta_high,
    )


def _normalize_leg_side(side) -> LegSide:
    if isinstance(side, LegSide):
        return side
    if isinstance(side, str):
        return LegSide.BUY if side.upper().startswith("B") else LegSide.SELL
    return LegSide.BUY


def _ensure_leg_security_id(leg: OptionLeg, symbol: str, expiry) -> Optional[int]:
    if leg.security_id:
        try:
            return int(float(leg.security_id))
        except Exception:
            return None
    resolved = _resolve_security_id_cached(symbol, expiry.isoformat(), leg.strike, leg.option_type.value)
    if resolved:
        try:
            return int(float(resolved))
        except Exception:
            return None
    return None


def _write_pid_file() -> None:
    data: Dict[str, Any] = {}
    if PID_FILE.exists():
        try:
            data = json.loads(PID_FILE.read_text()) or {}
        except Exception:
            data = {}
    data.update(
        {
            "pid": os.getpid(),
            "trade_mode": os.getenv("TRADE_MODE", "live"),
            "started_at": datetime.now().isoformat(),
        }
    )
    try:
        PID_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        log.warning("Failed to write pid file %s", PID_FILE)


def _cleanup_pid_file() -> None:
    try:
        if PID_FILE.exists():
            data = json.loads(PID_FILE.read_text())
            if data.get("pid") == os.getpid():
                PID_FILE.unlink()
    except Exception:
        pass


atexit.register(_cleanup_pid_file)


def _ensure_csv_headers(path: Path, headers: List[str]) -> None:
    if path.exists():
        return
    try:
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            writer.writeheader()
    except Exception as exc:
        log.warning("Failed to initialize CSV %s: %s", path, exc)


def _append_csv_row(path: Path, headers: List[str], row: Dict[str, Any]) -> None:
    _ensure_csv_headers(path, headers)
    try:
        with path.open("a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            writer.writerow(row)
    except Exception as exc:
        log.warning("Failed to append row to %s: %s", path, exc)


def _update_blotter_summary(entry: Dict[str, Any]) -> None:
    summary: Dict[str, Any] = {"warn_only_orders": 0, "executed_orders": 0}
    if TRADE_BLOTTER_SUMMARY.exists():
        try:
            summary = json.loads(TRADE_BLOTTER_SUMMARY.read_text()) or summary
        except Exception:
            summary = {"warn_only_orders": 0, "executed_orders": 0}
    summary["executed_orders"] = int(summary.get("executed_orders", 0)) + int(entry.get("executed", 0))
    summary["warn_only_orders"] = 0
    summary["last_order_ts"] = entry.get("timestamp")
    try:
        TRADE_BLOTTER_SUMMARY.write_text(json.dumps(summary, indent=2))
    except Exception as exc:
        log.warning("Failed to update blotter summary: %s", exc)


def _log_trade_event(*, trade_mode: str, side: str, leg: OptionLeg, order_type: str, notes: str = "") -> None:
    entry = {
        "timestamp": datetime.now().isoformat(),
        "trade_mode": trade_mode,
        "warn_only": 0,
        "executed": 1,
        "side": side,
        "order_type": order_type,
        "exchange_seg": "NSE_FNO",
        "product_type": "MARGIN",
        "security_id": _ensure_leg_security_id(leg, leg.symbol, leg.expiry) or "",
        "quantity": leg.quantity,
        "price": float(getattr(leg, "entry_price", 0.0) or leg.current_ltp or 0.0),
        "strike": leg.strike,
        "tag": getattr(leg, "strategy_type", StrategyType.NONE).name,
        "notes": notes,
    }
    _append_csv_row(TRADE_BLOTTER_PATH, BLOTTER_FIELDS, entry)
    _update_blotter_summary(entry)


def _log_feature_event(row: Dict[str, Any]) -> None:
    _append_csv_row(FEATURE_LOG_PATH, FEATURE_FIELDS, row)


def _log_environment_state(market: MarketSnapshot, trend_ctx: TrendContext, trade_mode: str) -> None:
    global _LAST_ENV_LOG
    now = datetime.now()
    if _LAST_ENV_LOG and (now - _LAST_ENV_LOG).total_seconds() < 60:
        return
    _LAST_ENV_LOG = now
    context = {
        "market_state": {
            "trend": trend_ctx.trend_side.name.lower(),
            "bull_score": trend_ctx.bull_score,
            "bear_score": trend_ctx.bear_score,
            "confidence": trend_ctx.confidence,
            "spot": market.spot,
            "vix": market.india_vix,
            "as_of": market.now.isoformat(),
        },
        "strategy_candidates": [],
    }
    row = {
        "timestamp": now.isoformat(),
        "strategy": "environment",
        "expiry": "",
        "exchange_seg": "NSE_FNO",
        "spot": market.spot,
        "ce_strike": "",
        "pe_strike": "",
        "ce_ltp": "",
        "pe_ltp": "",
        "ce_delta": "",
        "pe_delta": "",
        "ce_entry": "",
        "pe_entry": "",
        "net_credit": "",
        "warn_only": 0,
        "trade_mode": trade_mode,
        "context": json.dumps(context),
        "block_reason": "",
        "net_delta": "",
        "total_notional": "",
        "positions_summary": "",
    }
    _log_feature_event(row)


def _log_strategy_event(
    *,
    strategy: StrategyType,
    action: str,
    reason: str,
    market: MarketSnapshot,
    trade_mode: str,
    legs: List[OptionLeg],
) -> None:
    ce_leg = next((leg for leg in legs if leg.side == LegSide.SELL and leg.option_type == OptionType.CALL), None)
    pe_leg = next((leg for leg in legs if leg.side == LegSide.SELL and leg.option_type == OptionType.PUT), None)
    context = {"action": action, "reason": reason}
    row = {
        "timestamp": datetime.now().isoformat(),
        "strategy": strategy.name.lower(),
        "expiry": legs[0].expiry.isoformat() if legs else "",
        "exchange_seg": "NSE_FNO",
        "spot": market.spot,
        "ce_strike": ce_leg.strike if ce_leg else "",
        "pe_strike": pe_leg.strike if pe_leg else "",
        "ce_ltp": ce_leg.current_ltp if ce_leg else "",
        "pe_ltp": pe_leg.current_ltp if pe_leg else "",
        "ce_delta": "",
        "pe_delta": "",
        "ce_entry": ce_leg.entry_price if ce_leg else "",
        "pe_entry": pe_leg.entry_price if pe_leg else "",
        "net_credit": (
            (ce_leg.entry_price if ce_leg else 0.0) + (pe_leg.entry_price if pe_leg else 0.0)
            if action.startswith("OPEN")
            else ""
        ),
        "warn_only": 0,
        "trade_mode": trade_mode,
        "context": json.dumps(context),
        "block_reason": "",
        "net_delta": "",
        "total_notional": "",
        "positions_summary": "",
    }
    _log_feature_event(row)


_SECURITY_ID_CACHE: Dict[tuple[str, str, float, str], Optional[str]] = {}
_SECURITY_ID_MISS_LOGGED: set[tuple[str, str, float, str]] = set()
_SCRIP_MASTER_WARMED = False


def _warm_scrip_master(force: bool = False) -> None:
    global _SCRIP_MASTER_WARMED
    if _SCRIP_MASTER_WARMED and not force:
        return
    try:
        refresh_scrip_master(force=force)
        _SCRIP_MASTER_WARMED = True
    except Exception as exc:  # pragma: no cover - best effort
        log.warning("Failed to refresh Dhan scrip master (force=%s): %s", force, exc)


def _resolve_security_id_cached(symbol: str, expiry: str, strike: float, option_type: str) -> Optional[str]:
    strike_norm = float(strike)
    key = (symbol.upper(), expiry, strike_norm, option_type.upper())
    if key in _SECURITY_ID_CACHE:
        return _SECURITY_ID_CACHE[key]
    sec_id: Optional[str] = None
    try:
        sec_val = resolve_option_security_id(symbol, expiry, strike_norm, option_type)
        if sec_val:
            sec_id = str(sec_val)
        elif not _SCRIP_MASTER_WARMED:
            _warm_scrip_master(force=True)
            sec_val = resolve_option_security_id(symbol, expiry, strike_norm, option_type)
            if sec_val:
                sec_id = str(sec_val)
    except Exception as exc:  # pragma: no cover - defensive log
        log.warning(
            "SecurityId lookup failed symbol=%s expiry=%s strike=%s opt=%s err=%s",
            symbol,
            expiry,
            strike_norm,
            option_type,
            exc,
        )
    if not sec_id and key not in _SECURITY_ID_MISS_LOGGED:
        _SECURITY_ID_MISS_LOGGED.add(key)
        log.warning(
            "Missing security_id for symbol=%s expiry=%s strike=%s opt=%s",
            symbol,
            expiry,
            strike_norm,
            option_type,
        )
    _SECURITY_ID_CACHE[key] = sec_id
    return sec_id

# ── Helpers ─────────────────────────────────────────────────────────────────
def _as_bool(val: Optional[str], default: bool) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def _ist_now() -> datetime:
    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    return datetime.now(tz) if tz else datetime.now()


def _is_india_market_open(now: Optional[datetime] = None) -> bool:
    current = now or _ist_now()
    if current.weekday() >= 5:
        return False
    start = dtime(9, 15)
    end = dtime(15, 30)
    return start <= current.time() <= end


def _looks_like_chain_dict(obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    for val in obj.values():
        if isinstance(val, dict):
            lower_keys = {str(k).lower() for k in val.keys()}
            if {"ce", "pe"} & lower_keys:
                return True
    return False


def _rows_to_chain(rows: Any) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not isinstance(rows, list):
        return out
    for row in rows:
        if not isinstance(row, dict):
            continue
        strike = (
            row.get("strike")
            or row.get("strikePrice")
            or row.get("strikeprice")
            or row.get("StrikePrice")
        )
        if strike is None:
            continue
        try:
            strike_val = float(strike)
        except Exception:
            continue
        strike_key = str(int(strike_val)) if abs(strike_val - int(strike_val)) < 1e-6 else str(strike_val)
        ce = (
            row.get("ce")
            or row.get("CE")
            or row.get("call")
            or row.get("Call")
            or row.get("callOption")
        )
        pe = (
            row.get("pe")
            or row.get("PE")
            or row.get("put")
            or row.get("Put")
            or row.get("putOption")
        )
        entry: Dict[str, Any] = {}
        if ce:
            entry["ce"] = ce
        if pe:
            entry["pe"] = pe
        if entry:
            out[strike_key] = entry
    return out


def _coerce_chain_dict(chain_raw: Any) -> Dict[str, Dict[str, Any]]:
    """
    DHAN responses often nest the actual strike→legs dictionary under several keys.
    Walk those wrappers and return the first dict that looks like an option chain.
    """
    queue = [chain_raw]
    seen: set[int] = set()
    while queue:
        cur = queue.pop(0)
        if isinstance(cur, dict):
            oid = id(cur)
            if oid in seen:
                continue
            seen.add(oid)
            if _looks_like_chain_dict(cur):
                return cur  # already strike→legs
            for key in ("data", "Data", "DATA", "oc", "option_chain", "optionchain", "records", "chain"):
                if key in cur:
                    queue.append(cur[key])
            continue
        if isinstance(cur, list):
            converted = _rows_to_chain(cur)
            if converted:
                return converted
    return {}


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
            side = LegSide.SELL if net < 0 else LegSide.BUY
            symbol = r.get("tradingSymbol") or r.get("symbol") or ""

            def _parse_expiry_from_symbol(sym: str) -> Optional[dt_date]:
                if not sym:
                    return None
                sym = sym.upper().replace("-", " ")
                parts = sym.split()
                months = {
                    "JAN": 1,
                    "FEB": 2,
                    "MAR": 3,
                    "APR": 4,
                    "MAY": 5,
                    "JUN": 6,
                    "JUL": 7,
                    "AUG": 8,
                    "SEP": 9,
                    "OCT": 10,
                    "NOV": 11,
                    "DEC": 12,
                }
                for idx in range(len(parts) - 2):
                    if parts[idx].isdigit() and parts[idx + 1] in months:
                        day = int(parts[idx])
                        month = months[parts[idx + 1]]
                        year = datetime.now().year
                        try:
                            return dt_date(year, month, day)
                        except Exception:
                            continue
                return None

            expiry = (
                r.get("expiryDate")
                or r.get("expiry")
                or r.get("drvExpiryDate")
                or r.get("drvExpiry")
                or r.get("expirydate")
                or _parse_expiry_from_symbol(symbol)
            )
            expiry_date = datetime.fromisoformat(str(expiry)).date() if expiry else datetime.now().date()

            def _parse_strike_from_symbol(sym: str) -> Optional[float]:
                if not sym:
                    return None
                import re

                matches = re.findall(r"(\d{4,6}(?:\.\d+)?)", sym)
                if not matches:
                    return None
                try:
                    return float(matches[-1])
                except Exception:
                    return None

            strike = (
                r.get("strikePrice")
                or r.get("strike")
                or r.get("drvStrikePrice")
                or r.get("drvStrike")
                or _parse_strike_from_symbol(symbol)
                or 0
            )
            opt_type_raw = (
                r.get("optionType")
                or r.get("option_type")
                or r.get("drvOptionType")
                or r.get("drv_option_type")
                or symbol
                or "C"
            )

            def _coerce_option_type(val: str) -> OptionType:
                val = (val or "").upper()
                if "P" in val:
                    return OptionType.PUT
                return OptionType.CALL

            option_type = _coerce_option_type(opt_type_raw)

            legs.append(
                OptionLeg(
                    symbol="NIFTY",
                    expiry=expiry_date,
                    strike=float(strike),
                    option_type=option_type,
                    side=side,
                    quantity=qty,
                    entry_price=float(r.get("avgPrice") or r.get("avg_price") or 0.0),
                    security_id=str(r.get("securityId") or r.get("security_id") or ""),
                    strategy_type=StrategyType.NONE,
                )
            )
        except Exception:
            continue
    return legs


def _map_chain(chain_raw: dict, expiry, symbol: str) -> List[dict]:
    rows: List[dict] = []
    chain_dict = _coerce_chain_dict(chain_raw)
    if not isinstance(chain_dict, dict):
        return rows
    expiry_str = expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry)
    for strike, legs in chain_dict.items():
        if not isinstance(legs, dict):
            continue
        for opt_name, opt_data in (legs or {}).items():
            if not isinstance(opt_data, dict):
                continue
            option_type = "CE" if opt_name.lower().startswith("c") else "PE"
            sec_id = opt_data.get("securityId") or opt_data.get("security_id")
            if not sec_id:
                resolved = _resolve_security_id_cached(symbol, expiry_str, float(strike), option_type)
                if resolved:
                    sec_id = resolved
            rows.append(
                {
                    "expiry": expiry,
                    "option_type": option_type,
                    "strike": float(strike),
                    "ltp": (
                        opt_data.get("last_price")
                        or opt_data.get("LastPrice")
                        or opt_data.get("ltp")
                        or opt_data.get("close")
                    ),
                    "delta": opt_data.get("delta"),
                    "security_id": sec_id,
                }
            )
    return rows


def _update_leg_ltps_from_chain(legs: List[OptionLeg], chain_rows: List[dict]) -> None:
    lookup: Dict[Tuple[str, float], float] = {}
    for row in chain_rows or []:
        try:
            opt = str(row.get("option_type") or "").upper()
            strike = float(row.get("strike") or 0.0)
            ltp = row.get("ltp")
            if not opt or strike == 0 or ltp is None:
                continue
            lookup[(opt, strike)] = float(ltp)
        except Exception:
            continue
    for leg in legs or []:
        try:
            opt_name = leg.option_type.value if hasattr(leg.option_type, "value") else str(leg.option_type)
            opt_name = opt_name.upper()
            if "CALL" in opt_name and opt_name != "CE":
                opt_name = "CE"
            elif "PUT" in opt_name and opt_name != "PE":
                opt_name = "PE"
            strike = float(leg.strike)
            ltp = lookup.get((opt_name, strike))
            if ltp is not None:
                leg.current_ltp = ltp
        except Exception:
            continue


def _compute_mtm(legs: List[OptionLeg]) -> float:
    mtm = 0.0
    for leg in legs:
        if leg.current_ltp is None:
            continue
        side = _normalize_leg_side(leg.side)
        pnl = (leg.entry_price - leg.current_ltp) if side == LegSide.SELL else (leg.current_ltp - leg.entry_price)
        mtm += pnl * leg.quantity
    return mtm


def _fetch_expiry(dw: DhanWrapper):
    return _fetch_expiry_with_settings(dw, DEFAULT_SETTINGS)


def _weekday_to_int(value) -> Optional[int]:
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


def _parse_holiday_list(settings: Dict[str, Any]) -> set:
    holidays_raw = settings.get("holiday_list") or []
    # support comma separated string for backward compatibility
    if isinstance(holidays_raw, str):
        holidays_raw = [h.strip() for h in holidays_raw.replace(",", "\n").splitlines() if h.strip()]
    out = set()
    for item in holidays_raw:
        try:
            clean = str(item).split("T", 1)[0].split(" ", 1)[0]
            out.add(datetime.fromisoformat(clean).date())
        except Exception:
            continue
    return out


def _derive_shifted_holidays(target_weekday: Optional[int], expiries: List[dt_date]) -> set:
    """
    If the exchange has shifted expiry (e.g., no Tuesday but a Wednesday expiry),
    infer the missing target weekday immediately before the shifted expiry as a holiday.
    """
    if target_weekday is None or not expiries:
        return set()
    today = datetime.now().date()
    future = [e for e in expiries if e >= today]
    if not future:
        return set()
    future.sort()
    first_exp = future[0]
    if first_exp.weekday() == target_weekday:
        return set()
    # compute the most recent target_weekday before the shifted expiry
    delta_days = (first_exp.weekday() - target_weekday) % 7
    inferred_holiday = first_exp - timedelta(days=delta_days or 7)
    if inferred_holiday >= today - timedelta(days=7):
        return {inferred_holiday}
    return set()


def _fetch_expiry_with_settings(dw: DhanWrapper, settings: Dict[str, Any]):
    try:
        expiries = dw.get_optionchain_expirylist("IDX_I", 13)
    except AttributeError:
        try:
            expiries = dw.get_expiry_list(13, "IDX_I")
        except Exception:
            expiries = []
    except Exception:
        expiries = []

    # Normalize into a list of ISO strings if we received a dict
    expiry_list: List[str] = []
    if isinstance(expiries, list):
        expiry_list = [str(e) for e in expiries]
    elif isinstance(expiries, dict):
        for key in ("Expiry", "expiry", "expiries", "data"):
            val = expiries.get(key)
            if isinstance(val, list):
                expiry_list = [str(e) for e in val]
                break

    today = datetime.now().date()
    parsed: List[dt_date] = []
    for raw in expiry_list:
        clean = raw.strip()
        if "T" in clean:
            clean = clean.split("T", 1)[0]
        if " " in clean:
            clean = clean.split(" ", 1)[0]
        try:
            parsed.append(datetime.fromisoformat(clean).date())
        except Exception:
            continue
    parsed.sort()
    if not parsed:
        return today

    target_weekday = _weekday_to_int(settings.get("nifty_expiry_weekday", "Tuesday"))
    holidays = _parse_holiday_list(settings)
    shift_if_holiday = bool(settings.get("expiry_shift_if_holiday", True))
    holidays |= _derive_shifted_holidays(target_weekday, parsed)

    def is_valid(cand: dt_date) -> bool:
        if cand < today:
            return False
        if target_weekday is not None and cand.weekday() != target_weekday:
            return False
        if shift_if_holiday and cand in holidays:
            return False
        return True

    filtered = [c for c in parsed if is_valid(c)]
    if filtered:
        return filtered[0]

    # fallback: ignore weekday filter but still respect holiday shift
    filtered_no_weekday = [c for c in parsed if c >= today and (not shift_if_holiday or c not in holidays)]
    if filtered_no_weekday:
        return filtered_no_weekday[0]

    # ultimate fallback: last known expiry
    return parsed[-1]


def _place_leg_order(dw: DhanWrapper, leg: OptionLeg, *, close: bool, trade_mode: str) -> None:
    side = _normalize_leg_side(leg.side)
    if close:
        order_side = LegSide.BUY if side == LegSide.SELL else LegSide.SELL
    else:
        order_side = side
    security_id = _ensure_leg_security_id(leg, leg.symbol, leg.expiry)
    if not security_id:
        log.error("Skipping leg order due to missing security_id (strike=%s opt=%s)", leg.strike, leg.option_type)
        return
    log.info(
        "Placing %s order sec_id=%s strike=%s opt=%s qty=%s expiry=%s",
        order_side.value,
        security_id,
        leg.strike,
        leg.option_type.value if hasattr(leg.option_type, "value") else str(leg.option_type),
        leg.quantity,
        leg.expiry,
    )
    dw.place_order(
        side=order_side.value,
        exchange_seg="NSE_FNO",
        security_id=security_id,
        quantity=leg.quantity,
        product_type="MARGIN",
        order_type="MARKET",
    )
    _log_trade_event(trade_mode=trade_mode, side=order_side.value, leg=leg, order_type="MARKET")


def _cached_intraday_for_day(dw: DhanWrapper, trade_day: dt_date, interval: int = INTRADAY_INTERVAL_MIN) -> List[dict]:
    key = (trade_day.isoformat(), interval)
    now = datetime.now()
    entry = _INTRADAY_CACHE.get(key)
    ttl = 60 if trade_day == now.date() else 3600
    if entry:
        age = (now - entry["ts"]).total_seconds()
        if age < ttl:
            return entry["data"]
    candles = dw.get_intraday_candles(
        INDEX_SECURITY_ID,
        INDEX_EXCHANGE_SEG,
        INDEX_INSTRUMENT,
        interval=interval,
        from_date=trade_day.isoformat(),
        to_date=trade_day.isoformat(),
    )
    if candles:
        _INTRADAY_CACHE[key] = {"ts": now, "data": candles}
    elif trade_day < now.date():
        # cache empty for past non-trading days to avoid repeated calls
        _INTRADAY_CACHE[key] = {"ts": now, "data": []}
    return candles


def _simplify_candles(raw: List[dict], limit: int = 2) -> List[dict]:
    simplified: List[dict] = []
    for candle in raw[:limit]:
        simplified.append(
            {
                "open": float(candle.get("open") or 0.0),
                "high": float(candle.get("high") or candle.get("open") or 0.0),
                "low": float(candle.get("low") or candle.get("open") or 0.0),
                "close": float(candle.get("close") or candle.get("open") or 0.0),
                "volume": float(candle.get("volume") or 0.0),
            }
        )
    return simplified


def _calc_vwap_from_candles(candles: List[dict], fallback: float) -> float:
    total_volume = 0.0
    pv_sum = 0.0
    for candle in candles:
        vol = float(candle.get("volume") or 0.0)
        if vol <= 0:
            continue
        typical = (float(candle.get("open") or 0.0) + float(candle.get("high") or 0.0) + float(candle.get("low") or 0.0) + float(candle.get("close") or 0.0)) / 4.0
        total_volume += vol
        pv_sum += typical * vol
    if total_volume > 0:
        return pv_sum / total_volume
    closes = [float(candle.get("close") or 0.0) for candle in candles if candle.get("close") is not None]
    return closes[-1] if closes else fallback


def _average_volume(candles: List[dict]) -> float:
    vols = [float(candle.get("volume") or 0.0) for candle in candles if candle.get("volume") not in (None, "")]
    if not vols:
        return 0.0
    return sum(vols) / len(vols)


def _prev_day_levels(candles: List[dict], fallback: float) -> Tuple[float, float, float]:
    highs = [float(candle.get("high") or 0.0) for candle in candles if candle.get("high") is not None]
    lows = [float(candle.get("low") or 0.0) for candle in candles if candle.get("low") is not None]
    close = fallback
    for candle in reversed(candles):
        if candle.get("close") is not None:
            close = float(candle.get("close") or 0.0)
            break
    high = max(highs) if highs else fallback
    low = min(lows) if lows else fallback
    return high, low, close


def build_market_snapshot(dw: DhanWrapper, *, avg_volume_hint: float = 0.0) -> Tuple[MarketSnapshot, float, float, float]:
    now = datetime.now()
    spot = float(dw.get_ltp_once(INDEX_EXCHANGE_SEG, INDEX_SECURITY_ID) or 0.0)
    trade_day = now.date()
    todays_candles = _cached_intraday_for_day(dw, trade_day, INTRADAY_INTERVAL_MIN)
    if not spot and todays_candles:
        spot = float(todays_candles[-1].get("close") or 0.0)

    prev_candles: List[dict] = []
    probe = trade_day - timedelta(days=1)
    for _ in range(5):
        if probe < dt_date(2000, 1, 1):
            break
        prev = _cached_intraday_for_day(dw, probe, INTRADAY_INTERVAL_MIN)
        if prev:
            prev_candles = prev
            break
        probe -= timedelta(days=1)

    yesterday_high, yesterday_low, yesterday_close = _prev_day_levels(prev_candles, spot or 0.0)
    avg_volume = _average_volume(prev_candles)
    if avg_volume <= 0 and avg_volume_hint:
        avg_volume = float(avg_volume_hint)
    vwap = _calc_vwap_from_candles(todays_candles, spot or 0.0)
    pivot = (yesterday_high + yesterday_low + yesterday_close) / 3.0 if prev_candles else (spot or 0.0)

    snapshot = MarketSnapshot(
        symbol="NIFTY",
        spot=spot,
        candles_5m=_simplify_candles(todays_candles, limit=2),
        yesterday_high=yesterday_high,
        yesterday_low=yesterday_low,
        yesterday_close=yesterday_close,
        india_vix=0.0,
        now=now,
    )
    return snapshot, float(avg_volume), float(vwap), float(pivot)


def main() -> None:
    poll_sec = float(os.getenv("POLL_SEC", "10"))
    _warm_scrip_master(force=False)
    _ensure_dhan_credentials()
    settings = _load_agent_settings()
    dw = DhanWrapper(logger=logging.getLogger("dhan_wrapper"))
    _write_pid_file()
    lot_size = int(settings.get("lot_size", DEFAULT_SETTINGS["lot_size"]))
    selector = StrategySelector(symbol="NIFTY", lot_size=lot_size)
    risk = _build_risk_config(settings)
    smart_enabled = bool(settings.get("smart_selector_enabled", True))
    avg_volume_hint = float(settings.get("avg_5m_volume", 0.0))
    trade_mode = os.getenv("TRADE_MODE", "live")
    _ensure_csv_headers(TRADE_BLOTTER_PATH, BLOTTER_FIELDS)
    _ensure_csv_headers(INTEL_LOG_PATH, ["timestamp", "prob_bull", "prob_bear", "prob_sideways", "regime", "strategy", "size_multiplier", "notes", "gap_pct", "vix", "vwap_distance"])
    _ensure_csv_headers(FEATURE_LOG_PATH, FEATURE_FIELDS)
    log.info("Regime-aware agent started (mode=%s)", trade_mode)
    daily_trades = 0
    current_day = datetime.now().date()
    day_entries = 0
    day_legs = 0
    day_mode = "NORMAL"
    orb_config = ORBConfig()
    orb_state = ORBState()
    basket_peak_mtm = 0.0
    trail_active = False
    trail_floor = -1000.0
    day_pnl = 0.0
    feat_extractor = FeatureExtractor()
    regime_scorer = RegimeScorer()
    learning = LearningManager()
    policy = PolicyEngine(learning_manager=learning)

    while True:
        try:
            today = datetime.now().date()
            if today != current_day:
                current_day = today
                daily_trades = 0
                day_entries = 0
                day_legs = 0
                day_mode = "NORMAL"
                orb_state = ORBState()
                basket_peak_mtm = 0.0
                trail_active = False
                trail_floor = -1000.0
                day_pnl = 0.0
            market, avg_volume, vwap, pivot = build_market_snapshot(dw, avg_volume_hint=avg_volume_hint)
            market_open = datetime.combine(market.now.date(), dtime(9, 15))
            if orb_config.enabled and orb_state.orb_levels is None:
                levels = compute_orb_levels(market.candles_5m, market_open, orb_config)
                if levels:
                    orb_state.orb_levels = levels
            if orb_state.orb_levels:
                breakout = detect_orb_breakout(market, orb_state.orb_levels, orb_config)
                orb_state.breakout_signal = breakout if breakout.active else None
            sr_levels = compute_sr_levels(market, orb_state.orb_levels)
            expiry = _fetch_expiry_with_settings(dw, settings)
            positions_raw = dw.get_positions_raw()
            legs = _map_positions(positions_raw)
            if legs:
                log.info(
                    "[Positions detail] %s",
                    [
                        (
                            leg.side.value if hasattr(leg.side, "value") else str(leg.side),
                            leg.option_type.value if hasattr(leg.option_type, "value") else str(leg.option_type),
                            leg.strike,
                            leg.expiry,
                            leg.quantity,
                        )
                        for leg in legs
                    ],
            )
            expiry_str = expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry)
            chain_raw = dw.get_option_chain(INDEX_SECURITY_ID, INDEX_EXCHANGE_SEG, expiry_str)
            chain = _map_chain(chain_raw, expiry, symbol="NIFTY")
            _update_leg_ltps_from_chain(legs, chain)
            basket_mtm = _compute_mtm(legs)
            day_pnl = basket_mtm  # single-basket approximation for open PnL; realized adds into MTM over time
            if not legs:
                basket_peak_mtm = 0.0
                trail_active = False
                trail_floor = -1000.0
            # Daily hard stop: lock the day if PnL breaches daily_max_loss
            forced_decision: Optional[TradeAction] = None
            if day_mode != "LOCKED_RED" and day_pnl <= risk.daily_max_loss:
                forced_decision = TradeAction(
                    action_type="CLOSE_ALL",
                    strategy_type=StrategyType.NONE,
                    legs_to_open=[],
                    legs_to_close=legs,
                    reason="DAILY_LOCK_RED",
                )
                day_mode = "LOCKED_RED"
            if day_mode != "LOCKED_RED" and basket_mtm <= risk.daily_max_loss:
                day_mode = "LOCKED_RED"
            elif day_mode == "NORMAL" and basket_mtm >= risk.daily_target:
                day_mode = "LOCKED_GREEN"
            if legs:
                if basket_mtm > basket_peak_mtm:
                    basket_peak_mtm = basket_mtm
                # Hard per-basket SL always enforced
                if basket_mtm <= -1000.0 and forced_decision is None:
                    forced_decision = TradeAction(
                        action_type="CLOSE_ALL",
                        strategy_type=StrategyType.NONE,
                        legs_to_open=[],
                        legs_to_close=legs,
                        reason="BASKET_SL_HIT",
                    )
                if not trail_active and forced_decision is None:
                    if basket_mtm >= 2500.0:
                        trail_active = True
                        trail_floor = max(1500.0, basket_mtm - 1000.0)
                elif trail_active and forced_decision is None:
                    desired_floor = max(1500.0, basket_peak_mtm * 0.6)
                    if desired_floor > trail_floor:
                        trail_floor = desired_floor
                    if basket_mtm <= trail_floor:
                        forced_decision = TradeAction(
                            action_type="CLOSE_ALL",
                            strategy_type=StrategyType.NONE,
                            legs_to_open=[],
                            legs_to_close=legs,
                            reason="BASKET_TRAIL_HIT",
                        )
            if smart_enabled:
                trend_ctx = detect_trend_from_open(market, avg_volume, vwap=vwap, pivot=pivot)
            else:
                trend_ctx = TrendContext(TrendSide.SIDEWAYS, 0, 0, 0)
            _log_environment_state(market, trend_ctx, trade_mode)

            if forced_decision and forced_decision.reason == "BASKET_TRAIL_HIT" and basket_mtm >= risk.daily_target:
                day_mode = "LOCKED_GREEN"

            # Intelligence layer: features -> scores -> policy
            features = feat_extractor.compute(market, vwap, pivot)
            scores = regime_scorer.score(features)
            pol = policy.decide(scores.as_dict())
            # Override trend context with policy regime when trading
            if not pol.should_trade:
                decision = forced_decision or TradeAction("HOLD", StrategyType.NONE, [], [], pol.notes)
            else:
                if pol.regime == TrendSide.BULL:
                    trend_ctx = TrendContext(TrendSide.BULL, int(scores.prob_bull * 10), int(scores.prob_bear * 10), int(pol.confidence * 10))
                elif pol.regime == TrendSide.BEAR:
                    trend_ctx = TrendContext(TrendSide.BEAR, int(scores.prob_bull * 10), int(scores.prob_bear * 10), int(pol.confidence * 10))
                elif pol.regime == TrendSide.SIDEWAYS:
                    trend_ctx = TrendContext(TrendSide.SIDEWAYS, int(scores.prob_bull * 10), int(scores.prob_bear * 10), int(pol.confidence * 10))
                decision = (
                    forced_decision
                    if forced_decision
                    else selector.decide(
                        market=market,
                        option_chain=chain,
                        expiry=expiry,
                        risk=risk,
                        current_positions=legs,
                        basket_mtm=basket_mtm,
                        trend_ctx=trend_ctx,
                        daily_trades=daily_trades,
                        day_mode=day_mode,
                        entries_used=day_entries,
                        legs_used=day_legs,
                        orb_state=orb_state,
                        sr_levels=sr_levels,
                        reentry_buffer=orb_config.reentry_buffer_points,
                    )
                )
            # Log intelligence snapshot
            try:
                with INTEL_LOG_PATH.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [
                            _ist_now().isoformat(),
                            scores.prob_bull,
                            scores.prob_bear,
                            scores.prob_sideways,
                            pol.regime.name if hasattr(pol, "regime") else "NA",
                            pol.strategy.name if hasattr(pol, "strategy") else "NA",
                            pol.size_multiplier if hasattr(pol, "size_multiplier") else 0.0,
                            pol.notes if hasattr(pol, "notes") else "",
                            features.gap_pct,
                            features.vix_value,
                            features.vwap_distance,
                        ]
                    )
            except Exception:
                pass
            log.info(
                "Decision=%s strategy=%s reason=%s (trend=%s bull=%s bear=%s)",
                decision.action_type,
                decision.strategy_type.value,
                decision.reason,
                trend_ctx.trend_side.value,
                trend_ctx.bull_score,
                trend_ctx.bear_score,
            )
            if decision.action_type != "HOLD" and decision.strategy_type != StrategyType.NONE:
                legs_to_log = decision.legs_to_open if decision.legs_to_open else decision.legs_to_close or legs
                _log_strategy_event(
                    strategy=decision.strategy_type,
                    action=decision.action_type,
                    reason=decision.reason,
                    market=market,
                    trade_mode=trade_mode,
                    legs=legs_to_log,
                )
            if decision.action_type.startswith("OPEN"):
                if _is_india_market_open():
                    for leg in decision.legs_to_open:
                        leg.opened_at = _ist_now()
                        _place_leg_order(dw, leg, close=False, trade_mode=trade_mode)
                        day_legs += 1
                    daily_trades += 1
                    day_entries += 1
                else:
                    log.warning("Market closed; skipping OPEN action (%s)", decision.action_type)
            elif decision.action_type.startswith("CLOSE"):
                target_legs = decision.legs_to_close or legs
                for leg in target_legs:
                    _place_leg_order(dw, leg, close=True, trade_mode=trade_mode)
                    day_legs += 1
        except Exception as exc:
            log.exception("Agent loop error: %s", exc)
        time.sleep(poll_sec)

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()
