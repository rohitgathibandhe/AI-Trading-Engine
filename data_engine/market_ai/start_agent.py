#!/usr/bin/env python3
"""
Refactored agent entrypoint.

Uses the new StrategySelector framework (regime-aware) instead of the legacy
weekly_theta_strangle loop. Warn-only by default; flip WARN_ONLY=false to allow
order placement once verified.
"""
from __future__ import annotations

import atexit
import json
import logging
import os
import sys
import time
from datetime import datetime, date as dt_date, time as dtime
from pathlib import Path
from typing import Any, Dict, List, Optional

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
DEFAULT_SETTINGS: Dict[str, Any] = {
    "lot_size": 75,
}
CREDS_FILE = STATE_DIR / "creds.json"

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
        return merged
    except Exception as exc:
        log.warning("Failed to load agent settings from %s (%s); falling back to defaults", path, exc)
        return dict(DEFAULT_SETTINGS)


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
            expiry = r.get("expiryDate") or r.get("expiry")
            expiry_date = datetime.fromisoformat(str(expiry)).date() if expiry else datetime.now().date()
            legs.append(
                OptionLeg(
                    symbol="NIFTY",
                    expiry=expiry_date,
                    strike=float(r.get("strikePrice") or r.get("strike") or 0),
                    option_type=OptionType.CALL if "C" in str(r.get("optionType") or r.get("option_type") or "C").upper() else OptionType.PUT,
                    side=side,
                    quantity=qty,
                    entry_price=float(r.get("avgPrice") or r.get("avg_price") or 0.0),
                    security_id=str(r.get("securityId") or r.get("security_id") or ""),
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
    for candidate in parsed:
        if candidate >= today:
            return candidate
    return parsed[-1] if parsed else today


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
    _warm_scrip_master(force=False)
    _ensure_dhan_credentials()
    settings = _load_agent_settings()
    dw = DhanWrapper(logger=logging.getLogger("dhan_wrapper"))
    _write_pid_file()
    lot_size = int(settings.get("lot_size", DEFAULT_SETTINGS["lot_size"]))
    selector = StrategySelector(symbol="NIFTY", lot_size=lot_size)
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
            expiry_str = expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry)
            chain_raw = dw.get_option_chain(13, "IDX_I", expiry_str)
            chain = _map_chain(chain_raw, expiry, symbol="NIFTY")
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
                if not _is_india_market_open():
                    log.warning("Market closed; skipping OPEN action (%s)", decision.action_type)
                    continue
                for leg in decision.legs_to_open:
                    side = leg.side
                    if isinstance(side, str):
                        side = LegSide.BUY if side.upper().startswith("B") else LegSide.SELL
                    if not leg.security_id:
                        log.error(
                            "Skipping %s leg due to missing security_id: strike=%s opt=%s qty=%s expiry=%s",
                            side,
                            leg.strike,
                            leg.option_type.value,
                            leg.quantity,
                            leg.expiry,
                        )
                        continue
                    try:
                        sec_id = int(float(leg.security_id))
                    except Exception:
                        log.error(
                            "Skipping %s leg due to invalid security_id=%s (strike=%s opt=%s)",
                            side,
                            leg.security_id,
                            leg.strike,
                            leg.option_type.value,
                        )
                        continue
                    log.info(
                        "Placing %s order sec_id=%s strike=%s opt=%s qty=%s expiry=%s",
                        side.value if hasattr(side, "value") else str(side),
                        sec_id,
                        leg.strike,
                        leg.option_type.value,
                        leg.quantity,
                        leg.expiry,
                    )
                    resp = dw.place_order(
                        side=side,
                        exchange_seg="NSE_FNO",
                        security_id=sec_id,
                        quantity=leg.quantity,
                        product_type="MARGIN",
                        order_type="MARKET",
                    )
                    log.info("order placed resp=%s", resp)
            if not warn_only and decision.action_type.startswith("CLOSE"):
                if not _is_india_market_open():
                    log.warning("Market closed; skipping CLOSE action (%s)", decision.action_type)
                    continue
                for leg in decision.legs_to_close:
                    side = leg.side
                    if isinstance(side, str):
                        side = LegSide.SELL if side.upper().startswith("B") else LegSide.BUY
                    else:
                        side = LegSide.BUY if side == LegSide.SELL else LegSide.SELL
                    if not leg.security_id:
                        log.error(
                            "Skipping close leg due to missing security_id: strike=%s opt=%s qty=%s expiry=%s",
                            leg.strike,
                            leg.option_type.value,
                            leg.quantity,
                            leg.expiry,
                        )
                        continue
                    try:
                        sec_id = int(float(leg.security_id))
                    except Exception:
                        log.error(
                            "Skipping close leg due to invalid security_id=%s (strike=%s opt=%s)",
                            leg.security_id,
                            leg.strike,
                            leg.option_type.value,
                        )
                        continue
                    log.info(
                        "Placing close order side=%s sec_id=%s strike=%s opt=%s qty=%s expiry=%s",
                        side.value if hasattr(side, "value") else str(side),
                        sec_id,
                        leg.strike,
                        leg.option_type.value,
                        leg.quantity,
                        leg.expiry,
                    )
                    resp = dw.place_order(
                        side=side,
                        exchange_seg="NSE_FNO",
                        security_id=sec_id,
                        quantity=leg.quantity,
                        product_type="MARGIN",
                        order_type="MARKET",
                    )
                    log.info("close order resp=%s", resp)
        except Exception as exc:
            log.exception("Agent loop error: %s", exc)
        time.sleep(poll_sec)


if __name__ == "__main__":
    main()
