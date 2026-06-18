# -*- coding: utf-8 -*-
"""
Dhan wrapper (robust funds + LTP parsing, bulk LTP, and live positions).
No UI framework dependencies.
"""

from __future__ import annotations

import os
import sys
import threading
import time
import requests
from pathlib import Path
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, date as dt_date, time as dtime
from enum import Enum
import re
import importlib.util
import json
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# Shared singleton & locks for rate-limited option chain access
_SHARED_WRAPPER: Optional["DhanWrapper"] = None
_SHARED_WRAPPER_LOCK = threading.Lock()

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


# ---------------------------
# Enums expected by your agent
# ---------------------------

class Leg(Enum):
    SINGLE = "SINGLE"
    BULL_CALL = "BULL_CALL"
    BEAR_PUT = "BEAR_PUT"
    CUSTOM = "CUSTOM"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


__all__ = [
    "DhanWrapper",
    "Leg",
    "OrderSide",
]

# ---- Public API compatibility aliases (used by older modules/tests) ----
# Some parts of the project call `get_ltp_many`, `get_positions_raw`, or `positions()`.
# Provide thin shims so those call sites don't break under refactors.


# ---------------------------
# Import Dhan SDK safely
# ---------------------------
# Ensure local checkout is on sys.path so the vendored dhan_sdk works even when
# PYTHONPATH was not exported (common when launching from local tools/UI).
HERE = Path(__file__).resolve()
ROOT = HERE.parents[2]          # repo root
DATA_ENGINE_DIR = HERE.parents[1]  # .../data_engine
LOCAL_PKG_DIR = HERE.parent        # .../data_engine/market_ai (needed for top-level 'dhan_sdk')
for p in (str(ROOT), str(DATA_ENGINE_DIR), str(LOCAL_PKG_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

def _import_from_dhan_sdk():
    last_err = None
    candidates = [
        ("dhan_sdk.dhan_http", "DhanHTTP"),
        ("dhan_sdk._market_feed", "MarketFeed"),
        ("dhan_sdk._option_chain", "OptionChain"),
        ("dhan_sdk._portfolio", "Portfolio"),
        ("dhan_sdk._historical_data", "HistoricalData"),
        # project-local fallbacks
        ("market_ai.dhan_sdk.dhan_http", "DhanHTTP"),
        ("market_ai.dhan_sdk._market_feed", "MarketFeed"),
        ("market_ai.dhan_sdk._option_chain", "OptionChain"),
        ("market_ai.dhan_sdk._portfolio", "Portfolio"),
        ("market_ai.dhan_sdk._historical_data", "HistoricalData"),
    ]
    mods: Dict[str, Any] = {}
    for mod_name, symbol in candidates:
        try:
            mod = __import__(mod_name, fromlist=["*"])
            mods[symbol] = getattr(mod, symbol)
        except Exception as e:
            last_err = e
            # attempt to load from vendored source immediately
            fallback = HERE.parents[1] / "dhan_sdk" / f"{mod_name.split('.')[-1]}.py"
            if fallback.exists():
                try:
                    spec = importlib.util.spec_from_file_location(f"dhan_sdk_local_{symbol}", str(fallback))
                    module = importlib.util.module_from_spec(spec)
                    assert spec.loader is not None
                    spec.loader.exec_module(module)
                    mods[symbol] = getattr(module, symbol)
                    continue
                except Exception as exc:
                    last_err = exc
            continue
    needed = {"DhanHTTP", "MarketFeed", "OptionChain", "Portfolio", "HistoricalData"}
    if not needed.issubset(mods):
        raise ModuleNotFoundError(
            "Could not import dhan_sdk modules. Ensure 'dhan_sdk' is on PYTHONPATH. "
            f"Missing={needed - set(mods)}, last_error={last_err}"
        )
    return (
        mods["DhanHTTP"],
        mods["MarketFeed"],
        mods["OptionChain"],
        mods["Portfolio"],
        mods["HistoricalData"],
    )


DhanHTTP, MarketFeed, OptionChain, Portfolio, HistoricalData = _import_from_dhan_sdk()


@dataclass(frozen=True)
class DhanEndpoints:
    funds: str = os.getenv("DHAN_FUNDS_PATH", "/fundlimit")
    optionchain: str = os.getenv("DHAN_OPTIONCHAIN_PATH", "/optionchain")
    optionchain_expirylist: str = os.getenv("DHAN_OPTIONCHAIN_EXPIRYLIST_PATH", "/optionchain/expirylist")
    positions: str = os.getenv("DHAN_POSITIONS_PATH", "/positions")
    # positions are accessed via the Portfolio helper


def _ist_now() -> datetime:
    try:
        tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    except Exception:
        tz = None
    return datetime.now(tz) if tz else datetime.now()


def _is_india_market_open(now: Optional[datetime] = None) -> bool:
    if now is None:
        now = _ist_now()
    if now.weekday() >= 5:  # Sat/Sun
        return False
    start = dtime(9, 15)
    end = dtime(15, 30)
    return start <= now.time() <= end


class _DefaultLogger:
    def info(self, *args, **kwargs) -> None:
        msg = args[0] if args else ""
        print(msg)
    def warning(self, *args, **kwargs) -> None:
        msg = args[0] if args else ""
        print(msg)
    def error(self, *args, **kwargs) -> None:
        msg = args[0] if args else ""
        print(msg)
    def exception(self, *args, **kwargs) -> None:
        msg = args[0] if args else ""
        print(msg)


class _SDKContext:
    def __init__(self, http):
        self._http = http

    def get_dhan_http(self):
        return self._http


class DhanWrapper:
    """
    Thin wrapper around dhan_sdk:
      - Robust funds parsing
      - Robust single LTP + bulk LTP
      - Live positions with multiple fallbacks
      - Simple background LTP poller
    """

    # Shared throttle across all instances (UI + agent). Dhan allows 1 OC call / 3s.
    _GLOBAL_CHAIN_LOCK = threading.Lock()
    _GLOBAL_LAST_CHAIN_TS: float = 0.0
    _GLOBAL_CHAIN_COOLDOWN_UNTIL: Optional[float] = None

    @staticmethod
    def _normalize_expiry_value(expiry: Any) -> Optional[str]:
        """
        Accept str/date/datetime/number and return YYYY-MM-DD string if possible.
        """
        if expiry is None:
            return None
        if hasattr(expiry, "isoformat"):
            try:
                return expiry.isoformat()
            except Exception:
                pass
        if isinstance(expiry, (int, float)):
            return str(expiry)
        text = str(expiry).strip()
        if not text:
            return None
        # Trim any timestamp portion
        if "T" in text:
            text = text.split("T", 1)[0]
        if " " in text:
            text = text.split(" ", 1)[0]
        return text

    @staticmethod
    def _coerce_expiry_list(payload: Any) -> List[str]:
        """
        Flatten various DHAN expiry list payloads into a sorted list of ISO date strings.
        """
        stack: List[Any] = [payload]
        collected: List[str] = []
        seen: set[int] = set()
        while stack:
            cur = stack.pop()
            if isinstance(cur, list):
                for item in cur:
                    if isinstance(item, (list, dict)):
                        stack.append(item)
                    else:
                        norm = DhanWrapper._normalize_expiry_value(item)
                        if norm:
                            collected.append(norm)
                continue
            if isinstance(cur, dict):
                oid = id(cur)
                if oid in seen:
                    continue
                seen.add(oid)
                for key in ("expiryList", "ExpiryList", "Expiry", "expiry", "expiries", "data", "Data"):
                    if key in cur:
                        stack.append(cur[key])
                continue
            norm = DhanWrapper._normalize_expiry_value(cur)
            if norm:
                collected.append(norm)
        unique: List[str] = []
        for raw in collected:
            try:
                # Validate format (allow YYYY-MM-DD only)
                dt_date.fromisoformat(raw)
                if raw not in unique:
                    unique.append(raw)
            except Exception:
                continue
        return sorted(unique)
    def _refresh_token_if_stale(self) -> None:
        """Hot-reload token from creds.json if the file changed since last read.
        Called before every outbound API request so a mid-session token update
        (e.g. user pastes a new Dhan token) is picked up without restarting."""
        state_dir = Path(__file__).resolve().parent / "state"
        creds_file = state_dir / "creds.json"
        try:
            mtime = creds_file.stat().st_mtime
        except OSError:
            return
        if mtime <= self._creds_file_mtime:
            return
        try:
            data = json.loads(creds_file.read_text())
        except Exception:
            return
        tok = str(data.get("access_token") or "").strip()
        if not tok or tok == self.access_token:
            self._creds_file_mtime = mtime
            return
        self.access_token = tok
        # Propagate into the underlying HTTP layer so ongoing connections use the new token
        if hasattr(self.http, "access_token"):
            self.http.access_token = tok
        if hasattr(self.http, "header") and isinstance(self.http.header, dict):
            self.http.header["access-token"] = tok
        if hasattr(self.http, "_headers") and isinstance(self.http._headers, dict):
            self.http._headers["access-token"] = tok
        self._creds_file_mtime = mtime
        logging.getLogger("dhan_wrapper").info("Hot-reloaded DHAN access token from state/creds.json")

    def _order_headers(self) -> Dict[str, str]:
        self._refresh_token_if_stale()
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": self.access_token,
            "client-id": self.client_id,
        }

    def place_order(
        self,
        *,
        side: str,
        exchange_seg: str,
        security_id: int,
        quantity: int,
        product_type: str = "MARGIN",
        order_type: str = "MARKET",
        validity: str = "DAY",
        price: Optional[float] = None,
        client_order_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Minimal order placement by security_id. Uses /v2/orders directly.
        Side: BUY/SELL. exchange_seg: e.g., NSE_FNO.
        """
        url = f"{os.getenv('DHAN_API_BASE', 'https://api.dhan.co').rstrip('/')}/v2/orders"
        payload: Dict[str, Any] = {
            "transactionType": side.upper(),
            "exchangeSegment": exchange_seg,
            "productType": (product_type or "MARGIN").upper(),
            "orderType": order_type,
            "validity": validity,
            "securityId": int(security_id),
            "quantity": int(quantity),
            "price": float(price) if price is not None else 0,
            "afterMarketOrder": False,
            "amoTime": None,
            "dhanClientId": self.client_id,
        }
        if client_order_id:
            payload["clientOrderId"] = str(client_order_id)
        resp = requests.post(url, headers=self._order_headers(), json=payload, timeout=10)
        try:
            js = resp.json()
        except Exception:
            self.log.error("[order] non-JSON response %s", resp.text[:400])
            resp.raise_for_status()
            raise
        if resp.status_code >= 400 or (isinstance(js, dict) and js.get("status") == "failed"):
            error_info = js if isinstance(js, dict) else resp.text[:400]
            payload_snapshot = {
                "transactionType": payload.get("transactionType"),
                "exchangeSegment": payload.get("exchangeSegment"),
                "productType": payload.get("productType"),
                "orderType": payload.get("orderType"),
                "validity": payload.get("validity"),
                "securityId": payload.get("securityId"),
                "quantity": payload.get("quantity"),
                "price": payload.get("price"),
            }
            self.log.error(
                "[order] failed status=%s info=%s payload=%s",
                resp.status_code,
                error_info,
                payload_snapshot,
            )
            resp.raise_for_status()
        return js

    def order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Fetch order status from /v2/orders/{order_id}.
        """
        url = f"{os.getenv('DHAN_API_BASE', 'https://api.dhan.co').rstrip('/')}/v2/orders/{order_id}"
        resp = requests.get(url, headers=self._order_headers(), timeout=10)
        try:
            js = resp.json()
        except Exception:
            self.log.error("[order_status] non-JSON response %s", resp.text[:400])
            resp.raise_for_status()
            raise
        if resp.status_code >= 400 or (isinstance(js, dict) and js.get("status") == "failed"):
            self.log.error("[order_status] failed %s: %s", resp.status_code, str(js)[:400])
            resp.raise_for_status()
        return js

    # --- Compatibility shims -------------------------------------------------
    def get_ltp_many(self, exchange_seg: str, security_ids: list[int]) -> Dict[tuple[str, int], Optional[float]]:
        """
        Backwards-compatible alias for `get_ltp_bulk`.
        Returns a dict {(seg, id): ltp or None}.
        """
        pairs: list[tuple[str, int]] = []
        for sid in security_ids or []:
            try:
                pairs.append((exchange_seg, int(sid)))
            except Exception:
                continue
        return self.get_ltp_bulk(pairs)

    def get_positions_raw(self) -> List[Dict[str, Any]]:
        """
        Return raw list from /v2/positions without normalization,
        matching older call sites that expect the untouched payload.
        """
        resp = self.http.get(self.endpoints.positions)
        if isinstance(resp, dict):
            data = resp.get("data") or resp.get("positions") or resp.get("netPositions")
            return data if isinstance(data, list) else []
        return resp if isinstance(resp, list) else []

    def positions(self) -> List[Dict[str, Any]]:
        """
        Alias for normalized live positions.
        """
        return self.get_positions_live()

    def _load_creds_from_state(self) -> None:
        """
        Ensure env creds align with state/creds.json (used by local control UIs).
        Helps when code is run outside start_agent.
        """
        # state directory lives at data_engine/market_ai/state
        state_dir = Path(__file__).resolve().parent / "state"
        creds_file = state_dir / "creds.json"
        if not creds_file.exists():
            return
        try:
            data = json.loads(creds_file.read_text())
        except Exception:
            return
        if not isinstance(data, dict):
            return
        cid = str(data.get("client_id") or "").strip()
        tok = str(data.get("access_token") or "").strip()
        updated = False
        if cid and os.getenv("DHAN_CLIENT_ID", "").strip() != cid:
            os.environ["DHAN_CLIENT_ID"] = cid
            updated = True
        if tok and os.getenv("DHAN_ACCESS_TOKEN", "").strip() != tok:
            os.environ["DHAN_ACCESS_TOKEN"] = tok
            updated = True
        if updated:
            logging.getLogger("dhan_wrapper").info("Loaded DHAN creds from state/creds.json")

    def __init__(
        self,
        dhan_client_id: Optional[str] = None,
        access_token: Optional[str] = None,
        logger: Optional[Any] = None,
        http_pool: Optional[dict] = None,
        disable_ssl: bool = False,
    ):
        # Align env with saved creds before reading
        self._load_creds_from_state()
        self.client_id: str = (dhan_client_id or os.getenv("DHAN_CLIENT_ID", "")).strip()
        self.access_token: str = (access_token or os.getenv("DHAN_ACCESS_TOKEN", "")).strip()
        if not self.client_id or not self.access_token:
            raise TypeError(
                "DhanWrapper requires client_id and access_token. "
                "Pass them to the constructor or set DHAN_CLIENT_ID/DHAN_ACCESS_TOKEN."
            )

        self.log: Any = logger or _DefaultLogger()
        self.http: Any = DhanHTTP(
            client_id=self.client_id,
            access_token=self.access_token,
            disable_ssl=disable_ssl,
            pool=http_pool,
        )
        self.endpoints: DhanEndpoints = DhanEndpoints()
        self._sdk_context = _SDKContext(self.http)
        self._historical = HistoricalData(self._sdk_context)
        self._creds_file_mtime: float = 0.0
        # Simple in-memory throttling/cache for option-chain to avoid 429
        self._last_chain_ts: Optional[float] = None
        self._last_chain_key: Optional[str] = None
        self._last_chain_resp: Optional[Dict[str, Any]] = None
        self._chain_cooldown_until: Optional[float] = None

        # poller state
        self._stop: threading.Event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -------- tiny helpers (used across LTP & positions) --------

    @staticmethod
    def _as_float(x: Any) -> Optional[float]:
        """Safe float() with None on failure; name used by get_ltp_once."""
        try:
            return float(x)
        except Exception:
            return None

    # keep older name as alias, some call sites may still reference it
    _to_float = _as_float

    @staticmethod
    def _parse_timestamp(value: Any) -> Optional[datetime]:
        if value in (None, "", [], {}):
            return None
        if isinstance(value, (int, float)):
            try:
                epoch = float(value)
                if epoch > 1_000_000_000_000:
                    epoch /= 1000.0
                if epoch > 10_000_000:
                    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
                    parsed = datetime.fromtimestamp(epoch, tz=tz) if tz else datetime.fromtimestamp(epoch)
                    return parsed.replace(tzinfo=None)
            except Exception:
                return None
        text = str(value).strip()
        if not text:
            return None
        try:
            epoch = float(text)
            if epoch > 1_000_000_000_000:
                epoch /= 1000.0
            if epoch > 10_000_000:
                tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
                parsed = datetime.fromtimestamp(epoch, tz=tz) if tz else datetime.fromtimestamp(epoch)
                return parsed.replace(tzinfo=None)
        except Exception:
            pass
        text = text.replace("T", " ").replace("/", "-")
        try:
            return datetime.fromisoformat(text)
        except Exception:
            pass
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%d-%m-%Y %H:%M:%S"):
            try:
                return datetime.strptime(text, fmt)
            except Exception:
                continue
        return None

    @staticmethod
    def _pick(d: Dict[str, Any], *keys: str) -> Any:
        """Return the first non-empty value for any of the provided keys in dict d."""
        for k in keys:
            if k in d and d[k] not in (None, "", [], {}):
                return d[k]
        return None

    # -------- Funds --------

    def get_funds(self) -> Dict[str, Optional[float]]:
        """
        Returns a normalized dict with keys:
        - available, collateral, utilized, withdrawable
        """
        resp = self.http.get(self.endpoints.funds)

        payload: Any = resp
        if isinstance(resp, dict) and "data" in resp and resp.get("status") == "success":
            payload = resp["data"]

        out: Dict[str, Optional[float]] = {"available": None, "collateral": None, "utilized": None, "withdrawable": None}

        def _as_number(val: Any) -> Optional[float]:
            """Recursively parse a value as a float, handling nested dicts/lists and cleaning strings."""
            if val in (None, "", [], {}):
                return None

            # unwrap common nested structures (e.g. {'value': '12345'})
            if isinstance(val, dict):
                for key in ("value", "amount", "balance"):
                    if key in val:
                        return _as_number(val[key])
                # fall back to first numeric-looking value
                for v in val.values():
                    parsed = _as_number(v)
                    if parsed is not None:
                        return parsed
                return None

            if isinstance(val, (list, tuple)):
                for item in val:
                    parsed = _as_number(item)
                    if parsed is not None:
                        return parsed
                return None

            if isinstance(val, Decimal):
                return float(val)

            if isinstance(val, str):
                cleaned = re.sub(r"[^\d\.\-]", "", val).strip()
                if cleaned in ("", "-", ".", "-.", ".-", "--"):
                    return None
                val = cleaned

            return DhanWrapper._as_float(val)

        if isinstance(payload, dict):
            out["available"] = _as_number(self._pick(
                payload,
                "availableMargin",
                "available_margin",
                "availableBalance",
                "availabelBalance",
                "availableCash",
                "cashBalance",
                "availableCashBalance",
                "available",
            ))
            out["collateral"] = _as_number(self._pick(
                payload,
                "collateralMargin",
                "collateralAmount",
                "collateralValue",
                "collateral",
            ))
            out["utilized"] = _as_number(self._pick(
                payload,
                "utilised",
                "utilized",
                "utilizedMargin",
                "utilizedAmount",
                "usedMargin",
                "used_amount",
            ))
            out["withdrawable"] = _as_number(self._pick(
                payload,
                "withdrawableBalance",
                "withdrawableCashBalance",
                "withdrawableAmount",
                "withdrawableCash",
                "withdrawable",
            ))

        self.log.info(f"[Funds] parsed => {out}")
        return out

    # -------- Option Chain --------

    def get_expiry_list(self, underlying_security_id: int, underlying_seg: str) -> Dict[str, Any]:
        ctx = type("Ctx", (object,), {"get_dhan_http": lambda _: self.http})()
        oc = OptionChain(ctx)
        segs = []
        if underlying_seg and underlying_seg.upper().startswith("NSE"):
            segs.append("IDX_I")
        segs.append(underlying_seg)
        segs.append("NSE_FNO")
        seen = set()
        for seg in segs:
            if not seg or seg in seen:
                continue
            seen.add(seg)
            try:
                resp = oc.expiry_list(underlying_security_id, seg)
                if isinstance(resp, dict) and resp.get("data"):
                    return resp
                elif resp:
                    return resp
            except Exception:
                continue
        return {}

    def get_optionchain_expirylist(self, underlying_seg: str, underlying_security_id: int) -> List[str]:
        """
        Compatibility helper used by some agents/tests to fetch expiry list.
        Returns a sorted list of ISO date strings when possible.
        """
        try:
            raw = self.get_expiry_list(underlying_security_id, underlying_seg)
        except Exception:
            return []
        return self._coerce_expiry_list(raw)

    def get_option_chain(self, underlying_security_id: int, underlying_seg: str, expiry: Any) -> Dict[str, Any]:
        ctx = type("Ctx", (object,), {"get_dhan_http": lambda _: self.http})()
        oc = OptionChain(ctx)
        expiry_str = self._normalize_expiry_value(expiry)
        if not expiry_str:
            raise ValueError("expiry is required for get_option_chain")

        key = f"{underlying_security_id}:{underlying_seg}:{expiry_str}"
        now = time.time()

        # Global/shared throttle: Dhan allows 1 request / 3 seconds.
        with self._GLOBAL_CHAIN_LOCK:
            last_ts = max(
                self._GLOBAL_LAST_CHAIN_TS or 0.0,
                self._last_chain_ts or 0.0,
            )
            cooldown_until = (
                self._GLOBAL_CHAIN_COOLDOWN_UNTIL
                if self._GLOBAL_CHAIN_COOLDOWN_UNTIL is not None
                else self._chain_cooldown_until
            )

            # If we recently hit 429, honor cooldown for all instances
            if cooldown_until and now < cooldown_until:
                wait_for = cooldown_until - now
                self.log.warning(
                    "[option_chain] cooling down after 429; wait %.1fs",
                    wait_for,
                )
                time.sleep(max(0, wait_for))
                now = time.time()

            # Throttle to 1 call / 3s globally
            if last_ts and now - last_ts < 3.0:
                wait_for = 3.0 - (now - last_ts)
                self.log.debug(
                    "[option_chain] throttling for %.2fs to respect global 1/3s limit",
                    wait_for,
                )
                time.sleep(wait_for)
                now = time.time()

        # throttle: reuse recent response for same expiry
        if (
            self._last_chain_resp is not None
            and self._last_chain_key == key
            and self._last_chain_ts is not None
            and now - self._last_chain_ts < 3.0
        ):
            return self._last_chain_resp

        segs = []
        # Primary guess for NIFTY index
        if underlying_seg and underlying_seg.upper().startswith("NSE"):
            segs.append("IDX_I")
        segs.append(underlying_seg)
        # Legacy fallback
        segs.append("NSE_FNO")
        seen = set()

        def _try_fetch(seg: str) -> Optional[Dict[str, Any]]:
            try:
                resp = oc.option_chain(underlying_security_id, seg, expiry_str)
            except Exception as exc:
                self.log.warning(f"[option_chain] fetch failed seg={seg}: {exc}")
                return None
            if isinstance(resp, dict):
                status = resp.get("status")
                data = resp.get("data")
                # Explicit 429 handling
                if status == "failed" and isinstance(data, dict) and "805" in data:
                    self.log.warning("[option_chain] 429 rate limit hit; backing off 90s")
                    cooldown = time.time() + 90.0
                    self._chain_cooldown_until = cooldown
                    with self._GLOBAL_CHAIN_LOCK:
                        self._GLOBAL_CHAIN_COOLDOWN_UNTIL = cooldown
                    return None
                if status == "success" or data:
                    return resp
            elif resp:
                return resp
            return None

        resp: Optional[Dict[str, Any]] = None
        for seg in segs:
            if not seg or seg in seen:
                continue
            seen.add(seg)
            try:
                max_attempts = max(1, int(getattr(self, "option_chain_attempts", 3) or 3))
            except Exception:
                max_attempts = 3
            for attempt in range(max_attempts):
                resp = _try_fetch(seg)
                if resp is not None:
                    break
                # backoff to avoid 429
                time.sleep(1.5)
            if resp is not None:
                break

        if resp is None:
            return {}

        # cache successful fetch
        ts_now = time.time()
        self._last_chain_resp = resp
        self._last_chain_ts = ts_now
        # update shared markers
        with self._GLOBAL_CHAIN_LOCK:
            self._GLOBAL_LAST_CHAIN_TS = ts_now
            self._GLOBAL_CHAIN_COOLDOWN_UNTIL = None
        self._last_chain_key = key
        return resp

    def get_intraday_candles(
        self,
        security_id: int,
        exchange_segment: str,
        instrument_type: str = "INDEX",
        *,
        interval: int = 5,
        from_date: Optional[str] = None,
        to_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch intraday OHLCV from /charts/intraday via the HistoricalData helper.
        Returns a list of dicts sorted by timestamp with keys: timestamp, open, high, low, close, volume.
        """
        self._refresh_token_if_stale()
        from_str = (from_date or dt_date.today().isoformat()).strip()
        to_str = (to_date or from_str).strip()
        try:
            resp = self._historical.intraday_minute_data(
                security_id=security_id,
                exchange_segment=exchange_segment,
                instrument_type=instrument_type,
                from_date=from_str,
                to_date=to_str,
                interval=interval,
            )
        except Exception as exc:
            self.log.exception("[historical] intraday fetch failed: %s", exc)
            return []

        data: Any = resp
        if isinstance(data, dict) and data.get("status") == "failure":
            self.log.error("[historical] intraday fetch failure: %s", data.get("remarks", data))
            return []
        if isinstance(data, dict):
            for key in ("data", "Data", "candles", "CANDLES"):
                if key in data:
                    data = data[key]
                    break
        if isinstance(data, dict):
            for key in ("data", "candles"):
                if key in data:
                    data = data[key]
                    break
        if isinstance(data, dict) and {"open", "high", "low", "close"}.issubset(data.keys()):
            opens = data.get("open", [])
            highs = data.get("high", [])
            lows = data.get("low", [])
            closes = data.get("close", [])
            vols = data.get("volume", [])
            ts = data.get("timestamp", [])
            temp: List[Dict[str, Any]] = []
            length = min(len(opens), len(highs), len(lows), len(closes))
            for i in range(length):
                temp.append(
                    {
                        "timestamp": self._parse_timestamp(ts[i]) if i < len(ts) else None,
                        "open": opens[i],
                        "high": highs[i],
                        "low": lows[i],
                        "close": closes[i],
                        "volume": vols[i] if i < len(vols) else 0.0,
                    }
                )
            data = temp
        if not isinstance(data, list):
            self.log.warning("[historical] intraday unexpected response type=%s: %s", type(data).__name__, str(data)[:200])
            return []

        candles: List[Dict[str, Any]] = []

        for row in data:
            if not isinstance(row, dict):
                continue
            o = self._as_float(self._pick(row, "open", "openPrice", "Open", "o"))
            h = self._as_float(self._pick(row, "high", "highPrice", "High", "h"))
            l = self._as_float(self._pick(row, "low", "lowPrice", "Low", "l"))
            c = self._as_float(self._pick(row, "close", "closePrice", "Close", "c", "ltp"))
            v = self._as_float(self._pick(row, "volume", "volumeTraded", "Volume", "v"))
            ts = self._parse_timestamp(
                self._pick(
                    row,
                    "startTime",
                    "start_time",
                    "time",
                    "timestamp",
                    "tradeTime",
                    "dateTime",
                )
            )
            if o is None and h is None and l is None and c is None:
                continue
            candle = {
                "timestamp": ts,
                "open": o if o is not None else (h or l or c or 0.0),
                "high": h if h is not None else (o or l or c or 0.0),
                "low": l if l is not None else (o or h or c or 0.0),
                "close": c if c is not None else (o or h or l or 0.0),
                "volume": v if v is not None else 0.0,
            }
            candles.append(candle)
        candles.sort(key=lambda item: item.get("timestamp") or datetime.min)
        return candles

    def get_daily_candles(
        self,
        security_id: int,
        exchange_segment: str,
        instrument_type: str = "INDEX",
        *,
        from_date: str,
        to_date: str,
    ) -> List[Dict[str, Any]]:
        """
        Fetch daily OHLCV from /charts/historical via the HistoricalData helper.
        Returns a list of dicts sorted by timestamp with keys: timestamp, open, high, low, close, volume.
        """
        try:
            resp = self._historical.historical_daily_data(
                security_id=security_id,
                exchange_segment=exchange_segment,
                instrument_type=instrument_type,
                from_date=from_date,
                to_date=to_date,
                expiry_code=0,
            )
        except Exception as exc:
            self.log.exception("[historical] daily fetch failed: %s", exc)
            return []

        data: Any = resp
        if isinstance(data, dict):
            for key in ("data", "Data", "candles", "CANDLES"):
                if key in data:
                    data = data[key]
                    break
        if isinstance(data, dict):
            for key in ("data", "candles"):
                if key in data:
                    data = data[key]
                    break
        if isinstance(data, dict) and {"open", "high", "low", "close"}.issubset(data.keys()):
            opens = data.get("open", [])
            highs = data.get("high", [])
            lows = data.get("low", [])
            closes = data.get("close", [])
            vols = data.get("volume", [])
            ts = data.get("timestamp", [])
            temp_list: List[Dict[str, Any]] = []
            length = min(len(opens), len(highs), len(lows), len(closes))
            for i in range(length):
                temp_list.append(
                    {
                        "timestamp": self._parse_timestamp(ts[i]) if i < len(ts) else None,
                        "open": opens[i],
                        "high": highs[i],
                        "low": lows[i],
                        "close": closes[i],
                        "volume": vols[i] if i < len(vols) else 0.0,
                    }
                )
            data = temp_list
        if not isinstance(data, list):
            return []

        candles: List[Dict[str, Any]] = []
        for row in data:
            if not isinstance(row, dict):
                continue
            ts = self._parse_timestamp(
                self._pick(
                    row,
                    "timestamp",
                    "time",
                    "datetime",
                    "dateTime",
                )
            )
            o = self._as_float(self._pick(row, "open", "Open", "o"))
            h = self._as_float(self._pick(row, "high", "High", "h"))
            l = self._as_float(self._pick(row, "low", "Low", "l"))
            c = self._as_float(self._pick(row, "close", "Close", "c"))
            v = self._as_float(self._pick(row, "volume", "Volume", "v"))
            if None in (o, h, l, c):
                continue
            candles.append(
                {
                    "timestamp": ts,
                    "open": o,
                    "high": h,
                    "low": l,
                    "close": c,
                    "volume": v if v is not None else 0.0,
                }
            )
        candles.sort(key=lambda item: item.get("timestamp") or datetime.min)
        return candles

    # -------- LTP (single + bulk) --------

    @staticmethod
    def _unwrap_to_segment(obj: Any, exchange_seg: str) -> Any:
        """
        Dhan sometimes nests {'data': {..., 'data': {...}}}.
        Keep unwrapping 'data' while it looks like a wrapper and we haven't
        reached the structure that contains the exchange segment key or a list.
        """
        seen = set()
        cur = obj
        while isinstance(cur, dict):
            if exchange_seg in cur or isinstance(cur.get(exchange_seg), (dict, list)):
                return cur
            oid = id(cur)
            if oid in seen:
                return cur
            seen.add(oid)
            nxt = cur.get("data")
            if isinstance(nxt, dict):
                cur = nxt
                continue
            if isinstance(nxt, list):
                return nxt
            break
        return cur

    def _extract_ltp_from_response(self, resp: Any, exchange_seg: str, security_id: int) -> Optional[float]:
        """
        Recursively search for the LTP value in a dhan_sdk LTP/quote response.
        """
        if isinstance(resp, dict) and resp.get("status") == "failure" and not resp.get("data"):
            return None

        def iter_nodes(obj: Any, visited: set[int]) -> Iterable[Dict[str, Any]]:
            if isinstance(obj, dict):
                oid = id(obj)
                if oid in visited:
                    return
                visited.add(oid)
                yield obj
                for v in obj.values():
                    yield from iter_nodes(v, visited)
            elif isinstance(obj, list):
                for item in obj:
                    yield from iter_nodes(item, visited)

        def match_ltp(node: Dict[str, Any]) -> Optional[float]:
            seg = node.get("exchangeSegment") or node.get("exchange_segment") or node.get("segment") or node.get("segmentName")
            sid = node.get("securityId") or node.get("security_id") or node.get("id") or node.get("instrumentToken")
            if seg and str(seg).upper() != str(exchange_seg).upper():
                return None
            if sid is not None and str(sid) != str(security_id):
                return None
            for key in ("last_price", "LTP", "ltp", "close", "closingPrice", "price", "lastPrice"):
                if key in node:
                    val = DhanWrapper._as_float(node[key])
                    if val is not None:
                        return val
            return None

        data = resp
        if isinstance(resp, dict):
            data = resp.get("data", resp)
        data = self._unwrap_to_segment(data, exchange_seg)

        if isinstance(data, dict) and exchange_seg in data:
            direct = data[exchange_seg]
            result = self._extract_ltp_from_response(direct, exchange_seg, security_id)
            if result is not None:
                return result
        if isinstance(data, dict):
            maybe = data.get(str(security_id)) or data.get(security_id)
            if isinstance(maybe, dict):
                for key in ("last_price", "LTP", "ltp"):
                    if key in maybe:
                        val = DhanWrapper._as_float(maybe[key])
                        if val is not None:
                            return val

        visited: set[int] = set()
        for node in iter_nodes(data, visited):
            if not isinstance(node, dict):
                continue
            ltp = match_ltp(node)
            if ltp is not None:
                return ltp
        return None

    def get_ltp_once(self, exchange_seg: str, security_id: int) -> Optional[float]:
        """
        Robustly parse different shapes Dhan may return for /marketfeed/ltp with fallbacks.
        """
        ctx = type("Ctx", (object,), {"get_dhan_http": lambda _: self.http})()
        mf = MarketFeed(ctx)
        payload = {exchange_seg: [security_id]}
        responses: list[Tuple[str, Any]] = []

        for source in ("ticker", "quote"):
            try:
                if source == "ticker":
                    resp = mf.ticker_data(payload)
                else:
                    resp = mf.quote_data(payload)
                responses.append((source, resp))
                ltp = self._extract_ltp_from_response(resp, exchange_seg, security_id)
                if ltp is not None:
                    return ltp
            except Exception as e:
                self.log.exception(f"[LTP] {source} fetch error: {e}")

        if responses:
            if all(isinstance(r, dict) and r.get("status") == "failure" for _, r in responses):
                return None
            last_source, last_resp = responses[-1]
            self.log.warning(f"[LTP] Could not parse LTP from {last_source} response: {last_resp}")
        return None

    def get_ltp_bulk(self, pairs: list[tuple[str, int]]) -> Dict[tuple[str, int], Optional[float]]:
        """
        Fetch LTP for many (exchange_seg, security_id) pairs at once.
        Returns a dict {(seg, id): ltp or None}.
        Works with both dict-of-dicts and list-of-dicts marketfeed shapes.
        """
        if not pairs:
            return {}

        # Build payload per segment
        payload: Dict[str, list[int]] = {}
        for seg, sid in pairs:
            if not seg or sid is None:
                continue
            try:
                sid = int(sid)
            except Exception:
                continue
            payload.setdefault(seg, []).append(sid)

        if not payload:
            return {}

        ctx = type("Ctx", (object,), {"get_dhan_http": lambda _: self.http})()
        mf = MarketFeed(ctx)

        out: Dict[tuple[str, int], Optional[float]] = {}
        try:
            resp = mf.ticker_data(payload)
            if isinstance(resp, dict) and resp.get("status") == "failure":
                msg = resp.get("remarks", {}).get("error_message")
                if msg:
                    self.log.debug(f"[LTP-bulk] status=failure remarks={msg}")
                return out
            data = resp.get("data", resp) if isinstance(resp, dict) else resp

            if isinstance(data, dict):
                for seg, node in data.items():
                    # dict-of-dicts: {'NSE_FNO': {'52802': {'last_price': ...}, ...}}
                    if isinstance(node, dict):
                        for k, v in node.items():
                            if not isinstance(v, dict):
                                continue
                            ltp = (
                                self._as_float(v.get("last_price"))
                                or self._as_float(v.get("LTP"))
                                or self._as_float(v.get("ltp"))
                            )
                            try:
                                out[(seg, int(k))] = ltp
                            except Exception:
                                pass
                    # list-of-dicts: {'NSE_FNO': [{'securityId': 52802, 'last_price': ...}, ...]}
                    elif isinstance(node, list):
                        for item in node:
                            if not isinstance(item, dict):
                                continue
                            sid = item.get("securityId") or item.get("security_id") or item.get("id")
                            ltp = (
                                self._as_float(item.get("last_price"))
                                or self._as_float(item.get("LTP"))
                                or self._as_float(item.get("ltp"))
                            )
                            try:
                                out[(seg, int(sid))] = ltp
                            except Exception:
                                pass
        except Exception as e:
            self.log.exception(f"[LTP-bulk] error: {e}")

        return out
    # -------- Live Positions (via official REST API) --------
    def get_positions_live(self) -> list[Dict[str, Any]]:
        """
        Call /v2/positions and normalize rows for the UI with:
          symbol, product, qty, avg_price, ltp, pnl, exchange_seg, security_id, side
        PnL rule:
          - if realizedProfit != 0 use realizedProfit
          - else BUY (qty>0):  (buyAvg - costPrice) * qty
               SELL (qty<0):  (costPrice - sellAvg) * abs(qty)
        """
        try:
            resp = self.http.get(self.endpoints.positions)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch positions: {e}")

        payload: Any = resp
        if isinstance(resp, dict):
            payload = resp.get("data", resp.get("positions", resp.get("netPositions", [])))

        if not isinstance(payload, list):
            payload = []

        rows: list[Dict[str, Any]] = []

        for row in payload:
            if not isinstance(row, dict):
                continue

            symbol = self._pick(row, "tradingSymbol", "tradingsymbol", "symbol", "securityName", "displayName") or ""
            product = self._pick(row, "productType", "product", "product_type") or ""
            exchange_seg = self._pick(row, "exchangeSegment", "exchange_seg", "segment") or ""

            # ids & numbers
            security_id = self._pick(row, "securityId", "security_id", "id")
            try:
                security_id = int(security_id) if security_id is not None else None
            except Exception:
                security_id = None

            net_qty_val = self._as_float(self._pick(row, "netQty", "netQuantity", "qty", "netqty")) or 0.0
            qty_int = int(net_qty_val) if net_qty_val == int(net_qty_val) else int(net_qty_val)

            buy_qty = self._as_float(self._pick(row, "buyQty", "dayBuyQty", "carryForwardBuyQty"))
            sell_qty = self._as_float(self._pick(row, "sellQty", "daySellQty", "carryForwardSellQty"))

            cost_price = self._as_float(self._pick(row, "costPrice", "price"))
            buy_avg    = self._as_float(self._pick(row, "buyAvg", "buyAveragePrice"))
            sell_avg   = self._as_float(self._pick(row, "sellAvg", "sellAveragePrice"))
            realized   = self._as_float(self._pick(row, "realizedProfit", "realisedProfit"))
            unrealized = self._as_float(self._pick(row, "unrealizedProfit", "unrealisedProfit", "mtm"))
            ltp_value  = self._as_float(self._pick(row, "ltp", "lastPrice", "last_price", "closePrice"))

            if qty_int > 0:
                side = "BUY"
            elif qty_int < 0:
                side = "SELL"
            else:
                side = "FLAT"

            if qty_int != 0 and unrealized is not None:
                pnl = unrealized
            elif qty_int == 0 and realized is not None:
                pnl = realized
            else:
                pnl = realized if realized is not None else unrealized

            rows.append(
                {
                    "symbol": symbol,
                    "product": product,
                    "qty": qty_int,
                    "net_qty": qty_int,
                    "buy_qty": int(buy_qty) if buy_qty is not None else None,
                    "sell_qty": int(sell_qty) if sell_qty is not None else None,
                    "avg_price": cost_price,
                    "cost_price": cost_price,
                    "ltp": ltp_value,
                    "pnl": pnl,
                    "exchange_seg": exchange_seg,
                    "security_id": security_id,
                    "side": side,
                    "position_type": str(self._pick(row, "positionType", "side")).upper() if self._pick(row, "positionType", "side") else side,
                    "buy_avg": buy_avg,
                    "sell_avg": sell_avg,
                    "realized_profit": realized,
                    "unrealized_profit": unrealized,
                    "_raw": row,
                }
            )

        self.log.info(f"[Positions] normalized {len(rows)} rows (with side & pnl rule)")
        return rows

    def get_positions_live_with_ltp(self) -> list[Dict[str, Any]]:
        """
        Same as get_positions_live, but merges LTP via /marketfeed/ltp
        and fills 'ltp' and 'pnl' (pnl uses Dhan 'unrealizedProfit' when available).
        """
        rows = self.get_positions_live()

        # Build (seg,id) pairs for bulk LTP
        pairs: list[tuple[str, int]] = []
        for r in rows:
            seg = r.get("exchange_seg")
            sid = r.get("security_id")
            if isinstance(seg, str) and isinstance(sid, int):
                pairs.append((seg, sid))

        ltps = self.get_ltp_bulk(pairs) if pairs else {}

        for r in rows:
            seg = r.get("exchange_seg")
            sid = r.get("security_id")
            ltp_live = ltps.get((seg, sid)) if seg and sid else None

            raw = r.get("_raw", {})
            position_type = str(raw.get("positionType") or r.get("position_type") or r.get("side") or "").upper()
            net_qty = r.get("qty") or 0
            cost_price = self._as_float(raw.get("costPrice")) or self._as_float(r.get("avg_price"))
            buy_avg = self._as_float(raw.get("buyAvg"))
            sell_avg = self._as_float(raw.get("sellAvg"))
            buy_qty_f = self._as_float(raw.get("buyQty"))
            sell_qty_f = self._as_float(raw.get("sellQty"))
            buy_qty = int(buy_qty_f) if buy_qty_f is not None else None
            sell_qty = int(sell_qty_f) if sell_qty_f is not None else None
            unrealized = self._as_float(raw.get("unrealizedProfit"))
            realized = self._as_float(raw.get("realizedProfit"))
            derived_ltp = None
            if ltp_live is None and unrealized is not None and net_qty:
                qty_abs = abs(net_qty)
                if qty_abs > 0:
                    if net_qty > 0 and buy_avg is not None:
                        derived_ltp = buy_avg + (unrealized / qty_abs)
                    elif net_qty < 0 and sell_avg is not None:
                        derived_ltp = sell_avg - (unrealized / qty_abs)

            if position_type == "LONG" or (isinstance(net_qty, int) and net_qty > 0):
                qty_disp = buy_qty if buy_qty is not None else net_qty
                candidate_ltp = ltp_live if ltp_live is not None else derived_ltp
                ltp_val = candidate_ltp if candidate_ltp is not None else raw.get("ltp") or raw.get("lastPrice") or buy_avg or cost_price
                ltp_val = self._as_float(ltp_val) or self._as_float(cost_price)
                avg_val = buy_avg if buy_avg is not None else cost_price
                pnl_val = unrealized if unrealized is not None else r.get("pnl") or 0.0
                r["qty"] = int(qty_disp or 0)
                r["ltp"] = ltp_val
                r["pnl"] = pnl_val
                r["avg_price"] = avg_val
            elif position_type == "SHORT" or (isinstance(net_qty, int) and net_qty < 0):
                qty_disp = -(sell_qty if sell_qty is not None else abs(net_qty))
                candidate_ltp = ltp_live if ltp_live is not None else derived_ltp
                ltp_val = candidate_ltp if candidate_ltp is not None else raw.get("ltp") or raw.get("lastPrice") or sell_avg or cost_price
                ltp_val = self._as_float(ltp_val) or self._as_float(cost_price)
                avg_val = sell_avg if sell_avg is not None else cost_price
                pnl_val = unrealized if unrealized is not None else r.get("pnl") or 0.0
                r["qty"] = int(qty_disp or 0)
                r["ltp"] = ltp_val
                r["pnl"] = pnl_val
                r["avg_price"] = avg_val
            else:
                qty_disp = sell_qty or buy_qty or net_qty
                ltp_val = ltp_live if ltp_live is not None else cost_price
                ltp_val = self._as_float(ltp_val) or self._as_float(cost_price)
                pnl_val = realized if realized is not None else r.get("pnl") or 0.0
                r["qty"] = int(qty_disp or 0)
                r["ltp"] = ltp_val
                r["pnl"] = pnl_val
                r["avg_price"] = cost_price

        self.log.info(f"[Positions] normalized {len(rows)} rows (with LTP merge)")
        return rows

    # -------- Background single-LTP poller (for NIFTY tile) --------

    def start_ltp_poller(
        self,
        exchange_seg: str,
        security_id: int,
        interval_sec: float,
        on_update: Callable[[float, datetime], None],
        poll_when_closed: bool = False,
    ) -> None:
        self.stop_ltp_poller()
        self._stop.clear()

        def _loop():
            while not self._stop.is_set():
                try:
                    if poll_when_closed or _is_india_market_open():
                        ltp = self.get_ltp_once(exchange_seg, security_id)
                        if ltp is not None:
                            on_update(ltp, _ist_now())
                except Exception as e:
                    self.log.exception(f"[LTP] poller: {e}")
                self._stop.wait(max(0.25, float(interval_sec)))

        self._thread = threading.Thread(target=_loop, daemon=True, name="DhanLTPPoller")
        self._thread.start()

    def stop_ltp_poller(self) -> None:
        """Stop the background LTP poller thread if running."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None


def get_shared_dhan_wrapper() -> "DhanWrapper":
    """
    Return a process-wide shared DhanWrapper so that option-chain throttling
    (1 call / 3s) is respected across agent + UI callers.
    """
    global _SHARED_WRAPPER
    if _SHARED_WRAPPER is not None:
        return _SHARED_WRAPPER
    with _SHARED_WRAPPER_LOCK:
        if _SHARED_WRAPPER is None:
            _SHARED_WRAPPER = DhanWrapper()
    return _SHARED_WRAPPER
