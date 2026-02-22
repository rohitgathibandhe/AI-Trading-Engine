# market_ai/modules/data_fetch/dhan_option_chain.py
from __future__ import annotations
from typing import Dict, Any, List, Optional, Tuple
from datetime import date, datetime
import logging, time
import requests

from market_ai.modules.data_fetch.dhan_api import (
    SimpleDhanClient,
    get_expiry_list_for_underlying,
    get_option_chain_for,
    DhanError,
)

LOG = logging.getLogger(__name__)
if not LOG.handlers:
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

def _is_iso_date(s: str) -> bool:
    try:
        datetime.strptime(s, "%Y-%m-%d"); return True
    except Exception:
        return False

def _normalize(oc: Any) -> Dict[str, Any]:
    if isinstance(oc, dict):
        vals = list(oc.values())[:1]
        if vals and isinstance(vals[0], dict) and ("ce" in vals[0] or "pe" in vals[0]):
            return oc
        return oc
    if isinstance(oc, list):
        out: Dict[str, Any] = {}
        for row in oc:
            try:
                strike = row.get("strike") or row.get("StrikePrice") or row.get("strikePrice")
                ce = row.get("ce"); pe = row.get("pe")
                if strike is None:
                    ce_s = (ce or {}).get("strike"); pe_s = (pe or {}).get("strike")
                    strike = ce_s if ce_s is not None else pe_s
                if strike is None: continue
                out[str(float(strike))] = {"ce": ce, "pe": pe}
            except Exception:
                continue
        return out
    return {}


class RateLimitedOptionChain:
    """
    Lightweight shared client that respects Dhan's 1 call / 3s limit for /v2/optionchain.
    Caches recent responses per (UnderlyingScrip, Expiry) for min_interval_sec.
    """

    def __init__(self, base_url: str, client_id: str, access_token: str):
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.access_token = access_token
        self.min_interval_sec: float = 3.0  # as confirmed by Dhan support
        # (UnderlyingScrip, Expiry) -> (last_ts, response)
        self._cache: Dict[Tuple[int, str], Tuple[float, dict]] = {}
        self._last_global_call_ts: float = 0.0
        self.log = LOG

    def _headers(self) -> dict:
        return {
            "access-token": self.access_token,
            "client-id": self.client_id,
            "Content-Type": "application/json",
        }

    def get_option_chain(self, underlying_scrip: int, expiry: str, underlying_seg: str = "IDX_I") -> dict:
        key = (underlying_scrip, expiry)
        now = time.monotonic()

        # Return cached value if fresh
        if key in self._cache:
            ts, data = self._cache[key]
            if now - ts < self.min_interval_sec:
                age = now - ts
                self.log.info("OC CACHE HIT: underlying=%s expiry=%s age=%.2fs", underlying_scrip, expiry, age)
                return data

        # Enforce global min interval between real calls
        elapsed = now - self._last_global_call_ts
        if elapsed < self.min_interval_sec:
            # If we have cache, use it; else wait remaining
            if key in self._cache:
                ts, data = self._cache[key]
                age = now - ts
                self.log.info("OC CACHE HIT (global throttle): underlying=%s expiry=%s age=%.2fs", underlying_scrip, expiry, age)
                return data
            sleep_s = self.min_interval_sec - elapsed
            self.log.debug("OC throttling %.2fs before real call", sleep_s)
            time.sleep(max(0, sleep_s))

        url = f"{self.base_url}/v2/optionchain"
        payload = {
            "UnderlyingScrip": underlying_scrip,
            "UnderlyingSeg": underlying_seg,
            "Expiry": expiry,
        }

        def _do_call() -> requests.Response:
            return requests.post(url, headers=self._headers(), json=payload, timeout=5)

        resp = _do_call()
        if resp.status_code == 429:
            self.log.warning("OC 429 received; sleeping %.1fs and retrying once", self.min_interval_sec)
            time.sleep(self.min_interval_sec)
            resp = _do_call()
        resp.raise_for_status()

        data = resp.json()
        now = time.monotonic()
        self._last_global_call_ts = now
        self._cache[key] = (now, data)
        self.log.info("OC API CALL: underlying=%s expiry=%s (fresh)", underlying_scrip, expiry)
        return data

class OptionChain:
    """LIVE OC via DHAN (for *current/future* expiries). Not suitable for history."""
    def __init__(self, client: SimpleDhanClient, default_seg: str = "IDX_I"):
        self.client = client
        self.default_seg = default_seg
        self._expiry_cache: Dict[str, List[str]] = {}

    def get_option_chain(self, underlying_id: int, expiry_or_tag: Optional[str], underlying_seg: Optional[str] = None) -> Dict[str, Any]:
        seg = (underlying_seg or self.default_seg) or "IDX_I"
        tag = expiry_or_tag or ""

        # 0) No tag/expiry provided → default to first future expiry (front)
        if not tag:
            exps = self._expiries(underlying_id, seg)
            pick = _pick_first_future(exps)
            if not pick:
                return {}
            try:
                oc_raw = get_option_chain_for(self.client, underlying_id, expiry=pick, underlying_seg=seg)
                return _normalize(oc_raw)
            except DhanError as e:
                LOG.warning("DHAN front expiry %s failed: %s", pick, e)
                return {}

        # 1) If explicit ISO date, call exactly that expiry once. Do NOT remap to "next".
        if _is_iso_date(tag):
            try:
                oc_raw = get_option_chain_for(self.client, underlying_id, expiry=tag, underlying_seg=seg)
                return _normalize(oc_raw)
            except DhanError as e:
                LOG.warning("DHAN could not serve explicit expiry %s on %s: %s", tag, seg, e)
                return {}

        # 2) If tag 'weekly', resolve the next weekly from DHAN expiry list (live only).
        if tag.lower() == "weekly":
            exps = self._expiries(underlying_id, seg)
            pick = _pick_first_future(exps)
            if not pick:
                return {}
            try:
                oc_raw = get_option_chain_for(self.client, underlying_id, expiry=pick, underlying_seg=seg)
                return _normalize(oc_raw)
            except DhanError as e:
                LOG.warning("DHAN weekly %s failed: %s", pick, e)
                return {}

        # Unknown tag → return empty
        return {}

    def _expiries(self, underlying_id: int, seg: str) -> List[str]:
        key = f"{underlying_id}:{seg}"
        if key in self._expiry_cache:
            return self._expiry_cache[key]
        exps = get_expiry_list_for_underlying(self.client, underlying_id, seg)
        self._expiry_cache[key] = exps or []
        return self._expiry_cache[key]

def _pick_first_future(expiries: List[str]) -> Optional[str]:
    if not expiries: return None
    today = date.today()
    fut = []
    for e in expiries:
        try:
            d = datetime.strptime(e, "%Y-%m-%d").date()
            if d >= today: fut.append(d)
        except Exception:
            continue
    if not fut: return None
    return sorted(fut)[0].isoformat()

# ---------- Parsing helpers ----------

def _num(val) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        return None

def _first(*vals):
    for v in vals:
        if v is not None:
            return v
    return None

def parse_option_chain_rows(oc_resp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flatten the Dhan optionchain response into a list of rows (one per CE/PE per strike).

    Expected shapes we have seen:
      {"data": {
          "last_price": 24964.25,
          "oc": {
              "25000.000000": {"ce": {...}, "pe": {...}},
              ...
          }
       }, "status": "success"}

    Returns list of dicts with keys:
      strike, option_type, last_price, oi, volume, implied_volatility,
      delta, theta, gamma, vega, raw
    """
    if not isinstance(oc_resp, dict):
        return []

    data = oc_resp.get("data") or oc_resp
    oc = data.get("oc") if isinstance(data, dict) else None
    if not isinstance(oc, dict):
        # fall back: maybe response IS the oc dict already
        if isinstance(oc_resp.get("oc"), dict):
            oc = oc_resp["oc"]
        else:
            return []

    rows: List[Dict[str, Any]] = []

    for strike_key, node in oc.items():
        strike_f = None
        try:
            strike_f = float(str(strike_key))
        except Exception:
            continue
        for opt_field, opt_type in (("ce", "CE"), ("pe", "PE")):
            leg = node.get(opt_field) or {}
            greeks = leg.get("greeks") or leg.get("Greeks") or {}
            row = {
                "strike": strike_f,
                "option_type": opt_type,
                "last_price": _num(_first(
                    leg.get("last_price"), leg.get("lastPrice"),
                    leg.get("ltp"), leg.get("LTP"),
                )),
                "security_id": leg.get("securityId") or leg.get("security_id") or leg.get("instrumentId"),
                "symbol": leg.get("trading_symbol") or leg.get("tradingSymbol") or leg.get("symbol") or leg.get("instrument"),
                "oi": _num(leg.get("oi") or leg.get("open_interest")),
                "volume": _num(leg.get("volume")),
                "implied_volatility": _num(_first(
                    leg.get("implied_volatility"), leg.get("impliedVolatility"), leg.get("iv"),
                )),
                "delta": _num(_first(greeks.get("delta"), leg.get("delta"))),
                "theta": _num(greeks.get("theta")),
                "gamma": _num(greeks.get("gamma")),
                "vega": _num(greeks.get("vega")),
                "raw": leg,
            }
            rows.append(row)
    return rows
