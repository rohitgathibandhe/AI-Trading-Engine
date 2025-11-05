# market_ai/modules/data_fetch/live_option_chain.py
from __future__ import annotations

import os, time, logging, json
from datetime import datetime
from typing import Dict, Any, Optional, Iterable, List

import requests

LOG = logging.getLogger(__name__)
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

class LiveOptionChainMarket:
    BASE = os.environ.get("DHAN_API_BASE", "https://api.dhan.co").rstrip("/")
    PATH_OPTIONCHAIN = "/v2/optionchain"
    PATH_EXPIRYLIST  = "/v2/optionchain/expirylist"

    def __init__(self, client_id: Optional[str], access_token: Optional[str],
                 underlying_seg: str = "IDX_I", rate_sleep_s: float = 3.0, timeout_s: int = 15):
        self.client_id = client_id or os.environ.get("DHAN_CLIENT_ID")
        self.token     = access_token or os.environ.get("DHAN_ACCESS_TOKEN")
        if not self.client_id or not self.token:
            raise RuntimeError("DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN missing")
        self.underlying_seg = underlying_seg
        self.rate_sleep_s   = rate_sleep_s
        self.timeout_s      = timeout_s
        self._expiry_cache: Dict[int, str] = {}
        self._last_good_chain: Optional[Dict[str, Any]] = None

    # ---------- HTTP ----------
    def _headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "client-id": self.client_id,
            "access-token": self.token,
        }

    def _post_json(self, path: str, body: Dict[str, Any]) -> Any:
        url = f"{self.BASE}{path}"
        r = requests.post(url, headers=self._headers(), json=body, timeout=self.timeout_s)
        r.raise_for_status()
        try:
            return r.json()
        except Exception:
            LOG.error("Non-JSON response from %s: %s", path, r.text[:400])
            raise

    # ---------- utils ----------
    @staticmethod
    def _num(x) -> Optional[float]:
        try:
            return float(x) if x is not None else None
        except Exception:
            return None

    @staticmethod
    def _first(*vals, default=None):
        for v in vals:
            if v is not None:
                return v
        return default

    # ---------- Expiry ----------
    def _choose_expiry(self, underlying_scrip: int, desired_expiry: str) -> str:
        cached = self._expiry_cache.get(underlying_scrip)
        if cached:
            return cached

        body = {"UnderlyingScrip": int(underlying_scrip), "UnderlyingSeg": self.underlying_seg}
        js = self._post_json(self.PATH_EXPIRYLIST, body)
        time.sleep(self.rate_sleep_s)

        exps: List[str] = []
        if isinstance(js, list):
            exps = [str(x) for x in js]
        elif isinstance(js, dict):
            data = js.get("data")
            if isinstance(data, list):
                exps = [str(x) for x in data]
            elif isinstance(data, dict):
                if isinstance(data.get("items"), list):
                    exps = [str(x) for x in data["items"]]
                else:
                    exps = [str(k) for k in data.keys()]
        exps = [e for e in exps if e]
        if not exps:
            raise RuntimeError("Empty expiry list from Dhan.")

        if desired_expiry in exps:
            chosen = desired_expiry
        else:
            try:
                target = datetime.strptime(desired_expiry, "%Y-%m-%d").date()
                parsed = [datetime.strptime(e, "%Y-%m-%d").date() for e in exps]
                future = [d for d in parsed if d >= target]
                pick   = min(future) if future else max(parsed)
                chosen = pick.strftime("%Y-%m-%d")
                LOG.warning("Expiry %s not in list; using %s.", desired_expiry, chosen)
            except Exception:
                chosen = exps[0]
                LOG.warning("Expiry parse failed; using first: %s.", chosen)

        self._expiry_cache[underlying_scrip] = chosen
        return chosen

    # ---------- Parsers for two canonical shapes ----------
    def _parse_shape_oc_dict(self, js: dict) -> Dict[str, Any]:
        """
        Shape like:
          {"data": {
             "last_price": 25836.2,
             "oc": {
               "19100.000000": {"ce": {...}, "pe": {...}}, ...
             }
          }}
        Here the strike is the KEY of the 'oc' dict.
        """
        data = js.get("data") or {}
        oc   = data.get("oc")
        if not isinstance(oc, dict):
            return {}

        spot_val = self._num(self._first(
            data.get("spot"), data.get("last_price"), data.get("underlyingValue")
        ))
        chain: Dict[str, Dict[str, Any]] = {}

        for k_str, node in oc.items():
            # strike from KEY
            k_num = self._num(k_str)
            if k_num is None:
                try:
                    k_num = self._num(str(k_str))
                except Exception:
                    continue
            strike_key = str(int(round(k_num)))

            ce = node.get("ce") or node.get("CE") or {}
            pe = node.get("pe") or node.get("PE") or {}

            def last_px(n: Dict[str,Any]) -> float:
                v = self._first(n.get("last_price"), n.get("lastPrice"), n.get("ltp"), n.get("LTP"),
                                n.get("close"), n.get("Close"), 0.0)
                return float(self._num(v) or 0.0)

            def get_delta(n: Dict[str,Any]) -> Optional[float]:
                g = n.get("greeks") or n.get("Greeks") or {}
                d = self._first(g.get("delta"), n.get("delta"))
                return self._num(d)

            def get_iv(n: Dict[str,Any]) -> Optional[float]:
                # accept snake_case too
                return self._num(self._first(n.get("iv"), n.get("impliedVolatility"), n.get("implied_volatility"), n.get("IV")))

            chain[strike_key] = {
                "ce": {"last_price": last_px(ce), "iv": get_iv(ce), "greeks": {"delta": get_delta(ce)}, "spot": spot_val},
                "pe": {"last_price": last_px(pe), "iv": get_iv(pe), "greeks": {"delta": get_delta(pe)}, "spot": spot_val},
            }

        return chain

    def _parse_shape_row_list(self, js: dict | list) -> Dict[str, Any]:
        """
        Generic list/dict-of-rows shape where each row has a strike field.
        """
        # find candidate rows
        rows: List[dict] = []
        def looks_like_row(d: Any) -> bool:
            if not isinstance(d, dict):
                return False
            keys = {"strikePrice","strike_price","strike","k","StrikePrice","STRIKE_PRICE"}
            return any(k in d for k in keys)

        def walk(obj: Any, depth=0):
            if depth > 4:
                return
            if isinstance(obj, list):
                for it in obj:
                    if looks_like_row(it):
                        rows.append(it)
                    else:
                        walk(it, depth+1)
            elif isinstance(obj, dict):
                if looks_like_row(obj):
                    rows.append(obj)
                for v in obj.values():
                    if isinstance(v, (dict, list)):
                        walk(v, depth+1)

        walk(js)

        chain: Dict[str, Dict[str, Any]] = {}
        spot_val: Optional[float] = None

        for row in rows:
            sp = self._first(
                row.get("strikePrice"), row.get("strike_price"), row.get("strike"),
                row.get("k"), row.get("StrikePrice"), row.get("STRIKE_PRICE")
            )
            sp_f = self._num(sp)
            if sp_f is None:
                continue
            strike_key = str(int(round(sp_f)))

            # CE/PE
            ce = row.get("CE") or row.get("ce") or row.get("Call") or row.get("call") or {}
            pe = row.get("PE") or row.get("pe") or row.get("Put")  or row.get("put")  or {}

            def last_px(n: Dict[str,Any]) -> float:
                v = self._first(n.get("last_price"), n.get("lastPrice"), n.get("ltp"), n.get("LTP"),
                                n.get("close"), n.get("Close"), 0.0)
                return float(self._num(v) or 0.0)

            def get_delta(n: Dict[str,Any]) -> Optional[float]:
                g = n.get("greeks") or n.get("Greeks") or {}
                d = self._first(g.get("delta"), n.get("delta"))
                return self._num(d)

            def get_iv(n: Dict[str,Any]) -> Optional[float]:
                return self._num(self._first(n.get("iv"), n.get("impliedVolatility"), n.get("implied_volatility"), n.get("IV")))

            if spot_val is None:
                s = self._first(row.get("spot"), row.get("underlyingValue"),
                                ce.get("spot"), pe.get("spot"),
                                row.get("Spot"), row.get("UNDERLYING_VALUE"))
                spot_val = self._num(s) or spot_val

            chain[strike_key] = {
                "ce": {"last_price": last_px(ce), "iv": get_iv(ce), "greeks": {"delta": get_delta(ce)}, "spot": spot_val},
                "pe": {"last_price": last_px(pe), "iv": get_iv(pe), "greeks": {"delta": get_delta(pe)}, "spot": spot_val},
            }
        return chain

    # ---------- Main ----------
    def get_live_chain(self, underlying_scrip: int, desired_expiry: str) -> Dict[str, Any]:
        expiry = self._choose_expiry(underlying_scrip, desired_expiry)

        body = {"UnderlyingScrip": int(underlying_scrip), "UnderlyingSeg": self.underlying_seg, "Expiry": expiry}
        js = self._post_json(self.PATH_OPTIONCHAIN, body)
        time.sleep(self.rate_sleep_s)

        # Try shape: data.oc (dict keyed by strike)
        chain: Dict[str, Any] = {}
        if isinstance(js, dict) and isinstance(js.get("data"), dict) and isinstance(js["data"].get("oc"), dict):
            chain = self._parse_shape_oc_dict(js)

        # Else try generic row-list shape
        if not chain:
            chain = self._parse_shape_row_list(js if isinstance(js, (dict, list)) else {"data": js})

        strikes = len(chain)
        if strikes == 0:
            # Preview to aid debugging
            try:
                preview = json.dumps(js if isinstance(js, (dict, list)) else {"type": type(js).__name__}, default=str)[:400]
            except Exception:
                preview = f"<unserializable {type(js).__name__}>"
            LOG.warning("Optionchain returned no parseable rows; preview=%s", preview)
            if self._last_good_chain:
                LOG.warning("Using last good chain fallback.")
                return self._last_good_chain
            raise RuntimeError("Parsed 0 strikes from option chain (transient or schema shift).")

        self._last_good_chain = chain
        # get a spot (any row)
        any_row = next(iter(chain.values()))
        spot = any_row["ce"]["spot"] or any_row["pe"]["spot"] or "NA"
        LOG.info("Live chain OK | expiry=%s | strikes=%d | spot≈%s",
                 expiry, strikes, f"{float(spot):.2f}" if isinstance(spot,(int,float)) else str(spot))
        return chain
