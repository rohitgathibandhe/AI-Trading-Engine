# market_ai/modules/data_fetch/rolling_expired_options.py
from __future__ import annotations

import logging
import math
import os
import time
from datetime import datetime, timedelta, date
from typing import Any, Dict, List, Optional, Tuple

import requests

LOG = logging.getLogger(__name__)
if not LOG.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )

# -------------------- math helpers --------------------
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_delta(spot: float, strike: float, t_years: float, iv: float, r: float, call: bool) -> float:
    """Black–Scholes delta. iv/r are annualized decimals. + for CALL, negative for PUT."""
    try:
        if spot <= 0 or strike <= 0 or t_years <= 0 or iv <= 0:
            return 0.0
        d1 = (math.log(spot / strike) + (r + 0.5 * iv * iv) * t_years) / (iv * math.sqrt(t_years))
        cdf = _norm_cdf(d1)
        return cdf if call else (cdf - 1.0)
    except Exception:
        return 0.0

# -------------------- time helpers --------------------
def _parse_iso(d: str) -> date:
    return datetime.strptime(d, "%Y-%m-%d").date()

def _choose_last_idx_for_day(ts: List[int], target_day: date) -> Optional[int]:
    last = None
    for i, t in enumerate(ts or []):
        if datetime.utcfromtimestamp(int(t)).date() == target_day:
            last = i
    return last

def _expiry_code(as_of: date, expiry: date) -> int:
    if as_of.year == expiry.year and as_of.month == expiry.month:
        return 0
    diff = (expiry.year - as_of.year) * 12 + (expiry.month - as_of.month)
    return 1 if diff == 1 else (2 if diff > 1 else 0)

# -------------------- adapter --------------------
class RollingExpiredOptionsMarket:
    """
    DHAN /v2/charts/rollingoption adapter (bounded, laddered).

    • CALLs: fixed ladder ATM+[k0, k0+2, k0+4, k0+6]
      PUTs : fixed ladder ATM-[k0, k0+2, k0+4, k0+6]
      where k0 ≈ round(spot*seed_otm_pct/50) with min=2
    • Stops early if price <= min_premium or distance >= max_otm_pct
    • Uses fallback IV for delta so IV=None doesn’t stall
    • Ensures ≥3 unique strikes/side when possible
    """

    BASE = os.environ.get("DHAN_API_BASE", "https://api.dhan.co").rstrip("/")
    PATH = "/v2/charts/rollingoption"

    def __init__(
        self,
        client_id: Optional[str] = None,
        access_token: Optional[str] = None,
        exchange_segment: str = "NSE_FNO",
        instrument: str = "OPTIDX",
        expiry_flag: str = "MONTH",
        interval: str = "5",
        risk_free_rate: float = 0.06,
        throttle_s: float = 0.35,
        max_retries: int = 4,
        seed_otm_pct: float = 0.035,
        max_otm_pct: float = 0.060,
        min_premium: float = 10.0,
        max_requests: int = 8,
        iv_fallback: float = 0.20,
        neighborhood_span: int = 2,   # accepted for compatibility (not used by laddered plan)
        **_ignored                     # accept extra kwargs safely
    ):
        self.client_id = client_id or os.environ.get("DHAN_CLIENT_ID")
        self.access_token = access_token or os.environ.get("DHAN_ACCESS_TOKEN")
        if not self.client_id or not self.access_token:
            raise RuntimeError("Missing DHAN credentials (DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN).")
        self.exchange_segment = exchange_segment
        self.instrument = instrument
        self.expiry_flag = expiry_flag
        self.interval = interval
        self.r = float(risk_free_rate)
        self.throttle_s = float(throttle_s)
        self.max_retries = int(max_retries)
        self.seed_otm_pct = float(seed_otm_pct)
        self.max_otm_pct = float(max_otm_pct)
        self.min_premium = float(min_premium)
        self.max_requests = int(max_requests)
        self.iv_fallback = float(iv_fallback)
        self.neighborhood_span = int(neighborhood_span)

    # ---------- HTTP ----------
    def _headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": self.access_token,
            "client-id": self.client_id,
        }

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.BASE}{self.PATH}"
        for attempt in range(1, self.max_retries + 1):
            LOG.debug(
                "rollingoption payload: expiryFlag=%s expiryCode=%s strike=%s from=%s to=%s type=%s",
                payload.get("expiryFlag"),
                payload.get("expiryCode"),
                payload.get("strike"),
                payload.get("fromDate"),
                payload.get("toDate"),
                payload.get("drvOptionType"),
            )
            resp = requests.post(url, headers=self._headers(), json=payload, timeout=30)
            if resp.status_code in (429, 500, 502, 503, 504):
                wait = min(2.0 * attempt, 8.0)
                LOG.debug("rollingoption HTTP %s; retry in %.1fs", resp.status_code, wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, dict) and data.get("status") == "failed":
                raise RuntimeError(str(data)[:800])
            return data
        raise RuntimeError("rollingoption failed after retries")

    # ---------- core fetch ----------
    def _pull(self, security_id: int, selector: str, opt_type: str,
              from_date: str, to_date: str, expiry_code: int,
              want: List[str], target_day: date) -> Optional[Tuple[str, float, Dict[str, Any]]]:
        payload = {
            "exchangeSegment": self.exchange_segment,
            "interval": self.interval,
            "securityId": int(security_id),
            "instrument": self.instrument,
            "expiryFlag": self.expiry_flag,
            "expiryCode": int(expiry_code),
            "strike": selector,              # "ATM", "ATM+7", "ATM-12", ...
            "drvOptionType": opt_type,       # "CALL" / "PUT"
            "requiredData": want,
            "fromDate": from_date,
            "toDate": to_date
        }
        data = self._post(payload) or {}
        node = (data.get("data") or {}).get("ce" if opt_type == "CALL" else "pe") or {}
        ts = node.get("timestamp") or []
        idx = _choose_last_idx_for_day(ts, target_day)
        if idx is None:
            time.sleep(self.throttle_s); return None
        try:
            K = float(node["strike"][idx])
            close = float(node["close"][idx])
            iv = node.get("iv", [None])[idx]; iv = float(iv) if iv is not None else None
            spot = node.get("spot", [None])[idx]; spot = float(spot) if spot is not None else None
            oi = node.get("oi", [0.0])[idx] or 0.0
        except Exception:
            time.sleep(self.throttle_s); return None

        leg = {"last_price": close, "iv": iv, "oi": float(oi), "spot": spot, "strike": K}
        iv_eff = (iv / 100.0) if (iv is not None and iv > 1.0) else (iv if iv is not None else self.iv_fallback)
        if spot and K and iv_eff:
            d = bs_delta(float(spot), float(K), 1.0/12.0, float(iv_eff), self.r, call=(opt_type=="CALL"))
        else:
            d = None
        leg.setdefault("greeks", {})["delta"] = d
        time.sleep(self.throttle_s)
        return selector, K, leg

    def _ladder(self, spot: Optional[float]) -> List[int]:
        # assume 50pt spacing; seed ~3.5% OTM
        tick = 50.0
        k0 = 6 if spot is None else max(2, int((spot * self.seed_otm_pct) // tick))
        return [k0, k0 + 2, k0 + 4, k0 + 6]

    def _build_side(self, security_id: int, as_of_date: str, expiry_code: int, opt_type: str) -> List[Tuple[float, Dict[str, Any]]]:
        want = ["open","high","low","close","iv","volume","strike","oi","spot","timestamp"]
        day = _parse_iso(as_of_date); to_date = (day + timedelta(days=1)).isoformat()

        # 1) ATM to get spot
        at = self._pull(security_id, "ATM", opt_type, as_of_date, to_date, expiry_code, want, day)
        reqs = 1 if at else 0
        rows_by_selector: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        if at:
            rows_by_selector["ATM"] = (at[1], at[2])
        spot = at[2].get("spot") if at else None

        # 2) fixed small ladder
        ks = self._ladder(spot)
        for k in ks:
            if reqs >= self.max_requests: break
            sel = f"ATM+{k}" if opt_type=="CALL" else f"ATM-{k}"
            if sel in rows_by_selector: continue
            got = self._pull(security_id, sel, opt_type, as_of_date, to_date, expiry_code, want, day)
            reqs += 1
            if not got: continue
            _, K, leg = got
            price = float(leg["last_price"]); s = float(leg.get("spot") or 0.0)
            rows_by_selector[sel] = (K, leg)
            # early guards
            if price <= self.min_premium: break
            if s>0 and abs(K - s)/s >= self.max_otm_pct: break

        # ensure uniqueness by strike (server sometimes maps different selectors → same K)
        out_rows: List[Tuple[float, Dict[str, Any]]] = []
        seenK = set()
        for sel in rows_by_selector:
            K, leg = rows_by_selector[sel]
            if K in seenK: continue
            seenK.add(K); out_rows.append((K, leg))

        if len(out_rows) < 3:
            LOG.warning("Only %d unique %s strikes on %s; selectors=%s",
                        len(out_rows), opt_type, as_of_date, list(rows_by_selector.keys()))
        return out_rows

    # ---------- public ----------
    def get_option_chain(self, underlying_id: int, expiry_or_tag: str,
                         as_of_date: Optional[str] = None, underlying_seg: Optional[str] = None) -> Dict[str, Any]:
        if not as_of_date:
            raise RuntimeError("rolling market requires as_of_date=YYYY-MM-DD")

        LOG.info("RollingAdapter cfg: throttle=%.2fs, max_req=%d, seed=%.2f%%, min_px=%.2f, max_otm=%.1f%%, iv_fallback=%.0f%%",
                 self.throttle_s, self.max_requests, self.seed_otm_pct*100, self.min_premium,
                 self.max_otm_pct*100, self.iv_fallback*100)

        try:
            expiry_day = _parse_iso(expiry_or_tag)
        except Exception:
            expiry_day = _parse_iso(as_of_date) + timedelta(days=35)
        as_of = _parse_iso(as_of_date)
        exp_code = _expiry_code(as_of, expiry_day)

        ce_rows = self._build_side(int(underlying_id), as_of_date, exp_code, "CALL")
        pe_rows = self._build_side(int(underlying_id), as_of_date, exp_code, "PUT")

        out: Dict[str, Any] = {}
        for K, ce in ce_rows:
            out.setdefault(str(K), {})["ce"] = ce
        for K, pe in pe_rows:
            out.setdefault(str(K), {})["pe"] = pe

        LOG.info("Rolling OC built on %s for expiry=%s | strikes: CE=%d, PE=%d, total=%d",
                 as_of_date, expiry_or_tag, len(ce_rows), len(pe_rows), len(ce_rows)+len(pe_rows))
        return out
