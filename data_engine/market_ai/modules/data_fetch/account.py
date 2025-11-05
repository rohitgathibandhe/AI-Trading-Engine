# market_ai/modules/data_fetch/account.py
from __future__ import annotations
import os
import time
import logging
import json
import requests
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class AccountFetcher:
    """
    Fetch account / margin details from Dhan.

    - Reads config/broker_config.yaml or environment variables
    - Probes a short list of candidate endpoint paths
    - Caches result for cache_ttl seconds
    - Normalizes to keys:
        available_margin, used_margin, collateral, cash_balance, raw
    - Tolerant to minor misspellings (e.g. "availabelBalance")
    """

    _CANDIDATE_ENDPOINTS = [
        "/v2/fundlimit",
        "/fundlimit",
        "/v2/fundlimit/",
        "/v2/account/funds",
        "/funds",
    ]

    def __init__(self, config_path: str = "config/broker_config.yaml"):
        try:
            import yaml  # type: ignore
        except Exception:
            yaml = None  # type: ignore

        cfg: Dict[str, Any] = {}
        if yaml and os.path.exists(config_path):
            try:
                with open(config_path, "r") as fh:
                    cfg = yaml.safe_load(fh) or {}
            except Exception:
                LOG.debug("Failed to parse broker config (ignoring)", exc_info=True)

        self.access_token = cfg.get("access_token") or os.environ.get("DHAN_ACCESS_TOKEN")
        self.client_id = cfg.get("client_id") or os.environ.get("DHAN_CLIENT_ID")
        self.base_url = (cfg.get("api_base") or cfg.get("api_base_url") or os.environ.get("DHAN_API_BASE") or "https://api.dhan.co").rstrip("/")

        self.headers = {
            "Content-Type": "application/json",
            "access-token": self.access_token or "",
            "client-id": str(self.client_id or ""),
        }

        self.cache_ttl = int(cfg.get("cache_ttl_sec", 10))
        self.retry_delay = float(cfg.get("retry_delay_sec", 1.0))
        self.max_retries = int(cfg.get("max_retries", 3))

        self._last_fetch: Optional[datetime] = None
        self._cached: Optional[Dict[str, Any]] = None

    def _join(self, path: str) -> str:
        b = self.base_url.rstrip("/")
        if b.endswith("/v2") and path.startswith("/v2/"):
            path = path.replace("/v2/", "/", 1)
        if not path.startswith("/"):
            path = "/" + path
        return f"{b}{path}"

    def get_funds_raw(self, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        if not force_refresh and self._cached:
            if self._last_fetch and (datetime.now() - self._last_fetch) < timedelta(seconds=self.cache_ttl):
                LOG.debug("Returning cached funds (age %.1fs)", (datetime.now() - self._last_fetch).total_seconds())
                return self._cached

        for ep in self._CANDIDATE_ENDPOINTS:
            url = self._join(ep)
            LOG.debug("Attempt GET %s", url)
            for attempt in range(1, self.max_retries + 1):
                try:
                    r = requests.get(url, headers=self.headers, timeout=10)
                    LOG.debug("HTTP %s -> %s", url, r.status_code)
                    if r.status_code == 200:
                        try:
                            js = r.json()
                        except Exception:
                            LOG.debug("Non-JSON response from %s (first 300 chars): %s", url, r.text[:300])
                            js = None
                        if isinstance(js, dict):
                            LOG.debug("Fetched funds keys: %s", list(js.keys()))
                            self._cached = js
                            self._last_fetch = datetime.now()
                            return js
                        else:
                            LOG.debug("GET %s returned JSON but not dict (skipping)", url)
                    elif r.status_code in (401, 403):
                        LOG.warning("Auth failed for %s (status=%s) - check token/client-id", url, r.status_code)
                        return None
                    else:
                        LOG.debug("Endpoint %s returned status %s", url, r.status_code)
                except Exception as e:
                    LOG.debug("Request exception to %s attempt %d: %s", url, attempt, e)
                time.sleep(self.retry_delay * attempt)

        LOG.warning("No endpoint returned funds data")
        return None

    def get_margin_breakdown(self, force_refresh: bool = False) -> Dict[str, float]:
        raw = self.get_funds_raw(force_refresh=force_refresh) or {}

        if isinstance(raw, dict) and "data" in raw and isinstance(raw["data"], dict):
            d = raw["data"]
        else:
            d = raw if isinstance(raw, dict) else {}

        LOG.debug("Raw funds inspect keys: %s", list(d.keys()) if isinstance(d, dict) else "<not-dict>")

        def _as_float(x: Any) -> float:
            try:
                if x is None:
                    return 0.0
                if isinstance(x, (int, float)):
                    return float(x)
                if isinstance(x, str):
                    s = x.replace(",", "").strip()
                    if s == "":
                        return 0.0
                    return float(s)
                return float(x)
            except Exception:
                return 0.0

        def _find_exact(keys):
            if isinstance(d, dict):
                for k in keys:
                    if k in d and d.get(k) is not None:
                        return d.get(k)
                fd = d.get("funds")
                if isinstance(fd, dict):
                    for k in keys:
                        if k in fd and fd.get(k) is not None:
                            return fd.get(k)
            if isinstance(raw, dict):
                for k in keys:
                    if k in raw and raw.get(k) is not None:
                        return raw.get(k)
            return None

        def _find_loose(keys, substrings):
            """
            If exact keys fail, look for keys in the raw dict where the key name
            contains one of the substrings (case-insensitive). This catches
            small misspellings like "availabelBalance".
            """
            if not isinstance(raw, dict) and not isinstance(d, dict):
                return None
            # check d first
            for src in (d, raw):
                if not isinstance(src, dict):
                    continue
                for actual in src.keys():
                    low = actual.lower()
                    for s in substrings:
                        if s in low:
                            val = src.get(actual)
                            if val is not None:
                                LOG.debug("Loose match: using raw key '%s' for substring '%s'", actual, s)
                                return val
            return None

        # candidate lists
        avail_keys = ["availableMargin", "available_margin", "availableBalance", "availableCash", "cashBalance", "availableCashBalance"]
        used_keys = ["utilizedMargin", "used_margin", "utilizedAmount", "usedAmount"]
        coll_keys = ["collateral", "collateralAmount", "collateralValue"]
        cash_keys = ["cashBalance", "cash", "availableBalance", "cash_balance", "withdrawableBalance"]

        # exact search
        avail_val = _find_exact(avail_keys)
        used_val = _find_exact(used_keys)
        coll_val = _find_exact(coll_keys)
        cash_val = _find_exact(cash_keys)

        # If exact not found, try loose substring-based matching (handles misspellings)
        if avail_val is None:
            avail_val = _find_loose(avail_keys, ["avail", "available", "availabel"])
        if used_val is None:
            used_val = _find_loose(used_keys, ["utiliz", "used", "utilized"])
        if coll_val is None:
            coll_val = _find_loose(coll_keys, ["collat", "collateral"])
        if cash_val is None:
            cash_val = _find_loose(cash_keys, ["cash", "withdraw"])

        LOG.debug("Found candidates -> avail_val: %r, used_val: %r, coll_val: %r, cash_val: %r", avail_val, used_val, coll_val, cash_val)

        avail_f = _as_float(avail_val)
        used_f = _as_float(used_val)
        coll_f = _as_float(coll_val)
        cash_f = _as_float(cash_val)

        # If explicit availableMargin missing, try direct availableBalance usage or compute it
        if avail_f == 0.0:
            # if there is any available-like raw key, use it directly
            alt = _find_exact(["availableBalance", "available_balance"]) or _find_loose([], ["avail", "available", "availabel"])
            if alt is not None:
                alt_f = _as_float(alt)
                LOG.debug("available-like alt present -> %r (float %.2f)", alt, alt_f)
                if used_f > 0:
                    # prefer computing available = availableBalance - used
                    computed = max(0.0, alt_f - used_f)
                    LOG.debug("Computed available_margin = availableBalance - utilizedAmount = %.2f - %.2f = %.2f", alt_f, used_f, computed)
                    avail_f = computed
                else:
                    LOG.debug("Using availableBalance directly as available_margin = %.2f", alt_f)
                    avail_f = alt_f

        # fallback: if still zero but cash balance present, derive from cash - used
        if avail_f == 0.0 and cash_f > 0:
            computed2 = max(0.0, cash_f - used_f)
            LOG.debug("Fallback computed available_margin = cash_balance - used_margin = %.2f - %.2f = %.2f", cash_f, used_f, computed2)
            avail_f = computed2

        normalized = {
            "available_margin": float(avail_f),
            "used_margin": float(used_f),
            "collateral": float(coll_f),
            "cash_balance": float(cash_f),
            "raw": raw,
        }

        LOG.debug("Normalized margin breakdown: %s", normalized)
        return normalized

    def get_available_margin(self, force_refresh: bool = False) -> float:
        return self.get_margin_breakdown(force_refresh).get("available_margin", 0.0)

    def get_net_funds(self, force_refresh: bool = False) -> float:
        mb = self.get_margin_breakdown(force_refresh)
        return float(mb.get("cash_balance", 0.0) + mb.get("collateral", 0.0))

    def pretty_print_margin(self):
        mb = self.get_margin_breakdown(force_refresh=True)
        print("\n💰 Margin Snapshot:")
        for k, v in mb.items():
            if k == "raw":
                continue
            print(f"  {k:18s} : ₹{v:,.2f}")
        print("-" * 40)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    a = AccountFetcher()
    a.pretty_print_margin()
