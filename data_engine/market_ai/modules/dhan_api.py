# market_ai/modules/dhan_api.py
"""
Low-level Dhan API client wrapper.

Features:
 - Uses `requests` if available, falls back to `urllib3`.
 - Reads credentials from environment variables:
    DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN
 - Optional config via environment:
    DHAN_API_BASE (default "https://api.dhan.co")
    DHAN_REQUEST_TIMEOUT (seconds) default 10
    DHAN_CACHE_DIR (path) default .optionchain_cache
    DHAN_MIN_REQUEST_INTERVAL (float seconds) default 0.5
 - Automatic retry/backoff on 429/5xx with jitter.
 - Raises DhanError for non-transient errors (401/400).
 - Returns parsed JSON dicts (not raw responses).
"""

from __future__ import annotations
import os
import time
import json
import logging
import random
from typing import Any, Dict, Optional, Tuple

# add near top of market_ai/modules/data_fetch/dhan_api.py
import time
import random
import requests

def _request_with_backoff(method, url, headers=None, json=None, timeout=10.0, max_attempts=6):
    """
    Simple exponential backoff wrapper for requests; handles 429 + network errors.
    Returns requests.Response or raises RuntimeError after repeated failures.
    """
    attempt = 0
    backoff = 0.5
    while attempt < max_attempts:
        attempt += 1
        try:
            if method.lower() == "post":
                resp = requests.post(url, headers=headers, json=json, timeout=timeout)
            else:
                resp = requests.get(url, headers=headers, json=json, timeout=timeout)
        except requests.RequestException as e:
            # transient network error -> retry
            sleep = backoff + random.uniform(0, backoff * 0.1)
            time.sleep(sleep)
            backoff *= 2
            continue

        # If successful
        if resp.status_code == 200:
            return resp

        # 429 -> backoff then retry
        if resp.status_code == 429:
            sleep = backoff + random.uniform(0, backoff * 0.5)
            time.sleep(sleep)
            backoff *= 2
            attempt += 0  # attempt already incremented
            continue

        # For 400/4xx other than 429, return response (caller will inspect JSON and handle '813' Invalid Security)
        return resp

    raise RuntimeError(f"Request to {url} failed after {max_attempts} attempts")

LOG = logging.getLogger(__name__)
LOG.addHandler(logging.NullHandler())


class DhanError(Exception):
    """Non-retryable error from Dhan (e.g. auth failure, invalid request)."""


# read configuration from environment
_DHAN_BASE = os.environ.get("DHAN_API_BASE", "https://api.dhan.co").rstrip("/")
_DHAN_CLIENT_ID = os.environ.get("DHAN_CLIENT_ID", "")
_DHAN_ACCESS_TOKEN = os.environ.get("DHAN_ACCESS_TOKEN", "")
_REQUEST_TIMEOUT = float(os.environ.get("DHAN_REQUEST_TIMEOUT", "10.0"))
_CACHE_DIR = os.environ.get("DHAN_CACHE_DIR", ".optionchain_cache")
_MIN_REQUEST_INTERVAL = float(os.environ.get("DHAN_MIN_REQUEST_INTERVAL", "0.5"))
_MAX_RETRIES = int(os.environ.get("DHAN_MAX_RETRIES", "4"))
_BACKOFF_BASE = float(os.environ.get("DHAN_BACKOFF_BASE", "0.6"))  # seconds

# ensure cache dir exists
try:
    os.makedirs(_CACHE_DIR, exist_ok=True)
except Exception:
    # ignore if cannot make cache dir; caching becomes noop
    LOG.debug("Could not create cache dir %s (continuing without persistent cache)", _CACHE_DIR, exc_info=True)


# Small abstraction: prefer requests, fallback to urllib3
try:
    import requests  # type: ignore

    _HAS_REQUESTS = True
except Exception:
    _HAS_REQUESTS = False
    import urllib3  # type: ignore
    _HTTP_POOL = urllib3.PoolManager()


def _auth_headers() -> Dict[str, str]:
    headers = {
        "content-type": "application/json",
    }
    if _DHAN_ACCESS_TOKEN:
        headers["access-token"] = _DHAN_ACCESS_TOKEN
    if _DHAN_CLIENT_ID:
        headers["client-id"] = _DHAN_CLIENT_ID
    return headers


_last_request_time = 0.0


def _throttle_min_interval():
    """Ensure we don't send requests faster than configured min interval (basic client-side rate-limit)."""
    global _last_request_time
    now = time.time()
    elapsed = now - _last_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        wait = _MIN_REQUEST_INTERVAL - elapsed
        LOG.debug("Throttling: sleeping %.3fs to respect min interval", wait)
        time.sleep(wait)
    _last_request_time = time.time()


def _post_json(path: str, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    """
    POST JSON to Dhan path (relative to base). Returns (status_code, parsed_json)
    Raises DhanError for fatal client errors.
    Retries on 429/5xx up to _MAX_RETRIES with exponential backoff.
    """
    url = f"{_DHAN_BASE.rstrip('/')}/{path.lstrip('/')}"
    headers = _auth_headers()
    attempt = 0
    while True:
        attempt += 1
        _throttle_min_interval()
        try:
            if _HAS_REQUESTS:
                LOG.debug("Dhan POST (requests) %s attempt=%d payload=%s", url, attempt, payload)
                r = requests.post(url, json=payload, headers=headers, timeout=_REQUEST_TIMEOUT)
                status = r.status_code
                try:
                    body = r.json()
                except Exception:
                    body = {"__raw_text": r.text or ""}
            else:
                LOG.debug("Dhan POST (urllib3) %s attempt=%d payload=%s", url, attempt, payload)
                encoded = json.dumps(payload).encode("utf-8")
                r = _HTTP_POOL.request("POST", url, body=encoded, headers=headers, timeout=_REQUEST_TIMEOUT)
                status = r.status
                try:
                    body = json.loads(r.data.decode("utf-8") if isinstance(r.data, (bytes, bytearray)) else r.data)
                except Exception:
                    body = {"__raw_text": (r.data.decode("utf-8") if isinstance(r.data, (bytes, bytearray)) else r.data)}
        except Exception as e:
            # network error: retry a few times
            LOG.warning("Network error calling Dhan API: %s (attempt %d)", e, attempt, exc_info=True)
            if attempt > _MAX_RETRIES:
                raise DhanError(f"Dhan network error after {attempt} attempts: {e}") from e
            back = _BACKOFF_BASE * (2 ** (attempt - 1)) * (0.5 + random.random() * 0.8)
            time.sleep(back)
            continue

        # interpret response
        if status == 200:
            return status, body
        # auth problems -> immediate DhanError
        if status in (401, 403):
            LOG.error("Dhan auth failed (status=%d): %s", status, body)
            raise DhanError(f"Dhan auth failed (status={status}): {body}")
        # client bad request (400) often non-retryable
        if status == 400:
            LOG.error("Dhan returned 400 Bad Request: %s", body)
            raise DhanError(f"Dhan 400: {body}")
        # rate-limited or server error -> retry with backoff
        if status in (429, 502, 503, 504) or 500 <= status < 600:
            LOG.warning("Dhan transient error status=%d body=%s (attempt=%d)", status, body, attempt)
            if attempt > _MAX_RETRIES:
                raise DhanError(f"Dhan transient error after {attempt} attempts: status={status}, body={body}")
            # backoff with jitter
            back = _BACKOFF_BASE * (2 ** (attempt - 1)) * (0.5 + random.random() * 1.2)
            # cap sleep reasonably
            time.sleep(min(back, 10.0))
            continue
        # other statuses -> raise
        LOG.error("Unexpected Dhan response status=%d body=%s", status, body)
        raise DhanError(f"Unexpected Dhan status {status}: {body}")


# Public helper convenience wrappers
def optionchain_post(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    POST to the Dhan optionchain endpoint.
    Returns parsed JSON dict on success.
    May raise DhanError for fatal / retry exhausted errors.
    """
    status, body = _post_json("/v2/optionchain", payload)
    return body


def expiry_list_post(payload: Dict[str, Any]) -> Dict[str, Any]:
    """POST to the Dhan expiry list (same endpoint used by some APIs)."""
    status, body = _post_json("/v2/optionchain/expirylist", payload)
    return body


# Simple file cache helpers (persist to _CACHE_DIR) to reduce requests
def _cache_get(key: str, ttl: float = 30.0) -> Optional[Dict[str, Any]]:
    """
    key: simple string; we store as JSON file under _CACHE_DIR.
    ttl: seconds; if expired return None.
    """
    try:
        safe_name = key.replace("/", "_").replace(" ", "_")
        path = os.path.join(_CACHE_DIR, f"{safe_name}.json")
        if not os.path.exists(path):
            return None
        stat = os.stat(path)
        age = time.time() - stat.st_mtime
        if age > ttl:
            return None
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _cache_set(key: str, value: Dict[str, Any]) -> None:
    try:
        safe_name = key.replace("/", "_").replace(" ", "_")
        path = os.path.join(_CACHE_DIR, f"{safe_name}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(value, fh)
    except Exception:
        LOG.debug("Could not write cache file for %s", key, exc_info=True)


# Expose a small set of higher-level helpers commonly useful to callers.
def get_expiry_list_for_underlying(underlying_id: int, use_cache: bool = True, cache_ttl: float = 60.0) -> Dict[str, Any]:
    """
    Returns the raw response body for expirylist/optionchain metadata for underlying.
    Uses caching to avoid frequent requests.
    """
    if underlying_id is None:
        raise ValueError("underlying_id required")
    key = f"expirylist_{underlying_id}"
    if use_cache:
        cached = _cache_get(key, ttl=cache_ttl)
        if cached:
            LOG.debug("Using cached expirylist for %s", underlying_id)
            return cached
    payload = {"UnderlyingScrip": underlying_id}
    try:
        resp = expiry_list_post(payload)
        _cache_set(key, resp)
        return resp
    except DhanError:
        raise
    except Exception as e:
        LOG.exception("Unhandled error fetching expiry list")
        raise DhanError(f"Could not fetch expiry list: {e}") from e


def get_option_chain_for(underlying_id: int, expiry: Optional[str] = None, use_cache: bool = True, cache_ttl: float = 30.0) -> Dict[str, Any]:
    """
    Request option chain for an underlying (optionally for a specific expiry).
    Returns parsed JSON (body) on success.
    Caches responses to reduce rate limits.
    """
    if underlying_id is None:
        raise ValueError("underlying_id required")
    key = f"optionchain_{underlying_id}_{expiry or 'all'}"
    if use_cache:
        cached = _cache_get(key, ttl=cache_ttl)
        if cached:
            LOG.debug("Using cached optionchain for %s expiry=%s", underlying_id, expiry)
            return cached
    payload = {"UnderlyingScrip": underlying_id}
    if expiry:
        payload["Expiry"] = expiry
    try:
        resp = optionchain_post(payload)
        # store in cache
        _cache_set(key, resp)
        return resp
    except DhanError:
        raise
    except Exception as e:
        LOG.exception("Unhandled error fetching option chain")
        raise DhanError(f"Could not fetch option chain: {e}") from e
