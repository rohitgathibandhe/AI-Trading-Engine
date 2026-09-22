# market_ai/modules/data_fetch/dhan_api.py
from __future__ import annotations
import os, time, json, logging, random, threading
from typing import Any, Dict, Optional, List
import requests

LOG = logging.getLogger(__name__)
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

class DhanError(RuntimeError):
    pass

class SimpleDhanClient:
    def __init__(self, client_id: Optional[str] = None, access_token: Optional[str] = None,
                 base_url: Optional[str] = None, timeout: float = 20.0):
        self.client_id = client_id or os.environ.get("DHAN_CLIENT_ID")
        self.access_token = access_token or os.environ.get("DHAN_ACCESS_TOKEN")
        self.base_url = (base_url or os.environ.get("DHAN_API_BASE") or "https://api.dhan.co").rstrip("/")
        self.timeout = timeout
        if not self.client_id or not self.access_token:
            raise DhanError("Missing DHAN credentials: set DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN")
        self.headers = {
            "Content-Type": "application/json",
            "access-token": self.access_token,
            "client-id": self.client_id,
        }
        LOG.info("SimpleDhanClient initialised base_url=%s", self.base_url)

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path if path.startswith('/') else '/'+path}"

    def post(self, path: str, payload: Dict[str, Any]) -> requests.Response:
        url = self._url(path)
        LOG.debug("POST %s payload=%s", url, {k: payload.get(k) for k in ("UnderlyingScrip","UnderlyingSeg","Expiry")})
        return requests.post(url, headers=self.headers, json=payload, timeout=self.timeout)

def make_client(base_url: Optional[str] = None) -> SimpleDhanClient:
    return SimpleDhanClient(base_url=base_url)

def _raise_for_status(resp: requests.Response):
    try:
        resp.raise_for_status()
    except Exception as e:
        body = ""
        try:
            body = resp.text[:800]
        except Exception:
            pass
        raise DhanError(f"HTTP {resp.status_code}: {body}") from e

# ---------- global throttle (respects ~1 call / 3s for /optionchain) ----------
_MIN_SEC_BETWEEN_OC = float(os.environ.get("MARKET_AI_OC_RATE_LIMIT_S", "3.2"))
_last_post_ts_lock = threading.Lock()
_last_post_ts = 0.0

# ---------- 429 CIRCUIT-BREAKER ----------
# A 429 means "stop sending requests." The old code did the opposite: it retried 5x per call, and the
# main loop re-fired every ~43s, hammering Dhan ~7-10 req/min WHILE rate-limited — which keeps the
# block alive and can escalate it to a longer ban (2026-09-22: a transient 429 became a 3.5h outage).
# The breaker imposes an ESCALATING cooldown after a 429 during which NO chain request is sent at all,
# so Dhan's limit can actually reset. Cleared on the first success.
_OC_429_BASE_COOLDOWN_S = float(os.environ.get("MARKET_AI_OC_429_COOLDOWN_S", "60") or 60)
_OC_429_MAX_COOLDOWN_S = float(os.environ.get("MARKET_AI_OC_429_MAX_COOLDOWN_S", "300") or 300)
_oc_429_lock = threading.Lock()
_oc_429_cooldown_until = 0.0
_oc_429_streak = 0


class RateLimitCooldown(DhanError):
    """Raised (without hitting Dhan) while inside the 429 cooldown window."""


def _oc_cooldown_remaining() -> float:
    with _oc_429_lock:
        return max(0.0, _oc_429_cooldown_until - time.time())


def _note_429() -> float:
    """Record a 429 and arm/extend the cooldown (escalating). Returns the cooldown seconds set."""
    global _oc_429_cooldown_until, _oc_429_streak
    with _oc_429_lock:
        _oc_429_streak += 1
        cd = min(_OC_429_BASE_COOLDOWN_S * (2 ** (_oc_429_streak - 1)), _OC_429_MAX_COOLDOWN_S)
        _oc_429_cooldown_until = time.time() + cd
        return cd


def _clear_429() -> None:
    global _oc_429_cooldown_until, _oc_429_streak
    with _oc_429_lock:
        _oc_429_cooldown_until = 0.0
        _oc_429_streak = 0


def _respect_throttle_if_needed(path: str):
    if not path.endswith("/optionchain"):
        return
    remaining = _oc_cooldown_remaining()
    if remaining > 0:
        # Do NOT send a request while cooling down — that is what deepens the ban.
        raise RateLimitCooldown(f"optionchain in 429 cooldown for {remaining:.0f}s (backing off, not sending)")
    global _last_post_ts
    with _last_post_ts_lock:
        now = time.time()
        delta = now - _last_post_ts
        if delta < _MIN_SEC_BETWEEN_OC:
            sleep_s = _MIN_SEC_BETWEEN_OC - delta
            LOG.debug("Throttling /optionchain for %.2fs", sleep_s)
            time.sleep(sleep_s)
        _last_post_ts = time.time()

def _post_json_with_backoff(client: SimpleDhanClient, path: str, payload: Dict[str, Any],
                            attempts: int = 5, backoff_sec: float = 1.25) -> Dict[str, Any]:
    """
    Robust POST:
      - respects throttle for /optionchain
      - handles 429 with enforced 3.2s cooldown
      - exponential backoff with jitter
      - surfaces DHAN 'status':'failed' bodies verbatim
    """
    last = None
    for i in range(attempts):
        try:
            _respect_throttle_if_needed(path)   # raises RateLimitCooldown while cooling down (no request sent)
            resp = client.post(path, payload)
            # 429 = "stop sending." Do NOT keep retrying (each retry deepens the ban). Arm the escalating
            # cooldown and bail immediately so no further requests go out until Dhan resets.
            if resp.status_code == 429 or (400 <= resp.status_code < 500 and "805" in (getattr(resp, "text", "") or "")):
                cd = _note_429()
                LOG.warning("429/rate-limit on %s — CIRCUIT-BREAKER armed, backing off %.0fs (no more requests)", path, cd)
                raise RateLimitCooldown(f"429 on {path}; cooling down {cd:.0f}s")
            if 400 <= resp.status_code < 500:
                body = resp.text[:800] if hasattr(resp, "text") else ""
                raise DhanError(f"HTTP {resp.status_code}: {body}")
            _raise_for_status(resp)
            data = resp.json()
            if isinstance(data, dict) and data.get("status") == "failed":
                # a 'failed' body may also be a rate-limit (code 805) — treat those as a 429, else surface
                if "805" in json.dumps(data) or "too many requests" in json.dumps(data).lower():
                    cd = _note_429()
                    LOG.warning("rate-limit (805) on %s — CIRCUIT-BREAKER armed, backing off %.0fs", path, cd)
                    raise RateLimitCooldown(f"805 on {path}; cooling down {cd:.0f}s")
                raise DhanError(json.dumps(data)[:800])
            _clear_429()                          # a clean response ends any cooldown
            return data
        except RateLimitCooldown:
            raise                                 # don't retry a rate-limit — propagate up, loop skips this cycle
        except Exception as e:
            last = e
            sleep = backoff_sec * (2 ** i) + random.uniform(0, 0.25)
            LOG.warning("POST attempt %d/%d failed: %s | sleeping %.2fs", i+1, attempts, e, sleep)
            time.sleep(sleep)
    raise DhanError(f"POST failed after {attempts} attempts: {last}")

# ---- caches ----
_expiry_cache: Dict[str, List[str]] = {}
_oc_cache: Dict[str, Any] = {}

def get_expiry_list_for_underlying(client: SimpleDhanClient, underlying_id: int,
                                   underlying_seg: str, use_cache: bool = True) -> List[str]:
    key = f"expirylist:{underlying_id}:{underlying_seg}"
    if use_cache and key in _expiry_cache:
        return _expiry_cache[key]
    payload = {"UnderlyingScrip": int(underlying_id), "UnderlyingSeg": str(underlying_seg)}
    data = _post_json_with_backoff(client, "/v2/optionchain/expirylist", payload)
    expiries: List[str] = []
    if isinstance(data, dict):
        raw = data.get("data")
        if isinstance(raw, list):
            expiries = [str(x) for x in raw]
    if use_cache:
        _expiry_cache[key] = expiries
    return expiries

def get_option_chain_for(client: SimpleDhanClient, underlying_id: int, expiry: Optional[str] = None,
                         underlying_seg: str = "IDX_I", use_cache: bool = True) -> Dict[str, Any]:
    key = f"oc:{underlying_id}:{underlying_seg}:{expiry or '_'}"
    if use_cache and key in _oc_cache:
        return _oc_cache[key]
    payload = {"UnderlyingScrip": int(underlying_id), "UnderlyingSeg": str(underlying_seg)}
    if expiry:
        payload["Expiry"] = str(expiry)
    data = _post_json_with_backoff(client, "/v2/optionchain", payload)
    node = data.get("data", data) if isinstance(data, dict) else {}
    oc = node.get("oc") or node.get("optionchain") or node
    # normalize list → dict keyed by strike
    if isinstance(oc, list):
        out = {}
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
        oc = out
    if use_cache:
        _oc_cache[key] = oc if isinstance(oc, dict) else {}
    return oc if isinstance(oc, dict) else {}
