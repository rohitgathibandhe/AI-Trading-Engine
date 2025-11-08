# modules/vix_utils.py
"""
VIX utilities (robust + production-minded).

Exports:
  fallback_vix() -> float
  infer_vix_from_chain(optionchain_data) -> Optional[float]
  get_vix(use_optionchain=True, underlying_id=None, optionchain_data=None, vix_gate_min=None, force_vix=None) -> float
  should_skip_sells(vix: float, cfg: dict) -> (bool, str)

Notes:
 - This module performs lazy import of your Dhan helper (market_ai.modules.data_fetch.*)
 - It caches the last successful option-chain fetch for `underlying_id` for a short time window to reduce 429s.
 - `should_skip_sells` centralizes the gate logic used by the backtest engine/adapter.
"""

from __future__ import annotations
import os
import time
import logging
from typing import Optional, Any, Dict, Tuple

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# default fallbacks and simple cache settings
_DEFAULT_FALLBACK_VIX = float(os.environ.get("DEFAULT_FALLBACK_VIX", 10.0))
_CACHE_TTL_SEC = int(os.environ.get("VIX_CACHE_TTL_SEC", 3))  # small TTL to respect rate limits

# tiny in-process cache for optionchain data per (underlying_id, expiry)
_vix_cache: Dict[Tuple[int, Optional[str]], Tuple[float, float]] = {}
# structure: key -> (vix_value, timestamp)


def fallback_vix() -> float:
    """Return numeric fallback VIX when option-chain isn't available."""
    return float(os.environ.get("VIX_FALLBACK", _DEFAULT_FALLBACK_VIX))


def infer_vix_from_chain(optionchain_data: Dict[str, Any]) -> Optional[float]:
    """
    Defensive extraction of a VIX proxy from optionchain dict.
    Attempts to extract implied_volatility from CE/PE across strikes and return a median.

    Returns None if inference fails.
    """
    try:
        if not isinstance(optionchain_data, dict):
            return None

        # Dhan: top-level often {'data': {'oc': {...}, 'last_price': ...}}
        oc = optionchain_data.get("data", {}).get("oc")
        if not oc and isinstance(optionchain_data.get("oc"), dict):
            oc = optionchain_data.get("oc")
        # some caches might be the 'oc' dict itself
        if not oc and "ce" in optionchain_data:
            oc = optionchain_data

        if not oc or not isinstance(oc, dict):
            return None

        ivs = []
        for strike_key, strike_entry in oc.items():
            if not isinstance(strike_entry, dict):
                continue
            ce = strike_entry.get("ce")
            pe = strike_entry.get("pe")
            for opt in (ce, pe):
                if isinstance(opt, dict):
                    iv = opt.get("implied_volatility") or opt.get("iv") or opt.get("iv_percent")
                    if iv is None:
                        # sometimes greeks->vega etc hold info — skip if none
                        continue
                    try:
                        ivs.append(float(iv))
                    except Exception:
                        continue

        if not ivs:
            return None
        ivs.sort()
        n = len(ivs)
        mid = n // 2
        if n % 2 == 1:
            return float(ivs[mid])
        return float((ivs[mid - 1] + ivs[mid]) / 2.0)
    except Exception as e:
        logger.debug("infer_vix_from_chain failed: %s", e, exc_info=True)
        return None


def _fetch_remote_chain_and_infer(underlying_id: Optional[int] = None, expiry: Optional[str] = None) -> Optional[float]:
    """
    Attempt to call your Dhan wrapper to fetch optionchain and infer a VIX.
    Lazy-import safe: does not crash if dhan wrapper missing.
    """
    if underlying_id is None:
        return None

    key = (underlying_id, expiry)
    now = time.time()
    cached = _vix_cache.get(key)
    if cached:
        vix_val, ts = cached
        if now - ts <= _CACHE_TTL_SEC:
            logger.debug("vix_utils: using cached VIX for key=%s -> %s", key, vix_val)
            return float(vix_val)

    # lazy-import Dhan api wrapper(s) - support multiple possible module names used in this repo
    rhoi = None
    for mod_path in (
        "market_ai.modules.data_fetch.dhan_option_chain",
        "market_ai.modules.data_fetch.dhan_api",
    ):
        try:
            rhoi = __import__(mod_path, fromlist=["*"])  # type: ignore
            break
        except Exception:
            continue

    if rhoi is None:
        logger.debug("vix_utils: no Dhan helper module found (market_ai.modules.data_fetch.*)")
        return None

    # call the helper: we try a few argument patterns defensively
    try:
        params = {}
        params["UnderlyingScrip"] = underlying_id
        if getattr(rhoi, "DEFAULT_UNDERLYING_SEG", None):
            params["UnderlyingSeg"] = getattr(rhoi, "DEFAULT_UNDERLYING_SEG")
        else:
            params["UnderlyingSeg"] = params.get("UnderlyingSeg", "IDX_I")
        if expiry:
            params["Expiry"] = expiry

        logger.debug("vix_utils: calling Dhan option chain with params: %s", params)
        # many helper wrappers use option_chain(**params) or get_option_chain(...)
        oc_resp = None
        if hasattr(rhoi, "get_option_chain"):
            oc_resp = rhoi.get_option_chain(**params)
        elif hasattr(rhoi, "option_chain"):
            # some wrappers named option_chain
            oc_resp = rhoi.option_chain(**params)
        elif hasattr(rhoi, "fetch_option_chain"):
            oc_resp = rhoi.fetch_option_chain(**params)
        else:
            # fallback: try a generic _post_json helper if installed (very defensive)
            if hasattr(rhoi, "_post_json"):
                oc_resp = rhoi._post_json("optionchain", params)
            else:
                logger.debug("vix_utils: dhan helper present but no recognized call name")
                oc_resp = None

        if not oc_resp:
            logger.debug("vix_utils: Dhan helper returned empty response")
            return None

        iv = infer_vix_from_chain(oc_resp)
        if iv is not None:
            _vix_cache[key] = (float(iv), now)
        return iv
    except Exception as e:
        # Don't blow up the calling code; keep it in debug logs
        logger.debug("vix_utils: remote option chain fetch/infer failed: %s", e, exc_info=True)
        return None


def get_vix(use_optionchain: bool = True,
            underlying_id: Optional[int] = None,
            optionchain_data: Optional[Dict[str, Any]] = None,
            vix_gate_min: Optional[float] = None,
            force_vix: Optional[float] = None) -> float:
    """
    Main helper to obtain a VIX proxy.

    Priority:
     1) if force_vix provided -> return that
     2) if optionchain_data passed in -> infer
     3) if use_optionchain True and underlying_id provided -> try Dhan fetch + infer
     4) fallback_vix()

    vix_gate_min is not used to change the returned value — it's only for convenience if the caller wants to apply gate logic here.
    """
    # explicit override
    if force_vix is not None:
        logger.debug("vix_utils: force_vix override used -> %s", force_vix)
        return float(force_vix)

    # explicit provided chain
    if optionchain_data:
        iv = infer_vix_from_chain(optionchain_data)
        if iv is not None:
            logger.debug("vix_utils: inferred VIX from provided optionchain -> %s", iv)
            return float(iv)

    # try remote fetch (cached)
    if use_optionchain and underlying_id is not None:
        iv = _fetch_remote_chain_and_infer(underlying_id=underlying_id, expiry=None)
        if iv is not None:
            logger.debug("vix_utils: inferred VIX from remote Dhan chain -> %s", iv)
            return float(iv)

    # nothing worked -> fallback
    fb = fallback_vix()
    logger.debug("vix_utils: falling back to fallback_vix() -> %s", fb)
    return float(fb)


def should_skip_sells(vix: float,
                      cfg: Optional[Dict[str, Any]] = None) -> Tuple[bool, str]:
    """
    Decide whether to skip sell-based strategies based on VIX and strategy config.

    cfg is expected to contain an optional key:
      cfg.get("market_filters", {}).get("vix_min")  (numeric)
    Fallback behavior:
      - if no cfg or no vix_min configured -> return (False, reason) (do not skip)
      - if vix < vix_min -> return (True, reason)

    Returns (skip_bool, reason_string)
    """
    try:
        if cfg is None:
            return False, "no cfg provided -> allow sells"

        filters = cfg.get("market_filters", {}) if isinstance(cfg, dict) else {}
        vix_min = filters.get("vix_min", None)
        if vix_min is None:
            return False, "no vix_min configured; allow sells"

        try:
            vix_min = float(vix_min)
        except Exception:
            return False, "vix_min not numeric; allow sells"

        if float(vix) < vix_min:
            return True, f"vix {vix:.2f} < vix_min {vix_min:.2f} -> skip sells"
        return False, f"vix {vix:.2f} >= vix_min {vix_min:.2f} -> allow sells"
    except Exception as e:
        logger.debug("vix_utils.should_skip_sells error: %s", e, exc_info=True)
        return False, "error in vix gate -> allow sells"
