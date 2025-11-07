"""Dhan scrip-master cache + lookup helpers.

This module downloads https://images.dhan.co/api-data/api-scrip-master.csv
into data_engine/market_ai/state/ and exposes a tiny resolver so trading
strategies can map (symbol, expiry, strike, option_type) -> security_id.
"""

from __future__ import annotations

import csv
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import requests

LOG = logging.getLogger(__name__)
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

MARKET_AI_DIR = Path(__file__).resolve().parents[2]
STATE_DIR = MARKET_AI_DIR / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)

MASTER_URL = os.getenv("DHAN_SCRIP_MASTER_URL", "https://images.dhan.co/api-data/api-scrip-master.csv")
MASTER_PATH = STATE_DIR / "dhan_scrip_master.csv"
REFRESH_SECONDS = float(os.getenv("DHAN_SCRIP_MASTER_TTL", str(6 * 3600)))  # default 6 hours
DOWNLOAD_TIMEOUT = float(os.getenv("DHAN_SCRIP_MASTER_TIMEOUT", "60"))

_cache_lock = threading.Lock()
_option_cache: Dict[Tuple[str, str, int, str], int] = {}
_cached_mtime: Optional[float] = None


def _download_master(path: Path) -> Path:
    LOG.info("Fetching Dhan scrip master CSV → %s", path)
    resp = requests.get(MASTER_URL, timeout=DOWNLOAD_TIMEOUT)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def _ensure_master_file() -> Path:
    if MASTER_PATH.exists():
        age = time.time() - MASTER_PATH.stat().st_mtime
        if age <= REFRESH_SECONDS:
            return MASTER_PATH
        LOG.info("Scrip master is stale (age %.1fs); refreshing", age)
        try:
            return _download_master(MASTER_PATH)
        except Exception as exc:
            LOG.warning("Could not refresh scrip master (%s); using stale copy", exc)
            return MASTER_PATH
    # no existing file → download once
    try:
        return _download_master(MASTER_PATH)
    except Exception as exc:
        LOG.error("Failed to download Dhan scrip master: %s", exc)
        raise


def _normalize_symbol(raw: str) -> str:
    if not raw:
        return ""
    base = raw.strip().upper()
    if "-" in base:
        return base.split("-", 1)[0]
    return base


def _normalize_expiry(raw: str) -> str:
    if not raw:
        return ""
    raw = raw.strip()
    if " " in raw:
        raw = raw.split(" ", 1)[0]
    return raw


def _normalize_strike(raw: str) -> Optional[int]:
    if raw is None or raw == "":
        return None
    try:
        return int(round(float(raw)))
    except Exception:
        return None


def _load_cache_locked() -> None:
    global _option_cache, _cached_mtime
    path = _ensure_master_file()
    mtime = path.stat().st_mtime
    if _option_cache and _cached_mtime == mtime:
        return

    new_cache: Dict[Tuple[str, str, int, str], int] = {}
    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            instrument = (row.get("SEM_INSTRUMENT_NAME") or "").upper()
            if instrument not in ("OPTIDX", "OPTSTK"):
                continue
            opt_type = (row.get("SEM_OPTION_TYPE") or "").upper()
            if opt_type not in ("CE", "PE"):
                continue
            symbol = _normalize_symbol(row.get("SEM_TRADING_SYMBOL") or row.get("SM_SYMBOL_NAME") or "")
            expiry = _normalize_expiry(row.get("SEM_EXPIRY_DATE") or "")
            strike = _normalize_strike(row.get("SEM_STRIKE_PRICE") or "")
            if not symbol or not expiry or strike is None:
                continue
            try:
                sec_id = int(str(row.get("SEM_SMST_SECURITY_ID")).strip())
            except Exception:
                continue
            key = (symbol, expiry, strike, opt_type)
            new_cache[key] = sec_id

    if not new_cache:
        LOG.warning("Parsed 0 option rows from scrip master; keeping previous cache of %d entries", len(_option_cache))
        return

    _option_cache = new_cache
    _cached_mtime = mtime
    LOG.info("Loaded %d option rows from Dhan scrip master", len(_option_cache))


def resolve_option_security_id(symbol: str, expiry: str, strike: float, option_type: str) -> Optional[int]:
    """
    Return Dhan security id for given symbol (e.g. 'NIFTY'), expiry 'YYYY-MM-DD',
    strike (float/int), and option_type ('CALL'/'PUT' or 'CE'/'PE').
    """
    symbol_norm = _normalize_symbol(symbol)
    expiry_norm = _normalize_expiry(expiry)
    strike_norm = _normalize_strike(strike)
    opt = option_type.strip().upper() if isinstance(option_type, str) else ""
    if opt == "CALL":
        opt = "CE"
    elif opt == "PUT":
        opt = "PE"

    if not symbol_norm or not expiry_norm or strike_norm is None or opt not in ("CE", "PE"):
        return None

    with _cache_lock:
        _load_cache_locked()
        return _option_cache.get((symbol_norm, expiry_norm, strike_norm, opt))


def refresh_scrip_master(force: bool = False) -> None:
    """Force a fresh download/parsing cycle (used by maintenance tools)."""
    with _cache_lock:
        if force and MASTER_PATH.exists():
            try:
                MASTER_PATH.unlink()
            except Exception:
                pass
        _option_cache.clear()
        _cached_mtime = None
        _load_cache_locked()
