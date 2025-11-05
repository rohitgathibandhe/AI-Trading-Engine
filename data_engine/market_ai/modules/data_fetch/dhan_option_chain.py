# market_ai/modules/data_fetch/dhan_option_chain.py
from __future__ import annotations
from typing import Dict, Any, List, Optional
from datetime import date, datetime
import logging, time

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

class OptionChain:
    """LIVE OC via DHAN (for *current/future* expiries). Not suitable for history."""
    def __init__(self, client: SimpleDhanClient, default_seg: str = "IDX_I"):
        self.client = client
        self.default_seg = default_seg
        self._expiry_cache: Dict[str, List[str]] = {}

    def get_option_chain(self, underlying_id: int, expiry_or_tag: str, underlying_seg: Optional[str] = None) -> Dict[str, Any]:
        seg = (underlying_seg or self.default_seg) or "IDX_I"

        # 1) If explicit ISO date, call exactly that expiry once. Do NOT remap to "next".
        if _is_iso_date(expiry_or_tag):
            try:
                oc_raw = get_option_chain_for(self.client, underlying_id, expiry=expiry_or_tag, underlying_seg=seg)
                return _normalize(oc_raw)
            except DhanError as e:
                LOG.warning("DHAN could not serve explicit expiry %s on %s: %s", expiry_or_tag, seg, e)
                return {}

        # 2) If tag 'weekly', resolve the next weekly from DHAN expiry list (live only).
        if expiry_or_tag.lower() == "weekly":
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
