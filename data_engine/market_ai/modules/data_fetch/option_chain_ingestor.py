# -*- coding: utf-8 -*-
"""
Thin convenience layer over ``dhan_option_chain.OptionChain`` that returns a
tidy pandas DataFrame for downstream strategy logic (monthly strangle +
weekly hedge).

It handles:
  * Client construction from env (DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN)
  * Normalising the nested {"strike": {"ce": {...}, "pe": {...}}} response
  * Extracting common price / IV / OI fields with sensible fallbacks
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

from market_ai.modules.data_fetch.dhan_api import make_client, SimpleDhanClient
from market_ai.modules.data_fetch.dhan_option_chain import OptionChain


def _first(node: Dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        if k in node and node[k] not in (None, ""):
            return node[k]
    return None


def _as_float(val: Any, default: Optional[float] = None) -> Optional[float]:
    if val in (None, "", [], {}):
        return default
    try:
        return float(str(val).replace(",", ""))
    except Exception:
        return default


@dataclass
class ChainConfig:
    underlying_id: int
    underlying_seg: str = "IDX_I"   # DHAN segment code (NIFTY spot = IDX_I, F&O = NSE_FNO)


class OptionChainIngestor:
    """Fetches option-chain snapshots and returns a tidy DataFrame."""

    def __init__(self, chain_cfg: ChainConfig, client: Optional[SimpleDhanClient] = None):
        self.cfg = chain_cfg
        self.client = client or make_client()
        self.chain = OptionChain(self.client, default_seg=self.cfg.underlying_seg)

    def fetch(self, expiry: str) -> pd.DataFrame:
        """
        Fetch option-chain for ``expiry`` (YYYY-MM-DD) or tag ("weekly") and
        return a DataFrame sorted by strike.
        """
        raw = self.chain.get_option_chain(
            underlying_id=self.cfg.underlying_id,
            expiry_or_tag=expiry,
            underlying_seg=self.cfg.underlying_seg,
        )

        rows = []
        for strike, node in (raw or {}).items():
            try:
                k = float(str(strike))
            except Exception:
                continue

            ce = node.get("ce") or {}
            pe = node.get("pe") or {}

            rows.append(
                {
                    "strike": k,
                    "ltp_ce": _as_float(
                        _first(ce, ("ltp", "LTP", "lastPrice", "last_price", "close"))
                    ),
                    "ltp_pe": _as_float(
                        _first(pe, ("ltp", "LTP", "lastPrice", "last_price", "close"))
                    ),
                    "delta_ce": _as_float(
                        _first(ce.get("greeks", {}) if isinstance(ce, dict) else {}, ("delta", "Delta"))
                    ),
                    "delta_pe": _as_float(
                        _first(pe.get("greeks", {}) if isinstance(pe, dict) else {}, ("delta", "Delta"))
                    ),
                    "oi_ce": _as_float(_first(ce, ("oi", "openInterest", "OI"))),
                    "oi_pe": _as_float(_first(pe, ("oi", "openInterest", "OI"))),
                    "IV_CE": _as_float(_first(ce, ("iv", "impliedVolatility", "IV"))),
                    "IV_PE": _as_float(_first(pe, ("iv", "impliedVolatility", "IV"))),
                    "underlying_ltp": _as_float(
                        _first(
                            ce,
                            (
                                "underlying_ltp",
                                "underlyingValue",
                                "underlying_value",
                                "spot",
                                "underlyingPrice",
                                "underlying",
                            ),
                        )
                        or _first(
                            pe,
                            (
                                "underlying_ltp",
                                "underlyingValue",
                                "underlying_value",
                                "spot",
                                "underlyingPrice",
                                "underlying",
                            ),
                        )
                    ),
                    "raw_call": ce,
                    "raw_put": pe,
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values("strike").reset_index(drop=True)
        return df

    def list_expiries(self) -> pd.DataFrame:
        """Return a Series of upcoming expiries for the configured underlying."""
        exp = self.chain._expiries(self.cfg.underlying_id, self.cfg.underlying_seg)  # type: ignore[attr-defined]
        return pd.Series(exp, name="expiry")
