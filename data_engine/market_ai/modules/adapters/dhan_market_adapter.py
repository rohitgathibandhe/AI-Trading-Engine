# -*- coding: utf-8 -*-
"""
dhan_market_adapter.py
A thin market adapter used by the AdaptiveAgent/StrategySelector.

It MUST provide:
  - get_spot(symbol) -> float
  - get_ohlc(symbol, timeframe: "1D"|"1H"|"15m"|"5m", limit: int) -> pd.DataFrame
    (returns columns: ['datetime','open','high','low','close'])

This version tries to pull a real spot from your existing modules:
  - modules.data_fetch.dhan_option_chain (OptionChain / OptionChainAdapter)
  - modules.data_fetch.dhan_wrapper.get_option_chain / get_option_price
and falls back to a local synthetic generator so the agent keeps running.

Replace the fallback parts with your live API calls whenever ready.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import datetime as _dt
import random

import pandas as pd

# -------------------------------
# Resolve optional data sources
# -------------------------------

# Option-chain adapter (to derive spot if available)
_OptionChainCls = None
try:
    from modules.data_fetch import dhan_option_chain as _doc
    for _cand in ("OptionChain", "OptionChainAdapter", "DhanOptionChainAdapter", "DhanOptionChain"):
        if hasattr(_doc, _cand):
            _OptionChainCls = getattr(_doc, _cand)
            break
except Exception:
    _OptionChainCls = None

# Wrapper with helper functions like get_option_chain / get_option_price
_dhan_wrapper = None
try:
    from modules.data_fetch import dhan_wrapper as _dw
    _dhan_wrapper = _dw
except Exception:
    _dhan_wrapper = None


class DhanMarketAdapter:
    """
    Primary adapter used by the agent. Minimal, dependency-light, and resilient.
    """

    def __init__(self, config: Dict[str, Any]):
        self.cfg = config
        # Optional: instantiate option-chain facade if available
        self._chain = _OptionChainCls(config) if _OptionChainCls else None
        # Basic caches to make synthetic OHLC look smoother within a session
        self._last_spot: Dict[str, float] = {}

    # ------------------------------------------------------------
    # Public API expected by StrategySelector / AdaptiveAgent
    # ------------------------------------------------------------

    def get_spot(self, symbol: str) -> float:
        """
        Try to return a real spot (underlying index price). Fallback to a reasonable synthetic.
        Priority:
          1) Option chain adapter ATM row's underlying_ltp / underlying_value
          2) Wrapper.get_option_chain(symbol) -> dict with 'underlying_value'
          3) Synthetic spot around last cached value or 24000 baseline (NIFTY)
        """
        # 1) Option-chain adapter ATM row
        try:
            if self._chain:
                df = self._chain.get_current_week_chain(symbol)
                if isinstance(df, pd.DataFrame) and not df.empty:
                    # prefer explicit 'underlying_ltp' if provided
                    if "underlying_ltp" in df.columns:
                        val = float(df["underlying_ltp"].iloc[0])
                        if val > 0:
                            self._last_spot[symbol] = val
                            return val
                    # else: take ATM row and use any embedded field
                    atm = self._chain.get_atm_row(df)
                    for k in ("underlying_ltp", "underlying_value", "spot"):
                        if k in atm:
                            val = float(atm[k])
                            if val > 0:
                                self._last_spot[symbol] = val
                                return val
        except Exception:
            pass

        # 2) Wrapper.get_option_chain(symbol) -> dict with underlying_value
        try:
            if _dhan_wrapper and hasattr(_dhan_wrapper, "get_option_chain"):
                oc = _dhan_wrapper.get_option_chain(symbol)
                if isinstance(oc, dict):
                    uv = oc.get("underlying_value") or oc.get("underlyingLtp") or oc.get("spot")
                    if uv:
                        val = float(uv)
                        if val > 0:
                            self._last_spot[symbol] = val
                            return val
        except Exception:
            pass

        # 3) Fallback synthetic
        base = self._last_spot.get(symbol, 24000.0)
        # gentle random walk so ATR/regime calcs don't flatline
        step = random.uniform(-15, 15)
        val = max(1000.0, base + step)
        self._last_spot[symbol] = val
        return val

    def get_ohlc(self, symbol: str, timeframe: str = "1D", limit: int = 80) -> pd.DataFrame:
        """
        Return OHLC candles for the given timeframe.

        If you have a live/historical API, wire it here. For now, we synthesize
        a smooth series around the live/synthetic spot so the rest of the engine
        (ATR/regime gates) functions properly.

        timeframe: "1D" | "1H" | "15m" | "5m"
        """
        # Map to pandas frequency
        # Replace your freq map with this
        _freq_map = {"1D": "1D", "1H": "h", "15m": "15min", "5m": "5min"}
        freq = _freq_map.get(timeframe, "1D")

        end = _dt.datetime.now()
        # Ensure at least 20 bars even if someone passes too small a limit
        n = max(20, int(limit))

        # Build a gentle random walk around spot
        spot = float(self.get_spot(symbol))
        dates = pd.date_range(end=end, periods=n, freq=freq)

        # Random walk params tuned to timeframe so it "looks" plausible
        scale_map = {"1D": 40.0, "1H": 20.0, "15m": 10.0, "5m": 6.0}
        step_scale = scale_map.get(timeframe, 30.0)

        closes = []
        price = spot
        for _ in range(n):
            # mean-revert softly toward spot
            drift = (spot - price) * 0.02
            step = drift + random.uniform(-step_scale, step_scale)
            price = max(100.0, price + step)
            closes.append(price)

        # Build O/H/L/C around the close path
        opens = [closes[0]] + closes[:-1]
        highs = [c + abs(random.uniform(2, step_scale * 0.8)) for c in closes]
        lows  = [c - abs(random.uniform(2, step_scale * 0.8)) for c in closes]

        df = pd.DataFrame(
            {
                "datetime": dates,
                "open": opens,
                "high": highs,
                "low": lows,
                "close": closes,
            }
        )
        return df

# Backward-compatibility alias if other code imports MarketAdapter
MarketAdapter = DhanMarketAdapter
