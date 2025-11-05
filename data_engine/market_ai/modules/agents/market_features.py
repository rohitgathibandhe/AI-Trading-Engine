import numpy as np
import pandas as pd
from typing import Dict, Any

def _safe_series(arr):
    s = pd.Series(arr).astype(float)
    s.replace([np.inf, -np.inf], np.nan, inplace=True)
    return s.ffill().bfill()

def _atr_like(df: pd.DataFrame, period: int = 14) -> float:
    # Quick ATR approximation
    h, l, c = _safe_series(df["high"]), _safe_series(df["low"]), _safe_series(df["close"])
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=3).mean().iloc[-1]
    return float(atr) if pd.notna(atr) else 0.0

def _slope(df: pd.DataFrame, lookback: int = 20) -> float:
    # Linear regression slope on close
    c = _safe_series(df["close"])
    c = c[-lookback:]
    if len(c) < 5: return 0.0
    x = np.arange(len(c))
    m = np.polyfit(x, c.values, 1)[0]
    return float(m)

def summarize_timeframe(df: pd.DataFrame, tf_label: str) -> Dict[str, Any]:
    """
    Returns a compact summary including ATR%, slope, and a naive 'regime' label.
    Regime heuristic:
      - 'range' if ATR% is small and slope is tiny
      - 'trend_up' / 'trend_down' based on slope sign and magnitude
    """
    if df is None or len(df) < 10:
        return {"tf": tf_label, "atr_pct": 0.0, "slope": 0.0, "regime": "unknown"}

    close = _safe_series(df["close"])
    last = float(close.iloc[-1])
    atr = _atr_like(df, 14)
    atr_pct = (atr / max(1e-6, last)) * 100.0
    slope_val = _slope(df, 20)

    # Heuristics (tune as you wish)
    if abs(slope_val) < 0.5 and atr_pct < 0.8:
        regime = "range"
    elif slope_val >= 0.5:
        regime = "trend_up"
    elif slope_val <= -0.5:
        regime = "trend_down"
    else:
        regime = "neutral"

    return {"tf": tf_label, "atr_pct": atr_pct, "slope": slope_val, "regime": regime}

def topdown_gate(daily: Dict[str, Any], h1: Dict[str, Any], m15: Dict[str, Any]) -> bool:
    """
    Prefer entries when higher TFs aren’t trending hard:
      - Allow if D ∈ {range, neutral} AND H1 ∈ {range, neutral} AND M15 ≠ 'trend_extreme'
    """
    okD = daily["regime"] in ("range", "neutral")
    okH = h1["regime"] in ("range", "neutral")
    okM = m15["regime"] not in ("trend_up", "trend_down")
    return bool(okD and okH and okM)
