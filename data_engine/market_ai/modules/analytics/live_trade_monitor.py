from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class Zone:
    side: str  # "resistance" or "support"
    price: float
    timeframe: str  # e.g., "5m", "15m", "1h", "d"
    timestamp: datetime


@dataclass
class PositionHealth:
    pnl: float
    pnl_pct: Optional[float]
    spot: float
    nearest_support: Optional[Zone]
    nearest_resistance: Optional[Zone]
    distance_to_support_pct: Optional[float]
    distance_to_resistance_pct: Optional[float]
    bias: str  # "bullish"/"bearish"/"neutral"
    risk_flag: str  # "safe"/"caution"/"exit"
    notes: List[str]
    context: Dict[str, object]


def _pivot_points(prices: pd.DataFrame, lookback: int) -> List[Zone]:
    """Return swing highs/lows using a simple pivot window."""
    pivots: List[Zone] = []
    if prices.empty or lookback < 2:
        return pivots
    highs = prices["high"].values
    lows = prices["low"].values
    ts = np.array(prices["timestamp"].dt.to_pydatetime())
    for i in range(lookback, len(prices) - lookback):
        left_high = highs[i - lookback : i]
        right_high = highs[i + 1 : i + lookback + 1]
        if highs[i] >= left_high.max() and highs[i] >= right_high.max():
            pivots.append(
                Zone(
                    side="resistance",
                    price=float(highs[i]),
                    timeframe=prices.attrs.get("timeframe", "unk"),
                    timestamp=ts[i],
                )
            )
        left_low = lows[i - lookback : i]
        right_low = lows[i + 1 : i + lookback + 1]
        if lows[i] <= left_low.min() and lows[i] <= right_low.min():
            pivots.append(
                Zone(
                    side="support",
                    price=float(lows[i]),
                    timeframe=prices.attrs.get("timeframe", "unk"),
                    timestamp=ts[i],
                )
            )
    return pivots


def find_multi_tf_zones(candles: Dict[str, pd.DataFrame]) -> List[Zone]:
    """
    Build swing-derived support/resistance zones across timeframes.

    candles: mapping timeframe -> DataFrame with timestamp/high/low columns.
    """
    zones: List[Zone] = []
    for tf, df in candles.items():
        if df is None or df.empty or not {"timestamp", "high", "low"}.issubset(df.columns):
            continue
        frame = df.copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame = frame.dropna(subset=["timestamp", "high", "low"])
        frame.attrs["timeframe"] = tf
        if tf in {"1m", "5m"}:
            lookback = 3
        elif tf in {"15m", "30m", "1h"}:
            lookback = 2
        else:
            lookback = 1
        zones.extend(_pivot_points(frame, lookback))
    # Deduplicate nearby zones (within 0.05%)
    zones = sorted(zones, key=lambda z: (z.timeframe, z.timestamp))
    merged: List[Zone] = []
    thresh = 0.0005
    for z in zones:
        if not merged:
            merged.append(z)
            continue
        if abs(z.price - merged[-1].price) / max(z.price, 1.0) < thresh and z.side == merged[-1].side:
            continue
        merged.append(z)
    return merged


def nearest_zones(zones: Iterable[Zone], spot: float) -> Tuple[Optional[Zone], Optional[Zone]]:
    res_up = [z for z in zones if z.side == "resistance" and z.price >= spot]
    res_down = [z for z in zones if z.side == "resistance" and z.price < spot]
    sup_up = [z for z in zones if z.side == "support" and z.price >= spot]
    sup_down = [z for z in zones if z.side == "support" and z.price < spot]
    nearest_res = min(res_up, key=lambda z: z.price, default=None) or max(res_down, key=lambda z: z.price, default=None)
    nearest_sup = max(sup_down, key=lambda z: z.price, default=None) or min(sup_up, key=lambda z: z.price, default=None)
    return nearest_sup, nearest_res


def evaluate_position_health(
    *,
    pnl: float,
    pnl_pct: Optional[float],
    spot: float,
    zones: Iterable[Zone],
    min_buffer_pct: float = 0.003,
    caution_drawdown_pct: float = -0.01,
    exit_drawdown_pct: float = -0.02,
    gap_pct: Optional[float] = None,
    resistance: Optional[Zone] = None,
    support: Optional[Zone] = None,
) -> PositionHealth:
    sup, res = nearest_zones(zones, spot)
    if support is not None:
        sup = support
    if resistance is not None:
        res = resistance
    dist_sup_pct = (spot - sup.price) / spot if sup else None
    dist_res_pct = (res.price - spot) / spot if res else None
    bias = "neutral"
    if sup and res:
        if dist_sup_pct is not None and dist_res_pct is not None:
            bias = "bullish" if dist_sup_pct > dist_res_pct else "bearish"
    notes: List[str] = []
    risk_flag = "safe"
    if pnl_pct is not None:
        if pnl_pct <= exit_drawdown_pct * 100:
            risk_flag = "exit"
            notes.append(f"Drawdown {pnl_pct:.2f}% worse than {exit_drawdown_pct*100:.1f}%")
        elif pnl_pct <= caution_drawdown_pct * 100:
            risk_flag = "caution"
            notes.append(f"Drawdown {pnl_pct:.2f}% worse than {caution_drawdown_pct*100:.1f}%")
    if dist_sup_pct is not None and dist_sup_pct < min_buffer_pct:
        risk_flag = "caution" if risk_flag == "safe" else risk_flag
        notes.append(f"Support buffer thin ({dist_sup_pct*100:.2f}%)")
    if dist_res_pct is not None and dist_res_pct < min_buffer_pct:
        notes.append(f"Close to resistance ({dist_res_pct*100:.2f}%)")
    if gap_pct is not None:
        notes.append(f"Gap {gap_pct*100:.2f}%")
    return PositionHealth(
        pnl=pnl,
        pnl_pct=pnl_pct,
        spot=spot,
        nearest_support=sup,
        nearest_resistance=res,
        distance_to_support_pct=dist_sup_pct,
        distance_to_resistance_pct=dist_res_pct,
        bias=bias,
        risk_flag=risk_flag,
        notes=notes,
        context={"gap_pct": gap_pct},
    )


def summarize_position_health(health: PositionHealth) -> Dict[str, object]:
    return {
        "pnl": health.pnl,
        "pnl_pct": health.pnl_pct,
        "spot": health.spot,
        "risk_flag": health.risk_flag,
        "bias": health.bias,
        "nearest_support": _fmt_zone(health.nearest_support),
        "nearest_resistance": _fmt_zone(health.nearest_resistance),
        "distance_to_support_pct": _pct(health.distance_to_support_pct),
        "distance_to_resistance_pct": _pct(health.distance_to_resistance_pct),
        "notes": health.notes,
    }


def _fmt_zone(z: Optional[Zone]) -> Optional[Dict[str, object]]:
    if z is None:
        return None
    return {
        "side": z.side,
        "price": z.price,
        "timeframe": z.timeframe,
        "timestamp": z.timestamp.isoformat(),
    }


def _pct(val: Optional[float]) -> Optional[float]:
    return None if val is None else round(val * 100, 3)
