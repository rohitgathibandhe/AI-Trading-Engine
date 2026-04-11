#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import asdict
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DATA_ENGINE_ROOT = REPO_ROOT / "data_engine"
for path in (REPO_ROOT, DATA_ENGINE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from market_ai.modules.agents.intraday_option_selling_advisor import (  # noqa: E402
    IntradayOptionSellingAdvisor,
    IntradayOptionSellingAdvisorConfig,
)
from market_ai.modules.agents.intraday_learning import (  # noqa: E402
    IntradayLearningConfig,
    IntradayLearningManager,
)
from market_ai.modules.analytics.live_trade_monitor import find_multi_tf_zones, nearest_zones  # noqa: E402
from market_ai.modules.analytics.price_action_patterns import detect_recent_price_action  # noqa: E402
from market_ai.modules.agents.intraday_position_manager import IntradayPositionManager  # noqa: E402
from market_ai.modules.agents.decision_committee import DecisionCommittee  # noqa: E402
from market_ai.strategies import LegSide, OptionLeg, OptionType, StrategyType  # noqa: E402


IST = ZoneInfo("Asia/Kolkata") if ZoneInfo else None


class BacktestIntradayOptionSellingAdvisor(IntradayOptionSellingAdvisor):
    def _persist(self, when: Optional[datetime] = None) -> None:
        now = when or datetime.now(IST) if IST else datetime.now()
        self.state.updated_at = now.isoformat(timespec="seconds")
        if not self.state.current_session_date:
            self.state.current_session_date = now.date().isoformat()


class BacktestIntradayPositionManager(IntradayPositionManager):
    def __init__(self, *args, **kwargs) -> None:
        self.history_rows: List[Dict[str, Any]] = []
        super().__init__(*args, **kwargs)

    def _persist(self, when: Optional[datetime] = None) -> None:
        now = when or datetime.now(IST) if IST else datetime.now()
        self.state.updated_at = now.isoformat(timespec="seconds")
        if not self.state.current_session_date:
            self.state.current_session_date = now.date().isoformat()

    def _append_history(self, payload: Dict[str, Any]) -> None:
        self.history_rows.append(dict(payload))


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value in (None, "", "-", "--"):
        return default
    try:
        val = float(value)
        if math.isnan(val):
            return default
        return val
    except Exception:
        return default


def _parse_ts(value: Any) -> datetime:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        if IST:
            ts = ts.tz_localize(IST)
    elif IST:
        ts = ts.tz_convert(IST)
    return ts.to_pydatetime()


def _parse_time(value: str) -> dtime:
    return datetime.strptime(value, "%H:%M").time()


def _candles_to_zone_frame(candles: List[Dict[str, Any]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for candle in candles or []:
        if not isinstance(candle, dict):
            continue
        try:
            ts_raw = candle.get("timestamp") or candle.get("time")
            ts = _parse_ts(ts_raw)
            rows.append(
                {
                    "timestamp": ts,
                    "open": float(candle.get("open") or 0.0),
                    "high": float(candle.get("high") or candle.get("open") or 0.0),
                    "low": float(candle.get("low") or candle.get("open") or 0.0),
                    "close": float(candle.get("close") or candle.get("open") or 0.0),
                }
            )
        except Exception:
            continue
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame


def _summarize_swing_structure(zones: List[Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "bias": "NEUTRAL",
        "label": "RANGE",
        "confidence": 0.0,
        "reasons": [],
        "timeframes": {},
        "nearest_support": None,
        "nearest_resistance": None,
    }
    if not zones:
        return summary
    bull_score = 0.0
    bear_score = 0.0
    total_weight = 0.0
    tf_weights = {"5m": 1.0, "15m": 1.8, "60m": 2.4, "1h": 2.4}

    def _trend(prices: List[float]) -> str:
        if len(prices) < 2:
            return "UNKNOWN"
        delta = float(prices[-1]) - float(prices[-2])
        threshold = max(5.0, abs(float(prices[-2])) * 0.0005)
        if delta >= threshold:
            return "RISING"
        if delta <= -threshold:
            return "FALLING"
        return "FLAT"

    for tf, weight in tf_weights.items():
        tf_zones = [z for z in zones if str(getattr(z, "timeframe", "")) == tf]
        supports = sorted(
            [float(getattr(z, "price", 0.0)) for z in tf_zones if str(getattr(z, "side", "")) == "support"]
        )
        resistances = sorted(
            [float(getattr(z, "price", 0.0)) for z in tf_zones if str(getattr(z, "side", "")) == "resistance"]
        )
        support_trend = _trend(supports[-2:])
        resistance_trend = _trend(resistances[-2:])
        label = "RANGE"
        if support_trend == "RISING" and resistance_trend == "RISING":
            bull_score += weight
            label = "HH_HL_UPTREND"
        elif support_trend == "FALLING" and resistance_trend == "FALLING":
            bear_score += weight
            label = "LH_LL_DOWNTREND"
        elif support_trend == "RISING" or resistance_trend == "RISING":
            bull_score += weight * 0.4
            label = "BULLISH_TRANSITION"
        elif support_trend == "FALLING" or resistance_trend == "FALLING":
            bear_score += weight * 0.4
            label = "BEARISH_TRANSITION"
        total_weight += weight
        summary["timeframes"][tf] = {
            "support_trend": support_trend,
            "resistance_trend": resistance_trend,
            "support_count": len(supports),
            "resistance_count": len(resistances),
            "label": label,
        }
        if label not in {"RANGE", "BULLISH_TRANSITION", "BEARISH_TRANSITION"}:
            summary["reasons"].append(f"{tf}:{label}")
    if bull_score > bear_score and bull_score >= 0.8:
        summary["bias"] = "BULLISH"
        summary["label"] = "HH_HL_UPTREND" if bull_score >= bear_score + 0.5 else "BULLISH_TRANSITION"
    elif bear_score > bull_score and bear_score >= 0.8:
        summary["bias"] = "BEARISH"
        summary["label"] = "LH_LL_DOWNTREND" if bear_score >= bull_score + 0.5 else "BEARISH_TRANSITION"
    elif bull_score > 0.0 or bear_score > 0.0:
        summary["label"] = "TRANSITION"
    summary["confidence"] = round(float(min(1.0, abs(bull_score - bear_score) / max(1.0, total_weight))), 3)
    return summary


def _next_thursday(day: date) -> date:
    offset = (3 - day.weekday()) % 7
    return day + timedelta(days=offset)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _prefix_option_type(prefix: str) -> Optional[str]:
    p = str(prefix or "").strip().lower()
    if p == "atm_call" or p.startswith("call_"):
        return "CE"
    if p == "atm_put" or p.startswith("put_"):
        return "PE"
    return None


def _discover_option_prefixes(columns: Iterable[str]) -> Dict[str, str]:
    prefixes: Dict[str, str] = {}
    for col in columns:
        if not col.endswith("_strike"):
            continue
        prefix = col[: -len("_strike")]
        opt = _prefix_option_type(prefix)
        if opt:
            prefixes[prefix] = opt
    return prefixes


def _strategy_type_enum(name: str) -> StrategyType:
    strategy_name = str(name or "").upper()
    if strategy_name in {"PUT_CREDIT_SPREAD", "SHORT_PUT_WITH_HEDGE"}:
        return StrategyType.BULL_PUT_SPREAD
    if strategy_name in {"CALL_CREDIT_SPREAD", "SHORT_CALL_WITH_HEDGE"}:
        return StrategyType.BEAR_CALL_SPREAD
    return StrategyType.NONE


def _build_chain_rows(
    row: pd.Series,
    *,
    prefixes: Dict[str, str],
    prev_oi_map: Dict[Tuple[str, float], float],
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, float], float]]:
    merged: Dict[Tuple[str, float], Dict[str, Any]] = {}
    next_prev_oi = dict(prev_oi_map)
    for prefix, option_type in prefixes.items():
        strike = _safe_float(row.get(f"{prefix}_strike"))
        ltp = _safe_float(row.get(f"{prefix}_ltp"))
        if strike is None or ltp is None or strike <= 0:
            continue
        oi = _safe_float(row.get(f"{prefix}_oi"))
        key = (option_type, float(strike))
        oi_change = None
        if oi is not None:
            prev_oi = prev_oi_map.get(key)
            if prev_oi is not None:
                oi_change = float(oi) - float(prev_oi)
            next_prev_oi[key] = float(oi)
        payload = {
            "option_type": option_type,
            "strike": float(strike),
            "ltp": float(ltp),
            "oi": None if oi is None else float(oi),
            "oi_change": None if oi_change is None else float(oi_change),
            "iv": _safe_float(row.get(f"{prefix}_iv")),
            "volume": _safe_float(row.get(f"{prefix}_volume")),
            "delta": _safe_float(row.get(f"{prefix}_delta")),
        }
        existing = merged.get(key)
        if existing is None:
            merged[key] = payload
            continue
        existing_score = (existing.get("oi") is not None, existing.get("volume") is not None)
        payload_score = (payload.get("oi") is not None, payload.get("volume") is not None)
        if payload_score > existing_score:
            merged[key] = payload
    rows = sorted(merged.values(), key=lambda item: (str(item["option_type"]), float(item["strike"])))
    return rows, next_prev_oi


def _extend_otm_chain_rows(chain_rows: List[Dict[str, Any]], *, min_extra_steps: int = 6) -> List[Dict[str, Any]]:
    rows = [dict(row) for row in chain_rows if isinstance(row, dict)]
    strikes = sorted({float(row.get("strike") or 0.0) for row in rows if _safe_float(row.get("strike")) is not None})
    if len(strikes) < 2:
        return rows
    step = min(
        (strikes[i + 1] - strikes[i] for i in range(len(strikes) - 1) if (strikes[i + 1] - strikes[i]) > 0),
        default=50.0,
    )
    if step <= 0:
        step = 50.0

    def _sorted_side(option_type: str) -> List[Dict[str, Any]]:
        return sorted(
            [row for row in rows if str(row.get("option_type") or "").upper() == option_type],
            key=lambda item: float(item.get("strike") or 0.0),
        )

    def _append_extras(items: List[Dict[str, Any]], *, option_type: str, direction: str) -> None:
        if len(items) < 2:
            return
        if direction == "UP":
            base_last = items[-1]
            base_prev = items[-2]
            strike_seed = float(base_last.get("strike") or 0.0)
        else:
            base_last = items[0]
            base_prev = items[1]
            strike_seed = float(base_last.get("strike") or 0.0)

        last_ltp = float(_safe_float(base_last.get("ltp"), 0.0) or 0.0)
        prev_ltp = float(_safe_float(base_prev.get("ltp"), 0.0) or 0.0)
        if prev_ltp > 0:
            ltp_ratio = _clamp(last_ltp / prev_ltp, 0.35, 0.85)
        else:
            ltp_ratio = 0.65
        last_oi = _safe_float(base_last.get("oi"))
        prev_oi = _safe_float(base_prev.get("oi"))
        if last_oi is not None and prev_oi not in (None, 0.0):
            oi_ratio = _clamp(float(last_oi) / float(prev_oi), 0.60, 1.05)
        else:
            oi_ratio = 0.85
        last_volume = _safe_float(base_last.get("volume"))
        last_iv = _safe_float(base_last.get("iv"))
        next_ltp = last_ltp
        next_oi = float(last_oi) if last_oi is not None else None
        next_volume = float(last_volume) if last_volume is not None else None
        for idx in range(1, max(1, int(min_extra_steps)) + 1):
            strike = strike_seed + (step * idx if direction == "UP" else -step * idx)
            next_ltp = max(0.5, next_ltp * ltp_ratio)
            next_oi = None if next_oi is None else max(1.0, next_oi * oi_ratio)
            next_volume = None if next_volume is None else max(1.0, next_volume * 0.75)
            rows.append(
                {
                    "option_type": option_type,
                    "strike": float(strike),
                    "ltp": round(float(next_ltp), 2),
                    "oi": None if next_oi is None else round(float(next_oi), 2),
                    "oi_change": None,
                    "iv": last_iv,
                    "volume": None if next_volume is None else round(float(next_volume), 2),
                    "delta": None,
                    "synthetic": True,
                }
            )

    _append_extras(_sorted_side("CE"), option_type="CE", direction="UP")
    _append_extras(_sorted_side("PE"), option_type="PE", direction="DOWN")
    deduped: Dict[Tuple[str, float], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("option_type") or "").upper(), float(row.get("strike") or 0.0))
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = row
        elif existing.get("synthetic") and not row.get("synthetic"):
            deduped[key] = row
    return sorted(deduped.values(), key=lambda item: (str(item["option_type"]), float(item["strike"])))


def _summarize_option_chain_context(
    chain_rows: List[Dict[str, Any]],
    *,
    spot: float,
    oi_skew_hint: Optional[float],
) -> Dict[str, Any]:
    rows = [r for r in chain_rows if isinstance(r, dict)]
    ce_rows = [r for r in rows if str(r.get("option_type") or "").upper() == "CE"]
    pe_rows = [r for r in rows if str(r.get("option_type") or "").upper() == "PE"]

    def _sum_oi(items: List[Dict[str, Any]]) -> float:
        return sum(float(item.get("oi") or 0.0) for item in items if item.get("oi") is not None)

    def _max_oi_row(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        with_oi = [r for r in items if r.get("oi") is not None]
        if not with_oi:
            return None
        return max(with_oi, key=lambda item: float(item.get("oi") or 0.0))

    total_call_oi = _sum_oi(ce_rows)
    total_put_oi = _sum_oi(pe_rows)
    pcr_total = (total_put_oi / total_call_oi) if total_call_oi > 0 else None

    near_band = max(100.0, min(400.0, round(float(spot or 0.0) * 0.01 / 50.0) * 50.0))
    near_ce = [r for r in ce_rows if abs(float(r.get("strike") or 0.0) - float(spot or 0.0)) <= near_band]
    near_pe = [r for r in pe_rows if abs(float(r.get("strike") or 0.0) - float(spot or 0.0)) <= near_band]
    near_call_oi = _sum_oi(near_ce)
    near_put_oi = _sum_oi(near_pe)
    pcr_near = (near_put_oi / near_call_oi) if near_call_oi > 0 else None

    call_wall = _max_oi_row(ce_rows)
    put_wall = _max_oi_row(pe_rows)
    call_wall_above = _max_oi_row([r for r in ce_rows if float(r.get("strike") or 0.0) >= float(spot or 0.0)])
    put_wall_below = _max_oi_row([r for r in pe_rows if float(r.get("strike") or 0.0) <= float(spot or 0.0)])

    near_call_oi_delta = sum(float(r.get("oi_change") or 0.0) for r in near_ce if r.get("oi_change") is not None)
    near_put_oi_delta = sum(float(r.get("oi_change") or 0.0) for r in near_pe if r.get("oi_change") is not None)
    oi_delta_points = sum(1 for r in near_ce + near_pe if r.get("oi_change") is not None)
    pcr_oi_change_near = None
    if abs(near_call_oi_delta) > 0:
        pcr_oi_change_near = near_put_oi_delta / near_call_oi_delta

    oi_build_bias = "UNKNOWN"
    delta_diff = near_put_oi_delta - near_call_oi_delta
    if oi_delta_points > 0 and abs(delta_diff) >= 1:
        if delta_diff > 0:
            oi_build_bias = "BULLISH_SUPPORT"
        elif delta_diff < 0:
            oi_build_bias = "BEARISH_RESISTANCE"
    elif oi_skew_hint is not None:
        if oi_skew_hint >= 0.10:
            oi_build_bias = "BULLISH_SUPPORT"
        elif oi_skew_hint <= -0.10:
            oi_build_bias = "BEARISH_RESISTANCE"
        else:
            oi_build_bias = "NEUTRAL"

    pcr_bias = "NEUTRAL"
    pcr_anchor = pcr_near if pcr_near is not None else pcr_total
    if pcr_anchor is not None:
        if pcr_anchor >= 1.15:
            pcr_bias = "BULLISH"
        elif pcr_anchor <= 0.85:
            pcr_bias = "BEARISH"

    def _wall_payload(item: Optional[Dict[str, Any]], *, side: str) -> Dict[str, Any]:
        strike = _safe_float((item or {}).get("strike"))
        oi = _safe_float((item or {}).get("oi"))
        distance = None
        if strike is not None:
            distance = float(strike) - float(spot or 0.0) if side == "CALL" else float(spot or 0.0) - float(strike)
        return {
            "strike": strike,
            "oi": oi,
            "distance_from_spot": None if distance is None else round(float(distance), 2),
        }

    return {
        "rows_count": len(rows),
        "spot": round(float(spot or 0.0), 2),
        "pcr_total": None if pcr_total is None else round(float(pcr_total), 3),
        "pcr_near_atm": None if pcr_near is None else round(float(pcr_near), 3),
        "pcr_bias": pcr_bias,
        "call_wall": _wall_payload(call_wall, side="CALL"),
        "put_wall": _wall_payload(put_wall, side="PUT"),
        "call_wall_above": _wall_payload(call_wall_above, side="CALL"),
        "put_wall_below": _wall_payload(put_wall_below, side="PUT"),
        "oi_build": {
            "available": oi_delta_points > 0 or oi_skew_hint is not None,
            "points": int(oi_delta_points),
            "near_call_oi_change": round(float(near_call_oi_delta), 2) if oi_delta_points else None,
            "near_put_oi_change": round(float(near_put_oi_delta), 2) if oi_delta_points else None,
            "pcr_oi_change_near": None if pcr_oi_change_near is None else round(float(pcr_oi_change_near), 3),
            "bias": oi_build_bias,
        },
    }


def _tf_snapshot(
    history: List[Tuple[datetime, float]],
    *,
    now: datetime,
    interval_min: int,
    spot: float,
    iv_rank: Optional[float],
    combined_premium_pct: Optional[float],
    prev_day_high: Optional[float],
    prev_day_low: Optional[float],
) -> Dict[str, Any]:
    def _ema(values: List[float], period: int) -> Optional[float]:
        if not values:
            return None
        alpha = 2.0 / (float(period) + 1.0)
        ema_val = float(values[0])
        for value in values[1:]:
            ema_val = (float(value) * alpha) + (ema_val * (1.0 - alpha))
        return ema_val

    window_start = now - timedelta(minutes=int(interval_min))
    window = [pt for pt in history if pt[0] >= window_start]
    if len(window) < 2:
        window = history[-max(2, min(len(history), interval_min)) :]
    spots = [float(value) for _, value in window] if window else [float(spot or 0.0)]
    first = spots[0]
    last = spots[-1]
    low = min(spots)
    high = max(spots)
    range_pts = max(0.0, high - low)
    change = last - first
    change_pct = (change / max(1.0, abs(first))) * 100.0
    close_pos = (last - low) / range_pts if range_pts > 0 else 0.5
    base_thresh = max(float(spot or 0.0) * 0.00045, 12.0, range_pts * 0.28)
    if change >= base_thresh and close_pos >= 0.62:
        pattern = "UPTREND"
        trend = "BULLISH"
    elif change <= -base_thresh and close_pos <= 0.38:
        pattern = "DOWNTREND"
        trend = "BEARISH"
    else:
        pattern = "SIDEWAYS"
        trend = "RANGE"
    dir_score = _clamp(change / max(range_pts, 35.0), -1.0, 1.0)
    range_pct = range_pts / max(float(spot or 0.0), 1.0)
    ema_fast = _ema(spots, 5)
    ema_slow = _ema(spots, 20)
    ema_base = _ema(spots, 50)
    ema_fast_prev = _ema(spots[:-1], 5) if len(spots) > 1 else ema_fast
    ema_slow_prev = _ema(spots[:-1], 20) if len(spots) > 1 else ema_slow
    ema_base_prev = _ema(spots[:-1], 50) if len(spots) > 1 else ema_base
    ema_gap_pct = None
    ema_alignment = "NEUTRAL"
    if ema_fast is not None and ema_slow is not None:
        ema_gap_pct = (float(ema_fast - ema_slow) / max(1.0, abs(last)))
        fast_slope = 0.0 if ema_fast_prev is None else float(ema_fast - ema_fast_prev)
        slow_slope = 0.0 if ema_slow_prev is None else float(ema_slow - ema_slow_prev)
        if ema_gap_pct >= 0.0008 and fast_slope > 0 and slow_slope >= 0:
            ema_alignment = "BULLISH"
        elif ema_gap_pct <= -0.0008 and fast_slope < 0 and slow_slope <= 0:
            ema_alignment = "BEARISH"
    ema_base_slope = None if ema_base is None or ema_base_prev is None else float(ema_base - ema_base_prev)
    ema_20_50_gap_pct = None
    ema_20_50_alignment = "NEUTRAL"
    if ema_slow is not None and ema_base is not None:
        ema_20_50_gap_pct = (float(ema_slow - ema_base) / max(1.0, abs(last)))
        slow_slope = 0.0 if ema_slow_prev is None else float(ema_slow - ema_slow_prev)
        base_slope = 0.0 if ema_base_slope is None else float(ema_base_slope)
        if ema_20_50_gap_pct >= 0.0008 and slow_slope > 0 and base_slope >= 0:
            ema_20_50_alignment = "BULLISH"
        elif ema_20_50_gap_pct <= -0.0008 and slow_slope < 0 and base_slope <= 0:
            ema_20_50_alignment = "BEARISH"
    vol_regime = "NORMAL"
    if (iv_rank is not None and iv_rank >= 0.75) or (combined_premium_pct is not None and combined_premium_pct >= 0.022) or range_pct >= 0.007:
        vol_regime = "HIGH"
    elif (iv_rank is not None and iv_rank <= 0.35) and (combined_premium_pct is None or combined_premium_pct <= 0.015) and range_pct <= 0.0035:
        vol_regime = "LOW"
    breakout_confirmed = False
    breakout_dir = "NONE"
    breakout_buffer = max(float(spot or 0.0) * 0.0006, 20.0)
    if prev_day_high is not None and float(spot or 0.0) >= float(prev_day_high) + breakout_buffer:
        breakout_confirmed = True
        breakout_dir = "UP"
    elif prev_day_low is not None and float(spot or 0.0) <= float(prev_day_low) - breakout_buffer:
        breakout_confirmed = True
        breakout_dir = "DOWN"
    elif interval_min >= 15 and pattern in {"UPTREND", "DOWNTREND"} and abs(dir_score) >= 0.68:
        breakout_confirmed = True
        breakout_dir = "UP" if pattern == "UPTREND" else "DOWN"
    return {
        "interval_min": int(interval_min),
        "trend": trend,
        "pattern": pattern,
        "bars": len(spots),
        "change_points": round(float(change), 2),
        "change_pct": round(float(change_pct), 3),
        "range_points": round(float(range_pts), 2),
        "close_position_in_range": round(float(close_pos), 3),
        "dir_score": round(float(dir_score), 3),
        "support": round(float(low), 2),
        "resistance": round(float(high), 2),
        "distance_to_support": round(float(last - low), 2),
        "distance_to_resistance": round(float(high - last), 2),
        "atr_like_points": round(float(range_pts), 2),
        "ema_fast": None if ema_fast is None else round(float(ema_fast), 2),
        "ema_slow": None if ema_slow is None else round(float(ema_slow), 2),
        "ema_base": None if ema_base is None else round(float(ema_base), 2),
        "ema_gap_pct": None if ema_gap_pct is None else round(float(ema_gap_pct), 4),
        "ema_alignment": ema_alignment,
        "ema_base_slope": None if ema_base_slope is None else round(float(ema_base_slope), 2),
        "ema_20_50_gap_pct": None if ema_20_50_gap_pct is None else round(float(ema_20_50_gap_pct), 4),
        "ema_20_50_alignment": ema_20_50_alignment,
        "volatility_regime": vol_regime,
        "breakout_confirmed": bool(breakout_confirmed),
        "breakout_dir": breakout_dir,
    }


def _recent_interval_candles_from_history(
    history: List[Tuple[datetime, float]],
    *,
    now: datetime,
    interval_min: int,
    count: int = 6,
) -> List[Dict[str, float]]:
    if not history:
        return []
    day_points = [(ts, float(value)) for ts, value in history if ts.date() == now.date() and ts <= now]
    if not day_points:
        return []
    buckets: Dict[datetime, List[float]] = {}
    for ts, value in day_points:
        bucket_minute = (ts.minute // interval_min) * interval_min
        bucket_start = ts.replace(minute=bucket_minute, second=0, microsecond=0)
        buckets.setdefault(bucket_start, []).append(float(value))
    candles: List[Dict[str, float]] = []
    for bucket_start in sorted(buckets.keys())[-max(3, count):]:
        values = buckets[bucket_start]
        candles.append(
            {
                "open": float(values[0]),
                "high": float(max(values)),
                "low": float(min(values)),
                "close": float(values[-1]),
            }
        )
    return candles


def _price_action_from_interval_candles(
    candles: List[Dict[str, float]],
    *,
    support: Optional[float],
    resistance: Optional[float],
    atr_like_points: float,
    ema_slow: Optional[float] = None,
    ema_base: Optional[float] = None,
) -> Dict[str, Any]:
    return detect_recent_price_action(
        candles,
        support=support,
        resistance=resistance,
        atr_like_points=atr_like_points,
        ema_slow=ema_slow,
        ema_base=ema_base,
    )


def _build_trend_context(
    history: List[Tuple[datetime, float]],
    *,
    now: datetime,
    row: pd.Series,
) -> Dict[str, Any]:
    spot = float(_safe_float(row.get("spot"), 0.0) or 0.0)
    iv_rank = _safe_float(row.get("iv_rank"))
    combined_premium_pct = _safe_float(row.get("combined_premium_pct"))
    prev_day_high = _safe_float(row.get("prev_day_high"))
    prev_day_low = _safe_float(row.get("prev_day_low"))
    per_tf: Dict[str, Dict[str, Any]] = {}
    weighted_sum = 0.0
    weighted_denom = 0.0
    high_vol_votes = 0
    low_vol_votes = 0
    breakout_votes: List[str] = []
    for interval, weight in ((5, 1.0), (15, 2.0), (60, 3.0)):
        snap = _tf_snapshot(
            history,
            now=now,
            interval_min=interval,
            spot=spot,
            iv_rank=iv_rank,
            combined_premium_pct=combined_premium_pct,
            prev_day_high=prev_day_high,
            prev_day_low=prev_day_low,
        )
        interval_candles = _recent_interval_candles_from_history(history, now=now, interval_min=interval)
        snap.update(
            _price_action_from_interval_candles(
                interval_candles,
                support=_safe_float(snap.get("support")),
                resistance=_safe_float(snap.get("resistance")),
                atr_like_points=float(_safe_float(snap.get("atr_like_points"), 0.0) or 0.0),
                ema_slow=_safe_float(snap.get("ema_slow")),
                ema_base=_safe_float(snap.get("ema_base")),
            )
        )
        per_tf[str(interval)] = snap
        weighted_sum += float(snap.get("dir_score") or 0.0) * weight
        weighted_denom += weight
        if str(snap.get("volatility_regime") or "").upper() == "HIGH":
            high_vol_votes += 1
        elif str(snap.get("volatility_regime") or "").upper() == "LOW":
            low_vol_votes += 1
        if bool(snap.get("breakout_confirmed")):
            breakout_votes.append(str(snap.get("breakout_dir") or "NONE").upper())
    bias_score = (weighted_sum / weighted_denom) if weighted_denom > 0 else 0.0
    if _safe_float(row.get("oi_skew")) is not None:
        bias_score = (bias_score * 0.8) + (_clamp(float(_safe_float(row.get("oi_skew"), 0.0) or 0.0), -1.0, 1.0) * 0.2)
    bias = "NEUTRAL"
    if bias_score >= 0.35:
        bias = "BULLISH"
    elif bias_score <= -0.35:
        bias = "BEARISH"
    volatility_regime = "NORMAL"
    if high_vol_votes >= 2:
        volatility_regime = "HIGH"
    elif low_vol_votes >= 2:
        volatility_regime = "LOW"
    breakout_confirmation = "NONE"
    up_votes = sum(1 for vote in breakout_votes if vote == "UP")
    down_votes = sum(1 for vote in breakout_votes if vote == "DOWN")
    if up_votes > down_votes and up_votes >= 1:
        breakout_confirmation = "UP_CONFIRMED"
    elif down_votes > up_votes and down_votes >= 1:
        breakout_confirmation = "DOWN_CONFIRMED"
    elif per_tf["15"]["pattern"] == "UPTREND" and per_tf["60"]["pattern"] == "UPTREND" and bias_score >= 0.58:
        breakout_confirmation = "UP_CONFIRMED"
    elif per_tf["15"]["pattern"] == "DOWNTREND" and per_tf["60"]["pattern"] == "DOWNTREND" and bias_score <= -0.58:
        breakout_confirmation = "DOWN_CONFIRMED"
    orb_confirmation = "NONE"
    day_history = [pt for pt in history if pt[0].date() == now.date()]
    if day_history:
        orb_window = [pt for pt in day_history if pt[0].time() <= dtime(9, 30)]
        if len(orb_window) >= 2:
            orb_high = max(value for _, value in orb_window)
            orb_low = min(value for _, value in orb_window)
            orb_buffer = max(float(spot or 0.0) * 0.0003, 5.0)
            if float(spot or 0.0) >= float(orb_high) + orb_buffer:
                orb_confirmation = "UP_CONFIRMED"
            elif float(spot or 0.0) <= float(orb_low) - orb_buffer:
                orb_confirmation = "DOWN_CONFIRMED"
    pivot_zones: List[Dict[str, Any]] = []
    swing_structure: Dict[str, Any] = {
        "bias": "NEUTRAL",
        "label": "RANGE",
        "confidence": 0.0,
        "reasons": [],
        "timeframes": {},
        "nearest_support": None,
        "nearest_resistance": None,
    }
    try:
        zone_frames: Dict[str, pd.DataFrame] = {}
        for interval in (5, 15, 60):
            interval_candles = _recent_interval_candles_from_history(history, now=now, interval_min=interval)
            frame = _candles_to_zone_frame(interval_candles)
            if not frame.empty:
                zone_frames[f"{interval}m"] = frame
        if zone_frames:
            swing_zones = find_multi_tf_zones(zone_frames)
            nearest_support, nearest_resistance = nearest_zones(swing_zones, float(spot or 0.0))
            pivot_zones = [
                {
                    "side": str(getattr(zone, "side", "")),
                    "price": round(float(getattr(zone, "price", 0.0)), 2),
                    "timeframe": str(getattr(zone, "timeframe", "")),
                    "timestamp": getattr(zone, "timestamp", now).isoformat(timespec="seconds"),
                }
                for zone in swing_zones[-24:]
            ]
            swing_structure = _summarize_swing_structure(swing_zones)
            if nearest_support is not None:
                swing_structure["nearest_support"] = {
                    "side": str(getattr(nearest_support, "side", "")),
                    "price": round(float(getattr(nearest_support, "price", 0.0)), 2),
                    "timeframe": str(getattr(nearest_support, "timeframe", "")),
                    "timestamp": getattr(nearest_support, "timestamp", now).isoformat(timespec="seconds"),
                }
            if nearest_resistance is not None:
                swing_structure["nearest_resistance"] = {
                    "side": str(getattr(nearest_resistance, "side", "")),
                    "price": round(float(getattr(nearest_resistance, "price", 0.0)), 2),
                    "timeframe": str(getattr(nearest_resistance, "timeframe", "")),
                    "timestamp": getattr(nearest_resistance, "timestamp", now).isoformat(timespec="seconds"),
                }
    except Exception:
        pass
    return {
        "timeframes": per_tf,
        "bias_score": round(float(bias_score), 3),
        "bias": bias,
        "volatility_regime": volatility_regime,
        "breakout_confirmation": breakout_confirmation,
        "pivot_zones": pivot_zones,
        "swing_structure": swing_structure,
        "orb": {
            "timeframe_minutes": 15,
            "breakout_confirmation": orb_confirmation,
            "breakout_active": orb_confirmation != "NONE",
        },
        "errors": [],
    }


def _build_structure_context(
    *,
    spot: float,
    trend_ctx: Dict[str, Any],
    oc_ctx: Dict[str, Any],
    daily_candles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    tf_map = trend_ctx.get("timeframes") if isinstance(trend_ctx.get("timeframes"), dict) else {}
    spot_f = float(spot or 0.0)

    def _pick_nearest_support(cands: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        below = [c for c in cands if _safe_float(c.get("level")) is not None and float(c["level"]) <= spot_f]
        pool = below if below else [c for c in cands if _safe_float(c.get("level")) is not None]
        pool = sorted(pool, key=lambda c: abs(spot_f - float(c["level"])))
        return (pool[0] if pool else None, pool[1] if len(pool) > 1 else None)

    def _pick_nearest_resistance(cands: List[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        above = [c for c in cands if _safe_float(c.get("level")) is not None and float(c["level"]) >= spot_f]
        pool = above if above else [c for c in cands if _safe_float(c.get("level")) is not None]
        pool = sorted(pool, key=lambda c: abs(float(c["level"]) - spot_f))
        return (pool[0] if pool else None, pool[1] if len(pool) > 1 else None)

    def _level_payload(item: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        level = _safe_float((item or {}).get("level"))
        distance = None if level is None else abs(float(level) - spot_f)
        return {
            "level": None if level is None else round(float(level), 2),
            "source": (item or {}).get("source"),
            "strength": (item or {}).get("strength"),
            "distance_from_spot": None if distance is None else round(float(distance), 2),
        }

    intraday_support_cands: List[Dict[str, Any]] = []
    intraday_resistance_cands: List[Dict[str, Any]] = []
    swing_support_cands: List[Dict[str, Any]] = []
    swing_resistance_cands: List[Dict[str, Any]] = []
    for tf_key, tf_weight in (("5", 1.0), ("15", 1.5), ("60", 2.0)):
        tf = tf_map.get(tf_key) if isinstance(tf_map.get(tf_key), dict) else {}
        support = _safe_float(tf.get("support"))
        resistance = _safe_float(tf.get("resistance"))
        if support is not None:
            intraday_support_cands.append({"level": support, "source": f"{tf_key}m_price", "strength": tf_weight})
        if resistance is not None:
            intraday_resistance_cands.append({"level": resistance, "source": f"{tf_key}m_price", "strength": tf_weight})

    pivot_zones = trend_ctx.get("pivot_zones") if isinstance(trend_ctx.get("pivot_zones"), list) else []
    pivot_tf_weights = {"5m": 1.2, "15m": 1.9, "60m": 2.5, "1h": 2.5}
    for zone in pivot_zones:
        if not isinstance(zone, dict):
            continue
        level = _safe_float(zone.get("price"))
        side = str(zone.get("side") or "").lower()
        tf_label = str(zone.get("timeframe") or "")
        strength = float(pivot_tf_weights.get(tf_label, 1.0))
        payload = {"level": level, "source": f"swing_{tf_label}_{side}", "strength": strength}
        if level is None:
            continue
        if side == "support":
            swing_support_cands.append(payload)
            intraday_support_cands.append(payload)
        elif side == "resistance":
            swing_resistance_cands.append(payload)
            intraday_resistance_cands.append(payload)

    put_wall_below = oc_ctx.get("put_wall_below") if isinstance(oc_ctx.get("put_wall_below"), dict) else {}
    call_wall_above = oc_ctx.get("call_wall_above") if isinstance(oc_ctx.get("call_wall_above"), dict) else {}
    put_wall = oc_ctx.get("put_wall") if isinstance(oc_ctx.get("put_wall"), dict) else {}
    call_wall = oc_ctx.get("call_wall") if isinstance(oc_ctx.get("call_wall"), dict) else {}
    for wall, source, strength in (
        (put_wall_below, "oi_put_wall_below", 2.5),
        (put_wall, "oi_put_wall", 1.5),
    ):
        level = _safe_float(wall.get("strike"))
        if level is not None:
            intraday_support_cands.append({"level": level, "source": source, "strength": strength})
    for wall, source, strength in (
        (call_wall_above, "oi_call_wall_above", 2.5),
        (call_wall, "oi_call_wall", 1.5),
    ):
        level = _safe_float(wall.get("strike"))
        if level is not None:
            intraday_resistance_cands.append({"level": level, "source": source, "strength": strength})

    support_1, support_2 = _pick_nearest_support(intraday_support_cands)
    resistance_1, resistance_2 = _pick_nearest_resistance(intraday_resistance_cands)
    swing_support_1, swing_support_2 = _pick_nearest_support(swing_support_cands)
    swing_resistance_1, swing_resistance_2 = _pick_nearest_resistance(swing_resistance_cands)

    daily_window = daily_candles[-3:] if len(daily_candles) >= 3 else daily_candles
    daily_support = min((float(item["low"]) for item in daily_window), default=None)
    daily_resistance = max((float(item["high"]) for item in daily_window), default=None)
    weekly_window = daily_candles[-5:] if len(daily_candles) >= 5 else daily_candles
    weekly_support = min((float(item["low"]) for item in weekly_window), default=None)
    weekly_resistance = max((float(item["high"]) for item in weekly_window), default=None)
    weekly_mid = None
    if weekly_support is not None and weekly_resistance is not None:
        weekly_mid = (float(weekly_support) + float(weekly_resistance)) / 2.0
    weekly_close = float(weekly_window[-1]["close"]) if weekly_window else None
    weekly_bias = "UNKNOWN"
    if len(weekly_window) >= 2:
        first_close = float(weekly_window[0]["close"])
        last_close = float(weekly_window[-1]["close"])
        move_pct = ((last_close - first_close) / max(1.0, abs(first_close))) * 100.0
        if move_pct >= 0.35:
            weekly_bias = "BULLISH"
        elif move_pct <= -0.35:
            weekly_bias = "BEARISH"
        else:
            weekly_bias = "RANGE"

    price_action_patterns: List[str] = []
    price_action_weighted = 0.0
    price_action_weight_total = 0.0
    candle_confirm_hits = 0
    retest_confirm_hits = 0
    reversal_confirm_hits = 0
    breakout_confirm_hits = 0
    retest_status = "NONE"
    retest_bias = "NEUTRAL"
    retest_level = None
    retest_score = 0.0
    for tf_key, tf_weight in (("5", 1.0), ("15", 1.7), ("60", 0.7)):
        tf = tf_map.get(tf_key) if isinstance(tf_map.get(tf_key), dict) else {}
        pa_bias = str(tf.get("price_action_bias") or "NEUTRAL").upper()
        pa_conf = str(tf.get("price_action_confirmation") or "NONE").upper()
        if pa_bias == "BULLISH":
            price_action_weighted += tf_weight
            price_action_weight_total += tf_weight
        elif pa_bias == "BEARISH":
            price_action_weighted -= tf_weight
            price_action_weight_total += tf_weight
        if pa_conf in {"CANDLE_CONFIRMED", "CANDLE_AND_RETEST_CONFIRMED", "REVERSAL_CONFIRMED", "BREAKOUT_CONFIRMED"}:
            candle_confirm_hits += 1
        if pa_conf in {"RETEST_CONFIRMED", "CANDLE_AND_RETEST_CONFIRMED"}:
            retest_confirm_hits += 1
        if pa_conf == "REVERSAL_CONFIRMED":
            reversal_confirm_hits += 1
        elif pa_conf == "BREAKOUT_CONFIRMED":
            breakout_confirm_hits += 1
        for pattern in list(tf.get("price_action_patterns") or []):
            text = f"{tf_key}m:{str(pattern or '').upper()}"
            if text not in price_action_patterns:
                price_action_patterns.append(text)
        tf_retest_score = _safe_float(tf.get("retest_score"), 0.0) or 0.0
        if tf_retest_score >= float(retest_score):
            retest_status = str(tf.get("retest_status") or "NONE").upper()
            retest_bias = str(tf.get("retest_bias") or "NEUTRAL").upper()
            retest_level = _safe_float(tf.get("retest_level"))
            retest_score = float(tf_retest_score)

    price_action_bias = "NEUTRAL"
    if price_action_weight_total > 0:
        pa_score = price_action_weighted / price_action_weight_total
        if pa_score >= 0.35:
            price_action_bias = "BULLISH"
        elif pa_score <= -0.35:
            price_action_bias = "BEARISH"
    price_action_confirmation = "NONE"
    if candle_confirm_hits > 0 and retest_confirm_hits > 0 and price_action_bias != "NEUTRAL":
        price_action_confirmation = "CANDLE_AND_RETEST_CONFIRMED"
    elif reversal_confirm_hits > 0 and price_action_bias != "NEUTRAL":
        price_action_confirmation = "REVERSAL_CONFIRMED"
    elif breakout_confirm_hits > 0 and price_action_bias != "NEUTRAL":
        price_action_confirmation = "BREAKOUT_CONFIRMED"
    elif candle_confirm_hits > 0 and price_action_bias != "NEUTRAL":
        price_action_confirmation = "CANDLE_CONFIRMED"
    elif retest_confirm_hits > 0 and retest_bias != "NEUTRAL":
        price_action_bias = retest_bias
        price_action_confirmation = "RETEST_CONFIRMED"

    def _vote(label: Any, bullish_tokens: Tuple[str, ...], bearish_tokens: Tuple[str, ...]) -> int:
        text = str(label or "").upper()
        if any(token in text for token in bullish_tokens):
            return 1
        if any(token in text for token in bearish_tokens):
            return -1
        return 0

    trend_vote = _vote(trend_ctx.get("bias"), ("BULL",), ("BEAR",))
    pcr_vote = _vote(oc_ctx.get("pcr_bias"), ("BULL",), ("BEAR",))
    oi_build = oc_ctx.get("oi_build") if isinstance(oc_ctx.get("oi_build"), dict) else {}
    oi_vote = _vote(oi_build.get("bias"), ("BULL", "SUPPORT"), ("BEAR", "RESISTANCE"))
    breakout_vote = _vote(trend_ctx.get("breakout_confirmation"), ("UP", "BULL"), ("DOWN", "BEAR"))
    weekly_vote = _vote(weekly_bias, ("BULL",), ("BEAR",))
    swing_structure = trend_ctx.get("swing_structure") if isinstance(trend_ctx.get("swing_structure"), dict) else {}
    swing_bias = str(swing_structure.get("bias") or "NEUTRAL").upper()
    swing_confidence = float(_safe_float(swing_structure.get("confidence"), 0.0) or 0.0)
    swing_vote = _vote(swing_bias, ("BULL",), ("BEAR",))
    candle_vote = 1 if (price_action_bias == "BULLISH" and candle_confirm_hits > 0) else (-1 if (price_action_bias == "BEARISH" and candle_confirm_hits > 0) else 0)
    retest_fail_statuses = {"NONE", "RETEST_FAILED", "FAILED_BREAKOUT_RETEST", "FAILED_BREAKDOWN_RETEST"}
    retest_vote = 1 if (retest_bias == "BULLISH" and retest_status not in retest_fail_statuses) else (-1 if (retest_bias == "BEARISH" and retest_status not in retest_fail_statuses) else 0)
    votes = {
        "trend": trend_vote,
        "pcr": pcr_vote,
        "oi_build": oi_vote,
        "breakout": breakout_vote,
        "weekly": weekly_vote,
        "swing": swing_vote,
        "candle": candle_vote,
        "retest": retest_vote,
    }
    non_zero_votes = [vote for vote in votes.values() if vote != 0]
    bullish_votes = sum(1 for vote in non_zero_votes if vote > 0)
    bearish_votes = sum(1 for vote in non_zero_votes if vote < 0)
    known_votes = len(non_zero_votes)
    dominant_votes = max(bullish_votes, bearish_votes) if known_votes else 0
    conflict_votes = max(0, known_votes - dominant_votes)
    conflict_ratio = (conflict_votes / known_votes) if known_votes else 0.0
    trend_bias_score = abs(float(_safe_float(trend_ctx.get("bias_score"), 0.0) or 0.0))
    base_conf = (dominant_votes / known_votes) if known_votes else 0.0
    trend_confidence = (0.55 * base_conf) + (0.25 * min(1.0, trend_bias_score)) + (0.20 * (1.0 - conflict_ratio))

    pcr_total = _safe_float(oc_ctx.get("pcr_total"))
    pcr_near = _safe_float(oc_ctx.get("pcr_near_atm"))
    pcr_extreme = None
    for value in (pcr_near, pcr_total):
        if value is None:
            continue
        if pcr_extreme is None or abs(value - 1.0) > abs(pcr_extreme - 1.0):
            pcr_extreme = value
    pcr_unbalanced = False
    pcr_unbalanced_side = "NEUTRAL"
    if pcr_extreme is not None:
        if pcr_extreme >= 1.30:
            pcr_unbalanced = True
            pcr_unbalanced_side = "BULLISH"
            trend_confidence += 0.05 if bullish_votes >= bearish_votes else -0.05
        elif pcr_extreme <= 0.70:
            pcr_unbalanced = True
            pcr_unbalanced_side = "BEARISH"
            trend_confidence += 0.05 if bearish_votes >= bullish_votes else -0.05

    sr_alignment_hits = 0
    for a, b in (
        (support_1, {"level": _safe_float(put_wall_below.get("strike"))}),
        (resistance_1, {"level": _safe_float(call_wall_above.get("strike"))}),
    ):
        la = _safe_float((a or {}).get("level"))
        lb = _safe_float((b or {}).get("level"))
        if la is not None and lb is not None and abs(la - lb) <= 100.0:
            sr_alignment_hits += 1
    if sr_alignment_hits:
        trend_confidence += 0.03 * sr_alignment_hits
    if swing_vote != 0 and swing_confidence > 0:
        trend_confidence += 0.05 * min(1.0, swing_confidence)
    if price_action_confirmation in {"CANDLE_CONFIRMED", "RETEST_CONFIRMED", "REVERSAL_CONFIRMED", "BREAKOUT_CONFIRMED"}:
        trend_confidence += 0.04
    elif price_action_confirmation == "CANDLE_AND_RETEST_CONFIRMED":
        trend_confidence += 0.07

    volatility_regime = str(trend_ctx.get("volatility_regime") or "NORMAL").upper()
    if volatility_regime == "HIGH" and conflict_ratio > 0.25:
        trend_confidence -= 0.06
    trend_confidence = _clamp(float(trend_confidence), 0.05, 0.98)

    signal_conflict_score = _clamp(conflict_ratio * 100.0, 0.0, 100.0)
    if known_votes >= 3 and bullish_votes > 0 and bearish_votes > 0:
        signal_conflict_score = min(100.0, signal_conflict_score + 10.0)
    dominant_bias = "NEUTRAL"
    if bullish_votes > bearish_votes and bullish_votes >= 2:
        dominant_bias = "BULLISH"
    elif bearish_votes > bullish_votes and bearish_votes >= 2:
        dominant_bias = "BEARISH"
    return {
        "intraday_support": _level_payload(support_1),
        "intraday_support_secondary": _level_payload(support_2),
        "intraday_resistance": _level_payload(resistance_1),
        "intraday_resistance_secondary": _level_payload(resistance_2),
        "swing_support": _level_payload(swing_support_1),
        "swing_support_secondary": _level_payload(swing_support_2),
        "swing_resistance": _level_payload(swing_resistance_1),
        "swing_resistance_secondary": _level_payload(swing_resistance_2),
        "swing_structure_bias": swing_bias,
        "swing_structure_label": str(swing_structure.get("label") or "RANGE"),
        "swing_structure_confidence": round(float(swing_confidence), 3),
        "swing_structure_reasons": list(swing_structure.get("reasons") or [])[:6],
        "daily_support": {
            "level": None if daily_support is None else round(float(daily_support), 2),
            "source": "daily_3d_low",
            "strength": 1.8,
            "distance_from_spot": None if daily_support is None else round(abs(float(daily_support) - spot_f), 2),
        },
        "daily_resistance": {
            "level": None if daily_resistance is None else round(float(daily_resistance), 2),
            "source": "daily_3d_high",
            "strength": 1.8,
            "distance_from_spot": None if daily_resistance is None else round(abs(float(daily_resistance) - spot_f), 2),
        },
        "weekly_support": None if weekly_support is None else round(float(weekly_support), 2),
        "weekly_resistance": None if weekly_resistance is None else round(float(weekly_resistance), 2),
        "weekly_mid": None if weekly_mid is None else round(float(weekly_mid), 2),
        "weekly_close": None if weekly_close is None else round(float(weekly_close), 2),
        "weekly_bias": weekly_bias,
        "weekly_window_days": len(weekly_window),
        "trend_confidence": round(float(trend_confidence), 3),
        "signal_conflict_score": round(float(signal_conflict_score), 1),
        "dominant_signal_bias": dominant_bias,
        "votes": votes,
        "bullish_votes": bullish_votes,
        "bearish_votes": bearish_votes,
        "known_votes": known_votes,
        "conflict_votes": conflict_votes,
        "pcr_unbalanced": bool(pcr_unbalanced),
        "pcr_unbalanced_side": pcr_unbalanced_side,
        "pcr_extreme": None if pcr_extreme is None else round(float(pcr_extreme), 3),
        "sr_alignment_hits": sr_alignment_hits,
        "volatility_regime": volatility_regime,
        "price_action_bias": price_action_bias,
        "price_action_confirmation": price_action_confirmation,
        "primary_pattern": price_action_patterns[0] if price_action_patterns else "NONE",
        "price_action_patterns": price_action_patterns[:6],
        "retest_status": retest_status,
        "retest_bias": retest_bias,
        "retest_level": None if retest_level is None else round(float(retest_level), 2),
        "retest_score": round(float(retest_score), 3),
    }


def _rec_leg_to_option_leg(
    *,
    rec_leg: Dict[str, Any],
    strategy_name: str,
    expiry: date,
    lot_size: int,
    chain_rows: List[Dict[str, Any]],
) -> OptionLeg:
    side_raw = str(rec_leg.get("side") or "").upper()
    opt_raw = str(rec_leg.get("option_type") or "").upper()
    strike = float(rec_leg.get("strike") or 0.0)
    qty_lots = max(1, int(float(rec_leg.get("qty_lots") or 1)))
    qty = int(max(1, lot_size) * qty_lots)
    chain_map = {
        (str(row.get("option_type") or "").upper(), float(row.get("strike") or 0.0)): row
        for row in (chain_rows or [])
        if _safe_float(row.get("strike")) is not None
    }
    row = chain_map.get((opt_raw, float(strike))) or {}
    ltp = float(_safe_float(row.get("ltp"), 0.0) or 0.0)
    return OptionLeg(
        symbol="NIFTY",
        expiry=expiry,
        strike=float(strike),
        option_type=OptionType.CALL if opt_raw == "CE" else OptionType.PUT,
        side=LegSide.BUY if side_raw == "BUY" else LegSide.SELL,
        quantity=int(qty),
        entry_price=float(ltp),
        security_id=None,
        strategy_type=_strategy_type_enum(strategy_name),
    )


def _load_history(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for line in path.read_text().splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _trade_metrics(trades: pd.DataFrame) -> Dict[str, Any]:
    if trades.empty:
        return {
            "closed_trades": 0,
            "wins": 0,
            "losses": 0,
            "flat": 0,
            "win_rate_pct": 0.0,
            "realized_pnl_rs": 0.0,
            "avg_pnl_rs": 0.0,
            "median_pnl_rs": 0.0,
            "avg_hold_min": 0.0,
            "median_hold_min": 0.0,
            "best_trade_rs": 0.0,
            "worst_trade_rs": 0.0,
            "exit_reason_counts": {},
            "strategy_counts": {},
            "positive_days": 0,
            "trading_days": 0,
            "positive_day_rate_pct": 0.0,
            "avg_daily_pnl_rs": 0.0,
        }
    pnl = trades["pnl_rs"].astype(float)
    holds = trades["hold_minutes"].fillna(0.0).astype(float)
    realized = float(pnl.sum())
    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    flat = int((pnl == 0).sum())
    daily = trades.groupby("current_session_date", dropna=True)["pnl_rs"].sum()
    return {
        "closed_trades": int(len(trades)),
        "wins": wins,
        "losses": losses,
        "flat": flat,
        "win_rate_pct": round((wins / len(trades)) * 100.0, 2),
        "realized_pnl_rs": round(realized, 2),
        "avg_pnl_rs": round(float(pnl.mean()), 2),
        "median_pnl_rs": round(float(pnl.median()), 2),
        "avg_hold_min": round(float(holds.mean()), 2),
        "median_hold_min": round(float(holds.median()), 2),
        "best_trade_rs": round(float(pnl.max()), 2),
        "worst_trade_rs": round(float(pnl.min()), 2),
        "exit_reason_counts": {str(key): int(val) for key, val in Counter(trades["reason"]).items()},
        "strategy_counts": {str(key): int(val) for key, val in Counter(trades["strategy_type"]).items()},
        "positive_days": int((daily > 0).sum()),
        "trading_days": int(len(daily)),
        "positive_day_rate_pct": round(float(((daily > 0).sum() / len(daily)) * 100.0), 2) if len(daily) else 0.0,
        "avg_daily_pnl_rs": round(float(daily.mean()), 2) if len(daily) else 0.0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay backtest for the current intraday AI advisor and position manager.")
    parser.add_argument("--input", type=Path, required=True, help="Rolling intraday CSV input.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to write signals, trades and summary.")
    parser.add_argument("--max-trades-per-session", type=int, default=1, help="Maximum intraday trades per session.")
    parser.add_argument("--lot-size", type=int, default=65, help="Lot size for one set.")
    parser.add_argument("--entry-not-before", default="09:45", help="Advisor entry not before time (HH:MM).")
    parser.add_argument("--last-new-entry-time", default="14:20", help="Advisor last new entry time (HH:MM).")
    parser.add_argument("--max-hold-till", default="15:05", help="Max hold time (HH:MM).")
    parser.add_argument("--preferred-bias", default="NEUTRAL", help="NEUTRAL, BULLISH or BEARISH.")
    parser.add_argument(
        "--directional-structure",
        default="CREDIT_SPREAD",
        help="CREDIT_SPREAD or SHORT_OPTION_WITH_HEDGE.",
    )
    parser.add_argument("--naked-max-loss-rs", type=float, default=3000.0, help="Operational max loss per set.")
    parser.add_argument("--naked-trail-arm-rs", type=float, default=5000.0, help="Profit at which trailing becomes active.")
    parser.add_argument("--naked-trail-keep-pct", type=float, default=0.72, help="Fraction of peak PnL to retain after trailing arms.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    if "timestamp" not in df.columns or "spot" not in df.columns:
        raise ValueError("Input CSV must contain at least 'timestamp' and 'spot' columns.")

    df["timestamp"] = df["timestamp"].apply(_parse_ts)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["trade_date"] = df["timestamp"].apply(lambda ts: ts.date())

    daily_candle_map: Dict[date, Dict[str, Any]] = {}
    for trade_date, day_df in df.groupby("trade_date", sort=True):
        spots = day_df["spot"].astype(float)
        daily_candle_map[trade_date] = {
            "date": trade_date.isoformat(),
            "open": float(spots.iloc[0]),
            "high": float(spots.max()),
            "low": float(spots.min()),
            "close": float(spots.iloc[-1]),
        }

    prefixes = _discover_option_prefixes(df.columns)
    if not prefixes:
        raise ValueError("Input CSV does not contain recognizable option chain strike columns.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    advisor_status_path = args.output_dir / "intraday_ai_advisor_status.json"
    committee_status_path = args.output_dir / "decision_committee_status.json"
    committee_outcomes_path = args.output_dir / "decision_committee_outcomes.jsonl"
    learning_status_path = args.output_dir / "intraday_ai_learning_status.json"
    position_state_path = args.output_dir / "intraday_ai_position_status.json"
    history_path = args.output_dir / "intraday_ai_trade_history.jsonl"
    for path in (
        advisor_status_path,
        committee_status_path,
        committee_outcomes_path,
        learning_status_path,
        position_state_path,
        history_path,
    ):
        if path.exists():
            path.unlink()

    advisor = BacktestIntradayOptionSellingAdvisor(
        config=IntradayOptionSellingAdvisorConfig(
            lot_size=max(1, int(args.lot_size)),
            entry_not_before=args.entry_not_before,
            last_new_entry_time=args.last_new_entry_time,
            max_hold_till=args.max_hold_till,
            preferred_bias=str(args.preferred_bias or "NEUTRAL").upper(),
            directional_structure=str(args.directional_structure or "CREDIT_SPREAD").upper(),
            naked_operational_max_loss_rs=max(500.0, float(args.naked_max_loss_rs or 3000.0)),
            naked_profit_trail_arm_rs=max(500.0, float(args.naked_trail_arm_rs or 5000.0)),
            naked_profit_trail_keep_pct=max(0.35, min(0.95, float(args.naked_trail_keep_pct or 0.72))),
        ),
        status_path=advisor_status_path,
        logger=None,
    )
    position_manager = BacktestIntradayPositionManager(
        state_path=position_state_path,
        history_path=history_path,
        logger=None,
    )
    learning_manager = IntradayLearningManager(
        config=IntradayLearningConfig(),
        status_path=learning_status_path,
        outcomes_path=committee_outcomes_path,
        logger=None,
    )
    decision_committee = DecisionCommittee(
        status_path=committee_status_path,
        history_path=None,
        outcomes_path=committee_outcomes_path,
        learning_manager=learning_manager,
        logger=None,
    )

    signals: List[Dict[str, Any]] = []
    opened_positions = 0
    prev_oi_map: Dict[Tuple[str, float], float] = {}
    sorted_trade_dates = sorted(daily_candle_map.keys())

    for trade_date, day_df in df.groupby("trade_date", sort=True):
        prev_oi_map = {}
        day_history: List[Tuple[datetime, float]] = []
        prior_daily_candles = [daily_candle_map[d] for d in sorted_trade_dates if d < trade_date]
        day_rows = list(day_df.itertuples(index=False))
        last_chain_rows: List[Dict[str, Any]] = []
        last_signal: Dict[str, Any] = {}
        last_now: Optional[datetime] = None
        for row_obj in day_rows:
            row = pd.Series(row_obj._asdict())
            now = row["timestamp"]
            last_now = now
            spot = float(_safe_float(row.get("spot"), 0.0) or 0.0)
            day_history.append((now, spot))
            chain_rows, prev_oi_map = _build_chain_rows(row, prefixes=prefixes, prev_oi_map=prev_oi_map)
            chain_rows = _extend_otm_chain_rows(chain_rows)
            last_chain_rows = chain_rows
            trend_ctx = _build_trend_context(day_history, now=now, row=row)
            oc_ctx = _summarize_option_chain_context(
                chain_rows,
                spot=spot,
                oi_skew_hint=_safe_float(row.get("oi_skew")),
            )
            structure_ctx = _build_structure_context(
                spot=spot,
                trend_ctx=trend_ctx,
                oc_ctx=oc_ctx,
                daily_candles=prior_daily_candles,
            )
            context = {
                "computed_at": now.isoformat(timespec="seconds"),
                "option_chain": oc_ctx,
                "trend": trend_ctx,
                "structure": structure_ctx,
            }
            expiry = _next_thursday(trade_date).isoformat()
            advisor_out = advisor.update(
                now=now,
                expiry=expiry,
                spot=spot,
                chain_rows=chain_rows,
                context=context,
                has_open_bkm=False,
            )
            committee_out = decision_committee.evaluate_intraday(
                now=now,
                signal_payload=advisor_out,
                context=context,
                has_open_bkm=False,
                allow_parallel_with_bkm=False,
            )
            last_signal = advisor_out
            recommendation = advisor_out.get("recommendation") if isinstance(advisor_out.get("recommendation"), dict) else {}
            trade_plan = recommendation.get("plan") if isinstance(recommendation.get("plan"), dict) else {}
            signal_row = {
                "timestamp": now.isoformat(),
                "trade_date": trade_date.isoformat(),
                "signal": advisor_out.get("signal"),
                "strategy": advisor_out.get("strategy"),
                "market_bias": advisor_out.get("market_bias"),
                "trend_confidence": advisor_out.get("trend_confidence"),
                "signal_conflict_score": advisor_out.get("signal_conflict_score"),
                "breakout_confirmation": advisor_out.get("breakout_confirmation"),
                "volatility_regime": advisor_out.get("volatility_regime"),
                "spot": round(spot, 2),
                "signal_changed": bool(advisor_out.get("signal_changed")),
                "committee_verdict": committee_out.get("verdict"),
                "committee_bias": committee_out.get("consensus_bias"),
                "committee_confidence": committee_out.get("ensemble_confidence"),
                "advisor_reasons": "|".join(str(x) for x in (advisor_out.get("reasons") or []) if x),
                "advisor_headline": recommendation.get("headline"),
                "plan_posture": trade_plan.get("posture"),
                "plan_setup_type": trade_plan.get("setup_type"),
                "plan_entry_ready": bool(trade_plan.get("entry_ready")),
                "plan_pullback_touched": bool(trade_plan.get("pullback_touched")),
                "plan_pullback_near": bool(trade_plan.get("pullback_near")),
                "plan_pullback_ok": bool(trade_plan.get("pullback_ok")),
                "plan_pullback_score": _safe_float(trade_plan.get("pullback_score")),
                "plan_pullback_distance_points": _safe_float(trade_plan.get("pullback_distance_points")),
                "plan_pullback_tolerance_points": _safe_float(trade_plan.get("pullback_tolerance_points")),
                "plan_confirmation_ready": bool(trade_plan.get("confirmation_ready")),
                "plan_fake_breakout_conflict": bool(trade_plan.get("fake_breakout_conflict")),
                "plan_higher_tf_ok": bool(trade_plan.get("higher_tf_ok")),
                "plan_higher_tf_score": _safe_float(trade_plan.get("higher_tf_score")),
                "plan_buyer_seller_ok": bool(trade_plan.get("buyer_seller_ok")),
                "plan_divergence_supportive": bool(trade_plan.get("divergence_supportive")),
                "plan_block_reasons": "|".join(str(x) for x in (trade_plan.get("entry_block_reasons") or []) if x),
                "deployed": False,
                "deployed_position_id": None,
            }
            if position_manager.has_open_position():
                pos_eval = position_manager.evaluate(
                    now=now,
                    spot=spot,
                    chain_rows=chain_rows,
                    signal_payload=advisor_out,
                )
                if str(pos_eval.get("action") or "").upper() == "CLOSE":
                    close_state = position_manager.close_position(
                        reason=str(pos_eval.get("reason") or "CLOSED"),
                        pnl_rs=float(pos_eval.get("pnl_rs") or 0.0),
                        now=now,
                    )
                    trade_row = close_state.get("history_row") if isinstance(close_state, dict) else None
                    if isinstance(trade_row, dict) and trade_row:
                        decision_committee.record_outcome(trade_row=trade_row, now=now)
            if (
                not position_manager.has_open_position()
                and str(advisor_out.get("signal") or "").upper() == "ENTER_NOW"
                and str(committee_out.get("verdict") or "WAIT").upper() == "READY"
                and position_manager.can_open_new_position(
                    max_trades_per_session=max(1, int(args.max_trades_per_session)),
                    now=now,
                )
            ):
                rec = advisor_out.get("recommendation") if isinstance(advisor_out.get("recommendation"), dict) else {}
                legs_payload = rec.get("legs") if isinstance(rec.get("legs"), list) else []
                expiry_dt = datetime.fromisoformat(expiry).date()
                order_legs = [
                    _rec_leg_to_option_leg(
                        rec_leg=leg,
                        strategy_name=str(rec.get("strategy_type") or advisor_out.get("strategy") or ""),
                        expiry=expiry_dt,
                        lot_size=max(1, int(args.lot_size)),
                        chain_rows=chain_rows,
                    )
                    for leg in legs_payload
                    if isinstance(leg, dict)
                ]
                if order_legs:
                    position_id = f"{trade_date.isoformat()}-{opened_positions + 1}"
                    entry_features = rec.get("entry_features") if isinstance(rec.get("entry_features"), dict) else {}
                    if isinstance(entry_features, dict):
                        entry_features = dict(entry_features)
                    else:
                        entry_features = {}
                    entry_features["committee_snapshot"] = decision_committee.snapshot()
                    position_manager.open_position(
                        position_id=position_id,
                        trade_mode="paper",
                        strategy_type=str(rec.get("strategy_type") or advisor_out.get("strategy") or "INTRADAY_AI"),
                        strategy_label=str(rec.get("strategy_label") or rec.get("strategy_type") or "Intraday AI"),
                        expiry=expiry,
                        legs=order_legs,
                        entry_spot=spot,
                        sl_total_rs=float(_safe_float(((rec.get("sl") or {}) if isinstance(rec.get("sl"), dict) else {}).get("loss_rs_per_set"), 0.0) or 0.0),
                        tp_total_rs=float(_safe_float(((rec.get("tp") or {}) if isinstance(rec.get("tp"), dict) else {}).get("profit_rs_per_set"), 0.0) or 0.0),
                        invalidation_spot_level=_safe_float(((rec.get("invalidation") or {}) if isinstance(rec.get("invalidation"), dict) else {}).get("spot_level")),
                        max_hold_till=str(((rec.get("hold") or {}) if isinstance(rec.get("hold"), dict) else {}).get("max_hold_till") or args.max_hold_till),
                        trailing_enabled=True,
                        lots_multiplier=1,
                        entry_features=entry_features,
                        now=now,
                    )
                    opened_positions += 1
                    signal_row["deployed"] = True
                    signal_row["deployed_position_id"] = position_id
            signals.append(signal_row)

        if position_manager.has_open_position() and last_now is not None:
            forced_close_ts = datetime.combine(trade_date, _parse_time(args.max_hold_till), tzinfo=last_now.tzinfo)
            if forced_close_ts < last_now:
                forced_close_ts = last_now
            pos_eval = position_manager.evaluate(
                now=forced_close_ts,
                spot=float(_safe_float(day_df.iloc[-1]["spot"], 0.0) or 0.0),
                chain_rows=last_chain_rows,
                signal_payload=last_signal,
            )
            if str(pos_eval.get("action") or "").upper() == "CLOSE":
                close_state = position_manager.close_position(
                    reason=str(pos_eval.get("reason") or "CLOSED"),
                    pnl_rs=float(pos_eval.get("pnl_rs") or 0.0),
                    now=forced_close_ts,
                )
                trade_row = close_state.get("history_row") if isinstance(close_state, dict) else None
                if isinstance(trade_row, dict) and trade_row:
                    decision_committee.record_outcome(trade_row=trade_row, now=forced_close_ts)

    advisor_status_path.write_text(json.dumps(advisor.snapshot(), indent=2, default=str))
    committee_status_path.write_text(json.dumps(decision_committee.snapshot(), indent=2, default=str))
    learning_status_path.write_text(json.dumps(learning_manager.snapshot(), indent=2, default=str))
    position_state_path.write_text(json.dumps(position_manager.snapshot(), indent=2, default=str))
    history_path.write_text(
        "".join(json.dumps(row, default=str) + "\n" for row in position_manager.history_rows),
        encoding="utf-8",
    )

    trade_rows = list(position_manager.history_rows)
    trades_df = pd.DataFrame(trade_rows)
    if not trades_df.empty:
        trades_df = trades_df.sort_values(["opened_at", "closed_at"], na_position="last").reset_index(drop=True)
    signals_df = pd.DataFrame(signals)
    metrics = _trade_metrics(trades_df)
    summary = {
        "input": str(args.input),
        "rows": int(len(df)),
        "days": int(df["trade_date"].nunique()),
        "signals_total": int(len(signals_df)),
        "enter_signals": int((signals_df["signal"] == "ENTER_NOW").sum()) if not signals_df.empty else 0,
        "deployed_trades": int(opened_positions),
        "metrics": metrics,
    }

    signals_path = args.output_dir / "intraday_ai_signals.csv"
    trades_path = args.output_dir / "intraday_ai_trades.csv"
    summary_path = args.output_dir / "intraday_ai_summary.json"
    signals_df.to_csv(signals_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))

    print(f"Wrote signals to {signals_path}")
    print(f"Wrote trades to {trades_path}")
    print(f"Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
