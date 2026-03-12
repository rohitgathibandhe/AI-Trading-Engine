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
from market_ai.modules.agents.intraday_position_manager import IntradayPositionManager  # noqa: E402
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
    if strategy_name == "PUT_CREDIT_SPREAD":
        return StrategyType.BULL_PUT_SPREAD
    if strategy_name == "CALL_CREDIT_SPREAD":
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
    ema_fast_prev = _ema(spots[:-1], 5) if len(spots) > 1 else ema_fast
    ema_slow_prev = _ema(spots[:-1], 20) if len(spots) > 1 else ema_slow
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
        "ema_gap_pct": None if ema_gap_pct is None else round(float(ema_gap_pct), 4),
        "ema_alignment": ema_alignment,
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
) -> Dict[str, Any]:
    default = {
        "price_action_bias": "NEUTRAL",
        "price_action_confirmation": "NONE",
        "price_action_score": 0.0,
        "primary_pattern": "NONE",
        "price_action_patterns": [],
        "retest_status": "NONE",
        "retest_bias": "NEUTRAL",
        "retest_level": None,
        "retest_score": 0.0,
    }
    parsed: List[Dict[str, float]] = []
    for candle in (candles or [])[-6:]:
        try:
            open_px = float(candle.get("open") or candle.get("close") or 0.0)
            high_px = float(candle.get("high") or candle.get("close") or open_px)
            low_px = float(candle.get("low") or candle.get("close") or open_px)
            close_px = float(candle.get("close") or open_px)
        except Exception:
            continue
        rng = max(0.01, high_px - low_px)
        body = abs(close_px - open_px)
        upper_wick = max(0.0, high_px - max(open_px, close_px))
        lower_wick = max(0.0, min(open_px, close_px) - low_px)
        close_pos = (close_px - low_px) / rng
        parsed.append(
            {
                "open": open_px,
                "high": high_px,
                "low": low_px,
                "close": close_px,
                "range": rng,
                "body": body,
                "upper_wick": upper_wick,
                "lower_wick": lower_wick,
                "close_pos": close_pos,
            }
        )
    if len(parsed) < 3:
        return default
    last = parsed[-1]
    prev = parsed[-2]
    prev2 = parsed[-3]
    prior_rows = parsed[:-1]
    prior_high = max(row["high"] for row in prior_rows)
    prior_low = min(row["low"] for row in prior_rows)
    avg_range = max(0.01, float(atr_like_points or 0.0), sum(row["range"] for row in parsed) / len(parsed))
    level_buffer = max(5.0, avg_range * 0.25)
    bullish_score = 0.0
    bearish_score = 0.0
    bullish_patterns: List[str] = []
    bearish_patterns: List[str] = []
    retest_status = "NONE"
    retest_bias = "NEUTRAL"
    retest_level: Optional[float] = None
    retest_score = 0.0

    def _record_retest(*, status: str, bias: str, level: Optional[float], score: float) -> None:
        nonlocal retest_status, retest_bias, retest_level, retest_score
        if float(score) >= float(retest_score):
            retest_status = str(status)
            retest_bias = str(bias)
            retest_level = None if level is None else float(level)
            retest_score = float(score)

    if (
        last["close"] > last["open"]
        and prev["close"] < prev["open"]
        and last["open"] <= prev["close"]
        and last["close"] >= prev["open"]
        and last["body"] >= (prev["body"] * 0.9)
    ):
        bullish_score += 0.85
        bullish_patterns.append("BULLISH_ENGULFING")
    if (
        last["close"] < last["open"]
        and prev["close"] > prev["open"]
        and last["open"] >= prev["close"]
        and last["close"] <= prev["open"]
        and last["body"] >= (prev["body"] * 0.9)
    ):
        bearish_score += 0.85
        bearish_patterns.append("BEARISH_ENGULFING")
    if last["lower_wick"] >= max(last["body"] * 1.6, avg_range * 0.18) and last["close_pos"] >= 0.60:
        bullish_score += 0.55
        bullish_patterns.append("LOWER_REJECTION")
    if last["upper_wick"] >= max(last["body"] * 1.6, avg_range * 0.18) and last["close_pos"] <= 0.40:
        bearish_score += 0.55
        bearish_patterns.append("UPPER_REJECTION")
    if last["close"] > prior_high and last["close_pos"] >= 0.62:
        bullish_score += 0.70
        bullish_patterns.append("BREAKOUT_CONTINUATION_UP")
    if last["close"] < prior_low and last["close_pos"] <= 0.38:
        bearish_score += 0.70
        bearish_patterns.append("BREAKOUT_CONTINUATION_DOWN")
    if (
        prev["high"] <= prev2["high"]
        and prev["low"] >= prev2["low"]
        and last["close"] > prev["high"] + (level_buffer * 0.05)
        and last["close"] > last["open"]
    ):
        bullish_score += 0.50
        bullish_patterns.append("INSIDE_BAR_BREAK_UP")
    if (
        prev["high"] <= prev2["high"]
        and prev["low"] >= prev2["low"]
        and last["close"] < prev["low"] - (level_buffer * 0.05)
        and last["close"] < last["open"]
    ):
        bearish_score += 0.50
        bearish_patterns.append("INSIDE_BAR_BREAK_DOWN")
    if support is not None:
        support_f = float(support)
        if last["low"] <= support_f + level_buffer and last["close"] >= support_f + (level_buffer * 0.10) and last["close_pos"] >= 0.55:
            _record_retest(status="SUPPORT_HOLD", bias="BULLISH", level=support_f, score=0.72)
    if resistance is not None:
        resistance_f = float(resistance)
        if last["high"] >= resistance_f - level_buffer and last["close"] <= resistance_f - (level_buffer * 0.10) and last["close_pos"] <= 0.45:
            _record_retest(status="RESISTANCE_HOLD", bias="BEARISH", level=resistance_f, score=0.72)
        if prev["close"] > resistance_f and last["low"] <= resistance_f + level_buffer and last["close"] > resistance_f:
            _record_retest(status="BREAKOUT_RETEST_HOLD", bias="BULLISH", level=resistance_f, score=0.88)
        elif prev["close"] > resistance_f and last["close"] < resistance_f:
            _record_retest(status="RETEST_FAILED", bias="BEARISH", level=resistance_f, score=0.82)
    if support is not None:
        support_f = float(support)
        if prev["close"] < support_f and last["high"] >= support_f - level_buffer and last["close"] < support_f:
            _record_retest(status="BREAKDOWN_RETEST_HOLD", bias="BEARISH", level=support_f, score=0.88)
        elif prev["close"] < support_f and last["close"] > support_f:
            _record_retest(status="RETEST_FAILED", bias="BULLISH", level=support_f, score=0.82)

    price_action_bias = "NEUTRAL"
    price_action_patterns: List[str] = []
    price_action_score = 0.0
    if bullish_score >= max(0.55, bearish_score + 0.20):
        price_action_bias = "BULLISH"
        price_action_patterns = bullish_patterns
        price_action_score = bullish_score
    elif bearish_score >= max(0.55, bullish_score + 0.20):
        price_action_bias = "BEARISH"
        price_action_patterns = bearish_patterns
        price_action_score = bearish_score

    confirmation = "NONE"
    if price_action_bias != "NEUTRAL" and retest_bias == price_action_bias and retest_status not in {"NONE", "RETEST_FAILED"}:
        confirmation = "CANDLE_AND_RETEST_CONFIRMED"
    elif price_action_bias != "NEUTRAL":
        confirmation = "CANDLE_CONFIRMED"
    elif retest_bias != "NEUTRAL" and retest_status not in {"NONE", "RETEST_FAILED"}:
        price_action_bias = retest_bias
        confirmation = "RETEST_CONFIRMED"
        price_action_patterns = [retest_status]
        price_action_score = retest_score

    unique_patterns: List[str] = []
    for pattern in price_action_patterns:
        text = str(pattern or "").upper()
        if text and text not in unique_patterns:
            unique_patterns.append(text)
    return {
        "price_action_bias": price_action_bias,
        "price_action_confirmation": confirmation,
        "price_action_score": round(float(max(price_action_score, retest_score)), 3),
        "primary_pattern": unique_patterns[0] if unique_patterns else "NONE",
        "price_action_patterns": unique_patterns[:4],
        "retest_status": retest_status,
        "retest_bias": retest_bias,
        "retest_level": None if retest_level is None else round(float(retest_level), 2),
        "retest_score": round(float(retest_score), 3),
    }


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
    return {
        "timeframes": per_tf,
        "bias_score": round(float(bias_score), 3),
        "bias": bias,
        "volatility_regime": volatility_regime,
        "breakout_confirmation": breakout_confirmation,
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
    for tf_key, tf_weight in (("5", 1.0), ("15", 1.5), ("60", 2.0)):
        tf = tf_map.get(tf_key) if isinstance(tf_map.get(tf_key), dict) else {}
        support = _safe_float(tf.get("support"))
        resistance = _safe_float(tf.get("resistance"))
        if support is not None:
            intraday_support_cands.append({"level": support, "source": f"{tf_key}m_price", "strength": tf_weight})
        if resistance is not None:
            intraday_resistance_cands.append({"level": resistance, "source": f"{tf_key}m_price", "strength": tf_weight})

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
        if pa_conf in {"CANDLE_CONFIRMED", "CANDLE_AND_RETEST_CONFIRMED"}:
            candle_confirm_hits += 1
        if pa_conf in {"RETEST_CONFIRMED", "CANDLE_AND_RETEST_CONFIRMED"}:
            retest_confirm_hits += 1
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
    candle_vote = 1 if (price_action_bias == "BULLISH" and candle_confirm_hits > 0) else (-1 if (price_action_bias == "BEARISH" and candle_confirm_hits > 0) else 0)
    retest_vote = 1 if (retest_bias == "BULLISH" and retest_status not in {"NONE", "RETEST_FAILED"}) else (-1 if (retest_bias == "BEARISH" and retest_status not in {"NONE", "RETEST_FAILED"}) else 0)
    votes = {
        "trend": trend_vote,
        "pcr": pcr_vote,
        "oi_build": oi_vote,
        "breakout": breakout_vote,
        "weekly": weekly_vote,
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
    if price_action_confirmation in {"CANDLE_CONFIRMED", "RETEST_CONFIRMED"}:
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
    position_state_path = args.output_dir / "intraday_ai_position_status.json"
    history_path = args.output_dir / "intraday_ai_trade_history.jsonl"
    for path in (advisor_status_path, position_state_path, history_path):
        if path.exists():
            path.unlink()

    advisor = BacktestIntradayOptionSellingAdvisor(
        config=IntradayOptionSellingAdvisorConfig(
            lot_size=max(1, int(args.lot_size)),
            entry_not_before=args.entry_not_before,
            last_new_entry_time=args.last_new_entry_time,
            max_hold_till=args.max_hold_till,
            preferred_bias=str(args.preferred_bias or "NEUTRAL").upper(),
        ),
        status_path=advisor_status_path,
        logger=None,
    )
    position_manager = BacktestIntradayPositionManager(
        state_path=position_state_path,
        history_path=history_path,
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
            last_signal = advisor_out
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
                    position_manager.close_position(
                        reason=str(pos_eval.get("reason") or "CLOSED"),
                        pnl_rs=float(pos_eval.get("pnl_rs") or 0.0),
                        now=now,
                    )
            if (
                not position_manager.has_open_position()
                and str(advisor_out.get("signal") or "").upper() == "ENTER_NOW"
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
                        entry_features=rec.get("entry_features") if isinstance(rec.get("entry_features"), dict) else {},
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
                position_manager.close_position(
                    reason=str(pos_eval.get("reason") or "CLOSED"),
                    pnl_rs=float(pos_eval.get("pnl_rs") or 0.0),
                    now=forced_close_ts,
                )

    advisor_status_path.write_text(json.dumps(advisor.snapshot(), indent=2, default=str))
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
