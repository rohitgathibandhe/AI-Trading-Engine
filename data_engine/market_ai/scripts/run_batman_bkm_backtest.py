#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
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

from market_ai.modules.strategies.batman_bkm_monthly import (  # noqa: E402
    BatmanBKMConfig,
    BatmanBKMStrategy,
)
from market_ai.modules.agents.batman_bkm_learning import (  # noqa: E402
    BatmanBKMLearningConfig,
    BatmanBKMLearningManager,
)


IST = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
DEFAULT_CSV = REPO_ROOT / "reports" / "intraday_from_rolling_12m.csv"
DEFAULT_OUT_DIR = Path("/tmp/batman_bkm_backtest")


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    if value in (None, "", "-", "--"):
        return default
    try:
        parsed = float(value)
        if math.isnan(parsed):
            return default
        return parsed
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
        if not str(col).endswith("_strike"):
            continue
        prefix = str(col)[: -len("_strike")]
        opt = _prefix_option_type(prefix)
        if opt:
            prefixes[prefix] = opt
    return prefixes


def _build_chain_rows(row: pd.Series, *, prefixes: Dict[str, str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for prefix, option_type in prefixes.items():
        strike = _safe_float(row.get(f"{prefix}_strike"))
        ltp = _safe_float(row.get(f"{prefix}_ltp"))
        if strike is None or ltp is None or strike <= 0:
            continue
        rows.append(
            {
                "option_type": option_type,
                "strike": float(strike),
                "ltp": float(ltp),
                "oi": _safe_float(row.get(f"{prefix}_oi")),
                "iv": _safe_float(row.get(f"{prefix}_iv")),
                "volume": _safe_float(row.get(f"{prefix}_volume")),
                "security_id": f"{option_type}_{int(round(float(strike)))}",
            }
        )
    return sorted(rows, key=lambda item: (str(item["option_type"]), float(item["strike"])))


def _extend_otm_chain_rows(
    chain_rows: List[Dict[str, Any]],
    *,
    min_extra_steps: int = 30,
) -> List[Dict[str, Any]]:
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
        ltp_ratio = 0.65 if prev_ltp <= 0 else max(0.35, min(0.85, last_ltp / prev_ltp))
        last_oi = _safe_float(base_last.get("oi"))
        prev_oi = _safe_float(base_prev.get("oi"))
        oi_ratio = 0.85
        if last_oi is not None and prev_oi not in (None, 0.0):
            oi_ratio = max(0.60, min(1.05, float(last_oi) / float(prev_oi)))
        next_ltp = max(0.5, last_ltp)
        next_oi = float(last_oi) if last_oi is not None else None
        last_iv = _safe_float(base_last.get("iv"))
        last_volume = _safe_float(base_last.get("volume"))
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
                    "iv": last_iv,
                    "volume": None if next_volume is None else round(float(next_volume), 2),
                    "security_id": f"SYN_{option_type}_{int(round(float(strike)))}",
                    "synthetic": True,
                }
            )

    _append_extras(_sorted_side("CE"), option_type="CE", direction="UP")
    _append_extras(_sorted_side("PE"), option_type="PE", direction="DOWN")

    deduped: Dict[Tuple[str, float], Dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("option_type") or "").upper(), float(row.get("strike") or 0.0))
        existing = deduped.get(key)
        if existing is None or (existing.get("synthetic") and not row.get("synthetic")):
            deduped[key] = row
    return sorted(deduped.values(), key=lambda item: (str(item["option_type"]), float(item["strike"])))


def _last_tuesday(year: int, month: int) -> date:
    if month == 12:
        probe = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        probe = date(year, month + 1, 1) - timedelta(days=1)
    while probe.weekday() != 1:
        probe -= timedelta(days=1)
    return probe


def _next_month(year: int, month: int) -> Tuple[int, int]:
    if month == 12:
        return year + 1, 1
    return year, month + 1


def _roll_plan_for_trade_day(trade_day: date) -> Tuple[date, date]:
    anchor = _last_tuesday(trade_day.year, trade_day.month)
    if trade_day > anchor:
        next_year, next_month = _next_month(trade_day.year, trade_day.month)
        anchor = _last_tuesday(next_year, next_month)
    target_year, target_month = _next_month(anchor.year, anchor.month)
    target = _last_tuesday(target_year, target_month)
    return anchor, target


def _last_friday_before(expiry: date) -> date:
    probe = expiry
    while probe.weekday() != 4:
        probe -= timedelta(days=1)
    return probe


def _extract_leg_strikes(basket) -> Dict[str, Optional[float]]:
    longs_ce = sorted(float(leg.strike) for leg in basket.legs if leg.option_type == "CE" and leg.side == "BUY")
    sells_ce = sorted(float(leg.strike) for leg in basket.legs if leg.option_type == "CE" and leg.side == "SELL")
    longs_pe = sorted(float(leg.strike) for leg in basket.legs if leg.option_type == "PE" and leg.side == "BUY")
    sells_pe = sorted(float(leg.strike) for leg in basket.legs if leg.option_type == "PE" and leg.side == "SELL")
    return {
        "long_call": min(longs_ce) if longs_ce else None,
        "hedge_call": max(longs_ce) if len(longs_ce) > 1 else (longs_ce[0] if longs_ce else None),
        "short_call": sells_ce[0] if sells_ce else None,
        "long_put": max(longs_pe) if longs_pe else None,
        "hedge_put": min(longs_pe) if len(longs_pe) > 1 else (longs_pe[0] if longs_pe else None),
        "short_put": sells_pe[0] if sells_pe else None,
    }


def _bias_from_ratio(value: Optional[float], *, bullish_above: float, bearish_below: float) -> str:
    if value is None:
        return "NEUTRAL"
    if float(value) >= bullish_above:
        return "BULLISH"
    if float(value) <= bearish_below:
        return "BEARISH"
    return "NEUTRAL"


def _build_backtest_market_context(
    *,
    row: pd.Series,
    recent_spots: List[float],
    day_open: float,
    spot: float,
) -> Dict[str, Any]:
    atm_call_oi = _safe_float(row.get("atm_call_oi"))
    atm_put_oi = _safe_float(row.get("atm_put_oi"))
    pcr_total = None
    if atm_call_oi not in (None, 0.0) and atm_put_oi is not None:
        pcr_total = float(atm_put_oi) / max(1.0, float(atm_call_oi))
    oi_skew = _safe_float(row.get("oi_skew"))
    combined_premium_pct = _safe_float(row.get("combined_premium_pct"))
    iv_rank = _safe_float(row.get("iv_rank"))
    recent_anchor = recent_spots[max(0, len(recent_spots) - 6)] if recent_spots else spot
    intraday_ret = 0.0 if day_open <= 0 else (float(spot) - float(day_open)) / float(day_open)
    recent_ret = 0.0 if recent_anchor <= 0 else (float(spot) - float(recent_anchor)) / float(recent_anchor)
    bias_score = max(-1.0, min(1.0, (intraday_ret * 140.0) + (recent_ret * 220.0)))
    trend_bias = "NEUTRAL"
    if bias_score >= 0.32:
        trend_bias = "BULLISH"
    elif bias_score <= -0.32:
        trend_bias = "BEARISH"
    breakout_confirmation = "NONE"
    if intraday_ret >= 0.004 and recent_ret >= 0.0015:
        breakout_confirmation = "UP_CONFIRMED"
    elif intraday_ret <= -0.004 and recent_ret <= -0.0015:
        breakout_confirmation = "DOWN_CONFIRMED"
    volatility_regime = "NORMAL"
    if (iv_rank is not None and iv_rank >= 0.70) or (combined_premium_pct is not None and combined_premium_pct >= 0.018):
        volatility_regime = "HIGH"
    elif (iv_rank is not None and iv_rank <= 0.35) or (combined_premium_pct is not None and combined_premium_pct <= 0.009):
        volatility_regime = "LOW"
    pcr_bias = _bias_from_ratio(pcr_total, bullish_above=1.08, bearish_below=0.92)
    oi_build_bias = "UNKNOWN"
    if oi_skew is not None:
        if float(oi_skew) >= 0.05:
            oi_build_bias = "BULLISH_SUPPORT"
        elif float(oi_skew) <= -0.05:
            oi_build_bias = "BEARISH_RESISTANCE"
    dominant_signal_bias = trend_bias
    conflict = 12.0
    if trend_bias == "BULLISH" and pcr_bias == "BEARISH":
        conflict += 18.0
    elif trend_bias == "BEARISH" and pcr_bias == "BULLISH":
        conflict += 18.0
    if trend_bias == "NEUTRAL":
        conflict += 8.0
    call_wall_strike = _safe_float(row.get("call_oi_max_strike"))
    put_wall_strike = _safe_float(row.get("put_oi_max_strike"))
    return {
        "option_chain": {
            "pcr_bias": pcr_bias,
            "pcr_total": pcr_total,
            "pcr_near_atm": pcr_total,
            "oi_build": {"bias": oi_build_bias},
            "call_wall_above": {
                "strike": call_wall_strike,
                "distance_from_spot": None if call_wall_strike is None else round(float(call_wall_strike) - float(spot), 2),
            },
            "put_wall_below": {
                "strike": put_wall_strike,
                "distance_from_spot": None if put_wall_strike is None else round(float(spot) - float(put_wall_strike), 2),
            },
        },
        "trend": {
            "bias": trend_bias,
            "bias_score": round(float(bias_score), 3),
            "breakout_confirmation": breakout_confirmation,
            "volatility_regime": volatility_regime,
        },
        "structure": {
            "dominant_signal_bias": dominant_signal_bias,
            "trend_confidence": round(min(1.0, abs(float(bias_score))), 3),
            "signal_conflict_score": round(float(conflict), 2),
        },
    }


def _build_learning_entry_snapshot(
    *,
    now: datetime,
    expiry: date,
    spot: float,
    entry_mode: str,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    oc_ctx = context.get("option_chain") if isinstance(context.get("option_chain"), dict) else {}
    trend_ctx = context.get("trend") if isinstance(context.get("trend"), dict) else {}
    structure_ctx = context.get("structure") if isinstance(context.get("structure"), dict) else {}
    oi_build = oc_ctx.get("oi_build") if isinstance(oc_ctx.get("oi_build"), dict) else {}
    return {
        "timestamp": now.isoformat(timespec="seconds"),
        "trade_mode": "paper",
        "expiry": expiry.isoformat(),
        "spot": round(float(spot), 2),
        "entry_mode": str(entry_mode or "SCHEDULED").upper(),
        "market_bias": str(structure_ctx.get("dominant_signal_bias") or "NEUTRAL"),
        "dominant_signal_bias": str(structure_ctx.get("dominant_signal_bias") or "NEUTRAL"),
        "trend_confidence": structure_ctx.get("trend_confidence"),
        "signal_conflict_score": structure_ctx.get("signal_conflict_score"),
        "pcr_bias": str(oc_ctx.get("pcr_bias") or "NEUTRAL"),
        "oi_build_bias": str(oi_build.get("bias") or "UNKNOWN"),
        "volatility_regime": str(trend_ctx.get("volatility_regime") or "UNKNOWN"),
        "breakout_confirmation": str(trend_ctx.get("breakout_confirmation") or "NONE"),
    }


def run_backtest(
    *,
    csv_path: Path,
    out_dir: Path,
    cfg: BatmanBKMConfig,
    catchup_not_before: dtime,
) -> Dict[str, Any]:
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns or "spot" not in df.columns:
        raise ValueError("Input CSV must contain at least 'timestamp' and 'spot'.")
    df["timestamp"] = df["timestamp"].apply(_parse_ts)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["trade_date"] = df["timestamp"].apply(lambda ts: ts.date())
    prefixes = _discover_option_prefixes(df.columns)
    strategy = BatmanBKMStrategy(cfg)
    learning_manager = BatmanBKMLearningManager(
        config=BatmanBKMLearningConfig(
            enabled=True,
            allow_premarket_entry=False,
            opening_range_block_minutes=15,
            min_bucket_samples=2,
            review_window=100,
        ),
        status_path=out_dir / "batman_bkm_learning_status.json",
        outcomes_path=out_dir / "batman_bkm_learning_outcomes.jsonl",
    )
    trades: List[Dict[str, Any]] = []
    timeline: List[Dict[str, Any]] = []
    skipped_entry_reasons: Counter[str] = Counter()
    open_meta: Optional[Dict[str, Any]] = None
    attempted_entry_days: set[Tuple[date, date]] = set()
    last_chain: Optional[List[Dict[str, Any]]] = None
    last_ts: Optional[datetime] = None
    last_spot: Optional[float] = None

    for trade_date, day_df in df.groupby("trade_date", sort=True):
        anchor_expiry, target_expiry = _roll_plan_for_trade_day(trade_date)
        entry_day = _last_friday_before(anchor_expiry)
        day_open = float(_safe_float(day_df.iloc[0].get("spot"), 0.0) or 0.0)
        day_spots: List[float] = []
        for _, row in day_df.iterrows():
            now = row["timestamp"]
            spot = float(_safe_float(row.get("spot"), 0.0) or 0.0)
            day_spots.append(float(spot))
            chain = _extend_otm_chain_rows(_build_chain_rows(row, prefixes=prefixes))
            last_chain = chain
            last_ts = now
            last_spot = spot
            market_context = _build_backtest_market_context(
                row=row,
                recent_spots=day_spots,
                day_open=day_open,
                spot=spot,
            )

            if strategy.basket is not None:
                pnl = float(strategy.update_mtm(chain) or 0.0)
                timeline.append(
                    {
                        "timestamp": now.isoformat(),
                        "trade_date": trade_date.isoformat(),
                        "expiry": strategy.basket.expiry.isoformat(),
                        "spot": round(spot, 2),
                        "basket_mtm": round(pnl, 2),
                        "status": "OPEN",
                    }
                )
                exit_reason = strategy.maybe_exit(
                    pnl,
                    now,
                    spot=spot,
                    context=market_context,
                )
                if exit_reason:
                    basket = strategy.basket
                    strikes = _extract_leg_strikes(basket)
                    entry_spot = float((open_meta or {}).get("entry_spot") or 0.0)
                    trades.append(
                        {
                            "entry_time": (open_meta or {}).get("entry_time"),
                            "exit_time": now.isoformat(),
                            "exit_reason": exit_reason,
                            "expiry": basket.expiry.isoformat(),
                            "entry_spot": round(entry_spot, 2),
                            "exit_spot": round(spot, 2),
                            "realized_pnl": round(float(pnl), 2),
                            "deployed_capital": round(float(basket.margin_required), 2),
                            "net_credit": round(float(basket.net_credit), 2),
                            "credit_pct": round(float(basket.credit_pct), 3),
                            "quality_status": basket.quality_status,
                            "quality_score": round(float(basket.quality_score), 1),
                            "widened_iterations": int(basket.widened_iterations or 0),
                            "hedge_call_qty": int(basket.hedge_qty_call or 0),
                            "hedge_put_qty": int(basket.hedge_qty_put or 0),
                            "entry_mode": (open_meta or {}).get("entry_mode"),
                            **strikes,
                        }
                    )
                    if open_meta and open_meta.get("learning_entry"):
                        learning_manager.record_outcome(
                            expiry=basket.expiry.isoformat(),
                            trade_mode="paper",
                            opened_at=(open_meta or {}).get("entry_time"),
                            closed_at=now.isoformat(),
                            close_reason=exit_reason,
                            pnl_rs=float(pnl),
                            entry_snapshot=dict(open_meta.get("learning_entry") or {}),
                            close_context={
                                "market_bias": market_context.get("structure", {}).get("dominant_signal_bias"),
                                "breakout_confirmation": market_context.get("trend", {}).get("breakout_confirmation"),
                            },
                            quality_status=basket.quality_status,
                            quality_score=float(basket.quality_score),
                            net_credit_rs=float(basket.net_credit),
                        )
                    strategy.basket = None
                    open_meta = None
                continue

            if target_expiry in strategy.entered_expiries:
                continue

            is_scheduled_window = trade_date == entry_day and now.time() >= cfg.entry_time
            is_catchup_window = trade_date > entry_day and trade_date <= anchor_expiry and now.time() >= catchup_not_before
            if not (is_scheduled_window or is_catchup_window):
                continue

            cycle_key = (trade_date, target_expiry)
            if cycle_key in attempted_entry_days:
                continue
            attempted_entry_days.add(cycle_key)
            entry_mode = "SCHEDULED" if is_scheduled_window else "CATCHUP"
            learning_entry_snapshot = _build_learning_entry_snapshot(
                now=now,
                expiry=target_expiry,
                spot=spot,
                entry_mode=entry_mode,
                context=market_context,
            )
            learning_assessment = learning_manager.assess_entry(learning_entry_snapshot, when=now)
            if bool(learning_assessment.get("block_entry")):
                reason = "LEARNING_BLOCK"
                basket = None
            else:
                basket, reason = strategy.maybe_enter(
                    spot,
                    chain,
                    target_expiry,
                    market_context=market_context,
                    learning_assessment=learning_assessment,
                )
            if basket and reason == "ENTER":
                basket.entry_ts = now
                open_meta = {
                    "entry_time": now.isoformat(),
                    "entry_spot": round(spot, 2),
                    "entry_mode": entry_mode,
                    "learning_entry": {
                        **learning_entry_snapshot,
                        "assessment": learning_assessment,
                    },
                }
                timeline.append(
                    {
                        "timestamp": now.isoformat(),
                        "trade_date": trade_date.isoformat(),
                        "expiry": basket.expiry.isoformat(),
                        "spot": round(spot, 2),
                        "basket_mtm": 0.0,
                        "status": "ENTER",
                        "entry_mode": open_meta["entry_mode"],
                        "net_credit": round(float(basket.net_credit), 2),
                        "quality_status": basket.quality_status,
                        "quality_score": round(float(basket.quality_score), 1),
                    }
                )
            else:
                skipped_entry_reasons[str(reason or "UNKNOWN")] += 1
                timeline.append(
                    {
                        "timestamp": now.isoformat(),
                        "trade_date": trade_date.isoformat(),
                        "expiry": target_expiry.isoformat(),
                        "spot": round(spot, 2),
                        "status": "SKIP",
                        "skip_reason": str(reason or "UNKNOWN"),
                    }
                )

    if strategy.basket is not None and last_ts is not None:
        pnl = float(strategy.update_mtm(last_chain or []) or 0.0)
        basket = strategy.basket
        strikes = _extract_leg_strikes(basket)
        trades.append(
            {
                "entry_time": (open_meta or {}).get("entry_time"),
                "exit_time": last_ts.isoformat(),
                "exit_reason": "DATA_END",
                "expiry": basket.expiry.isoformat(),
                "entry_spot": round(float((open_meta or {}).get("entry_spot") or 0.0), 2),
                "exit_spot": round(float(last_spot or 0.0), 2),
                "realized_pnl": round(float(pnl), 2),
                "deployed_capital": round(float(basket.margin_required), 2),
                "net_credit": round(float(basket.net_credit), 2),
                "credit_pct": round(float(basket.credit_pct), 3),
                "quality_status": basket.quality_status,
                "quality_score": round(float(basket.quality_score), 1),
                "widened_iterations": int(basket.widened_iterations or 0),
                "hedge_call_qty": int(basket.hedge_qty_call or 0),
                "hedge_put_qty": int(basket.hedge_qty_put or 0),
                "entry_mode": (open_meta or {}).get("entry_mode"),
                **strikes,
            }
        )
        if open_meta and open_meta.get("learning_entry"):
            learning_manager.record_outcome(
                expiry=basket.expiry.isoformat(),
                trade_mode="paper",
                opened_at=(open_meta or {}).get("entry_time"),
                closed_at=last_ts.isoformat(),
                close_reason="DATA_END",
                pnl_rs=float(pnl),
                entry_snapshot=dict(open_meta.get("learning_entry") or {}),
                close_context={},
                quality_status=basket.quality_status,
                quality_score=float(basket.quality_score),
                net_credit_rs=float(basket.net_credit),
            )
        strategy.basket = None

    trades_df = pd.DataFrame(trades)
    timeline_df = pd.DataFrame(timeline)
    total_pnl = float(trades_df["realized_pnl"].sum()) if not trades_df.empty else 0.0
    wins = int((trades_df["realized_pnl"] > 0).sum()) if not trades_df.empty else 0
    losses = int((trades_df["realized_pnl"] < 0).sum()) if not trades_df.empty else 0
    flats = int((trades_df["realized_pnl"] == 0).sum()) if not trades_df.empty else 0
    summary = {
        "dataset": str(csv_path),
        "days": int(df["trade_date"].nunique()),
        "entries": int(len(trades_df)),
        "closed_trades": int(len(trades_df)),
        "wins": wins,
        "losses": losses,
        "flats": flats,
        "win_rate": round((wins / len(trades_df)) * 100.0, 2) if len(trades_df) else 0.0,
        "realized_pnl": round(total_pnl, 2),
        "avg_pnl_per_trade": round(total_pnl / len(trades_df), 2) if len(trades_df) else 0.0,
        "best_trade": round(float(trades_df["realized_pnl"].max()), 2) if not trades_df.empty else 0.0,
        "worst_trade": round(float(trades_df["realized_pnl"].min()), 2) if not trades_df.empty else 0.0,
        "exit_reason_counts": dict(Counter(str(x) for x in trades_df.get("exit_reason", pd.Series(dtype=str)).fillna("UNKNOWN"))),
        "skip_reason_counts": dict(skipped_entry_reasons),
        "quality_status_counts": dict(Counter(str(x) for x in trades_df.get("quality_status", pd.Series(dtype=str)).fillna("UNKNOWN"))),
        "open_trade_left": False,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "batman_bkm_summary.json"
    trades_path = out_dir / "batman_bkm_trades.csv"
    timeline_path = out_dir / "batman_bkm_timeline.csv"
    summary_path.write_text(json.dumps(summary, indent=2))
    trades_df.to_csv(trades_path, index=False)
    timeline_df.to_csv(timeline_path, index=False)
    return {
        "summary": summary,
        "summary_path": summary_path,
        "trades_path": trades_path,
        "timeline_path": timeline_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay backtest for batman_bkm_monthly using rolling option CSV.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--catchup-not-before", type=str, default="09:35")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BatmanBKMConfig()
    result = run_backtest(
        csv_path=args.csv,
        out_dir=args.out_dir,
        cfg=cfg,
        catchup_not_before=_parse_time(args.catchup_not_before),
    )
    print(json.dumps(
        {
            "summary_path": str(result["summary_path"]),
            "trades_path": str(result["trades_path"]),
            "timeline_path": str(result["timeline_path"]),
            "summary": result["summary"],
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
