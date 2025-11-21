#!/usr/bin/env python3
"""
Offline backtest for the regime-aware agent (StrategySelector + hedged strangle).

It replays a historical feature CSV (default: reports/intraday_from_rolling_latest.csv),
derives a lightweight option-chain snapshot from the provided strikes (ATM +/- offsets),
and feeds those into the same StrategySelector used in production. Trades are opened/closed
exactly as the live agent would, with MTM and realized P&L tracked over time.

Outputs:
  - reports/regime_agent_trades.csv (chronological trade log)
  - reports/regime_agent_summary.json (aggregate metrics)
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, date as dt_date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

import sys

SCRIPT_ROOT = Path(__file__).resolve()
REPO_ROOT = SCRIPT_ROOT.parents[3]
for candidate in (REPO_ROOT, REPO_ROOT / "data_engine"):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from market_ai.strategies import (
    StrategySelector,
    MarketSnapshot,
    RiskConfig,
    OptionLeg,
    TrendContext,
    TrendSide,
    LegSide,
    StrategyType,
)
from market_ai.strategies.trend_detector import detect_trend_from_open
from market_ai.strategies import strangle_engine, credit_spread_engine

# Backtest-mode overrides: relax hedge caps so structures don’t get skipped in sim.
strangle_engine.HEDGE_PREFERRED_MAX = 12.0
strangle_engine.HEDGE_FALLBACK_MAX = 20.0
strangle_engine.HEDGE_MAX_PRICE = strangle_engine.HEDGE_PREFERRED_MAX
credit_spread_engine.HEDGE_PREFERRED_MAX = 12.0
credit_spread_engine.HEDGE_FALLBACK_MAX = 20.0
credit_spread_engine.HEDGE_MAX_PRICE = credit_spread_engine.HEDGE_PREFERRED_MAX

DATA_ROOT = REPO_ROOT
DEFAULT_CSV = DATA_ROOT / "reports" / "intraday_from_rolling_latest.csv"
DEFAULT_TRADES = DATA_ROOT / "reports" / "regime_agent_trades.csv"
DEFAULT_SUMMARY = DATA_ROOT / "reports" / "regime_agent_summary.json"


CALL_PREFIXES = [
    ("atm_call", "CE"),
    ("call_atm_plus2", "CE"),
    ("call_atm_minus2", "CE"),
    ("call_atm_plus4", "CE"),
    ("call_atm_minus4", "CE"),
    ("call_atm_plus6", "CE"),
    ("call_atm_minus6", "CE"),
]

PUT_PREFIXES = [
    ("atm_put", "PE"),
    ("put_atm_plus2", "PE"),
    ("put_atm_minus2", "PE"),
    ("put_atm_plus4", "PE"),
    ("put_atm_minus4", "PE"),
    ("put_atm_plus6", "PE"),
    ("put_atm_minus6", "PE"),
]


def _coerce_float(val) -> Optional[float]:
    try:
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None


def _row_to_chain(row: pd.Series, expiry: dt_date) -> List[Dict[str, float]]:
    """Build a minimal option-chain list from the available prefixes."""
    rows: List[Dict[str, float]] = []

    def add(prefix: str, opt_type: str) -> None:
        strike = _coerce_float(row.get(f"{prefix}_strike"))
        ltp = _coerce_float(row.get(f"{prefix}_ltp"))
        delta = _coerce_float(row.get(f"{prefix}_delta"))
        if strike is None or ltp is None:
            return
        rows.append(
            {
                "expiry": expiry,
                "option_type": "CE" if opt_type == "CE" else "PE",
                "strike": strike,
                "ltp": ltp,
                "delta": delta,
                "security_id": f"{opt_type}_{int(round(strike))}",
            }
        )

    for prefix, opt in CALL_PREFIXES:
        add(prefix, opt)
    for prefix, opt in PUT_PREFIXES:
        add(prefix, opt)

    # Add synthetic far OTM hedges so engines can always find cheap wings.
    atm_ref = (
        _coerce_float(row.get("atm_call_strike"))
        or _coerce_float(row.get("atm_put_strike"))
        or _coerce_float(row.get("spot"))
    )
    if atm_ref:
        synthetic_offsets = [
            (400.0, 9.0),
            (600.0, 6.0),
            (800.0, 3.0),
            (1000.0, 2.0),
            (1200.0, 1.2),
        ]
        for offset, base_ltp in synthetic_offsets:
            noise = random.uniform(0.75, 1.35)
            ltp_ce = max(0.5, base_ltp * noise)
            noise = random.uniform(0.75, 1.35)
            ltp_pe = max(0.5, base_ltp * noise)
            rows.append(
                {
                    "expiry": expiry,
                    "option_type": "CE",
                    "strike": atm_ref + offset,
                    "ltp": ltp_ce,
                    "delta": None,
                    "security_id": f"SYN_CE_{int(round(atm_ref + offset))}",
                }
            )
            rows.append(
                {
                    "expiry": expiry,
                    "option_type": "PE",
                    "strike": atm_ref - offset,
                    "ltp": ltp_pe,
                    "delta": None,
                    "security_id": f"SYN_PE_{int(round(atm_ref - offset))}",
                }
            )
    return rows


def _volume_sum(block: pd.DataFrame) -> float:
    total = 0.0
    for col in ("atm_call_volume", "atm_put_volume"):
        if col in block.columns:
            total += block[col].fillna(0).astype(float).sum()
    return float(total)


def _calc_vwap(candles: List[dict]) -> float:
    total_volume = 0.0
    pv_sum = 0.0
    for candle in candles:
        vol = float(candle.get("volume") or 0.0)
        if vol <= 0:
            continue
        typical = (
            float(candle.get("open") or 0.0)
            + float(candle.get("high") or 0.0)
            + float(candle.get("low") or 0.0)
            + float(candle.get("close") or 0.0)
        ) / 4.0
        total_volume += vol
        pv_sum += typical * vol
    if total_volume > 0:
        return pv_sum / total_volume
    closes = [float(candle.get("close") or 0.0) for candle in candles if candle.get("close") is not None]
    return closes[-1] if closes else 0.0


def _build_daily_candles(df: pd.DataFrame) -> Tuple[Dict[dt_date, List[dict]], Dict[dt_date, float], Dict[dt_date, float]]:
    candles_lookup: Dict[dt_date, List[dict]] = {}
    avg_volume_lookup: Dict[dt_date, float] = {}
    vwap_lookup: Dict[dt_date, float] = {}
    grouped = df.groupby("trade_date")
    for trade_day, group in grouped:
        group = group.sort_values("timestamp")
        base = datetime.combine(trade_day, datetime.strptime("09:15", "%H:%M").time())
        candles: List[dict] = []
        for offset in (0, 5):
            window_start = base + timedelta(minutes=offset)
            window_end = window_start + timedelta(minutes=5)
            mask = (group["timestamp"] >= window_start) & (group["timestamp"] < window_end)
            subset = group.loc[mask]
            if subset.empty:
                continue
            candles.append(
                {
                    "open": float(subset["spot"].iloc[0]),
                    "high": float(subset["spot"].max()),
                    "low": float(subset["spot"].min()),
                    "close": float(subset["spot"].iloc[-1]),
                    "volume": _volume_sum(subset),
                }
            )
        candles_lookup[trade_day] = candles
        vols = [candle.get("volume", 0.0) for candle in candles if candle]
        avg_volume_lookup[trade_day] = float(sum(vols) / len(vols)) if vols else 0.0
        vwap_lookup[trade_day] = _calc_vwap(candles)
    return candles_lookup, avg_volume_lookup, vwap_lookup


def _as_snapshot(row: pd.Series, candles_lookup: Dict[dt_date, List[dict]]) -> MarketSnapshot:
    ts_val = row["timestamp"]
    if isinstance(ts_val, pd.Timestamp):
        now = ts_val.to_pydatetime()
    else:
        now = pd.to_datetime(ts_val).to_pydatetime()
    spot = float(row["spot"])
    prev_high = _coerce_float(row.get("prev_day_high")) or spot
    prev_low = _coerce_float(row.get("prev_day_low")) or spot
    prev_close = _coerce_float(row.get("prev_day_close")) or spot
    vix = _coerce_float(row.get("spot_volatility")) or 0.0
    candles = candles_lookup.get(now.date(), [])
    return MarketSnapshot(
        symbol="NIFTY",
        spot=spot,
        candles_5m=[dict(candle) for candle in candles],
        yesterday_high=prev_high,
        yesterday_low=prev_low,
        yesterday_close=prev_close,
        india_vix=vix,
        now=now,
    )


def _leg_key(leg: OptionLeg) -> Tuple[str, float]:
    opt = "CE" if leg.option_type.value.startswith("CALL") else "PE"
    return opt, float(leg.strike)


def _update_leg_ltps(legs: List[OptionLeg], chain_map: Dict[Tuple[str, float], float]) -> None:
    for leg in legs:
        ltp = chain_map.get(_leg_key(leg))
        if ltp is not None:
            leg.current_ltp = ltp


def _lookup_ltp(chain_map: Dict[Tuple[str, float], float], leg: OptionLeg) -> Optional[float]:
    return chain_map.get(_leg_key(leg))


def _compute_mtm(legs: List[OptionLeg]) -> float:
    mtm = 0.0
    for leg in legs:
        if leg.current_ltp is None:
            continue
        if leg.side.value if hasattr(leg.side, "value") else str(leg.side) == "SELL":
            pnl = (leg.entry_price - leg.current_ltp) * leg.quantity
        else:
            pnl = (leg.current_ltp - leg.entry_price) * leg.quantity
        mtm += pnl
    return mtm


def run_backtest(csv_path: Path, trades_out: Path, summary_out: Path) -> None:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    df["trade_date"] = df["timestamp"].dt.date
    df = df.sort_values("timestamp")
    candles_lookup, avg_volume_lookup, vwap_lookup = _build_daily_candles(df)
    selector = StrategySelector(symbol="NIFTY", lot_size=75)
    risk = RiskConfig(
        max_intraday_loss=-3000,
        intraday_target=4000,
        allow_carry_forward=True,
        max_carry_days=2,
        vix_carry_threshold=12.0,
        last_entry_time=datetime.strptime("14:45", "%H:%M").time(),
        force_exit_time=datetime.strptime("15:15", "%H:%M").time(),
        max_daily_loss_pct=0.03,
        max_daily_trades=5,
        partial_target_pct=0.65,
        per_leg_sl_mult=1.6,
        per_leg_tp_mult=0.5,
        vix_adaptive_low=12.0,
        vix_adaptive_high=20.0,
        strangle_delta_low=0.15,
        strangle_delta_high=0.15,
        strangle_offset_low=150.0,
        strangle_offset_high=150.0,
        spread_short_delta_low=0.25,
        spread_short_delta_high=0.25,
    )

    open_legs: List[OptionLeg] = []
    realized = 0.0
    trades: List[Dict[str, Any]] = []
    equity_curve: List[Tuple[datetime, float]] = []

    current_day = None
    daily_trades = 0
    day_realized = 0.0
    day_mode = "NORMAL"
    daily_max_loss = -2000.0
    max_baskets_per_day = 2
    baskets_opened_today = 0
    for _, row in df.iterrows():
        try:
            expiry = pd.to_datetime(row["expiryDate"]).date()
        except Exception:
            continue
        chain_rows = _row_to_chain(row, expiry)
        if not chain_rows:
            continue
        chain_map = {(r["option_type"], r["strike"]): r["ltp"] for r in chain_rows}
        _update_leg_ltps(open_legs, chain_map)
        market = _as_snapshot(row, candles_lookup)
        basket_mtm = _compute_mtm(open_legs)
        trade_day = market.now.date()
        prev_day = trade_day - timedelta(days=1)
        avg_volume = avg_volume_lookup.get(prev_day, avg_volume_lookup.get(trade_day, 0.0))
        vwap = vwap_lookup.get(trade_day, market.spot)
        pivot = (market.yesterday_high + market.yesterday_low + market.yesterday_close) / 3.0
        trend_ctx = detect_trend_from_open(market, avg_volume or 0.0, vwap=vwap, pivot=pivot)
        if trade_day != current_day:
            current_day = trade_day
            daily_trades = 0
            day_realized = 0.0
            day_mode = "NORMAL"
            baskets_opened_today = 0

        def _close_all(reason: str) -> None:
            nonlocal realized, day_realized, open_legs
            for leg in list(open_legs):
                ltp_close = _lookup_ltp(chain_map, leg)
                if ltp_close is None:
                    continue
                pnl_raw = (leg.entry_price - ltp_close) * leg.quantity if leg.side == LegSide.SELL else (ltp_close - leg.entry_price) * leg.quantity
                remaining = daily_max_loss - day_realized
                pnl = max(pnl_raw, remaining)
                realized += pnl
                day_realized += pnl
                trades.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "event": "CLOSE",
                        "action": reason,
                        "strike": leg.strike,
                        "option_type": leg.option_type.value,
                        "side": leg.side.value,
                        "quantity": leg.quantity,
                        "price": ltp_close,
                        "pnl": pnl,
                        "strategy_type": getattr(leg, "strategy_type", StrategyType.NONE).name,
                        "regime": getattr(leg, "regime_label", trend_ctx.trend_side.name),
                    }
                )
                open_legs.remove(leg)

        decision = selector.decide(
            market=market,
            option_chain=chain_rows,
            expiry=expiry,
            risk=risk,
            current_positions=open_legs,
            basket_mtm=basket_mtm,
            trend_ctx=trend_ctx,
            daily_trades=daily_trades,
        )
        timestamp = market.now
        action = decision.action_type
        basket_mtm = _compute_mtm(open_legs)
        day_pnl = day_realized + basket_mtm

        # Hard daily lockout
        if day_mode != "LOCKED_RED" and day_pnl <= daily_max_loss:
            _close_all("DAILY_LOCK_RED")
            day_realized = max(day_realized, daily_max_loss)
            day_mode = "LOCKED_RED"
            equity_curve.append((timestamp, realized + _compute_mtm(open_legs)))
            continue

        # Hard per-basket SL
        if open_legs and basket_mtm <= -1000.0:
            _close_all("BASKET_SL_HIT")
            day_realized = max(day_realized, daily_max_loss)
            equity_curve.append((timestamp, realized + _compute_mtm(open_legs)))
            continue

        if day_mode == "LOCKED_RED":
            # skip all opens when locked
            if action.startswith("OPEN"):
                equity_curve.append((timestamp, realized + _compute_mtm(open_legs)))
                continue

        if action.startswith("OPEN"):
            if baskets_opened_today >= max_baskets_per_day:
                equity_curve.append((timestamp, realized + _compute_mtm(open_legs)))
                continue
            opened_any = False
            for leg in decision.legs_to_open:
                ltp = _lookup_ltp(chain_map, leg)
                if ltp is None:
                    continue
                new_leg = OptionLeg(
                    symbol=leg.symbol,
                    expiry=leg.expiry,
                    strike=leg.strike,
                    option_type=leg.option_type,
                    side=leg.side,
                    quantity=leg.quantity,
                    entry_price=ltp,
                    security_id=leg.security_id,
                    current_ltp=ltp,
                    strategy_type=decision.strategy_type,
                    opened_at=timestamp,
                )
                setattr(new_leg, "regime_label", trend_ctx.trend_side.name)
                open_legs.append(new_leg)
                trades.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "event": "OPEN",
                        "action": action,
                        "strike": leg.strike,
                        "option_type": leg.option_type.value,
                        "side": leg.side.value,
                        "quantity": leg.quantity,
                        "price": ltp,
                        "strategy_type": decision.strategy_type.name,
                        "regime": trend_ctx.trend_side.name,
                    }
                )
                opened_any = True
            if opened_any:
                daily_trades += 1
                baskets_opened_today += 1
        elif action.startswith("CLOSE"):
            closing = decision.legs_to_close if decision.legs_to_close else open_legs.copy()
            for leg in closing:
                match = next(
                    (
                        existing
                        for existing in open_legs
                        if existing.strike == leg.strike
                        and existing.option_type == leg.option_type
                        and existing.side == leg.side
                    ),
                    None,
                )
                if not match:
                    continue
                ltp = _lookup_ltp(chain_map, match)
                if ltp is None:
                    continue
                if match.side == LegSide.SELL:
                    pnl = (match.entry_price - ltp) * match.quantity
                else:
                    pnl = (ltp - match.entry_price) * match.quantity
                remaining = daily_max_loss - day_realized
                pnl = max(pnl, remaining)
                realized += pnl
                day_realized += pnl
                trades.append(
                    {
                        "timestamp": timestamp.isoformat(),
                        "event": "CLOSE",
                        "action": action,
                        "strike": match.strike,
                        "option_type": match.option_type.value,
                        "side": match.side.value,
                        "quantity": match.quantity,
                        "price": ltp,
                        "pnl": pnl,
                        "strategy_type": getattr(match, "strategy_type", StrategyType.NONE).name,
                        "regime": getattr(match, "regime_label", trend_ctx.trend_side.name),
                    }
                )
                open_legs.remove(match)
            if day_realized <= daily_max_loss:
                _close_all("DAILY_LOCK_RED")
                day_realized = max(day_realized, daily_max_loss)
                day_mode = "LOCKED_RED"
        equity_curve.append((timestamp, realized + _compute_mtm(open_legs)))

    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(trades_out, index=False)
    breakdown: List[Dict[str, Any]] = []
    if not trades_df.empty:
        close_df = trades_df[trades_df["event"] == "CLOSE"].copy()
        if not close_df.empty:
            grouped = close_df.groupby([close_df["regime"], close_df["strategy_type"]])
            for (regime, strat), grp in grouped:
                breakdown.append(
                    {
                        "regime": regime,
                        "strategy_type": strat,
                        "trades": int(len(grp)),
                        "realized_pnl": float(grp.get("pnl", 0).sum()),
                    }
                )
    summary = {
        "trades": len(trades),
        "realized_pnl": realized,
        "open_positions": len(open_legs),
        "final_equity": equity_curve[-1][1] if equity_curve else realized,
        "breakdown": breakdown,
    }
    summary_out.write_text(json.dumps(summary, indent=2))
    print(f"Backtest completed: {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Regime-aware agent backtest")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="Input historical features CSV")
    parser.add_argument("--trades-out", type=Path, default=DEFAULT_TRADES)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    run_backtest(args.csv, args.trades_out, args.summary_out)


if __name__ == "__main__":
    main()
