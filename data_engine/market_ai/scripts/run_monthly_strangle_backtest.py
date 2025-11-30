"""
Monthly Strangle backtest using rolling_*.json cache (with synthetic greeks).

We synthesize an option chain snapshot from the rolling cache for each entry day,
compute deltas from spot/strike/iv/time, pick strikes via monthly_strangle_manager,
and simulate MTM until TP/SL/expiry/max-hold.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

import sys
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from market_ai.strategies.monthly_strangle_manager import (
    MonthlyStrangleConfig,
    manage_basket,
    propose_entry,
)
from market_ai.strategies.models import LegSide, OptionLeg, OptionType, StrategyType

log = logging.getLogger("monthly_strangle_backtest")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROLL_DIR = ROOT / "state" / "rolling_option"


def bs_delta(spot: float, strike: float, iv: float, t_years: float, call: bool = True) -> float:
    if spot <= 0 or strike <= 0 or iv <= 0 or t_years <= 0:
        return 0.0
    try:
        sigma = iv / 100.0
        d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * t_years) / (sigma * math.sqrt(t_years))
        from math import erf, sqrt
        nd1 = 0.5 * (1.0 + erf(d1 / math.sqrt(2.0)))
        return nd1 if call else nd1 - 1.0
    except Exception:
        return 0.0


def parse_rolling_files() -> List[dict]:
    rows: List[dict] = []
    for path in sorted(ROLL_DIR.glob("rolling_*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        ce = data.get("data", {}).get("ce")
        pe = data.get("data", {}).get("pe")
        for opt_key, payload in [("CE", ce), ("PE", pe)]:
            if not payload:
                continue
            ts_list = payload.get("timestamp", [])
            strike_list = payload.get("strike", [])
            spot_list = payload.get("spot", [])
            close_list = payload.get("close", [])
            iv_list = payload.get("iv", [])
            for ts, k, s, c, iv in zip(ts_list, strike_list, spot_list, close_list, iv_list):
                try:
                    dt = datetime.fromtimestamp(ts)
                except Exception:
                    continue
                rows.append(
                    {
                        "timestamp": dt,
                        "date": dt.date(),
                        "time": dt.time(),
                        "strike": float(k),
                        "spot": float(s),
                        "ltp": float(c),
                        "iv": float(iv),
                        "option_type": opt_key,
                    }
                )
    return rows


def next_monthly_expiry(d: date) -> date:
    # Use next Tuesday in the same month if possible; else first Tuesday of next month.
    cur = d
    while True:
        if cur.weekday() == 1:  # Tuesday
            if cur.month == d.month:
                return cur
            # first Tuesday of next month
            return cur
        cur += timedelta(days=1)


def build_chain_for_date(rows: List[dict], trade_date: date, entry_end: time) -> List[dict]:
    daily = [r for r in rows if r["date"] == trade_date and r["time"] <= entry_end]
    if not daily:
        return []
    # Use last quote per strike/opt before entry_end
    best: Dict[Tuple[float, str], dict] = {}
    for r in daily:
        key = (r["strike"], r["option_type"])
        if key not in best or r["timestamp"] > best[key]["timestamp"]:
            best[key] = r
    trade_ts = max(r["timestamp"] for r in daily)
    expiry = next_monthly_expiry(trade_date)
    t_years = max((expiry - trade_date).days, 1) / 365.0
    chain: List[dict] = []
    for (strike, opt), r in best.items():
        delta = bs_delta(r["spot"], strike, r.get("iv", 0.0), t_years, call=opt == "CE")
        chain.append(
            {
                "strike": strike,
                "option_type": opt,
                "ltp": r["ltp"],
                "spot": r["spot"],
                "delta": delta,
                "expiry": expiry,
                "security_id": None,
                "symbol": "NIFTY",
            }
        )
    return chain


def _quotes_by_date(rows: List[dict], expiry: date) -> Dict[Tuple[date, float, str], dict]:
    """Return last quote of each date/strike/option_type for the chosen expiry."""
    by_key: Dict[Tuple[date, float, str], dict] = {}
    for r in rows:
        if r["expiry"] != expiry:
            continue
        key = (r["date"], r["strike"], r["option_type"])
        if key not in by_key or r["timestamp"] > by_key[key]["timestamp"]:
            by_key[key] = r
    return by_key


def _adx_by_date(rows: List[dict]) -> Dict[date, float]:
    """
    Very lightweight ADX approximation from daily spot highs/lows/closes.
    If insufficient data, ADX will not be populated for that date.
    """
    from collections import defaultdict

    daily: Dict[date, Dict[str, float]] = defaultdict(lambda: {"high": -1e9, "low": 1e9, "close": None})
    for r in rows:
        d = r["date"]
        spot = r.get("spot")
        if spot is None:
            continue
        daily[d]["high"] = max(daily[d]["high"], spot)
        daily[d]["low"] = min(daily[d]["low"], spot)
        # use last quote as close
        if daily[d]["close"] is None or r["timestamp"] > datetime.combine(d, daily[d].get("time_hint", time.min)):
            daily[d]["close"] = spot
            daily[d]["time_hint"] = r["timestamp"].time()

    dates = sorted(daily.keys())
    if len(dates) < 15:
        return {}

    dx_list: List[Tuple[date, float]] = []
    prev_tr = prev_dm_pos = prev_dm_neg = None
    prev_close = None
    for d in dates:
        o = daily[d]
        if o["close"] is None:
            continue
        high = o["high"]
        low = o["low"]
        close = o["close"]
        if prev_close is None:
            prev_close = close
            continue
        tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
        dm_pos = max(high - prev_close, 0)
        dm_neg = max(prev_close - low, 0)
        if prev_tr is None:
            prev_tr, prev_dm_pos, prev_dm_neg = tr, dm_pos, dm_neg
        else:
            prev_tr = prev_tr * 13 / 14 + tr
            prev_dm_pos = prev_dm_pos * 13 / 14 + dm_pos
            prev_dm_neg = prev_dm_neg * 13 / 14 + dm_neg
        if prev_tr == 0:
            prev_close = close
            continue
        di_pos = 100 * (prev_dm_pos / prev_tr)
        di_neg = 100 * (prev_dm_neg / prev_tr)
        dx = 100 * abs(di_pos - di_neg) / max(di_pos + di_neg, 1e-9)
        dx_list.append((d, dx))
        prev_close = close

    adx: Dict[date, float] = {}
    if not dx_list:
        return adx
    # rolling average of DX
    window = 14
    for i in range(len(dx_list)):
        if i + 1 < window:
            continue
        window_dx = [dx for _, dx in dx_list[i + 1 - window : i + 1]]
        adx[dx_list[i][0]] = sum(window_dx) / len(window_dx)
    return adx


def simulate_entry_and_exit(chain_rows: List[dict], cfg: MonthlyStrangleConfig, lot_size: int, trade_date: date) -> dict:
    entry_time = datetime.combine(trade_date, time(9, 30))
    if not chain_rows:
        return {"entry_date": trade_date, "reason": "NO_CHAIN", "pnl": 0.0}
    market_spot = chain_rows[0].get("spot") or 0.0
    decision = propose_entry(
        now=entry_time,
        spot=market_spot,
        adx=0.0,
        max_body_pct=0.0,
        gap_pct=0.0,
        monthly_range_frac=0.5,
        vix=14.0,
        vix_rising=True,
        option_chain=chain_rows,
        expiry=chain_rows[0]["expiry"],
        lot_size=lot_size,
        cfg=cfg,
    )
    if decision.action_type != "OPEN":
        return {"entry_date": trade_date, "reason": decision.reason, "pnl": 0.0}
    legs = decision.legs_to_open
    # Populate entry prices from chain snapshot
    entry_lookup = {(c["strike"], c["option_type"]): c for c in chain_rows}
    for leg in legs:
        key = (leg.strike, "CE" if leg.option_type == OptionType.CALL else "PE")
        leg.entry_price = entry_lookup.get(key, {}).get("ltp", leg.entry_price or 0.0)

    expiry = chain_rows[0]["expiry"]
    # Build quote lookup for entire horizon
    by_date = _quotes_by_date(chain_rows, expiry)
    # ADX approximation from spot path of this expiry
    adx_by_date = _adx_by_date(chain_rows)

    daily_dates = sorted({d for (d, _, _) in by_date.keys() if d >= trade_date})
    if not daily_dates:
        return {"entry_date": trade_date, "expiry": expiry, "reason": "NO_QUOTES", "pnl": 0.0}

    sign = lambda leg: -1 if leg.side == LegSide.SELL else 1
    margin = sum(abs(leg.entry_price * leg.quantity) for leg in legs if leg.side == LegSide.SELL)
    tp_level = margin * cfg.basket.tp_pct
    sl_level = -abs(margin * cfg.basket.sl_pct)
    trail_trigger = margin * 0.006  # +0.6%
    trail_floor = margin * 0.004    # lock +0.4%
    trail_active = False

    def leg_delta(spot: float, leg: OptionLeg, iv: float, cur_date: date) -> float:
        t_years = max((expiry - cur_date).days, 1) / 365.0
        return bs_delta(spot, leg.strike, iv, t_years, call=leg.option_type == OptionType.CALL)

    for day in daily_dates:
        # MTM for the day (last quote)
        mtm = 0.0
        both_high_delta = False
        for leg in legs:
            key = (day, leg.strike, "CE" if leg.option_type == OptionType.CALL else "PE")
            quote = by_date.get(key)
            if not quote:
                continue
            mtm += (quote["ltp"] - leg.entry_price) * leg.quantity * sign(leg)
        # Delta guard (dual delta)
        deltas = []
        for leg in legs:
            key = (day, leg.strike, "CE" if leg.option_type == OptionType.CALL else "PE")
            quote = by_date.get(key)
            if not quote:
                continue
            deltas.append(abs(leg_delta(quote["spot"], leg, quote.get("iv", 0.0), day)))
        if len(deltas) == len(legs) and all(d > 0.25 for d in deltas):
            return {"entry_date": trade_date, "expiry": expiry, "reason": "DELTA_EXIT", "pnl": mtm}
        # ADX guard
        adx_val = adx_by_date.get(day)
        if adx_val is not None and adx_val > 30:
            return {"entry_date": trade_date, "expiry": expiry, "reason": "ADX_EXIT", "pnl": mtm}

        if mtm >= tp_level:
            return {"entry_date": trade_date, "expiry": expiry, "reason": "TP", "pnl": mtm}
        if mtm <= sl_level:
            return {"entry_date": trade_date, "expiry": expiry, "reason": "SL", "pnl": mtm}
        if not trail_active and mtm >= trail_trigger:
            trail_active = True
        if trail_active and mtm <= trail_floor:
            return {"entry_date": trade_date, "expiry": expiry, "reason": "TRAIL_EXIT", "pnl": mtm}
        if (expiry - day).days <= cfg.basket.hard_exit_days_to_expiry:
            return {"entry_date": trade_date, "expiry": expiry, "reason": "DTE_EXIT", "pnl": mtm}
        if (day - trade_date).days >= cfg.basket.max_hold_days:
            return {"entry_date": trade_date, "expiry": expiry, "reason": "MAX_HOLD_EXIT", "pnl": mtm}

    # If no exit triggered, mark hold end at last MTM
    last_day = daily_dates[-1]
    mtm = 0.0
    for leg in legs:
        key = (last_day, leg.strike, "CE" if leg.option_type == OptionType.CALL else "PE")
        quote = by_date.get(key)
        if not quote:
            continue
        mtm += (quote["ltp"] - leg.entry_price) * leg.quantity * sign(leg)
    return {"entry_date": trade_date, "expiry": expiry, "reason": "HOLD_END", "pnl": mtm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=str, default=(date.today() - timedelta(days=365)).isoformat())
    ap.add_argument("--end", type=str, default=date.today().isoformat())
    ap.add_argument("--lot-size", type=int, default=75)
    args = ap.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    rows = parse_rolling_files()
    if not rows:
        log.error("No rolling cache rows found.")
        return
    cfg = MonthlyStrangleConfig()
    results = []
    # Entry on first 7 days of each month if data present
    for d in sorted({r["date"] for r in rows if start <= r["date"] <= end}):
        if not (1 <= d.day <= 7):
            continue
        chain = build_chain_for_date(rows, d, time(10, 30))
        res = simulate_entry_and_exit(chain, cfg, args.lot_size, d)
        results.append(res)
    out_dir = ROOT / "reports" / "monthly_strangle_backtest"
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_path = out_dir / "monthly_strangle_trades.csv"
    pd.DataFrame(results).to_csv(trades_path, index=False)
    summary = {
        "rows": len(results),
        "pnl_sum": float(sum(r.get("pnl", 0.0) for r in results)),
    }
    (out_dir / "monthly_strangle_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Backtest complete. rows=%s pnl_sum=%.2f saved in %s", len(results), summary["pnl_sum"], trades_path)


if __name__ == "__main__":
    main()
