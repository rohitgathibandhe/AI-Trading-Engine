#!/usr/bin/env python
"""Backtest the VWAP-anchored intraday ruleset on our own candle history (state/candle_cache_*_5m.json).

The ruleset (from the research + the price-action gap we found):
  REGIME (first hour): Opening Range = first 3 candles (09:15-09:30). VWAP + its slope. Trend = price
    on one side of a SLOPING VWAP and beyond the OR; Range = flat VWAP, price oscillating in the OR.
  ENTRY (confirmation, never fade the wrong way):
    Trend-up   -> LONG on a PULLBACK: a candle dips to VWAP and closes back ABOVE it (reclaim).
    Trend-down -> SHORT on a pullback: a candle pops to VWAP and closes back BELOW it.
    Range      -> FADE the extreme toward VWAP: rejection candle at OR-high -> short; at OR-low -> long.
  STOP/TARGET: trend -> stop = STOP_R x OR range, target = TARGET_R x OR range. Range -> target = VWAP,
    stop = beyond the OR extreme. First-touch, else exit at close.

Directional P&L in index points (a proxy for the debit/credit structure's move). Tunable via the knobs
below so we can push it toward profitable and see what actually moves the needle.

  python scripts/intraday_ruleset_backtest.py
"""
from __future__ import annotations

import glob
import json
import os
import statistics
from pathlib import Path

STATE = Path(__file__).resolve().parents[1] / "data_engine" / "market_ai" / "state"


def _e(name, default):
    try:
        return type(default)(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ── Tunables (env-overridable so we can sweep) ────────────────────────────────────────────────
OR_BARS         = _e("BT_OR_BARS", 3)
ENTER_AFTER     = _e("BT_ENTER_AFTER", 6)
VWAP_SLOPE_LB   = _e("BT_SLOPE_LB", 4)
TREND_SLOPE_PTS = _e("BT_TREND_SLOPE", 3.0)
VWAP_TOUCH      = _e("BT_VWAP_TOUCH", 0.0007)
TARGET_R        = _e("BT_TARGET_R", 1.0)
STOP_R          = _e("BT_STOP_R", 0.8)
RANGE_STOP_BUF  = _e("BT_RANGE_STOP_BUF", 0.4)
TREND_ONLY      = _e("BT_TREND_ONLY", 0)     # 1 = skip range fades (test trend setups alone)


def _load(day_path):
    d = json.loads(Path(day_path).read_text())
    return d if isinstance(d, list) else (d.get("candles") or d.get("data") or [])


def _vwap_series(cs):
    pv = v = 0.0
    out = []
    for c in cs:
        tp = (c["high"] + c["low"] + c["close"]) / 3.0
        vol = c.get("volume") or 1.0
        pv += tp * vol; v += vol
        out.append(pv / v if v else tp)
    return out


def backtest_day(cs):
    if len(cs) < ENTER_AFTER + 4:
        return None
    highs = [c["high"] for c in cs]; lows = [c["low"] for c in cs]
    opens = [c["open"] for c in cs]; closes = [c["close"] for c in cs]
    vwap = _vwap_series(cs)
    or_hi = max(highs[:OR_BARS]); or_lo = min(lows[:OR_BARS]); or_rng = or_hi - or_lo
    if or_rng <= 0:
        return None

    for i in range(ENTER_AFTER, len(cs) - 1):
        slope = vwap[i] - vwap[i - VWAP_SLOPE_LB]
        trending = abs(slope) >= TREND_SLOPE_PTS
        entry = side = tgt = stop = None

        if trending and slope > 0 and closes[i] > vwap[i] and closes[i] > or_hi:
            # trend-up: LONG on a pullback that reclaims VWAP
            if lows[i] <= vwap[i] * (1 + VWAP_TOUCH) and closes[i] > vwap[i] and closes[i] > opens[i]:
                entry, side = closes[i], "LONG"
                tgt, stop = entry + TARGET_R * or_rng, entry - STOP_R * or_rng
        elif trending and slope < 0 and closes[i] < vwap[i] and closes[i] < or_lo:
            if highs[i] >= vwap[i] * (1 - VWAP_TOUCH) and closes[i] < vwap[i] and closes[i] < opens[i]:
                entry, side = closes[i], "SHORT"
                tgt, stop = entry - TARGET_R * or_rng, entry + STOP_R * or_rng
        elif not trending and not TREND_ONLY:
            # range: fade the extreme toward VWAP
            if highs[i] >= or_hi * (1 - 0.0003) and closes[i] < opens[i]:
                entry, side = closes[i], "SHORT"
                tgt, stop = vwap[i], or_hi + RANGE_STOP_BUF * or_rng
            elif lows[i] <= or_lo * (1 + 0.0003) and closes[i] > opens[i]:
                entry, side = closes[i], "LONG"
                tgt, stop = vwap[i], or_lo - RANGE_STOP_BUF * or_rng

        if entry is None:
            continue
        # walk forward: first touch of target or stop, else exit at close
        for j in range(i + 1, len(cs)):
            if side == "LONG":
                if highs[j] >= tgt: return (side, round(tgt - entry, 1), "target")
                if lows[j] <= stop: return (side, round(stop - entry, 1), "stop")
            else:
                if lows[j] <= tgt: return (side, round(entry - tgt, 1), "target")
                if highs[j] >= stop: return (side, round(entry - stop, 1), "stop")
        exitp = closes[-1]
        return (side, round((exitp - entry) if side == "LONG" else (entry - exitp), 1), "eod")
    return None


def day_signal(cs):
    """The ruleset's DAY-LEVEL read at ~09:45 (after the OR forms): 'TREND_UP' / 'TREND_DOWN' / 'RANGE'.
    Used to pick the option structure to score against the shadow book's real fills."""
    if len(cs) < ENTER_AFTER + 2:
        return None
    highs = [c["high"] for c in cs]; lows = [c["low"] for c in cs]; closes = [c["close"] for c in cs]
    vwap = _vwap_series(cs)
    i = ENTER_AFTER
    or_hi = max(highs[:OR_BARS]); or_lo = min(lows[:OR_BARS])
    slope = vwap[i] - vwap[i - VWAP_SLOPE_LB]
    if slope >= TREND_SLOPE_PTS and closes[i] > vwap[i] and closes[i] > or_hi:
        return "TREND_UP"
    if slope <= -TREND_SLOPE_PTS and closes[i] < vwap[i] and closes[i] < or_lo:
        return "TREND_DOWN"
    return "RANGE"


def main():
    files = sorted(glob.glob(str(STATE / "candle_cache_*_5m.json")))
    results = []
    for f in files:
        day = f.split("candle_cache_")[1][:10]
        try:
            r = backtest_day(_load(f))
        except Exception as e:  # noqa: BLE001
            r = None
        if r:
            results.append((day, *r))
    if not results:
        print("no trades generated"); return 0
    pnls = [p for _, _, p, _ in results]
    wins = [p for p in pnls if p > 0]
    print(f"RULESET BACKTEST — {len(results)} trading days with a signal")
    print(f"  total {sum(pnls):+.0f} pts | avg {statistics.mean(pnls):+.1f} | win {100*len(wins)/len(pnls):.0f}% "
          f"| best {max(pnls):+.0f} | worst {min(pnls):+.0f}")
    print(f"  knobs: OR={OR_BARS}bars slope>={TREND_SLOPE_PTS} target={TARGET_R}xOR stop={STOP_R}xOR\n")
    for day, side, pnl, how in results:
        print(f"   {day}  {side:5} {pnl:+6.0f} pts  ({how})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
