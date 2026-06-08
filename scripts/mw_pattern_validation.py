#!/usr/bin/env python3
"""
Validate M (double-top) / W (double-bottom) reversal patterns as trade signals.

Reconstructs 5-min OHLC bars from the per-minute spot in the option-chain archive
(state/rolling_option/<date>/), detects double-top / double-bottom with a neckline
break, and measures whether the reversal actually played out over the next N bars.

Question: does an M near resistance predict a DOWN move, and a W near support
predict an UP move? If yes, these become high-quality reversal triggers for the
v83 engine (especially valuable because the raw directional bias is only ~36-44%).
"""
from __future__ import annotations
import os, glob
import pandas as pd

CHAINS = '/Users/Rohit/AI-Trading-Engine/data_engine/market_ai/state/rolling_option'
TOL = 0.0015      # two tops/bottoms within 0.15% = "equal"
LOOKFWD = 6       # bars after neckline break to measure reversal (6*5m = 30min)
PIVOT_W = 2       # bars each side for a local pivot


def bars_from_spot(date):
    p = f'{CHAINS}/{date}/rolling_option.parquet'
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    # one spot per minute (any strike row carries spot)
    s = df.groupby('tradeTime')['spot'].first().reset_index()
    s['t'] = pd.to_datetime('2000-01-01 ' + s['tradeTime'])
    s = s.set_index('t')['spot']
    o = s.resample('5min').agg(['first', 'max', 'min', 'last']).dropna()
    o.columns = ['open', 'high', 'low', 'close']
    return o.reset_index(drop=True)


def pivots(vals, kind):
    out = []
    for i in range(PIVOT_W, len(vals) - PIVOT_W):
        w = vals[i - PIVOT_W:i + PIVOT_W + 1]
        if kind == 'high' and vals[i] == max(w):
            out.append((i, vals[i]))
        if kind == 'low' and vals[i] == min(w):
            out.append((i, vals[i]))
    return out


def detect(bars):
    """Yield (idx_breakbar, kind, neckline) for confirmed M/W."""
    highs = pivots(list(bars['high']), 'high')
    lows = pivots(list(bars['low']), 'low')
    sig = []
    # M / double top: two near-equal highs, neckline = trough between, break below
    for a in range(len(highs)):
        for b in range(a + 1, len(highs)):
            i1, h1 = highs[a]; i2, h2 = highs[b]
            if i2 - i1 < 2 or i2 - i1 > 12:
                continue
            if abs(h1 - h2) / h1 > TOL:
                continue
            trough = min(bars['low'][i1:i2 + 1])
            # neckline break: a close below trough after i2
            for k in range(i2 + 1, min(len(bars), i2 + 8)):
                if bars['close'][k] < trough:
                    sig.append((k, 'M', trough, max(h1, h2)))
                    break
    # W / double bottom: two near-equal lows, neckline = peak between, break above
    for a in range(len(lows)):
        for b in range(a + 1, len(lows)):
            i1, l1 = lows[a]; i2, l2 = lows[b]
            if i2 - i1 < 2 or i2 - i1 > 12:
                continue
            if abs(l1 - l2) / l1 > TOL:
                continue
            peak = max(bars['high'][i1:i2 + 1])
            for k in range(i2 + 1, min(len(bars), i2 + 8)):
                if bars['close'][k] > peak:
                    sig.append((k, 'W', peak, min(l1, l2)))
                    break
    return sig


def main():
    dates = sorted(os.listdir(CHAINS))
    m_hits = m_tot = w_hits = w_tot = 0
    detail = []
    for d in dates:
        bars = bars_from_spot(d)
        if bars is None or len(bars) < 12:
            continue
        for k, kind, neck, extreme in detect(bars):
            if k + 1 >= len(bars):
                continue
            entry = bars['close'][k]
            fwd = bars['close'][min(len(bars) - 1, k + LOOKFWD)]
            move = fwd - entry
            if kind == 'M':            # expect DOWN
                hit = move < 0
                m_tot += 1; m_hits += hit
            else:                      # W expect UP
                hit = move > 0
                w_tot += 1; w_hits += hit
            detail.append((d, kind, round(entry, 1), round(move, 1), hit))

    print(f"Validated over {len(dates)} archived sessions\n")
    print("=== M (double-top) → expect DOWN reversal ===")
    if m_tot:
        print(f"  {m_tot} signals, reversal hit {100*m_hits/m_tot:.0f}% over next {LOOKFWD*5}min")
    print("=== W (double-bottom) → expect UP reversal ===")
    if w_tot:
        print(f"  {w_tot} signals, reversal hit {100*w_hits/w_tot:.0f}% over next {LOOKFWD*5}min")
    print("\n=== sample signals ===")
    for d, kind, e, mv, hit in detail[:20]:
        print(f"  {d} {kind} entry={e} move={mv:+.0f} {'HIT' if hit else 'miss'}")


if __name__ == '__main__':
    main()
