#!/usr/bin/env python3
"""
Real-premium backtest — true rupee P&L, no spot proxy.

For every historical decision on days where we also have the archived option
chain (state/rolling_option/<date>/rolling_option.parquet), construct the actual
credit spread the agent would place at the OI wall, price it with REAL premiums,
simulate the rest-of-session with TP/SL exits, and compute realized P&L.

Spread construction (defined-risk credit spread, width = WIDTH points):
  - bearish -> BEAR_CALL : short call at call_wall, long call at call_wall+WIDTH
  - bullish -> BULL_PUT  : short put  at put_wall,  long put  at put_wall-WIDTH

Exit rules (intraday):
  - TP: buy spread back when its value <= (1-TP_FRAC) * entry_credit  (keep TP_FRAC)
  - SL: when spread value >= entry_credit + SL_MULT*entry_credit       (lose SL_MULT x)
  - else square off at SQUAREOFF time

Outputs win rate, avg win/loss, expectancy in rupees — the numbers needed to
trust the edge with real money.

Usage: python scripts/real_premium_backtest.py
"""
from __future__ import annotations
import sqlite3, json, os
from collections import defaultdict
import pandas as pd

STATE = '/Users/Rohit/AI-Trading-Engine/data_engine/market_ai/state'
DB = f'{STATE}/intraday_v83_paper_live_learning.sqlite3'
CHAINS = f'{STATE}/rolling_option'

LOT = 65
WIDTH = 100          # spread width in points
TP_FRAC = 0.50       # take profit at 50% of credit captured
SL_MULT = 1.0        # stop at 1x credit loss (1:1)
SQUAREOFF = '15:20'
ENTRY_MIN_HOUR = 0   # set >0 to restrict entry window (we sweep separately)


def spot_of(m):
    orh, pv = m.get('or_high'), m.get('price_vs_or_high')
    return round(orh + pv, 2) if (orh is not None and pv is not None) else None


def ist_hm(ts):
    h = int(ts[11:13]) + 5; mi = int(ts[14:16]) + 30
    if mi >= 60:
        h += 1; mi -= 60
    return h, mi


def load_chain(date):
    p = f'{CHAINS}/{date}/rolling_option.parquet'
    if not os.path.exists(p):
        return None
    df = pd.read_parquet(p)
    # nearest (front) expiry only
    exp = sorted(df.expiryDate.unique())[0]
    df = df[df.expiryDate == exp].copy()
    # index: (tradeTime, strike, optionType) -> ltp
    df['key'] = list(zip(df.tradeTime, df.strikePrice, df.optionType))
    return df.set_index('key')['ltp'].to_dict(), sorted(df.tradeTime.unique())


def ltp(chain, t, strike, otype):
    return chain.get((t, float(strike), otype))


def nearest_time(times, hh, mm):
    target = f"{hh:02d}:{mm:02d}:00"
    le = [t for t in times if t <= target]
    return le[-1] if le else None


def simulate(chain, times, entry_t, dirn, short_k, long_k, otype):
    s0 = ltp(chain, entry_t, short_k, otype)
    l0 = ltp(chain, entry_t, long_k, otype)
    if s0 is None or l0 is None:
        return None
    credit = s0 - l0
    if credit <= 0:
        return None  # not a real credit spread at these strikes
    tp_val = credit * (1 - TP_FRAC)
    sl_val = credit * (1 + SL_MULT)
    future = [t for t in times if t > entry_t and t <= f"{SQUAREOFF}:00"]
    exit_val = credit  # default if no data
    exit_reason = 'SQUAREOFF'
    for t in future:
        s = ltp(chain, t, short_k, otype); l = ltp(chain, t, long_k, otype)
        if s is None or l is None:
            continue
        val = s - l
        if val <= tp_val:
            exit_val = val; exit_reason = 'TP'; break
        if val >= sl_val:
            exit_val = min(val, WIDTH)  # capped by spread width
            exit_reason = 'SL'; break
        exit_val = val
    pnl = (credit - exit_val) * LOT
    return {'credit': credit, 'exit_val': exit_val, 'pnl': pnl,
            'reason': exit_reason}


def round_to_50(x):
    return round(x / 50) * 50


def main():
    conn = sqlite3.connect(DB)
    rows = conn.execute("SELECT session_date, ts, action, payload_json FROM decisions ORDER BY session_date, id").fetchall()
    conn.close()
    sessions = defaultdict(list)
    for sd, ts, action, pj in rows:
        m = json.loads(pj).get('metadata', {})
        s = spot_of(m)
        if s is None:
            continue
        h, mi = ist_hm(ts)
        sessions[sd].append({'ts': ts, 'h': h, 'm_': mi, 'action': action, 'spot': s, 'm': m})

    chain_dates = set(os.listdir(CHAINS))
    overlap = sorted(set(sessions) & chain_dates)
    print(f"Real-premium backtest over {len(overlap)} sessions: {overlap}\n")

    trades = []
    for sd in overlap:
        loaded = load_chain(sd)
        if not loaded:
            continue
        chain, times = loaded
        # take at most ONE trade per day per direction: first qualifying decision
        taken_dir = set()
        for dec in sessions[sd]:
            m = dec['m']; bias = m.get('market_state_bias')
            cw = m.get('current_call_wall'); pw = m.get('current_put_wall')
            if dec['h'] < ENTRY_MIN_HOUR:
                continue
            if bias == 'BEARISH' and cw and 'BEAR' not in taken_dir:
                dirn = 'BEAR'; otype = 'CE'
                short_k = round_to_50(cw); long_k = short_k + WIDTH
            elif bias == 'BULLISH' and pw and 'BULL' not in taken_dir:
                dirn = 'BULL'; otype = 'PE'
                short_k = round_to_50(pw); long_k = short_k - WIDTH
            else:
                continue
            t = nearest_time(times, dec['h']-0, dec['m_'])  # h already IST? no—convert
            # dec h/m_ are IST; chain tradeTime is IST clock too (09:15..15:23)
            t = nearest_time(times, dec['h'], dec['m_'])
            if t is None:
                continue
            sim = simulate(chain, times, t, dirn, short_k, long_k, otype)
            if sim is None:
                continue
            trades.append({'sd': sd, 'dir': dirn, 'hour': dec['h'],
                           'short_k': short_k, 'spot': dec['spot'],
                           'dist': abs(short_k - dec['spot']), **sim})
            taken_dir.add(dirn)

    if not trades:
        print("No priceable spreads constructed."); return

    def summarize(name, ts):
        if not ts:
            print(f"{name}: no trades"); return
        wins = [t for t in ts if t['pnl'] > 0]
        pnl = sum(t['pnl'] for t in ts)
        aw = (sum(t['pnl'] for t in wins)/len(wins)) if wins else 0
        losers = [t for t in ts if t['pnl'] <= 0]
        al = (sum(t['pnl'] for t in losers)/len(losers)) if losers else 0
        print(f"{name}: {len(ts)} trades | win {100*len(wins)/len(ts):.0f}% | "
              f"net ₹{pnl:,.0f} | avg win ₹{aw:,.0f} | avg loss ₹{al:,.0f} | "
              f"expectancy ₹{pnl/len(ts):,.0f}/trade")

    print("=== Real-premium P&L (1 trade/day/direction, width=%d, TP=%d%%, SL=%dx) ==="
          % (WIDTH, int(TP_FRAC*100), int(SL_MULT)))
    summarize("ALL", trades)
    summarize("BEAR_CALL", [t for t in trades if t['dir'] == 'BEAR'])
    summarize("BULL_PUT", [t for t in trades if t['dir'] == 'BULL'])

    print("\n=== By entry hour (IST) ===")
    for hr in range(9, 16):
        sub = [t for t in trades if t['hour'] == hr]
        if sub:
            summarize(f"  {hr}:00", sub)

    print("\n=== By distance-to-short-strike ===")
    for lo, hi in [(0,60),(60,100),(100,160),(160,300)]:
        sub = [t for t in trades if lo <= t['dist'] < hi]
        if sub:
            summarize(f"  {lo}-{hi}pt", sub)

    print("\n=== Per-trade detail ===")
    for t in sorted(trades, key=lambda x: (x['sd'], x['hour'])):
        print(f"  {t['sd']} {t['hour']:02d}h {t['dir']:<4} short={t['short_k']} "
              f"dist={t['dist']:.0f} credit={t['credit']:.1f} "
              f"exit={t['exit_val']:.1f} {t['reason']:<9} P&L ₹{t['pnl']:,.0f}")


if __name__ == '__main__':
    main()
