#!/usr/bin/env python3
"""
Gate calibration backtest.

For every historical v83 decision, reconstruct the intraday spot path and label
whether a directional credit spread (placed at the OI wall) would have won or
lost, based on the rest-of-session price action. Then measure which gate scores
separate winners from losers — so thresholds can be set from outcomes, not by feel.

A credit-spread seller wins when the underlying stays AWAY from the short strike:
  - bullish bias  -> bull put spread, short put at the put wall  -> loses if spot
                     falls through (put_wall + buffer)
  - bearish bias  -> bear call spread, short call at the call wall -> loses if spot
                     rises through (call_wall - buffer)

We hold to session close (defined-risk intraday). "Win" = short strike never
breached AND adverse excursion stayed under half the distance-to-strike (spread
held value / decay realized).
"""
from __future__ import annotations
import sqlite3, json
from collections import defaultdict
from statistics import mean, median

DB = '/Users/Rohit/AI-Trading-Engine/data_engine/market_ai/state/intraday_v83_paper_live_learning.sqlite3'
BUFFER = 0.0


def spot_of(m: dict):
    orh, pv = m.get('or_high'), m.get('price_vs_or_high')
    if orh is not None and pv is not None:
        return round(orh + pv, 2)
    return None


def load_sessions():
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT session_date, ts, action, regime, confidence, payload_json "
        "FROM decisions ORDER BY session_date, id"
    ).fetchall()
    conn.close()
    sessions = defaultdict(list)
    for sd, ts, action, regime, conf, pj in rows:
        try:
            m = json.loads(pj).get('metadata', {})
        except Exception:
            continue
        s = spot_of(m)
        if s is None:
            continue
        sessions[sd].append({'ts': ts, 'action': action, 'regime': regime,
                             'conf': conf, 'spot': s, 'm': m})
    return sessions


def label_trade(dec, future_spots):
    m = dec['m']
    bias = m.get('market_state_bias')
    spot = dec['spot']
    call_wall = m.get('current_call_wall')
    put_wall = m.get('current_put_wall')
    if not future_spots:
        return None
    fmax = max(future_spots); fmin = min(future_spots)

    if bias == 'BULLISH' and put_wall:
        direction = 'BULL_PUT'; short_strike = put_wall + BUFFER
        adv = spot - fmin; fav = fmax - spot; breached = fmin <= short_strike
    elif bias == 'BEARISH' and call_wall:
        direction = 'BEAR_CALL'; short_strike = call_wall - BUFFER
        adv = fmax - spot; fav = spot - fmin; breached = fmax >= short_strike
    else:
        return None

    dist = abs(short_strike - spot)
    if dist <= 0:
        return None
    won = (not breached) and (adv <= 0.5 * dist)
    return {'direction': direction, 'won': won, 'fav': fav, 'adv': adv,
            'dist_to_strike': dist, 'breached': breached}


def main():
    sessions = load_sessions()
    print(f"Loaded {len(sessions)} sessions\n")

    labeled = []
    for sd, decs in sessions.items():
        spots = [d['spot'] for d in decs]
        for i, dec in enumerate(decs):
            lab = label_trade(dec, spots[i+1:])
            if lab:
                labeled.append({**dec, **lab, 'session': sd})

    n = len(labeled)
    wins = [d for d in labeled if d['won']]
    losers = [d for d in labeled if not d['won']]
    print("=== Counterfactual opportunities (bias-directional spread at wall) ===")
    print(f"Total evaluable: {n}")
    if n:
        print(f"Would-win: {len(wins)} ({100*len(wins)/n:.1f}%)  "
              f"Would-lose: {len(losers)} ({100*len(losers)/n:.1f}%)")

    def med(decs, field):
        vals = [d['m'].get(field) for d in decs
                if isinstance(d['m'].get(field), (int, float))]
        return round(median(vals), 2) if vals else None

    fields = ['bullish_trade_score', 'bearish_trade_score',
              'bullish_entry_score', 'bearish_entry_score',
              'bullish_confluence_score', 'bullish_support_quality_score',
              'tradability_score', 'distance_to_call_wall',
              'distance_to_put_wall', 'state_confidence']
    print("\n=== Gate scores: WINNER median vs LOSER median ===")
    print(f"{'field':<34}{'win':>9}{'lose':>9}{'sep':>9}")
    for f in fields:
        w, l = med(wins, f), med(losers, f)
        if w is not None and l is not None:
            print(f"{f:<34}{w:>9}{l:>9}{round(w-l,2):>9}")

    print("\n=== By direction ===")
    for dirn in ('BULL_PUT', 'BEAR_CALL'):
        sub = [d for d in labeled if d['direction'] == dirn]
        sw = sum(1 for d in sub if d['won'])
        if sub:
            print(f"  {dirn}: {len(sub)} opps, {sw} wins ({100*sw/len(sub):.1f}%)")

    print("\n=== Sweep bullish_trade_score (BULL_PUT) ===")
    bull = [d for d in labeled if d['direction'] == 'BULL_PUT']
    for thr in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]:
        sel = [d for d in bull if (d['m'].get('bullish_trade_score') or 0) >= thr]
        if sel:
            wr = 100*sum(1 for d in sel if d['won'])/len(sel)
            print(f"  >={thr}: {len(sel):>4} trades, win {wr:.1f}%")

    print("\n=== Sweep bearish_trade_score (BEAR_CALL) ===")
    bear = [d for d in labeled if d['direction'] == 'BEAR_CALL']
    for thr in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5]:
        sel = [d for d in bear if (d['m'].get('bearish_trade_score') or 0) >= thr]
        if sel:
            wr = 100*sum(1 for d in sel if d['won'])/len(sel)
            print(f"  >={thr}: {len(sel):>4} trades, win {wr:.1f}%")

    # Combined gate: trade_score threshold + min wall distance, dedup to 1/day
    print("\n=== Realistic policy: 1 trade/day, score>=thr, wall_dist>=40 ===")
    for thr in [3.0, 3.5, 4.0, 4.5]:
        per_day = {}
        for d in labeled:
            sc = (d['m'].get('bullish_trade_score') if d['direction']=='BULL_PUT'
                  else d['m'].get('bearish_trade_score')) or 0
            wall = (d['m'].get('distance_to_put_wall') if d['direction']=='BULL_PUT'
                    else d['m'].get('distance_to_call_wall')) or 0
            if sc >= thr and wall >= 40:
                if d['session'] not in per_day:  # first qualifying of day
                    per_day[d['session']] = d
        if per_day:
            sel = list(per_day.values())
            wr = 100*sum(1 for d in sel if d['won'])/len(sel)
            print(f"  score>={thr}: {len(sel)} trading days, win {wr:.1f}%")


if __name__ == '__main__':
    main()
