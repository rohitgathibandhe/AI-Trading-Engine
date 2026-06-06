#!/usr/bin/env python3
"""
Post-session trading review — "think like a professional trader".

Run after market close (or for any past session_date). For the day it:
  1. Summarises what the agent did (regimes, trades, dominant blockers).
  2. Grades every would-be setup against the backtested edge
     (bearish + midday 11:00-14:00 + >=80pt from wall = high probability).
  3. Flags MISSED high-probability setups the gates rejected, and
     TAKEN low-probability setups (overtrading) — the two failure modes.
  4. Writes a dated journal entry to state/session_journal/<date>.md

Usage:
  python scripts/session_review.py            # today (IST)
  python scripts/session_review.py 2026-06-04 # specific date
"""
from __future__ import annotations
import sqlite3, json, sys, os
from collections import defaultdict
from datetime import datetime, timezone, timedelta

DB = '/Users/Rohit/AI-Trading-Engine/data_engine/market_ai/state/intraday_v83_paper_live_learning.sqlite3'
JOURNAL_DIR = '/Users/Rohit/AI-Trading-Engine/data_engine/market_ai/state/session_journal'
IST = timezone(timedelta(hours=5, minutes=30))

# Backtest-derived "high probability" context (see docs/gate_calibration_findings.md)
HP_MIN_HOUR = 11          # don't trust pre-11:00 IST
HP_MIN_WALL_DIST = 80     # >=80pt from short strike
HP_PREFERRED_DIR = 'BEARISH'  # bearish edge is 60-80% vs bullish 27-39%


def spot_of(m):
    orh, pv = m.get('or_high'), m.get('price_vs_or_high')
    return round(orh + pv, 2) if (orh is not None and pv is not None) else None


def ist_hm(ts):
    h = int(ts[11:13]) + 5; mi = int(ts[14:16]) + 30
    if mi >= 60:
        h += 1; mi -= 60
    return h, mi


def load_day(date):
    conn = sqlite3.connect(DB)
    rows = conn.execute(
        "SELECT ts, action, regime, confidence, payload_json FROM decisions "
        "WHERE session_date=? ORDER BY id", (date,)).fetchall()
    outs = conn.execute(
        "SELECT playbook, strategy, pnl_rupees, max_drawdown_rupees FROM outcomes "
        "WHERE session_date=?", (date,)).fetchall()
    conn.close()
    decs = []
    for ts, action, regime, conf, pj in rows:
        m = json.loads(pj).get('metadata', {})
        s = spot_of(m)
        if s is None:
            continue
        h, mi = ist_hm(ts)
        decs.append({'ts': ts, 'h': h, 'm_': mi, 'action': action,
                     'regime': regime, 'conf': conf, 'spot': s, 'm': m})
    return decs, outs


def hp_label(dec, future_spots):
    """Would this decision have been a high-probability winning setup?"""
    m = dec['m']; bias = m.get('market_state_bias'); spot = dec['spot']
    cw = m.get('current_call_wall'); pw = m.get('current_put_wall')
    if not future_spots:
        return None
    fmax = max(future_spots); fmin = min(future_spots)
    if bias == 'BULLISH' and pw:
        dirn = 'BULL_PUT'; strike = pw; adv = spot - fmin; dist = spot - strike
        breached = fmin <= strike
    elif bias == 'BEARISH' and cw:
        dirn = 'BEAR_CALL'; strike = cw; adv = fmax - spot; dist = strike - spot
        breached = fmax >= strike
    else:
        return None
    if dist <= 0:
        return None
    won = (not breached) and (adv <= 0.5 * dist)
    high_prob_context = (dec['h'] >= HP_MIN_HOUR and dist >= HP_MIN_WALL_DIST
                         and bias == HP_PREFERRED_DIR)
    return {'dirn': dirn, 'won': won, 'dist': dist, 'hp': high_prob_context,
            'bias': bias}


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now(IST).strftime('%Y-%m-%d')
    decs, outs = load_day(date)
    lines = []

    def out(s=''):
        print(s); lines.append(s)

    out(f"# Session Review — {date}")
    out(f"_Generated {datetime.now(IST).strftime('%Y-%m-%d %H:%M IST')}_\n")

    if not decs:
        out("No decisions recorded for this session.")
        _write(date, lines); return

    trades = [d for d in decs if d['action'] == 'TRADE']
    out(f"## Activity")
    out(f"- Decisions: {len(decs)}  |  Trade signals: {len(trades)}  |  "
        f"Closed outcomes: {len(outs)}")
    if outs:
        pnl = sum(o[2] or 0 for o in outs)
        out(f"- Realized PnL: ₹{pnl:.0f}")
        for o in outs:
            out(f"    - {o[0]} / {o[1]}: PnL ₹{o[2]:.0f}, max drawdown ₹{o[3]:.0f}")

    # Regime time-in-state
    reg_time = defaultdict(int)
    for d in decs:
        reg_time[d['regime']] += 1
    out(f"\n## Regime distribution")
    for r, c in sorted(reg_time.items(), key=lambda x: -x[1]):
        out(f"- {r}: {c} ticks ({100*c/len(decs):.0f}%)")

    # Grade every decision vs the edge
    spots = [d['spot'] for d in decs]
    graded = []
    for i, d in enumerate(decs):
        lab = hp_label(d, spots[i+1:])
        if lab:
            graded.append({**d, **lab})

    hp_setups = [g for g in graded if g['hp']]
    hp_wins = [g for g in hp_setups if g['won']]
    hp_winrate = (len(hp_wins) / len(hp_setups)) if hp_setups else 0.0
    # A setup bucket is only genuinely tradable if it was majority-winning that
    # day. On a trending day the "high-prob" directional filter can still lose
    # (e.g. bearish call-spreads into a rally) — standing aside is then correct.
    bucket_tradable = hp_winrate >= 0.55
    out(f"\n## Edge grading (vs backtested high-probability filter)")
    out(f"High-prob filter: bias=BEARISH, hour>={HP_MIN_HOUR}:00 IST, "
        f">={HP_MIN_WALL_DIST}pt from wall")
    if hp_setups:
        bucket_label = ("TRADABLE" if bucket_tradable else
                        "NOT tradable today (directional filter lost — likely counter-trend)")
        out(f"- High-prob setups available today: {len(hp_setups)} ticks "
            f"({len(set((g['h'],g['m_']) for g in hp_setups))} distinct minutes)")
        out(f"- Of those, would-win: {len(hp_wins)} ({100*hp_winrate:.0f}%) "
            f"→ bucket {bucket_label}")
    else:
        out(f"- No high-probability setups occurred today (correct to stand aside).")

    # Two failure modes
    out(f"\n## Behaviour check")
    # 1. Missed: a genuinely tradable bucket (majority-winning) was rejected.
    missed = [g for g in hp_setups if g['action'] == 'NO_TRADE' and g['won']]
    if bucket_tradable and missed:
        first = missed[0]
        out(f"- ⚠️ MISSED a tradable bucket ({100*hp_winrate:.0f}% win). "
            f"First winner {first['h']:02d}:{first['m_']:02d} IST "
            f"({first['dirn']}, {first['dist']:.0f}pt cushion). "
            f"Investigate which gate blocked it.")
    elif missed:
        out(f"- ✅ Stood aside on the high-prob directional bucket — correct, it "
            f"only won {100*hp_winrate:.0f}% today (counter-trend).")
    else:
        out(f"- ✅ No missed high-probability winning setups.")
    # 2. Overtrading: took a trade in a low-prob context
    bad_trades = [g for g in graded if g['action'] == 'TRADE' and not g['hp']]
    if bad_trades:
        out(f"- ⚠️ Took {len(bad_trades)} trade(s) OUTSIDE the high-prob filter "
            f"(early / close-to-wall / bullish). Review for overtrading.")
    elif trades:
        out(f"- ✅ All trades taken were within the high-prob filter.")
    else:
        out(f"- Stood aside all day. Acceptable IF no high-prob setups (see above).")

    # First high-prob setup detail (for the trade thesis the agent should log)
    if hp_setups:
        g = hp_setups[0]; m = g['m']
        out(f"\n## First high-prob setup detail ({g['h']:02d}:{g['m_']:02d} IST)")
        out(f"- Direction: {g['dirn']}  bias={g['bias']}")
        out(f"- Spot {g['spot']} | short strike "
            f"{m.get('current_call_wall') if g['dirn']=='BEAR_CALL' else m.get('current_put_wall')} "
            f"| cushion {g['dist']:.0f}pt")
        out(f"- Scores: bear_trade={m.get('bearish_trade_score')} "
            f"bull_trade={m.get('bullish_trade_score')} "
            f"tradability={m.get('tradability_class')}")
        out(f"- Gate that blocked (if NO_TRADE): "
            f"{m.get('trade_funnel',{}).get('canonical_rejection_reason')}")

    out(f"\n## Verdict")
    if bucket_tradable and missed:
        out(f"Gates too tight — a majority-winning bucket was rejected. Recalibrate "
            f"the blocking gate (see first missed setup above).")
    elif bad_trades:
        out(f"Gates too loose — trades taken in low-prob contexts. Tighten "
            f"time/wall/direction filters.")
    elif hp_setups and trades:
        out(f"On-edge: took high-prob setups, avoided junk. Track realized PnL.")
    elif not hp_setups:
        out(f"Correct stand-aside day — market offered no high-prob setup.")
    else:
        out(f"Stood aside despite high-prob setups present but none would have won "
            f"— acceptable.")

    _write(date, lines)


def _write(date, lines):
    os.makedirs(JOURNAL_DIR, exist_ok=True)
    path = os.path.join(JOURNAL_DIR, f"{date}.md")
    with open(path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n[journal written: {path}]")


if __name__ == '__main__':
    main()
