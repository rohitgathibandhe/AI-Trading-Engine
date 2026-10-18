#!/usr/bin/env python3
"""Loss autopsy — for EVERY losing trade, what was the market doing, what did the agent do, why did it lose, and
what should it have done. Then rank the mistakes by rupees so the biggest recurring flaw is fixed first.

  python scripts/loss_autopsy.py [--ledger <trades.jsonl>] [--since 2026-07-01] [--json out.json]

Owner (2026-10-18): "for any day the trade went into loss, analyze in detail what went wrong with the logic and
what the agent should have done ... if a professional can make money in that market, why can't the agent".

Everything is read from completed 5m candles at the trade's own timestamps (no hindsight in the CONTEXT columns);
the OUTCOME columns (what happened next, hold-to-close) are hindsight by design — that is the lesson.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, time, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "data_engine"))
import agent_skill_report as asr  # noqa: E402  (5m candle loader with cache)

LEDGER = ROOT / "data_engine" / "market_ai" / "state" / "intraday_v83_paper_live_trades.jsonl"
DIR = {"CALL_DEBIT_SPREAD": 1, "BULL_PUT_CREDIT_SPREAD": 1, "PUT_DEBIT_SPREAD": -1, "BEAR_CALL_CREDIT_SPREAD": -1}
DEBIT = {"CALL_DEBIT_SPREAD", "PUT_DEBIT_SPREAD"}

FIX = {
    "TWO_SIDED_RANGE_DIRECTIONAL": "Range day (opening range broken BOTH ways before entry): no directional bet — sell premium "
                                   "beyond the walls or stand aside. (Guarded since PR #41.)",
    "CHASED_EXTENSION": "Entered at the edge of the day's range after the move had already run — wait for a pullback to "
                        "VWAP/EMA or the retest of the broken level; never buy the high / sell the low.",
    "AGAINST_SESSION": "Traded against the session's own structure (VWAP side + opening-range break). Only trade with the "
                       "session, or fade only at a defended level with a tight stop.",
    "EXIT_TOO_EARLY": "Thesis was right: the stop/exit fired on noise and the market then went the trade's way. The exit "
                      "must key on structure (VWAP/swing break), not a quick premium or patience rule.",
    "GAVE_BACK_WINNER": "Was solidly in profit and let it turn into a loss — bank part at the measured move and trail the "
                        "rest behind structure.",
    "HELD_THROUGH_REVERSAL": "The reason for the trade broke (price closed back through VWAP against it) long before the "
                             "exit; cut there, small, instead of waiting for the premium stop.",
    "RIGHT_DIRECTION_WRONG_STRUCTURE": "Direction was right but the structure lost (credit capped / strike buffer / decay "
                                       "or a debit bought too far OTM) — the structure did not fit the move.",
    "WRONG_READ": "Direction read was wrong with no single flaw — check the context columns (trend state, walls, day "
                  "shape) for what the read missed.",
}


def _vwap(bars):
    tp = [(b["h"] + b["l"] + b["c"]) / 3 for b in bars]
    return sum(tp) / len(tp) if tp else None


def _ts(s):
    return datetime.fromisoformat(str(s)).replace(tzinfo=None)


def _shape(day_bars):
    segs = [(time(9, 15), time(11, 0)), (time(11, 0), time(13, 0)), (time(13, 0), time(15, 15))]
    rng = max(b["h"] for b in day_bars) - min(b["l"] for b in day_bars)
    thr = max(0.15 * rng, 40)
    out = ""
    for a, z in segs:
        s = [b for b in day_bars if a <= b["t"].time() < z]
        if len(s) < 3:
            out += "?"
            continue
        mv = s[-1]["c"] - s[0]["o"]
        r = max(b["h"] for b in s) - min(b["l"] for b in s)
        out += ("U" if mv > 0 else "D") if abs(mv) >= thr and abs(mv) / max(r, 1) >= 0.5 else "S"
    return out


def autopsy(entry: dict, exit_: dict) -> dict | None:
    st = entry.get("strategy")
    d = DIR.get(st)
    if d is None:
        return None
    t_in, t_out = _ts(entry["timestamp"]), _ts(exit_.get("exit_timestamp") or exit_["timestamp"])
    day = t_in.date().isoformat()
    bars = [b for b in asr._candles(day) if b["t"].time() < time(15, 30)]
    if len(bars) < 20:
        return None
    closed = [b for b in bars if b["t"] + timedelta(minutes=5) <= t_in]
    if len(closed) < 3:
        return None
    spot = float(entry.get("spot_at_entry") or closed[-1]["c"])
    orb = [b for b in bars if b["t"].time() < time(9, 30)]
    oh, ol = max(b["h"] for b in orb), min(b["l"] for b in orb)
    post_or = [b for b in closed if b["t"].time() >= time(9, 30)]
    broke_up = any(b["c"] > oh for b in post_or)
    broke_dn = any(b["c"] < ol for b in post_or)
    vw = _vwap(closed)
    hi, lo = max(b["h"] for b in closed), min(b["l"] for b in closed)
    pos = (spot - lo) / max(hi - lo, 1)                       # 0 = at the day low so far, 1 = at the high
    session = (1 if (spot > vw and broke_up and not broke_dn) else -1 if (spot < vw and broke_dn and not broke_up) else 0)
    held = [b for b in bars if t_in <= b["t"] < t_out]
    after = [b for b in bars if b["t"] >= t_in and b["t"].time() < time(15, 15)]
    post_exit = [b for b in bars if b["t"] >= t_out and b["t"].time() < time(15, 15)]
    if not after:
        return None
    fav = max(((b["h"] - spot) if d > 0 else (spot - b["l"])) for b in (held or after[:1]))
    adv = max(((spot - b["l"]) if d > 0 else (b["h"] - spot)) for b in (held or after[:1]))
    exit_spot = (held[-1]["c"] if held else spot)
    close = after[-1]["c"]
    hold_close_pts = (close - spot) * d
    at_exit_pts = (exit_spot - spot) * d
    best_after_exit = max([((b["h"] - spot) if d > 0 else (spot - b["l"])) for b in post_exit] or [at_exit_pts])
    # first completed bar after entry that closed through VWAP AGAINST the trade (thesis break)
    thesis_break = None
    for i, b in enumerate(held):
        vw_i = _vwap([x for x in bars if x["t"] <= b["t"]])
        if ((b["c"] < vw_i - 10) if d > 0 else (b["c"] > vw_i + 10)) and b["t"] - t_in >= timedelta(minutes=10):
            thesis_break = b["t"]
            break
    pnl = float(exit_.get("realized_paper_pnl") or 0)
    mfe_rs = float(exit_.get("mfe_rupees") or 0)
    # root cause, first match (ordered from the most actionable)
    if broke_up and broke_dn and st in DEBIT:
        cause = "TWO_SIDED_RANGE_DIRECTIONAL"
    elif at_exit_pts > 5:
        cause = "RIGHT_DIRECTION_WRONG_STRUCTURE"
    elif mfe_rs >= 1000 or fav >= 45:
        cause = "GAVE_BACK_WINNER"
    elif best_after_exit >= 30 and hold_close_pts > 0:
        cause = "EXIT_TOO_EARLY"
    elif session == -d:
        cause = "AGAINST_SESSION"
    elif (d > 0 and pos >= 0.85) or (d < 0 and pos <= 0.15):
        cause = "CHASED_EXTENSION"
    elif thesis_break is not None and (t_out - thesis_break) >= timedelta(minutes=30):
        cause = "HELD_THROUGH_REVERSAL"
    else:
        cause = "WRONG_READ"
    return {
        "day": day, "shape": _shape(bars), "entry": t_in.strftime("%H:%M"), "exit": t_out.strftime("%H:%M"),
        "strategy": st, "playbook": entry.get("playbook"), "dir": "LONG" if d > 0 else "SHORT", "pnl": round(pnl),
        "exit_reason": exit_.get("exit_reason"), "spot": round(spot), "vs_vwap": round(spot - vw),
        "pos_in_range": round(pos, 2), "or_broken": ("BOTH" if broke_up and broke_dn else "UP" if broke_up else "DOWN" if broke_dn else "NONE"),
        "session": {1: "UP", -1: "DOWN", 0: "MIXED"}[session], "fav_pts": round(fav), "adv_pts": round(adv),
        "at_exit_pts": round(at_exit_pts), "hold_to_close_pts": round(hold_close_pts),
        "best_after_exit_pts": round(best_after_exit),
        "thesis_break": thesis_break.strftime("%H:%M") if thesis_break else None, "cause": cause, "fix": FIX[cause],
    }


def capture(entry: dict, exit_: dict) -> dict | None:
    """For a WINNING trade: how far did the market move our way after entry, and how much did we take?"""
    st = entry.get("strategy")
    d = DIR.get(st)
    if d is None:
        return None
    t_in, t_out = _ts(entry["timestamp"]), _ts(exit_.get("exit_timestamp") or exit_["timestamp"])
    bars = [b for b in asr._candles(t_in.date().isoformat()) if b["t"].time() < time(15, 15)]
    after = [b for b in bars if b["t"] >= t_in]
    held = [b for b in after if b["t"] < t_out]
    if len(bars) < 20 or not after:
        return None
    spot = float(entry.get("spot_at_entry") or after[0]["o"])
    avail = max(((b["h"] - spot) if d > 0 else (spot - b["l"])) for b in after)      # best move our way, entry -> 15:15
    t_best = max(after, key=lambda b: (b["h"] - spot) if d > 0 else (spot - b["l"]))["t"].strftime("%H:%M")
    exit_spot = held[-1]["c"] if held else spot
    took = (exit_spot - spot) * d
    o, c = bars[0]["o"], bars[-1]["c"]
    day_move = c - o
    pnl = float(exit_.get("realized_paper_pnl") or 0)
    lots_qty = sum(abs(float(l.get("quantity") or 0)) for l in (entry.get("legs") or [])[:1]) or 130
    # what a with-direction ~0.5-delta position would have earned over the available move (rupees per point)
    return {"day": t_in.date().isoformat(), "entry": t_in.strftime("%H:%M"), "exit": t_out.strftime("%H:%M"),
            "dir": "LONG" if d > 0 else "SHORT", "strategy": st, "exit_reason": exit_.get("exit_reason"), "pnl": round(pnl),
            "day_move": round(day_move), "available_pts": round(avail), "best_at": t_best, "captured_pts": round(took),
            "capture_pct": round(100 * took / avail) if avail > 0 else 0,
            "rs_per_pt": round(pnl / took, 1) if took > 0 else None, "qty": lots_qty}


def main_winners(a) -> int:
    rows = [json.loads(x) for x in open(a.ledger) if x.strip().startswith("{")]
    entries = {str(t.get("entry_timestamp") or t["timestamp"]): t for t in rows if t.get("event") == "PAPER_ENTRY"}
    out = []
    for t in rows:
        if t.get("event") != "PAPER_EXIT" or float(t.get("realized_paper_pnl") or 0) <= 0:
            continue
        if str(t.get("timestamp", ""))[:10] < a.since:
            continue
        e = entries.get(str(t.get("entry_timestamp")))
        r = capture(e, t) if e else None
        if r:
            out.append(r)
    print(f"WINNER CAPTURE — {len(out)} winning trades, total Rs {sum(r['pnl'] for r in out):,}\n")
    print(f"{'day':10s} {'in':5s} {'out':5s} {'dir':5s} {'structure':24s} {'exit':22s} {'Rs':>7s} {'day mv':>6s} "
          f"{'avail':>5s} {'@':5s} {'took':>5s} {'cap%':>5s} {'Rs/pt':>6s}")
    for r in sorted(out, key=lambda x: x["day"] + x["entry"]):
        print(f"{r['day']} {r['entry']} {r['exit']} {r['dir']:5s} {r['strategy']:24s} {str(r['exit_reason'])[:22]:22s} "
              f"{r['pnl']:>7,} {r['day_move']:>+6d} {r['available_pts']:>5d} {r['best_at']} {r['captured_pts']:>5d} "
              f"{r['capture_pct']:>4d}% {str(r['rs_per_pt'] or '-'):>6s}")
    av = sum(r["available_pts"] for r in out); tk = sum(max(r["captured_pts"], 0) for r in out)
    print(f"\nTOTAL available {av} pts, captured {tk} pts = {100*tk/max(av,1):.0f}% of the move")
    by = defaultdict(lambda: [0, 0, 0, 0])
    for r in out:
        k = by[r["strategy"]]; k[0] += 1; k[1] += r["available_pts"]; k[2] += max(r["captured_pts"], 0); k[3] += r["pnl"]
    for k, v in by.items():
        print(f"  {k:24s} n={v[0]:3d} avail {v[1]:5d} pts captured {v[2]:5d} ({100*v[2]/max(v[1],1):.0f}%)  Rs {v[3]:,}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=str(LEDGER))
    ap.add_argument("--since", default="2026-07-01")
    ap.add_argument("--json", default="")
    ap.add_argument("--winners", action="store_true", help="capture analysis of WINNING trades instead")
    a = ap.parse_args()
    if a.winners:
        return main_winners(a)
    rows = [json.loads(x) for x in open(a.ledger) if x.strip().startswith("{")]
    entries = {}
    out = []
    for t in rows:
        if t.get("event") == "PAPER_ENTRY":
            entries[str(t.get("entry_timestamp") or t["timestamp"])] = t
    for t in rows:
        if t.get("event") != "PAPER_EXIT" or float(t.get("realized_paper_pnl") or 0) >= 0:
            continue
        if str(t.get("timestamp", ""))[:10] < a.since:
            continue
        e = entries.get(str(t.get("entry_timestamp")))
        if not e:
            continue
        try:
            r = autopsy(e, t)
        except Exception as exc:  # noqa: BLE001 — one bad day must not hide the rest
            r = None
            print(f"[skip] {t.get('timestamp')}: {exc}", file=sys.stderr)
        if r:
            out.append(r)
    total = sum(r["pnl"] for r in out)
    print(f"LOSS AUTOPSY — {len(out)} losing trades, total Rs {total:,}\n")
    for r in sorted(out, key=lambda x: x["day"] + x["entry"]):
        print(f"{r['day']} [{r['shape']}] {r['entry']}->{r['exit']} {r['dir']:5s} {r['strategy']:24s} {r['playbook'] or '':20s} "
              f"Rs {r['pnl']:>7,}  exit {r['exit_reason']}")
        print(f"    context: spot {r['spot']} ({r['vs_vwap']:+d} vs VWAP), {int(r['pos_in_range']*100)}% of day range, "
              f"OR broken {r['or_broken']}, session {r['session']}")
        print(f"    outcome: best {r['fav_pts']:+d} / worst -{r['adv_pts']} pts while held, {r['at_exit_pts']:+d} at exit, "
              f"{r['hold_to_close_pts']:+d} if held to close, best after exit {r['best_after_exit_pts']:+d}"
              + (f", thesis broke {r['thesis_break']}" if r['thesis_break'] else ""))
        print(f"    CAUSE {r['cause']}: {r['fix']}\n")
    by = defaultdict(list)
    for r in out:
        by[r["cause"]].append(r["pnl"])
    print("RANKED MISTAKES (by rupees lost)")
    for k, v in sorted(by.items(), key=lambda kv: sum(kv[1])):
        print(f"  {k:32s} {len(v):3d} trades  Rs {sum(v):>9,}")
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
