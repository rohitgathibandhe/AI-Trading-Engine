#!/usr/bin/env python3
"""DAY AUDIT — every session in the history, not hand-picked days. For each day:

  1. the MOVE that was there (largest run-down / run-up after 09:30, day range, day type);
  2. what a DISCIPLINED trader would have taken, live-decidable on CLOSED 5m candles only (owner's rules):
       short = two consecutive closes below the 15-min opening-range low (an ACCEPTED break, not a poke) while below
               VWAP, or a close below the last confirmed swing low while below VWAP with a lower high (M top);
       long  = the mirror. Entries 09:35-14:30, stop beyond the last swing (15-60 pts), breakeven at +1R, then trail
               behind each new confirmed swing, flat at 15:15. Max 2 trades a day, no re-entry on the same side
               after a stop. Points are Nifty points (no option pricing);
  3. what the AGENT did that day (live-path replay outputs: real recorded chains where we have them, otherwise the
     synthetic-chain replay of the same code on the real candles);
  4. the verdict: CAUGHT / LATE / WRONG_SIDE / LOSS / MISSED / OVERTRADE / CORRECT_STAND_ASIDE, and the blockers
     the agent logged on missed days.

  python scripts/day_audit.py --bars /Users/Rohit/trading_research/nifty_5m.json \
      --real /Users/Rohit/trading_research/year/ESB0_REAL.out --synthetic '/Users/Rohit/trading_research/year/SBASE_*.out' \
      --out /Users/Rohit/trading_research/day_audit
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, time
from pathlib import Path

OR_END = time(9, 30)
LAST_ENTRY = time(14, 30)
FLAT = time(15, 15)


def load_bars(p):
    raw = json.load(open(p))
    days = defaultdict(list)
    for x in raw:
        t = datetime.fromisoformat(x["timestamp"])
        if t.time() < FLAT:
            days[t.date()].append({"t": t, "o": x["open"], "h": x["high"], "l": x["low"], "c": x["close"]})
    return {d: v for d, v in days.items() if len(v) >= 60}


def move_profile(T):
    post = [b for b in T if b["t"].time() >= OR_END]
    hi, lo, dn, up = -1e9, 1e9, (0, None, None), (0, None, None)
    hi_t = lo_t = None
    for b in post:
        if b["h"] > hi:
            hi, hi_t = b["h"], b["t"]
        if b["l"] < lo:
            lo, lo_t = b["l"], b["t"]
        if hi - b["l"] > dn[0]:
            dn = (hi - b["l"], hi_t, b["t"])
        if b["h"] - lo > up[0]:
            up = (b["h"] - lo, lo_t, b["t"])
    rng = max(b["h"] for b in T) - min(b["l"] for b in T)
    big = max(dn[0], up[0])
    kind = ("TREND_DOWN" if dn[0] >= up[0] else "TREND_UP") if big >= 0.7 * rng and big >= 120 else \
           ("BIG_TWO_WAY" if big >= 120 else "RANGE")
    return {"range": round(rng), "down": round(dn[0]), "down_from": dn[1], "down_to": dn[2],
            "up": round(up[0]), "up_from": up[1], "up_to": up[2], "type": kind}


def pivots(T, i):
    H = [k for k in range(2, i - 1) if T[k]["h"] >= max(T[k - 2]["h"], T[k - 1]["h"], T[k + 1]["h"], T[k + 2]["h"])]
    L = [k for k in range(2, i - 1) if T[k]["l"] <= min(T[k - 2]["l"], T[k - 1]["l"], T[k + 1]["l"], T[k + 2]["l"])]
    return H, L


def disciplined(T, prev):
    """The owner's playbook, closed candles only. Returns a list of trades."""
    orb = [b for b in T if b["t"].time() < OR_END]
    oh, ol = max(b["h"] for b in orb), min(b["l"] for b in orb)
    vw, pv = [], 0.0
    for k, b in enumerate(T):
        pv += (b["h"] + b["l"] + b["c"]) / 3
        vw.append(pv / (k + 1))
    trades, pos, stopped = [], None, set()
    for i in range(3, len(T)):
        b = T[i]
        H, L = pivots(T, i)
        if pos:
            d = pos["dir"]
            adverse = (b["h"] >= pos["stop"]) if d < 0 else (b["l"] <= pos["stop"])
            if adverse:
                pts = (pos["entry"] - pos["stop"]) if d < 0 else (pos["stop"] - pos["entry"])
                pos.update(exit=b["t"], pts=round(pts, 1), why="STOP")
                if pos["pts"] < 0:
                    stopped.add(d)
                trades.append(pos); pos = None
                continue
            fav = (pos["entry"] - b["l"]) if d < 0 else (b["h"] - pos["entry"])
            if fav >= pos["risk"] and ((pos["stop"] > pos["entry"]) if d < 0 else (pos["stop"] < pos["entry"])):
                pos["stop"] = pos["entry"]                                   # breakeven at +1R
            if d < 0 and H and T[H[-1]]["h"] + 2 < pos["stop"] and H[-1] > pos["i"]:
                pos["stop"] = T[H[-1]]["h"] + 2                              # trail behind each new lower high
            if d > 0 and L and T[L[-1]]["l"] - 2 > pos["stop"] and L[-1] > pos["i"]:
                pos["stop"] = T[L[-1]]["l"] - 2
            if i == len(T) - 1:
                pos.update(exit=b["t"], pts=round((pos["entry"] - b["c"]) if d < 0 else (b["c"] - pos["entry"]), 1), why="CLOSE")
                trades.append(pos); pos = None
            continue
        if len(trades) >= 2 or b["t"].time() < time(9, 35) or b["t"].time() > LAST_ENTRY:
            continue
        c, p1 = b["c"], T[i - 1]["c"]
        for d in (-1, 1):
            if d in stopped:
                continue
            below = c < vw[i] if d < 0 else c > vw[i]
            acc = (c < ol and p1 < ol) if d < 0 else (c > oh and p1 > oh)
            first_acc = acc and not ((T[i - 2]["c"] < ol) if d < 0 else (T[i - 2]["c"] > oh))
            swing = False
            if d < 0 and L and H and len(H) >= 2:
                swing = c < T[L[-1]]["l"] and p1 >= T[L[-1]]["l"] and T[H[-1]]["h"] < T[H[-2]]["h"]
            if d > 0 and H and L and len(L) >= 2:
                swing = c > T[H[-1]]["h"] and p1 <= T[H[-1]]["h"] and T[L[-1]]["l"] > T[L[-2]]["l"]
            if not (below and (first_acc or swing)):
                continue
            ref = (T[H[-1]]["h"] + 2) if (d < 0 and H) else (T[L[-1]]["l"] - 2) if (d > 0 and L) else None
            if ref is None or ((ref - c) if d < 0 else (c - ref)) <= 0:
                ref = c + 30 if d < 0 else c - 30
            risk = min(max(abs(ref - c), 15.0), 60.0)
            pos = {"dir": d, "i": i, "at": b["t"], "entry": c, "risk": risk, "stop": c + risk if d < 0 else c - risk,
                   "trigger": "OR_ACCEPT" if first_acc else "SWING_BREAK"}
            break
    return trades


def agent_days(paths):
    """{date: {'pnl': rupees, 'trades': [(time, structure, pnl)], 'blockers': str, 'src': file}} from replay outputs.
    Days the replay covered but printed nothing for are no-trade days, filled in by the caller."""
    out = {}
    for p in paths:
        m = re.search(r"LIVE-PATH REPLAY (\d{4}-\d\d-\d\d)\.\.(\d{4}-\d\d-\d\d)", open(p).read())
        if m:
            out.setdefault("_windows", []).append((m.group(1), m.group(2)))
        day = None
        pend = None
        for ln in open(p):
            m = re.match(r"(\d{4}-\d\d-\d\d)\s+P&L Rs ([-\d,]+)", ln)
            if m:
                day = m.group(1)
                out[day] = {"pnl": int(m.group(2).replace(",", "")), "trades": [], "blockers": "", "src": Path(p).name}
                continue
            m = re.match(r"\s+ENTRY (\d\d:\d\d) (\w+)", ln)
            if m and day:
                pend = (m.group(1), m.group(2)); continue
            m = re.match(r"\s+EXIT\s+\d\d:\d\d (\w+) Rs ([-\d,]+)", ln)
            if m and day:
                if pend:
                    out[day]["trades"].append((pend[0], pend[1], int(m.group(2).replace(",", "")), m.group(1)))
                pend = None
                continue
            m = re.match(r"\s+blockers: (.*)", ln)
            if m and day:
                out[day]["blockers"] = m.group(1)[:160]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars", required=True)
    ap.add_argument("--real", default="")
    ap.add_argument("--synthetic", default="")
    ap.add_argument("--recorded", default="/Users/Rohit/AI-Trading-Engine/data_engine/market_ai/state/rolling_option_live")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    days = load_bars(a.bars)
    syn = agent_days(sorted(glob.glob(a.synthetic))) if a.synthetic else {}
    real = agent_days([a.real]) if a.real else {}
    recorded = {p.name for p in Path(a.recorded).iterdir()} if Path(a.recorded).exists() else set()
    empty = {"pnl": 0, "trades": [], "blockers": ""}

    def covered(book, ds):
        return any(s0 <= ds <= s1 for s0, s1 in book.get("_windows", []))
    rows = []
    dl = sorted(days)
    for k, d in enumerate(dl):
        if k == 0:
            continue
        T = days[d]
        mp = move_profile(T)
        dt = disciplined(T, days[dl[k - 1]])
        ds = str(d)
        if ds in recorded and covered(real, ds):
            ag, src = real.get(ds, empty), "real"
        elif covered(syn, ds):
            ag, src = syn.get(ds, empty), "synthetic"
        else:
            continue
        ref_pts = sum(t["pts"] for t in dt)
        dom = -1 if mp["down"] >= mp["up"] else 1
        opp = ref_pts >= 40                      # catchable LIVE by the disciplined rules, not just a big hindsight move
        big = max(mp["down"], mp["up"]) >= 150
        at = ag["trades"]
        sides = {(-1 if s.startswith("PUT") or s.startswith("BEAR") else 1) for _, s, _, _ in at}
        first_ref = min((t["at"] for t in dt if t["pts"] > 0), default=None)
        if at:
            agent_first = at[0][0]
            if opp and dom in sides and ag["pnl"] > 0:
                lag = (datetime.strptime(agent_first, "%H:%M") - datetime.strptime(first_ref.strftime("%H:%M"), "%H:%M")).total_seconds() \
                    if first_ref is not None else 0
                late = lag >= 45 * 60
                verdict = "LATE" if late else "CAUGHT"
            elif (opp or big) and dom not in sides:
                verdict = "WRONG_SIDE"
            elif ag["pnl"] <= 0 and not opp:
                verdict = "OVERTRADE"
            else:
                verdict = "LOSS" if ag["pnl"] <= 0 else "CAUGHT_OTHER"
        else:
            verdict = "MISSED" if opp else ("MISSED_BIG_MOVE" if big else "CORRECT_STAND_ASIDE")
        rows.append({"day": ds, "src": src, "recorded_live": ds in recorded, "type": mp["type"], "range": mp["range"],
                     "down": mp["down"], "up": mp["up"],
                     "ref": [{"at": t["at"].strftime("%H:%M"), "dir": "SHORT" if t["dir"] < 0 else "LONG",
                              "trigger": t["trigger"], "pts": t["pts"], "why": t["why"]} for t in dt],
                     "ref_pts": round(ref_pts, 1), "big_move": big, "agent_pnl": ag["pnl"],
                     "agent": [f"{t} {s} {p:+,} {w}" for t, s, p, w in at], "blockers": ag["blockers"], "verdict": verdict})
    json.dump(rows, open(out / "day_audit.json", "w"), indent=1)
    L = []
    P = L.append
    n = len(rows)
    P(f"DAY AUDIT — {n} sessions ({sum(r['src']=='real' for r in rows)} on real recorded chains, "
      f"{sum(r['src']=='synthetic' for r in rows)} synthetic-chain replay of the same code on real candles)")
    P(f"day types: {dict(Counter(r['type'] for r in rows))}")
    v = Counter(r["verdict"] for r in rows)
    P(f"verdicts: {dict(v)}")
    opp = [r for r in rows if r["ref_pts"] >= 40]
    caught = [r for r in opp if r["verdict"] in ("CAUGHT", "LATE")]
    P(f"catchable days (disciplined rules made >= 40 pts live) {len(opp)}: agent caught {len(caught)} "
      f"({len(caught)/max(len(opp),1):.0%}); of those late {sum(r['verdict']=='LATE' for r in opp)}; "
      f"missed {sum(r['verdict']=='MISSED' for r in opp)}, wrong side {sum(r['verdict']=='WRONG_SIDE' for r in opp)}, "
      f"lost {sum(r['verdict']=='LOSS' for r in opp)}")
    bigd = [r for r in rows if r["big_move"]]
    P(f"big-move days (>=150 pts one way) {len(bigd)}: agent profitable on {sum(r['agent_pnl']>0 for r in bigd)}, "
      f"no trade on {sum(not r['agent'] for r in bigd)}, lost on {sum(r['agent_pnl']<0 for r in bigd)}")
    nt = [r for r in rows if not r["agent"]]
    P(f"no-trade days {len(nt)}: disciplined rules on those days {sum(r['ref_pts'] for r in nt):+.0f} pts "
      f"({sum(r['ref_pts']>=40 for r in nt)} catchable, {sum(r['ref_pts']<0 for r in nt)} where they would have lost)")
    P(f"disciplined reference: {sum(r['ref_pts'] for r in rows):+.0f} Nifty pts over {sum(len(r['ref']) for r in rows)} trades; "
      f"win days {sum(r['ref_pts']>0 for r in rows)}, loss days {sum(r['ref_pts']<0 for r in rows)}")
    P(f"agent: Rs {sum(r['agent_pnl'] for r in rows):+,} gross over {sum(len(r['agent']) for r in rows)} trades")
    P("\nby day type:  days | ref pts | agent Rs | caught/missed/wrong/loss")
    for ty in ("TREND_DOWN", "TREND_UP", "BIG_TWO_WAY", "RANGE"):
        g = [r for r in rows if r["type"] == ty]
        c = Counter(r["verdict"] for r in g)
        P(f"  {ty:12s} {len(g):4d} | {sum(r['ref_pts'] for r in g):+7.0f} | {sum(r['agent_pnl'] for r in g):+9,} | "
          f"{c['CAUGHT']+c['LATE']}/{c['MISSED']}/{c['WRONG_SIDE']}/{c['LOSS']}  (stand-aside {c['CORRECT_STAND_ASIDE']}, overtrade {c['OVERTRADE']})")
    P("\nwhy the agent missed (blocker that dominated the day, MISSED days):")
    bc = Counter()
    for r in rows:
        if r["verdict"] in ("MISSED", "MISSED_BIG_MOVE"):
            m = re.findall(r"'([A-Z0-9_]+)[^']*': (\d+)", r["blockers"])
            m = [x for x in m if x[0] not in ("MIN_5M_CANDLES_NOT_READY", "NONE")]
            bc[max(m, key=lambda x: int(x[1]))[0] if m else "SELECTOR_STAND_ASIDE (no blocker logged)"] += 1
    for k, c in bc.most_common():
        P(f"  {c:3d}  {k}")
    P("\nevery catchable or big-move day (newest first):")
    for r in sorted([r for r in rows if r["ref_pts"] >= 40 or r["big_move"]], key=lambda r: r["day"], reverse=True):
        ref = "; ".join(f"{t['at']} {t['dir']} {t['trigger']} {t['pts']:+.0f}" for t in r["ref"]) or "-"
        P(f"  {r['day']} {r['src'][:4]} {r['type']:11s} dn {r['down']:3d} up {r['up']:3d} | ref {ref} | agent "
          f"{'; '.join(r['agent']) or '-'} | {r['verdict']}")
    text = "\n".join(L)
    (out / "report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
