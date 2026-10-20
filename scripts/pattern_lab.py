#!/usr/bin/env python3
"""PATTERN LAB — read every 5m chart pattern in the history, record its context (incl. higher-timeframe trend) at the
moment it completes, measure what happened next with a structure stop, and learn which pattern+context combinations
win often and win BIG. Walk-forward: learn on the first 2/3 of days, judge on the last 1/3 (owner, 2026-10-20).

Patterns (all on COMPLETED 5m bars, live-decidable): SWING_BREAK, OR_BREAK, OR_FAIL, DOUBLE_TOP/BOTTOM (M/W),
VWAP_REJECT/RECLAIM, EMA_PULLBACK. Context: daily / 60m / 15m trend vs trade direction, VWAP & 5m EMA20 side, trigger
candle strength, two-sided day, stretch from VWAP and room to the day extreme (in ATR), time of day, gap.
Outcome: entry = trigger close; stop = the pattern's invalidation level (clamped 10..80 pts); first touch of the stop
vs 1R/2R/3R targets until 15:15 (stop assumed first when both touch in one bar).

  python scripts/pattern_lab.py --data /Users/Rohit/trading_research/nifty_5m.json --out /Users/Rohit/trading_research/pattern_lab
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, time
from pathlib import Path

import numpy as np


def load(p):
    out = []
    for b in json.load(open(p)):
        t = b["timestamp"]
        t = datetime.fromisoformat(str(t).replace("Z", "")) if not isinstance(t, (int, float)) else datetime.fromtimestamp(t)
        t = t.replace(tzinfo=None)
        if time(9, 15) <= t.time() < time(15, 30):
            out.append({"t": t, "o": float(b["open"]), "h": float(b["high"]), "l": float(b["low"]), "c": float(b["close"])})
    return sorted(out, key=lambda x: x["t"])


def ema_series(vals, n):
    k, e, out = 2 / (n + 1), None, []
    for v in vals:
        e = v if e is None else e + k * (v - e)
        out.append(e)
    return out


def resample(bars, mins):
    g = defaultdict(list)
    for b in bars:
        m = (b["t"].hour * 60 + b["t"].minute - 555) // mins
        g[(b["t"].date(), m)].append(b)
    return [{"t": v[0]["t"], "end": v[-1]["t"], "c": v[-1]["c"], "h": max(x["h"] for x in v), "l": min(x["l"] for x in v)}
            for _, v in sorted(g.items())]


def sgn(x):
    return (x > 0) - (x < 0)


def build_events(bars):
    days = defaultdict(list)
    for b in bars:
        days[b["t"].date()].append(b)
    dlist = sorted(d for d in days if len(days[d]) >= 60)
    daily = [{"d": d, "h": max(b["h"] for b in days[d]), "l": min(b["l"] for b in days[d]), "c": days[d][-1]["c"]} for d in dlist]
    dema = ema_series([x["c"] for x in daily], 20)
    allb = [b for d in dlist for b in days[d]]
    e5 = ema_series([b["c"] for b in allb], 20)
    idx5 = {id(b): i for i, b in enumerate(allb)}
    m15 = resample(allb, 15); e15 = ema_series([x["c"] for x in m15], 20)
    m60 = resample(allb, 60); e60 = ema_series([x["c"] for x in m60], 20)
    end15 = [x["end"] for x in m15]; end60 = [x["end"] for x in m60]
    import bisect
    events = []
    for di, d in enumerate(dlist):
        if di < 21:
            continue
        T = [b for b in days[d] if b["t"].time() < time(15, 15)]
        pd_ = daily[di - 1]
        dtrend = sgn(pd_["c"] - dema[di - 1]) + sgn(dema[di - 1] - dema[di - 6])
        orb = [b for b in T if b["t"].time() < time(9, 30)]
        oh, ol = max(b["h"] for b in orb), min(b["l"] for b in orb)
        gap = T[0]["o"] - pd_["c"]
        pv = 0.0
        vw = []
        for k, b in enumerate(T):
            pv += (b["h"] + b["l"] + b["c"]) / 3
            vw.append(pv / (k + 1))
        broke_up = broke_dn = False
        or_break_dir = None
        or_break_k = None
        fired = defaultdict(int)
        for i in range(6, len(T)):
            b = T[i]
            if b["t"].time() > time(14, 30):
                break
            gi = idx5[id(b)]
            atr = sum(x["h"] - x["l"] for x in allb[max(0, gi - 14):gi + 1]) / min(15, gi + 1)
            # higher-timeframe bars COMPLETED by this bar's close
            j15 = bisect.bisect_right(end15, b["t"]) - 1
            j60 = bisect.bisect_right(end60, b["t"]) - 1
            if j15 < 4 or j60 < 4:
                continue
            t15 = sgn(e15[j15] - e15[j15 - 3])
            t60 = sgn(e60[j60] - e60[j60 - 3]) + sgn(m60[j60]["c"] - e60[j60])
            c, vv, ee = b["c"], vw[i], e5[gi]
            if b["t"].time() >= time(9, 30):
                if c > oh: broke_up = True
                if c < ol: broke_dn = True
            two = broke_up and broke_dn
            hi_so, lo_so = max(x["h"] for x in T[:i + 1]), min(x["l"] for x in T[:i + 1])
            rng = max(b["h"] - b["l"], 0.5)
            body = (b["c"] - b["o"]) / rng
            pivH = [k for k in range(2, i - 1) if T[k]["h"] >= max(T[k - 2]["h"], T[k - 1]["h"], T[k + 1]["h"], T[k + 2]["h"])]
            pivL = [k for k in range(2, i - 1) if T[k]["l"] <= min(T[k - 2]["l"], T[k - 1]["l"], T[k + 1]["l"], T[k + 2]["l"])]
            cands = []   # (pattern, dir, stop)
            # SWING BREAK
            if pivL:
                L = T[pivL[-1]]["l"]
                if c < L and T[i - 1]["c"] >= L:
                    st = max([T[k]["h"] for k in pivH if k > pivL[-1]] or [max(x["h"] for x in T[i - 3:i + 1])])
                    cands.append(("SWING_BREAK", -1, st))
            if pivH:
                H = T[pivH[-1]]["h"]
                if c > H and T[i - 1]["c"] <= H:
                    st = min([T[k]["l"] for k in pivL if k > pivH[-1]] or [min(x["l"] for x in T[i - 3:i + 1])])
                    cands.append(("SWING_BREAK", 1, st))
            # OPENING RANGE break / failure
            if b["t"].time() >= time(9, 30):
                if or_break_dir is None and (c > oh or c < ol):
                    or_break_dir, or_break_k = (1 if c > oh else -1), i
                    cands.append(("OR_BREAK", or_break_dir, (oh + ol) / 2))
                elif or_break_dir is not None and or_break_k is not None and i - or_break_k <= 3 and fired["OR_FAIL"] == 0:
                    if (or_break_dir > 0 and c < oh) or (or_break_dir < 0 and c > ol):
                        ext = max(x["h"] for x in T[or_break_k:i + 1]) if or_break_dir > 0 else min(x["l"] for x in T[or_break_k:i + 1])
                        cands.append(("OR_FAIL", -or_break_dir, ext))
            # M / W
            if len(pivH) >= 2:
                a, b2 = pivH[-2], pivH[-1]
                if b2 - a >= 4 and abs(T[a]["h"] - T[b2]["h"]) <= 0.0015 * c:
                    neck = min(x["l"] for x in T[a:b2 + 1])
                    if c < neck and T[i - 1]["c"] >= neck and i > b2:
                        cands.append(("DOUBLE_TOP", -1, max(T[a]["h"], T[b2]["h"])))
            if len(pivL) >= 2:
                a, b2 = pivL[-2], pivL[-1]
                if b2 - a >= 4 and abs(T[a]["l"] - T[b2]["l"]) <= 0.0015 * c:
                    neck = max(x["h"] for x in T[a:b2 + 1])
                    if c > neck and T[i - 1]["c"] <= neck and i > b2:
                        cands.append(("DOUBLE_BOTTOM", 1, min(T[a]["l"], T[b2]["l"])))
            # VWAP rejection / reclaim with a strong candle
            if b["h"] >= vv and c < vv - 0.2 * atr and body <= -0.5 and all(T[k]["c"] < vw[k] for k in range(i - 3, i)):
                cands.append(("VWAP_REJECT", -1, b["h"]))
            if b["l"] <= vv and c > vv + 0.2 * atr and body >= 0.5 and all(T[k]["c"] > vw[k] for k in range(i - 3, i)):
                cands.append(("VWAP_REJECT", 1, b["l"]))
            # 20 EMA pullback continuation
            if e5[gi] < e5[gi - 3] and b["h"] >= ee and c < ee and body <= -0.3:
                cands.append(("EMA_PULLBACK", -1, b["h"]))
            if e5[gi] > e5[gi - 3] and b["l"] <= ee and c > ee and body >= 0.3:
                cands.append(("EMA_PULLBACK", 1, b["l"]))
            for pat, dr, stop in cands:
                key = (pat, dr)
                if fired[key] >= 2:
                    continue
                fired[key] += 1
                if pat == "OR_FAIL":
                    fired["OR_FAIL"] += 1
                risk = min(max(abs(c - stop), 10.0), 80.0)
                stop_px = c + risk if dr < 0 else c - risk
                res, mfe = None, 0.0
                hits = {1: None, 2: None, 3: None}
                for x in T[i + 1:]:
                    adv = (x["h"] - c) if dr < 0 else (c - x["l"])
                    fav = (c - x["l"]) if dr < 0 else (x["h"] - c)
                    if adv >= risk:
                        res = res or "STOP"
                        break
                    mfe = max(mfe, fav)
                    for r in (1, 2, 3):
                        if hits[r] is None and fav >= r * risk:
                            hits[r] = x["t"]
                end_r = (((T[-1]["c"] - c) * dr) / risk) if res != "STOP" else -1.0
                hr = lambda h: h.hour + h.minute / 60
                events.append({
                    "day": str(d), "t": b["t"].strftime("%H:%M"), "pattern": pat, "dir": dr, "risk": round(risk, 1),
                    "won1": hits[1] is not None, "won2": hits[2] is not None, "won3": hits[3] is not None,
                    "stopped": res == "STOP", "end_r": round(end_r, 2), "mfe_r": round(mfe / risk, 2),
                    "f_daily": dtrend * dr, "f_60m": t60 * dr, "f_15m": t15 * dr, "f_vwap": sgn(c - vv) * dr,
                    "f_ema": sgn(c - ee) * dr, "f_body": round(body * dr, 2), "f_two": int(two),
                    "f_stretch": round((c - vv) * dr / max(atr, 1), 2), "f_room": round(((c - lo_so) if dr < 0 else (hi_so - c)) / max(atr, 1), 2),
                    "f_hour": hr(b["t"]), "f_gap": sgn(gap) * dr, "f_risk_atr": round(risk / max(atr, 1), 2),
                })
    return events


# ---------- learning (numpy logistic regression on one-hot buckets) ----------
def bucket(e):
    f = []
    f.append("pat=" + e["pattern"] + ("_UP" if e["dir"] > 0 else "_DN"))
    f.append(f"daily={e['f_daily']:+d}"); f.append(f"h60={e['f_60m']:+d}"); f.append(f"m15={e['f_15m']:+d}")
    f.append(f"vwap={e['f_vwap']:+d}"); f.append(f"ema={e['f_ema']:+d}"); f.append(f"two={e['f_two']}"); f.append(f"gap={e['f_gap']:+d}")
    f.append("body=" + ("strong" if e["f_body"] >= 0.5 else "weak" if e["f_body"] >= 0 else "against"))
    f.append("stretch=" + ("far" if e["f_stretch"] > 1.5 else "near" if e["f_stretch"] > 0 else "behind"))
    f.append("room=" + ("big" if e["f_room"] > 3 else "mid" if e["f_room"] > 1.5 else "small"))
    f.append("time=" + ("am" if e["f_hour"] < 11 else "mid" if e["f_hour"] < 13 else "pm"))
    f.append("risk=" + ("tight" if e["f_risk_atr"] < 1 else "normal" if e["f_risk_atr"] < 2 else "wide"))
    return f


def fit_logit(X, y, l2=1.0, it=400, lr=0.5):
    w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w))
        g = X.T @ (p - y) / len(y) + l2 * w / len(y)
        w -= lr * g
    return w


def ev_of(g, target=2):
    """Expected R per trade with a fixed target of `target`R and a 1R stop; open trades marked at the 15:15 close."""
    tot = 0.0
    for e in g:
        if e[f"won{target}"]:
            tot += target
        elif e["stopped"]:
            tot -= 1
        else:
            tot += max(min(e["end_r"], target), -1)
    return tot / max(len(g), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    ev = build_events(load(a.data))
    json.dump(ev, open(out / "events.json", "w"))
    days = sorted({e["day"] for e in ev}); cut = days[int(len(days) * 2 / 3)]
    tr = [e for e in ev if e["day"] < cut]; te = [e for e in ev if e["day"] >= cut]
    L = []
    P = L.append
    P(f"PATTERN LAB — {len(ev)} pattern events on {len(days)} days; learn on {len(tr)} (to {cut}), test on {len(te)}")
    P(f"baseline: P(2R before stop) train {np.mean([e['won2'] for e in tr]):.0%} test {np.mean([e['won2'] for e in te]):.0%} | "
      f"EV(2R target) train {ev_of(tr):+.2f}R test {ev_of(te):+.2f}R")
    P("\nBY PATTERN (test period)  n  P(1R) P(2R) P(3R)  EV@2R  EV@3R  avg MFE")
    for pat in sorted({(e['pattern'], e['dir']) for e in ev}):
        g = [e for e in te if (e['pattern'], e['dir']) == pat]
        g2 = [e for e in tr if (e['pattern'], e['dir']) == pat]
        if len(g) < 20:
            continue
        P(f"  {pat[0]+('_UP' if pat[1]>0 else '_DN'):18s} {len(g):4d}  {np.mean([e['won1'] for e in g]):.0%}  {np.mean([e['won2'] for e in g]):.0%}  "
          f"{np.mean([e['won3'] for e in g]):.0%}  {ev_of(g):+.2f}  {ev_of(g,3):+.2f}  {np.mean([e['mfe_r'] for e in g]):.1f}R   (train EV@2R {ev_of(g2):+.2f})")
    # context effects: does each context bucket raise P(2R) on BOTH periods?
    P("\nCONTEXT that moves the odds (P(2R) train -> test, EV@2R test)")
    keys = defaultdict(lambda: ([], []))
    for e in tr:
        for f in bucket(e)[1:]:
            keys[f][0].append(e)
    for e in te:
        for f in bucket(e)[1:]:
            keys[f][1].append(e)
    rows = []
    for f, (a1, b1) in keys.items():
        if len(a1) >= 80 and len(b1) >= 40:
            rows.append((np.mean([e['won2'] for e in b1]) - np.mean([e['won2'] for e in te]), f, a1, b1))
    for d, f, a1, b1 in sorted(rows, reverse=True):
        P(f"  {f:16s} train {np.mean([e['won2'] for e in a1]):.0%} n={len(a1):5d} | test {np.mean([e['won2'] for e in b1]):.0%} n={len(b1):5d}  EV {ev_of(b1):+.2f}R")
    # model: learned probability on the TEST period, by predicted bucket
    vocab = sorted({f for e in ev for f in bucket(e)})
    ix = {f: i for i, f in enumerate(vocab)}
    def X(es):
        m = np.zeros((len(es), len(vocab) + 1)); m[:, -1] = 1
        for r, e in enumerate(es):
            for f in bucket(e):
                m[r, ix[f]] = 1
        return m
    w = fit_logit(X(tr), np.array([e['won2'] for e in tr], float))
    pt = 1 / (1 + np.exp(-X(te) @ w))
    order = np.argsort(pt)
    P("\nLEARNED PROBABILITY on unseen test days (sorted into fifths by predicted P(2R))")
    for q in range(5):
        sel = order[int(q * len(te) / 5):int((q + 1) * len(te) / 5)]
        g = [te[i] for i in sel]
        P(f"  fifth {q+1}: predicted {pt[sel].mean():.0%}  actual P(2R) {np.mean([e['won2'] for e in g]):.0%}  EV@2R {ev_of(g):+.2f}R  EV@3R {ev_of(g,3):+.2f}R  n={len(g)}")
    top = sorted(zip(vocab, w[:-1]), key=lambda kv: -abs(kv[1]))[:14]
    P("\nstrongest learned drivers: " + ", ".join(f"{k} {v:+.2f}" for k, v in top))
    json.dump({"vocab": vocab, "w": w.tolist(), "cut": cut}, open(out / "model.json", "w"))
    text = "\n".join(L)
    (out / "report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
