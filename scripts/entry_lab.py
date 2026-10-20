#!/usr/bin/env python3
"""ENTRY LAB — every completed 5m candle in the history is a candidate entry, LONG and SHORT, with the stop just
beyond the last confirmed swing (intraday_defined_risk/entry_score.features — the SAME function the live gate uses).
Label: did it run 3R before the stop (by 15:15)? Learn what the market looked like at the candles that did,
walk-forward (first 2/3 of days learn, last 1/3 judge). Owner, 2026-10-20: "learn every historical 5m chart; the
optimal entry point with a small SL; on some days the first 5-min candle's H/L matter, on others the first 15-min,
depending on volatility; understand the swings forming."

  python scripts/entry_lab.py --data /Users/Rohit/trading_research/nifty_5m.json --out /Users/Rohit/trading_research/entry_lab
  (--ship writes the model to intraday_defined_risk/entry_score_model.json, trained on ALL days)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import time, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_engine"))
import pattern_lab as pl                                      # noqa: E402  loader + logistic fit
from market_ai.intraday_defined_risk import entry_score as es  # noqa: E402


def rows_for(bars):
    days = defaultdict(list)
    for b in bars:
        days[b["t"].date()].append(b)
    dlist = sorted(d for d in days if len(days[d]) >= 60)
    daily = [{"t": d.isoformat(), "o": days[d][0]["o"], "h": max(b["h"] for b in days[d]),
              "l": min(b["l"] for b in days[d]), "c": days[d][-1]["c"]} for d in dlist]
    out = []
    for di, d in enumerate(dlist):
        if di < 26:
            continue
        prior = [x for x in daily[:di] if x["t"] >= (d - timedelta(days=90)).isoformat()]   # what live fetches
        ctx = es.daily_context(prior)
        T = [{k: b[k] for k in ("t", "o", "h", "l", "c")} for b in days[d] if b["t"].time() < time(15, 15)]
        for i in range(2, len(T) - 1):
            if T[i]["t"].time() > time(14, 30):
                break
            c = T[i]["c"]
            for dr in (1, -1):
                f = es.features(T, i, dr, ctx)
                if f is None:
                    continue
                risk = f["risk_pts"]
                hit3 = stop = False
                fav = 0.0
                for x in T[i + 1:]:
                    if ((c - x["l"]) if dr > 0 else (x["h"] - c)) >= risk:
                        stop = True
                        break
                    fav = max(fav, (x["h"] - c) if dr > 0 else (c - x["l"]))
                    if fav >= 3 * risk:
                        hit3 = True
                        break
                out.append({"day": str(d), "t": T[i]["t"].strftime("%H:%M"), "risk": risk, "win3": hit3,
                            "stop": stop, "mfe": fav, "f": f})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ship", action="store_true")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    R = rows_for(pl.load(a.data))
    days = sorted({r["day"] for r in R}); cut = days[int(len(days) * 2 / 3)]
    tr = [r for r in R if r["day"] < cut]; te = [r for r in R if r["day"] >= cut]
    vocab = sorted({x for r in R for x in es.tokens(r["f"])}); ix = {x: i for i, x in enumerate(vocab)}

    def X(rs):
        m = np.zeros((len(rs), len(vocab) + 1), dtype=np.float32); m[:, -1] = 1
        for n, r in enumerate(rs):
            for x in es.tokens(r["f"]):
                m[n, ix[x]] = 1
        return m

    def fit(rs):
        Xm = X(rs)
        w = pl.fit_logit(Xm, np.array([r["win3"] for r in rs], float), it=500)
        return w, 1 / (1 + np.exp(-Xm @ w))
    w, ptr = fit(tr)
    pt = 1 / (1 + np.exp(-X(te) @ w))
    for r, p in zip(te, pt):
        r["p"] = float(p)
    L = []
    P = L.append

    def ev(g):   # Nifty points per trade: +3R on a win, -1R on a stop, else 0 (flat at 15:15)
        return sum(3 * r["risk"] if r["win3"] else (-r["risk"] if r["stop"] else 0.0) for r in g) / max(len(g), 1)
    tdays = len({r['day'] for r in te})
    P(f"ENTRY LAB [swing stop] — {len(R)} candle-entries on {len(days)} days; learn to {cut}, judge on {len(te)} ({tdays} days)")
    P(f"baseline (any candle): P(3R before stop) {np.mean([r['win3'] for r in te]):.1%}, {ev(te):+.1f} pts/trade")
    P("\nLEARNED SCORE on unseen days — thresholds fixed from the TRAINING score distribution (no peeking)")
    for q in (0.2, 0.1, 0.05, 0.02, 0.01):
        thr = float(np.quantile(ptr, 1 - q))
        g = [r for r in te if r["p"] >= thr]
        if g:
            P(f"  train-top {q:>4.0%} (p>={thr:.3f}): P(3R) {np.mean([r['win3'] for r in g]):5.1%}  {ev(g):+5.1f} pts/trade  "
              f"risk {np.mean([r['risk'] for r in g]):4.1f}  {len(g)/tdays:4.1f}/day  days {len({r['day'] for r in g})}")
    P("\nstrongest learned drivers:")
    for k, v in sorted(zip(vocab, w[:-1]), key=lambda kv: -kv[1])[:14]:
        P(f"  + {k:34s} {v:+.2f}")
    for k, v in sorted(zip(vocab, w[:-1]), key=lambda kv: kv[1])[:10]:
        P(f"  - {k:34s} {v:+.2f}")
    if a.ship:
        wa, pall = fit(R)
        model = {"vocab": vocab, "w": [float(x) for x in wa[:-1]], "bias": float(wa[-1]),
                 "threshold": float(np.quantile(pall, 0.98)), "thresholds": {f"top{int(q*100)}": float(np.quantile(pall, 1 - q))
                                                                            for q in (0.01, 0.02, 0.05, 0.10)},
                 "trained_days": [days[0], days[-1]], "label": "3R before swing stop by 15:15"}
        p = Path(es.__file__).with_name("entry_score_model.json")
        p.write_text(json.dumps(model, indent=0))
        P(f"\nshipped model (all days) -> {p}  thresholds {model['thresholds']}")
    text = "\n".join(L); (out / "report.txt").write_text(text); print(text)


if __name__ == "__main__":
    main()
