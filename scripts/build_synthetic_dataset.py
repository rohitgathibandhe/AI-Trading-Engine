#!/usr/bin/env python
"""Year-scale replay dataset: real Nifty 5m candles + a SYNTHETIC option chain every 5 minutes, so the real
run_live loop (scripts/live_path_replay.py --data) can be validated on ~350 sessions instead of the ~30 with
recorded chains.

Pricing: Black-Scholes, ATM IV = India VIX x f(DTE), per-strike smile, bid/ask — all CALIBRATED on the recorded
live chains (2026-09/10): ATM/VIX 1.21 (0DTE) 1.07 (1) ~0.85 (3-5) 0.92 (6); put skew 1.12 (-0.5sd) .. 1.29 (-2sd);
spreads 0.05 (<20) 0.10 (<60) 0.20 (<200). NO look-ahead: the chain at bucket t prices off the previous bar's
close (the replay serves it at t+4m). OI is left EMPTY — it cannot be reconstructed for past days, so OI/wall
signals are neutral in this replay; the recorded-chain replay remains the test for those.

  python scripts/build_synthetic_dataset.py --start 2025-05-01 --end 2025-05-31 --out <dir> [--source <research dir>]
Source: JSON from Dhan intraday API (nifty_5m.json, vix_5m.json) — fetched by scripts in the research scratchpad.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from bisect import bisect_left
from collections import defaultdict
from datetime import date, datetime, time, timedelta
from pathlib import Path

ATM_BY_DTE = {0: 1.21, 1: 1.07, 2: 0.95, 3: 0.83, 4: 0.86, 5: 0.87, 6: 0.92}
# ATM IV / VIX by DTE x HOUR, measured on the recorded live chains (2026-07-15..10-16; DTE 2 interpolated): IV MELTS
# intraday (DTE 3: 0.91 -> 0.73; expiry 1.26 -> 0.97). The flat-by-DTE table over-priced intraday option buying.
ATM_BY_DTE_HOUR = {
    0: {9: 1.26, 10: 1.21, 11: 1.19, 12: 1.14, 13: 1.06, 14: 1.22, 15: 0.97},
    1: {9: 1.08, 10: 1.10, 11: 1.13, 12: 1.09, 13: 1.04, 14: 1.02, 15: 1.02},
    2: {9: 1.00, 10: 0.99, 11: 0.99, 12: 0.96, 13: 0.91, 14: 0.89, 15: 0.88},
    3: {9: 0.91, 10: 0.88, 11: 0.85, 12: 0.82, 13: 0.78, 14: 0.75, 15: 0.73},
    4: {9: 0.87, 10: 0.88, 11: 0.86, 12: 0.85, 13: 0.81, 14: 0.83, 15: 0.81},
    5: {9: 0.88, 10: 0.87, 11: 0.87, 12: 0.85, 13: 0.88, 14: 0.85, 15: 0.86},
    6: {9: 0.93, 10: 0.92, 11: 0.91, 12: 0.91, 13: 0.95, 14: 0.94, 15: 0.91},
}
import os as _os
IV_MODE = _os.environ.get("SYN_IV_MODE", "DTE_HOUR")      # DTE_HOUR (calibrated) | DTE (legacy flat)


def _atm_ratio(dte: int, hour: int) -> float:
    if IV_MODE != "DTE_HOUR":
        return ATM_BY_DTE.get(min(dte, 6), 0.92)
    row = ATM_BY_DTE_HOUR.get(min(dte, 6), ATM_BY_DTE_HOUR[6])
    return row.get(min(max(hour, 9), 15), 0.9)
SMILE = [(-3.0, 1.78), (-2.5, 1.41), (-2.0, 1.29), (-1.5, 1.21), (-1.0, 1.15), (-0.5, 1.12), (0.0, 1.0),
         (0.5, 0.90), (1.0, 0.92), (1.5, 0.97), (2.0, 1.05), (2.5, 1.16), (3.0, 1.53)]
STRIKE_SPAN, STEP = 1000, 50


def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _bs(S, K, T, iv, put):
    s = iv * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * s * s) / s
    d2 = d1 - s
    px = (K * _ncdf(-d2) - S * _ncdf(-d1)) if put else (S * _ncdf(d1) - K * _ncdf(d2))
    delta = (_ncdf(d1) - 1) if put else _ncdf(d1)
    return max(px, 0.05), delta


def _smile(x):
    x = max(min(x, 3.0), -3.0)
    for (x0, y0), (x1, y1) in zip(SMILE, SMILE[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return 1.0


def _spread(px):
    return 0.05 if px < 20 else 0.10 if px < 60 else 0.20 if px < 200 else 0.40


def _load(p):
    out = []
    for b in json.load(open(p)):
        t = b["timestamp"]
        t = datetime.fromisoformat(str(t).replace("Z", "")) if not isinstance(t, (int, float)) else datetime.fromtimestamp(t)
        out.append({"t": t.replace(tzinfo=None), "o": float(b["open"]), "h": float(b["high"]), "l": float(b["low"]),
                    "c": float(b["close"]), "v": float(b.get("volume") or 0)})
    return sorted(out, key=lambda x: x["t"])


def _expiry(d, trading_days):
    wd = 1 if d >= date(2025, 9, 1) else 3          # Tuesday from 2025-09-01, Thursday before
    e = d + timedelta(days=(wd - d.weekday()) % 7)
    while e not in trading_days and e > d:           # holiday -> previous trading day
        e -= timedelta(days=1)
    return e


def build(src: Path, start: date, end: date, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    N = [b for b in _load(src / "nifty_5m.json") if time(9, 15) <= b["t"].time() < time(15, 30)]
    V = _load(src / "vix_5m.json")
    vt = [b["t"] for b in V]
    days = sorted({b["t"].date() for b in N})
    tdays = set(days)
    keep = [d for d in days if start <= d <= end]
    prev = [d for d in days if d < start][-3:]
    bars = [b for b in N if b["t"].date() in set(keep) | set(prev)]
    by_day = defaultdict(list)
    for b in bars:
        by_day[b["t"].date()].append(b)
    rows = []
    for d in keep:
        exp = _expiry(d, tdays)
        dte = (exp - d).days
        day = by_day[d]
        for i, b in enumerate(day):
            spot = day[i - 1]["c"] if i else b["o"]        # price AT the bucket start — no look-ahead
            k = bisect_left(vt, b["t"]) - 1
            vix = V[k]["c"] if k >= 0 and V[k]["t"].date() == d else (V[max(k, 0)]["c"] if V else 14.0)
            T = max((datetime.combine(exp, time(15, 30)) - b["t"]).total_seconds(), 900) / (365 * 86400)
            atm = vix / 100 * _atm_ratio(dte, b["t"].hour)
            k0 = round(spot / STEP) * STEP
            for K in range(int(k0 - STRIKE_SPAN), int(k0 + STRIKE_SPAN) + STEP, STEP):
                x = math.log(K / spot) / (atm * math.sqrt(T))
                iv = atm * _smile(x)
                for side, put in (("CALL", False), ("PUT", True)):
                    px, dl = _bs(spot, K, T, iv, put)
                    h = _spread(px) / 2
                    rows.append({"timestamp": b["t"].isoformat(), "expiry": exp.isoformat(), "spot": round(spot, 2),
                                 "strike": float(K), "option_type": side, "bid": round(max(px - h, 0.05), 2),
                                 "ask": round(px + h, 2), "ltp": round(px, 2), "delta": round(dl, 4),
                                 "iv": round(iv * 100, 2), "oi": ""})
    cols = ["timestamp", "expiry", "spot", "strike", "option_type", "bid", "ask", "ltp", "delta", "iv", "oi"]
    with (out / "options_chain.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    def _w(path, bb):
        with path.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
            for x in bb:
                w.writerow([x["t"].isoformat(), x["o"], x["h"], x["l"], x["c"], x["v"]])
    _w(out / "nifty_5m.csv", bars)
    g = defaultdict(list)
    for b in bars:
        g[b["t"].replace(minute=(b["t"].minute // 15) * 15)].append(b)
    _w(out / "nifty_15m.csv", [{"t": k, "o": v[0]["o"], "h": max(x["h"] for x in v), "l": min(x["l"] for x in v),
                                "c": v[-1]["c"], "v": sum(x["v"] for x in v)} for k, v in sorted(g.items())])
    (out / "dataset_metadata.json").write_text(json.dumps({"source": "SYNTHETIC chain (BS+VIX, calibrated smile)",
                                                          "days": len(keep), "chain_rows": len(rows)}))
    print(f"days={len(keep)} chain_rows={len(rows)} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", default="/private/tmp/claude-501/-Users-Rohit-AI-Trading-Engine/2206980f-ba53-4074-bf0e-9d1a606e0751/scratchpad/research")
    a = ap.parse_args()
    build(Path(a.source), date.fromisoformat(a.start), date.fromisoformat(a.end), Path(a.out))
