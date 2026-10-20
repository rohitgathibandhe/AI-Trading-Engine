"""ENTRY SCORE — the learned probability that an entry at the latest completed 5m candle, with the stop just beyond
the last confirmed swing, runs 3R before the stop (by 15:15). Learned walk-forward on every candle of 361 Nifty
sessions by scripts/entry_lab.py (one feature function — this one — for training, replay and live).

What the model learned (2026-10-20, swing stop, unseen last third of the history):
  * the top 1% of candles hit 3R 41.5% of the time vs 16.7% for any candle; the top 2% 33%;
  * which opening range matters depends on the open: after a HUGE first 5m candle (> 0.6 of a normal day's range)
    the break beyond the 15-min / 5-min range is the strongest positive, and trading back against it the strongest
    negative; after a merely BIG first candle the 5-min break is a fake-out (negative);
  * enter AT / NEAR the swing (small structure stop), never far from it; not stretched from VWAP; not afternoon.

Inputs are only what the live loop has: today's 5m bars and the prior sessions' daily bars (the live provider fetches
90 days). SEL_ENTRY_SCORE=1 gates edge-only entries on it (see strategy_selector._edge_only).
"""
from __future__ import annotations

import json
import math
from datetime import datetime, time
from pathlib import Path
from typing import Any, Sequence

MODEL_PATH = Path(__file__).with_name("entry_score_model.json")


def _sgn(x: float) -> int:
    return (x > 0) - (x < 0)


def _ema(vals: Sequence[float], n: int = 20) -> list[float]:
    k, out = 2 / (n + 1), []
    for v in vals:
        out.append(v if not out else out[-1] + k * (v - out[-1]))
    return out


def _g(b: Any, k: str):
    if isinstance(b, dict):
        return b[k]
    return getattr(b, {"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close"}[k])


def norm(bars: Sequence[Any]) -> list[dict]:
    out = []
    for b in bars:
        t = _g(b, "t")
        t = datetime.fromisoformat(str(t)) if not isinstance(t, datetime) else t
        out.append({"t": t.replace(tzinfo=None), "o": float(_g(b, "o")), "h": float(_g(b, "h")),
                    "l": float(_g(b, "l")), "c": float(_g(b, "c"))})
    return out


def daily_context(daily: Sequence[Any]) -> dict:
    """Prior sessions (oldest first): typical day range and the daily trend."""
    d = norm(daily)
    if len(d) < 10:
        return {"atr_d": None, "dtrend": 0}
    atr_d = sum(x["h"] - x["l"] for x in d[-10:]) / 10
    e = _ema([x["c"] for x in d], 20)
    dtrend = (_sgn(d[-1]["c"] - e[-1]) + _sgn(e[-1] - e[-6])) if len(d) >= 26 else 0
    return {"atr_d": atr_d, "dtrend": dtrend}


def features(T: list[dict], i: int, dr: int, ctx: dict) -> dict | None:
    """Features of an entry in direction dr (+1 long / -1 short) at the close of session bar T[i], with the stop
    beyond the last confirmed swing. None when there is no valid structure stop (or it is wider than 60 pts)."""
    if i < 2 or not T:
        return None
    b = T[i]
    seen = T[:i + 1]
    pivH = [k for k in range(2, i - 1) if T[k]["h"] >= max(T[k - 2]["h"], T[k - 1]["h"], T[k + 1]["h"], T[k + 2]["h"])]
    pivL = [k for k in range(2, i - 1) if T[k]["l"] <= min(T[k - 2]["l"], T[k - 1]["l"], T[k + 1]["l"], T[k + 2]["l"])]
    c = b["c"]
    if dr > 0 and pivL and T[pivL[-1]]["l"] < c:
        risk = c - T[pivL[-1]]["l"] + 2
    elif dr < 0 and pivH and T[pivH[-1]]["h"] > c:
        risk = T[pivH[-1]]["h"] - c + 2
    else:
        return None
    if risk > 60:
        return None
    risk = max(risk, 8.0)
    atr5 = sum(x["h"] - x["l"] for x in T[max(0, i - 11):i + 1]) / min(12, i + 1)
    atr_d = ctx.get("atr_d") or max(x["h"] for x in seen) - min(x["l"] for x in seen)
    o5h, o5l = T[0]["h"], T[0]["l"]
    o15 = [x for x in T if x["t"].time() < time(9, 30)] or T[:1]
    o15h, o15l = max(x["h"] for x in o15), min(x["l"] for x in o15)
    first5_size = (o5h - o5l) / max(atr_d, 1)
    pv = 0.0
    for k in range(i + 1):
        pv += (T[k]["h"] + T[k]["l"] + T[k]["c"]) / 3
    v = pv / (i + 1)
    e5 = _ema([x["c"] for x in seen], 20)
    m15 = [seen[k:k + 3][-1]["c"] for k in range(0, len(seen) - len(seen) % 3, 3)]
    t15 = _sgn(m15[-1] - m15[-4]) if len(m15) >= 4 else 0
    hh = len(pivH) >= 2 and T[pivH[-1]]["h"] > T[pivH[-2]]["h"]
    hl = len(pivL) >= 2 and T[pivL[-1]]["l"] > T[pivL[-2]]["l"]
    lh = len(pivH) >= 2 and T[pivH[-1]]["h"] < T[pivH[-2]]["h"]
    ll = len(pivL) >= 2 and T[pivL[-1]]["l"] < T[pivL[-2]]["l"]
    rng = max(b["h"] - b["l"], 0.5)
    body = (b["c"] - b["o"]) / rng
    up_wick = (b["h"] - max(b["c"], b["o"])) / rng
    dn_wick = (min(b["c"], b["o"]) - b["l"]) / rng
    hi_so, lo_so = max(x["h"] for x in seen), min(x["l"] for x in seen)
    broke = (bool(pivH) and dr > 0 and c > T[pivH[-1]]["h"] and T[i - 1]["c"] <= T[pivH[-1]]["h"]) or \
            (bool(pivL) and dr < 0 and c < T[pivL[-1]]["l"] and T[i - 1]["c"] >= T[pivL[-1]]["l"])
    near = (c - T[pivL[-1]]["l"]) / max(atr5, 1) if dr > 0 else (T[pivH[-1]]["h"] - c) / max(atr5, 1)
    tm = b["t"].time()
    return {
        "risk_pts": risk,
        "dir": dr, "first5": "huge" if first5_size > 0.6 else "big" if first5_size > 0.35 else "normal",
        "or5": "beyond" if (c > o5h if dr > 0 else c < o5l) else "against" if (c < o5l if dr > 0 else c > o5h) else "inside",
        "or15": "beyond" if (c > o15h if dr > 0 else c < o15l) else "against" if (c < o15l if dr > 0 else c > o15h) else "inside",
        "struct": "with" if ((hh and hl) if dr > 0 else (lh and ll)) else "against" if ((lh and ll) if dr > 0 else (hh and hl)) else "mixed",
        "broke_swing": bool(broke),
        "near_swing": "at" if near < 0.6 else "near" if near < 1.5 else "far",
        "vwap": "with" if (c - v) * dr > 0 else "against",
        "stretch": "far" if abs(c - v) / max(atr5, 1) > 2.5 else "near",
        "ema": "with" if (c - e5[-1]) * dr > 0 else "against",
        "ema_slope": "with" if len(e5) > 3 and (e5[-1] - e5[-4]) * dr > 0 else "against",
        "m15": t15 * dr, "daily": int(ctx.get("dtrend") or 0) * dr,
        "body": "strong" if body * dr >= 0.5 else "weak" if body * dr >= 0 else "against",
        "wick": "reject" if (dn_wick if dr > 0 else up_wick) >= 0.4 else "none",
        "room": "big" if (((hi_so - c) if dr < 0 else (c - lo_so)) / max(atr5, 1) > 3) else "small",
        "time": "open" if tm < time(9, 45) else "am" if tm < time(11, 0) else "mid" if tm < time(13, 0) else "pm",
        "risk": "tiny" if risk <= 12 else "small" if risk <= 22 else "med" if risk <= 40 else "wide",
    }


def tokens(f: dict) -> list[str]:
    base = [f"{k}={v}" for k, v in f.items() if k != "risk_pts"]
    return base + [f"dir={f['dir']}|struct={f['struct']}", f"dir={f['dir']}|or15={f['or15']}",
                   f"first5={f['first5']}|or5={f['or5']}", f"first5={f['first5']}|or15={f['or15']}",
                   f"broke={f['broke_swing']}|vwap={f['vwap']}", f"dir={f['dir']}|daily={f['daily']}"]


_MODEL: dict | None = None


def _model() -> dict | None:
    global _MODEL
    if _MODEL is None:
        try:
            m = json.loads(MODEL_PATH.read_text())
            m["_w"] = dict(zip(m["vocab"], m["w"]))
            _MODEL = m
        except Exception:  # noqa: BLE001 — no model = no score (the gate then fails open, see caller)
            _MODEL = {}
    return _MODEL or None


def prob(f: dict, model: dict | None = None) -> float:
    m = model or _model()
    w = m["_w"]
    z = m["bias"] + sum(w.get(t, 0.0) for t in tokens(f))
    return 1 / (1 + math.exp(-z))


def evaluate(bars_5m: Sequence[Any], now: datetime, daily: Sequence[Any] | None = None) -> dict:
    """Score LONG and SHORT at the latest completed 5m bar of today. {'long': p|None, 'short': p|None, ...}."""
    m = _model()
    now = now.replace(tzinfo=None)
    T = [b for b in norm(bars_5m) if b["t"].date() == now.date() and b["t"].timestamp() + 300 <= now.timestamp()]
    T = [b for b in T if b["t"].time() < time(15, 15)]
    out: dict = {"at": T[-1]["t"].isoformat() if T else None, "long": None, "short": None,
                 "threshold": (m or {}).get("threshold")}
    if not m or len(T) < 3:
        return out
    ctx = daily_context(daily or [])
    for dr, key in ((1, "long"), (-1, "short")):
        f = features(T, len(T) - 1, dr, ctx)
        if f is not None:
            out[key] = round(prob(f, m), 4)
            out[key + "_risk"] = round(f["risk_pts"], 1)
    return out
