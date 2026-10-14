"""TREND-DAY MODE — be positioned with the day's trend instead of waiting for one perfect moment.

Owner's standard (2026-10-14): on a trending day, being flat is the failure. 10-13 closed +358 at its high
and the agent made +416; its single-moment "high-probability" gate was missed or blocked most of the day.
This module turns the agent's existing building blocks into a session playbook:

  STATE (decides the day's direction, re-evaluated every cycle on COMPLETED 5m bars)
    1. 15-min ORB: a completed bar closed beyond the 09:15-09:30 opening range           (required)
    2. VWAP hold : the last VWAP_HOLD_BARS completed bars all closed on that side of VWAP  (required)
    3. structure : 15m higher lows (up) / lower highs (down)                               (one of 3/4)
    4. OI        : put writing dominates (up) / call writing dominates (down)              (one of 3/4)

  TRIGGERS (entry, in the trend's direction only)
    PULLBACK      price retraced >= PULLBACK_MIN_PTS from the day extreme into the VWAP/EMA20 zone, then a
                  completed bar resumed (closed beyond the prior bar's high/low) on the trend side of VWAP
    DAY_BREAKOUT  a completed bar closed beyond the prior day-high (up) / day-low (down) with volume
    RANGE_BREAKOUT a tight consolidation (last RANGE_BARS bars within RANGE_MAX_PTS) broke in the trend's way
    LATE_SAFETY   the trend has held >= 30 min, it's >= 11:30 and no trigger fired in the last hour ->
                  enter on the next completed bar that closes in the trend's direction (missed-trend net)

Pure functions on bars — no I/O — so the same logic runs live (regime.py) and in tests/replays.
"""
from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any, Sequence

OR_END = time(9, 30)
VWAP_HOLD_BARS = 6            # 30 minutes on one side of VWAP
PULLBACK_MIN_PTS = 20.0
PULLBACK_ZONE_PTS = 15.0      # "into the VWAP / EMA20 zone" = within this many points of either
RANGE_BARS = 6
RANGE_MAX_PTS = 40.0
SAFETY_FROM = time(11, 30)
SAFETY_IDLE = timedelta(minutes=60)
LAST_ENTRY = time(14, 0)       # replay 10-08/09/12/13: entries after ~14:00 were exhausted (-20..-51pts on 3/4 days)


def _get(b, k):
    return b[k] if isinstance(b, dict) else getattr(b, {"t": "timestamp", "o": "open", "h": "high", "l": "low",
                                                         "c": "close", "v": "volume"}[k])


def _norm(bars: Sequence[Any]) -> list[dict]:
    out = []
    for b in bars:
        t = _get(b, "t")
        t = datetime.fromisoformat(str(t)) if not isinstance(t, datetime) else t
        out.append({"t": t.replace(tzinfo=None), "o": float(_get(b, "o")), "h": float(_get(b, "h")),
                    "l": float(_get(b, "l")), "c": float(_get(b, "c")), "v": float(_get(b, "v") or 0.0)})
    return out


def completed(bars: Sequence[Any], now: datetime, interval_min: int = 5) -> list[dict]:
    """Session bars whose 5-minute window has fully closed at `now`."""
    now = now.replace(tzinfo=None)
    return [b for b in _norm(bars) if b["t"] + timedelta(minutes=interval_min) <= now and b["t"].date() == now.date()]


def _vwap_series(bars: list[dict]) -> list[float]:
    pv = vv = 0.0
    out = []
    for b in bars:
        tp = (b["h"] + b["l"] + b["c"]) / 3.0
        vol = b["v"] if b["v"] > 0 else 1.0
        pv += tp * vol
        vv += vol
        out.append(pv / vv)
    return out


def _ema(vals: list[float], n: int = 20) -> list[float]:
    k = 2 / (n + 1)
    out, e = [], None
    for v in vals:
        e = v if e is None else v * k + e * (1 - k)
        out.append(e)
    return out


def _fifteen(bars: list[dict]) -> list[dict]:
    groups: dict[datetime, list[dict]] = {}
    for b in bars:
        key = b["t"].replace(minute=(b["t"].minute // 15) * 15)
        groups.setdefault(key, []).append(b)
    return [{"t": k, "h": max(x["h"] for x in g), "l": min(x["l"] for x in g)}
            for k, g in sorted(groups.items()) if len(g) == 3]


def trend_state(bars: list[dict], oi: dict | None = None) -> dict:
    """{'state': 'TREND_UP'|'TREND_DOWN'|'NONE', 'since': datetime|None, 'checks': {...}} on COMPLETED bars."""
    oi = oi or {}
    res = {"state": "NONE", "since": None, "checks": {}}
    or_bars = [b for b in bars if b["t"].time() < OR_END]
    post = [b for b in bars if b["t"].time() >= OR_END]
    if len(or_bars) < 3 or len(post) < VWAP_HOLD_BARS:
        return res
    or_h, or_l = max(b["h"] for b in or_bars), min(b["l"] for b in or_bars)
    vw = _vwap_series(bars)
    f15 = _fifteen(bars)
    for side in ("UP", "DOWN"):
        up = side == "UP"
        orb = any((b["c"] > or_h) if up else (b["c"] < or_l) for b in post)
        tail = list(zip(bars, vw))[-VWAP_HOLD_BARS:]
        hold = all((b["c"] > v) if up else (b["c"] < v) for b, v in tail)
        # vs the 15m bar TWO back: a single pullback bar makes a lower low by definition, and comparing with the
        # immediately previous bar switched the trend OFF during the very pullback we want to enter on
        struct = len(f15) >= 3 and ((f15[-1]["l"] > f15[-3]["l"]) if up else (f15[-1]["h"] < f15[-3]["h"]))
        p_chg, c_chg = float(oi.get("put_oi_change") or 0.0), float(oi.get("call_oi_change") or 0.0)
        smb = str(oi.get("smart_money_bias") or "").upper()
        oi_ok = ((p_chg > c_chg and p_chg > 0) or smb == "BULLISH") if up else ((c_chg > p_chg and c_chg > 0) or smb == "BEARISH")
        if orb and hold and (struct or oi_ok):
            # since = the first bar of the current unbroken VWAP-side run (how long the trend has held)
            since = bars[-1]["t"]
            for b, v in reversed(list(zip(bars, vw))):
                if (b["c"] > v) if up else (b["c"] < v):
                    since = b["t"]
                else:
                    break
            res = {"state": f"TREND_{side}", "since": since,
                   "checks": {"orb": orb, "vwap_hold": hold, "structure": struct, "oi": oi_ok,
                              "or_high": or_h, "or_low": or_l, "vwap": round(vw[-1], 2)}}
            return res
    return res


def trigger(bars: list[dict], state: dict, *, last_trigger_at: datetime | None = None) -> dict | None:
    """Entry trigger on the LAST completed bar, in the state's direction. None if nothing fired."""
    if state.get("state") not in ("TREND_UP", "TREND_DOWN") or len(bars) < RANGE_BARS + 2:
        return None
    up = state["state"] == "TREND_UP"
    last, prev = bars[-1], bars[-2]
    if last["t"].time() > LAST_ENTRY:
        return None
    vw = _vwap_series(bars)
    ema = _ema([b["c"] for b in bars])
    on_side = (last["c"] > vw[-1]) if up else (last["c"] < vw[-1])
    if not on_side:
        return None
    prior = bars[:-1]
    day_ext = max(b["h"] for b in prior) if up else min(b["l"] for b in prior)
    avg_v = sum(b["v"] for b in prior[-12:]) / max(len(prior[-12:]), 1)

    # DAY_BREAKOUT: close beyond the prior day extreme, with at-least-average volume
    if (last["c"] > day_ext if up else last["c"] < day_ext) and last["v"] >= avg_v:
        return {"kind": "DAY_BREAKOUT", "at": last["t"], "spot": last["c"],
                "why": f"closed {'above day-high' if up else 'below day-low'} {day_ext:.0f} with volume"}

    # RANGE_BREAKOUT: tight consolidation, then a close beyond it
    box = prior[-RANGE_BARS:]
    hi, lo = max(b["h"] for b in box), min(b["l"] for b in box)
    if hi - lo <= RANGE_MAX_PTS and (last["c"] > hi if up else last["c"] < lo):
        return {"kind": "RANGE_BREAKOUT", "at": last["t"], "spot": last["c"],
                "why": f"broke a {hi - lo:.0f}pt range {'up' if up else 'down'}"}

    # PULLBACK: retraced from the extreme into the VWAP/EMA zone within the last 6 bars, then resumed
    recent = prior[-6:]
    depth = (day_ext - min(b["l"] for b in recent)) if up else (max(b["h"] for b in recent) - day_ext)
    zone_touch = any(
        (min(abs(b["l"] - v), abs(b["l"] - e)) if up else min(abs(b["h"] - v), abs(b["h"] - e))) <= PULLBACK_ZONE_PTS
        for b, v, e in zip(recent, vw[-7:-1], ema[-7:-1]))
    resumed = (last["c"] > prev["h"]) if up else (last["c"] < prev["l"])
    if depth >= PULLBACK_MIN_PTS and zone_touch and resumed:
        return {"kind": "PULLBACK", "at": last["t"], "spot": last["c"],
                "why": f"{depth:.0f}pt pullback into VWAP/EMA zone, resumed {'up' if up else 'down'}"}

    # LATE_SAFETY: trend held >= 30 min, >= 11:30, nothing fired in the last hour -> join on a trend-way bar
    since = state.get("since")
    idle = last_trigger_at is None or (last["t"] - last_trigger_at) >= SAFETY_IDLE
    if (since is not None and last["t"] - since >= timedelta(minutes=30) and last["t"].time() >= SAFETY_FROM
            and idle and ((last["c"] > last["o"]) if up else (last["c"] < last["o"]))):
        return {"kind": "LATE_SAFETY", "at": last["t"], "spot": last["c"],
                "why": "trend held >=30 min and still flat after 11:30 — join on a trend-way bar"}
    return None


_LAST_TRIGGER: dict[str, datetime] = {}       # session date -> last trigger the selector ACTED on


def note_trigger_used(at_iso: str) -> None:
    try:
        t = datetime.fromisoformat(at_iso)
        _LAST_TRIGGER[t.date().isoformat()] = t
    except Exception:
        pass


def evaluate(bars: Sequence[Any], now: datetime, oi: dict | None = None,
             last_trigger_at: datetime | None = None) -> dict:
    """One call for the live path: state + trigger on completed bars at `now`."""
    cb = completed(bars, now)
    if last_trigger_at is None:
        last_trigger_at = _LAST_TRIGGER.get(now.replace(tzinfo=None).date().isoformat())
    st = trend_state(cb, oi)
    tg = trigger(cb, st, last_trigger_at=last_trigger_at) if st["state"] != "NONE" else None
    if tg is not None:
        tg = dict(tg, at=tg["at"].isoformat())
    return {"state": st["state"], "since": st["since"].isoformat() if st["since"] else None,
            "checks": st["checks"], "trigger": tg}
