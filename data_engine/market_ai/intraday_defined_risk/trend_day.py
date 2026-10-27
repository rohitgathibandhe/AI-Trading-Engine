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
LAST_ENTRY = time(14, 0)
# A session whose completed bars have CLOSED beyond BOTH sides of the opening range is a two-sided / range day,
# never a trend day. 357 sessions (2026-07-16..10-16, both halves): one-sided OR breaks became trend days 39%|37%
# (median efficiency .60|.63); two-sided only 12%|7% (efficiency .33|.37). 2026-10-07 broke below the OR at 09:35
# and above it at 10:55 inside a 170pt range; the detector flipped DOWN -> UP -> DOWN and the trend-day debit
# bought the false 10:55 breakout (-4,999 in replay). TD_TWO_SIDED_RANGE=0 disables.
import os as _os
TWO_SIDED_RANGE = _os.environ.get("TD_TWO_SIDED_RANGE", "1") == "1"
# ACCEPTANCE (TD_BREAK_ACCEPT_BARS=N, default 0 = off): a side of the opening range counts as BROKEN only after N
# consecutive closes beyond it. A single close that falls straight back inside is a failed break (a trap), not a
# break — 2026-10-01: one 10:00 close 10pts over the 15m high (22,600 vs 22,590), next bar back inside; the 12:15/12:20
# closes below the low were the real M-top breakdown (-300pts to 14:00) but the session was labelled RANGE_TWO_SIDED
# and every bearish entry from 12:25 was vetoed.
BREAK_ACCEPT_BARS = int(_os.environ.get("TD_BREAK_ACCEPT_BARS", "0") or 0)


# RELEASE (TD_TWO_SIDED_RELEASE_X=x, default 0 = off): a two-sided session stops being a "range" once price closes
# beyond the LATER-broken side by >= x * the opening-range size, on that side of VWAP — a failed break that reversed
# into a trend. 2026-10-21: 09:30/09:35 closed above the 15m high, 10:05/10:10 below the low -> RANGE_TWO_SIDED all
# day; every put debit (11:45, 14:25, 14:45) was vetoed while Nifty fell 150-400 pts below the range (-470 on the day).
TWO_SIDED_RELEASE_X = float(_os.environ.get("TD_TWO_SIDED_RELEASE_X", "0") or 0)


def _two_sided_released(bars: list[dict], post: list[dict], or_h: float, or_l: float) -> bool:
    if TWO_SIDED_RELEASE_X <= 0 or not post:
        return False
    first_up = next((k for k, b in enumerate(post) if b["c"] > or_h), None)
    first_dn = next((k for k, b in enumerate(post) if b["c"] < or_l), None)
    if first_up is None or first_dn is None:
        return False
    last, vw = bars[-1], _vwap_series(bars)[-1]
    ext = TWO_SIDED_RELEASE_X * max(or_h - or_l, 1.0)
    if first_dn > first_up:                                  # down was the later break
        return last["c"] <= or_l - ext and last["c"] < vw
    return last["c"] >= or_h + ext and last["c"] > vw


def _accepted(post: list[dict], beyond) -> bool:
    n = max(BREAK_ACCEPT_BARS, 1)
    run = 0
    for b in post:
        run = run + 1 if beyond(b) else 0
        if run >= n:
            return True
    return False       # replay 10-08/09/12/13: entries after ~14:00 were exhausted (-20..-51pts on 3/4 days)


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


# FAILED-BREAK RESET (TD_FAILED_RESETS=1): once a side's break FAILED (failed_break), that side's level becomes the trap
# extreme + TD_RESET_MARGIN_OR x OR range, and it counts as broken again only after 2 consecutive closes beyond it.
# 2026-10-27: the 09:45 up-break failed at 10:20 (trap 22,804) yet at 12:30-12:45 the trend detector and accepted_break
# reused the morning closes -> stale TREND_UP -> 12:45 call at the range top (-3,398) while the classifier said RANGE.
FAILED_RESETS = _os.environ.get("TD_FAILED_RESETS", "0") == "1"
RESET_MARGIN_OR = float(_os.environ.get("TD_RESET_MARGIN_OR", "0.25") or 0.25)


def _reset_levels(bars: list[dict], or_h: float, or_l: float) -> tuple[float, float, dict]:
    """Effective (up_level, down_level) after failed breaks; also which sides were reset."""
    if not FAILED_RESETS:
        return or_h, or_l, {}
    post = [b for b in bars if b["t"].time() >= OR_END]
    vw = _vwap_series(bars)
    off = len(bars) - len(post)
    up_lvl, dn_lvl, info = or_h, or_l, {}
    margin = RESET_MARGIN_OR * max(or_h - or_l, 1.0)
    for side in ("UP", "DOWN"):
        lvl = or_h if side == "UP" else or_l
        k0 = next((k for k, b in enumerate(post) if (b["c"] > lvl if side == "UP" else b["c"] < lvl)), None)
        if k0 is None:
            continue
        for k in range(k0 + 1, len(post)):
            b, v = post[k], vw[off + k]
            if (b["c"] < lvl and b["c"] < v) if side == "UP" else (b["c"] > lvl and b["c"] > v):
                ext = max(x["h"] for x in post[k0:k + 1]) if side == "UP" else min(x["l"] for x in post[k0:k + 1])
                if side == "UP":
                    up_lvl = ext + margin
                else:
                    dn_lvl = ext - margin
                info[side] = {"failed_at": b["t"].isoformat(), "trap": ext}
                break
    return up_lvl, dn_lvl, info


def trend_state(bars: list[dict], oi: dict | None = None) -> dict:
    """{'state': 'TREND_UP'|'TREND_DOWN'|'NONE', 'since': datetime|None, 'checks': {...}} on COMPLETED bars."""
    oi = oi or {}
    res = {"state": "NONE", "since": None, "checks": {}}
    or_bars = [b for b in bars if b["t"].time() < OR_END]
    post = [b for b in bars if b["t"].time() >= OR_END]
    if len(or_bars) < 3 or len(post) < VWAP_HOLD_BARS:
        return res
    or_h, or_l = max(b["h"] for b in or_bars), min(b["l"] for b in or_bars)
    lv_h, lv_l, _reset = _reset_levels(bars, or_h, or_l)
    if TWO_SIDED_RANGE and _accepted(post, lambda b: b["c"] > or_h) and _accepted(post, lambda b: b["c"] < or_l) \
            and not _two_sided_released(bars, post, or_h, or_l):
        return {"state": "RANGE_TWO_SIDED", "since": None,
                "checks": {"two_sided": True, "or_high": or_h, "or_low": or_l}}
    vw = _vwap_series(bars)
    f15 = _fifteen(bars)
    for side in ("UP", "DOWN"):
        up = side == "UP"
        if (side in _reset):            # failed earlier: needs 2 consecutive closes beyond the trap level (after the failure)
            _fa = _reset[side]["failed_at"]
            _after = [b for b in post if b["t"].isoformat() > _fa]
            orb = _accepted(_after, (lambda b: b["c"] > lv_h) if up else (lambda b: b["c"] < lv_l)) if len(_after) >= 2 else False
        else:
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


def accepted_break(bars: list[dict]) -> dict | None:
    """The owner's trigger on COMPLETED bars: the first time two consecutive closes sit beyond the 15-min opening range
    (an ACCEPTED break, not a one-bar poke). {'dir': 'UP'|'DOWN', 'at', 'bars_since', 'valid', 'two_sided'}; valid =
    the latest close is still beyond that side AND on the same side of VWAP."""
    or_bars = [b for b in bars if b["t"].time() < OR_END]
    post = [b for b in bars if b["t"].time() >= OR_END]
    if len(or_bars) < 3 or len(post) < 2:
        return None
    or_h, or_l = max(b["h"] for b in or_bars), min(b["l"] for b in or_bars)
    first = {}
    for k in range(1, len(post)):
        for side, beyond in (("UP", lambda b: b["c"] > or_h), ("DOWN", lambda b: b["c"] < or_l)):
            if side not in first and beyond(post[k]) and beyond(post[k - 1]):
                first[side] = k
    if not first:
        return None
    if FAILED_RESETS:
        _uh, _dl, _rs = _reset_levels(bars, or_h, or_l)
        for _sd in list(first):
            if _sd in _rs:
                _after = [b for b in post if b["t"].isoformat() > _rs[_sd]["failed_at"]]
                _lvl = _uh if _sd == "UP" else _dl
                _re = next((k for k in range(1, len(_after)) if ((_after[k]["c"] > _lvl and _after[k - 1]["c"] > _lvl) if _sd == "UP"
                                                                 else (_after[k]["c"] < _lvl and _after[k - 1]["c"] < _lvl))), None)
                if _re is None:
                    del first[_sd]
                else:
                    first[_sd] = post.index(_after[_re])
        if not first:
            return None
    side = min(first, key=first.get) if len(first) == 1 else max(first, key=first.get)   # the latest accepted side
    k = first[side]
    last, vw = bars[-1], _vwap_series(bars)[-1]
    up = side == "UP"
    valid = (last["c"] > or_h and last["c"] > vw) if up else (last["c"] < or_l and last["c"] < vw)
    return {"dir": side, "at": post[k]["t"].isoformat(), "bars_since": len(post) - 1 - k, "valid": bool(valid),
            "two_sided": len(first) == 2, "or_high": or_h, "or_low": or_l}


def last_swings(bars: list[dict]) -> dict:
    """Last CONFIRMED 5m swing high / low (a bar whose high/low beats the two bars either side), for chart stops."""
    hi = lo = None
    for k in range(2, len(bars) - 2):
        b = bars[k]
        if b["h"] >= max(bars[k - 2]["h"], bars[k - 1]["h"], bars[k + 1]["h"], bars[k + 2]["h"]):
            hi = b["h"]
        if b["l"] <= min(bars[k - 2]["l"], bars[k - 1]["l"], bars[k + 1]["l"], bars[k + 2]["l"]):
            lo = b["l"]
    return {"high": hi, "low": lo, "day_high": max((b["h"] for b in bars), default=None),
            "day_low": min((b["l"] for b in bars), default=None)}


def fib_pullback(bars: list[dict], min_impulse: float = 80.0) -> dict | None:
    """Owner's setup: after an impulse leg (>= min_impulse pts), price pulls back into the 61.8-68% retracement zone
    and the LAST CLOSED candle is a strong candle off the zone in the impulse direction. Invalid if price closed
    beyond the 78.6% level. {'dir': 'UP'|'DOWN', 'zone': (lo, hi), 'inval', 'impulse': (from, to), 'bars_since_touch'}"""
    post = [b for b in bars if b["t"].time() >= OR_END]
    if len(post) < 6:
        return None
    last = post[-1]
    rng = max(last["h"] - last["l"], 0.5)
    out = None
    for up in (True, False):
        if up:      # impulse UP: lowest low, then highest high after it
            i0 = min(range(len(post) - 1), key=lambda k: post[k]["l"])
            if i0 >= len(post) - 3:
                continue
            i1 = max(range(i0, len(post) - 1), key=lambda k: post[k]["h"])
            a, z = post[i0]["l"], post[i1]["h"]
        else:
            i0 = max(range(len(post) - 1), key=lambda k: post[k]["h"])
            if i0 >= len(post) - 3:
                continue
            i1 = min(range(i0, len(post) - 1), key=lambda k: post[k]["l"])
            a, z = post[i0]["h"], post[i1]["l"]
        imp = abs(z - a)
        if imp < min_impulse or i1 >= len(post) - 1:
            continue
        lvl = (lambda f: z - f * (z - a))
        z618, z68, z786 = lvl(0.618), lvl(0.68), lvl(0.786)
        lo, hi = min(z618, z68), max(z618, z68)
        after = post[i1 + 1:]
        if up:
            if any(b["c"] < z786 for b in after):
                continue
            touch = [k for k, b in enumerate(after) if b["l"] <= hi]
            strong = (last["c"] - last["o"]) / rng >= 0.5 and (last["c"] - last["l"]) / rng >= 0.7 and last["c"] > hi
        else:
            if any(b["c"] > z786 for b in after):
                continue
            touch = [k for k, b in enumerate(after) if b["h"] >= lo]
            strong = (last["o"] - last["c"]) / rng >= 0.5 and (last["h"] - last["c"]) / rng >= 0.7 and last["c"] < lo
        if not touch:
            continue
        since = len(after) - 1 - touch[-1]
        if strong and since <= 3:
            out = {"dir": "UP" if up else "DOWN", "zone": (round(lo, 2), round(hi, 2)), "inval": round(z786, 2),
                   "impulse": (round(a, 2), round(z, 2)), "bars_since_touch": since, "at": last["t"].isoformat()}
    return out


def mw_neckline_break(bars: list[dict], lookback: int = 30, tol_frac: float = 0.0025) -> dict | None:
    """M top / W bottom (incl. the LOWER-high M / HIGHER-low W) with a neckline break on the LAST CLOSED candle.
    Owner's 2026-02-01 read: top 25,441 (11:05), dip to 25,356 (11:20), lower top 25,403 (11:25-11:30), 11:40 strong close
    25,312 through the neckline -> -740 pts. First top: a swing high (2 bars each side); second top: the highest high
    after the trough, confirmed by 1 bar each side, not above the first and within tol_frac of it; neckline: lowest
    low between them; trigger: last close through the neckline, previous close not. {'dir','neckline','top','at'}"""
    if len(bars) < 12:
        return None
    seg = bars[-lookback:]
    last, prev = seg[-1], seg[-2]
    n = len(seg)
    for down in (True, False):
        hi = (lambda x: x["h"]) if down else (lambda x: -x["l"])     # mirror W into M
        p1 = [k for k in range(2, n - 4) if hi(seg[k]) >= max(hi(seg[k - 2]), hi(seg[k - 1]), hi(seg[k + 1]), hi(seg[k + 2]))]
        for k1 in reversed(p1):
            t1 = hi(seg[k1])
            lo = (lambda x: x["l"]) if down else (lambda x: -x["h"])
            span = list(range(k1 + 1, n - 3))
            if not span:
                continue
            kt = min(span, key=lambda k: lo(seg[k]))                 # the dip (neckline) after the first top
            rest = list(range(kt + 1, n - 2))
            if not rest:
                continue
            k2 = max(rest, key=lambda k: hi(seg[k]))                 # the second (lower) top after the dip
            t2 = hi(seg[k2])
            if not (hi(seg[k2 - 1]) <= t2 >= hi(seg[k2 + 1])) or t2 > t1 or (t1 - t2) > tol_frac * abs(t1):
                continue
            neck = seg[kt]["l"] if down else seg[kt]["h"]
            crossed = (last["c"] < neck <= prev["c"]) if down else (last["c"] > neck >= prev["c"])
            if crossed:
                top = max(seg[k1]["h"], seg[k2]["h"]) if down else min(seg[k1]["l"], seg[k2]["l"])
                return {"dir": "DOWN" if down else "UP", "neckline": round(neck, 2), "top": round(top, 2),
                        "at": last["t"].isoformat()}
            break
    return None


def failed_break(bars: list[dict]) -> dict | None:
    """FAILED OPENING-RANGE BREAK (a trap) on COMPLETED bars: a close beyond one side of the 15-min opening range, then a
    later close back INSIDE the range and on the other side of VWAP. Trade the reversal. 2026-10-21: 09:30/09:35 closed
    above the 15m high (23,087), 09:45 closed back at 23,058 under VWAP -> the -470pt day (owner sold the 23,100 call;
    +14.6k on 2 lots at 09:45 vs +3k at the agent's 11:50). Pattern lab (361 days): OR_FAIL +0.18R short / +0.30R long.
    {'dir': reversal 'UP'|'DOWN', 'at', 'bars_since', 'valid', 'trap_extreme'}; valid = latest close still inside/through
    the range on the reversal side of VWAP."""
    or_bars = [b for b in bars if b["t"].time() < OR_END]
    post = [b for b in bars if b["t"].time() >= OR_END]
    if len(or_bars) < 3 or len(post) < 2:
        return None
    or_h, or_l = max(b["h"] for b in or_bars), min(b["l"] for b in or_bars)
    vw = _vwap_series(bars)
    off = len(bars) - len(post)
    best = None
    for side in ("UP", "DOWN"):
        k0 = next((k for k, b in enumerate(post) if (b["c"] > or_h if side == "UP" else b["c"] < or_l)), None)
        if k0 is None:
            continue
        for k in range(k0 + 1, len(post)):
            b, v = post[k], vw[off + k]
            back = (b["c"] < or_h and b["c"] < v) if side == "UP" else (b["c"] > or_l and b["c"] > v)
            if back:
                ext = max(x["h"] for x in post[k0:k + 1]) if side == "UP" else min(x["l"] for x in post[k0:k + 1])
                ev = {"dir": "DOWN" if side == "UP" else "UP", "at": b["t"].isoformat(), "k": k, "trap_extreme": ext}
                if best is None or k > best["k"]:
                    best = ev
                break
    if best is None:
        return None
    last, v = bars[-1], vw[-1]
    down = best["dir"] == "DOWN"
    valid = (last["c"] < or_h and last["c"] < v) if down else (last["c"] > or_l and last["c"] > v)
    return {"dir": best["dir"], "at": best["at"], "bars_since": len(post) - 1 - best.pop("k"), "valid": bool(valid),
            "trap_extreme": best["trap_extreme"], "or_high": or_h, "or_low": or_l}


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
            "checks": st["checks"], "trigger": tg, "accepted_break": accepted_break(cb),
            "failed_break": failed_break(cb), "swings": last_swings(cb),
            "last_bar": ({k: cb[-1][k] for k in ("o", "h", "l", "c")} if cb else None),
            "prev_close_bar": ({k: cb[-2][k] for k in ("o", "h", "l", "c")} if len(cb) > 1 else None),
            "fib": fib_pullback(cb), "mw": mw_neckline_break(cb),
            "avg_range12": (sum(b["h"] - b["l"] for b in cb[-13:-1]) / 12 if len(cb) >= 13 else None)}
