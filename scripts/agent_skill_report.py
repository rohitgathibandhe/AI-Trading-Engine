#!/usr/bin/env python3
"""AGENT SKILL REPORT — judge the TRADER, not one structure.

The user's standard (2026-10-10): markets differ every day, so the deciding factor for going live is
whether the agent READS the market correctly, ENTERS at good moments and EXITS well — across every
kind of day — not whether one structure (put-debit) has 20 trades. This report measures exactly that,
structure-agnostic, from the forward record:

  READ    did spot move the way the agent's direction said? (60 min after entry, and by the close)
          + every recorded directional READ from pair_shadow (bias vs the next 60 min of spot).
  ENTRY   where the entry sat inside the +/-30 min window: 1.0 = sold the exact high / bought the
          exact low in the trade's favour; 0.5 = middle; 0 = the worst price in the window.
  HEAT    worst adverse spot excursion (points) between entry and exit.
  EXIT    capture = realized / peak open profit (MFE); and realized vs simply holding to 15:15.
  DAY     each trade's session classified from Nifty 5m candles: TREND_UP / TREND_DOWN (directional
          efficiency >= 0.5), RANGE (< 0.25), MIXED — every metric is also shown per day type, since
          skill must hold on all of them, not only the friendly ones.

Candles come from state/candle_cache_<date>_5m.json, else Dhan (cached under state/skill_candles/).
Decides nothing. Writes state/agent_skill.json (consumed by live_readiness).

  python scripts/agent_skill_report.py [--since 2026-09-17]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data_engine" / "market_ai" / "state"
CANDLE_DIR = STATE / "skill_candles"
OUT = STATE / "agent_skill.json"
DEFAULT_SINCE = "2026-09-17"      # current selector config (FORWARD_EPOCH in promotion_gate)

# Skill bars (the "competence gate"); per day type a bar is only judged with >= MIN_CELL samples.
BARS = {"read_60m": 0.60, "entry_score": 0.55, "exit_capture": 0.50}
MIN_CELL = 5

_DIR = {"PUT_DEBIT": -1, "BEAR_CALL": -1, "CALL_DEBIT": 1, "BULL_PUT": 1}


def _jsonl(p: Path) -> list[dict]:
    try:
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    except OSError:
        return []


def _direction(strategy: str) -> int:
    s = (strategy or "").upper()
    for k, v in _DIR.items():
        if k in s:
            return v
    return 0


def _ts(x) -> datetime | None:
    try:
        return datetime.fromisoformat(str(x)).replace(tzinfo=None)
    except Exception:
        return None


_dw = None


def _candles(day: str) -> list[dict]:
    """5m Nifty bars for a session: [{t, o, h, l, c}] (naive local times)."""
    def norm(rows):
        out = []
        for b in rows or []:
            t = _ts(b.get("timestamp") or b.get("ts") or b.get("t"))
            if t is None:
                continue
            out.append({"t": t, "o": float(b["open"]), "h": float(b["high"]), "l": float(b["low"]), "c": float(b["close"])})
        return sorted(out, key=lambda x: x["t"])
    for p in (STATE / f"candle_cache_{day}_5m.json", CANDLE_DIR / f"{day}.json"):
        try:
            raw = json.loads(p.read_text())
            rows = raw if isinstance(raw, list) else (raw.get("candles") or raw.get("bars") or raw.get("data") or [])
            bars = norm(rows)
            if len(bars) >= 60:                      # a full-ish session; partial caches are re-fetched
                return bars
        except Exception:
            pass
    global _dw
    try:
        if _dw is None:
            sys.path.insert(0, str(ROOT / "data_engine"))
            from market_ai.dhan_wrapper import DhanWrapper
            cr = json.loads((STATE / "creds.json").read_text())
            _dw = DhanWrapper(dhan_client_id=cr["client_id"], access_token=cr["access_token"])
        time.sleep(1.2)
        nxt = (datetime.fromisoformat(day) + timedelta(days=1)).date().isoformat()
        rows = _dw.get_intraday_candles(13, "IDX_I", interval=5, from_date=day, to_date=nxt)
        rows = [r for r in rows or [] if str(r.get("timestamp"))[:10] == day]
        CANDLE_DIR.mkdir(exist_ok=True)
        (CANDLE_DIR / f"{day}.json").write_text(json.dumps(rows, default=str))
        return norm(rows)
    except Exception as e:
        print(f"  [candles] {day}: unavailable ({type(e).__name__})", file=sys.stderr)
        return []


def _day_type(bars: list[dict]) -> str:
    if len(bars) < 20:
        return "UNKNOWN"
    o, c = bars[0]["o"], bars[-1]["c"]
    h, l = max(b["h"] for b in bars), min(b["l"] for b in bars)
    eff = abs(c - o) / (h - l) if h > l else 0.0
    if eff >= 0.5:
        return "TREND_UP" if c > o else "TREND_DOWN"
    return "RANGE" if eff < 0.25 else "MIXED"


def _spot_at(bars, t: datetime) -> float | None:
    prev = None
    for b in bars:
        if b["t"] > t:
            break
        prev = b
    return prev["c"] if prev else None


def _trade_metrics(entry: dict, exit_: dict, bars: list[dict], toclose: dict | None) -> dict:
    d = _direction(entry.get("strategy"))
    et, xt = _ts(entry.get("entry_timestamp") or entry.get("timestamp")), _ts(exit_.get("exit_timestamp") or exit_.get("timestamp"))
    spot = float(entry.get("spot_at_entry") or 0) or (_spot_at(bars, et) if et else None)
    pnl = float(exit_.get("realized_paper_pnl") or 0.0)
    mfe = float(exit_.get("mfe_rupees") or 0.0)
    m = {"date": str(et.date()) if et else None, "entry": et.isoformat(timespec="minutes") if et else None,
         "strategy": entry.get("strategy"), "direction": d, "pnl": round(pnl), "win": pnl > 0,
         "exit_reason": exit_.get("exit_reason"), "day_type": _day_type(bars)}
    # Capture is only meaningful once the trade was really in the money (MFE >= Rs 300); a loser that
    # peaked at +Rs 50 would otherwise read as -100% and swamp the median.
    m["exit_capture"] = round(max(-1.0, min(1.0, pnl / mfe)), 2) if mfe >= 300 else None
    hold = (toclose or {}).get("hold_to_close_rupees")
    m["exit_vs_hold"] = round(pnl - hold) if hold is not None else None
    # LOSS CONTROL: for a losing trade, did the exit cut it vs holding to the close? (True = cut cheaper)
    m["loss_cut"] = (pnl > hold) if (pnl < 0 and hold is not None) else None
    if d == 0 or not bars or not et or not spot:
        return m
    s60 = _spot_at(bars, et + timedelta(minutes=60))
    sclose = bars[-1]["c"]
    m["read_60m"] = (d * (s60 - spot) > 0) if s60 is not None else None
    m["read_close"] = d * (sclose - spot) > 0
    win = [b for b in bars if et - timedelta(minutes=30) <= b["t"] <= et + timedelta(minutes=30)]
    if win:
        hi, lo = max(b["h"] for b in win), min(b["l"] for b in win)
        if hi > lo:
            m["entry_score"] = round(((spot - lo) if d < 0 else (hi - spot)) / (hi - lo), 2)
    held = [b for b in bars if et <= b["t"] <= (xt or bars[-1]["t"])]
    if held:
        adverse = max(b["h"] for b in held) - spot if d < 0 else spot - min(b["l"] for b in held)
        m["heat_pts"] = round(max(0.0, adverse), 1)
    return m


def _read_samples(since: str) -> list[dict]:
    """Directional READs from pair_shadow: bias vs spot 60 min later (from the ghost's marks)."""
    out = []
    for r in _jsonl(STATE / "pair_shadow.jsonl"):
        if str(r.get("session_date")) < since or r.get("trigger") != "READ":
            continue
        d = 1 if r.get("direction") == "BULLISH" else -1
        t0 = _ts(r.get("opened_at"))
        legs = list((r.get("structures") or {}).values())
        marks = []
        try:     # pair rows don't keep raw marks; fall back to the close
            pass
        finally:
            pass
        s0, sc = r.get("spot"), r.get("spot_at_close")
        bars = _candles(str(r.get("session_date")))
        s60 = _spot_at(bars, t0 + timedelta(minutes=60)) if (bars and t0) else None
        out.append({"date": r.get("session_date"), "t": r.get("opened_at"), "direction": d,
                    "day_type": _day_type(bars), "read_60m": (d * (s60 - s0) > 0) if s60 else None,
                    "read_close": d * (sc - s0) > 0 if (sc and s0) else None})
    return out


def _agg(rows: list[dict]) -> dict:
    def rate(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return (round(sum(1 for x in v if x) / len(v), 2), len(v)) if v else (None, 0)
    def med(k):
        v = [r[k] for r in rows if r.get(k) is not None]
        return (round(st.median(v), 2), len(v)) if v else (None, 0)
    pnl = [r["pnl"] for r in rows if "pnl" in r]
    return {"n": len(rows), "net": round(sum(pnl)) if pnl else None,
            "win_rate": round(sum(1 for x in pnl if x > 0) / len(pnl), 2) if pnl else None,
            "read_60m": rate("read_60m"), "read_close": rate("read_close"), "entry_score": med("entry_score"),
            "exit_capture": med("exit_capture"), "heat_pts": med("heat_pts"), "loss_cut": rate("loss_cut"),
            "exit_vs_hold": (round(sum(r["exit_vs_hold"] for r in rows if r.get("exit_vs_hold") is not None))
                             if any(r.get("exit_vs_hold") is not None for r in rows) else None)}


def evaluate(since: str = DEFAULT_SINCE) -> dict:
    trades = _jsonl(STATE / "intraday_v83_paper_live_trades.jsonl")
    entries = {str(r.get("entry_timestamp") or r.get("timestamp")): r for r in trades if r.get("event") == "PAPER_ENTRY"}
    toclose = {str(r.get("entry_timestamp")): r for r in _jsonl(STATE / "exit_shadow_toclose.jsonl")}
    rows = []
    for x in trades:
        if x.get("event") != "PAPER_EXIT" or x.get("realized_paper_pnl") is None:
            continue
        key = str(x.get("entry_timestamp"))
        if key[:10] < since:
            continue
        en = entries.get(key) or {"strategy": x.get("strategy"), "entry_timestamp": key,
                                  "spot_at_entry": x.get("spot_at_entry")}
        en.setdefault("strategy", x.get("strategy"))
        rows.append(_trade_metrics(en, x, _candles(key[:10]), toclose.get(key)))
    reads = _read_samples(since)

    by_day = defaultdict(list)
    for r in rows:
        by_day[r["day_type"]].append(r)
    reads_by_day = defaultdict(list)
    for r in reads:
        reads_by_day[r["day_type"]].append(r)

    overall = _agg(rows)
    # Bars: overall value must clear the bar, and no day type with >= MIN_CELL samples may fall short.
    checks = {}
    for k, bar in BARS.items():
        val, n = overall[k] if isinstance(overall[k], tuple) else (overall[k], overall["n"])
        weak = [dt for dt, g in by_day.items()
                if (_agg(g)[k][1] >= MIN_CELL and _agg(g)[k][0] is not None and _agg(g)[k][0] < bar)]
        checks[k] = {"value": val, "n": n, "bar": bar, "ok": bool(val is not None and val >= bar and not weak),
                     "weak_day_types": weak}
    res = {"generated_at": datetime.now().isoformat(timespec="seconds"), "since": since, "bars": BARS,
           "min_cell": MIN_CELL, "overall": overall, "by_day_type": {k: _agg(v) for k, v in by_day.items()},
           "reads": {"n": len(reads), "read_60m": _agg(reads)["read_60m"], "read_close": _agg(reads)["read_close"],
                     "by_day_type": {k: _agg(v)["read_60m"] for k, v in reads_by_day.items()}},
           "checks": checks, "skill_ok": all(c["ok"] for c in checks.values()), "trades": rows}
    OUT.write_text(json.dumps(res, indent=2, default=str))
    return res


def _fmt(t):
    return "—" if t is None or t[0] is None else f"{t[0]:.0%}" + f" (n={t[1]})" if isinstance(t, tuple) else str(t)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=DEFAULT_SINCE)
    a = ap.parse_args()
    r = evaluate(a.since)
    o = r["overall"]
    print(f"AGENT SKILL REPORT — trades since {r['since']}  (structure-agnostic)\n")
    print(f"{'':16}{'trades':>7}{'net':>9}{'win':>6}{'read60':>14}{'readEOD':>14}{'entry':>13}{'exitCap':>13}{'heat':>12}{'exit-hold':>10}{'lossCut':>9}")
    def line(lbl, g):
        def p(t, pct=True):
            if t[0] is None:
                return "—"
            return (f"{t[0]:.0%}" if pct else f"{t[0]:g}") + f"/{t[1]}"
        print(f"{lbl:16}{g['n']:>7}{(g['net'] or 0):>9,}{(g['win_rate'] or 0):>6.0%}{p(g['read_60m']):>14}"
              f"{p(g['read_close']):>14}{p(g['entry_score'], False):>13}{p(g['exit_capture'], False):>13}"
              f"{p(g['heat_pts'], False):>12}{(g['exit_vs_hold'] if g['exit_vs_hold'] is not None else '—'):>10}"
              f"{p(g['loss_cut']):>9}")
    line("ALL", o)
    for k, g in sorted(r["by_day_type"].items()):
        line("  " + k, g)
    rd = r["reads"]
    print(f"\nDirectional READS (pair_shadow, every read incl. stand-asides): n={rd['n']}, "
          f"right 60 min later {_fmt(rd['read_60m'])}, right by close {_fmt(rd['read_close'])}")
    for k, v in rd["by_day_type"].items():
        print(f"  {k:12} {_fmt(v)}")
    print("\nSKILL BARS (overall AND every day type with >= %d samples):" % r["min_cell"])
    for k, c in r["checks"].items():
        print(f"  [{'PASS' if c['ok'] else 'FAIL'}] {k:13} {c['value']} (n={c['n']}) vs bar {c['bar']}"
              + (f"  weak on: {', '.join(c['weak_day_types'])}" if c['weak_day_types'] else ""))
    print(f"\n=> skill {'PROVEN' if r['skill_ok'] else 'NOT YET PROVEN'}  -> {OUT}")
    print("\nPer trade:")
    for t in r["trades"]:
        print(f"  {t['entry']}  {str(t['strategy'])[:16]:16} {t['day_type']:10} pnl {t['pnl']:>6}  read60 "
              f"{t.get('read_60m','—')!s:5} entry {t.get('entry_score','—')!s:5} heat {t.get('heat_pts','—')!s:6} "
              f"capture {t.get('exit_capture','—')!s:5} exit {t.get('exit_reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
