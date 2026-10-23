#!/usr/bin/env python3
"""LIVE WATCH — deterministic market-hours watcher (launchd, every 60 s). It cannot hang on a permission prompt the way
the scheduled AI session did on 2026-10-23 (stuck at 09:19, nothing watched a -415pt trend day). Telegram on:

  * ENTRY / EXIT of every paper trade (structure, legs, playbook, P&L, exit reason);
  * NO DATA: the agent's last decisions are all INSUFFICIENT_DATA after 09:40 -> if Dhan's chart API is failing
    (DH-905 in the runner log) it starts scripts/candle_recorder.py itself (the fallback built on 10-23);
  * MISSED TREND: trend_day state TREND_UP/DOWN on closed candles for >= 15 min while the agent is flat and has not
    entered for 30 min -> names the blocker the selector is logging (EDGE-ONLY skip / blackout / chop / build fail).

Each alert key is throttled to once per 30 min. State: state/live_watch_state.json.
"""
from __future__ import annotations

import html
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "data_engine" / "market_ai" / "state"
sys.path.insert(0, str(ROOT / "data_engine"))
WSTATE = STATE / ("live_watch_state_dry.json" if __import__("os").environ.get("LIVE_WATCH_DRYRUN") == "1" else "live_watch_state.json")
PY = "/Users/Rohit/.venvs/data_engine_py310/bin/python"


import os
DRY = os.environ.get("LIVE_WATCH_DRYRUN") == "1"          # test mode: print instead of Telegram, no auto-start


def _tg(text: str) -> None:
    if DRY:
        print("WOULD SEND:", text.replace("\n", " | "), flush=True)
        return
    c = json.loads((STATE / "creds.json").read_text())
    try:
        requests.post(f"https://api.telegram.org/bot{c['telegram_bot_token']}/sendMessage",
                      json={"chat_id": c["telegram_chat_id"], "text": html.escape(text)}, timeout=10)
    except Exception:  # noqa: BLE001
        pass
    print(datetime.now().strftime("%H:%M:%S"), text.replace("\n", " | "), flush=True)


def _tail(path: Path, n: int = 400_000) -> list[str]:
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n))
            return f.read().decode("utf-8", "ignore").splitlines()
    except Exception:  # noqa: BLE001
        return []


def _jsonl(path: Path, day: str, now: datetime, n: int = 400_000) -> list[dict]:
    out = []
    for l in _tail(path, n):
        if f'"{day}' not in l:
            continue
        try:
            j = json.loads(l)
        except Exception:  # noqa: BLE001 — a tail can start mid-line
            continue
        ts = str(j.get("timestamp") or "")
        try:
            if ts and datetime.fromisoformat(ts).replace(tzinfo=None) > now:
                continue
        except ValueError:
            pass
        out.append(j)
    return out


def main() -> None:
    now = datetime.fromisoformat(os.environ["LIVE_WATCH_NOW"]) if os.environ.get("LIVE_WATCH_NOW") else datetime.now()
    if now.weekday() >= 5 or not ("0915" <= now.strftime("%H%M") <= "1530"):
        return
    day = now.date().isoformat()
    st = json.loads(WSTATE.read_text()) if WSTATE.exists() else {}
    if st.get("day") != day:
        st = {"day": day, "seen_trades": 0, "last_alert": {}, "trend_since": None}

    def alert(key: str, text: str, every_min: int = 30) -> None:
        last = st["last_alert"].get(key)
        if last and (now - datetime.fromisoformat(last)) < timedelta(minutes=every_min):
            return
        st["last_alert"][key] = now.isoformat()
        _tg(text)

    # 1) trade events
    trades = _jsonl(STATE / "intraday_v83_paper_live_trades.jsonl", day, now)
    for r in trades[st["seen_trades"]:]:
        legs = ", ".join(f"{x.get('action')} {x.get('strike'):.0f}{(x.get('option_type') or '')[:1]}" for x in (r.get("legs") or []))
        if r.get("event") == "PAPER_ENTRY":
            _tg(f"AGENT ENTRY {str(r.get('timestamp'))[11:16]}: {r.get('strategy')} ({r.get('playbook')})\n{legs}")
        elif r.get("event") == "PAPER_EXIT":
            _tg(f"AGENT EXIT {str(r.get('timestamp'))[11:16]}: {r.get('strategy')} P&L Rs {float(r.get('realized_paper_pnl') or 0):,.0f}"
                f" — {r.get('exit_reason')}\n{legs}")
    st["seen_trades"] = len(trades)

    # 2) data health
    decs = _jsonl(STATE / "intraday_v83_paper_live_validation_decisions.jsonl", day, now, 3_000_000 if DRY else 300_000)[-8:]
    if now.strftime("%H%M") >= "0942" and decs and all(d.get("block_reason") == "INSUFFICIENT_DATA" for d in decs):
        log = "\n".join(_tail(STATE / "intraday_v83_runner.log", 60_000)[-200:])
        chart_down = "DH-905" in log or "intraday fetch failure" in log
        started = ""
        if chart_down and not DRY and "candle_recorder.py" not in subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout:
            subprocess.Popen([PY, str(ROOT / "scripts" / "candle_recorder.py")], cwd=str(ROOT),
                             stdout=open(ROOT.parent / "trading_research" / "candle_recorder.log", "a"),
                             stderr=subprocess.STDOUT, start_new_session=True)
            started = "\nAUTO-FIX: started the LTP candle recorder (fallback candles)."
        alert("nodata", f"AGENT HAS NO DATA since {decs[0]['timestamp'][11:16]} (INSUFFICIENT_DATA)."
                        f"{' Dhan chart API failing.' if chart_down else ''}{started}", 15)

    # 3) missed trend
    pos_open = False
    try:
        pos_open = bool(json.loads((STATE / "intraday_v83_paper_state.json").read_text()).get("active_position"))
    except Exception:  # noqa: BLE001
        pass
    try:
        from market_ai.intraday_defined_risk import trend_day as td
        bars = [{"t": datetime.fromisoformat(b["timestamp"]), "o": b["open"], "h": b["high"], "l": b["low"],
                 "c": b["close"], "v": b.get("volume") or 0}
                for b in json.loads((STATE / f"candle_cache_{day}_5m.json").read_text())
                if datetime.fromisoformat(b["timestamp"]) + timedelta(minutes=5) <= now]
        ev = td.evaluate(bars, now)
        state = ev.get("state")
    except Exception:  # noqa: BLE001
        state, bars = None, []
    if state in ("TREND_UP", "TREND_DOWN"):
        st["trend_since"] = st.get("trend_since") or now.isoformat()
    else:
        st["trend_since"] = None
    last_entry = max((datetime.fromisoformat(str(r["timestamp"])).replace(tzinfo=None) for r in trades
                      if r.get("event") == "PAPER_ENTRY"), default=None)
    if (st.get("trend_since") and not pos_open and now - datetime.fromisoformat(st["trend_since"]) >= timedelta(minutes=15)
            and (last_entry is None or now - last_entry >= timedelta(minutes=30))):
        why = Counter()
        _rl = [l for l in _tail(STATE / "intraday_v83_runner.log", 6_000_000 if DRY else 200_000) if l.startswith("{")]
        if DRY:
            _rl = [l for l in _rl if f'"{day}T' in l and l.split('"emitted_at": "')[-1][:19] <= now.isoformat()[:19]]
        for l in _rl[-120:]:
            if not l.startswith("{"):
                continue
            try:
                rat = " ".join(json.loads(l).get("rationale") or [])
            except Exception:  # noqa: BLE001
                continue
            if "EDGE-ONLY" in rat:
                why["edge-only skip: " + rat.split("[EDGE-ONLY:")[-1][:70]] += 1
            elif "blackout" in rat:
                why["debit blackout"] += 1
            elif "efficiency" in rat:
                why["chop/efficiency filter"] += 1
            elif "confluence" in rat:
                why["no confluence"] += 1
        blk = Counter(d.get("block_reason") for d in decs)
        move = (bars[-1]["c"] - bars[0]["o"]) if bars else 0
        alert("trend", f"MISSED TREND? {state} since {st['trend_since'][11:16]}, Nifty {move:+.0f} pts on the day, agent FLAT.\n"
                       f"Blockers: {', '.join(f'{k} x{v}' for k, v in why.most_common(2)) or 'n/a'}; "
                       f"block_reason {dict(blk.most_common(2))}")
    WSTATE.write_text(json.dumps(st))


if __name__ == "__main__":
    main()
