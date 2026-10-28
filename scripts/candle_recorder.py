#!/usr/bin/env python3
"""FALLBACK CANDLE RECORDER — when Dhan's /charts/intraday endpoint fails (2026-10-23: DH-905 on every chart request,
every instrument, while LTP / option chain / profile worked and the data plan was active), build Nifty 5m/15m candles
from Dhan LTP polling and write them to the agent's own disk candle cache (state/candle_cache_<date>_<n>m.json), which
live_runtime._candles falls back to when the live fetch returns nothing.

Before the recorder started, the morning is approximated from the option-chain recorder's spot snapshots (one every
~5 min: O=prev spot, C=spot, H/L=max/min of the two) and marked approx in a side file. Exits by itself as soon as the
real chart endpoint answers again (so it never overwrites real candles) or after 15:31.

  PYTHONPATH=data_engine python scripts/candle_recorder.py              (fallback mode: exits when charts recover)
  PYTHONPATH=data_engine python scripts/candle_recorder.py --standing   (daily from 09:13 via launchd: records LTP
      ticks all session so the opening range is exact; writes candles only while the chart API fails)
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

STATE = Path(__file__).resolve().parents[1] / "data_engine" / "market_ai" / "state"
POLL_S = 5


def _creds():
    c = json.loads((STATE / "creds.json").read_text())
    return {"access-token": c["access_token"], "client-id": str(c["client_id"]),
            "Content-Type": "application/json", "Accept": "application/json"}


def _ltp(h):
    r = requests.post("https://api.dhan.co/v2/marketfeed/ltp", headers=h, json={"IDX_I": [13]}, timeout=8)
    return float(r.json()["data"]["IDX_I"]["13"]["last_price"])


def _chart_ok(h, day):
    r = requests.post("https://api.dhan.co/v2/charts/intraday", headers=h, timeout=10,
                      json={"securityId": "13", "exchangeSegment": "IDX_I", "instrument": "INDEX", "interval": "5",
                            "oi": False, "fromDate": f"{day} 09:15:00", "toDate": f"{day} 15:30:00"})
    return r.status_code == 200 and "timestamp" in (r.json() if r.text.startswith("{") else {})


def _seed(day):
    """(timestamp, price) points from the chain recorder's spot snapshots."""
    pts = []
    f = STATE / "rolling_option_live" / day / "chain_snapshots.jsonl"
    if f.exists():
        for ln in f.open():
            try:
                j = json.loads(ln)
                t = datetime.fromisoformat(j["captured_at"]).replace(tzinfo=None)
                p = float(j["response"]["data"]["data"]["last_price"])
                if t.time() >= datetime.strptime("09:15", "%H:%M").time():
                    pts.append((t, p))
            except Exception:  # noqa: BLE001
                continue
    return pts


def _bars(points, minutes):
    out = {}
    prev = None
    for t, p in sorted(points):
        if t.time() < datetime.strptime("09:15", "%H:%M").time():
            prev = p          # the pre-open price IS the 09:15 open, but must not form a 09:10 bar
            continue          # pre-open ticks (09:13 start) made a 09:10 bar that shifted the opening range (2026-10-28)
        k = t.replace(second=0, microsecond=0)
        k = k - timedelta(minutes=(k.minute - 15) % minutes if minutes == 15 else k.minute % minutes)
        b = out.get(k)
        if b is None:
            o = prev if prev is not None else p
            out[k] = b = {"timestamp": k.isoformat(), "open": o, "high": max(o, p), "low": min(o, p), "close": p, "volume": 0.0}
        b["high"] = max(b["high"], p)
        b["low"] = min(b["low"], p)
        b["close"] = p
        prev = p
    return [out[k] for k in sorted(out)]


def main():
    day = datetime.now().date().isoformat()
    h = _creds()
    seed = _seed(day)
    ticks = []
    tick_file = STATE / f"ltp_ticks_{day}.jsonl"
    if tick_file.exists():
        for ln in tick_file.open():
            try:
                j = json.loads(ln)
                ticks.append((datetime.fromisoformat(j["t"]), float(j["p"])))
            except Exception:  # noqa: BLE001
                pass
    first_tick = ticks[0][0] if ticks else datetime.now()
    (STATE / f"candle_cache_{day}_APPROX_UNTIL.txt").write_text(first_tick.isoformat())
    last_check = 0.0
    print(f"recorder start {datetime.now():%H:%M:%S}; seed points {len(seed)}", flush=True)
    standing = "--standing" in sys.argv          # daily mode: always record ticks, write candles only while charts fail
    chart_down = not standing
    while datetime.now().strftime("%H%M") <= "1531":
        if time.time() - last_check > 60:
            last_check = time.time()
            try:
                ok = _chart_ok(h, day)
                if ok and not standing:
                    print(f"{datetime.now():%H:%M:%S} Dhan chart endpoint is back — recorder exits", flush=True)
                    return
                if ok != (not chart_down):
                    print(f"{datetime.now():%H:%M:%S} Dhan chart endpoint {'OK' if ok else 'FAILING — writing fallback candles'}", flush=True)
                chart_down = not ok
            except Exception:  # noqa: BLE001
                pass
        try:
            p = _ltp(h)
            now = datetime.now()
            ticks.append((now, p))
            with tick_file.open("a") as f:
                f.write(json.dumps({"t": now.isoformat(), "p": p}) + "\n")
        except Exception as e:  # noqa: BLE001
            print(f"ltp fail {e}", flush=True)
        if chart_down:
            pts = [x for x in seed if x[0] < first_tick] + ticks
            for m in (5, 15):
                (STATE / f"candle_cache_{day}_{m}m.json").write_text(json.dumps(_bars(pts, m)))
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
