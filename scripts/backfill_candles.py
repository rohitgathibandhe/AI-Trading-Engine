#!/usr/bin/env python
"""BACKFILL index candle caches for days missed during a data outage.

Only the INDEX OHLC candles are recoverable: Dhan's charts/intraday endpoint serves historical
candles. The option chain is NOT backfillable — Dhan's option-chain API is a live snapshot with no
historical date parameter — so the shadow book / OI-IV snapshots for the gap days cannot be
reconstructed; they only ever exist if captured live during market hours.

Writes state/candle_cache_<date>_<interval>m.json as a list of
{timestamp, open, high, low, close, volume} rows (the format the agent's cache uses). Skips days the
API returns empty (weekends/holidays). Idempotent — safe to re-run; overwrites the same files.

Run:  python scripts/backfill_candles.py 2026-08-13 2026-08-27      (inclusive date range)
"""
from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import requests

STATE = Path(__file__).resolve().parents[1] / "data_engine" / "market_ai" / "state"
CREDS = STATE / "creds.json"
IST = timezone(timedelta(hours=5, minutes=30))
NIFTY = {"securityId": "13", "exchangeSegment": "IDX_I", "instrument": "INDEX"}
INTERVALS = ("5", "15")


def _headers() -> dict:
    c = json.loads(CREDS.read_text())
    return {"access-token": str(c.get("access_token") or ""), "client-id": str(c.get("client_id") or ""),
            "Content-Type": "application/json"}


def _fetch(h: dict, day: str, interval: str) -> list[dict]:
    r = requests.post("https://api.dhan.co/v2/charts/intraday", headers=h,
                      json={**NIFTY, "interval": interval, "fromDate": day, "toDate": day}, timeout=25)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text[:160]}")
    d = r.json()
    ts = d.get("timestamp") or []
    rows = []
    for i in range(len(ts)):
        rows.append({
            "timestamp": datetime.fromtimestamp(float(ts[i]), IST).strftime("%Y-%m-%dT%H:%M:%S"),
            "open": d["open"][i], "high": d["high"][i], "low": d["low"][i],
            "close": d["close"][i], "volume": d["volume"][i],
        })
    return rows


def _weekdays(start: date, end: date):
    d = start
    while d <= end:
        if d.weekday() < 5:
            yield d
        d += timedelta(days=1)


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: backfill_candles.py <from YYYY-MM-DD> <to YYYY-MM-DD>")
        return 2
    start = date.fromisoformat(sys.argv[1])
    end = date.fromisoformat(sys.argv[2])
    h = _headers()
    filled, empty, failed = [], [], []
    for d in _weekdays(start, end):
        day = d.isoformat()
        got_any = False
        for interval in INTERVALS:
            try:
                rows = _fetch(h, day, interval)
            except Exception as e:  # noqa: BLE001
                failed.append(f"{day}/{interval}m: {e}")
                continue
            if not rows:
                continue
            (STATE / f"candle_cache_{day}_{interval}m.json").write_text(json.dumps(rows))
            got_any = True
            time.sleep(0.4)  # be gentle on the data API
        if got_any:
            filled.append(day)
            print(f"  {day}: backfilled ({INTERVALS[0]}m + {INTERVALS[1]}m)")
        else:
            empty.append(day)
            print(f"  {day}: no candles (holiday / non-trading)")
    print(f"\nBACKFILL DONE — {len(filled)} day(s) filled, {len(empty)} empty, {len(failed)} failed.")
    for f in failed:
        print(f"  FAILED {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
