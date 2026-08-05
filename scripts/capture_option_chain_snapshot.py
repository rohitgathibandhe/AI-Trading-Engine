#!/usr/bin/env python
"""Capture one full option-chain snapshot from Dhan's /v2/optionchain and append it to a
daily file. Scheduled via launchd every ~3 min during market hours (the Dhan chain API is
1 req / 3s and returns the ENTIRE chain — all strikes with bid/ask + greeks + oi + iv).

This is the FORWARD "gold" data source that fixes both historical gaps at once:
  - depth  : full chain (not the ATM±10 cap of the expired-options rolling endpoint)
  - bid/ask: the rolling/expired endpoint has none; the live chain does.

Design: LOSSLESS — we store the raw API response per snapshot so no field is lost; the
dataset builder parses exact field names later. The script self-guards on market hours and
exits 0 (no error) when the market is closed / the chain is empty, so a naive every-3-min
launchd timer is safe.

Root-cause note: authenticate via state/creds.json (the daily-refreshed token), NOT the
stale DHAN_ACCESS_TOKEN that shell profiles leak into the environment. See
memory/project_data_capture.md.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, time, timezone, timedelta
from pathlib import Path

ENGINE_DIR = Path(__file__).resolve().parents[1] / "data_engine" / "market_ai"
STATE = ENGINE_DIR / "state"
OUT_ROOT = STATE / "rolling_option_live"
IST = timezone(timedelta(hours=5, minutes=30))
INDEX_SECURITY_ID = int(os.getenv("MARKET_AI_INDEX_SECURITY_ID", "13"))
INDEX_SEG = "IDX_I"


def _load_creds_into_env() -> bool:
    try:
        creds = json.loads((STATE / "creds.json").read_text())
    except Exception as exc:  # noqa: BLE001
        print(f"[capture] cannot read creds.json: {exc}", file=sys.stderr)
        return False
    tok, cid = creds.get("access_token"), creds.get("client_id")
    if not tok or not cid:
        print("[capture] creds.json missing token/client_id", file=sys.stderr)
        return False
    os.environ["DHAN_ACCESS_TOKEN"] = str(tok)
    os.environ["DHAN_CLIENT_ID"] = str(cid)
    return True


def _market_open(now_ist: datetime) -> bool:
    if now_ist.weekday() >= 5:  # Sat/Sun
        return False
    # small pre/post pad so we catch the 09:15 open and 15:30 close cleanly
    return time(9, 10) <= now_ist.time() <= time(15, 35)


def main() -> int:
    now_ist = datetime.now(IST)
    if not _market_open(now_ist):
        print(f"[capture] market closed at {now_ist:%Y-%m-%d %H:%M} IST — skip")
        return 0
    if not _load_creds_into_env():
        return 0  # don't error the timer; just skip

    sys.path.insert(0, str(ENGINE_DIR.parent))
    try:
        from market_ai.dhan_wrapper import DhanWrapper
    except Exception as exc:  # noqa: BLE001
        print(f"[capture] import DhanWrapper failed: {exc}", file=sys.stderr)
        return 0

    dw = DhanWrapper(logger=None)
    try:
        expiries = dw.get_optionchain_expirylist(INDEX_SEG, INDEX_SECURITY_ID) or []
    except Exception as exc:  # noqa: BLE001
        print(f"[capture] expirylist failed (market closed?): {exc}")
        return 0
    if not expiries:
        print("[capture] no expiries returned — market likely closed")
        return 0
    # Pick the nearest NON-EXPIRED expiry. The list can still include yesterday's just-expired
    # weekly (e.g. on Wed 07-15 the list starts with the expired Tue 07-14), so filter to
    # expiry >= today; include today itself (0-DTE trades on the Tuesday expiry day).
    today_str = now_ist.date().isoformat()
    future = [e for e in expiries if str(e) >= today_str]
    expiry = future[0] if future else expiries[0]

    try:
        resp = dw.get_option_chain(INDEX_SECURITY_ID, INDEX_SEG, expiry)
    except Exception as exc:  # noqa: BLE001
        print(f"[capture] get_option_chain failed: {exc}")
        return 0

    # Validate non-empty (after-hours returns empty/None)
    oc = (((resp or {}).get("data") or {}).get("data") or {}).get("oc") or {}
    if not oc:
        print("[capture] empty chain (market closed) — skip")
        return 0

    day = now_ist.strftime("%Y-%m-%d")
    out_dir = OUT_ROOT / day
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "chain_snapshots.jsonl"
    record = {
        "captured_at": now_ist.isoformat(),
        "expiry": expiry,
        "security_id": INDEX_SECURITY_ID,
        "seg": INDEX_SEG,
        "response": resp,  # LOSSLESS raw response
    }
    with out_file.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"[capture] wrote snapshot {now_ist:%H:%M:%S} exp={expiry} strikes={len(oc)} -> {out_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
