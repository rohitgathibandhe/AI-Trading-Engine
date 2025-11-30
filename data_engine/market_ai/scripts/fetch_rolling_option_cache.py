"""
Incremental rolling option fetcher (Dhan).

Uses DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN from env (or creds.json via start_agent)
to download rolling option data in weekly chunks and store them under
data_engine/market_ai/state/rolling_option/. A manifest keeps track of
what has been fetched so we don't re-download the same window.

Usage:
  python3 data_engine/market_ai/scripts/fetch_rolling_option_cache.py \
      --start 2024-01-01 --end 2024-12-31 --interval 60 --strike ATM
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List

import requests

ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = ROOT / "state"
ROLLING_DIR = STATE_DIR / "rolling_option"
ROLLING_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST = STATE_DIR / "rolling_option_manifest.json"

log = logging.getLogger("fetch_rolling_option_cache")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_manifest() -> Dict[str, Dict[str, str]]:
    if MANIFEST.exists():
        try:
            data = json.loads(MANIFEST.read_text())
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def save_manifest(manifest: Dict[str, Dict[str, str]]) -> None:
    MANIFEST.write_text(json.dumps(manifest, indent=2))


def daterange_chunks(start: date, end: date, chunk_days: int = 7):
    cur = start
    while cur <= end:
        nxt = min(cur + timedelta(days=chunk_days - 1), end)
        yield cur, nxt
        cur = nxt + timedelta(days=1)


def fetch_chunk(
    from_d: date,
    to_d: date,
    option_type: str,
    interval: int,
    strike: str,
    headers: Dict[str, str],
) -> Dict:
    url = "https://api.dhan.co/v2/charts/rollingoption"
    payload = {
        "exchangeSegment": "NSE_FNO",
        "interval": str(interval),
        "securityId": 13,
        "instrument": "OPTIDX",
        "expiryFlag": "MONTH",
        "expiryCode": 1,
        "strike": strike,
        "drvOptionType": option_type,
        "requiredData": ["open", "close", "strike", "oi", "iv", "volume", "spot"],
        "fromDate": from_d.isoformat(),
        "toDate": to_d.isoformat(),
    }
    last_exc = None
    for attempt in range(4):  # initial + 3 retries
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as exc:
            last_exc = exc
            status = exc.response.status_code if exc.response is not None else None
            # backoff on 429/401, otherwise fail fast
            if status in (401, 429):
                sleep_s = 2 ** attempt
                log.warning("Fetch %s-%s %s failed (%s), retrying in %ss", from_d, to_d, option_type, status, sleep_s)
                time.sleep(sleep_s)
                continue
            raise
        except Exception as exc:
            last_exc = exc
            sleep_s = 2 ** attempt
            log.warning("Fetch %s-%s %s failed (%s), retrying in %ss", from_d, to_d, option_type, exc, sleep_s)
            time.sleep(sleep_s)
            continue
    if last_exc:
        raise last_exc
    raise RuntimeError("Fetch failed without exception")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True, help="YYYY-MM-DD")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD")
    ap.add_argument("--interval", type=int, default=60, choices=[1, 5, 15, 25, 60])
    ap.add_argument("--strike", default="ATM")
    args = ap.parse_args()

    cid = os.getenv("DHAN_CLIENT_ID", "").strip()
    tok = os.getenv("DHAN_ACCESS_TOKEN", "").strip()
    if not cid or not tok:
        raise SystemExit("DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN not set")
    headers = {
        "access-token": tok,
        "client-id": cid,
        "Content-Type": "application/json",
    }

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    manifest = load_manifest()

    for opt in ("CALL", "PUT"):
        for frm, to in daterange_chunks(start, end, chunk_days=7):
            key = f"{frm.isoformat()}_{to.isoformat()}_{opt}"
            if key in manifest:
                log.info("Skipping cached window %s", key)
                continue
            try:
                data = fetch_chunk(frm, to, opt, args.interval, args.strike, headers)
            except Exception as exc:
                log.warning("Fetch failed for %s: %s", key, exc)
                continue
            out = ROLLING_DIR / f"rolling_{key}.json"
            out.write_text(json.dumps(data))
            manifest[key] = {
                "from": frm.isoformat(),
                "to": to.isoformat(),
                "option_type": opt,
                "interval": args.interval,
                "strike": args.strike,
                "saved_at": datetime.utcnow().isoformat(),
            }
            save_manifest(manifest)
            log.info("Saved %s", out.name)


if __name__ == "__main__":
    main()
