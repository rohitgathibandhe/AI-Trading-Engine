"""
Fetch a small set of public event feeds (no API keys) and normalize into the event schema.

Sources:
- RBI press releases RSS
- SEBI press releases RSS
- NSE trading holidays (JSON)

Output:
- Appends JSON lines to state/events.jsonl with normalized fields.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Any, List

import feedparser
import requests

LOG = logging.getLogger("events")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

ROOT = Path(__file__).resolve().parents[2]
STATE_DIR = ROOT / "state"
STATE_DIR.mkdir(parents=True, exist_ok=True)
EVENT_LOG = STATE_DIR / "events.jsonl"


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_events(events: List[Dict[str, Any]]) -> None:
    if not events:
        return
    with EVENT_LOG.open("a", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev, default=str) + "\n")
    LOG.info("Wrote %d events to %s", len(events), EVENT_LOG)


def _parse_rss(url: str, source: str, event_type: str, importance: int = 2) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    feed = feedparser.parse(url)
    for entry in feed.entries[:25]:
        ts = None
        for k in ("published", "updated", "created"):
            if k in entry:
                try:
                    ts = datetime(*entry[k + "_parsed"][:6], tzinfo=timezone.utc).isoformat()
                    break
                except Exception:
                    continue
        if not ts:
            ts = iso_now()
        headline = entry.get("title", "").strip()
        link = entry.get("link")
        out.append(
            {
                "id": f"{source}:{hash(headline)}",
                "timestamp": ts,
                "source": source,
                "event_type": event_type,
                "country": "IN",
                "symbols": ["NIFTY"],
                "headline": headline,
                "details": {"link": link},
                "importance": importance,
                "sentiment": None,
                "raw_payload": {"link": link},
            }
        )
    return out


def fetch_rbi() -> List[Dict[str, Any]]:
    return _parse_rss("https://rbi.org.in/shared/rss/pressreleases.xml", "rbi_rss", "regulatory", importance=3)


def fetch_sebi() -> List[Dict[str, Any]]:
    return _parse_rss("https://www.sebi.gov.in/sebiweb/rss/PRs.xml", "sebi_rss", "regulatory", importance=3)


def fetch_nse_holidays() -> List[Dict[str, Any]]:
    url = "https://www.nseindia.com/api/holiday-master?type=trading"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        js = resp.json()
    except Exception as e:
        LOG.warning("Failed to fetch NSE holidays: %s", e)
        return []
    rows = []
    for row in js.get("data", []):
        try:
            date_str = row.get("tradingDate") or row.get("holidayDate")
            ts = datetime.strptime(date_str, "%d-%b-%Y").date()
            cat = (row.get("description") or "").lower()
            ev = {
                "id": f"nse_holiday:{date_str}",
                "timestamp": datetime.combine(ts, datetime.min.time(), tzinfo=timezone.utc).isoformat(),
                "source": "nse_holiday",
                "event_type": "holiday",
                "country": "IN",
                "symbols": ["NIFTY"],
                "headline": f"Trading holiday: {row.get('description')}",
                "details": row,
                "importance": 3,
                "sentiment": None,
                "raw_payload": row,
            }
            rows.append(ev)
        except Exception:
            continue
    return rows


def main():
    events: List[Dict[str, Any]] = []
    for fn in (fetch_rbi, fetch_sebi, fetch_nse_holidays):
        try:
            events.extend(fn())
        except Exception as e:
            LOG.warning("Source failed: %s", e)
    write_events(events)


if __name__ == "__main__":
    main()
