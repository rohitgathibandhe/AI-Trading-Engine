from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Iterable, List, Optional
from zoneinfo import ZoneInfo

import feedparser
import requests

from .schema import EventRecord

LOG = logging.getLogger(__name__)

IN_TZ = ZoneInfo("Asia/Kolkata")
DEFAULT_HEADERS = {
    "User-Agent": "market-ai-event-bot/1.0",
    "Accept": "application/json, text/xml;q=0.9",
    "Connection": "keep-alive",
}


def _request_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(DEFAULT_HEADERS)
    return session


def _parse_date(value: str) -> Optional[datetime]:
    if not value:
        return None
    fmts = ["%d-%b-%Y", "%Y-%m-%d", "%d-%b-%y", "%d %b %Y"]
    for fmt in fmts:
        try:
            return datetime.strptime(value.strip(), fmt).replace(tzinfo=IN_TZ)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value)
    except Exception:
        return None


def collect_nse_holidays(year: Optional[int] = None) -> List[EventRecord]:
    year = year or datetime.now(tz=IN_TZ).year
    url = f"https://www.nseindia.com/api/holiday-master?type=trading&year={year}"
    session = _request_session()
    try:
        session.get("https://www.nseindia.com", timeout=10)
    except requests.RequestException:
        LOG.warning("Could not warm up NSE session")
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        LOG.error("Failed to fetch NSE holidays: %s", exc)
        return []
    try:
        payload = resp.json()
    except json.JSONDecodeError:
        LOG.error("NSE holiday response not JSON")
        return []
    out: List[EventRecord] = []
    segments = payload.get("data") or []
    for block in segments:
        holidays = block.get("holidayList") or []
        for row in holidays:
            datestr = row.get("tradingDate") or row.get("date")
            timestamp = _parse_date(datestr) or datetime(year, 1, 1, tzinfo=IN_TZ)
            headline = row.get("description") or row.get("tradingDate") or "Holiday"
            event_id = f"nse_holiday:{timestamp.date().isoformat()}:{row.get('description','').strip()}"
            out.append(
                EventRecord(
                    id=event_id,
                    timestamp=timestamp,
                    source="nse_holiday_api",
                    event_type="holiday",
                    country="IN",
                    symbols=["NIFTY", "BANKNIFTY"],
                    headline=headline,
                    details={"segment": row.get("tradingSegment"), "category": row.get("category")},
                    importance=3,
                    raw_payload=row,
                )
            )
    return out


def collect_rss_feed(
    url: str,
    *,
    source: str,
    event_type: str,
    country: str = "IN",
    symbols: Optional[List[str]] = None,
    importance: int = 2,
) -> List[EventRecord]:
    symbols = symbols or ["NIFTY"]
    feed = feedparser.parse(url)
    out: List[EventRecord] = []
    for entry in feed.entries:
        published = entry.get("published_parsed") or entry.get("updated_parsed")
        if published:
            ts = datetime.fromtimestamp(time.mktime(published), tz=timezone.utc)
        else:
            ts = datetime.now(tz=timezone.utc)
        event_id = entry.get("id") or entry.get("link") or f"{source}:{ts.timestamp()}"
        headline = entry.get("title", "").strip()
        summary = entry.get("summary", "")
        out.append(
            EventRecord(
                id=f"{source}:{event_id}",
                timestamp=ts,
                source=source,
                event_type=event_type,
                country=country,
                symbols=symbols,
                headline=headline or source,
                details={"summary": summary, "link": entry.get("link")},
                importance=importance,
                raw_payload={k: entry[k] for k in entry.keys()},
            )
        )
    return out


def collect_rbi_press_releases() -> List[EventRecord]:
    return collect_rss_feed(
        "https://www.rbi.org.in/rss/PressReleases.xml",
        source="rbi_press_release",
        event_type="regulatory",
        country="IN",
        symbols=["NIFTY", "BANKNIFTY"],
        importance=3,
    )


def collect_sebi_press_releases() -> List[EventRecord]:
    return collect_rss_feed(
        "https://www.sebi.gov.in/sebiweb/rss/PRs.xml",
        source="sebi_press_release",
        event_type="regulatory",
        country="IN",
        symbols=["NIFTY", "BANKNIFTY"],
        importance=3,
    )


def collect_macro_forexfactory(countries: Optional[List[str]] = None) -> List[EventRecord]:
    countries = countries or ["IN", "US"]
    url = "https://cdn-nfs.faireconomy.media/ff_calendar_thisweek.json"
    session = _request_session()
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        LOG.error("Failed to fetch ForexFactory calendar: %s", exc)
        return []
    try:
        data = resp.json()
    except json.JSONDecodeError:
        LOG.error("ForexFactory payload not JSON")
        return []
    impact_map = {"Low": 1, "Medium": 2, "High": 3, "Holiday": 3}
    out: List[EventRecord] = []
    for row in data:
        country = row.get("country")
        if country not in countries:
            continue
        ts_val = row.get("timestamp") or row.get("date")
        timestamp = None
        if isinstance(ts_val, (int, float)):
            scale = 1000 if ts_val > 1e12 else 1
            timestamp = datetime.fromtimestamp(ts_val / scale, tz=timezone.utc)
        elif isinstance(ts_val, str):
            timestamp = _parse_date(ts_val)
        if timestamp is None:
            continue
        impact = row.get("impact") or row.get("importance") or "Low"
        importance = impact_map.get(str(impact).title(), 1)
        headline = row.get("title") or row.get("event") or "Macro Event"
        symbols = ["NIFTY"]
        if country == "US":
            symbols.append("SPX")
        event_id = f"forexfactory:{row.get('id', '')}:{timestamp.isoformat()}"
        out.append(
            EventRecord(
                id=event_id,
                timestamp=timestamp,
                source="forexfactory",
                event_type="macro",
                country=country or "US",
                symbols=symbols,
                headline=headline,
                details={
                    "actual": row.get("actual"),
                    "forecast": row.get("forecast"),
                    "previous": row.get("previous"),
                    "impact": impact,
                },
                importance=importance,
                raw_payload=row,
            )
        )
    return out


def collect_events(default_symbols: Optional[List[str]] = None) -> List[EventRecord]:
    default_symbols = default_symbols or ["NIFTY", "BANKNIFTY"]
    events: List[EventRecord] = []
    events.extend(collect_nse_holidays())
    for func in (collect_rbi_press_releases, collect_sebi_press_releases):
        try:
            events.extend(func())
        except Exception as exc:
            LOG.error("Collector %s failed: %s", func.__name__, exc)
    try:
        events.extend(collect_macro_forexfactory())
    except Exception as exc:
        LOG.error("Macro collector failed: %s", exc)
    for event in events:
        if not event.symbols:
            event.symbols = default_symbols
    return events
