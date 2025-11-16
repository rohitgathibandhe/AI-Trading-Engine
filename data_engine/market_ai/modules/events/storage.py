from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .schema import EventRecord


class EventStore:
    """Persist events to JSONL (by day) and a SQLite index for fast queries."""

    def __init__(self, base_dir: Optional[Path] = None):
        default_dir = Path(__file__).resolve().parents[2] / "state" / "events"
        self.base_dir = Path(base_dir or default_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.base_dir / "events.db"
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    event_type TEXT,
                    importance INTEGER,
                    country TEXT,
                    symbols TEXT,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp)")

    def append(self, events: Iterable[EventRecord]) -> int:
        events = list(events)
        if not events:
            return 0
        grouped = defaultdict(list)
        for event in events:
            day = event.timestamp.astimezone().date().isoformat()
            grouped[day].append(event)
        for day, rows in grouped.items():
            path = self.base_dir / f"events_{day}.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row.to_dict(), default=str) + "\n")
        with sqlite3.connect(self.db_path) as conn:
            for event in events:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO events
                    (id, timestamp, event_type, importance, country, symbols, payload)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        event.id,
                        event.timestamp.isoformat(),
                        event.event_type,
                        event.importance,
                        event.country,
                        ",".join(event.symbols),
                        json.dumps(event.to_dict(), default=str),
                    ),
                )
            conn.commit()
        return len(events)

    def query(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        min_importance: int = 1,
        event_types: Optional[Sequence[str]] = None,
        countries: Optional[Sequence[str]] = None,
        symbols: Optional[Sequence[str]] = None,
    ) -> List[EventRecord]:
        clauses = []
        params: List[str] = []
        if start:
            clauses.append("timestamp >= ?")
            params.append(start.isoformat())
        if end:
            clauses.append("timestamp <= ?")
            params.append(end.isoformat())
        if event_types:
            placeholders = ",".join("?" for _ in event_types)
            clauses.append(f"event_type IN ({placeholders})")
            params.extend(event_types)
        if countries:
            placeholders = ",".join("?" for _ in countries)
            clauses.append(f"country IN ({placeholders})")
            params.extend(countries)
        clauses.append("importance >= ?")
        params.append(str(min_importance))
        sql = "SELECT payload FROM events"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY timestamp ASC"
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(sql, params).fetchall()
        records = [EventRecord.from_dict(json.loads(payload)) for (payload,) in rows]
        if symbols:
            wanted = set(symbols)
            records = [rec for rec in records if wanted.intersection(rec.symbols)]
        return records
