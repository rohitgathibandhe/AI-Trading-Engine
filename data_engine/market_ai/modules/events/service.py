from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .schema import EventRecord
from .storage import EventStore


class EventService:
    """Lightweight façade over the EventStore."""

    def __init__(self, store: Optional[EventStore] = None, base_dir: Optional[Path] = None):
        self.store = store or EventStore(base_dir=base_dir)

    def add_events(self, events: Iterable[EventRecord]) -> int:
        return self.store.append(events)

    def events_between(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        *,
        min_importance: int = 1,
        event_types: Optional[Sequence[str]] = None,
        countries: Optional[Sequence[str]] = None,
        symbols: Optional[Sequence[str]] = None,
    ) -> List[EventRecord]:
        return self.store.query(
            start=start,
            end=end,
            min_importance=min_importance,
            event_types=event_types,
            countries=countries,
            symbols=symbols,
        )

    def upcoming_events(
        self,
        window_minutes: int = 60,
        *,
        now: Optional[datetime] = None,
        min_importance: int = 2,
        symbols: Optional[Sequence[str]] = None,
    ) -> List[EventRecord]:
        now = now or datetime.now(timezone.utc)
        end = now + timedelta(minutes=window_minutes)
        return self.events_between(
            start=now,
            end=end,
            min_importance=min_importance,
            symbols=symbols,
        )
