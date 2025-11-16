from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class EventRecord:
    """Normalized event payload used across collectors."""

    id: str
    timestamp: datetime
    source: str
    event_type: str
    country: str
    symbols: List[str]
    headline: str
    details: Dict[str, Any] = field(default_factory=dict)
    importance: int = 1
    sentiment: Optional[float] = None
    raw_payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "event_type": self.event_type,
            "country": self.country,
            "symbols": self.symbols,
            "headline": self.headline,
            "details": self.details,
            "importance": self.importance,
            "sentiment": self.sentiment,
            "raw_payload": self.raw_payload,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EventRecord":
        ts = data.get("timestamp")
        if isinstance(ts, str):
            timestamp = datetime.fromisoformat(ts)
        elif isinstance(ts, datetime):
            timestamp = ts
        else:
            raise ValueError("timestamp is required")
        return cls(
            id=str(data["id"]),
            timestamp=timestamp,
            source=data.get("source", "unknown"),
            event_type=data.get("event_type", "macro"),
            country=data.get("country", "IN"),
            symbols=list(data.get("symbols", [])),
            headline=data.get("headline", ""),
            details=dict(data.get("details", {})),
            importance=int(data.get("importance", 1)),
            sentiment=data.get("sentiment"),
            raw_payload=dict(data.get("raw_payload", {})),
        )
