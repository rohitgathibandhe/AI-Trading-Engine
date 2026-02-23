from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def _now_ist() -> datetime:
    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    return datetime.now(tz) if tz else datetime.now()


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


@dataclass
class HeartbeatConfig:
    interval_sec: float = 10.0

    @classmethod
    def from_settings(cls, settings: Dict[str, Any]) -> "HeartbeatConfig":
        return cls(interval_sec=max(0.5, float(settings.get("ops_heartbeat_interval_sec", 10.0))))


class AgentHeartbeat:
    def __init__(self, *, path: Path, config: HeartbeatConfig, logger: Any = None) -> None:
        self.path = path
        self.config = config
        self.log = logger
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_write_ts: Optional[float] = None

    def _log(self, level: str, msg: str, *args: Any) -> None:
        if self.log and hasattr(self.log, level):
            getattr(self.log, level)(msg, *args)

    def beat(self, payload: Dict[str, Any], *, when: Optional[datetime] = None, force: bool = False) -> bool:
        now = when or _now_ist()
        now_ts = now.timestamp()
        if not force and self._last_write_ts is not None:
            if (now_ts - self._last_write_ts) < float(self.config.interval_sec):
                return False
        row = dict(payload or {})
        row["timestamp"] = now.isoformat(timespec="seconds")
        try:
            self.path.write_text(json.dumps(row, indent=2, default=str))
            self._last_write_ts = now_ts
            return True
        except Exception:
            self._log("exception", "Failed to write heartbeat %s", self.path)
            return False

    def snapshot(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            data = json.loads(self.path.read_text())
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}


@dataclass
class AlertConfig:
    dedupe_sec: float = 60.0

    @classmethod
    def from_settings(cls, settings: Dict[str, Any]) -> "AlertConfig":
        return cls(dedupe_sec=max(0.0, float(settings.get("ops_alert_dedupe_sec", 60.0))))


class AlertJournal:
    def __init__(self, *, path: Path, config: AlertConfig, logger: Any = None) -> None:
        self.path = path
        self.config = config
        self.log = logger
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            try:
                self.path.write_text("")
            except Exception:
                pass
        self._last_emit: Dict[str, float] = {}

    def _log(self, level: str, msg: str, *args: Any) -> None:
        if self.log and hasattr(self.log, level):
            getattr(self.log, level)(msg, *args)

    def emit(
        self,
        *,
        severity: str,
        code: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        source: str = "agent",
        when: Optional[datetime] = None,
        dedupe_key: Optional[str] = None,
        force: bool = False,
    ) -> bool:
        now = when or _now_ist()
        key = dedupe_key or f"{severity}|{code}|{source}"
        now_ts = now.timestamp()
        if not force and self.config.dedupe_sec > 0:
            last_ts = self._last_emit.get(key)
            if last_ts is not None and (now_ts - last_ts) < float(self.config.dedupe_sec):
                return False
        row: Dict[str, Any] = {
            "timestamp": now.isoformat(timespec="seconds"),
            "severity": str(severity).upper(),
            "code": str(code),
            "message": str(message),
            "source": str(source),
        }
        if details:
            row["details"] = details
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
            self._last_emit[key] = now_ts
            return True
        except Exception:
            self._log("exception", "Failed to append alert %s", self.path)
            return False

    def tail(self, limit: int = 100) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []
        out: List[Dict[str, Any]] = []
        for raw in lines[-max(0, int(limit)) :]:
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
            except Exception:
                continue
            if isinstance(payload, dict):
                out.append(payload)
        return out

    def clear(self) -> None:
        try:
            self.path.write_text("")
        except Exception:
            self._log("exception", "Failed to clear alerts %s", self.path)


def compute_watchdog_status(
    *,
    heartbeat_payload: Dict[str, Any],
    stale_after_sec: float,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    now_dt = now or _now_ist()
    stale_sec = max(1.0, float(stale_after_sec))
    if not isinstance(heartbeat_payload, dict) or not heartbeat_payload:
        return {
            "status": "MISSING",
            "stale_after_sec": stale_sec,
            "age_sec": None,
            "last_heartbeat_at": None,
            "heartbeat": {},
        }
    hb_ts = _parse_dt(heartbeat_payload.get("timestamp"))
    age_sec: Optional[float] = None
    if hb_ts is not None:
        age_sec = max(0.0, (now_dt - hb_ts).total_seconds())
    status = "OK"
    if hb_ts is None:
        status = "INVALID"
    elif age_sec is not None and age_sec > stale_sec:
        status = "STALE"
    return {
        "status": status,
        "stale_after_sec": stale_sec,
        "age_sec": round(age_sec, 3) if age_sec is not None else None,
        "last_heartbeat_at": heartbeat_payload.get("timestamp"),
        "heartbeat": heartbeat_payload,
    }

