from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def _now_ist() -> datetime:
    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    return datetime.now(tz) if tz else datetime.now()


@dataclass
class PositionReconcilerConfig:
    enabled: bool = True
    grace_sec: int = 30
    mismatch_confirm_count: int = 2
    hard_lock_on_mismatch: bool = True

    @classmethod
    def from_settings(cls, settings: Dict[str, Any]) -> "PositionReconcilerConfig":
        return cls(
            enabled=bool(settings.get("live_reconcile_enabled", True)),
            grace_sec=max(0, int(settings.get("live_reconcile_grace_sec", 30))),
            mismatch_confirm_count=max(1, int(settings.get("live_reconcile_mismatch_confirm_count", 2))),
            hard_lock_on_mismatch=bool(settings.get("live_reconcile_hard_lock", True)),
        )


@dataclass
class PositionReconcilerState:
    status: str = "OK"  # OK|LOCKED
    hard_lock: bool = False
    current_session_date: str = ""
    locked_for_date: Optional[str] = None
    mismatch_streak: int = 0
    check_grace_until: Optional[str] = None
    last_check_at: Optional[str] = None
    last_ok_at: Optional[str] = None
    last_mismatch_reason: Optional[str] = None
    last_diff_summary: Dict[str, Any] = field(default_factory=dict)
    updated_at: str = ""

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "PositionReconcilerState":
        return cls(
            status=str(payload.get("status", "OK")),
            hard_lock=bool(payload.get("hard_lock", False)),
            current_session_date=str(payload.get("current_session_date", "")),
            locked_for_date=payload.get("locked_for_date"),
            mismatch_streak=int(payload.get("mismatch_streak", 0)),
            check_grace_until=payload.get("check_grace_until"),
            last_check_at=payload.get("last_check_at"),
            last_ok_at=payload.get("last_ok_at"),
            last_mismatch_reason=payload.get("last_mismatch_reason"),
            last_diff_summary=payload.get("last_diff_summary") if isinstance(payload.get("last_diff_summary"), dict) else {},
            updated_at=str(payload.get("updated_at", "")),
        )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PositionReconciler:
    def __init__(self, *, config: PositionReconcilerConfig, status_path: Path, logger: Any = None) -> None:
        self.config = config
        self.status_path = status_path
        self.log = logger
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        today = self._parse_date(self.state.current_session_date) or _now_ist().date()
        self._session_date = today
        self.state.current_session_date = today.isoformat()
        self._persist_state()

    def _default_state(self) -> PositionReconcilerState:
        now = _now_ist()
        return PositionReconcilerState(
            status="OK",
            hard_lock=False,
            current_session_date=now.date().isoformat(),
            locked_for_date=None,
            mismatch_streak=0,
            check_grace_until=None,
            last_check_at=None,
            last_ok_at=None,
            last_mismatch_reason=None,
            last_diff_summary={},
            updated_at=now.isoformat(timespec="seconds"),
        )

    def _load_state(self) -> PositionReconcilerState:
        if not self.status_path.exists():
            return self._default_state()
        try:
            payload = json.loads(self.status_path.read_text())
            if not isinstance(payload, dict):
                return self._default_state()
            return PositionReconcilerState.from_payload(payload)
        except Exception:
            return self._default_state()

    def _persist_state(self) -> None:
        self.state.updated_at = _now_ist().isoformat(timespec="seconds")
        try:
            self.status_path.write_text(json.dumps(self.state.as_dict(), indent=2))
        except Exception:
            if self.log:
                self.log.exception("Failed to write reconcile status: %s", self.status_path)

    def _parse_date(self, value: Any) -> Optional[date]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value)).date()
        except Exception:
            return None

    def _parse_dt(self, value: Any) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value))
        except Exception:
            return None

    def on_tick(self, when: Optional[datetime] = None) -> None:
        now = when or _now_ist()
        today = now.date()
        if today != self._session_date:
            self._session_date = today
            self.state.current_session_date = today.isoformat()
            if not self.state.hard_lock:
                self.state.locked_for_date = None
                self.state.status = "OK"
                self.state.mismatch_streak = 0
                self.state.last_mismatch_reason = None
                self.state.last_diff_summary = {}
            self._persist_state()
        else:
            self.state.current_session_date = today.isoformat()

    def snapshot(self) -> Dict[str, Any]:
        return self.state.as_dict()

    def is_locked(self, when: Optional[datetime] = None) -> bool:
        now = when or _now_ist()
        if self.state.hard_lock:
            return True
        if not self.state.locked_for_date:
            return False
        return self.state.locked_for_date == now.date().isoformat()

    def should_block_entries(self, when: Optional[datetime] = None) -> bool:
        return self.is_locked(when)

    def defer_checks(self, *, seconds: Optional[int] = None, reason: str = "", when: Optional[datetime] = None) -> None:
        now = when or _now_ist()
        sec = max(0, int(seconds if seconds is not None else self.config.grace_sec))
        grace_until = now + timedelta(seconds=sec)
        self.state.check_grace_until = grace_until.isoformat(timespec="seconds")
        if reason:
            self.state.last_diff_summary = {"grace_reason": reason, "grace_sec": sec}
        self._persist_state()

    def in_grace_period(self, when: Optional[datetime] = None) -> bool:
        now = when or _now_ist()
        grace_until = self._parse_dt(self.state.check_grace_until)
        if not grace_until:
            return False
        return now < grace_until

    def _leg_key(self, leg: Any) -> Optional[str]:
        try:
            expiry = getattr(leg, "expiry", None)
            expiry_str = expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry or "")
            strike = float(getattr(leg, "strike", 0.0) or 0.0)
            strike_norm = int(round(strike)) if abs(strike - round(strike)) < 1e-6 else round(strike, 2)
            option_type = getattr(leg, "option_type", "")
            opt = option_type.value if hasattr(option_type, "value") else str(option_type)
            side = getattr(leg, "side", "")
            side_s = side.value if hasattr(side, "value") else str(side)
            qty = int(getattr(leg, "quantity", 0) or 0)
            return f"{expiry_str}|{str(opt).upper()}|{str(side_s).upper()}|{strike_norm}|{qty}"
        except Exception:
            return None

    def _to_counter(self, legs: Iterable[Any]) -> Counter:
        c: Counter = Counter()
        for leg in legs or []:
            key = self._leg_key(leg)
            if not key:
                continue
            c[key] += 1
        return c

    def acknowledge_external_positions(
        self,
        *,
        broker_legs: Iterable[Any],
        reason: str = "BROKER_ONLY_EXTERNAL_POSITIONS",
        when: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        now = when or _now_ist()
        self.on_tick(now)
        broker = self._to_counter(broker_legs)
        self.state.status = "OK"
        self.state.hard_lock = False
        self.state.locked_for_date = None
        self.state.mismatch_streak = 0
        self.state.last_mismatch_reason = None
        self.state.last_diff_summary = {
            "match": True,
            "expected_count": 0,
            "broker_count": sum(broker.values()),
            "external": dict(broker),
            "reason": str(reason or "BROKER_ONLY_EXTERNAL_POSITIONS"),
        }
        self.state.last_ok_at = now.isoformat(timespec="seconds")
        self._persist_state()
        return {
            "ok": True,
            "skipped": False,
            "locked": False,
            "reason": str(reason or "BROKER_ONLY_EXTERNAL_POSITIONS"),
            "external_positions_present": bool(broker),
            "diff": self.state.last_diff_summary,
        }

    def evaluate(
        self,
        *,
        expected_legs: Iterable[Any],
        broker_legs: Iterable[Any],
        when: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        now = when or _now_ist()
        self.on_tick(now)
        self.state.last_check_at = now.isoformat(timespec="seconds")

        if self.in_grace_period(now):
            self._persist_state()
            return {"ok": True, "skipped": True, "locked": self.is_locked(now), "reason": "GRACE_PERIOD"}

        if self.state.hard_lock:
            self._persist_state()
            return {"ok": False, "skipped": False, "locked": True, "reason": "HARD_LOCKED"}

        expected = self._to_counter(expected_legs)
        broker = self._to_counter(broker_legs)

        if expected == broker:
            self.state.status = "OK"
            self.state.mismatch_streak = 0
            self.state.last_mismatch_reason = None
            self.state.last_diff_summary = {
                "expected_count": sum(expected.values()),
                "broker_count": sum(broker.values()),
                "match": True,
            }
            self.state.last_ok_at = now.isoformat(timespec="seconds")
            self._persist_state()
            return {"ok": True, "skipped": False, "locked": self.is_locked(now), "reason": "MATCH"}

        missing = expected - broker
        extra = broker - expected
        self.state.mismatch_streak += 1
        self.state.last_mismatch_reason = "BROKER_LOCAL_POSITION_MISMATCH"
        self.state.last_diff_summary = {
            "match": False,
            "expected_count": sum(expected.values()),
            "broker_count": sum(broker.values()),
            "missing": dict(missing),
            "extra": dict(extra),
        }

        locked = False
        if self.state.mismatch_streak >= self.config.mismatch_confirm_count:
            self.state.status = "LOCKED"
            locked = True
            if self.config.hard_lock_on_mismatch:
                self.state.hard_lock = True
                self.state.locked_for_date = None
            else:
                self.state.locked_for_date = now.date().isoformat()
        self._persist_state()
        return {
            "ok": False,
            "skipped": False,
            "locked": locked or self.is_locked(now),
            "reason": "BROKER_LOCAL_POSITION_MISMATCH",
            "mismatch_streak": self.state.mismatch_streak,
            "diff": self.state.last_diff_summary,
        }

    def reset(self) -> Dict[str, Any]:
        self.state = self._default_state()
        self._session_date = _now_ist().date()
        self.state.current_session_date = self._session_date.isoformat()
        self._persist_state()
        return self.snapshot()
