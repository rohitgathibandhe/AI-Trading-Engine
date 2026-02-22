from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Deque, Dict, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def _now_ist() -> datetime:
    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    return datetime.now(tz) if tz else datetime.now()


@dataclass
class LiveGateConfig:
    enabled: bool = True
    probation_sessions: int = 10
    probation_min_pass: int = 8
    stage1_lot_multiplier: int = 1
    stage2_lot_multiplier: int = 2
    daily_loss_cap_abs: float = 5000.0
    data_fail_zero_spot_count: int = 3
    data_fail_zero_spot_window_sec: int = 120
    data_fail_conn_count: int = 3
    data_fail_conn_window_sec: int = 600
    data_fail_action: str = "AUTO_FLATTEN_LOCK_DAY"
    allow_normal_carry: bool = True
    probation_cum_mtm_floor: float = -15000.0

    @classmethod
    def from_settings(cls, settings: Dict[str, Any]) -> "LiveGateConfig":
        return cls(
            enabled=bool(settings.get("live_gate_enabled", True)),
            probation_sessions=max(1, int(settings.get("live_probation_sessions", 10))),
            probation_min_pass=max(1, int(settings.get("live_probation_min_pass", 8))),
            stage1_lot_multiplier=max(1, int(settings.get("live_stage1_lot_multiplier", 1))),
            stage2_lot_multiplier=max(1, int(settings.get("live_stage2_lot_multiplier", 2))),
            daily_loss_cap_abs=max(1.0, float(settings.get("live_daily_loss_cap_abs", 5000.0))),
            data_fail_zero_spot_count=max(1, int(settings.get("live_data_fail_zero_spot_count", 3))),
            data_fail_zero_spot_window_sec=max(1, int(settings.get("live_data_fail_zero_spot_window_sec", 120))),
            data_fail_conn_count=max(1, int(settings.get("live_data_fail_conn_count", 3))),
            data_fail_conn_window_sec=max(1, int(settings.get("live_data_fail_conn_window_sec", 600))),
            data_fail_action=str(settings.get("live_data_fail_action", "AUTO_FLATTEN_LOCK_DAY")),
            allow_normal_carry=bool(settings.get("live_allow_normal_carry", True)),
            probation_cum_mtm_floor=float(settings.get("live_probation_cum_mtm_floor", -15000.0)),
        )


@dataclass
class LiveGateState:
    status: str = "PROBATION"
    stage: str = "S1"
    lot_multiplier_active: int = 1
    sessions_total: int = 0
    sessions_pass: int = 0
    sessions_fail: int = 0
    cum_mtm: float = 0.0
    current_session_date: str = ""
    last_fail_reason: Optional[str] = None
    locked_for_date: Optional[str] = None
    updated_at: str = ""
    hard_lock: bool = False
    consecutive_failures: int = 0

    @classmethod
    def from_payload(cls, payload: Dict[str, Any], default_lot: int) -> "LiveGateState":
        return cls(
            status=str(payload.get("status", "PROBATION")),
            stage=str(payload.get("stage", "S1")),
            lot_multiplier_active=int(payload.get("lot_multiplier_active", default_lot)),
            sessions_total=int(payload.get("sessions_total", 0)),
            sessions_pass=int(payload.get("sessions_pass", 0)),
            sessions_fail=int(payload.get("sessions_fail", 0)),
            cum_mtm=float(payload.get("cum_mtm", 0.0)),
            current_session_date=str(payload.get("current_session_date", "")),
            last_fail_reason=payload.get("last_fail_reason"),
            locked_for_date=payload.get("locked_for_date"),
            updated_at=str(payload.get("updated_at", "")),
            hard_lock=bool(payload.get("hard_lock", False)),
            consecutive_failures=int(payload.get("consecutive_failures", 0)),
        )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LiveGate:
    def __init__(
        self,
        *,
        config: LiveGateConfig,
        status_path: Path,
        sessions_path: Path,
        logger: Any = None,
    ) -> None:
        self.config = config
        self.status_path = status_path
        self.sessions_path = sessions_path
        self.log = logger

        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.sessions_path.parent.mkdir(parents=True, exist_ok=True)

        self.state = self._load_state()
        self._spot_zero_events: Deque[float] = deque()
        self._conn_fail_events: Deque[float] = deque()

        session_date = self._parse_date(self.state.current_session_date) or _now_ist().date()
        self._session_date: date = session_date
        self._session_has_event = False
        self._session_had_failsafe = False
        self._session_had_daily_lock = False
        self._session_had_loop_error = False
        self._session_last_mtm = 0.0

        self.state.current_session_date = self._session_date.isoformat()
        self._persist_state()

    def _parse_date(self, value: Any) -> Optional[date]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value)).date()
        except Exception:
            return None

    def _epoch(self, when: datetime) -> float:
        try:
            return when.timestamp()
        except Exception:
            return _now_ist().timestamp()

    def _prune(self, queue: Deque[float], window_sec: int, now_ts: float) -> None:
        cutoff = now_ts - float(window_sec)
        while queue and queue[0] < cutoff:
            queue.popleft()

    def _default_state(self) -> LiveGateState:
        now = _now_ist()
        return LiveGateState(
            status="PROBATION",
            stage="S1",
            lot_multiplier_active=self.config.stage1_lot_multiplier,
            sessions_total=0,
            sessions_pass=0,
            sessions_fail=0,
            cum_mtm=0.0,
            current_session_date=now.date().isoformat(),
            last_fail_reason=None,
            locked_for_date=None,
            updated_at=now.isoformat(timespec="seconds"),
            hard_lock=False,
            consecutive_failures=0,
        )

    def _load_state(self) -> LiveGateState:
        if not self.status_path.exists():
            return self._default_state()
        try:
            payload = json.loads(self.status_path.read_text())
            if not isinstance(payload, dict):
                return self._default_state()
            return LiveGateState.from_payload(payload, self.config.stage1_lot_multiplier)
        except Exception:
            return self._default_state()

    def _persist_state(self) -> None:
        self.state.updated_at = _now_ist().isoformat(timespec="seconds")
        try:
            self.status_path.write_text(json.dumps(self.state.as_dict(), indent=2))
        except Exception:
            if self.log:
                self.log.exception("Failed to write live gate status: %s", self.status_path)

    def _append_session(self, payload: Dict[str, Any]) -> None:
        try:
            with self.sessions_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(payload, default=str) + "\n")
        except Exception:
            if self.log:
                self.log.exception("Failed to append live gate session record")

    def snapshot(self) -> Dict[str, Any]:
        return self.state.as_dict()

    def is_hard_locked(self) -> bool:
        return bool(self.state.hard_lock)

    def get_lot_multiplier(self) -> int:
        return int(self.state.lot_multiplier_active or self.config.stage1_lot_multiplier)

    def should_block_entries(self, when: Optional[datetime] = None) -> bool:
        now = when or _now_ist()
        if self.state.hard_lock:
            return True
        if not self.state.locked_for_date:
            return False
        return self.state.locked_for_date == now.date().isoformat()

    def on_tick(self, when: Optional[datetime] = None) -> None:
        now = when or _now_ist()
        today = now.date()
        if today != self._session_date:
            self._finalize_session(self._session_date)
            self._reset_session(today)
            # Day-level lock is only for the session that triggered it.
            if not self.state.hard_lock:
                self.state.locked_for_date = None
                if self.state.stage == "S2":
                    self.state.status = "PROMOTED"
                else:
                    self.state.status = "PROBATION"
            self._persist_state()
        else:
            self.state.current_session_date = today.isoformat()

    def _reset_session(self, new_date: date) -> None:
        self._session_date = new_date
        self._session_has_event = False
        self._session_had_failsafe = False
        self._session_had_daily_lock = False
        self._session_had_loop_error = False
        self._session_last_mtm = 0.0
        self.state.current_session_date = new_date.isoformat()

    def note_live_event(self, *, mtm: Optional[float] = None, when: Optional[datetime] = None) -> None:
        self.on_tick(when)
        self._session_has_event = True
        if mtm is not None:
            try:
                self._session_last_mtm = float(mtm)
            except Exception:
                pass

    def mark_daily_lock(self, reason: str = "DAILY_LOCK_RED", when: Optional[datetime] = None) -> None:
        self.on_tick(when)
        self._session_had_daily_lock = True
        self._session_has_event = True
        self.state.last_fail_reason = reason
        self._persist_state()

    def mark_loop_error(self, reason: str = "AGENT_LOOP_ERROR", when: Optional[datetime] = None) -> None:
        self.on_tick(when)
        self._session_had_loop_error = True
        self._session_has_event = True
        self.state.last_fail_reason = reason
        self._persist_state()

    def register_spot(self, spot: Optional[float], when: Optional[datetime] = None) -> Tuple[bool, str]:
        now = when or _now_ist()
        now_ts = self._epoch(now)
        if spot is not None and float(spot) > 0:
            self._prune(self._spot_zero_events, self.config.data_fail_zero_spot_window_sec, now_ts)
            return False, ""
        self._spot_zero_events.append(now_ts)
        self._prune(self._spot_zero_events, self.config.data_fail_zero_spot_window_sec, now_ts)
        if len(self._spot_zero_events) >= self.config.data_fail_zero_spot_count:
            reason = (
                f"DATA_FAILSAFE_SPOT_ZERO_{len(self._spot_zero_events)}"
                f"_IN_{self.config.data_fail_zero_spot_window_sec}s"
            )
            return True, reason
        return False, ""

    def register_connection_failure(self, when: Optional[datetime] = None) -> Tuple[bool, str]:
        now = when or _now_ist()
        now_ts = self._epoch(now)
        self._conn_fail_events.append(now_ts)
        self._prune(self._conn_fail_events, self.config.data_fail_conn_window_sec, now_ts)
        if len(self._conn_fail_events) >= self.config.data_fail_conn_count:
            reason = (
                f"DATA_FAILSAFE_CONN_{len(self._conn_fail_events)}"
                f"_IN_{self.config.data_fail_conn_window_sec}s"
            )
            return True, reason
        return False, ""

    def trigger_failsafe(self, reason: str, when: Optional[datetime] = None) -> None:
        now = when or _now_ist()
        self.on_tick(now)
        self._session_has_event = True
        self._session_had_failsafe = True
        self._session_had_daily_lock = True
        self.state.status = "LOCKED"
        self.state.locked_for_date = now.date().isoformat()
        self.state.last_fail_reason = reason
        self._persist_state()

    def _finalize_session(self, session_date: date) -> None:
        if not self._session_has_event:
            return

        reasons = []
        if self._session_had_failsafe:
            reasons.append("DATA_FAILSAFE_LOCK")
        if self._session_had_daily_lock:
            reasons.append("DAILY_LOCK_RED")
        if self._session_had_loop_error:
            reasons.append("AGENT_LOOP_ERROR")

        passed = len(reasons) == 0
        result = "PASS" if passed else "FAIL"

        self.state.sessions_total += 1
        self.state.cum_mtm += float(self._session_last_mtm or 0.0)

        if passed:
            self.state.sessions_pass += 1
            self.state.consecutive_failures = 0
        else:
            self.state.sessions_fail += 1
            self.state.consecutive_failures += 1
            self.state.last_fail_reason = reasons[0] if reasons else "UNKNOWN_FAIL"

        self._append_session(
            {
                "session_date": session_date.isoformat(),
                "result": result,
                "mtm_close": float(self._session_last_mtm or 0.0),
                "reasons": reasons,
                "status_before": self.state.status,
                "stage_before": self.state.stage,
                "sessions_total": self.state.sessions_total,
                "sessions_pass": self.state.sessions_pass,
                "sessions_fail": self.state.sessions_fail,
                "cum_mtm": self.state.cum_mtm,
                "timestamp": _now_ist().isoformat(timespec="seconds"),
            }
        )

        # Severe fail escalation: requires manual reset.
        severe_fail = False
        severe_reason = None
        if self.state.consecutive_failures >= 2:
            severe_fail = True
            severe_reason = "LIVE_GATE_SEVERE_2_CONSEC_FAILS"
        if (
            self.state.sessions_total < self.config.probation_sessions
            and self.state.cum_mtm < self.config.probation_cum_mtm_floor
        ):
            severe_fail = True
            severe_reason = "LIVE_GATE_SEVERE_CUM_MTM_FLOOR_BREACH"

        if severe_fail:
            self.state.hard_lock = True
            self.state.status = "LOCKED"
            self.state.locked_for_date = None
            self.state.last_fail_reason = severe_reason
            self.state.stage = "S1"
            self.state.lot_multiplier_active = self.config.stage1_lot_multiplier
            self._persist_state()
            return

        # Promotion gate after probation.
        if (
            self.state.sessions_total >= self.config.probation_sessions
            and self.state.sessions_pass >= self.config.probation_min_pass
            and self.state.cum_mtm >= self.config.probation_cum_mtm_floor
            and not self.state.hard_lock
        ):
            self.state.stage = "S2"
            self.state.status = "PROMOTED"
            self.state.lot_multiplier_active = self.config.stage2_lot_multiplier
        else:
            # Keep stage/status deterministic during probation.
            if self.state.stage != "S2":
                self.state.stage = "S1"
                self.state.status = "PROBATION"
                self.state.lot_multiplier_active = self.config.stage1_lot_multiplier

        self._persist_state()

    def reset(self) -> Dict[str, Any]:
        self.state = self._default_state()
        self._spot_zero_events.clear()
        self._conn_fail_events.clear()
        self._reset_session(_now_ist().date())
        try:
            self.sessions_path.write_text("")
        except Exception:
            if self.log:
                self.log.exception("Failed to clear live gate sessions file")
        self._persist_state()
        return self.snapshot()
