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


def _parse_dt(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except Exception:
        return None


def _parse_date(value: Any) -> Optional[date]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value)).date()
    except Exception:
        return None


@dataclass
class ExecutionRecoveryConfig:
    enabled: bool = True
    hard_lock_on_issue: bool = True
    journal_lookback_days: int = 45

    @classmethod
    def from_settings(cls, settings: Dict[str, Any]) -> "ExecutionRecoveryConfig":
        return cls(
            enabled=bool(settings.get("live_exec_recovery_enabled", True)),
            hard_lock_on_issue=bool(settings.get("live_exec_recovery_hard_lock", True)),
            journal_lookback_days=max(1, int(settings.get("live_exec_recovery_lookback_days", 45))),
        )


@dataclass
class ExecutionRecoveryState:
    status: str = "OK"  # OK|LOCKED
    hard_lock: bool = False
    current_session_date: str = ""
    locked_for_date: Optional[str] = None
    last_check_at: Optional[str] = None
    last_ok_at: Optional[str] = None
    last_reason: Optional[str] = None
    last_details: Dict[str, Any] = field(default_factory=dict)
    updated_at: str = ""

    @classmethod
    def from_payload(cls, payload: Dict[str, Any]) -> "ExecutionRecoveryState":
        return cls(
            status=str(payload.get("status", "OK")),
            hard_lock=bool(payload.get("hard_lock", False)),
            current_session_date=str(payload.get("current_session_date", "")),
            locked_for_date=payload.get("locked_for_date"),
            last_check_at=payload.get("last_check_at"),
            last_ok_at=payload.get("last_ok_at"),
            last_reason=payload.get("last_reason"),
            last_details=payload.get("last_details") if isinstance(payload.get("last_details"), dict) else {},
            updated_at=str(payload.get("updated_at", "")),
        )

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExecutionJournal:
    def __init__(self, *, journal_path: Path, logger: Any = None) -> None:
        self.journal_path = journal_path
        self.log = logger
        self.journal_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.journal_path.exists():
            try:
                self.journal_path.write_text("")
            except Exception:
                pass

    def _log(self, level: str, msg: str, *args: Any) -> None:
        if self.log and hasattr(self.log, level):
            getattr(self.log, level)(msg, *args)

    def record(self, event_type: str, payload: Optional[Dict[str, Any]] = None, *, when: Optional[datetime] = None) -> None:
        now = when or _now_ist()
        row: Dict[str, Any] = {
            "timestamp": now.isoformat(timespec="seconds"),
            "event_type": str(event_type),
        }
        if payload:
            row.update(payload)
        try:
            with self.journal_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, default=str) + "\n")
        except Exception:
            self._log("exception", "Failed to append execution journal %s", self.journal_path)

    def _load_events(self, *, lookback_days: int, when: Optional[datetime] = None) -> List[Dict[str, Any]]:
        now = when or _now_ist()
        cutoff = now - timedelta(days=max(1, int(lookback_days)))
        out: List[Dict[str, Any]] = []
        if not self.journal_path.exists():
            return out
        try:
            for line in self.journal_path.read_text(encoding="utf-8").splitlines():
                raw = line.strip()
                if not raw:
                    continue
                try:
                    ev = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(ev, dict):
                    continue
                ts = _parse_dt(ev.get("timestamp"))
                if ts and ts < cutoff:
                    continue
                out.append(ev)
        except Exception:
            self._log("exception", "Failed to read execution journal %s", self.journal_path)
        return out

    def analyze_bkm(self, *, lookback_days: int = 45, when: Optional[datetime] = None) -> Dict[str, Any]:
        events = self._load_events(lookback_days=lookback_days, when=when)
        op_state: Dict[str, Dict[str, Any]] = {}
        active_baskets: Dict[str, Dict[str, Any]] = {}
        last_event_at: Optional[str] = None

        for ev in events:
            if str(ev.get("strategy") or "") != "BATMAN_BKM":
                continue
            event_type = str(ev.get("event_type") or "")
            ts = str(ev.get("timestamp") or "")
            if ts:
                last_event_at = ts
            op_id = str(ev.get("op_id") or "").strip()
            if op_id:
                st = op_state.setdefault(
                    op_id,
                    {
                        "op_id": op_id,
                        "event_type": event_type,
                        "begin_event_type": None,
                        "terminal_event_type": None,
                        "status": None,
                        "timestamp": ts,
                        "expiry": ev.get("expiry"),
                    },
                )
                st["timestamp"] = ts or st.get("timestamp")
                st["expiry"] = ev.get("expiry") or st.get("expiry")
                if event_type.endswith("_BEGIN"):
                    st["begin_event_type"] = event_type
                    st["status"] = st.get("status") or "BEGIN"
                    if not st.get("event_type"):
                        st["event_type"] = event_type
                elif event_type.endswith("_SUCCESS"):
                    st["terminal_event_type"] = event_type
                    st["status"] = "SUCCESS"
                elif event_type.endswith("_FAIL"):
                    st["terminal_event_type"] = event_type
                    st["status"] = "FAIL"
                    st["details"] = ev.get("details") if isinstance(ev.get("details"), dict) else {}

            expiry = str(ev.get("expiry") or "").strip()
            if not expiry:
                continue
            if event_type == "BKM_OPEN_SUCCESS":
                active_baskets[expiry] = {
                    "expiry": expiry,
                    "legs": ev.get("legs") if isinstance(ev.get("legs"), list) else [],
                    "meta": ev.get("meta") if isinstance(ev.get("meta"), dict) else {},
                    "opened_at": ts,
                    "source_op_id": op_id or None,
                }
            elif event_type == "BKM_CLOSE_SUCCESS":
                active_baskets.pop(expiry, None)

        unresolved_ops: List[Dict[str, Any]] = []
        failed_ops: List[Dict[str, Any]] = []
        for st in op_state.values():
            if st.get("begin_event_type") and not st.get("terminal_event_type"):
                unresolved_ops.append(st)
            elif st.get("status") == "FAIL":
                failed_ops.append(st)
        unresolved_ops.sort(key=lambda x: str(x.get("timestamp") or ""))
        failed_ops.sort(key=lambda x: str(x.get("timestamp") or ""))
        return {
            "lookback_days": int(lookback_days),
            "events_scanned": len(events),
            "last_event_at": last_event_at,
            "unresolved_ops": unresolved_ops,
            "failed_ops": failed_ops,
            "active_baskets": active_baskets,
        }


class ExecutionRecoveryGuard:
    def __init__(self, *, config: ExecutionRecoveryConfig, status_path: Path, logger: Any = None) -> None:
        self.config = config
        self.status_path = status_path
        self.log = logger
        self.status_path.parent.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        today = _parse_date(self.state.current_session_date) or _now_ist().date()
        self._session_date = today
        self.state.current_session_date = today.isoformat()
        self._persist_state()

    def _default_state(self) -> ExecutionRecoveryState:
        now = _now_ist()
        return ExecutionRecoveryState(
            status="OK",
            hard_lock=False,
            current_session_date=now.date().isoformat(),
            locked_for_date=None,
            last_check_at=None,
            last_ok_at=None,
            last_reason=None,
            last_details={},
            updated_at=now.isoformat(timespec="seconds"),
        )

    def _load_state(self) -> ExecutionRecoveryState:
        if not self.status_path.exists():
            return self._default_state()
        try:
            payload = json.loads(self.status_path.read_text())
            if not isinstance(payload, dict):
                return self._default_state()
            return ExecutionRecoveryState.from_payload(payload)
        except Exception:
            return self._default_state()

    def _persist_state(self) -> None:
        self.state.updated_at = _now_ist().isoformat(timespec="seconds")
        try:
            self.status_path.write_text(json.dumps(self.state.as_dict(), indent=2))
        except Exception:
            if self.log:
                self.log.exception("Failed to write execution recovery status: %s", self.status_path)

    def on_tick(self, when: Optional[datetime] = None) -> None:
        now = when or _now_ist()
        today = now.date()
        if today != self._session_date:
            self._session_date = today
            self.state.current_session_date = today.isoformat()
            if not self.state.hard_lock:
                self.state.status = "OK"
                self.state.locked_for_date = None
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

    def lock(self, reason: str, *, details: Optional[Dict[str, Any]] = None, when: Optional[datetime] = None) -> Dict[str, Any]:
        now = when or _now_ist()
        self.state.status = "LOCKED"
        self.state.last_reason = str(reason)
        self.state.last_details = details if isinstance(details, dict) else {}
        if self.config.hard_lock_on_issue:
            self.state.hard_lock = True
            self.state.locked_for_date = None
        else:
            self.state.locked_for_date = now.date().isoformat()
        self.state.last_check_at = now.isoformat(timespec="seconds")
        self._persist_state()
        return self.snapshot()

    def _leg_key_from_obj(self, leg: Any) -> Optional[str]:
        try:
            expiry = getattr(leg, "expiry", None)
            expiry_s = expiry.isoformat() if hasattr(expiry, "isoformat") else str(expiry or "")
            strike = float(getattr(leg, "strike", 0.0) or 0.0)
            strike_norm = int(round(strike)) if abs(strike - round(strike)) < 1e-6 else round(strike, 2)
            option_type = getattr(leg, "option_type", None)
            opt = option_type.value if hasattr(option_type, "value") else str(option_type or "")
            opt_u = str(opt).upper()
            if opt_u == "CE":
                opt_u = "CALL"
            elif opt_u == "PE":
                opt_u = "PUT"
            side = getattr(leg, "side", None)
            side_s = side.value if hasattr(side, "value") else str(side or "")
            qty = int(getattr(leg, "quantity", 0) or 0)
            return f"{expiry_s}|{opt_u}|{str(side_s).upper()}|{strike_norm}|{qty}"
        except Exception:
            return None

    def _leg_key_from_dict(self, leg: Dict[str, Any]) -> Optional[str]:
        try:
            expiry = str(leg.get("expiry") or "")
            strike = float(leg.get("strike") or 0.0)
            strike_norm = int(round(strike)) if abs(strike - round(strike)) < 1e-6 else round(strike, 2)
            opt = str(leg.get("option_type") or leg.get("opt") or "").upper()
            if opt == "CE":
                opt = "CALL"
            elif opt == "PE":
                opt = "PUT"
            side = str(leg.get("side") or "").upper()
            qty = int(float(leg.get("quantity") or leg.get("qty") or 0))
            return f"{expiry}|{opt}|{side}|{strike_norm}|{qty}"
        except Exception:
            return None

    def _to_counter_obj(self, legs: Iterable[Any]) -> Counter:
        c: Counter = Counter()
        for leg in legs or []:
            key = self._leg_key_from_obj(leg)
            if key:
                c[key] += 1
        return c

    def _to_counter_dict(self, legs: Iterable[Dict[str, Any]]) -> Counter:
        c: Counter = Counter()
        for leg in legs or []:
            if not isinstance(leg, dict):
                continue
            key = self._leg_key_from_dict(leg)
            if key:
                c[key] += 1
        return c

    def evaluate_startup(
        self,
        *,
        local_bkm_state: Dict[str, Any],
        broker_legs: Iterable[Any],
        journal_summary: Dict[str, Any],
        when: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        now = when or _now_ist()
        self.on_tick(now)
        self.state.last_check_at = now.isoformat(timespec="seconds")

        unresolved_ops = journal_summary.get("unresolved_ops") or []
        failed_ops = journal_summary.get("failed_ops") or []
        active_baskets = journal_summary.get("active_baskets") or {}
        if unresolved_ops:
            details = {"unresolved_ops": unresolved_ops[:5], "count": len(unresolved_ops)}
            self.lock("EXEC_JOURNAL_UNRESOLVED_OP", details=details, when=now)
            return {"ok": False, "locked": True, "reason": "EXEC_JOURNAL_UNRESOLVED_OP", "details": details}
        if failed_ops:
            details = {"failed_ops": failed_ops[:5], "count": len(failed_ops)}
            self.lock("EXEC_JOURNAL_FAILED_OP", details=details, when=now)
            return {"ok": False, "locked": True, "reason": "EXEC_JOURNAL_FAILED_OP", "details": details}

        local_open_expiries = sorted(
            str(k)
            for k, v in (local_bkm_state or {}).items()
            if isinstance(v, dict) and str(v.get("status") or "").upper() == "OPEN"
        )
        broker_expiries = sorted(
            {
                (getattr(leg, "expiry").isoformat() if hasattr(getattr(leg, "expiry", None), "isoformat") else str(getattr(leg, "expiry", "")))
                for leg in (broker_legs or [])
            }
        )
        broker_expiries = [e for e in broker_expiries if e]
        journal_expiries = sorted(str(e) for e in active_baskets.keys())

        if len(journal_expiries) > 1:
            details = {"journal_expiries": journal_expiries}
            self.lock("EXEC_RECOVERY_MULTI_EXPIRY_UNSUPPORTED", details=details, when=now)
            return {"ok": False, "locked": True, "reason": "EXEC_RECOVERY_MULTI_EXPIRY_UNSUPPORTED", "details": details}

        if set(local_open_expiries) != set(journal_expiries):
            details = {
                "local_open_expiries": local_open_expiries,
                "journal_active_expiries": journal_expiries,
                "broker_expiries": broker_expiries,
            }
            self.lock("EXEC_RECOVERY_LOCAL_JOURNAL_MISMATCH", details=details, when=now)
            return {"ok": False, "locked": True, "reason": "EXEC_RECOVERY_LOCAL_JOURNAL_MISMATCH", "details": details}

        if set(broker_expiries) != set(journal_expiries):
            details = {
                "local_open_expiries": local_open_expiries,
                "journal_active_expiries": journal_expiries,
                "broker_expiries": broker_expiries,
            }
            self.lock("EXEC_RECOVERY_BROKER_JOURNAL_MISMATCH", details=details, when=now)
            return {"ok": False, "locked": True, "reason": "EXEC_RECOVERY_BROKER_JOURNAL_MISMATCH", "details": details}

        # If flat everywhere, startup is clean.
        if not journal_expiries:
            self.state.status = "OK"
            self.state.last_reason = None
            self.state.last_details = {
                "startup": "CLEAN_FLAT",
                "journal_events_scanned": int(journal_summary.get("events_scanned") or 0),
            }
            self.state.last_ok_at = now.isoformat(timespec="seconds")
            self._persist_state()
            return {"ok": True, "locked": False, "reason": "CLEAN_FLAT", "active_baskets": {}}

        # Exact broker-vs-journal compare for the active expiry.
        active_expiry = journal_expiries[0]
        expected_journal_legs = active_baskets.get(active_expiry, {}).get("legs") or []
        broker_legs_for_expiry = [
            leg
            for leg in (broker_legs or [])
            if (
                (getattr(leg, "expiry").isoformat() if hasattr(getattr(leg, "expiry", None), "isoformat") else str(getattr(leg, "expiry", "")))
                == active_expiry
            )
        ]
        expected_counter = self._to_counter_dict(expected_journal_legs)
        broker_counter = self._to_counter_obj(broker_legs_for_expiry)
        if expected_counter != broker_counter:
            details = {
                "expiry": active_expiry,
                "expected": dict(expected_counter),
                "broker": dict(broker_counter),
                "missing": dict(expected_counter - broker_counter),
                "extra": dict(broker_counter - expected_counter),
            }
            self.lock("EXEC_RECOVERY_BROKER_POSITION_MISMATCH", details=details, when=now)
            return {"ok": False, "locked": True, "reason": "EXEC_RECOVERY_BROKER_POSITION_MISMATCH", "details": details}

        self.state.status = "OK"
        self.state.last_reason = None
        self.state.last_details = {
            "startup": "RESUME_READY",
            "active_expiry": active_expiry,
            "active_legs": sum(expected_counter.values()),
        }
        self.state.last_ok_at = now.isoformat(timespec="seconds")
        self._persist_state()
        return {"ok": True, "locked": False, "reason": "RESUME_READY", "active_baskets": active_baskets}

    def reset(self) -> Dict[str, Any]:
        self.state = self._default_state()
        self._session_date = _now_ist().date()
        self.state.current_session_date = self._session_date.isoformat()
        self._persist_state()
        return self.snapshot()
