from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


def _now_ist() -> datetime:
    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    return datetime.now(tz) if tz else datetime.now()


def _json_read(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _jsonl_read(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if isinstance(row, dict):
                rows.append(row)
    except Exception:
        return []
    return rows


def _json_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, default=str) + "\n")


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except Exception:
        return default


def _safe_int(val: Any, default: int = 0) -> int:
    try:
        return int(float(val))
    except Exception:
        return default


@dataclass(frozen=True)
class TuningPaths:
    settings_path: Path
    strategy_state_path: Path
    live_gate_status_path: Path
    live_gate_sessions_path: Path
    advice_path: Path
    history_path: Path


_TUNABLE_BOUNDS: Dict[str, Dict[str, float]] = {
    "batman_bkm_max_credit_pct": {"min": 3.0, "max": 8.0},
    "batman_bkm_tp_pct": {"min": 0.01, "max": 0.04},
    "batman_bkm_sl_pct": {"min": 0.015, "max": 0.05},
    "batman_bkm_balance_tolerance": {"min": 2000.0, "max": 15000.0},
}


def _proposal_id(kind: str, key: str, current: Any, proposed: Any) -> str:
    raw = f"{kind}|{key}|{current}|{proposed}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _clamp_setting(key: str, value: float) -> float:
    bounds = _TUNABLE_BOUNDS.get(key)
    if not bounds:
        return value
    return max(bounds["min"], min(bounds["max"], value))


def _extract_bkm_state_metrics(strategy_state: Dict[str, Any]) -> Dict[str, Any]:
    bkm = strategy_state.get("BATMAN_BKM", {}) if isinstance(strategy_state, dict) else {}
    if not isinstance(bkm, dict):
        bkm = {}
    closed = []
    open_count = 0
    exit_reason_counts: Dict[str, int] = {}
    for _expiry, payload in bkm.items():
        if not isinstance(payload, dict):
            continue
        status = str(payload.get("status") or "").upper()
        if status == "OPEN":
            open_count += 1
        if status != "CLOSED":
            continue
        pnl = payload.get("pnl")
        reason = str(payload.get("reason") or "")
        if reason:
            exit_reason_counts[reason] = exit_reason_counts.get(reason, 0) + 1
        if pnl is not None:
            closed.append(_safe_float(pnl, 0.0))
    closed_count = len(closed)
    wins = sum(1 for x in closed if x > 0)
    return {
        "closed_count": closed_count,
        "open_count": open_count,
        "win_rate": (wins / closed_count) if closed_count else None,
        "avg_pnl": (sum(closed) / closed_count) if closed_count else None,
        "exit_reason_counts": exit_reason_counts,
    }


def _extract_live_gate_metrics(status: Dict[str, Any], sessions: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(sessions)
    fail_reason_counts: Dict[str, int] = {}
    data_failsafe_count = 0
    daily_lock_count = 0
    loop_error_count = 0
    recent_10 = rows[-10:]
    for row in recent_10:
        reasons = row.get("reasons") or []
        if not isinstance(reasons, list):
            reasons = [str(reasons)]
        seen = set()
        for r in reasons:
            rs = str(r or "")
            if not rs:
                continue
            fail_reason_counts[rs] = fail_reason_counts.get(rs, 0) + 1
            if rs in seen:
                continue
            if "DATA_FAILSAFE" in rs:
                data_failsafe_count += 1
            if "DAILY_LOCK_RED" in rs:
                daily_lock_count += 1
            if "AGENT_LOOP_ERROR" in rs:
                loop_error_count += 1
            seen.add(rs)
    sessions_total = _safe_int(status.get("sessions_total"), len(rows))
    sessions_pass = _safe_int(status.get("sessions_pass"), 0)
    sessions_fail = _safe_int(status.get("sessions_fail"), 0)
    pass_rate = (sessions_pass / sessions_total) if sessions_total > 0 else None
    return {
        "sessions_total": sessions_total,
        "sessions_pass": sessions_pass,
        "sessions_fail": sessions_fail,
        "pass_rate": pass_rate,
        "cum_mtm": _safe_float(status.get("cum_mtm"), 0.0),
        "stage": status.get("stage") or "S1",
        "status": status.get("status") or "PROBATION",
        "locked_for_date": status.get("locked_for_date"),
        "hard_lock": bool(status.get("hard_lock", False)),
        "recent_fail_reason_counts": fail_reason_counts,
        "recent_data_failsafe_count": data_failsafe_count,
        "recent_daily_lock_count": daily_lock_count,
        "recent_loop_error_count": loop_error_count,
    }


def _proposal(
    *,
    kind: str,
    key: str,
    current: Any,
    proposed: Any,
    reason: str,
    expected_effect: str,
    risk: str,
    confidence: float,
    guardrails: Optional[List[str]] = None,
) -> Dict[str, Any]:
    return {
        "id": _proposal_id(kind, key, current, proposed),
        "kind": kind,
        "setting_key": key,
        "current_value": current,
        "proposed_value": proposed,
        "reason": reason,
        "expected_effect": expected_effect,
        "risk": risk,
        "confidence": round(float(confidence), 2),
        "guardrails": guardrails or [
            "Manual approval required",
            "Apply only when agent is not running in live mode",
            "Change one proposal at a time and observe at least 5 sessions",
        ],
    }


def generate_advice(paths: TuningPaths) -> Dict[str, Any]:
    settings = _json_read(paths.settings_path)
    strategy_state = _json_read(paths.strategy_state_path)
    live_gate_status = _json_read(paths.live_gate_status_path)
    live_gate_sessions = _jsonl_read(paths.live_gate_sessions_path)

    gate_metrics = _extract_live_gate_metrics(live_gate_status, live_gate_sessions)
    bkm_metrics = _extract_bkm_state_metrics(strategy_state)

    observations: List[str] = []
    proposals: List[Dict[str, Any]] = []
    no_change_reason: Optional[str] = None

    sessions_total = int(gate_metrics["sessions_total"] or 0)
    pass_rate = gate_metrics.get("pass_rate")
    cum_mtm = float(gate_metrics.get("cum_mtm") or 0.0)
    recent_daily_lock_count = int(gate_metrics.get("recent_daily_lock_count") or 0)
    recent_data_failsafe_count = int(gate_metrics.get("recent_data_failsafe_count") or 0)
    recent_loop_error_count = int(gate_metrics.get("recent_loop_error_count") or 0)

    if sessions_total < int(settings.get("live_probation_sessions", 10) or 10):
        no_change_reason = (
            f"Probation still in progress ({sessions_total}/"
            f"{int(settings.get('live_probation_sessions', 10) or 10)} sessions). "
            "Hold strategy parameters steady so rollout metrics remain comparable."
        )
        observations.append("No tuning during live probation unless there is a severe operational issue.")
    else:
        if recent_data_failsafe_count > 0 or recent_loop_error_count > 0:
            no_change_reason = (
                "Recent operational failures detected (data failsafe or loop errors). "
                "Fix infra/broker/feed stability before changing strategy parameters."
            )
            observations.append(
                f"Recent operational faults: data_failsafe={recent_data_failsafe_count}, "
                f"loop_error={recent_loop_error_count} (last 10 sessions)."
            )
        else:
            if recent_daily_lock_count >= 2 and cum_mtm < 0:
                cur_credit = _safe_float(settings.get("batman_bkm_max_credit_pct"), 6.0)
                next_credit = round(_clamp_setting("batman_bkm_max_credit_pct", cur_credit - 0.5), 2)
                if next_credit < cur_credit:
                    proposals.append(
                        _proposal(
                            kind="PARAM_TUNE",
                            key="batman_bkm_max_credit_pct",
                            current=cur_credit,
                            proposed=next_credit,
                            reason=(
                                f"{recent_daily_lock_count} recent DAILY_LOCK_RED failures with negative cum MTM "
                                "suggest entry premium / structure aggressiveness is too high for current conditions."
                            ),
                            expected_effect="Lower entry aggressiveness and reduce tail-risk frequency.",
                            risk="May reduce premium captured and total upside in calm markets.",
                            confidence=0.62,
                        )
                    )
                cur_sl = _safe_float(settings.get("batman_bkm_sl_pct"), 0.025)
                next_sl = round(_clamp_setting("batman_bkm_sl_pct", cur_sl - 0.0025), 4)
                if next_sl < cur_sl:
                    proposals.append(
                        _proposal(
                            kind="PARAM_TUNE",
                            key="batman_bkm_sl_pct",
                            current=cur_sl,
                            proposed=next_sl,
                            reason=(
                                "Repeated daily lockouts indicate losses are reaching the day cap before strategy exits. "
                                "A slightly tighter strategy SL can cut adverse moves earlier."
                            ),
                            expected_effect="Reduce loss per adverse basket and lower probability of hitting day hard stop.",
                            risk="Higher chance of premature exits / whipsaws.",
                            confidence=0.55,
                        )
                    )
            elif (
                sessions_total >= 20
                and pass_rate is not None
                and pass_rate >= 0.8
                and cum_mtm > 0
                and recent_daily_lock_count == 0
            ):
                cur_tp = _safe_float(settings.get("batman_bkm_tp_pct"), 0.02)
                next_tp = round(_clamp_setting("batman_bkm_tp_pct", cur_tp + 0.0025), 4)
                if next_tp > cur_tp:
                    proposals.append(
                        _proposal(
                            kind="PARAM_TUNE",
                            key="batman_bkm_tp_pct",
                            current=cur_tp,
                            proposed=next_tp,
                            reason=(
                                "Performance is stable after probation with no recent daily lock failures. "
                                "A small TP increase is a controlled profit-maximization experiment."
                            ),
                            expected_effect="Capture slightly more profit per winning basket.",
                            risk="Could reduce win rate if exits become too ambitious.",
                            confidence=0.43,
                            guardrails=[
                                "Manual approval required",
                                "Prefer paper test first for 5 sessions",
                                "Apply only when agent is not running in live mode",
                                "Rollback if pass rate drops below 70% over next 5 sessions",
                            ],
                        )
                    )

    if not proposals and no_change_reason is None:
        no_change_reason = "No statistically strong tuning signal found in current data window."

    report = {
        "advisor": "BATMAN_BKM_TUNING_ADVISOR",
        "mode": "PROPOSE_ONLY",
        "manual_approval_required": True,
        "generated_at": _now_ist().isoformat(timespec="seconds"),
        "strategy": "batman_bkm_monthly",
        "summary": {
            "status": "PROPOSALS_READY" if proposals else "NO_CHANGE",
            "proposal_count": len(proposals),
            "no_change_reason": no_change_reason,
        },
        "metrics": {
            "live_gate": gate_metrics,
            "batman_bkm_state": bkm_metrics,
        },
        "observations": observations,
        "proposals": proposals,
        "apply_policy": {
            "requires_manual_approval": True,
            "applies_to": "agent_settings.json",
            "do_not_apply_while_live_running": True,
            "one_change_at_a_time_recommended": True,
        },
    }
    return report


def refresh_advice(paths: TuningPaths) -> Dict[str, Any]:
    report = generate_advice(paths)
    _json_write(paths.advice_path, report)
    return report


def load_or_refresh_advice(paths: TuningPaths) -> Dict[str, Any]:
    current = _json_read(paths.advice_path)
    if current:
        return current
    return refresh_advice(paths)


def apply_proposal(*, paths: TuningPaths, proposal_id: str) -> Dict[str, Any]:
    report = _json_read(paths.advice_path)
    if not report:
        report = refresh_advice(paths)
    if not isinstance(report, dict):
        raise RuntimeError("Invalid advice report payload")
    proposals = report.get("proposals") or []
    if not isinstance(proposals, list):
        raise RuntimeError("Invalid advice report proposals")
    match = None
    for p in proposals:
        if isinstance(p, dict) and str(p.get("id")) == str(proposal_id):
            match = p
            break
    if not match:
        raise RuntimeError(f"Proposal not found: {proposal_id}")
    if str(match.get("kind")) != "PARAM_TUNE":
        raise RuntimeError("Only PARAM_TUNE proposals can be applied")

    key = str(match.get("setting_key") or "")
    if key not in _TUNABLE_BOUNDS:
        raise RuntimeError(f"Setting not allowlisted for auto-apply: {key}")

    proposed_value = match.get("proposed_value")
    current_settings = _json_read(paths.settings_path)
    current_value = current_settings.get(key)
    if current_value != match.get("current_value"):
        raise RuntimeError(
            f"Settings drift detected for {key}: current={current_value!r}, "
            f"expected={match.get('current_value')!r}. Refresh advice first."
        )

    # Apply bounded value.
    if isinstance(proposed_value, (int, float)):
        bounded = _clamp_setting(key, float(proposed_value))
        if isinstance(match.get("current_value"), int) and not isinstance(match.get("current_value"), bool):
            proposed_value = int(round(bounded))
        else:
            proposed_value = round(bounded, 6)

    current_settings[key] = proposed_value
    _json_write(paths.settings_path, current_settings)

    record = {
        "timestamp": _now_ist().isoformat(timespec="seconds"),
        "strategy": "batman_bkm_monthly",
        "proposal_id": match.get("id"),
        "setting_key": key,
        "old_value": match.get("current_value"),
        "new_value": proposed_value,
        "reason": match.get("reason"),
        "source_advice_generated_at": report.get("generated_at"),
    }
    _append_jsonl(paths.history_path, record)
    return {
        "ok": True,
        "applied": record,
        "proposal": match,
        "settings": current_settings,
    }
