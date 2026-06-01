from __future__ import annotations

import json
import os
from datetime import datetime, time as dt_time
from pathlib import Path
from typing import Any

from .collector import IST


STATE_ROOT = Path(__file__).resolve().parents[1] / "state"
DEFAULT_STRUCTURED_ROOT = STATE_ROOT / "intraday_structured_dataset"
DEFAULT_RESEARCH_ROOT = STATE_ROOT / "intraday_research_dataset_2025_01_01_2026_03_30_5m"
DEFAULT_PIPELINE_STATUS = STATE_ROOT / "intraday_training_pipeline_status.json"
DEFAULT_COLLECTOR_PID = STATE_ROOT / "intraday_chain_collector.pid"
DEFAULT_COLLECTOR_LOG = STATE_ROOT / "intraday_chain_collector.log"
DEFAULT_AGENT_HEARTBEAT = STATE_ROOT / "agent_heartbeat.json"

FRESHNESS_THRESHOLDS_SEC = {
    "structured_5m_dataset": 60.0 * 60.0 * 24.0,
    "structured_option_chain_dataset": 60.0 * 60.0 * 24.0,
    "research_5m_dataset": 60.0 * 60.0 * 48.0,
    "research_option_chain_dataset": 60.0 * 60.0 * 48.0,
    "collector_heartbeat": 60.0 * 10.0,
    "collector_log": 60.0 * 30.0,
    "agent_heartbeat": 60.0 * 2.0,
}

_MARKET_CLOSE_IST = dt_time(15, 30)
_MARKET_OPEN_IST = dt_time(9, 15)


def _market_aware_stale_sec(base_sec: float, now: datetime) -> float:
    """Extend staleness threshold when markets are closed (weekends, pre-open).

    NIFTY data only updates during 09:15–15:30 IST on trading weekdays.
    On weekends and before market open, the staleness clock should be anchored
    to the last expected market-close, not wall-clock time.
    """
    weekday = now.weekday()  # 0=Monday … 6=Sunday
    current_time = now.time().replace(tzinfo=None)
    if weekday == 5:  # Saturday — data from Friday close, ~20-44h ago
        return base_sec + 60.0 * 60.0 * 24.0
    if weekday == 6:  # Sunday — data from Friday close, ~44-68h ago
        return base_sec + 60.0 * 60.0 * 48.0
    # Weekday before market open — data from previous close, ~18h old
    if current_time < _MARKET_OPEN_IST:
        return base_sec + 60.0 * 60.0 * 18.0
    return base_sec


def _now_ist() -> datetime:
    return datetime.now(IST)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _parse_dt(value: object) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=IST)
    return parsed.astimezone(IST)


def _file_check(name: str, path: Path, *, stale_after_sec: float, now: datetime) -> dict[str, Any]:
    if not path.exists():
        return {
            "name": name,
            "path": str(path),
            "exists": False,
            "status": "MISSING",
            "fresh": False,
            "age_sec": None,
            "stale_after_sec": stale_after_sec,
        }
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=IST)
    age_sec = max((now - mtime).total_seconds(), 0.0)
    fresh = age_sec <= stale_after_sec
    return {
        "name": name,
        "path": str(path),
        "exists": True,
        "status": "FRESH" if fresh else "STALE",
        "fresh": fresh,
        "age_sec": round(age_sec, 2),
        "updated_at": mtime.isoformat(),
        "stale_after_sec": stale_after_sec,
        "size_bytes": path.stat().st_size,
    }


def _json_time_check(
    name: str,
    path: Path,
    *,
    stale_after_sec: float,
    timestamp_keys: tuple[str, ...],
    now: datetime,
) -> dict[str, Any]:
    file_check = _file_check(name, path, stale_after_sec=stale_after_sec, now=now)
    if not file_check["exists"]:
        return file_check
    payload = _load_json(path)
    ts = None
    for key in timestamp_keys:
        ts = _parse_dt(payload.get(key))
        if ts is not None:
            break
    if ts is None:
        file_check["status"] = "INVALID"
        file_check["fresh"] = False
        file_check["age_sec"] = None
        file_check["status_reason"] = "TIMESTAMP_MISSING"
        file_check["payload_status"] = payload.get("status")
        file_check["payload_error"] = payload.get("error") or payload.get("fetch_error")
        return file_check
    age_sec = max((now - ts).total_seconds(), 0.0)
    fresh = age_sec <= stale_after_sec
    file_check.update(
        {
            "status": "FRESH" if fresh else "STALE",
            "fresh": fresh,
            "age_sec": round(age_sec, 2),
            "updated_at": ts.isoformat(),
            "payload_status": payload.get("status"),
            "payload_error": payload.get("error") or payload.get("fetch_error"),
            "session_date": payload.get("session_date"),
        }
    )
    return file_check


def _pid_status(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "pid": None, "alive": False, "status": "MISSING"}
    try:
        pid = int(path.read_text().strip())
    except Exception:
        return {"exists": True, "pid": None, "alive": False, "status": "INVALID"}
    try:
        os.kill(pid, 0)
    except PermissionError:
        alive = True
    except OSError:
        alive = False
    else:
        alive = True
    return {
        "exists": True,
        "pid": pid,
        "alive": alive,
        "status": "RUNNING" if alive else "STALE_PID",
    }


def build_data_freshness_report(
    *,
    state_root: str | Path = STATE_ROOT,
    structured_root: str | Path = DEFAULT_STRUCTURED_ROOT,
    research_root: str | Path = DEFAULT_RESEARCH_ROOT,
    now: datetime | None = None,
) -> dict[str, Any]:
    now_ist = (now or _now_ist()).astimezone(IST)
    state_root = Path(state_root)
    structured_root = Path(structured_root)
    research_root = Path(research_root)

    checks = {
        "structured_5m_dataset": _file_check(
            "structured_5m_dataset",
            structured_root / "nifty_5m.csv",
            stale_after_sec=_market_aware_stale_sec(FRESHNESS_THRESHOLDS_SEC["structured_5m_dataset"], now_ist),
            now=now_ist,
        ),
        "structured_option_chain_dataset": _file_check(
            "structured_option_chain_dataset",
            structured_root / "option_chain_decision_times.csv",
            stale_after_sec=_market_aware_stale_sec(FRESHNESS_THRESHOLDS_SEC["structured_option_chain_dataset"], now_ist),
            now=now_ist,
        ),
        "research_5m_dataset": _file_check(
            "research_5m_dataset",
            research_root / "nifty_5m.csv",
            stale_after_sec=_market_aware_stale_sec(FRESHNESS_THRESHOLDS_SEC["research_5m_dataset"], now_ist),
            now=now_ist,
        ),
        "research_option_chain_dataset": _file_check(
            "research_option_chain_dataset",
            research_root / "options_chain.csv",
            stale_after_sec=_market_aware_stale_sec(FRESHNESS_THRESHOLDS_SEC["research_option_chain_dataset"], now_ist),
            now=now_ist,
        ),
        "collector_heartbeat": _json_time_check(
            "collector_heartbeat",
            state_root / DEFAULT_PIPELINE_STATUS.name,
            stale_after_sec=FRESHNESS_THRESHOLDS_SEC["collector_heartbeat"],
            timestamp_keys=("as_of",),
            now=now_ist,
        ),
        "collector_log": _file_check(
            "collector_log",
            state_root / DEFAULT_COLLECTOR_LOG.name,
            stale_after_sec=FRESHNESS_THRESHOLDS_SEC["collector_log"],
            now=now_ist,
        ),
        "agent_heartbeat": _json_time_check(
            "agent_heartbeat",
            state_root / DEFAULT_AGENT_HEARTBEAT.name,
            stale_after_sec=FRESHNESS_THRESHOLDS_SEC["agent_heartbeat"],
            timestamp_keys=("timestamp", "now_ist"),
            now=now_ist,
        ),
    }

    pid_status = _pid_status(state_root / DEFAULT_COLLECTOR_PID.name)
    stale_reasons: list[str] = []
    mapping = {
        "structured_5m_dataset": "STRUCTURED_5M_DATASET_STALE",
        "structured_option_chain_dataset": "STRUCTURED_OPTION_CHAIN_STALE",
        "research_5m_dataset": "RESEARCH_5M_DATASET_STALE",
        "research_option_chain_dataset": "RESEARCH_OPTION_CHAIN_STALE",
        "collector_heartbeat": "COLLECTOR_HEARTBEAT_STALE",
        "collector_log": "COLLECTOR_LOG_STALE",
        "agent_heartbeat": "AGENT_HEARTBEAT_STALE",
    }
    for name, reason in mapping.items():
        if not checks[name]["fresh"]:
            stale_reasons.append(reason)
    collector_error = str(checks["collector_heartbeat"].get("payload_error") or "")
    if "Invalid_Authentication" in collector_error or "invalid or expired" in collector_error or "Authentication Failed" in collector_error:
        stale_reasons.append("COLLECTOR_AUTH_INVALID")
    if not pid_status["alive"]:
        stale_reasons.append("COLLECTOR_PROCESS_NOT_RUNNING")
    if not (state_root / "option_chain_history").exists():
        stale_reasons.append("RAW_OPTION_CHAIN_ROOT_MISSING")

    status = "HEALTHY"
    if stale_reasons:
        status = "STALE"
    elif any(check["status"] == "INVALID" for check in checks.values()):
        status = "DEGRADED"

    return {
        "as_of": now_ist.isoformat(),
        "status": status,
        "fresh": not stale_reasons,
        "stale_reasons": stale_reasons,
        "collector_pid_status": pid_status,
        "checks": checks,
    }


def enrich_status_payload(
    payload: dict[str, Any],
    *,
    state_root: str | Path = STATE_ROOT,
    structured_root: str | Path = DEFAULT_STRUCTURED_ROOT,
    research_root: str | Path = DEFAULT_RESEARCH_ROOT,
    now: datetime | None = None,
) -> dict[str, Any]:
    report = build_data_freshness_report(
        state_root=state_root,
        structured_root=structured_root,
        research_root=research_root,
        now=now,
    )
    enriched = dict(payload)
    enriched["data_freshness"] = report
    enriched["stale_data_block_reasons"] = report["stale_reasons"]
    enriched["data_fresh"] = report["fresh"]
    return enriched


def write_data_freshness_report(
    output_path: str | Path,
    *,
    state_root: str | Path = STATE_ROOT,
    structured_root: str | Path = DEFAULT_STRUCTURED_ROOT,
    research_root: str | Path = DEFAULT_RESEARCH_ROOT,
    now: datetime | None = None,
) -> dict[str, Any]:
    report = build_data_freshness_report(
        state_root=state_root,
        structured_root=structured_root,
        research_root=research_root,
        now=now,
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
