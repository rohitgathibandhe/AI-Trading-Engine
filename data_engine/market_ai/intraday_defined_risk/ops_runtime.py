from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, time
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

from .collector import IST
from .data_models import (
    DecisionOutput,
    MarketSnapshot,
    OpenPosition,
    OptionType,
    OptionsContractQuote,
    StrategyLeg,
    StrategyType,
    TradeStructure,
)
from .data_pipeline_health import build_data_freshness_report
from .execution import build_open_position, evaluate_exit


STATE_ROOT = Path(__file__).resolve().parents[1] / "state"

V83_LIVE_PLAYBOOKS = {
    # Bearish playbooks
    "SIDEWAYS_TO_BEARISH_REJECTION",
    "GAP_DOWN_BEARISH_CONTINUATION",
    "GAP_UP_BEARISH_FAILURE",
    "HIGH_CONFLUENCE_BEARISH_CONTINUATION",
    "EARLY_BALANCE_BEARISH_FAILED_RECLAIM",
    "BEARISH_CONTINUATION",
    "BEARISH_FAILED_RECLAIM",
    "EARLY_STRUCTURE_BEARISH",
    # Bullish playbooks
    "OPEN_DRIVE_BULLISH",
    "GAP_UP_BULLISH_CONTINUATION",
    "GAP_DOWN_BULLISH_RECOVERY",
    "SIDEWAYS_TO_BULLISH_RECLAIM",
    "HIGH_CONFLUENCE_BULLISH_CONTINUATION",
    "EARLY_BALANCE_BULLISH_RECLAIM",
    "AFTERNOON_TREND_HOLD_BULLISH",
    "EARLY_STRUCTURE_BULLISH",
    # Range / neutral playbooks
    "RANGE_BALANCED_CONDOR",
}
V83_LIVE_STRATEGIES = {
    StrategyType.BEAR_CALL_CREDIT_SPREAD.value,
    StrategyType.BULL_PUT_CREDIT_SPREAD.value,
    StrategyType.IRON_CONDOR.value,
    StrategyType.SHORT_STRANGLE.value,
    StrategyType.SHORT_STRADDLE.value,
}
V83_APPROVED_LIVE_STATES = {"TREND_DOWN", "TREND_UP", "TRANSITION", "DIRECTIONAL_BALANCE", "TRUE_RANGE"}
# Kept for backward-compat — points to the same set
V83_APPROVED_BEARISH_STATES = V83_APPROVED_LIVE_STATES


class RuntimeMode(str, Enum):
    RESEARCH = "RESEARCH"
    SHADOW_LIVE = "SHADOW_LIVE"
    PAPER_LIVE = "PAPER_LIVE"
    MICRO_LIVE = "MICRO_LIVE"
    LIVE_DISABLED = "LIVE_DISABLED"


class HealthStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    BLOCKED = "BLOCKED"


class ReconciliationStatus(str, Enum):
    MATCHED = "MATCHED"
    NO_POSITIONS = "NO_POSITIONS"
    POSITION_MISMATCH = "POSITION_MISMATCH"
    ORPHAN_POSITION = "ORPHAN_POSITION"
    ORPHAN_STATE = "ORPHAN_STATE"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class RuntimeRiskGovernance:
    max_lots_per_trade: int = 6
    max_open_structures: int = 1
    max_trades_per_day: int = 2
    max_daily_realized_loss_rupees: float = 10_000.0
    max_daily_total_loss_rupees: float = 10_000.0
    no_new_entries_after_hhmm: str = "14:30"
    stop_after_first_full_loss: bool = True
    require_manual_live_arm: bool = True
    live_only_bearish: bool = False
    second_trade_requires_first_trade_profit: bool = True
    second_trade_min_gap_minutes: int = 90

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RuntimeRiskGovernance":
        payload = payload or {}
        return cls(
            max_lots_per_trade=max(1, int(float(payload.get("max_lots_per_trade", 1) or 1))),
            max_open_structures=max(1, int(float(payload.get("max_open_structures", 1) or 1))),
            max_trades_per_day=max(1, int(float(payload.get("max_trades_per_day", 2) or 2))),
            max_daily_realized_loss_rupees=max(0.0, float(payload.get("max_daily_realized_loss_rupees", 10_000.0) or 0.0)),
            max_daily_total_loss_rupees=max(0.0, float(payload.get("max_daily_total_loss_rupees", 10_000.0) or 0.0)),
            no_new_entries_after_hhmm=str(payload.get("no_new_entries_after_hhmm") or "14:30"),
            stop_after_first_full_loss=bool(payload.get("stop_after_first_full_loss", True)),
            require_manual_live_arm=bool(payload.get("require_manual_live_arm", True)),
            live_only_bearish=bool(payload.get("live_only_bearish", True)),
            second_trade_requires_first_trade_profit=bool(payload.get("second_trade_requires_first_trade_profit", True)),
            second_trade_min_gap_minutes=max(0, int(float(payload.get("second_trade_min_gap_minutes", 90) or 0))),
        )


@dataclass(frozen=True)
class RuntimeConfig:
    mode: RuntimeMode = RuntimeMode.LIVE_DISABLED
    live_arm: bool = False
    v83_frozen: bool = True
    live_enabled_playbooks: tuple[str, ...] = tuple(sorted(V83_LIVE_PLAYBOOKS))
    live_enabled_strategies: tuple[str, ...] = tuple(sorted(V83_LIVE_STRATEGIES))
    approved_bearish_live_states: tuple[str, ...] = tuple(sorted(V83_APPROVED_BEARISH_STATES))
    bearish_score_margin: float = 1.5
    directional_balance_score_margin_override: float = 0.08
    allowed_entry_start_hhmm: str = "09:30"
    allowed_entry_end_hhmm: str = "14:30"
    allowed_bullish_entry_start_hhmm: str = "10:00"
    paper_override_bearish_score_threshold: float = 6.5
    paper_override_margin: float = 1.0
    paper_experiment_entry_end_hhmm: str = "14:30"
    health_required_for_paper: bool = True
    health_required_for_micro: bool = True
    allowed_playbook_tiers: tuple[str, ...] = ("A", "B")
    circuit_breaker_active: bool = False
    risk: RuntimeRiskGovernance = field(default_factory=RuntimeRiskGovernance)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RuntimeConfig":
        payload = payload or {}
        raw_mode = str(payload.get("mode") or RuntimeMode.LIVE_DISABLED.value).upper()
        try:
            mode = RuntimeMode(raw_mode)
        except ValueError:
            mode = RuntimeMode.LIVE_DISABLED
        approved_states = payload.get("approved_bearish_live_states")
        if approved_states is None:
            approved_states = payload.get("live_approved_market_states")
        return cls(
            mode=mode,
            live_arm=bool(payload.get("live_arm", False)),
            v83_frozen=bool(payload.get("v83_frozen", True)),
            live_enabled_playbooks=tuple(sorted(str(item) for item in payload.get("live_enabled_playbooks", sorted(V83_LIVE_PLAYBOOKS)))),
            live_enabled_strategies=tuple(sorted(str(item) for item in payload.get("live_enabled_strategies", sorted(V83_LIVE_STRATEGIES)))),
            approved_bearish_live_states=tuple(sorted(str(item) for item in approved_states or sorted(V83_APPROVED_BEARISH_STATES))),
            bearish_score_margin=float(payload.get("bearish_score_margin", 1.5) or 1.5),
            directional_balance_score_margin_override=float(
                payload.get(
                    "directional_balance_score_margin_override",
                    (
                        payload.get("risk", {}).get("directional_balance_score_margin_override")
                        if isinstance(payload.get("risk"), dict)
                        else 0.08
                    ),
                )
                or 0.08
            ),
            allowed_entry_start_hhmm=str(payload.get("allowed_entry_start_hhmm") or "09:30"),
            allowed_entry_end_hhmm=str(payload.get("allowed_entry_end_hhmm") or "14:30"),
            allowed_bullish_entry_start_hhmm=str(payload.get("allowed_bullish_entry_start_hhmm") or "10:00"),
            paper_override_bearish_score_threshold=float(payload.get("paper_override_bearish_score_threshold", 6.5) or 6.5),
            paper_override_margin=float(payload.get("paper_override_margin", 1.0) or 1.0),
            paper_experiment_entry_end_hhmm=str(payload.get("paper_experiment_entry_end_hhmm") or "14:30"),
            health_required_for_paper=bool(payload.get("health_required_for_paper", True)),
            health_required_for_micro=bool(payload.get("health_required_for_micro", True)),
            allowed_playbook_tiers=tuple(sorted(set(str(t) for t in payload.get("allowed_playbook_tiers", ["A", "B"])))),
            circuit_breaker_active=bool(payload.get("circuit_breaker_active", False)),
            risk=RuntimeRiskGovernance.from_payload(payload.get("risk") if isinstance(payload.get("risk"), dict) else payload),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        payload["live_approved_market_states"] = list(self.approved_bearish_live_states)
        return payload


@dataclass(frozen=True)
class OpsPaths:
    state_root: Path = STATE_ROOT
    runtime_config: Path = STATE_ROOT / "intraday_v83_runtime_config.json"
    runtime_state: Path = STATE_ROOT / "intraday_v83_runtime_state.json"
    paper_state: Path = STATE_ROOT / "intraday_v83_paper_state.json"
    shadow_decisions: Path = STATE_ROOT / "intraday_v83_shadow_live_decisions.jsonl"
    paper_trades: Path = STATE_ROOT / "intraday_v83_paper_live_trades.jsonl"
    operator_events: Path = STATE_ROOT / "intraday_v83_operator_events.jsonl"
    reconciliation_status: Path = STATE_ROOT / "intraday_v83_reconciliation_status.json"
    reconciliation_events: Path = STATE_ROOT / "intraday_v83_reconciliation_events.jsonl"
    recovery_state: Path = STATE_ROOT / "intraday_v83_recovery_state.json"
    emergency_flatten_events: Path = STATE_ROOT / "intraday_v83_emergency_flatten_events.jsonl"
    validation_decisions: Path = STATE_ROOT / "intraday_v83_paper_live_validation_decisions.jsonl"
    shadow_report: Path = STATE_ROOT / "intraday_v83_shadow_live_report.json"
    paper_report: Path = STATE_ROOT / "intraday_v83_paper_live_report.json"
    validation_report: Path = STATE_ROOT / "intraday_v83_paper_live_validation_report.json"
    paper_context_override_report: Path = STATE_ROOT / "paper_context_override_report.json"
    near_trade_candidates_csv: Path = STATE_ROOT / "near_trade_candidates.csv"
    paper_time_window_audit: Path = STATE_ROOT / "paper_time_window_audit.json"
    operator_status_report: Path = STATE_ROOT / "intraday_v83_operator_status_report.json"
    creds_path: Path = STATE_ROOT / "creds.json"


def _now() -> datetime:
    return datetime.now(IST)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


_JSONL_MAX_LINES: dict[str, int] = {
    "validation_decisions": 2000,
    "operator_events": 1000,
    "reconciliation_events": 500,
    "emergency_flatten_events": 300,
    "shadow_live_decisions": 500,
}
_JSONL_DEFAULT_MAX_LINES = 1000


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    """Append one JSON line and auto-rotate the file if it exceeds its line limit."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")
    # Auto-rotate: keep only the last N lines to prevent unbounded growth.
    stem = path.stem  # e.g. "intraday_v83_paper_live_validation_decisions"
    max_lines = next(
        (v for k, v in _JSONL_MAX_LINES.items() if k in stem),
        _JSONL_DEFAULT_MAX_LINES,
    )
    try:
        # Use a counter pass first to avoid reading the whole file unnecessarily.
        with path.open("r", encoding="utf-8") as f:
            count = sum(1 for ln in f if ln.strip())
        if count > max_lines * 1.25:  # only trim when 25% over limit
            with path.open("r", encoding="utf-8") as f:
                lines = [ln for ln in f if ln.strip()]
            with path.open("w", encoding="utf-8") as f:
                f.writelines(lines[-max_lines:])
    except Exception:
        pass


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _read_jsonl_tail(path: Path, n: int) -> list[dict[str, Any]]:
    """Read only the last *n* valid JSON lines from a JSONL file — much faster
    than reading the whole file when only recent rows are needed."""
    if not path.exists():
        return []
    # Collect raw lines from the end without loading the full file into one string.
    chunk_size = 65536
    rows_raw: list[str] = []
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            remaining = f.tell()
            buf = b""
            while remaining > 0 and len(rows_raw) < n + 1:
                read_size = min(chunk_size, remaining)
                remaining -= read_size
                f.seek(remaining)
                buf = f.read(read_size) + buf
                rows_raw = buf.decode("utf-8", errors="replace").splitlines()
    except Exception:
        return _read_jsonl(path)  # fallback to full read
    result: list[dict[str, Any]] = []
    for line in rows_raw:
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            if isinstance(payload, dict):
                result.append(payload)
        except Exception:
            continue
    return result[-n:]


def _parse_time(value: str, default: time) -> time:
    try:
        hour, minute = str(value).split(":", 1)
        return time(int(hour), int(minute))
    except Exception:
        return default


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


def _file_age_payload(path: Path, stale_after_sec: float) -> dict[str, Any]:
    now = _now()
    if not path.exists():
        return {
            "path": str(path),
            "exists": False,
            "status": HealthStatus.BLOCKED.value,
            "age_sec": None,
            "last_update": None,
            "block_reasons": ["FILE_MISSING"],
        }
    updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=IST)
    age_sec = max((now - updated_at).total_seconds(), 0.0)
    status = HealthStatus.HEALTHY.value if age_sec <= stale_after_sec else HealthStatus.DEGRADED.value
    return {
        "path": str(path),
        "exists": True,
        "status": status,
        "age_sec": round(age_sec, 2),
        "last_update": updated_at.isoformat(),
        "block_reasons": [] if status == HealthStatus.HEALTHY.value else ["FILE_STALE"],
    }


def load_runtime_config(paths: OpsPaths | None = None) -> RuntimeConfig:
    paths = paths or OpsPaths()
    payload = _read_json(paths.runtime_config)
    if not payload:
        config = RuntimeConfig()
        save_runtime_config(config, paths=paths)
        return config
    return RuntimeConfig.from_payload(payload)


def save_runtime_config(config: RuntimeConfig, *, paths: OpsPaths | None = None) -> None:
    paths = paths or OpsPaths()
    payload = config.to_dict()
    payload["updated_at"] = _now().isoformat(timespec="seconds")
    _write_json(paths.runtime_config, payload)


def set_runtime_mode(
    mode: RuntimeMode | str,
    *,
    live_arm: bool | None = None,
    paths: OpsPaths | None = None,
    source: str = "api",
) -> RuntimeConfig:
    paths = paths or OpsPaths()
    current = load_runtime_config(paths)
    new_mode = mode if isinstance(mode, RuntimeMode) else RuntimeMode(str(mode).upper())
    armed = current.live_arm if live_arm is None else bool(live_arm)
    if new_mode != RuntimeMode.MICRO_LIVE:
        armed = False
    config = RuntimeConfig(
        mode=new_mode,
        live_arm=armed,
        v83_frozen=current.v83_frozen,
        live_enabled_playbooks=current.live_enabled_playbooks,
        live_enabled_strategies=current.live_enabled_strategies,
        approved_bearish_live_states=current.approved_bearish_live_states,
        bearish_score_margin=current.bearish_score_margin,
        directional_balance_score_margin_override=current.directional_balance_score_margin_override,
        allowed_entry_start_hhmm=current.allowed_entry_start_hhmm,
        allowed_entry_end_hhmm=current.allowed_entry_end_hhmm,
        allowed_bullish_entry_start_hhmm=current.allowed_bullish_entry_start_hhmm,
        paper_override_bearish_score_threshold=current.paper_override_bearish_score_threshold,
        paper_override_margin=current.paper_override_margin,
        paper_experiment_entry_end_hhmm=current.paper_experiment_entry_end_hhmm,
        health_required_for_paper=current.health_required_for_paper,
        health_required_for_micro=current.health_required_for_micro,
        allowed_playbook_tiers=current.allowed_playbook_tiers,
        risk=current.risk,
    )
    save_runtime_config(config, paths=paths)
    _append_jsonl(
        paths.operator_events,
        {
            "timestamp": _now().isoformat(timespec="seconds"),
            "event": "RUNTIME_MODE_CHANGED",
            "source": source,
            "mode": config.mode.value,
            "live_arm": config.live_arm,
        },
    )
    return config


def _default_runtime_state(config: RuntimeConfig) -> dict[str, Any]:
    now = _now()
    return {
        "mode": config.mode.value,
        "live_arm": config.live_arm,
        "session_date": now.date().isoformat(),
        "daily_lock": {"active": False, "reason": None, "locked_at": None},
        "session_trade_count": 0,
        "trade_sequence": [],
        "realized_pnl_rupees": 0.0,
        "total_pnl_rupees": 0.0,
        "primary_block_reason": "MODE_NOT_EVALUATED",
        "last_order": None,
        "last_simulated_order": None,
        "last_exit_reason": None,
        "last_candidate": None,
        "last_decision_at": None,
        "updated_at": now.isoformat(timespec="seconds"),
    }


def load_runtime_state(paths: OpsPaths | None = None, config: RuntimeConfig | None = None) -> dict[str, Any]:
    paths = paths or OpsPaths()
    config = config or load_runtime_config(paths)
    state = _read_json(paths.runtime_state)
    if not state:
        state = _default_runtime_state(config)
        _write_json(paths.runtime_state, state)
    today = _now().date().isoformat()
    if state.get("session_date") != today:
        state.update(_default_runtime_state(config))
        _write_json(paths.runtime_state, state)
    return state


def save_runtime_state(state: dict[str, Any], *, paths: OpsPaths | None = None) -> None:
    paths = paths or OpsPaths()
    payload = dict(state)
    payload["updated_at"] = _now().isoformat(timespec="seconds")
    _write_json(paths.runtime_state, payload)


def _health_component(
    name: str,
    status: HealthStatus | str,
    *,
    last_update: str | None = None,
    age_sec: float | None = None,
    block_reasons: Iterable[str] = (),
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    status_value = status.value if isinstance(status, HealthStatus) else str(status)
    return {
        "name": name,
        "status": status_value,
        "last_update": last_update,
        "age_sec": round(float(age_sec), 2) if age_sec is not None else None,
        "block_reasons": list(block_reasons),
        "details": details or {},
    }


def _snapshot_age(snapshot: MarketSnapshot | None) -> float | None:
    if snapshot is None:
        return None
    ts = snapshot.timestamp
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=IST)
    return max((_now() - ts.astimezone(IST)).total_seconds(), 0.0)


def build_unified_health(
    *,
    config: RuntimeConfig | None = None,
    paths: OpsPaths | None = None,
    snapshot: MarketSnapshot | None = None,
) -> dict[str, Any]:
    paths = paths or OpsPaths()
    config = config or load_runtime_config(paths)
    now = _now()
    components: dict[str, dict[str, Any]] = {}

    creds = _read_json(paths.creds_path)
    if str(creds.get("client_id") or "").strip() and str(creds.get("access_token") or "").strip():
        components["broker_auth_health"] = _health_component(
            "broker_auth_health",
            HealthStatus.HEALTHY,
            last_update=now.isoformat(timespec="seconds"),
        )
    else:
        components["broker_auth_health"] = _health_component(
            "broker_auth_health",
            HealthStatus.BLOCKED,
            last_update=now.isoformat(timespec="seconds"),
            block_reasons=["BROKER_AUTH_MISSING"],
        )

    snap_age = _snapshot_age(snapshot)
    if snapshot is not None and snap_age is not None:
        market_status = HealthStatus.HEALTHY if snap_age <= 900.0 else HealthStatus.DEGRADED
        components["market_feed_health"] = _health_component(
            "market_feed_health",
            market_status,
            last_update=snapshot.timestamp.isoformat(),
            age_sec=snap_age,
            block_reasons=[] if market_status == HealthStatus.HEALTHY else ["MARKET_FEED_STALE"],
            details={"source": "live_snapshot"},
        )
        chain_age = snap_age
        chain_status = HealthStatus.HEALTHY if chain_age <= 900.0 else HealthStatus.BLOCKED
        components["option_chain_health"] = _health_component(
            "option_chain_health",
            chain_status,
            last_update=snapshot.option_chain.timestamp.isoformat(),
            age_sec=chain_age,
            block_reasons=[] if chain_status == HealthStatus.HEALTHY else ["OPTION_CHAIN_STALE"],
            details={"source": "live_snapshot"},
        )
    else:
        market_payload = _file_age_payload(paths.state_root / "intraday_structured_dataset" / "nifty_5m.csv", 86_400.0)
        components["market_feed_health"] = _health_component(
            "market_feed_health",
            market_payload["status"],
            last_update=market_payload.get("last_update"),
            age_sec=market_payload.get("age_sec"),
            block_reasons=market_payload.get("block_reasons") or [],
            details={"source": "structured_dataset", "path": market_payload.get("path")},
        )
        chain_payload = _file_age_payload(paths.state_root / "intraday_structured_dataset" / "option_chain_decision_times.csv", 86_400.0)
        components["option_chain_health"] = _health_component(
            "option_chain_health",
            chain_payload["status"],
            last_update=chain_payload.get("last_update"),
            age_sec=chain_payload.get("age_sec"),
            block_reasons=chain_payload.get("block_reasons") or [],
            details={"source": "structured_dataset", "path": chain_payload.get("path")},
        )

    reconcile = _read_json(paths.reconciliation_status)
    rec_status = str(reconcile.get("status") or ReconciliationStatus.UNKNOWN.value)
    if rec_status in {ReconciliationStatus.MATCHED.value, ReconciliationStatus.NO_POSITIONS.value}:
        sync_status = HealthStatus.HEALTHY
        sync_reasons: list[str] = []
    elif rec_status == ReconciliationStatus.UNKNOWN.value:
        sync_status = HealthStatus.DEGRADED
        sync_reasons = ["BROKER_POSITION_SYNC_UNKNOWN"]
    else:
        sync_status = HealthStatus.BLOCKED
        sync_reasons = [rec_status]
    components["broker_position_sync_health"] = _health_component(
        "broker_position_sync_health",
        sync_status,
        last_update=reconcile.get("updated_at"),
        block_reasons=sync_reasons,
        details=reconcile,
    )

    try:
        paths.state_root.mkdir(parents=True, exist_ok=True)
        probe = paths.state_root / ".intraday_v83_state_store_probe"
        probe.write_text(now.isoformat(timespec="seconds"))
        probe.unlink(missing_ok=True)
        components["state_store_health"] = _health_component("state_store_health", HealthStatus.HEALTHY, last_update=now.isoformat(timespec="seconds"))
    except Exception as exc:  # noqa: BLE001
        components["state_store_health"] = _health_component(
            "state_store_health",
            HealthStatus.BLOCKED,
            last_update=now.isoformat(timespec="seconds"),
            block_reasons=["STATE_STORE_WRITE_FAILED"],
            details={"error": str(exc)},
        )

    if config.v83_frozen and set(config.live_enabled_playbooks).issubset(V83_LIVE_PLAYBOOKS) and set(config.live_enabled_strategies).issubset(V83_LIVE_STRATEGIES):
        components["strategy_engine_health"] = _health_component(
            "strategy_engine_health",
            HealthStatus.HEALTHY,
            last_update=now.isoformat(timespec="seconds"),
            details={"v83_frozen": True},
        )
    else:
        components["strategy_engine_health"] = _health_component(
            "strategy_engine_health",
            HealthStatus.BLOCKED,
            last_update=now.isoformat(timespec="seconds"),
            block_reasons=["V83_FREEZE_OR_ALLOWED_BOOK_VIOLATION"],
            details={"v83_frozen": config.v83_frozen},
        )

    pipeline = build_data_freshness_report(state_root=paths.state_root)
    pipeline_reasons = list(pipeline.get("stale_reasons") or [])
    degraded_only_pipeline_reasons = {
        "COLLECTOR_LOG_STALE",
        "RESEARCH_5M_DATASET_STALE",
        "RESEARCH_OPTION_CHAIN_STALE",
    }
    hard_pipeline_reasons = [reason for reason in pipeline_reasons if reason not in degraded_only_pipeline_reasons]
    pipeline_checks = pipeline.get("checks") if isinstance(pipeline.get("checks"), dict) else {}
    structured_5m_ok = bool((pipeline_checks.get("structured_5m_dataset") or {}).get("fresh"))
    structured_chain_ok = bool((pipeline_checks.get("structured_option_chain_dataset") or {}).get("fresh"))
    collector_pid = pipeline.get("collector_pid_status") if isinstance(pipeline.get("collector_pid_status"), dict) else {}
    before_first_collector_capture = now.time() < time(9, 30)
    active_market_session = time(9, 15) <= now.time() <= time(15, 15)
    live_feed_ready = components.get("market_feed_health", {}).get("status") == HealthStatus.HEALTHY.value
    live_chain_ready = components.get("option_chain_health", {}).get("status") == HealthStatus.HEALTHY.value
    if (
        "COLLECTOR_HEARTBEAT_STALE" in hard_pipeline_reasons
        and bool(collector_pid.get("alive"))
        and before_first_collector_capture
        and structured_5m_ok
        and structured_chain_ok
    ):
        # The monitor intentionally does not write a same-day capture heartbeat before
        # the first 09:30 decision snapshot. Treat this as degraded pre-market
        # readiness, not a paper-live hard block.
        hard_pipeline_reasons = [reason for reason in hard_pipeline_reasons if reason != "COLLECTOR_HEARTBEAT_STALE"]
    if (
        config.mode == RuntimeMode.PAPER_LIVE
        and active_market_session
        and live_feed_ready
        and live_chain_ready
    ):
        # PAPER_LIVE intraday entries use the live market feed and current option chain.
        # Structured/research dataset files and the async collector are not part of the
        # live decision path. Demote all of these to degraded-only so that stale training
        # artifacts or a dead collector cannot hard-block paper trades when live data is healthy.
        paper_session_degraded_only = {
            "STRUCTURED_5M_DATASET_STALE",
            "STRUCTURED_OPTION_CHAIN_STALE",
            "COLLECTOR_HEARTBEAT_STALE",
            "COLLECTOR_PROCESS_NOT_RUNNING",
            "COLLECTOR_LOG_STALE",
        }
        hard_pipeline_reasons = [reason for reason in hard_pipeline_reasons if reason not in paper_session_degraded_only]
    if pipeline.get("fresh"):
        pipeline_status = HealthStatus.HEALTHY
    elif hard_pipeline_reasons:
        pipeline_status = HealthStatus.BLOCKED
    else:
        pipeline_status = HealthStatus.DEGRADED
    components["data_pipeline_health"] = _health_component(
        "data_pipeline_health",
        pipeline_status,
        last_update=pipeline.get("as_of"),
        block_reasons=hard_pipeline_reasons if pipeline_status == HealthStatus.BLOCKED else pipeline_reasons,
        details=pipeline,
    )

    block_reasons: list[str] = []
    if any(component["status"] == HealthStatus.BLOCKED.value for component in components.values()):
        status = HealthStatus.BLOCKED
    elif any(component["status"] == HealthStatus.DEGRADED.value for component in components.values()):
        status = HealthStatus.DEGRADED
    else:
        status = HealthStatus.HEALTHY
    for name, component in components.items():
        for reason in component.get("block_reasons") or []:
            block_reasons.append(f"{name}:{reason}")
    return {
        "status": status.value,
        "as_of": now.isoformat(timespec="seconds"),
        "components": components,
        "block_reasons": block_reasons,
    }


def _decision_metadata(decision: DecisionOutput) -> dict[str, Any]:
    return decision.metadata if isinstance(decision.metadata, dict) else {}


def _trade_funnel(decision: DecisionOutput) -> dict[str, Any]:
    metadata = _decision_metadata(decision)
    funnel = metadata.get("trade_funnel")
    return funnel if isinstance(funnel, dict) else {}


def _primary_block(reasons: list[str]) -> str:
    return reasons[0] if reasons else "NONE"


def _decision_origin(decision: DecisionOutput) -> str | None:
    metadata = _decision_metadata(decision)
    value = metadata.get("paper_trade_attribution") or metadata.get("decision_origin")
    if value:
        return str(value)
    return None


def _is_bullish_context(metadata: dict[str, Any], funnel: dict[str, Any]) -> bool:
    direction = str(metadata.get("setup_direction") or funnel.get("setup_direction") or "").upper()
    state_bias = str(metadata.get("market_state_bias") or funnel.get("market_state_bias") or "").upper()
    playbook = str(metadata.get("playbook") or funnel.get("playbook") or "").upper()
    return (
        direction == "BULLISH"
        or state_bias == "BULLISH"
        or playbook.startswith("OPEN_DRIVE_BULLISH")
        or playbook.startswith("GAP_UP_BULLISH")
        or playbook.startswith("HIGH_CONFLUENCE_BULLISH")
        or playbook.startswith("LATE_SESSION_BULLISH")
        or playbook.startswith("SIDEWAYS_TO_BULLISH")
    )


def _active_paper_position_count(paths: OpsPaths) -> int:
    state = _read_json(paths.paper_state)
    active = state.get("active_position") if isinstance(state, dict) else None
    return 1 if isinstance(active, dict) and active else 0


def _has_recovery_block(paths: OpsPaths) -> tuple[bool, str | None]:
    recovery = _read_json(paths.recovery_state)
    if bool(recovery.get("active")):
        return True, str(recovery.get("reason") or "RECOVERY_ACTIVE")
    reconcile = _read_json(paths.reconciliation_status)
    status = str(reconcile.get("status") or ReconciliationStatus.UNKNOWN.value)
    if status in {
        ReconciliationStatus.POSITION_MISMATCH.value,
        ReconciliationStatus.ORPHAN_POSITION.value,
        ReconciliationStatus.ORPHAN_STATE.value,
    }:
        return True, status
    return False, None


def _trade_sequence(state: dict[str, Any]) -> list[dict[str, Any]]:
    rows = state.get("trade_sequence")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _append_trade_sequence_entry(
    state: dict[str, Any],
    *,
    entry_timestamp: str,
    playbook: str,
    strategy: str,
    decision_origin: str,
) -> None:
    rows = _trade_sequence(state)
    rows.append(
        {
            "entry_timestamp": entry_timestamp,
            "exit_timestamp": None,
            "pnl_rupees": None,
            "exit_reason": None,
            "playbook": playbook,
            "strategy": strategy,
            "decision_origin": decision_origin,
        }
    )
    state["trade_sequence"] = rows


def _close_trade_sequence_entry(
    state: dict[str, Any],
    *,
    entry_timestamp: str,
    exit_timestamp: str,
    pnl_rupees: float,
    exit_reason: str,
) -> None:
    rows = _trade_sequence(state)
    for row in rows:
        if str(row.get("entry_timestamp") or "") == entry_timestamp and row.get("exit_timestamp") in {None, ""}:
            row["exit_timestamp"] = exit_timestamp
            row["pnl_rupees"] = round(float(pnl_rupees), 2)
            row["exit_reason"] = exit_reason
            break
    state["trade_sequence"] = rows


def record_runtime_trade_entry(
    decision: DecisionOutput,
    snapshot: MarketSnapshot,
    *,
    decision_origin: str,
    paths: OpsPaths | None = None,
) -> None:
    paths = paths or OpsPaths()
    config = load_runtime_config(paths)
    state = load_runtime_state(paths, config)
    state["session_trade_count"] = int(state.get("session_trade_count") or 0) + 1
    _append_trade_sequence_entry(
        state,
        entry_timestamp=snapshot.timestamp.isoformat(),
        playbook=str(decision.metadata.get("playbook") or "UNKNOWN"),
        strategy=decision.strategy.value if isinstance(decision.strategy, StrategyType) else str(decision.strategy),
        decision_origin=decision_origin,
    )
    save_runtime_state(state, paths=paths)


def record_runtime_trade_exit(
    position: OpenPosition,
    snapshot: MarketSnapshot,
    *,
    pnl_rupees: float,
    exit_reason: str,
    paths: OpsPaths | None = None,
) -> None:
    paths = paths or OpsPaths()
    config = load_runtime_config(paths)
    state = load_runtime_state(paths, config)
    _close_trade_sequence_entry(
        state,
        entry_timestamp=position.entry_time.isoformat(),
        exit_timestamp=snapshot.timestamp.isoformat(),
        pnl_rupees=pnl_rupees,
        exit_reason=exit_reason,
    )
    state["last_exit_reason"] = exit_reason
    save_runtime_state(state, paths=paths)


def _score_margin_required(
    metadata: dict[str, Any],
    funnel: dict[str, Any],
    *,
    config: RuntimeConfig,
    market_state: str,
    no_trade_score: float,
) -> float:
    configured = float(config.bearish_score_margin)
    dynamic = float(
        metadata.get("trade_score_margin_bearish")
        or funnel.get("trade_score_margin_bearish")
        or configured
    )
    required = dynamic if dynamic > 0 else configured
    if market_state == "DIRECTIONAL_BALANCE":
        required = max(
            required,
            max(abs(no_trade_score), 1.0) * max(config.directional_balance_score_margin_override, 0.0),
        )
    return round(required, 4)


def _second_trade_block_reason(
    state: dict[str, Any],
    *,
    now: datetime,
    config: RuntimeConfig,
) -> str | None:
    if not config.risk.second_trade_requires_first_trade_profit:
        return None
    if int(state.get("session_trade_count") or 0) < 1:
        return None
    rows = _trade_sequence(state)
    if not rows:
        return "SECOND_TRADE_FIRST_RESULT_MISSING"
    first = rows[0]
    first_entry = _parse_dt(first.get("entry_timestamp"))
    first_exit = _parse_dt(first.get("exit_timestamp"))
    if first_entry is None:
        return "SECOND_TRADE_FIRST_RESULT_MISSING"
    if first_exit is None:
        return "SECOND_TRADE_FIRST_NOT_COMPLETED"
    if float(first.get("pnl_rupees") or 0.0) <= 0.0:
        return "SECOND_TRADE_REQUIRES_FIRST_PROFIT"
    if first_entry.tzinfo is None and now.tzinfo is not None:
        first_entry = first_entry.replace(tzinfo=now.tzinfo)
    elif first_entry.tzinfo is not None and now.tzinfo is None:
        first_entry = first_entry.replace(tzinfo=None)
    elapsed_minutes = max((now - first_entry).total_seconds() / 60.0, 0.0)
    if elapsed_minutes < max(config.risk.second_trade_min_gap_minutes, 0):
        return "SECOND_TRADE_MIN_GAP_NOT_MET"
    return None


def evaluate_entry_gate(
    decision: DecisionOutput,
    snapshot: MarketSnapshot,
    *,
    config: RuntimeConfig,
    health: dict[str, Any],
    paths: OpsPaths | None = None,
) -> dict[str, Any]:
    paths = paths or OpsPaths()
    state = load_runtime_state(paths, config)
    metadata = _decision_metadata(decision)
    funnel = _trade_funnel(decision)
    reasons: list[str] = []
    mode = config.mode
    if mode not in {RuntimeMode.PAPER_LIVE, RuntimeMode.MICRO_LIVE}:
        reasons.append("MODE_NOT_PAPER_OR_MICRO")
    if mode == RuntimeMode.MICRO_LIVE:
        if config.risk.require_manual_live_arm and not config.live_arm:
            reasons.append("MICRO_LIVE_NOT_ARMED")
    if decision.action != "TRADE":
        reasons.append("NO_TRADE_DECISION")
    playbook = str(metadata.get("playbook") or funnel.get("playbook") or "")
    if playbook not in set(config.live_enabled_playbooks):
        reasons.append("PLAYBOOK_NOT_LIVE_ENABLED")
    strategy = decision.strategy.value if isinstance(decision.strategy, StrategyType) else str(decision.strategy)
    if strategy not in set(config.live_enabled_strategies):
        reasons.append("STRATEGY_NOT_LIVE_ENABLED")
    market_state = str(metadata.get("market_state") or funnel.get("market_state") or "")
    if market_state not in set(config.approved_bearish_live_states):
        reasons.append("MARKET_STATE_NOT_APPROVED")
    setup_direction = str(metadata.get("setup_direction") or funnel.get("setup_direction") or "").upper()
    bearish_score = float(metadata.get("bearish_trade_score") or funnel.get("bearish_trade_score") or 0.0)
    bullish_score = float(metadata.get("bullish_trade_score") or funnel.get("bullish_trade_score") or 0.0)
    no_trade_score = float(metadata.get("no_trade_score") or funnel.get("no_trade_score") or 0.0)
    score_margin_required = _score_margin_required(
        metadata,
        funnel,
        config=config,
        market_state=market_state,
        no_trade_score=no_trade_score,
    )
    # Use directional score: bullish strategies use bullish_trade_score, others use bearish_trade_score
    if setup_direction == "BULLISH":
        directional_score = bullish_score
    elif setup_direction == "RANGE":
        directional_score = max(bearish_score, bullish_score)
    else:
        directional_score = bearish_score
    if directional_score <= no_trade_score + score_margin_required:
        reasons.append("DIRECTIONAL_SCORE_MARGIN_FAIL")
    # Early structure playbooks fire in the 9:20–9:55 window before full confirmation.
    # They bypass the normal 10:00 bullish start and the 11:00 bearish start.
    if playbook in {"EARLY_STRUCTURE_BULLISH", "EARLY_STRUCTURE_BEARISH"}:
        start_t = time(9, 20)
    # Bullish setups use an earlier start time so gap-up continuation plays can
    # be entered from 10:00 AM instead of waiting for the bearish window (11:00).
    elif setup_direction == "BULLISH":
        start_t = _parse_time(config.allowed_bullish_entry_start_hhmm, time(10, 0))
    else:
        start_t = _parse_time(config.allowed_entry_start_hhmm, time(9, 30))
    end_t = min(
        _parse_time(config.allowed_entry_end_hhmm, time(14, 30)),
        _parse_time(config.risk.no_new_entries_after_hhmm, time(14, 30)),
    )
    if not (start_t <= snapshot.timestamp.time() <= end_t):
        reasons.append("ENTRY_WINDOW_CLOSED")
    daily_lock = state.get("daily_lock") if isinstance(state.get("daily_lock"), dict) else {}
    if bool(daily_lock.get("active")):
        reasons.append(str(daily_lock.get("reason") or "DAILY_LOCK_ACTIVE"))
    if _active_paper_position_count(paths) >= config.risk.max_open_structures:
        reasons.append("ACTIVE_STRUCTURE_EXISTS")
    session_trade_count = int(state.get("session_trade_count") or 0)
    if session_trade_count >= config.risk.max_trades_per_day:
        reasons.append("MAX_TRADES_PER_DAY_REACHED")
    elif session_trade_count >= 1:
        second_trade_reason = _second_trade_block_reason(state, now=snapshot.timestamp, config=config)
        if second_trade_reason:
            reasons.append(second_trade_reason)
    if int(decision.lots or 0) > config.risk.max_lots_per_trade:
        reasons.append("LOTS_EXCEED_RUNTIME_LIMIT")
    if float(state.get("realized_pnl_rupees") or 0.0) <= -abs(config.risk.max_daily_realized_loss_rupees):
        reasons.append("DAILY_REALIZED_LOSS_LOCK")
    if float(state.get("total_pnl_rupees") or 0.0) <= -abs(config.risk.max_daily_total_loss_rupees):
        reasons.append("DAILY_TOTAL_LOSS_LOCK")
    # Expiry-day guard: credit spreads on same-day expiry carry extreme gamma risk
    # (near-ATM options can move 10–50× in minutes). Block all new entries.
    try:
        chain_expiry = snapshot.option_chain.expiry
        if isinstance(chain_expiry, date) and chain_expiry == snapshot.timestamp.date():
            reasons.append("EXPIRY_DAY_BLOCKED")
    except Exception:
        pass
    if str(health.get("status") or "") == HealthStatus.BLOCKED.value:
        reasons.append("HEALTH_BLOCKED")
    components = health.get("components") if isinstance(health.get("components"), dict) else {}
    required = [
        "broker_auth_health",
        "market_feed_health",
        "option_chain_health",
        "broker_position_sync_health",
        "state_store_health",
        "strategy_engine_health",
        "data_pipeline_health",
    ]
    for name in required:
        component = components.get(name) if isinstance(components, dict) else None
        if not isinstance(component, dict) or str(component.get("status") or "") == HealthStatus.BLOCKED.value:
            reasons.append(f"{name.upper()}_BLOCKED")
    recovery_blocked, recovery_reason = _has_recovery_block(paths)
    if recovery_blocked:
        reasons.append(f"RECOVERY_BLOCK:{recovery_reason}")
    allowed = not reasons
    return {
        "allowed": allowed,
        "primary_block_reason": _primary_block(reasons),
        "block_reasons": reasons,
        "mode": mode.value,
        "playbook": playbook,
        "strategy": strategy,
        "market_state": market_state,
        "tradability_class": metadata.get("tradability_class") or funnel.get("tradability_class"),
        "bearish_trade_score": round(bearish_score, 4),
        "no_trade_score": round(no_trade_score, 4),
        "score_margin_required": score_margin_required,
    }


def evaluate_paper_context_override_gate(
    decision: DecisionOutput,
    snapshot: MarketSnapshot,
    *,
    config: RuntimeConfig,
    health: dict[str, Any],
    paths: OpsPaths | None = None,
) -> dict[str, Any]:
    paths = paths or OpsPaths()
    state = load_runtime_state(paths, config)
    metadata = _decision_metadata(decision)
    funnel = _trade_funnel(decision)
    reasons: list[str] = []

    if config.mode != RuntimeMode.PAPER_LIVE:
        reasons.append("MODE_NOT_PAPER_LIVE")
    _origin = _decision_origin(decision)
    if _origin not in {"PAPER_CONTEXT_OVERRIDE", "PAPER_CONTEXT_OVERRIDE_BULLISH"}:
        reasons.append("NOT_PAPER_CONTEXT_OVERRIDE")
    if decision.action != "TRADE":
        reasons.append("NO_TRADE_DECISION")

    strategy = decision.strategy.value if isinstance(decision.strategy, StrategyType) else str(decision.strategy)
    if strategy not in set(config.live_enabled_strategies):
        reasons.append("STRATEGY_NOT_LIVE_ENABLED")

    market_state = str(metadata.get("market_state") or funnel.get("market_state") or "")
    if market_state not in {"TRANSITION", "TREND_DOWN", "TREND_UP", "DIRECTIONAL_BALANCE"}:
        reasons.append("MARKET_STATE_NOT_PAPER_OVERRIDE_APPROVED")

    _is_bullish_override = (_origin == "PAPER_CONTEXT_OVERRIDE_BULLISH")
    bearish_score = float(metadata.get("bearish_trade_score") or funnel.get("bearish_trade_score") or 0.0)
    bullish_score = float(metadata.get("bullish_trade_score") or funnel.get("bullish_trade_score") or 0.0)
    no_trade_score = float(metadata.get("no_trade_score") or funnel.get("no_trade_score") or 0.0)
    _active_score = bullish_score if _is_bullish_override else bearish_score
    _score_threshold = float(getattr(config, "paper_override_bullish_score_threshold", config.paper_override_bearish_score_threshold)) if _is_bullish_override else config.paper_override_bearish_score_threshold
    if config.circuit_breaker_active:
        _score_threshold += 1.0  # raise bar by 1.0 after 2 consecutive session losses
    if _active_score < _score_threshold:
        reasons.append("PAPER_OVERRIDE_SCORE_TOO_LOW")
    if _active_score <= no_trade_score + config.paper_override_margin:
        reasons.append("PAPER_OVERRIDE_MARGIN_FAIL")

    start_t = _parse_time(config.allowed_entry_start_hhmm, time(9, 30))
    end_t = _parse_time(config.paper_experiment_entry_end_hhmm, time(14, 30))
    inside_window = start_t <= snapshot.timestamp.time() <= end_t
    if not inside_window:
        reasons.append("PAPER_EXPERIMENT_WINDOW_CLOSED")

    daily_lock = state.get("daily_lock") if isinstance(state.get("daily_lock"), dict) else {}
    if bool(daily_lock.get("active")):
        reasons.append(str(daily_lock.get("reason") or "DAILY_LOCK_ACTIVE"))
    if _active_paper_position_count(paths) >= config.risk.max_open_structures:
        reasons.append("ACTIVE_STRUCTURE_EXISTS")
    session_trade_count = int(state.get("session_trade_count") or 0)
    if session_trade_count >= config.risk.max_trades_per_day:
        reasons.append("MAX_TRADES_PER_DAY_REACHED")
    elif session_trade_count >= 1:
        second_trade_reason = _second_trade_block_reason(state, now=snapshot.timestamp, config=config)
        if second_trade_reason:
            reasons.append(second_trade_reason)
    if int(decision.lots or 0) > config.risk.max_lots_per_trade:
        reasons.append("LOTS_EXCEED_RUNTIME_LIMIT")
    if float(state.get("realized_pnl_rupees") or 0.0) <= -abs(config.risk.max_daily_realized_loss_rupees):
        reasons.append("DAILY_REALIZED_LOSS_LOCK")
    if float(state.get("total_pnl_rupees") or 0.0) <= -abs(config.risk.max_daily_total_loss_rupees):
        reasons.append("DAILY_TOTAL_LOSS_LOCK")
    # Expiry-day guard: same-day expiry = extreme gamma risk, block paper entries too
    try:
        chain_expiry = snapshot.option_chain.expiry
        if isinstance(chain_expiry, date) and chain_expiry == snapshot.timestamp.date():
            reasons.append("EXPIRY_DAY_BLOCKED")
    except Exception:
        pass

    if config.health_required_for_paper:
        if str(health.get("status") or "") == HealthStatus.BLOCKED.value:
            reasons.append("HEALTH_BLOCKED")
        components = health.get("components") if isinstance(health.get("components"), dict) else {}
        required = [
            "broker_auth_health",
            "market_feed_health",
            "option_chain_health",
            "broker_position_sync_health",
            "state_store_health",
            "strategy_engine_health",
            "data_pipeline_health",
        ]
        for name in required:
            component = components.get(name) if isinstance(components, dict) else None
            if not isinstance(component, dict) or str(component.get("status") or "") == HealthStatus.BLOCKED.value:
                reasons.append(f"{name.upper()}_BLOCKED")

    recovery_blocked, recovery_reason = _has_recovery_block(paths)
    if recovery_blocked:
        reasons.append(f"RECOVERY_BLOCK:{recovery_reason}")

    # ── OI directional conflict guards ──────────────────────────────────────
    # For bearish override trades: block if option chain signals strongly oppose the direction.
    # Smart money (institutional OI flow) and wall migration are the most reliable signals.
    # PCR < 0.65 = call-heavy market = institutions positioned bullish = high risk for bear spreads.
    _is_bearish_strategy = strategy in {
        StrategyType.BEAR_CALL_CREDIT_SPREAD.value,
        "BEAR_CALL_CREDIT_SPREAD",
    }
    if _is_bearish_strategy:
        smart_money_bias = str(metadata.get("smart_money_bias") or funnel.get("smart_money_bias") or "NEUTRAL")
        wall_migration_bias = str(metadata.get("wall_migration_bias") or funnel.get("wall_migration_bias") or "NEUTRAL")
        pcr = metadata.get("pcr_by_oi") or funnel.get("pcr_by_oi")
        pcr_val = float(pcr) if pcr is not None else None
        if smart_money_bias == "BULLISH":
            reasons.append("OI_FLOW_OPPOSES_BEARISH_ENTRY")
        if wall_migration_bias == "BULLISH":
            reasons.append("WALL_MIGRATION_OPPOSES_BEARISH_ENTRY")
        if pcr_val is not None and pcr_val < 0.65:
            reasons.append(f"PCR_TOO_LOW_FOR_BEARISH_ENTRY:{pcr_val:.2f}")

    allowed = not reasons
    return {
        "allowed": allowed,
        "primary_block_reason": _primary_block(reasons),
        "block_reasons": reasons,
        "mode": config.mode.value,
        "playbook": metadata.get("playbook") or funnel.get("playbook"),
        "strategy": strategy,
        "market_state": market_state,
        "tradability_class": metadata.get("tradability_class") or funnel.get("tradability_class"),
        "failure_type": metadata.get("failure_type") or funnel.get("failure_type"),
        "bearish_trade_score": round(bearish_score, 4),
        "no_trade_score": round(no_trade_score, 4),
        "score_threshold_required": config.paper_override_bearish_score_threshold,
        "score_margin_required": config.paper_override_margin,
        "paper_experiment_entry_end_hhmm": config.paper_experiment_entry_end_hhmm,
        "inside_paper_experiment_window": inside_window,
        "decision_origin": "PAPER_CONTEXT_OVERRIDE",
    }


def _leg_from_payload(payload: dict[str, Any]) -> StrategyLeg:
    option_type_raw = str(payload.get("option_type") or payload.get("quote", {}).get("option_type") or "").upper()
    if option_type_raw in {"CE", "CALL"}:
        option_type = OptionType.CALL
    elif option_type_raw in {"PE", "PUT"}:
        option_type = OptionType.PUT
    else:
        option_type = OptionType(option_type_raw)
    quote = OptionsContractQuote(
        strike=float(payload.get("strike") or 0.0),
        option_type=option_type,
        bid=float(payload.get("bid") or 0.0),
        ask=float(payload.get("ask") or 0.0),
        ltp=float(payload.get("ltp") or 0.0),
        delta=float(payload["delta"]) if payload.get("delta") is not None else None,
        iv=float(payload["iv"]) if payload.get("iv") is not None else None,
        oi=int(payload["oi"]) if payload.get("oi") is not None else None,
        symbol=str(payload.get("symbol")) if payload.get("symbol") else None,
    )
    return StrategyLeg(
        action=str(payload.get("action") or "").upper(),
        option_type=option_type,
        strike=float(payload.get("strike") or 0.0),
        quote=quote,
    )


def structure_from_decision(decision: DecisionOutput) -> TradeStructure:
    metadata = _decision_metadata(decision)
    legs = [_leg_from_payload(dict(item)) for item in decision.legs]
    width = float(metadata.get("structure_width_points") or 0.0)
    if width <= 0.0 and legs:
        sell_strikes = [leg.strike for leg in legs if leg.action == "SELL"]
        buy_strikes = [leg.strike for leg in legs if leg.action == "BUY"]
        if sell_strikes and buy_strikes:
            width = abs(max(buy_strikes) - max(sell_strikes)) or abs(min(buy_strikes) - min(sell_strikes))
    strategy = decision.strategy if isinstance(decision.strategy, StrategyType) else StrategyType(str(decision.strategy))
    return TradeStructure(
        strategy=strategy,
        legs=legs,
        credit_points=float(metadata.get("structure_credit_points") or decision.entry.get("expected_credit_points") or 0.0),
        width_points=width,
        call_width_points=width if strategy == StrategyType.BEAR_CALL_CREDIT_SPREAD else 0.0,
        put_width_points=width if strategy == StrategyType.BULL_PUT_CREDIT_SPREAD else 0.0,
        margin_estimate_per_lot=None,
        rationale=list(decision.rationale),
        metadata=dict(metadata),
    )


def open_position_from_decision(decision: DecisionOutput, snapshot: MarketSnapshot) -> OpenPosition:
    structure = structure_from_decision(decision)
    return build_open_position(
        structure=structure,
        lots=int(decision.lots or 1),
        lot_size=snapshot.lot_size,
        entry_time=snapshot.timestamp,
        entry_credit_points=float(decision.entry.get("expected_credit_points") or structure.credit_points),
        max_loss_rupees_per_lot=float(decision.max_loss_rupees_per_lot or 0.0),
        extra_metadata=decision.metadata,
    )


def serialize_open_position(position: OpenPosition) -> dict[str, Any]:
    return {
        "structure": {
            "strategy": position.structure.strategy.value,
            "legs": [
                {
                    "action": leg.action,
                    "option_type": leg.option_type.value,
                    "strike": leg.strike,
                    "bid": leg.quote.bid,
                    "ask": leg.quote.ask,
                    "ltp": leg.quote.ltp,
                    "delta": leg.quote.delta,
                    "iv": leg.quote.iv,
                    "oi": leg.quote.oi,
                    "symbol": leg.quote.symbol,
                }
                for leg in position.structure.legs
            ],
            "credit_points": position.structure.credit_points,
            "width_points": position.structure.width_points,
            "call_width_points": position.structure.call_width_points,
            "put_width_points": position.structure.put_width_points,
            "margin_estimate_per_lot": position.structure.margin_estimate_per_lot,
            "rationale": position.structure.rationale,
            "metadata": position.structure.metadata,
        },
        "lots": position.lots,
        "lot_size": position.lot_size,
        "entry_time": position.entry_time.isoformat(),
        "entry_credit_points": position.entry_credit_points,
        "target_value_points": position.target_value_points,
        "stop_value_points": position.stop_value_points,
        "max_loss_rupees_per_lot": position.max_loss_rupees_per_lot,
        "take_profit_capture_pct": position.take_profit_capture_pct,
        "metadata": position.metadata,
    }


def deserialize_open_position(payload: dict[str, Any]) -> OpenPosition | None:
    if not isinstance(payload, dict) or not payload:
        return None
    structure_payload = payload.get("structure")
    if not isinstance(structure_payload, dict):
        return None
    legs = [_leg_from_payload(dict(item)) for item in structure_payload.get("legs", []) if isinstance(item, dict)]
    if not legs:
        return None
    structure = TradeStructure(
        strategy=StrategyType(str(structure_payload.get("strategy"))),
        legs=legs,
        credit_points=float(structure_payload.get("credit_points") or 0.0),
        width_points=float(structure_payload.get("width_points") or 0.0),
        call_width_points=float(structure_payload.get("call_width_points") or 0.0),
        put_width_points=float(structure_payload.get("put_width_points") or 0.0),
        margin_estimate_per_lot=(
            float(structure_payload["margin_estimate_per_lot"])
            if structure_payload.get("margin_estimate_per_lot") is not None
            else None
        ),
        rationale=list(structure_payload.get("rationale") or []),
        metadata=dict(structure_payload.get("metadata") or {}),
    )
    entry_time = _parse_dt(payload.get("entry_time")) or _now()
    return OpenPosition(
        structure=structure,
        lots=int(payload.get("lots") or 1),
        lot_size=int(payload.get("lot_size") or 1),
        entry_time=entry_time.replace(tzinfo=None),
        entry_credit_points=float(payload.get("entry_credit_points") or 0.0),
        target_value_points=float(payload.get("target_value_points") or 0.0),
        stop_value_points=float(payload.get("stop_value_points") or 0.0),
        max_loss_rupees_per_lot=float(payload.get("max_loss_rupees_per_lot") or 0.0),
        take_profit_capture_pct=float(payload.get("take_profit_capture_pct") or 0.0),
        metadata=dict(payload.get("metadata") or {}),
    )


def load_paper_position(paths: OpsPaths | None = None) -> OpenPosition | None:
    paths = paths or OpsPaths()
    state = _read_json(paths.paper_state)
    active = state.get("active_position") if isinstance(state, dict) else None
    return deserialize_open_position(active if isinstance(active, dict) else {})


def save_paper_position(
    position: OpenPosition | None,
    *,
    paths: OpsPaths | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    paths = paths or OpsPaths()
    prior = _read_json(paths.paper_state)
    payload = dict(prior) if isinstance(prior, dict) else {}
    payload["updated_at"] = _now().isoformat(timespec="seconds")
    payload["active_position"] = serialize_open_position(position) if position else None
    if extra:
        payload.update(extra)
    _write_json(paths.paper_state, payload)


def _decision_log_payload(decision: DecisionOutput, *, snapshot: MarketSnapshot, action: str, block_reason: str | None = None) -> dict[str, Any]:
    metadata = _decision_metadata(decision)
    funnel = _trade_funnel(decision)
    return {
        "timestamp": snapshot.timestamp.isoformat(),
        "session_date": snapshot.timestamp.date().isoformat(),
        "market_state": metadata.get("market_state") or funnel.get("market_state"),
        "tradability_class": metadata.get("tradability_class") or funnel.get("tradability_class"),
        "playbook": metadata.get("playbook") or funnel.get("playbook"),
        "subtype": metadata.get("setup_subtype") or metadata.get("bearish_shadow_subtype") or metadata.get("bearish_subtype") or metadata.get("bullish_shadow_subtype"),
        "failure_type": metadata.get("failure_type"),
        "setup_quality_score": metadata.get("setup_quality_score") or funnel.get("setup_quality_score"),
        "market_state_score": metadata.get("market_state_score") or funnel.get("market_state_score"),
        "trend_quality_score": metadata.get("trend_quality_score") or funnel.get("trend_quality_score"),
        "failure_score": metadata.get("failure_score") or funnel.get("failure_score"),
        "location_score": metadata.get("location_score") or funnel.get("location_score"),
        "option_chain_pressure_score": metadata.get("option_chain_pressure_score") or funnel.get("option_chain_pressure_score"),
        "monetization_score": metadata.get("live_monetization_score") or funnel.get("monetization_score"),
        "tradability_score": metadata.get("tradability_score") or funnel.get("tradability_score"),
        "bearish_trade_score": metadata.get("bearish_trade_score") or funnel.get("bearish_trade_score"),
        "bullish_trade_score": metadata.get("bullish_trade_score") or funnel.get("bullish_trade_score"),
        "no_trade_score": metadata.get("no_trade_score") or funnel.get("no_trade_score"),
        # OI intelligence fields — key for directional conflict diagnosis
        "pcr_by_oi": metadata.get("pcr_by_oi") or funnel.get("pcr_by_oi"),
        "pcr_trend": metadata.get("pcr_trend") or funnel.get("pcr_trend"),
        "smart_money_bias": metadata.get("smart_money_bias") or funnel.get("smart_money_bias"),
        "wall_migration_bias": metadata.get("wall_migration_bias") or funnel.get("wall_migration_bias"),
        "bearish_option_chain_pressure_score": metadata.get("bearish_option_chain_pressure_score") or funnel.get("bearish_option_chain_pressure_score"),
        "bullish_option_chain_pressure_score": metadata.get("bullish_option_chain_pressure_score") or funnel.get("bullish_option_chain_pressure_score"),
        "chosen_action": action,
        "chosen_strategy": decision.strategy.value if isinstance(decision.strategy, StrategyType) else str(decision.strategy),
        "block_reason": block_reason,
        "hypothetical_entry_timestamp": decision.entry.get("timestamp") if decision.action == "TRADE" else None,
        "hypothetical_exit_timestamp": decision.time_exit if decision.action == "TRADE" else None,
        "exit_reason": "TIME_EXIT_HYPOTHESIS" if decision.action == "TRADE" else None,
        "hypothetical_pnl_rupees": None,
    }


def _snapshot_data_summary(
    *,
    snapshot: MarketSnapshot | None = None,
    data_status: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data_status = dict(data_status or {})
    if snapshot is None:
        return {
            "snapshot_status": str(data_status.get("snapshot_status") or "UNAVAILABLE"),
            "reason": data_status.get("reason"),
            "candle_count": int(data_status.get("candle_count") or 0),
            "first_timestamp": data_status.get("first_timestamp"),
            "last_timestamp": data_status.get("last_timestamp"),
            "timestamps_valid": bool(data_status.get("timestamps_valid", False)),
            "minimum_required_5m": data_status.get("minimum_required_5m"),
            "option_quote_count": int(data_status.get("option_quote_count") or 0),
            "option_chain_available": bool(data_status.get("option_chain_available", False)),
            "spot_price": data_status.get("spot_price"),
            "spot_available": bool(data_status.get("spot_available", False)),
        } | {
            key: value
            for key, value in data_status.items()
            if key not in {
                "snapshot_status",
                "reason",
                "candle_count",
                "first_timestamp",
                "last_timestamp",
                "timestamps_valid",
                "minimum_required_5m",
                "option_quote_count",
                "option_chain_available",
                "spot_price",
                "spot_available",
            }
        }

    bars = list(snapshot.nifty_5m.bars or [])
    return {
        "snapshot_status": str(data_status.get("snapshot_status") or "READY"),
        "reason": data_status.get("reason"),
        "candle_count": len(bars),
        "first_timestamp": bars[0].timestamp.isoformat() if bars else None,
        "last_timestamp": bars[-1].timestamp.isoformat() if bars else None,
        "timestamps_valid": all(bar.timestamp is not None for bar in bars),
        "minimum_required_5m": data_status.get("minimum_required_5m") or 5,
        "option_quote_count": len(snapshot.option_chain.quotes),
        "option_chain_available": bool(snapshot.option_chain.quotes),
        "spot_price": snapshot.option_chain.spot,
        "spot_available": snapshot.option_chain.spot > 0.0,
        "option_chain_timestamp": snapshot.option_chain.timestamp.isoformat(),
    }


def _fallback_block_reason(decision: DecisionOutput, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    metadata = _decision_metadata(decision)
    funnel = _trade_funnel(decision)
    for key in ("primary_block_reason", "canonical_rejection_reason", "rejection_reason"):
        value = metadata.get(key)
        if value:
            return str(value)
    value = funnel.get("canonical_rejection_reason")
    if value:
        return str(value)
    if decision.action == "TRADE":
        return "NONE"
    return "NO_TRADE_DECISION"


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _validation_anomalies(payload: dict[str, Any], *, paths: OpsPaths) -> list[dict[str, Any]]:
    rows = _read_jsonl(paths.validation_decisions)
    session_date = str(payload.get("session_date") or "")
    recent = [row for row in rows[-120:] if str(row.get("session_date") or "") == session_date]
    anomalies: list[dict[str, Any]] = []
    block_reason = str(payload.get("block_reason") or "NONE")
    decision = str(payload.get("decision") or "")
    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
    snapshot_status = str(data.get("snapshot_status") or "")

    if decision == "NO_TRADE" and block_reason != "NONE":
        streak = 1
        for row in reversed(recent):
            if str(row.get("decision") or "") != "NO_TRADE":
                break
            if str(row.get("block_reason") or "NONE") != block_reason:
                break
            streak += 1
        if streak > 50:
            anomalies.append(
                {
                    "type": "REPEATED_IDENTICAL_NO_TRADE_REASON",
                    "severity": "WARNING",
                    "block_reason": block_reason,
                    "consecutive_count": streak,
                }
            )

    current_count = int(data.get("candle_count") or 0)
    previous_with_count = next(
        (
            row for row in reversed(recent)
            if isinstance(row.get("data"), dict) and int(row.get("data", {}).get("candle_count") or 0) > 0
        ),
        None,
    )
    if previous_with_count is not None:
        previous_count = int(previous_with_count.get("data", {}).get("candle_count") or 0)
        if current_count > 0 and previous_count > 0 and current_count < previous_count:
            anomalies.append(
                {
                    "type": "SUDDEN_CANDLE_COUNT_DROP",
                    "severity": "WARNING",
                    "previous_candle_count": previous_count,
                    "current_candle_count": current_count,
                }
            )

    if snapshot_status == "READY" and block_reason in {"SNAPSHOT_UNAVAILABLE", "INSUFFICIENT_DATA"}:
        anomalies.append(
            {
                "type": "SNAPSHOT_PRESENT_BUT_IGNORED",
                "severity": "WARNING",
                "block_reason": block_reason,
            }
        )
    if snapshot_status == "READY" and not payload.get("market_state"):
        anomalies.append(
            {
                "type": "STATE_CLASSIFICATION_MISSING",
                "severity": "WARNING",
            }
        )
    if snapshot_status == "READY" and (
        payload.get("bearish_trade_score") is None or payload.get("no_trade_score") is None
    ):
        anomalies.append(
            {
                "type": "SCORE_CALCULATION_SKIPPED",
                "severity": "WARNING",
            }
        )
    return anomalies


def log_validation_decision(
    decision: DecisionOutput,
    *,
    snapshot: MarketSnapshot | None = None,
    paths: OpsPaths | None = None,
    block_reason: str | None = None,
    data_status: dict[str, Any] | None = None,
    event_type: str = "DECISION",
    paper_candidate: dict[str, Any] | None = None,
    actual_trade_created: bool | None = None,
    decision_origin: str | None = None,
) -> dict[str, Any]:
    paths = paths or OpsPaths()
    metadata = _decision_metadata(decision)
    funnel = _trade_funnel(decision)
    timestamp = snapshot.timestamp if snapshot is not None else _now()
    chosen_strategy = decision.strategy.value if isinstance(decision.strategy, StrategyType) else str(decision.strategy)
    resolved_block = _fallback_block_reason(decision, block_reason)
    data = _snapshot_data_summary(snapshot=snapshot, data_status=data_status)
    v83_decision = "TRADE" if decision.action == "TRADE" and resolved_block == "NONE" else "NO_TRADE"
    resolved_origin = decision_origin or _decision_origin(decision)
    paper_candidate = dict(paper_candidate or {})
    effective_paper_candidate_decision = paper_candidate.get("paper_candidate_decision")
    if effective_paper_candidate_decision is None and resolved_origin == "V83_APPROVED":
        effective_paper_candidate_decision = "TRADE"
    if effective_paper_candidate_decision is None:
        effective_paper_candidate_decision = "NO_TRADE"
    effective_paper_candidate_reason = paper_candidate.get("paper_candidate_reason")
    if effective_paper_candidate_reason is None and resolved_origin == "V83_APPROVED":
        effective_paper_candidate_reason = "V83_APPROVED"
    if effective_paper_candidate_reason is None:
        effective_paper_candidate_reason = "NOT_EVALUATED"
    effective_mapped_strategy = paper_candidate.get("mapped_strategy")
    if effective_mapped_strategy is None and resolved_origin == "V83_APPROVED":
        effective_mapped_strategy = chosen_strategy
    effective_spread_status = paper_candidate.get("spread_construction_status")
    if effective_spread_status is None and resolved_origin == "V83_APPROVED":
        effective_spread_status = "PASSED"
    if effective_spread_status is None:
        effective_spread_status = "NOT_EVALUATED"
    effective_would_create_trade = bool(
        actual_trade_created
        if actual_trade_created is not None
        else paper_candidate.get("would_create_paper_trade", v83_decision == "TRADE")
    )
    effective_decision = "TRADE" if effective_would_create_trade else "NO_TRADE"
    effective_chosen_strategy = effective_mapped_strategy if effective_would_create_trade and resolved_origin == "PAPER_CONTEXT_OVERRIDE" else chosen_strategy
    payload = {
        "event_type": event_type,
        "timestamp": timestamp.isoformat(),
        "session_date": timestamp.date().isoformat(),
        "market_state": metadata.get("market_state") or funnel.get("market_state"),
        "tradability_class": metadata.get("tradability_class") or funnel.get("tradability_class"),
        "playbook": metadata.get("playbook") or funnel.get("playbook"),
        "subtype": metadata.get("setup_subtype") or metadata.get("bearish_shadow_subtype") or metadata.get("bearish_subtype") or metadata.get("bullish_shadow_subtype"),
        "failure_type": metadata.get("failure_type"),
        "bearish_trade_score": _first_present(metadata.get("bearish_trade_score"), funnel.get("bearish_trade_score")),
        "no_trade_score": _first_present(metadata.get("no_trade_score"), funnel.get("no_trade_score")),
        "decision": effective_decision,
        "raw_action": decision.action,
        "v83_decision": v83_decision,
        "v83_rejection_reason": "NONE" if v83_decision == "TRADE" else resolved_block,
        "paper_candidate_decision": effective_paper_candidate_decision,
        "paper_candidate_reason": effective_paper_candidate_reason,
        "paper_trade_attribution": resolved_origin,
        "mapped_strategy": effective_mapped_strategy,
        "spread_construction_status": effective_spread_status,
        "would_create_paper_trade": effective_would_create_trade,
        "paper_experiment_window_allows": paper_candidate.get("paper_experiment_window_allows"),
        "chosen_strategy": effective_chosen_strategy,
        "block_reason": resolved_block,
        "data": data,
    }
    anomalies = _validation_anomalies(payload, paths=paths)
    payload["anomalies"] = anomalies
    if anomalies:
        _append_jsonl(
            paths.operator_events,
            {
                "timestamp": _now().isoformat(timespec="seconds"),
                "event": "PAPER_LIVE_VALIDATION_WARNING",
                "session_date": payload["session_date"],
                "anomalies": anomalies,
            },
        )
    _append_jsonl(paths.validation_decisions, payload)
    return payload


def log_shadow_decision(
    decision: DecisionOutput,
    *,
    snapshot: MarketSnapshot,
    paths: OpsPaths | None = None,
    block_reason: str | None = None,
    force_would_trade: bool = False,
) -> None:
    paths = paths or OpsPaths()
    action = "WOULD_TRADE" if decision.action == "TRADE" and (force_would_trade or not block_reason) else "NO_TRADE"
    _append_jsonl(paths.shadow_decisions, _decision_log_payload(decision, snapshot=snapshot, action=action, block_reason=block_reason))


def record_paper_entry(
    position: OpenPosition,
    *,
    decision: DecisionOutput,
    snapshot: MarketSnapshot,
    gate: dict[str, Any],
    paths: OpsPaths | None = None,
) -> None:
    paths = paths or OpsPaths()
    save_paper_position(position, paths=paths, extra={"last_entry_gate": gate})
    attribution = str(decision.metadata.get("paper_trade_attribution") or "V83_APPROVED")
    event = {
        "event": "PAPER_ENTRY",
        "timestamp": snapshot.timestamp.isoformat(),
        "session_date": snapshot.timestamp.date().isoformat(),
        "entry_timestamp": snapshot.timestamp.isoformat(),
        "playbook": decision.metadata.get("playbook"),
        "subtype": decision.metadata.get("setup_subtype") or decision.metadata.get("bearish_subtype"),
        "paper_trade_attribution": attribution,
        "paper_candidate_reason": decision.metadata.get("paper_candidate_reason"),
        "strategy": decision.strategy.value if isinstance(decision.strategy, StrategyType) else str(decision.strategy),
        "legs": decision.legs,
        "entry_price_assumptions": decision.entry,
        "gate": gate,
        "realized_paper_pnl": None,
        "mfe_rupees": 0.0,
        "mae_rupees": 0.0,
    }
    _append_jsonl(paths.paper_trades, event)
    record_runtime_trade_entry(decision, snapshot, decision_origin=attribution, paths=paths)
    state = load_runtime_state(paths, load_runtime_config(paths))
    state["last_simulated_order"] = event
    state["primary_block_reason"] = "NONE"
    save_runtime_state(state, paths=paths)


def manage_paper_position(
    snapshot: MarketSnapshot,
    *,
    paths: OpsPaths | None = None,
) -> OpenPosition | None:
    paths = paths or OpsPaths()
    position = load_paper_position(paths)
    if position is None:
        return None
    from .regime import classify_regime

    exit_decision = evaluate_exit(
        position,
        current_snapshot=snapshot,
        current_regime=classify_regime(snapshot),
        now=snapshot.timestamp,
    )
    open_pnl = exit_decision.pnl_rupees
    state = _read_json(paths.paper_state)
    mfe = max(float(state.get("mfe_rupees") or 0.0), open_pnl)
    mae = min(float(state.get("mae_rupees") or 0.0), open_pnl)
    save_paper_position(position, paths=paths, extra={"mfe_rupees": mfe, "mae_rupees": mae, "last_mark_pnl_rupees": open_pnl})
    if not exit_decision.should_exit:
        return position
    event = {
        "event": "PAPER_EXIT",
        "timestamp": snapshot.timestamp.isoformat(),
        "session_date": snapshot.timestamp.date().isoformat(),
        "entry_timestamp": position.entry_time.isoformat(),
        "exit_timestamp": snapshot.timestamp.isoformat(),
        "exit_reason": exit_decision.reason,
        "playbook": position.metadata.get("playbook"),
        "subtype": position.metadata.get("setup_subtype") or position.metadata.get("bearish_subtype"),
        "paper_trade_attribution": position.metadata.get("paper_trade_attribution") or "V83_APPROVED",
        "paper_candidate_reason": position.metadata.get("paper_candidate_reason"),
        "strategy": position.structure.strategy.value,
        "exit_mode": position.metadata.get("exit_mode") or "STANDARD_TRAIL",
        "realized_paper_pnl": round(exit_decision.pnl_rupees, 2),
        "mfe_rupees": round(mfe, 2),
        "mae_rupees": round(mae, 2),
        "legs": serialize_open_position(position)["structure"]["legs"],
    }
    _append_jsonl(paths.paper_trades, event)
    save_paper_position(None, paths=paths, extra={"last_exit": event, "mfe_rupees": 0.0, "mae_rupees": 0.0})
    config = load_runtime_config(paths)
    runtime_state = load_runtime_state(paths, config)
    _close_trade_sequence_entry(
        runtime_state,
        entry_timestamp=position.entry_time.isoformat(),
        exit_timestamp=snapshot.timestamp.isoformat(),
        pnl_rupees=exit_decision.pnl_rupees,
        exit_reason=exit_decision.reason,
    )
    runtime_state["realized_pnl_rupees"] = round(float(runtime_state.get("realized_pnl_rupees") or 0.0) + exit_decision.pnl_rupees, 2)
    runtime_state["total_pnl_rupees"] = runtime_state["realized_pnl_rupees"]
    runtime_state["last_exit_reason"] = exit_decision.reason
    if exit_decision.pnl_rupees <= -abs(position.max_loss_rupees_per_lot * position.lots) and config.risk.stop_after_first_full_loss:
        runtime_state["daily_lock"] = {
            "active": True,
            "reason": "FIRST_FULL_LOSS_LOCK",
            "locked_at": snapshot.timestamp.isoformat(),
        }
    if runtime_state["realized_pnl_rupees"] <= -abs(config.risk.max_daily_realized_loss_rupees):
        runtime_state["daily_lock"] = {
            "active": True,
            "reason": "DAILY_REALIZED_LOSS_LOCK",
            "locked_at": snapshot.timestamp.isoformat(),
        }
    save_runtime_state(runtime_state, paths=paths)
    return None


def _position_identity_from_open(position: OpenPosition) -> set[tuple[str, float, str]]:
    out: set[tuple[str, float, str]] = set()
    for leg in position.structure.legs:
        out.add((leg.action, float(leg.strike), leg.option_type.value))
    return out


def _position_identity_from_broker_rows(rows: Iterable[dict[str, Any]]) -> set[tuple[str, float, str]]:
    out: set[tuple[str, float, str]] = set()
    for row in rows or []:
        side = str(row.get("side") or row.get("transactionType") or row.get("positionType") or "").upper()
        action = "SELL" if side.startswith("S") or side == "SHORT" else "BUY"
        opt_raw = str(row.get("option_type") or row.get("optionType") or row.get("instrumentType") or row.get("strike") or "").upper()
        option_type = "CALL" if "CE" in opt_raw or "CALL" in opt_raw else ("PUT" if "PE" in opt_raw or "PUT" in opt_raw else "")
        try:
            strike = float(row.get("strike_price") or row.get("strike") or row.get("strikePrice") or 0.0)
        except Exception:
            strike = 0.0
        if strike > 0 and option_type:
            out.add((action, strike, option_type))
    return out


def reconcile_positions(
    *,
    broker_positions: Iterable[dict[str, Any]] | None = None,
    paths: OpsPaths | None = None,
    mode: RuntimeMode | str | None = None,
) -> dict[str, Any]:
    paths = paths or OpsPaths()
    now = _now()
    runtime_mode = mode if isinstance(mode, RuntimeMode) else RuntimeMode(str(mode or load_runtime_config(paths).mode.value).upper())
    paper_position = load_paper_position(paths)
    broker_rows = list(broker_positions or [])
    internal_has = paper_position is not None
    broker_has = bool(broker_rows)
    if runtime_mode != RuntimeMode.MICRO_LIVE:
        internal_has = False
    if not internal_has and not broker_has:
        status = ReconciliationStatus.NO_POSITIONS
        reason = None
    elif internal_has and not broker_has:
        status = ReconciliationStatus.ORPHAN_STATE
        reason = "INTERNAL_OPEN_WITHOUT_BROKER_POSITION"
    elif broker_has and not internal_has:
        status = ReconciliationStatus.ORPHAN_POSITION
        reason = "BROKER_OPEN_WITHOUT_INTERNAL_STATE"
    else:
        internal_ids = _position_identity_from_open(paper_position) if paper_position else set()
        broker_ids = _position_identity_from_broker_rows(broker_rows)
        if not broker_ids:
            status = ReconciliationStatus.UNKNOWN
            reason = "BROKER_POSITION_IDENTITY_UNAVAILABLE"
        elif internal_ids == broker_ids:
            status = ReconciliationStatus.MATCHED
            reason = None
        else:
            status = ReconciliationStatus.POSITION_MISMATCH
            reason = "INTERNAL_BROKER_LEG_SET_DIFFERS"
    payload = {
        "status": status.value,
        "reason": reason,
        "hard_lock": status in {
            ReconciliationStatus.POSITION_MISMATCH,
            ReconciliationStatus.ORPHAN_POSITION,
            ReconciliationStatus.ORPHAN_STATE,
        },
        "internal_open": internal_has,
        "broker_open_count": len(broker_rows),
        "updated_at": now.isoformat(timespec="seconds"),
        "adopt_existing_position_monitoring_supported": True,
        "adopt_existing_position_monitoring_active": False,
    }
    _write_json(paths.reconciliation_status, payload)
    _append_jsonl(paths.reconciliation_events, payload)
    return payload


def flatten_all_emergency(
    *,
    paths: OpsPaths | None = None,
    source: str = "api",
    broker_flatten_fn: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    paths = paths or OpsPaths()
    config = load_runtime_config(paths)
    now = _now()
    results: list[dict[str, Any]] = []
    paper_position = load_paper_position(paths)
    if paper_position is not None:
        for leg in paper_position.structure.legs:
            results.append(
                {
                    "scope": "PAPER",
                    "leg": {
                        "action": "BUY" if leg.action == "SELL" else "SELL",
                        "option_type": leg.option_type.value,
                        "strike": leg.strike,
                    },
                    "status": "CLOSED_LOCALLY",
                    "error": None,
                }
            )
        save_paper_position(None, paths=paths, extra={"last_exit": {"exit_reason": "EMERGENCY_FLATTEN", "exit_timestamp": now.isoformat(timespec="seconds")}})
    if config.mode == RuntimeMode.MICRO_LIVE and config.live_arm and broker_flatten_fn is not None:
        try:
            broker_result = broker_flatten_fn()
            results.append({"scope": "BROKER", "status": "REQUESTED", "result": broker_result})
        except Exception as exc:  # noqa: BLE001
            results.append({"scope": "BROKER", "status": "FAILED", "error": str(exc)})
    elif config.mode == RuntimeMode.MICRO_LIVE:
        results.append({"scope": "BROKER", "status": "SKIPPED", "error": "NO_BROKER_FLATTEN_HANDLER"})
    recovery = {
        "active": True,
        "reason": "EMERGENCY_FLATTEN_REQUESTED",
        "source": source,
        "started_at": now.isoformat(timespec="seconds"),
        "cleared_at": None,
    }
    _write_json(paths.recovery_state, recovery)
    event = {
        "timestamp": now.isoformat(timespec="seconds"),
        "source": source,
        "mode": config.mode.value,
        "live_arm": config.live_arm,
        "results": results,
    }
    _append_jsonl(paths.emergency_flatten_events, event)
    return {"ok": True, "recovery_state": recovery, "results": results}


def build_shadow_live_report(paths: OpsPaths | None = None, *, write: bool = True) -> dict[str, Any]:
    paths = paths or OpsPaths()
    rows = _read_jsonl(paths.shadow_decisions)
    sessions = {str(row.get("session_date") or "") for row in rows if row.get("session_date")}
    no_trade_reasons = Counter(str(row.get("block_reason") or "NONE") for row in rows if row.get("chosen_action") != "WOULD_TRADE")
    playbooks = Counter(str(row.get("playbook") or "UNKNOWN") for row in rows)
    states = Counter(str(row.get("market_state") or "UNKNOWN") for row in rows)
    would_trades = [row for row in rows if row.get("chosen_action") == "WOULD_TRADE"]
    payload = {
        "generated_at": _now().isoformat(timespec="seconds"),
        "sessions_run": len(sessions),
        "session_dates": sorted(sessions),
        "decisions_made": len(rows),
        "would_trade_count": len(would_trades),
        "no_trade_reasons": dict(no_trade_reasons),
        "playbook_frequency": dict(playbooks),
        "state_frequency": dict(states),
        "hypothetical_pnl_summary": {
            "known_count": sum(1 for row in rows if row.get("hypothetical_pnl_rupees") is not None),
            "total_pnl_rupees": round(sum(float(row.get("hypothetical_pnl_rupees") or 0.0) for row in rows), 2),
        },
        "latest_would_trades": would_trades[-10:],
    }
    if write:
        _write_json(paths.shadow_report, payload)
    return payload


def build_paper_live_report(paths: OpsPaths | None = None, *, write: bool = True) -> dict[str, Any]:
    paths = paths or OpsPaths()
    rows = _read_jsonl(paths.paper_trades)
    entries = [row for row in rows if row.get("event") == "PAPER_ENTRY"]
    exits = [row for row in rows if row.get("event") == "PAPER_EXIT"]
    pnl_values = [float(row.get("realized_paper_pnl") or 0.0) for row in exits]
    attribution_summary: dict[str, dict[str, Any]] = defaultdict(lambda: {"entries": 0, "exits": 0, "pnl_rupees": 0.0, "wins": 0, "losses": 0})
    for row in entries:
        attribution = str(row.get("paper_trade_attribution") or "UNKNOWN")
        attribution_summary[attribution]["entries"] += 1
    for row in exits:
        attribution = str(row.get("paper_trade_attribution") or "UNKNOWN")
        pnl = float(row.get("realized_paper_pnl") or 0.0)
        attribution_summary[attribution]["exits"] += 1
        attribution_summary[attribution]["pnl_rupees"] += pnl
        if pnl > 0.0:
            attribution_summary[attribution]["wins"] += 1
        elif pnl < 0.0:
            attribution_summary[attribution]["losses"] += 1
    by_day: dict[str, dict[str, Any]] = defaultdict(lambda: {"trades": 0, "pnl_rupees": 0.0, "exits": Counter()})
    for row in exits:
        day = str(row.get("session_date") or "")
        by_day[day]["trades"] += 1
        by_day[day]["pnl_rupees"] += float(row.get("realized_paper_pnl") or 0.0)
        by_day[day]["exits"][str(row.get("exit_reason") or "UNKNOWN")] += 1
    payload = {
        "generated_at": _now().isoformat(timespec="seconds"),
        "simulated_trades": len(entries),
        "closed_trades": len(exits),
        "realized_paper_pnl": round(sum(pnl_values), 2),
        "wins": sum(1 for value in pnl_values if value > 0.0),
        "losses": sum(1 for value in pnl_values if value < 0.0),
        "attribution_summary": {
            key: {
                "entries": value["entries"],
                "exits": value["exits"],
                "pnl_rupees": round(float(value["pnl_rupees"]), 2),
                "wins": value["wins"],
                "losses": value["losses"],
            }
            for key, value in sorted(attribution_summary.items())
        },
        "exit_reason_distribution": dict(Counter(str(row.get("exit_reason") or "UNKNOWN") for row in exits)),
        "lock_triggers": [
            row for row in _read_jsonl_tail(paths.operator_events, 200)
            if "LOCK" in str(row.get("event") or "") or "BLOCK" in str(row.get("event") or "")
        ],
        "mismatch_recovery_incidents": _read_jsonl_tail(paths.reconciliation_events, 25) + _read_jsonl_tail(paths.emergency_flatten_events, 25),
        "day_by_day": {
            day: {
                "trades": payload["trades"],
                "pnl_rupees": round(payload["pnl_rupees"], 2),
                "exit_reasons": dict(payload["exits"]),
            }
            for day, payload in sorted(by_day.items())
        },
    }
    if write:
        _write_json(paths.paper_report, payload)
    return payload


def build_paper_live_validation_report(
    paths: OpsPaths | None = None,
    *,
    session_date: str | None = None,
    write: bool = True,
) -> dict[str, Any]:
    paths = paths or OpsPaths()
    rows = _read_jsonl(paths.validation_decisions)
    available_sessions = sorted({str(row.get("session_date") or "") for row in rows if row.get("session_date")})
    target_session = session_date or (available_sessions[-1] if available_sessions else _now().date().isoformat())
    session_rows = [row for row in rows if str(row.get("session_date") or "") == target_session]
    top_no_trade_reasons = Counter(
        str(row.get("block_reason") or "UNKNOWN")
        for row in session_rows
        if str(row.get("decision") or "") != "TRADE"
    )
    states = Counter(str(row.get("market_state") or "UNKNOWN") for row in session_rows)
    playbooks = Counter(str(row.get("playbook") or "UNKNOWN") for row in session_rows)
    data_rows = [row.get("data") for row in session_rows if isinstance(row.get("data"), dict)]
    candle_counts = [int(data.get("candle_count") or 0) for data in data_rows]
    data_first = [str(data.get("first_timestamp")) for data in data_rows if data.get("first_timestamp")]
    data_last = [str(data.get("last_timestamp")) for data in data_rows if data.get("last_timestamp")]
    snapshot_statuses = Counter(str(data.get("snapshot_status") or "UNKNOWN") for data in data_rows)
    anomalies = [
        {
            "timestamp": row.get("timestamp"),
            "block_reason": row.get("block_reason"),
            "anomaly": anomaly,
        }
        for row in session_rows
        for anomaly in (row.get("anomalies") if isinstance(row.get("anomalies"), list) else [])
    ]
    payload = {
        "generated_at": _now().isoformat(timespec="seconds"),
        "session_date": target_session,
        "available_sessions": available_sessions,
        "total_decisions": len(session_rows),
        "trade_count": sum(1 for row in session_rows if row.get("decision") == "TRADE"),
        "no_trade_count": sum(1 for row in session_rows if row.get("decision") != "TRADE"),
        "top_no_trade_reasons": dict(top_no_trade_reasons.most_common(20)),
        "market_state_distribution": dict(states),
        "playbook_candidate_distribution": dict(playbooks),
        "data_availability_summary": {
            "snapshot_status_distribution": dict(snapshot_statuses),
            "ready_count": snapshot_statuses.get("READY", 0),
            "insufficient_data_count": snapshot_statuses.get("INSUFFICIENT_DATA", 0),
            "unavailable_count": snapshot_statuses.get("UNAVAILABLE", 0),
            "min_candle_count": min(candle_counts) if candle_counts else 0,
            "max_candle_count": max(candle_counts) if candle_counts else 0,
            "latest_candle_count": candle_counts[-1] if candle_counts else 0,
            "first_data_timestamp_seen": min(data_first) if data_first else None,
            "last_data_timestamp_seen": max(data_last) if data_last else None,
            "latest_snapshot_status": str(data_rows[-1].get("snapshot_status") or "UNKNOWN") if data_rows else "UNKNOWN",
        },
        "anomaly_count": len(anomalies),
        "anomalies_by_type": dict(Counter(str(item.get("anomaly", {}).get("type") or "UNKNOWN") for item in anomalies)),
        "anomalies": anomalies[-50:],
        "sample_decision_logs": session_rows[-10:],
    }
    if write:
        _write_json(paths.validation_report, payload)
    return payload


def _normalized_validation_row(
    row: dict[str, Any],
    *,
    config: RuntimeConfig,
) -> dict[str, Any]:
    timestamp = _parse_dt(row.get("timestamp"))
    paper_candidate_decision = row.get("paper_candidate_decision")
    paper_candidate_reason = row.get("paper_candidate_reason")
    mapped_strategy = row.get("mapped_strategy")
    spread_status = row.get("spread_construction_status")
    paper_experiment_window_allows = row.get("paper_experiment_window_allows")
    if timestamp is not None and paper_experiment_window_allows is None:
        start_t = _parse_time(config.allowed_entry_start_hhmm, time(9, 30))
        end_t = _parse_time(config.paper_experiment_entry_end_hhmm, time(14, 30))
        paper_experiment_window_allows = bool(start_t <= timestamp.time() <= end_t)
    if paper_candidate_decision is None:
        market_state = str(row.get("market_state") or "")
        tradability = str(row.get("tradability_class") or "")
        failure_type = str(row.get("failure_type") or "")
        bearish_score = float(row.get("bearish_trade_score") or 0.0)
        no_trade_score = float(row.get("no_trade_score") or 0.0)
        eligible = bool(
            market_state in {"TRANSITION", "TREND_DOWN", "DIRECTIONAL_BALANCE"}
            and tradability == "TRADABLE"
            and failure_type in {"FAILED_RECLAIM", "FAILED_BOUNCE", "FAILED_BREAKOUT", "ACCEPTED_BREAKDOWN"}
            and bearish_score >= config.paper_override_bearish_score_threshold
            and bearish_score > no_trade_score + config.paper_override_margin
            and paper_experiment_window_allows is not False
        )
        paper_candidate_decision = "NO_TRADE"
        if eligible:
            paper_candidate_reason = "LEGACY_DECISION_ROW_REQUIRES_REPLAY"
            mapped_strategy = StrategyType.BEAR_CALL_CREDIT_SPREAD.value
            spread_status = "NOT_EVALUATED"
        else:
            paper_candidate_reason = "NOT_EVALUATED"
            spread_status = "NOT_EVALUATED"
    return dict(row) | {
        "paper_candidate_decision": paper_candidate_decision,
        "paper_candidate_reason": paper_candidate_reason,
        "mapped_strategy": mapped_strategy,
        "spread_construction_status": spread_status,
        "paper_experiment_window_allows": paper_experiment_window_allows,
    }


def build_near_trade_candidates_csv(
    paths: OpsPaths | None = None,
    *,
    config: RuntimeConfig | None = None,
    write: bool = True,
) -> list[dict[str, Any]]:
    paths = paths or OpsPaths()
    config = config or load_runtime_config(paths)
    rows = [
        _normalized_validation_row(row, config=config)
        for row in _read_jsonl(paths.validation_decisions)
        if str(row.get("event_type") or "DECISION") == "DECISION"
    ]
    opportunities: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    current_key: tuple[Any, ...] | None = None
    current_ts: datetime | None = None
    merge_gap_sec = 300.0

    for row in rows:
        timestamp = _parse_dt(row.get("timestamp"))
        if timestamp is None:
            continue
        key = (
            str(row.get("session_date") or timestamp.date().isoformat()),
            str(row.get("market_state") or ""),
            str(row.get("tradability_class") or ""),
            str(row.get("failure_type") or ""),
            str(row.get("subtype") or ""),
            str(row.get("v83_decision") or row.get("decision") or "NO_TRADE"),
            str(row.get("v83_rejection_reason") or row.get("block_reason") or "NONE"),
            str(row.get("paper_candidate_decision") or "NO_TRADE"),
            str(row.get("paper_candidate_reason") or "NOT_EVALUATED"),
            str(row.get("mapped_strategy") or ""),
            str(row.get("spread_construction_status") or "NOT_EVALUATED"),
            bool(row.get("would_create_paper_trade", False)),
        )
        if (
            current is None
            or current_key != key
            or current_ts is None
            or (timestamp - current_ts).total_seconds() > merge_gap_sec
        ):
            if current is not None:
                opportunities.append(current)
            current = {
                "date": str(row.get("session_date") or timestamp.date().isoformat()),
                "first_seen_time": timestamp.isoformat(),
                "last_seen_time": timestamp.isoformat(),
                "market_state": row.get("market_state"),
                "tradability_class": row.get("tradability_class"),
                "failure_type": row.get("failure_type"),
                "subtype": row.get("subtype"),
                "bearish_trade_score": float(row.get("bearish_trade_score") or 0.0),
                "no_trade_score": float(row.get("no_trade_score") or 0.0),
                "v83_decision": row.get("v83_decision") or row.get("decision") or "NO_TRADE",
                "v83_rejection_reason": row.get("v83_rejection_reason") or row.get("block_reason") or "NONE",
                "paper_candidate_decision": row.get("paper_candidate_decision") or "NO_TRADE",
                "paper_candidate_reason": row.get("paper_candidate_reason") or "NOT_EVALUATED",
                "mapped_strategy": row.get("mapped_strategy"),
                "spread_construction_status": row.get("spread_construction_status") or "NOT_EVALUATED",
                "would_create_paper_trade": bool(row.get("would_create_paper_trade", False)),
            }
            current_key = key
        else:
            current["last_seen_time"] = timestamp.isoformat()
            current["bearish_trade_score"] = round(max(float(current.get("bearish_trade_score") or 0.0), float(row.get("bearish_trade_score") or 0.0)), 4)
            current["no_trade_score"] = round(min(float(current.get("no_trade_score") or 0.0), float(row.get("no_trade_score") or 0.0)), 4)
            current["would_create_paper_trade"] = bool(current.get("would_create_paper_trade") or row.get("would_create_paper_trade", False))
        current_ts = timestamp
    if current is not None:
        opportunities.append(current)

    if write:
        paths.near_trade_candidates_csv.parent.mkdir(parents=True, exist_ok=True)
        with paths.near_trade_candidates_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "date",
                    "first_seen_time",
                    "last_seen_time",
                    "market_state",
                    "tradability_class",
                    "failure_type",
                    "subtype",
                    "bearish_trade_score",
                    "no_trade_score",
                    "v83_decision",
                    "v83_rejection_reason",
                    "paper_candidate_decision",
                    "paper_candidate_reason",
                    "mapped_strategy",
                    "spread_construction_status",
                    "would_create_paper_trade",
                ],
            )
            writer.writeheader()
            writer.writerows(opportunities)
    return opportunities


def build_paper_time_window_audit(
    paths: OpsPaths | None = None,
    *,
    config: RuntimeConfig | None = None,
    write: bool = True,
) -> dict[str, Any]:
    paths = paths or OpsPaths()
    config = config or load_runtime_config(paths)
    rows = [
        _normalized_validation_row(row, config=config)
        for row in _read_jsonl(paths.validation_decisions)
        if str(row.get("event_type") or "DECISION") == "DECISION"
    ]
    time_window_rows = [
        row for row in rows
        if str(row.get("v83_rejection_reason") or row.get("block_reason") or "") in {"TIME_WINDOW", "ENTRY_TIME_INVALID"}
    ]
    allowed_count = sum(1 for row in time_window_rows if row.get("paper_experiment_window_allows") is True)
    blocked_count = sum(1 for row in time_window_rows if row.get("paper_experiment_window_allows") is False)
    payload = {
        "generated_at": _now().isoformat(timespec="seconds"),
        "paper_experiment_entry_end_hhmm": config.paper_experiment_entry_end_hhmm,
        "total_time_window_rejects": len(time_window_rows),
        "paper_window_would_allow_count": allowed_count,
        "paper_window_would_still_block_count": blocked_count,
        "by_subtype": dict(Counter(str(row.get("subtype") or "UNKNOWN") for row in time_window_rows)),
        "sample_rows": time_window_rows[-25:],
    }
    if write:
        _write_json(paths.paper_time_window_audit, payload)
    return payload


def build_paper_context_override_report(
    paths: OpsPaths | None = None,
    *,
    config: RuntimeConfig | None = None,
    write: bool = True,
) -> dict[str, Any]:
    paths = paths or OpsPaths()
    config = config or load_runtime_config(paths)
    validation_rows = [
        _normalized_validation_row(row, config=config)
        for row in _read_jsonl(paths.validation_decisions)
        if str(row.get("event_type") or "DECISION") == "DECISION"
    ]
    opportunities = build_near_trade_candidates_csv(paths=paths, config=config, write=write)
    time_window_audit = build_paper_time_window_audit(paths=paths, config=config, write=write)
    paper_report = build_paper_live_report(paths=paths, write=False)

    override_candidates = [
        row for row in validation_rows
        if str(row.get("paper_candidate_reason") or "NOT_EVALUATED") not in {"NOT_EVALUATED", "V83_APPROVED"}
        or str(row.get("paper_trade_attribution") or "") == "PAPER_CONTEXT_OVERRIDE"
    ]
    override_trades = [row for row in validation_rows if str(row.get("paper_trade_attribution") or "") == "PAPER_CONTEXT_OVERRIDE" and bool(row.get("would_create_paper_trade"))]
    v83_approved_trades = [row for row in validation_rows if str(row.get("paper_trade_attribution") or "") == "V83_APPROVED" and bool(row.get("would_create_paper_trade"))]
    override_rejections = Counter(
        str(row.get("paper_candidate_reason") or "UNKNOWN")
        for row in override_candidates
        if not bool(row.get("would_create_paper_trade"))
    )
    override_opportunities = [
        row for row in opportunities
        if str(row.get("paper_candidate_reason") or "NOT_EVALUATED") not in {"NOT_EVALUATED", "V83_APPROVED"}
    ]
    trade_creation_rate = (
        len([row for row in override_opportunities if bool(row.get("would_create_paper_trade"))]) / len(override_opportunities)
        if override_opportunities else 0.0
    )
    low_quality_flow_appears = bool(override_opportunities and trade_creation_rate < 0.25)
    payload = {
        "generated_at": _now().isoformat(timespec="seconds"),
        "paper_override_config": {
            "paper_override_bearish_score_threshold": config.paper_override_bearish_score_threshold,
            "paper_override_margin": config.paper_override_margin,
            "paper_experiment_entry_end_hhmm": config.paper_experiment_entry_end_hhmm,
        },
        "v83_approved_paper_trades": len(v83_approved_trades),
        "paper_context_override_candidates": len(override_candidates),
        "paper_context_override_paper_trades": len(override_trades),
        "rejection_reasons_after_override": dict(override_rejections),
        "pnl_split": paper_report.get("attribution_summary", {}),
        "low_quality_flow_appears": low_quality_flow_appears,
        "override_opportunity_windows": len(override_opportunities),
        "override_trade_creation_rate": round(trade_creation_rate, 4),
        "near_trade_candidates_path": str(paths.near_trade_candidates_csv),
        "paper_time_window_audit_path": str(paths.paper_time_window_audit),
        "paper_time_window_audit": time_window_audit,
    }
    if write:
        _write_json(paths.paper_context_override_report, payload)
    return payload


def build_promotion_gates(
    *,
    paths: OpsPaths | None = None,
    config: RuntimeConfig | None = None,
    health: dict[str, Any] | None = None,
) -> dict[str, Any]:
    paths = paths or OpsPaths()
    config = config or load_runtime_config(paths)
    health = health or build_unified_health(config=config, paths=paths)
    shadow = build_shadow_live_report(paths, write=False)
    paper = build_paper_live_report(paths, write=False)
    operator_visibility_ok = paths.operator_status_report.exists() or bool(health)
    research_to_shadow = {
        "passed": bool(config.v83_frozen and health and operator_visibility_ok),
        "checks": {
            "v83_frozen": config.v83_frozen,
            "health_model_implemented": bool(health),
            "execution_gates_implemented": True,
            "ui_mode_status_visible": operator_visibility_ok,
        },
    }
    shadow_to_paper = {
        "passed": bool(
            shadow["sessions_run"] >= 5
            and int(shadow["would_trade_count"]) <= max(int(shadow["sessions_run"]) * config.risk.max_trades_per_day, 0)
            and health.get("status") != HealthStatus.BLOCKED.value
            and operator_visibility_ok
        ),
        "checks": {
            "shadow_sessions_completed": shadow["sessions_run"],
            "requires_at_least_5_shadow_sessions": shadow["sessions_run"] >= 5,
            "no_unstable_routing_bursts": int(shadow["would_trade_count"]) <= max(int(shadow["sessions_run"]) * config.risk.max_trades_per_day, 0),
            "no_unexplained_health_flapping": health.get("status") != HealthStatus.BLOCKED.value,
            "no_missing_status_visibility": operator_visibility_ok,
        },
    }
    paper_to_micro = {
        "passed": bool(
            paper["closed_trades"] >= 5
            and not load_paper_position(paths)
            and health.get("status") != HealthStatus.BLOCKED.value
            and not (_read_json(paths.recovery_state).get("active"))
        ),
        "checks": {
            "paper_sessions_or_closed_trades_completed": paper["closed_trades"],
            "requires_at_least_5_paper_sessions": paper["closed_trades"] >= 5,
            "no_phantom_paper_positions": not bool(load_paper_position(paths)),
            "daily_locks_visible": True,
            "restart_recovery_visible": True,
            "reconciliation_visible": True,
            "recovery_state_clear": not bool(_read_json(paths.recovery_state).get("active")),
        },
    }
    return {
        "research_to_shadow_live": research_to_shadow,
        "shadow_live_to_paper_live": shadow_to_paper,
        "paper_live_to_micro_live": paper_to_micro,
        "auto_promote": False,
    }


def build_operator_status_report(
    *,
    paths: OpsPaths | None = None,
    snapshot: MarketSnapshot | None = None,
    broker_positions: Iterable[dict[str, Any]] | None = None,
    write: bool = True,
) -> dict[str, Any]:
    paths = paths or OpsPaths()
    config = load_runtime_config(paths)
    state = load_runtime_state(paths, config)
    reconcile = reconcile_positions(paths=paths, mode=config.mode, broker_positions=broker_positions)
    health = build_unified_health(config=config, paths=paths, snapshot=snapshot)
    recovery = _read_json(paths.recovery_state) or {"active": False, "reason": None}
    shadow = build_shadow_live_report(paths, write=True)
    paper = build_paper_live_report(paths, write=True)
    validation = build_paper_live_validation_report(paths, write=True)
    paper_override = build_paper_context_override_report(paths, config=config, write=True)
    events = _read_jsonl_tail(paths.operator_events, 100)
    flatten_events = _read_jsonl_tail(paths.emergency_flatten_events, 25)
    payload = {
        "generated_at": _now().isoformat(timespec="seconds"),
        "runtime_config": config.to_dict(),
        "runtime_state": state,
        "health": health,
        "broker_sync_status": reconcile,
        "recovery_status": recovery,
        "shadow_live_report_path": str(paths.shadow_report),
        "paper_live_report_path": str(paths.paper_report),
        "paper_live_validation_report_path": str(paths.validation_report),
        "paper_context_override_report_path": str(paths.paper_context_override_report),
        "near_trade_candidates_csv_path": str(paths.near_trade_candidates_csv),
        "paper_time_window_audit_path": str(paths.paper_time_window_audit),
        "operator_status_report_path": str(paths.operator_status_report),
        "operator_status": {
            "health_incidents": [
                reason for reason in health.get("block_reasons", [])
                if "HEALTH" in reason or reason
            ],
            "stale_data_incidents": health.get("components", {}).get("data_pipeline_health", {}).get("block_reasons", []),
            "broker_mismatch_incidents": _read_jsonl_tail(paths.reconciliation_events, 25),
            "flatten_all_invocations": flatten_events[-25:],
            "manual_blocks": [
                row for row in events[-50:]
                if "BLOCK" in str(row.get("event") or "") or "LOCK" in str(row.get("event") or "")
            ],
            "validation_anomalies": validation.get("anomalies", []),
        },
        "promotion_gates": build_promotion_gates(paths=paths, config=config, health=health),
        "shadow_summary": shadow,
        "paper_summary": paper,
        "paper_live_validation_summary": validation,
        "paper_context_override_summary": paper_override,
    }
    if write:
        _write_json(paths.operator_status_report, payload)
    return payload
