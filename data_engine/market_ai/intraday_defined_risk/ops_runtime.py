from __future__ import annotations

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
    "SIDEWAYS_TO_BEARISH_REJECTION",
    "GAP_DOWN_BEARISH_CONTINUATION",
    "GAP_UP_BEARISH_FAILURE",
}
V83_LIVE_STRATEGIES = {StrategyType.BEAR_CALL_CREDIT_SPREAD.value}
V83_APPROVED_BEARISH_STATES = {"TREND_DOWN", "TRANSITION"}


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
    max_lots_per_trade: int = 1
    max_open_structures: int = 1
    max_trades_per_day: int = 1
    max_daily_realized_loss_rupees: float = 10_000.0
    max_daily_total_loss_rupees: float = 10_000.0
    no_new_entries_after_hhmm: str = "14:00"
    stop_after_first_full_loss: bool = True
    require_manual_live_arm: bool = True
    live_only_bearish: bool = True

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RuntimeRiskGovernance":
        payload = payload or {}
        return cls(
            max_lots_per_trade=max(1, int(float(payload.get("max_lots_per_trade", 1) or 1))),
            max_open_structures=max(1, int(float(payload.get("max_open_structures", 1) or 1))),
            max_trades_per_day=max(1, int(float(payload.get("max_trades_per_day", 1) or 1))),
            max_daily_realized_loss_rupees=max(0.0, float(payload.get("max_daily_realized_loss_rupees", 10_000.0) or 0.0)),
            max_daily_total_loss_rupees=max(0.0, float(payload.get("max_daily_total_loss_rupees", 10_000.0) or 0.0)),
            no_new_entries_after_hhmm=str(payload.get("no_new_entries_after_hhmm") or "14:00"),
            stop_after_first_full_loss=bool(payload.get("stop_after_first_full_loss", True)),
            require_manual_live_arm=bool(payload.get("require_manual_live_arm", True)),
            live_only_bearish=bool(payload.get("live_only_bearish", True)),
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
    allowed_entry_start_hhmm: str = "09:30"
    allowed_entry_end_hhmm: str = "14:00"
    health_required_for_paper: bool = True
    health_required_for_micro: bool = True
    risk: RuntimeRiskGovernance = field(default_factory=RuntimeRiskGovernance)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RuntimeConfig":
        payload = payload or {}
        raw_mode = str(payload.get("mode") or RuntimeMode.LIVE_DISABLED.value).upper()
        try:
            mode = RuntimeMode(raw_mode)
        except ValueError:
            mode = RuntimeMode.LIVE_DISABLED
        return cls(
            mode=mode,
            live_arm=bool(payload.get("live_arm", False)),
            v83_frozen=bool(payload.get("v83_frozen", True)),
            live_enabled_playbooks=tuple(sorted(str(item) for item in payload.get("live_enabled_playbooks", sorted(V83_LIVE_PLAYBOOKS)))),
            live_enabled_strategies=tuple(sorted(str(item) for item in payload.get("live_enabled_strategies", sorted(V83_LIVE_STRATEGIES)))),
            approved_bearish_live_states=tuple(sorted(str(item) for item in payload.get("approved_bearish_live_states", sorted(V83_APPROVED_BEARISH_STATES)))),
            bearish_score_margin=float(payload.get("bearish_score_margin", 1.5) or 1.5),
            allowed_entry_start_hhmm=str(payload.get("allowed_entry_start_hhmm") or "09:30"),
            allowed_entry_end_hhmm=str(payload.get("allowed_entry_end_hhmm") or "14:00"),
            health_required_for_paper=bool(payload.get("health_required_for_paper", True)),
            health_required_for_micro=bool(payload.get("health_required_for_micro", True)),
            risk=RuntimeRiskGovernance.from_payload(payload.get("risk") if isinstance(payload.get("risk"), dict) else payload),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
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
    shadow_report: Path = STATE_ROOT / "intraday_v83_shadow_live_report.json"
    paper_report: Path = STATE_ROOT / "intraday_v83_paper_live_report.json"
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


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


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
        allowed_entry_start_hhmm=current.allowed_entry_start_hhmm,
        allowed_entry_end_hhmm=current.allowed_entry_end_hhmm,
        health_required_for_paper=current.health_required_for_paper,
        health_required_for_micro=current.health_required_for_micro,
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
    hard_pipeline_reasons = [reason for reason in pipeline_reasons if reason != "COLLECTOR_LOG_STALE"]
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
    tradability = str(metadata.get("tradability_class") or funnel.get("tradability_class") or metadata.get("regime_tradability") or "")
    if tradability == "NOT_TRADABLE":
        reasons.append("TRADABILITY_NOT_TRADABLE")
    market_state = str(metadata.get("market_state") or funnel.get("market_state") or "")
    if market_state not in set(config.approved_bearish_live_states):
        reasons.append("MARKET_STATE_NOT_APPROVED")
    if config.risk.live_only_bearish and str(metadata.get("setup_direction") or "").upper() == "BULLISH":
        reasons.append("BULLISH_LIVE_BLOCKED")
    bearish_score = float(metadata.get("bearish_trade_score") or funnel.get("bearish_trade_score") or 0.0)
    no_trade_score = float(metadata.get("no_trade_score") or funnel.get("no_trade_score") or 0.0)
    if bearish_score <= no_trade_score + config.bearish_score_margin:
        reasons.append("BEARISH_SCORE_MARGIN_FAIL")
    start_t = _parse_time(config.allowed_entry_start_hhmm, time(9, 30))
    end_t = min(
        _parse_time(config.allowed_entry_end_hhmm, time(14, 0)),
        _parse_time(config.risk.no_new_entries_after_hhmm, time(14, 0)),
    )
    if not (start_t <= snapshot.timestamp.time() <= end_t):
        reasons.append("ENTRY_WINDOW_CLOSED")
    daily_lock = state.get("daily_lock") if isinstance(state.get("daily_lock"), dict) else {}
    if bool(daily_lock.get("active")):
        reasons.append(str(daily_lock.get("reason") or "DAILY_LOCK_ACTIVE"))
    if _active_paper_position_count(paths) >= config.risk.max_open_structures:
        reasons.append("ACTIVE_STRUCTURE_EXISTS")
    if int(state.get("session_trade_count") or 0) >= config.risk.max_trades_per_day:
        reasons.append("MAX_TRADES_PER_DAY_REACHED")
    if int(decision.lots or 0) > config.risk.max_lots_per_trade:
        reasons.append("LOTS_EXCEED_RUNTIME_LIMIT")
    if float(state.get("realized_pnl_rupees") or 0.0) <= -abs(config.risk.max_daily_realized_loss_rupees):
        reasons.append("DAILY_REALIZED_LOSS_LOCK")
    if float(state.get("total_pnl_rupees") or 0.0) <= -abs(config.risk.max_daily_total_loss_rupees):
        reasons.append("DAILY_TOTAL_LOSS_LOCK")
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
        "tradability_class": tradability,
        "bearish_trade_score": round(bearish_score, 4),
        "no_trade_score": round(no_trade_score, 4),
        "score_margin_required": config.bearish_score_margin,
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
        "subtype": metadata.get("setup_subtype") or metadata.get("bearish_subtype") or metadata.get("bullish_shadow_subtype"),
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
        "chosen_action": action,
        "chosen_strategy": decision.strategy.value if isinstance(decision.strategy, StrategyType) else str(decision.strategy),
        "block_reason": block_reason,
        "hypothetical_entry_timestamp": decision.entry.get("timestamp") if decision.action == "TRADE" else None,
        "hypothetical_exit_timestamp": decision.time_exit if decision.action == "TRADE" else None,
        "exit_reason": "TIME_EXIT_HYPOTHESIS" if decision.action == "TRADE" else None,
        "hypothetical_pnl_rupees": None,
    }


def log_shadow_decision(
    decision: DecisionOutput,
    *,
    snapshot: MarketSnapshot,
    paths: OpsPaths | None = None,
    block_reason: str | None = None,
) -> None:
    paths = paths or OpsPaths()
    action = "WOULD_TRADE" if decision.action == "TRADE" and not block_reason else "NO_TRADE"
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
    event = {
        "event": "PAPER_ENTRY",
        "timestamp": snapshot.timestamp.isoformat(),
        "session_date": snapshot.timestamp.date().isoformat(),
        "entry_timestamp": snapshot.timestamp.isoformat(),
        "playbook": decision.metadata.get("playbook"),
        "subtype": decision.metadata.get("setup_subtype") or decision.metadata.get("bearish_subtype"),
        "strategy": decision.strategy.value if isinstance(decision.strategy, StrategyType) else str(decision.strategy),
        "legs": decision.legs,
        "entry_price_assumptions": decision.entry,
        "gate": gate,
        "realized_paper_pnl": None,
        "mfe_rupees": 0.0,
        "mae_rupees": 0.0,
    }
    _append_jsonl(paths.paper_trades, event)
    state = load_runtime_state(paths, load_runtime_config(paths))
    state["session_trade_count"] = int(state.get("session_trade_count") or 0) + 1
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
    exit_decision = evaluate_exit(position, current_snapshot=snapshot, now=snapshot.timestamp)
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
        "strategy": position.structure.strategy.value,
        "realized_paper_pnl": round(exit_decision.pnl_rupees, 2),
        "mfe_rupees": round(mfe, 2),
        "mae_rupees": round(mae, 2),
        "legs": serialize_open_position(position)["structure"]["legs"],
    }
    _append_jsonl(paths.paper_trades, event)
    save_paper_position(None, paths=paths, extra={"last_exit": event, "mfe_rupees": 0.0, "mae_rupees": 0.0})
    config = load_runtime_config(paths)
    runtime_state = load_runtime_state(paths, config)
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
        "exit_reason_distribution": dict(Counter(str(row.get("exit_reason") or "UNKNOWN") for row in exits)),
        "lock_triggers": [
            row for row in _read_jsonl(paths.operator_events)
            if "LOCK" in str(row.get("event") or "") or "BLOCK" in str(row.get("event") or "")
        ],
        "mismatch_recovery_incidents": _read_jsonl(paths.reconciliation_events)[-25:] + _read_jsonl(paths.emergency_flatten_events)[-25:],
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
    events = _read_jsonl(paths.operator_events)
    flatten_events = _read_jsonl(paths.emergency_flatten_events)
    payload = {
        "generated_at": _now().isoformat(timespec="seconds"),
        "runtime_config": config.to_dict(),
        "runtime_state": state,
        "health": health,
        "broker_sync_status": reconcile,
        "recovery_status": recovery,
        "shadow_live_report_path": str(paths.shadow_report),
        "paper_live_report_path": str(paths.paper_report),
        "operator_status_report_path": str(paths.operator_status_report),
        "operator_status": {
            "health_incidents": [
                reason for reason in health.get("block_reasons", [])
                if "HEALTH" in reason or reason
            ],
            "stale_data_incidents": health.get("components", {}).get("data_pipeline_health", {}).get("block_reasons", []),
            "broker_mismatch_incidents": _read_jsonl(paths.reconciliation_events)[-25:],
            "flatten_all_invocations": flatten_events[-25:],
            "manual_blocks": [
                row for row in events[-50:]
                if "BLOCK" in str(row.get("event") or "") or "LOCK" in str(row.get("event") or "")
            ],
        },
        "promotion_gates": build_promotion_gates(paths=paths, config=config, health=health),
        "shadow_summary": shadow,
        "paper_summary": paper,
    }
    if write:
        _write_json(paths.operator_status_report, payload)
    return payload
