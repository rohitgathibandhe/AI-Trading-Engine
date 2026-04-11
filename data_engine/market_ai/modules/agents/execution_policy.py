from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict


_EXECUTION_MODE_ALIASES = {
    "ADVISOR": "ADVISOR",
    "ANALYZE": "ADVISOR",
    "ANALYSIS": "ADVISOR",
    "MONITOR": "MONITOR",
    "WATCH": "MONITOR",
    "MONITOR_ONLY": "MONITOR",
    "MANAGE": "MANAGE_ONLY",
    "MANAGE_ONLY": "MANAGE_ONLY",
    "PROTECT": "MANAGE_ONLY",
    "PROTECT_ONLY": "MANAGE_ONLY",
    "EXECUTION": "EXECUTION",
    "EXECUTE": "EXECUTION",
    "AUTO": "EXECUTION",
    "AUTO_EXECUTE": "EXECUTION",
}

_INTRADAY_MODE_ALIASES = {
    "EXECUTION": "EXECUTION",
    "EXECUTE": "EXECUTION",
    "AUTO": "EXECUTION",
    "AUTO_EXECUTE": "EXECUTION",
    "ADVISOR": "ADVISOR",
    "ANALYZE": "ADVISOR",
    "ANALYSIS": "ADVISOR",
}


def normalize_agent_execution_mode(value: Any, default: str = "EXECUTION") -> str:
    mode = str(value or "").strip().upper()
    if not mode:
        return default
    return _EXECUTION_MODE_ALIASES.get(mode, default)


def intraday_ai_mode_name(settings: Dict[str, Any]) -> str:
    mode = str((settings or {}).get("intraday_ai_mode", "ADVISOR")).strip().upper()
    if not mode:
        return "ADVISOR"
    return _INTRADAY_MODE_ALIASES.get(mode, "ADVISOR")


@dataclass(frozen=True)
class ExecutionPolicy:
    mode: str
    trade_mode: str
    analysis_allowed: bool
    monitoring_allowed: bool
    manage_existing_allowed: bool
    new_entries_allowed: bool
    any_execution_allowed: bool
    paper_execution_allowed: bool
    live_execution_allowed: bool
    intraday_enabled: bool
    intraday_ai_mode: str
    intraday_new_entries_allowed: bool
    intraday_manage_existing_allowed: bool
    batman_new_entries_allowed: bool
    batman_manage_existing_allowed: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_execution_policy(settings: Dict[str, Any], trade_mode: str) -> Dict[str, Any]:
    settings = settings or {}
    runtime_trade_mode = str(trade_mode or settings.get("trade_mode") or "").strip().lower()
    mode = normalize_agent_execution_mode(settings.get("agent_execution_mode"), default="EXECUTION")

    analysis_allowed = True
    monitoring_allowed = mode in {"MONITOR", "MANAGE_ONLY", "EXECUTION"}
    manage_existing_allowed = mode in {"MANAGE_ONLY", "EXECUTION"}
    new_entries_allowed = mode == "EXECUTION"
    any_execution_allowed = manage_existing_allowed or new_entries_allowed
    paper_execution_allowed = any_execution_allowed and runtime_trade_mode == "paper"
    live_execution_allowed = any_execution_allowed and runtime_trade_mode == "live"

    intraday_enabled = bool(settings.get("intraday_ai_enabled", True))
    intraday_mode = intraday_ai_mode_name(settings)
    intraday_mode_allows_new_entries = intraday_enabled and intraday_mode == "EXECUTION"
    intraday_mode_allows_manage_existing = intraday_enabled and manage_existing_allowed
    if runtime_trade_mode == "paper":
        intraday_trade_mode_enabled = bool(settings.get("intraday_ai_auto_deploy_paper_enabled", True))
    elif runtime_trade_mode == "live":
        intraday_trade_mode_enabled = bool(settings.get("intraday_ai_auto_deploy_live_enabled", False))
    else:
        intraday_trade_mode_enabled = False

    policy = ExecutionPolicy(
        mode=mode,
        trade_mode=runtime_trade_mode,
        analysis_allowed=analysis_allowed,
        monitoring_allowed=monitoring_allowed,
        manage_existing_allowed=manage_existing_allowed,
        new_entries_allowed=new_entries_allowed,
        any_execution_allowed=any_execution_allowed,
        paper_execution_allowed=paper_execution_allowed,
        live_execution_allowed=live_execution_allowed,
        intraday_enabled=intraday_enabled,
        intraday_ai_mode=intraday_mode,
        intraday_new_entries_allowed=bool(
            new_entries_allowed and intraday_mode_allows_new_entries and intraday_trade_mode_enabled
        ),
        intraday_manage_existing_allowed=bool(intraday_mode_allows_manage_existing and any_execution_allowed),
        batman_new_entries_allowed=bool(new_entries_allowed),
        batman_manage_existing_allowed=bool(manage_existing_allowed),
    )
    return policy.to_dict()
