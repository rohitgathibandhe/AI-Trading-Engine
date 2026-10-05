"""Per-strategy REAL-MONEY promotion gate (runtime side).

Reads state/promotion_state.json (written by scripts/promotion_gate.py from the forward shadow-book
record) and answers one question the runtime asks before placing a REAL order: has THIS strategy
earned the right to trade real money?

Graduation is automatic and PER-STRATEGY: a structure trades real money only once the gate has marked
it ELIGIBLE_FOR_LIVE (>= N qualifying days of forward edge, and not currently drifting). Until then it
stays PAPER even when the runtime is live-armed — the proven strategies go live, the rest keep
paper-testing until they earn it, with zero manual intervention.

Safe default: if the file is missing/empty/unreadable, NOTHING is real-eligible (fail-closed — an
un-vetted strategy never reaches real money by accident). Cached, refreshed on file mtime change so a
fresh gate run takes effect within the next decision cycle without a restart.
"""
from __future__ import annotations

import json
from pathlib import Path

_STATE_FILE = Path(__file__).resolve().parent.parent / "state" / "promotion_state.json"
_cache: dict = {"mtime": -1.0, "eligible": frozenset()}


def _eligible() -> frozenset:
    try:
        mt = _STATE_FILE.stat().st_mtime
    except OSError:
        _cache["mtime"] = -1.0
        _cache["eligible"] = frozenset()
        return _cache["eligible"]
    if mt != _cache["mtime"]:
        try:
            data = json.loads(_STATE_FILE.read_text())
            _cache["eligible"] = frozenset(str(s) for s in (data.get("eligible_for_live") or []))
        except (OSError, ValueError, json.JSONDecodeError):
            _cache["eligible"] = frozenset()
        _cache["mtime"] = mt
    return _cache["eligible"]


def _name(strategy) -> str:
    return strategy.value if hasattr(strategy, "value") else str(strategy)


def real_money_eligible(strategy) -> bool:
    """True only if this strategy has cleared the promotion gate on the forward record."""
    return _name(strategy) in _eligible()


def eligible_set() -> frozenset:
    return _eligible()
