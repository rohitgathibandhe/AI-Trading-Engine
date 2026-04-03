from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from market_ai.modules.agents.execution_recovery_guard import (
    ExecutionJournal,
    ExecutionRecoveryConfig,
    ExecutionRecoveryGuard,
)
from market_ai.strategies import LegSide, OptionLeg, OptionType, StrategyType


def _ist_dt(day: int, hour: int = 10, minute: int = 0) -> datetime:
    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    return datetime(2026, 1, day, hour, minute, tzinfo=tz)


def _guard(tmp_path: Path, **kwargs) -> ExecutionRecoveryGuard:
    return ExecutionRecoveryGuard(
        config=ExecutionRecoveryConfig(**kwargs),
        status_path=tmp_path / "execution_recovery_status.json",
        logger=None,
    )


def _journal(tmp_path: Path) -> ExecutionJournal:
    return ExecutionJournal(journal_path=tmp_path / "execution_journal.jsonl", logger=None)


def _broker_leg(*, expiry: str, strike: float, opt: OptionType, side: LegSide, qty: int) -> OptionLeg:
    return OptionLeg(
        symbol="NIFTY",
        expiry=date.fromisoformat(expiry),
        strike=strike,
        option_type=opt,
        side=side,
        quantity=qty,
        entry_price=100.0,
        strategy_type=StrategyType.NONE,
        security_id="",
    )


def test_execution_journal_tracks_active_basket_and_terminal_events(tmp_path: Path) -> None:
    j = _journal(tmp_path)
    t = _ist_dt(5)
    legs = [
        {"expiry": "2026-01-29", "strike": 22000, "option_type": "CALL", "side": "SELL", "quantity": 65, "entry_price": 100.0},
        {"expiry": "2026-01-29", "strike": 21000, "option_type": "PUT", "side": "BUY", "quantity": 65, "entry_price": 80.0},
    ]
    j.record("BKM_OPEN_BEGIN", {"strategy": "BATMAN_BKM", "op_id": "o1", "expiry": "2026-01-29", "legs": legs}, when=t)
    j.record("BKM_OPEN_SUCCESS", {"strategy": "BATMAN_BKM", "op_id": "o1", "expiry": "2026-01-29", "legs": legs}, when=t)
    j.record("BKM_CLOSE_BEGIN", {"strategy": "BATMAN_BKM", "op_id": "c1", "expiry": "2026-01-29", "legs": legs}, when=t)
    j.record("BKM_CLOSE_SUCCESS", {"strategy": "BATMAN_BKM", "op_id": "c1", "expiry": "2026-01-29", "legs": legs}, when=t)

    summary = j.analyze_bkm(lookback_days=45, when=t)
    assert summary["unresolved_ops"] == []
    assert summary["failed_ops"] == []
    assert summary["active_baskets"] == {}


def test_execution_journal_detects_unresolved_and_failed_ops(tmp_path: Path) -> None:
    j = _journal(tmp_path)
    t = _ist_dt(6)
    j.record("BKM_OPEN_BEGIN", {"strategy": "BATMAN_BKM", "op_id": "o1", "expiry": "2026-01-29"}, when=t)
    j.record("BKM_CLOSE_BEGIN", {"strategy": "BATMAN_BKM", "op_id": "c1", "expiry": "2026-01-29"}, when=t)
    j.record("BKM_CLOSE_FAIL", {"strategy": "BATMAN_BKM", "op_id": "c1", "expiry": "2026-01-29"}, when=t)

    summary = j.analyze_bkm(lookback_days=45, when=t)
    assert len(summary["unresolved_ops"]) == 1
    assert summary["unresolved_ops"][0]["op_id"] == "o1"
    assert len(summary["failed_ops"]) == 1
    assert summary["failed_ops"][0]["op_id"] == "c1"


def test_guard_locks_on_unresolved_journal_ops(tmp_path: Path) -> None:
    g = _guard(tmp_path)
    t = _ist_dt(7)
    out = g.evaluate_startup(
        local_bkm_state={},
        broker_legs=[],
        journal_summary={"unresolved_ops": [{"op_id": "x"}], "failed_ops": [], "active_baskets": {}},
        when=t,
    )
    assert out["ok"] is False
    assert out["locked"] is True
    assert out["reason"] == "EXEC_JOURNAL_UNRESOLVED_OP"
    assert g.snapshot()["status"] == "LOCKED"


def test_guard_resume_ready_when_local_broker_and_journal_match(tmp_path: Path) -> None:
    g = _guard(tmp_path)
    g.state.hard_lock = True
    t = _ist_dt(8)
    active = {
        "2026-01-29": {
            "expiry": "2026-01-29",
            "legs": [
                {"expiry": "2026-01-29", "strike": 22000, "option_type": "CALL", "side": "SELL", "quantity": 65},
                {"expiry": "2026-01-29", "strike": 21000, "option_type": "PUT", "side": "BUY", "quantity": 65},
            ],
            "meta": {"net_credit": 650.0, "margin_required": 1000000.0, "credit_pct": 0.065},
        }
    }
    broker = [
        _broker_leg(expiry="2026-01-29", strike=22000, opt=OptionType.CALL, side=LegSide.SELL, qty=65),
        _broker_leg(expiry="2026-01-29", strike=21000, opt=OptionType.PUT, side=LegSide.BUY, qty=65),
    ]
    out = g.evaluate_startup(
        local_bkm_state={"2026-01-29": {"status": "OPEN"}},
        broker_legs=broker,
        journal_summary={"unresolved_ops": [], "failed_ops": [], "active_baskets": active, "events_scanned": 4},
        when=t,
    )
    assert out["ok"] is True
    assert out["reason"] == "RESUME_READY"
    snap = g.snapshot()
    assert snap["status"] == "OK"
    assert snap["hard_lock"] is False
    assert snap["last_details"]["startup"] == "RESUME_READY"


def test_guard_locks_on_broker_journal_mismatch(tmp_path: Path) -> None:
    g = _guard(tmp_path)
    t = _ist_dt(9)
    active = {
        "2026-01-29": {
            "expiry": "2026-01-29",
            "legs": [
                {"expiry": "2026-01-29", "strike": 22000, "option_type": "CALL", "side": "SELL", "quantity": 65},
            ],
        }
    }
    broker = [_broker_leg(expiry="2026-01-29", strike=22000, opt=OptionType.CALL, side=LegSide.SELL, qty=130)]
    out = g.evaluate_startup(
        local_bkm_state={"2026-01-29": {"status": "OPEN"}},
        broker_legs=broker,
        journal_summary={"unresolved_ops": [], "failed_ops": [], "active_baskets": active},
        when=t,
    )
    assert out["ok"] is False
    assert out["reason"] == "EXEC_RECOVERY_BROKER_POSITION_MISMATCH"
    assert g.snapshot()["status"] == "LOCKED"


def test_guard_allows_external_broker_positions_when_bkm_is_flat(tmp_path: Path) -> None:
    g = _guard(tmp_path)
    t = _ist_dt(10)
    broker = [_broker_leg(expiry="2026-01-29", strike=23000, opt=OptionType.PUT, side=LegSide.BUY, qty=65)]
    out = g.evaluate_startup(
        local_bkm_state={},
        broker_legs=broker,
        journal_summary={"unresolved_ops": [], "failed_ops": [], "active_baskets": {}, "events_scanned": 2},
        when=t,
    )
    assert out["ok"] is True
    assert out["reason"] == "BROKER_ONLY_EXTERNAL_POSITIONS"
    snap = g.snapshot()
    assert snap["status"] == "OK"
    assert snap["hard_lock"] is False
    assert snap["last_details"]["broker_leg_count"] == 1


def test_guard_ignores_stale_failed_ops_without_live_footprint(tmp_path: Path) -> None:
    g = _guard(tmp_path)
    t = _ist_dt(11)
    out = g.evaluate_startup(
        local_bkm_state={},
        broker_legs=[],
        journal_summary={
            "unresolved_ops": [],
            "failed_ops": [{"op_id": "c1", "expiry": "2026-01-29", "timestamp": "2026-01-05T10:00:00+05:30"}],
            "active_baskets": {},
            "events_scanned": 5,
        },
        when=t,
    )
    assert out["ok"] is True
    assert out["reason"] == "CLEAN_FLAT"
    snap = g.snapshot()
    assert snap["status"] == "OK"
    assert snap["last_details"]["ignored_failed_ops_count"] == 1


def test_guard_allows_resume_when_failed_op_exists_but_active_basket_matches(tmp_path: Path) -> None:
    g = _guard(tmp_path)
    g.state.hard_lock = True
    t = _ist_dt(12)
    active = {
        "2026-01-29": {
            "expiry": "2026-01-29",
            "legs": [
                {"expiry": "2026-01-29", "strike": 22000, "option_type": "CALL", "side": "SELL", "quantity": 130},
                {"expiry": "2026-01-29", "strike": 22500, "option_type": "CALL", "side": "BUY", "quantity": 65},
                {"expiry": "2026-01-29", "strike": 21000, "option_type": "PUT", "side": "BUY", "quantity": 65},
            ],
            "meta": {"net_credit": 650.0, "margin_required": 1000000.0, "credit_pct": 0.065},
        }
    }
    broker = [
        _broker_leg(expiry="2026-01-29", strike=22000, opt=OptionType.CALL, side=LegSide.SELL, qty=130),
        _broker_leg(expiry="2026-01-29", strike=22500, opt=OptionType.CALL, side=LegSide.BUY, qty=65),
        _broker_leg(expiry="2026-01-29", strike=21000, opt=OptionType.PUT, side=LegSide.BUY, qty=65),
    ]
    out = g.evaluate_startup(
        local_bkm_state={"2026-01-29": {"status": "OPEN"}},
        broker_legs=broker,
        journal_summary={
            "unresolved_ops": [],
            "failed_ops": [{"op_id": "c1", "expiry": "2026-01-29", "timestamp": "2026-01-12T10:00:00+05:30"}],
            "active_baskets": active,
            "events_scanned": 6,
        },
        when=t,
    )
    assert out["ok"] is True
    assert out["reason"] == "RESUME_READY"
    snap = g.snapshot()
    assert snap["status"] == "OK"
    assert snap["hard_lock"] is False
    assert snap["last_details"]["startup"] == "RESUME_READY"
    assert snap["last_details"]["resume_override"] == "MATCHED_ACTIVE_BASKET"
