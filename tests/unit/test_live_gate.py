from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from market_ai.modules.agents.live_gate import LiveGate, LiveGateConfig


def _ist_dt(day: int, hour: int = 10, minute: int = 0) -> datetime:
    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    return datetime(2026, 1, day, hour, minute, tzinfo=tz)


def _new_gate(tmp_path: Path, **cfg_overrides) -> LiveGate:
    cfg = LiveGateConfig(**cfg_overrides)
    return LiveGate(
        config=cfg,
        status_path=tmp_path / "live_gate_status.json",
        sessions_path=tmp_path / "live_gate_sessions.jsonl",
        logger=None,
    )


def test_promotes_on_exact_probation_threshold(tmp_path: Path) -> None:
    gate = _new_gate(tmp_path, probation_sessions=10, probation_min_pass=8, probation_cum_mtm_floor=-15000.0)

    # Exactly 10 sessions, 8 PASS, 2 FAIL (non-consecutive), cum MTM above floor.
    outcomes = [True, True, False, True, True, False, True, True, True, True]
    for idx, passed in enumerate(outcomes, start=1):
        session_ts = _ist_dt(idx)
        gate.on_tick(session_ts)
        gate.note_live_event(mtm=1000.0 if passed else -500.0, when=session_ts)
        if not passed:
            gate.mark_daily_lock("DAILY_LOCK_RED", when=session_ts)
        gate.on_tick(session_ts + timedelta(days=1))

    snap = gate.snapshot()
    assert snap["sessions_total"] == 10
    assert snap["sessions_pass"] == 8
    assert snap["sessions_fail"] == 2
    assert snap["stage"] == "S2"
    assert snap["status"] == "PROMOTED"
    assert snap["lot_multiplier_active"] == 2
    assert snap["cum_mtm"] >= 0

    sessions_file = tmp_path / "live_gate_sessions.jsonl"
    lines = [json.loads(line) for line in sessions_file.read_text().splitlines() if line.strip()]
    assert len(lines) == 10


def test_hard_lock_after_two_consecutive_fails(tmp_path: Path) -> None:
    gate = _new_gate(tmp_path, probation_sessions=10, probation_min_pass=8)

    day1 = _ist_dt(1)
    gate.on_tick(day1)
    gate.note_live_event(mtm=-1000.0, when=day1)
    gate.mark_daily_lock("DAILY_LOCK_RED", when=day1)
    gate.on_tick(day1 + timedelta(days=1))

    day2 = _ist_dt(2)
    gate.on_tick(day2)
    gate.note_live_event(mtm=-1200.0, when=day2)
    gate.mark_daily_lock("DAILY_LOCK_RED", when=day2)
    gate.on_tick(day2 + timedelta(days=1))

    snap = gate.snapshot()
    assert snap["sessions_total"] == 2
    assert snap["sessions_fail"] == 2
    assert snap["status"] == "LOCKED"
    assert snap["hard_lock"] is True
    assert snap["last_fail_reason"] == "LIVE_GATE_SEVERE_2_CONSEC_FAILS"
    assert gate.should_block_entries(_ist_dt(3)) is True


def test_spot_zero_threshold_triggers_lock_for_day(tmp_path: Path) -> None:
    gate = _new_gate(tmp_path, data_fail_zero_spot_count=3, data_fail_zero_spot_window_sec=120)
    t0 = _ist_dt(5, 11, 0)

    assert gate.register_spot(0.0, when=t0)[0] is False
    assert gate.register_spot(0.0, when=t0 + timedelta(seconds=40))[0] is False
    triggered, reason = gate.register_spot(0.0, when=t0 + timedelta(seconds=80))
    assert triggered is True
    assert "DATA_FAILSAFE_SPOT_ZERO_3_IN_120s" in reason

    gate.trigger_failsafe(reason, when=t0 + timedelta(seconds=80))
    assert gate.should_block_entries(t0 + timedelta(seconds=81)) is True

    # Day lock should clear next day if not hard-locked.
    next_day = t0 + timedelta(days=1)
    gate.on_tick(next_day)
    assert gate.should_block_entries(next_day) is False


def test_connection_failure_threshold_triggers_lock_without_open_legs(tmp_path: Path) -> None:
    gate = _new_gate(tmp_path, data_fail_conn_count=3, data_fail_conn_window_sec=600)
    t0 = _ist_dt(7, 12, 0)

    assert gate.register_connection_failure(when=t0)[0] is False
    assert gate.register_connection_failure(when=t0 + timedelta(seconds=120))[0] is False
    triggered, reason = gate.register_connection_failure(when=t0 + timedelta(seconds=240))
    assert triggered is True
    assert "DATA_FAILSAFE_CONN_3_IN_600s" in reason

    gate.trigger_failsafe(reason, when=t0 + timedelta(seconds=240))
    assert gate.should_block_entries(t0 + timedelta(seconds=241)) is True
