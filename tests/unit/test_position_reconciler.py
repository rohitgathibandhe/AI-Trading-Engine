from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from market_ai.modules.agents.position_reconciler import (
    PositionReconciler,
    PositionReconcilerConfig,
)
from market_ai.strategies import LegSide, OptionLeg, OptionType, StrategyType


def _ist_dt(day: int, hour: int = 10, minute: int = 0) -> datetime:
    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    return datetime(2026, 1, day, hour, minute, tzinfo=tz)


def _leg(*, strike: float, opt: OptionType, side: LegSide, qty: int, expiry: date | None = None) -> OptionLeg:
    return OptionLeg(
        symbol="NIFTY",
        expiry=expiry or date(2026, 1, 29),
        strike=strike,
        option_type=opt,
        side=side,
        quantity=qty,
        entry_price=100.0,
        strategy_type=StrategyType.NONE,
    )


def _new_reconciler(tmp_path: Path, **kwargs) -> PositionReconciler:
    cfg = PositionReconcilerConfig(**kwargs)
    return PositionReconciler(config=cfg, status_path=tmp_path / "position_reconcile_status.json", logger=None)


def test_match_resets_mismatch_streak(tmp_path: Path) -> None:
    r = _new_reconciler(tmp_path)
    t = _ist_dt(5)
    legs = [_leg(strike=22000, opt=OptionType.CALL, side=LegSide.SELL, qty=65)]
    out = r.evaluate(expected_legs=legs, broker_legs=legs, when=t)
    assert out["ok"] is True
    snap = r.snapshot()
    assert snap["status"] == "OK"
    assert snap["mismatch_streak"] == 0


def test_mismatch_locks_after_confirm_threshold(tmp_path: Path) -> None:
    r = _new_reconciler(tmp_path, mismatch_confirm_count=2, hard_lock_on_mismatch=True)
    t = _ist_dt(6)
    expected = [_leg(strike=22000, opt=OptionType.CALL, side=LegSide.SELL, qty=65)]
    broker = []

    out1 = r.evaluate(expected_legs=expected, broker_legs=broker, when=t)
    assert out1["ok"] is False
    assert out1["locked"] is False

    out2 = r.evaluate(expected_legs=expected, broker_legs=broker, when=t + timedelta(seconds=10))
    assert out2["ok"] is False
    assert out2["locked"] is True
    snap = r.snapshot()
    assert snap["status"] == "LOCKED"
    assert snap["hard_lock"] is True
    assert r.should_block_entries(t + timedelta(seconds=10)) is True


def test_grace_period_skips_check_after_order_submission(tmp_path: Path) -> None:
    r = _new_reconciler(tmp_path, grace_sec=30, mismatch_confirm_count=1)
    t = _ist_dt(7)
    r.defer_checks(seconds=30, reason="BKM_OPEN_SUBMITTED", when=t)
    out = r.evaluate(
        expected_legs=[_leg(strike=22000, opt=OptionType.CALL, side=LegSide.SELL, qty=65)],
        broker_legs=[],
        when=t + timedelta(seconds=5),
    )
    assert out["ok"] is True
    assert out["skipped"] is True
    assert out["reason"] == "GRACE_PERIOD"


def test_day_lock_clears_next_day_when_not_hard_lock(tmp_path: Path) -> None:
    r = _new_reconciler(tmp_path, mismatch_confirm_count=1, hard_lock_on_mismatch=False)
    t = _ist_dt(8)
    r.evaluate(
        expected_legs=[_leg(strike=22000, opt=OptionType.CALL, side=LegSide.SELL, qty=65)],
        broker_legs=[],
        when=t,
    )
    assert r.should_block_entries(t) is True
    nxt = t + timedelta(days=1)
    r.on_tick(nxt)
    assert r.should_block_entries(nxt) is False
    assert r.snapshot()["status"] == "OK"


def test_reset_clears_lock_and_diff(tmp_path: Path) -> None:
    r = _new_reconciler(tmp_path, mismatch_confirm_count=1)
    t = _ist_dt(9)
    r.evaluate(
        expected_legs=[_leg(strike=22000, opt=OptionType.CALL, side=LegSide.SELL, qty=65)],
        broker_legs=[],
        when=t,
    )
    snap = r.snapshot()
    assert snap["status"] == "LOCKED"
    out = r.reset()
    assert out["status"] == "OK"
    assert out["hard_lock"] is False
    assert out["mismatch_streak"] == 0


def test_acknowledge_external_positions_clears_existing_hard_lock(tmp_path: Path) -> None:
    r = _new_reconciler(tmp_path, mismatch_confirm_count=1, hard_lock_on_mismatch=True)
    t = _ist_dt(10)
    r.evaluate(
        expected_legs=[_leg(strike=22000, opt=OptionType.CALL, side=LegSide.SELL, qty=65)],
        broker_legs=[],
        when=t,
    )
    assert r.snapshot()["hard_lock"] is True

    out = r.acknowledge_external_positions(
        broker_legs=[_leg(strike=23000, opt=OptionType.PUT, side=LegSide.BUY, qty=65)],
        when=t + timedelta(minutes=1),
    )
    assert out["ok"] is True
    assert out["reason"] == "BROKER_ONLY_EXTERNAL_POSITIONS"
    snap = r.snapshot()
    assert snap["status"] == "OK"
    assert snap["hard_lock"] is False
    assert snap["mismatch_streak"] == 0
    assert snap["last_diff_summary"]["broker_count"] == 1
