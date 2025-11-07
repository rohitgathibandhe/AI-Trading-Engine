import pytest

from market_ai.modules.risk_manager import RiskLimits, RiskManager

pytestmark = pytest.mark.unit


def test_allow_trade_blocks_on_daily_loss():
    rm = RiskManager(RiskLimits(max_daily_loss=1000.0, per_leg_sl_mult=2.0, max_rolls_per_day=3))
    rm.on_mtm(-1500.0)
    assert rm.allow_trade() is False


def test_per_leg_stop_hit_triggers_when_threshold_crossed():
    rm = RiskManager(RiskLimits(max_daily_loss=5000.0, per_leg_sl_mult=1.5, max_rolls_per_day=5))
    assert rm.per_leg_stop_hit(entry_leg_px=10.0, current_leg_ltp=14.0) is False
    assert rm.per_leg_stop_hit(entry_leg_px=10.0, current_leg_ltp=15.1) is True


def test_can_allocate_respects_exposure_cap():
    limits = RiskLimits(
        max_daily_loss=5000.0,
        per_leg_sl_mult=2.0,
        max_rolls_per_day=5,
        equity=100_000.0,
        max_exposure_pct=0.05,
    )
    rm = RiskManager(limits)
    assert rm.can_allocate(4_000.0) is True
    assert rm.can_allocate(6_000.0) is False


def test_daily_drawdown_flat_and_lock():
    rm = RiskManager(RiskLimits(max_daily_loss=2000.0, per_leg_sl_mult=2.0, max_rolls_per_day=5))
    rm.on_mtm(-2500.0)
    assert rm.allow_trade() is False  # breach triggers lock
    rm.on_mtm(0.0)
    rm.l.hard_kill = True  # manual flat-and-lock
    assert rm.allow_trade() is False


def test_portfolio_stop_and_delta_guard():
    limits = RiskLimits(
        max_daily_loss=2000.0,
        per_leg_sl_mult=2.0,
        max_rolls_per_day=5,
        overall_sl_pct=2.5,
        max_portfolio_delta=0.20,
    )
    rm = RiskManager(limits)
    assert rm.portfolio_stop_hit(entry_credit=1000.0, realized_pnl=-2000.0) is False
    assert rm.portfolio_stop_hit(entry_credit=1000.0, realized_pnl=-2600.0) is True
    assert rm.delta_within_bounds(0.15) is True
    assert rm.delta_within_bounds(-0.25) is False
    assert rm.delta_within_bounds(0.30) is False
