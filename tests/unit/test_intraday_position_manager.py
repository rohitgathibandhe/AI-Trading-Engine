from __future__ import annotations

import csv
from datetime import date, datetime
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from market_ai.modules.agents.intraday_position_manager import IntradayPositionManager
from market_ai.strategies import LegSide, OptionLeg, OptionType, StrategyType


def _ist_dt(day: int, hour: int = 10, minute: int = 0) -> datetime:
    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    return datetime(2026, 3, day, hour, minute, tzinfo=tz)


def _leg(*, strike: float, opt: OptionType, side: LegSide, qty: int, entry_price: float) -> OptionLeg:
    return OptionLeg(
        symbol="NIFTY",
        expiry=date(2026, 3, 10),
        strike=float(strike),
        option_type=opt,
        side=side,
        quantity=int(qty),
        entry_price=float(entry_price),
        strategy_type=StrategyType.BEAR_CALL_SPREAD,
    )


def _manager(tmp_path: Path) -> IntradayPositionManager:
    return IntradayPositionManager(
        state_path=tmp_path / "intraday_ai_position_status.json",
        history_path=tmp_path / "intraday_ai_trade_history.jsonl",
        logger=None,
    )


def test_session_trade_limit_blocks_reentry_same_day(tmp_path: Path) -> None:
    mgr = _manager(tmp_path)
    now = _ist_dt(6, 10, 15)
    legs = [
        _leg(strike=25500, opt=OptionType.CALL, side=LegSide.SELL, qty=65, entry_price=100.0),
        _leg(strike=25700, opt=OptionType.CALL, side=LegSide.BUY, qty=65, entry_price=60.0),
    ]

    mgr.open_position(
        position_id="POS-1",
        trade_mode="paper",
        strategy_type="CALL_CREDIT_SPREAD",
        strategy_label="Bear Call Credit Spread",
        expiry="2026-03-10",
        legs=legs,
        entry_spot=25100.0,
        sl_total_rs=1600.0,
        tp_total_rs=800.0,
        invalidation_spot_level=25320.0,
        max_hold_till="15:05",
        trailing_enabled=True,
        lots_multiplier=1,
        entry_features={"market_bias": "BEARISH", "price_action_bias": "BEARISH"},
        now=now,
    )
    mgr.close_position(reason="TARGET_HIT", pnl_rs=900.0, now=now)

    assert mgr.snapshot()["session_trade_count"] == 1
    assert mgr.can_open_new_position(max_trades_per_session=1, now=now) is False
    assert mgr.can_open_new_position(max_trades_per_session=1, now=_ist_dt(7, 10, 15)) is True


def test_evaluate_closes_bear_call_spread_on_spot_invalidation(tmp_path: Path) -> None:
    mgr = _manager(tmp_path)
    now = _ist_dt(6, 11, 0)
    legs = [
        _leg(strike=25500, opt=OptionType.CALL, side=LegSide.SELL, qty=65, entry_price=100.0),
        _leg(strike=25700, opt=OptionType.CALL, side=LegSide.BUY, qty=65, entry_price=60.0),
    ]
    mgr.open_position(
        position_id="POS-2",
        trade_mode="paper",
        strategy_type="CALL_CREDIT_SPREAD",
        strategy_label="Bear Call Credit Spread",
        expiry="2026-03-10",
        legs=legs,
        entry_spot=25100.0,
        sl_total_rs=1600.0,
        tp_total_rs=800.0,
        invalidation_spot_level=25320.0,
        max_hold_till="15:05",
        trailing_enabled=True,
        lots_multiplier=1,
        entry_features={"market_bias": "BEARISH", "price_action_bias": "BEARISH"},
        now=now,
    )

    out = mgr.evaluate(
        now=_ist_dt(6, 11, 5),
        spot=25340.0,
        chain_rows=[
            {"option_type": "CE", "strike": 25500.0, "ltp": 130.0},
            {"option_type": "CE", "strike": 25700.0, "ltp": 82.0},
        ],
        signal_payload={},
    )

    assert out["action"] == "CLOSE"
    assert out["reason"] == "SPOT_INVALIDATION"


def test_close_position_appends_history_row(tmp_path: Path) -> None:
    mgr = _manager(tmp_path)
    now = _ist_dt(6, 10, 10)
    mgr.open_position(
        position_id="POS-3",
        trade_mode="paper",
        strategy_type="CALL_CREDIT_SPREAD",
        strategy_label="Bear Call Credit Spread",
        expiry="2026-03-10",
        legs=[_leg(strike=25500, opt=OptionType.CALL, side=LegSide.SELL, qty=65, entry_price=100.0)],
        entry_spot=25100.0,
        sl_total_rs=1500.0,
        tp_total_rs=700.0,
        invalidation_spot_level=25320.0,
        max_hold_till="15:05",
        trailing_enabled=True,
        lots_multiplier=1,
        entry_features={"market_bias": "BEARISH", "price_action_bias": "BEARISH"},
        now=now,
    )
    mgr.close_position(reason="TARGET_HIT", pnl_rs=850.0, now=_ist_dt(6, 10, 42))

    history_path = tmp_path / "intraday_ai_trade_history.jsonl"
    lines = [line for line in history_path.read_text().splitlines() if line.strip()]
    assert len(lines) == 1
    assert "\"reason\": \"TARGET_HIT\"" in lines[0]
    assert "\"result\": \"WIN\"" in lines[0]
    assert "\"price_action_bias\": \"BEARISH\"" in lines[0]


def test_evaluate_closes_bear_call_spread_on_price_action_reversal_before_full_signal_reversal(tmp_path: Path) -> None:
    mgr = _manager(tmp_path)
    now = _ist_dt(6, 12, 0)
    legs = [
        _leg(strike=25500, opt=OptionType.CALL, side=LegSide.SELL, qty=65, entry_price=100.0),
        _leg(strike=25700, opt=OptionType.CALL, side=LegSide.BUY, qty=65, entry_price=60.0),
    ]
    mgr.open_position(
        position_id="POS-4",
        trade_mode="paper",
        strategy_type="CALL_CREDIT_SPREAD",
        strategy_label="Bear Call Credit Spread",
        expiry="2026-03-10",
        legs=legs,
        entry_spot=25100.0,
        sl_total_rs=1500.0,
        tp_total_rs=700.0,
        invalidation_spot_level=25320.0,
        max_hold_till="15:05",
        trailing_enabled=True,
        lots_multiplier=1,
        entry_features={"market_bias": "BEARISH", "price_action_bias": "BEARISH"},
        now=now,
    )
    out = mgr.evaluate(
        now=_ist_dt(6, 12, 5),
        spot=25210.0,
        chain_rows=[
            {"option_type": "CE", "strike": 25500.0, "ltp": 99.0},
            {"option_type": "CE", "strike": 25700.0, "ltp": 58.0},
        ],
        signal_payload={
            "market_bias": "BULLISH",
            "breakout_confirmation": "NONE",
            "trend_confidence": 0.62,
            "signal_conflict_score": 40.0,
            "price_action_bias": "BULLISH",
            "price_action_confirmation": "CANDLE_AND_RETEST_CONFIRMED",
            "retest_status": "SUPPORT_HOLD",
            "pcr_bias": "BULLISH",
            "oi_build_bias": "BULLISH_SUPPORT",
            "pcr_unbalanced_side": "BULLISH",
        },
    )
    assert out["action"] == "CLOSE"
    assert out["reason"] == "PRICE_ACTION_REVERSAL"


def test_evaluate_closes_bull_put_spread_on_chain_conflict_when_losing(tmp_path: Path) -> None:
    mgr = _manager(tmp_path)
    now = _ist_dt(6, 12, 15)
    legs = [
        _leg(strike=24800, opt=OptionType.PUT, side=LegSide.SELL, qty=65, entry_price=100.0),
        _leg(strike=24600, opt=OptionType.PUT, side=LegSide.BUY, qty=65, entry_price=60.0),
    ]
    mgr.open_position(
        position_id="POS-5",
        trade_mode="paper",
        strategy_type="PUT_CREDIT_SPREAD",
        strategy_label="Bull Put Credit Spread",
        expiry="2026-03-10",
        legs=legs,
        entry_spot=25100.0,
        sl_total_rs=1000.0,
        tp_total_rs=500.0,
        invalidation_spot_level=24720.0,
        max_hold_till="15:05",
        trailing_enabled=True,
        lots_multiplier=1,
        entry_features={"market_bias": "BULLISH", "price_action_bias": "BULLISH"},
        now=now,
    )
    out = mgr.evaluate(
        now=_ist_dt(6, 12, 20),
        spot=24960.0,
        chain_rows=[
            {"option_type": "PE", "strike": 24800.0, "ltp": 104.0},
            {"option_type": "PE", "strike": 24600.0, "ltp": 61.0},
        ],
        signal_payload={
            "market_bias": "NEUTRAL",
            "breakout_confirmation": "NONE",
            "trend_confidence": 0.42,
            "signal_conflict_score": 65.0,
            "price_action_bias": "NEUTRAL",
            "price_action_confirmation": "NONE",
            "retest_status": "NONE",
            "pcr_bias": "BEARISH",
            "oi_build_bias": "BEARISH_RESISTANCE",
            "pcr_unbalanced_side": "BEARISH",
        },
    )
    assert out["action"] == "CLOSE"
    assert out["reason"] == "CHAIN_CONFLICT_EXIT"


def test_evaluate_tightens_profit_protection_after_peak_when_signal_weakens(tmp_path: Path) -> None:
    mgr = _manager(tmp_path)
    now = _ist_dt(6, 10, 45)
    legs = [
        _leg(strike=25500, opt=OptionType.CALL, side=LegSide.SELL, qty=65, entry_price=100.0),
        _leg(strike=25700, opt=OptionType.CALL, side=LegSide.BUY, qty=65, entry_price=60.0),
    ]
    mgr.open_position(
        position_id="POS-6",
        trade_mode="paper",
        strategy_type="CALL_CREDIT_SPREAD",
        strategy_label="Bear Call Credit Spread",
        expiry="2026-03-10",
        legs=legs,
        entry_spot=25100.0,
        sl_total_rs=1200.0,
        tp_total_rs=700.0,
        invalidation_spot_level=25320.0,
        max_hold_till="15:05",
        trailing_enabled=True,
        lots_multiplier=1,
        entry_features={"market_bias": "BEARISH", "price_action_bias": "BEARISH", "volatility_regime": "NORMAL"},
        now=now,
    )
    first = mgr.evaluate(
        now=_ist_dt(6, 10, 50),
        spot=25040.0,
        chain_rows=[
            {"option_type": "CE", "strike": 25500.0, "ltp": 95.0},
            {"option_type": "CE", "strike": 25700.0, "ltp": 58.0},
        ],
        signal_payload={"volatility_regime": "NORMAL"},
    )
    assert first["action"] == "HOLD"
    assert first["pnl_rs"] > 100.0

    second = mgr.evaluate(
        now=_ist_dt(6, 11, 5),
        spot=25110.0,
        chain_rows=[
            {"option_type": "CE", "strike": 25500.0, "ltp": 98.0},
            {"option_type": "CE", "strike": 25700.0, "ltp": 59.0},
        ],
        signal_payload={
            "market_bias": "BULLISH",
            "breakout_confirmation": "UP_CONFIRMED",
            "trend_confidence": 0.71,
            "signal_conflict_score": 38.0,
            "price_action_bias": "BULLISH",
            "price_action_confirmation": "CANDLE_CONFIRMED",
            "retest_status": "SUPPORT_HOLD",
            "pcr_bias": "BULLISH",
            "oi_build_bias": "BULLISH_SUPPORT",
            "pcr_unbalanced_side": "BULLISH",
            "volatility_regime": "NORMAL",
        },
    )
    assert second["action"] == "CLOSE"
    assert second["reason"] in {"PROFIT_PROTECT", "TRAIL_FLOOR_HIT"}
