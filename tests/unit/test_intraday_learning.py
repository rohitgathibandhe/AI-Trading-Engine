from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from market_ai.modules.agents.intraday_learning import IntradayLearningConfig, IntradayLearningManager


def _ist_dt(day: int, hour: int = 10, minute: int = 0) -> datetime:
    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    return datetime(2026, 4, day, hour, minute, tzinfo=tz)


def _outcome(ts: datetime, *, pnl_rs: float, result: str = "LOSS") -> dict:
    return {
        "timestamp": ts.isoformat(timespec="seconds"),
        "position_id": f"id-{ts.day}",
        "trade_mode": "paper",
        "strategy_type": "CALL_CREDIT_SPREAD",
        "strategy_label": "Bear Call Spread",
        "expiry": "2026-04-30",
        "opened_at": ts.isoformat(timespec="seconds"),
        "closed_at": ts.replace(minute=ts.minute + 20).isoformat(timespec="seconds"),
        "current_session_date": ts.date().isoformat(),
        "hold_minutes": 20.0,
        "pnl_rs": pnl_rs,
        "result": result,
        "exit_reason": "PRICE_ACTION_REVERSAL",
        "entry_market_bias": "BEARISH",
        "entry_price_action_bias": "BEARISH",
        "entry_retest_status": "BREAKDOWN_RETEST_HOLD",
        "committee_verdict_at_entry": "READY",
        "committee_bias_at_entry": "BEARISH",
        "committee_confidence_at_entry": 0.81,
        "committee_signal_at_entry": "ENTER_NOW",
        "committee_strategy_at_entry": "CALL_CREDIT_SPREAD",
        "committee_reasons_at_entry": ["STRUCTURE_BEARISH"],
        "committee_components_at_entry": {},
        "entry_features": {
            "volatility_regime": "HIGH",
            "breakout_confirmation": "DOWN_CONFIRMED",
            "pcr_bias": "BEARISH",
            "oi_build_bias": "BEARISH_RESISTANCE",
        },
    }


def _mgr(tmp_path: Path) -> IntradayLearningManager:
    return IntradayLearningManager(
        config=IntradayLearningConfig(),
        status_path=tmp_path / "intraday_ai_learning_status.json",
        outcomes_path=tmp_path / "decision_committee_outcomes.jsonl",
        logger=None,
    )


def test_intraday_learning_blocks_repeated_negative_setup(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path)
    outcomes_path = tmp_path / "decision_committee_outcomes.jsonl"
    rows = [_outcome(_ist_dt(day), pnl_rs=-120.0, result="LOSS") for day in (1, 2, 3)]
    outcomes_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    mgr.refresh_summary()

    assessment = mgr.assess_signal(
        signal_payload={
            "signal": "ENTER_NOW",
            "strategy": "CALL_CREDIT_SPREAD",
            "market_bias": "BEARISH",
            "price_action_bias": "BEARISH",
            "retest_status": "BREAKDOWN_RETEST_HOLD",
            "volatility_regime": "HIGH",
            "breakout_confirmation": "DOWN_CONFIRMED",
            "pcr_bias": "BEARISH",
            "oi_build_bias": "BEARISH_RESISTANCE",
        },
        context={},
        consensus_bias="BEARISH",
        base_confidence=0.8,
    )

    assert assessment["status"] == "ACTIVE"
    assert assessment["block_entry"] is True
    assert assessment["confidence_adjust"] < 0
    assert any(str(reason).startswith("LEARNING_NEGATIVE_") for reason in assessment["reasons"])


def test_intraday_learning_boosts_positive_setup(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path)
    outcomes_path = tmp_path / "decision_committee_outcomes.jsonl"
    rows = [_outcome(_ist_dt(day), pnl_rs=55.0, result="WIN") for day in (1, 2, 3)]
    outcomes_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    mgr.refresh_summary()

    assessment = mgr.assess_signal(
        signal_payload={
            "signal": "ENTER_NOW",
            "strategy": "CALL_CREDIT_SPREAD",
            "market_bias": "BEARISH",
            "price_action_bias": "BEARISH",
            "retest_status": "BREAKDOWN_RETEST_HOLD",
            "volatility_regime": "HIGH",
            "breakout_confirmation": "DOWN_CONFIRMED",
            "pcr_bias": "BEARISH",
            "oi_build_bias": "BEARISH_RESISTANCE",
        },
        context={},
        consensus_bias="BEARISH",
        base_confidence=0.7,
    )

    assert assessment["block_entry"] is False
    assert assessment["confidence_adjust"] > 0
    assert assessment["component"]["stance"] == "SUPPORTIVE"
