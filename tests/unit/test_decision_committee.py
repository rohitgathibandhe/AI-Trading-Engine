from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from market_ai.modules.agents.decision_committee import DecisionCommittee
from market_ai.modules.agents.intraday_learning import IntradayLearningConfig, IntradayLearningManager


def _ist_dt(day: int, hour: int = 10, minute: int = 0) -> datetime:
    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    return datetime(2026, 4, day, hour, minute, tzinfo=tz)


def _committee(tmp_path: Path, *, with_learning: bool = False) -> DecisionCommittee:
    learning_manager = None
    if with_learning:
        learning_manager = IntradayLearningManager(
            config=IntradayLearningConfig(),
            status_path=tmp_path / "intraday_ai_learning_status.json",
            outcomes_path=tmp_path / "decision_committee_outcomes.jsonl",
            logger=None,
        )
    return DecisionCommittee(
        status_path=tmp_path / "decision_committee_status.json",
        history_path=tmp_path / "decision_committee_history.jsonl",
        outcomes_path=tmp_path / "decision_committee_outcomes.jsonl",
        learning_manager=learning_manager,
        logger=None,
    )


def _context(*, bias: str = "BEARISH", conflict: float = 20.0, trend_conf: float = 0.84) -> dict:
    return {
        "trend": {
            "bias": bias,
            "breakout_confirmation": "DOWN_CONFIRMED" if bias == "BEARISH" else "UP_CONFIRMED",
        },
        "option_chain": {
            "pcr_bias": bias,
            "oi_build": {"bias": "BEARISH_RESISTANCE" if bias == "BEARISH" else "BULLISH_SUPPORT"},
        },
        "structure": {
            "dominant_signal_bias": bias,
            "trend_confidence": trend_conf,
            "signal_conflict_score": conflict,
            "price_action_confirmation": "CANDLE_AND_RETEST_CONFIRMED",
            "retest_status": "BREAKDOWN_RETEST_HOLD" if bias == "BEARISH" else "BREAKOUT_RETEST_HOLD",
            "volatility_regime": "NORMAL",
            "pcr_unbalanced": False,
            "pcr_unbalanced_side": "NEUTRAL",
        },
    }


def test_decision_committee_marks_ready_for_clean_enter_signal(tmp_path: Path) -> None:
    committee = _committee(tmp_path)
    out = committee.evaluate_intraday(
        now=_ist_dt(3, 10, 45),
        signal_payload={
            "signal": "ENTER_NOW",
            "strategy": "CALL_CREDIT_SPREAD",
            "expiry": "2026-04-28",
            "spot": 22850.0,
            "signal_conflict_score": 18.0,
            "recommendation": {
                "strategy_type": "CALL_CREDIT_SPREAD",
                "est_credit_rs_per_set": 720.0,
            },
        },
        context=_context(bias="BEARISH", conflict=18.0, trend_conf=0.88),
        has_open_bkm=False,
        allow_parallel_with_bkm=False,
    )

    assert out["status"] == "ACTIVE"
    assert out["verdict"] == "READY"
    assert out["consensus_bias"] == "BEARISH"
    assert out["ensemble_confidence"] >= 0.68
    assert out["components"]["risk_critic"]["veto"] is False


def test_decision_committee_vetoes_when_bkm_is_open_and_parallel_blocked(tmp_path: Path) -> None:
    committee = _committee(tmp_path)
    out = committee.evaluate_intraday(
        now=_ist_dt(3, 11, 0),
        signal_payload={
            "signal": "ENTER_NOW",
            "strategy": "PUT_CREDIT_SPREAD",
            "expiry": "2026-04-28",
            "spot": 22850.0,
            "signal_conflict_score": 12.0,
            "recommendation": {
                "strategy_type": "PUT_CREDIT_SPREAD",
                "est_credit_rs_per_set": 650.0,
            },
        },
        context=_context(bias="BULLISH", conflict=12.0, trend_conf=0.81),
        has_open_bkm=True,
        allow_parallel_with_bkm=False,
    )

    assert out["verdict"] == "VETO"
    assert out["components"]["risk_critic"]["veto"] is True
    assert "OPEN_BKM_BLOCK" in out["components"]["risk_critic"]["reasons"]


def test_decision_committee_can_mark_inactive(tmp_path: Path) -> None:
    committee = _committee(tmp_path)
    out = committee.set_inactive(
        now=_ist_dt(3, 9, 0),
        status="DISABLED",
        reason="INTRADAY_AI_DISABLED",
        has_open_bkm=True,
        expiry="2026-04-28",
    )

    assert out["status"] == "DISABLED"
    assert out["verdict"] == "IDLE"
    assert out["has_open_bkm"] is True
    assert out["reasons"] == ["INTRADAY_AI_DISABLED"]


def test_decision_committee_appends_history_only_on_meaningful_change(tmp_path: Path) -> None:
    committee = _committee(tmp_path)
    committee.set_inactive(
        now=_ist_dt(3, 9, 0),
        status="DISABLED",
        reason="INTRADAY_AI_DISABLED",
        has_open_bkm=False,
        expiry=None,
    )
    committee.set_inactive(
        now=_ist_dt(3, 9, 1),
        status="DISABLED",
        reason="INTRADAY_AI_DISABLED",
        has_open_bkm=False,
        expiry=None,
    )
    hist = (tmp_path / "decision_committee_history.jsonl").read_text().strip().splitlines()
    assert len(hist) == 1


def test_decision_committee_records_trade_outcome(tmp_path: Path) -> None:
    committee = _committee(tmp_path)
    committee.record_outcome(
        now=_ist_dt(3, 12, 0),
        trade_row={
            "position_id": "abc123",
            "trade_mode": "paper",
            "strategy_type": "CALL_CREDIT_SPREAD",
            "strategy_label": "Call Credit Spread",
            "expiry": "2026-04-28",
            "opened_at": "2026-04-03T10:30:00+05:30",
            "closed_at": "2026-04-03T11:10:00+05:30",
            "current_session_date": "2026-04-03",
            "hold_minutes": 40.0,
            "pnl_rs": 88.5,
            "reason": "PROFIT_PROTECT",
            "result": "WIN",
            "entry_features": {
                "market_bias": "BEARISH",
                "price_action_bias": "BEARISH",
                "retest_status": "BREAKDOWN_RETEST_HOLD",
                "committee_snapshot": {
                    "status": "ACTIVE",
                    "verdict": "READY",
                    "consensus_bias": "BEARISH",
                    "ensemble_confidence": 0.79,
                    "advisor_signal": "ENTER_NOW",
                    "advisor_strategy": "CALL_CREDIT_SPREAD",
                    "reasons": ["TREND_ALIGNED"],
                    "components": {"market_structure": {"score": 0.8}},
                },
            },
        },
    )

    lines = (tmp_path / "decision_committee_outcomes.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["committee_verdict_at_entry"] == "READY"
    assert payload["committee_bias_at_entry"] == "BEARISH"
    assert payload["exit_reason"] == "PROFIT_PROTECT"
    assert payload["pnl_rs"] == 88.5


def test_decision_committee_vetoes_when_learning_marks_setup_negative(tmp_path: Path) -> None:
    committee = _committee(tmp_path, with_learning=True)
    outcomes_path = tmp_path / "decision_committee_outcomes.jsonl"
    outcomes = []
    for day in (1, 2, 3):
        outcomes.append(
            {
                "timestamp": _ist_dt(day, 12, 0).isoformat(timespec="seconds"),
                "position_id": f"id-{day}",
                "trade_mode": "paper",
                "strategy_type": "CALL_CREDIT_SPREAD",
                "strategy_label": "Call Credit Spread",
                "expiry": "2026-04-28",
                "opened_at": _ist_dt(day, 10, 30).isoformat(timespec="seconds"),
                "closed_at": _ist_dt(day, 11, 10).isoformat(timespec="seconds"),
                "current_session_date": f"2026-04-0{day}",
                "hold_minutes": 40.0,
                "pnl_rs": -120.0,
                "reason": "PRICE_ACTION_REVERSAL",
                "exit_reason": "PRICE_ACTION_REVERSAL",
                "result": "LOSS",
                "entry_market_bias": "BEARISH",
                "entry_price_action_bias": "BEARISH",
                "entry_retest_status": "BREAKDOWN_RETEST_HOLD",
                "committee_status_at_entry": "ACTIVE",
                "committee_verdict_at_entry": "READY",
                "committee_bias_at_entry": "BEARISH",
                "committee_confidence_at_entry": 0.8,
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
        )
    outcomes_path.write_text("".join(json.dumps(row) + "\n" for row in outcomes), encoding="utf-8")
    committee.learning_manager.refresh_summary()

    out = committee.evaluate_intraday(
        now=_ist_dt(4, 10, 45),
        signal_payload={
            "signal": "ENTER_NOW",
            "strategy": "CALL_CREDIT_SPREAD",
            "expiry": "2026-04-28",
            "spot": 22850.0,
            "signal_conflict_score": 18.0,
            "market_bias": "BEARISH",
            "price_action_bias": "BEARISH",
            "retest_status": "BREAKDOWN_RETEST_HOLD",
            "volatility_regime": "HIGH",
            "breakout_confirmation": "DOWN_CONFIRMED",
            "pcr_bias": "BEARISH",
            "oi_build_bias": "BEARISH_RESISTANCE",
            "recommendation": {
                "strategy_type": "CALL_CREDIT_SPREAD",
                "est_credit_rs_per_set": 720.0,
            },
        },
        context=_context(bias="BEARISH", conflict=18.0, trend_conf=0.88),
        has_open_bkm=False,
        allow_parallel_with_bkm=False,
    )

    assert out["verdict"] == "VETO"
    assert out["components"]["adaptive_learning"]["veto"] is True
    assert "ADAPTIVE_LEARNING_BLOCK" in out["components"]["risk_critic"]["reasons"]
