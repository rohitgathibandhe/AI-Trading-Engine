from __future__ import annotations

from datetime import datetime
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from market_ai.modules.agents.batman_bkm_learning import (
    BatmanBKMLearningConfig,
    BatmanBKMLearningManager,
)


def _ist_dt(day: int, hour: int, minute: int) -> datetime:
    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    return datetime(2026, 4, day, hour, minute, tzinfo=tz)


def _snapshot(ts: datetime) -> dict:
    return {
        "timestamp": ts.isoformat(timespec="seconds"),
        "entry_mode": "SCHEDULED",
        "market_bias": "BEARISH",
        "pcr_bias": "BEARISH",
        "oi_build_bias": "BEARISH_PRESSURE",
        "volatility_regime": "HIGH",
        "breakout_confirmation": "DOWN_CONFIRMED",
        "dominant_signal_bias": "BEARISH",
        "signal_conflict_score": 20.0,
        "trend_confidence": 0.86,
        "quality_status": "PASS",
        "quality_score": 98.0,
    }


def _mgr(tmp_path: Path) -> BatmanBKMLearningManager:
    return BatmanBKMLearningManager(
        config=BatmanBKMLearningConfig(),
        status_path=tmp_path / "batman_bkm_learning_status.json",
        outcomes_path=tmp_path / "batman_bkm_learning_outcomes.jsonl",
        logger=None,
    )


def test_learning_manager_blocks_premarket_entry(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path)
    assessment = mgr.assess_entry(_snapshot(_ist_dt(1, 9, 0)), when=_ist_dt(1, 9, 0))

    assert assessment["status"] == "BLOCK"
    assert assessment["block_entry"] is True
    assert "LEARNING_BLOCK_PRE_MARKET_ENTRY" in assessment["reasons"]


def test_learning_manager_penalizes_repeated_negative_bucket_patterns(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path)
    for day in (1, 2, 3):
        ts = _ist_dt(day, 9, 45)
        mgr.record_outcome(
            expiry="2026-05-26",
            trade_mode="paper",
            opened_at=ts.isoformat(timespec="seconds"),
            closed_at=_ist_dt(day, 10, 5).isoformat(timespec="seconds"),
            close_reason="DAILY_LOCK_RED",
            pnl_rs=-3500.0,
            entry_snapshot=_snapshot(ts),
            close_context={"market_bias": "BEARISH"},
            quality_status="PASS",
            quality_score=98.0,
            net_credit_rs=22000.0,
        )

    assessment = mgr.assess_entry(_snapshot(_ist_dt(4, 9, 45)), when=_ist_dt(4, 9, 45))

    assert assessment["status"] == "BLOCK"
    assert assessment["block_entry"] is True
    assert float(assessment["risk_score_adjust"]) >= 12.0
    assert any(str(reason).startswith("LEARNING_NEGATIVE_") for reason in assessment["reasons"])
    snap = mgr.snapshot()
    assert snap["total_outcomes"] == 3
    assert snap["losses"] == 3
    assert snap["top_negative_buckets"]
