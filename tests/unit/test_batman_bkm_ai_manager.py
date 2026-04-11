from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from market_ai.modules.agents.batman_bkm_ai_manager import BatmanBKMAIConfig, BatmanBKMAIManager


def _ist_dt(day: int, hour: int = 10, minute: int = 0) -> datetime:
    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    return datetime(2026, 2, day, hour, minute, tzinfo=tz)


def _bkm_basket(now: datetime):
    entry_ts = now - timedelta(minutes=10)
    legs = [
        SimpleNamespace(option_type="PE", side="BUY", strike=25200.0, qty=65),
        SimpleNamespace(option_type="PE", side="SELL", strike=25000.0, qty=195),
        SimpleNamespace(option_type="PE", side="BUY", strike=24200.0, qty=130),
        SimpleNamespace(option_type="CE", side="BUY", strike=26000.0, qty=65),
        SimpleNamespace(option_type="CE", side="SELL", strike=26200.0, qty=195),
        SimpleNamespace(option_type="CE", side="BUY", strike=27000.0, qty=130),
    ]
    return SimpleNamespace(
        expiry=date(2026, 3, 30),
        entry_ts=entry_ts,
        margin_required=1_000_000.0,
        net_credit=22000.0,
        credit_pct=2.2,
        legs=legs,
    )


def _mgr(tmp_path: Path) -> BatmanBKMAIManager:
    return BatmanBKMAIManager(
        config=BatmanBKMAIConfig(),
        status_path=tmp_path / "batman_bkm_ai_status.json",
        events_path=tmp_path / "batman_bkm_ai_events.jsonl",
        logger=None,
    )


def _mgr_with_mode(tmp_path: Path, mode: str) -> BatmanBKMAIManager:
    cfg = BatmanBKMAIConfig()
    cfg.mode = mode
    return BatmanBKMAIManager(
        config=cfg,
        status_path=tmp_path / f"batman_bkm_ai_status_{mode}.json",
        events_path=tmp_path / f"batman_bkm_ai_events_{mode}.jsonl",
        logger=None,
    )


def test_ai_manager_holds_when_spot_inside_range(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path)
    now = _ist_dt(25, 10, 15)
    basket = _bkm_basket(now)

    snap = mgr.update_open(
        basket=basket,
        spot=25600.0,
        pnl=500.0,
        tp=20000.0,
        sl=-25000.0,
        day_mode="NORMAL",
        when=now,
    )

    assert snap["status"] == "ACTIVE"
    assert snap["action"] == "HOLD"
    assert snap["severity"] == "INFO"
    assert snap["outside_short_range"] is False
    assert (tmp_path / "batman_bkm_ai_status.json").exists()


def test_ai_manager_recommends_flatten_outside_outer_wing(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path)
    now = _ist_dt(25, 11, 0)
    basket = _bkm_basket(now)

    # Seed a couple of points so velocity features are available.
    mgr.update_open(
        basket=basket,
        spot=26100.0,
        pnl=-1000.0,
        tp=20000.0,
        sl=-25000.0,
        day_mode="NORMAL",
        when=now - timedelta(minutes=2),
    )
    snap = mgr.update_open(
        basket=basket,
        spot=27150.0,  # beyond CE outer wing 27000
        pnl=-9000.0,
        tp=20000.0,
        sl=-25000.0,
        day_mode="NORMAL",
        when=now,
    )

    assert snap["action"] == "FLATTEN_RECOMMEND"
    assert snap["severity"] == "CRITICAL"
    assert snap["outside_outer_range"] is True
    assert float(snap["score"]) >= 80.0
    events = (tmp_path / "batman_bkm_ai_events.jsonl").read_text().splitlines()
    assert len([ln for ln in events if ln.strip()]) >= 1


def test_ai_manager_uses_market_context_for_explainable_bias(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path)
    now = _ist_dt(25, 12, 0)
    basket = _bkm_basket(now)

    snap = mgr.update_open(
        basket=basket,
        spot=26170.0,  # close to call short 26200
        pnl=250.0,
        tp=20000.0,
        sl=-25000.0,
        day_mode="NORMAL",
        context={
            "trend": {"bias": "BULLISH", "bias_score": 0.72, "timeframes": {"5": {"trend": "UP"}}},
            "option_chain": {
                "pcr_bias": "BULLISH",
                "pcr_total": 1.18,
                "pcr_near_atm": 1.24,
                "oi_build": {"bias": "BULLISH_SUPPORT", "available": True},
                "call_wall_above": {"distance_from_spot": 420.0},
            },
        },
        when=now,
    )

    assert snap["status"] == "ACTIVE"
    assert "market_context" in snap and isinstance(snap["market_context"], dict)
    assert "plan" in snap and isinstance(snap["plan"], dict)
    assert snap["position_risk_side"] in {"CALL", "PUT", "CENTER"}
    assert snap["market_bias"] in {"BULLISH", "BEARISH", "NEUTRAL"}
    assert any("CONTEXT" in r or "WALL" in r for r in (snap.get("reasons") or []))


def test_ai_manager_auto_protect_sets_entry_lock_in_plan(tmp_path: Path) -> None:
    mgr = _mgr_with_mode(tmp_path, "AUTO_PROTECT")
    now = _ist_dt(25, 13, 0)
    basket = _bkm_basket(now)

    # Build score above reduce threshold by moving near call short with bullish context.
    snap = mgr.update_open(
        basket=basket,
        spot=27150.0,  # beyond CE outer wing -> high risk on call side
        pnl=-3500.0,
        tp=20000.0,
        sl=-25000.0,
        day_mode="NORMAL",
        context={
            "trend": {"bias": "BULLISH", "bias_score": 0.8, "volatility_regime": "HIGH", "breakout_confirmation": "UP_CONFIRMED"},
            "option_chain": {
                "pcr_bias": "BULLISH",
                "pcr_near_atm": 1.22,
                "oi_build": {"bias": "BULLISH_SUPPORT"},
                "call_wall_above": {"distance_from_spot": 500.0},
            },
        },
        when=now,
    )

    assert snap["plan"]["mode"] == "AUTO_PROTECT"
    assert snap["action"] == "FLATTEN_RECOMMEND"
    assert snap["plan"]["entry_lock_active"] is True


def test_ai_manager_applies_learning_context_adjustment(tmp_path: Path) -> None:
    mgr = _mgr(tmp_path)
    now = _ist_dt(25, 12, 30)
    basket = _bkm_basket(now)

    snap = mgr.update_open(
        basket=basket,
        spot=26170.0,
        pnl=-400.0,
        tp=20000.0,
        sl=-25000.0,
        day_mode="NORMAL",
        context={
            "learning": {
                "status": "CAUTION",
                "risk_score_adjust": 6.0,
                "reasons": ["LEARNING_NEGATIVE_ENTRY_PHASE_OPENING_RANGE"],
                "supportive_reasons": ["LEARNING_POSITIVE_QUALITY_STATUS_PASS"],
            }
        },
        when=now,
    )

    assert "LEARNING_NEGATIVE_ENTRY_PHASE_OPENING_RANGE" in (snap.get("reasons") or [])
    assert "LEARNING_POSITIVE_QUALITY_STATUS_PASS" in (snap.get("reasons") or [])
    market_context = snap.get("market_context") if isinstance(snap.get("market_context"), dict) else {}
    learning = market_context.get("learning") if isinstance(market_context.get("learning"), dict) else {}
    assert float(learning.get("risk_score_adjust") or 0.0) == 6.0
