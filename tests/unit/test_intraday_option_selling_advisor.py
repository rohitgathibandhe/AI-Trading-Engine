from __future__ import annotations

from datetime import datetime
from pathlib import Path

try:
    from zoneinfo import ZoneInfo
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from market_ai.modules.agents.intraday_option_selling_advisor import (
    IntradayOptionSellingAdvisor,
    IntradayOptionSellingAdvisorConfig,
)


def _ist_dt(day: int, hour: int = 10, minute: int = 0) -> datetime:
    tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    return datetime(2026, 2, day, hour, minute, tzinfo=tz)


def _advisor(tmp_path: Path) -> IntradayOptionSellingAdvisor:
    return IntradayOptionSellingAdvisor(
        config=IntradayOptionSellingAdvisorConfig(),
        status_path=tmp_path / "intraday_ai_advisor_status.json",
        logger=None,
    )


def _chain_rows() -> list[dict]:
    rows: list[dict] = []
    for strike in [24000, 24200, 24400, 24500, 24600, 24750, 24800, 25000, 25200, 25400, 25600, 25800, 26000, 26200, 26400, 26500, 26600, 26800, 27000, 27200]:
        # Coarse synthetic premium curve: higher put strikes carry more premium; higher call strikes carry more premium.
        rows.append({"option_type": "PE", "strike": float(strike), "ltp": max(5.0, (strike - 24000) / 10.0 + 10.0)})
        rows.append({"option_type": "CE", "strike": float(strike), "ltp": max(5.0, (strike - 25000) / 10.0 + 25.0)})
    return rows


def _context(*, conflict: float = 15.0, trend_conf: float = 0.78, bias: str = "BULLISH") -> dict:
    return {
        "trend": {
            "bias": bias,
            "bias_score": 0.72 if bias == "BULLISH" else (-0.72 if bias == "BEARISH" else 0.05),
            "volatility_regime": "NORMAL",
            "breakout_confirmation": "UP_CONFIRMED" if bias == "BULLISH" else ("DOWN_CONFIRMED" if bias == "BEARISH" else "NONE"),
            "timeframes": {
                "15": {"atr_like_points": 145.0, "pattern": "UPTREND" if bias == "BULLISH" else "DOWNTREND"},
                "5": {"pattern": "UPTREND" if bias == "BULLISH" else "DOWNTREND"},
                "60": {"pattern": "UPTREND" if bias == "BULLISH" else "DOWNTREND"},
            },
        },
        "option_chain": {
            "pcr_bias": bias,
            "pcr_total": 1.12 if bias == "BULLISH" else 0.78,
            "pcr_near_atm": 1.21 if bias == "BULLISH" else 0.79,
            "put_wall_below": {"strike": 25000.0, "distance_from_spot": 600.0},
            "call_wall_above": {"strike": 26500.0, "distance_from_spot": 900.0},
            "oi_build": {"bias": "BULLISH_SUPPORT" if bias == "BULLISH" else "BEARISH_RESISTANCE"},
        },
        "structure": {
            "trend_confidence": trend_conf,
            "signal_conflict_score": conflict,
            "dominant_signal_bias": bias,
            "pcr_unbalanced": False,
            "pcr_unbalanced_side": "NEUTRAL",
            "volatility_regime": "NORMAL",
            "intraday_support": {"level": 25120.0, "source": "oi_put_wall_below"},
            "intraday_resistance": {"level": 26240.0, "source": "oi_call_wall_above"},
            "weekly_support": 24850.0,
            "weekly_resistance": 26680.0,
        },
    }


def test_intraday_advisor_no_trade_when_bkm_open(tmp_path: Path) -> None:
    adv = _advisor(tmp_path)
    out = adv.update(
        now=_ist_dt(26, 10, 10),
        expiry="2026-03-03",
        spot=25600.0,
        chain_rows=_chain_rows(),
        context=_context(),
        has_open_bkm=True,
    )
    assert out["signal"] == "NO_TRADE"
    assert out["recommendation"]["headline"].lower().startswith("no new intraday trade")
    assert "overtrading" in out["recommendation"]["signal_text"].lower()


def test_intraday_advisor_no_trade_on_high_conflict(tmp_path: Path) -> None:
    adv = _advisor(tmp_path)
    out = adv.update(
        now=_ist_dt(26, 10, 25),
        expiry="2026-03-03",
        spot=25600.0,
        chain_rows=_chain_rows(),
        context=_context(conflict=78.0, trend_conf=0.55, bias="NEUTRAL"),
        has_open_bkm=False,
    )
    assert out["signal"] == "NO_TRADE"
    assert "SIGNAL_CONFLICT_HIGH" in (out.get("reasons") or [])
    assert "conflicting" in out["recommendation"]["headline"].lower()


def test_intraday_advisor_recommends_bull_put_spread_with_sl_tp(tmp_path: Path) -> None:
    adv = _advisor(tmp_path)
    out = adv.update(
        now=_ist_dt(26, 11, 0),
        expiry="2026-03-03",
        spot=25600.0,
        chain_rows=_chain_rows(),
        context=_context(conflict=18.0, trend_conf=0.82, bias="BULLISH"),
        has_open_bkm=False,
    )
    assert out["signal"] == "ENTER_NOW"
    rec = out["recommendation"]
    assert rec["strategy_type"] in {"PUT_CREDIT_SPREAD", "IRON_CONDOR", "CALL_CREDIT_SPREAD"}
    assert "what_to_enter" in rec and rec["what_to_enter"]
    assert "SL:" in rec["sl"]["text"]
    assert "TP:" in rec["tp"]["text"]
    assert "Hold till" in rec["hold"]["text"]
    assert rec["legs"]
    # When bullish bias is strong, the default expectation is a put credit spread.
    assert rec["strategy_type"] == "PUT_CREDIT_SPREAD"
