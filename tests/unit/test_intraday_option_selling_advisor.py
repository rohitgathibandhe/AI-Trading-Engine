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


def _advisor_with_config(tmp_path: Path, **cfg_kwargs) -> IntradayOptionSellingAdvisor:
    return IntradayOptionSellingAdvisor(
        config=IntradayOptionSellingAdvisorConfig(**cfg_kwargs),
        status_path=tmp_path / "intraday_ai_advisor_status.json",
        logger=None,
    )


def _chain_rows() -> list[dict]:
    rows: list[dict] = []
    for strike in [24000, 24200, 24400, 24500, 24600, 24750, 24800, 25000, 25200, 25400, 25600, 25800, 26000, 26200, 26400, 26500, 26600, 26800, 27000, 27200]:
        # Coarse synthetic premium curve: higher put strikes carry more premium; lower call strikes carry more premium.
        rows.append({"option_type": "PE", "strike": float(strike), "ltp": max(5.0, (strike - 24000) / 10.0 + 10.0)})
        rows.append({"option_type": "CE", "strike": float(strike), "ltp": max(5.0, (27200 - strike) / 10.0 + 25.0)})
    return rows


def _low_credit_bearish_chain_rows() -> list[dict]:
    rows: list[dict] = []
    ce_map = {
        26200: 20.0,
        26300: 17.0,
        26400: 14.0,
        26500: 15.0,
        26600: 11.0,
        26700: 8.0,
        26800: 4.0,
        26900: 3.0,
        27000: 2.0,
    }
    pe_map = {
        24800: 12.0,
        25000: 18.0,
        25200: 24.0,
        25400: 32.0,
    }
    for strike, ltp in ce_map.items():
        rows.append({"option_type": "CE", "strike": float(strike), "ltp": float(ltp)})
    for strike, ltp in pe_map.items():
        rows.append({"option_type": "PE", "strike": float(strike), "ltp": float(ltp)})
    return rows


def _context(
    *,
    conflict: float = 15.0,
    trend_conf: float = 0.78,
    bias: str = "BULLISH",
    spot: float = 25600.0,
) -> dict:
    ema_alignment = "BULLISH" if bias == "BULLISH" else ("BEARISH" if bias == "BEARISH" else "NEUTRAL")
    ema_2050_alignment = ema_alignment
    orb_confirmation = "UP_CONFIRMED" if bias == "BULLISH" else ("DOWN_CONFIRMED" if bias == "BEARISH" else "NONE")
    price_action_pattern = (
        "BREAKOUT_CONTINUATION_UP"
        if bias == "BULLISH"
        else ("BREAKOUT_CONTINUATION_DOWN" if bias == "BEARISH" else "NONE")
    )
    strong_candle_bias = bias if bias in {"BULLISH", "BEARISH"} else "NEUTRAL"
    strong_candle_status = "STRONG_GREEN_5M" if bias == "BULLISH" else ("STRONG_RED_5M" if bias == "BEARISH" else "NONE")
    strong_candle_desc = (
        "Strong bullish 5m candle closed near the high after the pullback."
        if bias == "BULLISH"
        else (
            "Strong bearish 5m candle closed near the low after the pullback."
            if bias == "BEARISH"
            else "No strong 5m confirmation candle."
        )
    )
    if bias == "BULLISH":
        intraday_support = spot - 40.0
        intraday_resistance = spot + 640.0
        swing_support = intraday_support - 20.0
        swing_resistance = intraday_resistance + 10.0
        put_wall = intraday_support - 120.0
        call_wall = intraday_resistance + 260.0
    elif bias == "BEARISH":
        intraday_support = spot - 640.0
        intraday_resistance = spot + 40.0
        swing_support = intraday_support - 10.0
        swing_resistance = intraday_resistance + 20.0
        put_wall = intraday_support - 260.0
        call_wall = intraday_resistance + 120.0
    else:
        intraday_support = spot - 260.0
        intraday_resistance = spot + 260.0
        swing_support = intraday_support - 20.0
        swing_resistance = intraday_resistance + 20.0
        put_wall = intraday_support - 120.0
        call_wall = intraday_resistance + 120.0
    return {
        "trend": {
            "bias": bias,
            "bias_score": 0.72 if bias == "BULLISH" else (-0.72 if bias == "BEARISH" else 0.05),
            "volatility_regime": "NORMAL",
            "breakout_confirmation": "UP_CONFIRMED" if bias == "BULLISH" else ("DOWN_CONFIRMED" if bias == "BEARISH" else "NONE"),
            "timeframes": {
                "15": {
                    "atr_like_points": 145.0,
                    "pattern": "UPTREND" if bias == "BULLISH" else ("DOWNTREND" if bias == "BEARISH" else "RANGE"),
                    "ema_alignment": ema_alignment,
                    "ema_20_50_alignment": ema_2050_alignment,
                },
                "5": {
                    "atr_like_points": 145.0,
                    "pattern": "UPTREND" if bias == "BULLISH" else ("DOWNTREND" if bias == "BEARISH" else "RANGE"),
                    "ema_alignment": ema_alignment,
                    "ema_20_50_alignment": ema_2050_alignment,
                    "strong_candle_bias": strong_candle_bias,
                    "strong_candle_confirmed": strong_candle_bias in {"BULLISH", "BEARISH"},
                    "strong_candle_status": strong_candle_status,
                    "strong_candle_description": strong_candle_desc,
                },
                "60": {
                    "atr_like_points": 145.0,
                    "pattern": "UPTREND" if bias == "BULLISH" else ("DOWNTREND" if bias == "BEARISH" else "RANGE"),
                    "ema_alignment": ema_alignment,
                    "ema_20_50_alignment": ema_2050_alignment,
                },
            },
            "orb": {"breakout_confirmation": orb_confirmation, "breakout_active": orb_confirmation != "NONE"},
        },
        "option_chain": {
            "pcr_bias": bias,
            "pcr_total": 1.12 if bias == "BULLISH" else 0.78,
            "pcr_near_atm": 1.21 if bias == "BULLISH" else 0.79,
            "put_wall_below": {"strike": float(put_wall), "distance_from_spot": abs(float(spot) - float(put_wall))},
            "call_wall_above": {"strike": float(call_wall), "distance_from_spot": abs(float(call_wall) - float(spot))},
            "oi_build": {"bias": "BULLISH_SUPPORT" if bias == "BULLISH" else "BEARISH_RESISTANCE"},
        },
        "structure": {
            "trend_confidence": trend_conf,
            "signal_conflict_score": conflict,
            "dominant_signal_bias": bias,
            "pcr_unbalanced": False,
            "pcr_unbalanced_side": "NEUTRAL",
            "volatility_regime": "NORMAL",
            "swing_structure_bias": bias,
            "swing_structure_label": "HH_HL_UPTREND" if bias == "BULLISH" else ("LH_LL_DOWNTREND" if bias == "BEARISH" else "RANGE"),
            "swing_structure_confidence": 0.72 if bias in {"BULLISH", "BEARISH"} else 0.0,
            "price_action_bias": bias,
            "price_action_confirmation": "CANDLE_AND_RETEST_CONFIRMED" if bias in {"BULLISH", "BEARISH"} else "NONE",
            "price_action_patterns": [f"5m:{price_action_pattern}", f"15m:{price_action_pattern}"] if price_action_pattern != "NONE" else [],
            "retest_status": "SUPPORT_HOLD" if bias == "BULLISH" else ("RESISTANCE_HOLD" if bias == "BEARISH" else "NONE"),
            "retest_bias": bias,
            "intraday_support": {"level": float(intraday_support), "source": "oi_put_wall_below"},
            "intraday_resistance": {"level": float(intraday_resistance), "source": "oi_call_wall_above"},
            "swing_support": {"level": float(swing_support), "source": "swing_15m_support"},
            "swing_resistance": {"level": float(swing_resistance), "source": "swing_15m_resistance"},
            "weekly_support": float(swing_support - 180.0),
            "weekly_resistance": float(swing_resistance + 180.0),
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
    adv = _advisor_with_config(tmp_path, signal_persistence_bars=1)
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


def test_intraday_advisor_can_emit_short_put_with_emergency_hedge_and_unlimited_trail(tmp_path: Path) -> None:
    adv = _advisor_with_config(
        tmp_path,
        signal_persistence_bars=1,
        directional_structure="SHORT_OPTION_WITH_HEDGE",
        naked_operational_max_loss_rs=3000.0,
        naked_profit_trail_arm_rs=5000.0,
    )
    out = adv.update(
        now=_ist_dt(26, 11, 0),
        expiry="2026-03-03",
        spot=25600.0,
        chain_rows=_chain_rows(),
        context=_context(conflict=14.0, trend_conf=0.84, bias="BULLISH"),
        has_open_bkm=False,
    )
    assert out["signal"] == "ENTER_NOW"
    rec = out["recommendation"]
    assert rec["strategy_type"] == "SHORT_PUT_WITH_HEDGE"
    assert rec["sl"]["loss_rs_per_set"] == 3000.0
    assert rec["tp"]["kind"] == "UNLIMITED_TRAIL"
    assert rec["tp"]["profit_rs_per_set"] == 0.0
    assert rec["trail"]["arm_profit_rs_per_set"] == 5000.0
    assert rec["trail"]["unlimited_profit_mode"] is True


def test_intraday_advisor_relaxes_bearish_directional_threshold_on_strong_alignment(tmp_path: Path) -> None:
    adv = _advisor_with_config(tmp_path, signal_persistence_bars=1)
    out = adv.update(
        now=_ist_dt(26, 10, 35),
        expiry="2026-03-03",
        spot=25600.0,
        chain_rows=_chain_rows(),
        context=_context(conflict=54.0, trend_conf=0.58, bias="BEARISH"),
        has_open_bkm=False,
    )
    assert out["signal"] == "ENTER_NOW"
    assert out["recommendation"]["strategy_type"] == "CALL_CREDIT_SPREAD"


def test_intraday_advisor_accepts_planner_confirmed_bearish_setup_with_blank_option_context(tmp_path: Path) -> None:
    adv = _advisor_with_config(tmp_path, signal_persistence_bars=1)
    ctx = _context(conflict=18.0, trend_conf=0.45, bias="BEARISH", spot=25600.0)
    ctx["option_chain"] = {}
    out = adv.update(
        now=_ist_dt(26, 11, 0),
        expiry="2026-03-03",
        spot=25550.0,
        chain_rows=_chain_rows(),
        context=ctx,
        has_open_bkm=False,
    )
    assert out["signal"] == "ENTER_NOW"
    assert out["recommendation"]["strategy_type"] == "CALL_CREDIT_SPREAD"


def test_intraday_advisor_accepts_lower_credit_when_strong_bearish_setup_is_clean(tmp_path: Path) -> None:
    adv = _advisor_with_config(tmp_path, min_credit_per_set_rs=500.0, signal_persistence_bars=1)
    ctx = _context(conflict=18.0, trend_conf=0.86, bias="BEARISH")
    ctx["option_chain"]["call_wall_above"]["strike"] = 26400.0
    out = adv.update(
        now=_ist_dt(26, 10, 40),
        expiry="2026-03-03",
        spot=25600.0,
        chain_rows=_low_credit_bearish_chain_rows(),
        context=ctx,
        has_open_bkm=False,
    )
    assert out["signal"] == "ENTER_NOW"
    rec = out["recommendation"]
    assert rec["strategy_type"] == "CALL_CREDIT_SPREAD"
    assert rec["est_credit_rs_per_set"] >= 400.0


def test_intraday_advisor_waits_for_signal_persistence_before_enter(tmp_path: Path) -> None:
    adv = _advisor_with_config(tmp_path, signal_persistence_bars=2)
    kwargs = {
        "now": _ist_dt(26, 11, 0),
        "expiry": "2026-03-03",
        "spot": 25600.0,
        "chain_rows": _chain_rows(),
        "context": _context(conflict=18.0, trend_conf=0.82, bias="BULLISH"),
        "has_open_bkm": False,
    }
    first = adv.update(**kwargs)
    assert first["signal"] == "WAIT"
    assert "SIGNAL_NOT_PERSISTED_YET" in (first.get("reasons") or [])
    second = adv.update(**kwargs)
    assert second["signal"] == "ENTER_NOW"
    assert second["recommendation"]["strategy_type"] == "PUT_CREDIT_SPREAD"


def test_intraday_advisor_requires_pullback_zone_before_enter(tmp_path: Path) -> None:
    adv = _advisor_with_config(tmp_path, signal_persistence_bars=1)
    ctx = _context(conflict=18.0, trend_conf=0.84, bias="BEARISH", spot=25600.0)
    out = adv.update(
        now=_ist_dt(26, 11, 0),
        expiry="2026-03-03",
        spot=25480.0,
        chain_rows=_chain_rows(),
        context=ctx,
        has_open_bkm=False,
    )
    assert out["signal"] == "ENTER_NOW"
    assert out["recommendation"]["strategy_type"] == "CALL_CREDIT_SPREAD"
    assert "TREND_CONTINUATION_READY" in (out.get("reasons") or [])


def test_intraday_advisor_accepts_atr_tolerant_pullback_near_zone(tmp_path: Path) -> None:
    adv = _advisor_with_config(tmp_path, signal_persistence_bars=1)
    ctx = _context(conflict=18.0, trend_conf=0.86, bias="BEARISH", spot=25600.0)
    out = adv.update(
        now=_ist_dt(26, 11, 0),
        expiry="2026-03-03",
        spot=25550.0,
        chain_rows=_chain_rows(),
        context=ctx,
        has_open_bkm=False,
    )
    assert out["signal"] == "ENTER_NOW"
    assert out["recommendation"]["strategy_type"] == "CALL_CREDIT_SPREAD"


def test_intraday_advisor_accepts_strong_bearish_trend_continuation_without_pullback_touch(tmp_path: Path) -> None:
    adv = _advisor_with_config(tmp_path, signal_persistence_bars=1)
    ctx = _context(conflict=18.0, trend_conf=0.88, bias="BEARISH", spot=25600.0)
    out = adv.update(
        now=_ist_dt(26, 11, 5),
        expiry="2026-03-03",
        spot=25440.0,
        chain_rows=_chain_rows(),
        context=ctx,
        has_open_bkm=False,
    )
    assert out["signal"] == "ENTER_NOW"
    assert "TREND_CONTINUATION_READY" in (out.get("reasons") or [])
    assert out["recommendation"]["strategy_type"] == "CALL_CREDIT_SPREAD"


def test_intraday_advisor_persistence_ignores_small_strike_shifts(tmp_path: Path) -> None:
    adv = _advisor_with_config(tmp_path, signal_persistence_bars=2)
    first = adv.update(
        now=_ist_dt(26, 11, 0),
        expiry="2026-03-03",
        spot=25600.0,
        chain_rows=_chain_rows(),
        context=_context(conflict=18.0, trend_conf=0.82, bias="BULLISH"),
        has_open_bkm=False,
    )
    assert first["signal"] == "WAIT"
    second = adv.update(
        now=_ist_dt(26, 11, 1),
        expiry="2026-03-03",
        spot=25580.0,
        chain_rows=_chain_rows(),
        context=_context(conflict=18.0, trend_conf=0.82, bias="BULLISH"),
        has_open_bkm=False,
    )
    assert second["signal"] == "ENTER_NOW"
    assert second["recommendation"]["strategy_type"] == "PUT_CREDIT_SPREAD"


def test_intraday_advisor_fast_tracks_strong_bearish_trend_day(tmp_path: Path) -> None:
    adv = _advisor_with_config(tmp_path, signal_persistence_bars=2, preferred_bias="BEARISH")
    out = adv.update(
        now=_ist_dt(26, 10, 41),
        expiry="2026-03-03",
        spot=25600.0,
        chain_rows=_chain_rows(),
        context=_context(conflict=35.0, trend_conf=0.90, bias="BEARISH"),
        has_open_bkm=False,
    )
    assert out["signal"] == "ENTER_NOW"
    assert out["recommendation"]["strategy_type"] == "CALL_CREDIT_SPREAD"


def test_intraday_advisor_delays_high_vol_bear_call_before_1015(tmp_path: Path) -> None:
    adv = _advisor_with_config(tmp_path, signal_persistence_bars=1, preferred_bias="BEARISH")
    ctx = _context(conflict=24.0, trend_conf=0.95, bias="BEARISH")
    ctx["trend"]["volatility_regime"] = "HIGH"
    ctx["structure"]["volatility_regime"] = "HIGH"
    out = adv.update(
        now=_ist_dt(26, 10, 7),
        expiry="2026-03-03",
        spot=25600.0,
        chain_rows=_chain_rows(),
        context=ctx,
        has_open_bkm=False,
    )
    assert out["signal"] == "WAIT"
    assert "HIGH_VOL_BEAR_CALL_DELAY" in (out.get("reasons") or [])


def test_intraday_advisor_blocks_one_bar_countertrend_flip_and_preserves_candidate(tmp_path: Path) -> None:
    adv = _advisor_with_config(
        tmp_path,
        signal_persistence_bars=2,
        preferred_bias="BEARISH",
        trend_day_fast_track_min_trend_confidence=0.90,
    )
    first = adv.update(
        now=_ist_dt(26, 10, 41),
        expiry="2026-03-03",
        spot=25600.0,
        chain_rows=_chain_rows(),
        context=_context(conflict=35.0, trend_conf=0.74, bias="BEARISH"),
        has_open_bkm=False,
    )
    assert first["signal"] == "WAIT"
    assert first["strategy"] == "CALL_CREDIT_SPREAD"
    first_signature = first.get("candidate_setup_signature")
    second = adv.update(
        now=_ist_dt(26, 10, 43),
        expiry="2026-03-03",
        spot=25620.0,
        chain_rows=_chain_rows(),
        context=_context(conflict=52.0, trend_conf=0.79, bias="BULLISH", spot=25620.0),
        has_open_bkm=False,
    )
    assert second["signal"] == "WAIT"
    assert "COUNTERTREND_FLIP_BLOCKED" in (second.get("reasons") or [])
    assert second.get("candidate_setup_signature") == first_signature
    assert second.get("candidate_signal_streak") == 1
    third = adv.update(
        now=_ist_dt(26, 10, 47),
        expiry="2026-03-03",
        spot=25570.0,
        chain_rows=_chain_rows(),
        context=_context(conflict=43.0, trend_conf=0.80, bias="BEARISH", spot=25570.0),
        has_open_bkm=False,
    )
    assert third["signal"] == "ENTER_NOW"
    assert third["recommendation"]["strategy_type"] == "CALL_CREDIT_SPREAD"


def test_intraday_advisor_blocks_directional_entry_when_ema_not_aligned(tmp_path: Path) -> None:
    adv = _advisor_with_config(tmp_path, signal_persistence_bars=1)
    ctx = _context(conflict=18.0, trend_conf=0.82, bias="BULLISH")
    ctx["trend"]["timeframes"]["5"]["ema_alignment"] = "BEARISH"
    out = adv.update(
        now=_ist_dt(26, 11, 0),
        expiry="2026-03-03",
        spot=25600.0,
        chain_rows=_chain_rows(),
        context=ctx,
        has_open_bkm=False,
    )
    assert out["signal"] == "WAIT"
    assert "EMA_ALIGNMENT_MISSING" in (out.get("reasons") or [])


def test_intraday_advisor_requires_strong_5m_candle_after_pullback(tmp_path: Path) -> None:
    adv = _advisor_with_config(tmp_path, signal_persistence_bars=1)
    ctx = _context(conflict=18.0, trend_conf=0.82, bias="BULLISH")
    ctx["trend"]["timeframes"]["5"]["strong_candle_bias"] = "NEUTRAL"
    ctx["trend"]["timeframes"]["5"]["strong_candle_confirmed"] = False
    ctx["trend"]["timeframes"]["5"]["strong_candle_status"] = "NONE"
    ctx["trend"]["timeframes"]["5"]["strong_candle_description"] = "No strong 5m confirmation candle."
    out = adv.update(
        now=_ist_dt(26, 11, 0),
        expiry="2026-03-03",
        spot=25600.0,
        chain_rows=_chain_rows(),
        context=ctx,
        has_open_bkm=False,
    )
    assert out["signal"] == "WAIT"
    assert "STRONG_5M_CANDLE_MISSING" in (out.get("reasons") or [])


def test_intraday_advisor_blocks_directional_entry_when_orb_not_confirmed(tmp_path: Path) -> None:
    adv = _advisor_with_config(tmp_path, signal_persistence_bars=1)
    ctx = _context(conflict=18.0, trend_conf=0.82, bias="BEARISH")
    ctx["trend"]["orb"]["breakout_confirmation"] = "NONE"
    ctx["trend"]["orb"]["breakout_active"] = False
    out = adv.update(
        now=_ist_dt(26, 11, 0),
        expiry="2026-03-03",
        spot=25600.0,
        chain_rows=_chain_rows(),
        context=ctx,
        has_open_bkm=False,
    )
    assert out["signal"] == "WAIT"
    assert "ORB_CONFIRMATION_MISSING" in (out.get("reasons") or [])


def test_intraday_advisor_blocks_directional_entry_when_price_action_not_confirmed(tmp_path: Path) -> None:
    adv = _advisor_with_config(tmp_path, signal_persistence_bars=1)
    ctx = _context(conflict=18.0, trend_conf=0.82, bias="BEARISH")
    ctx["structure"]["price_action_bias"] = "NEUTRAL"
    ctx["structure"]["price_action_confirmation"] = "NONE"
    ctx["structure"]["price_action_patterns"] = []
    ctx["structure"]["retest_status"] = "NONE"
    ctx["structure"]["retest_bias"] = "NEUTRAL"
    out = adv.update(
        now=_ist_dt(26, 11, 0),
        expiry="2026-03-03",
        spot=25600.0,
        chain_rows=_chain_rows(),
        context=ctx,
        has_open_bkm=False,
    )
    assert out["signal"] == "WAIT"
    assert "PRICE_ACTION_CONFIRMATION_MISSING" in (out.get("reasons") or [])


def test_intraday_advisor_allows_bearish_reversal_trade_with_fake_break_confirmation(tmp_path: Path) -> None:
    adv = _advisor_with_config(tmp_path, signal_persistence_bars=1, preferred_bias="BEARISH")
    ctx = _context(conflict=18.0, trend_conf=0.84, bias="BEARISH")
    ctx["structure"]["price_action_bias"] = "BEARISH"
    ctx["structure"]["price_action_confirmation"] = "REVERSAL_CONFIRMED"
    ctx["structure"]["price_action_patterns"] = ["5m:FAKE_BREAKOUT_UP", "15m:EMA20_BREAKDOWN_DOWN"]
    ctx["structure"]["retest_status"] = "FAILED_BREAKOUT_RETEST"
    ctx["structure"]["retest_bias"] = "BEARISH"
    out = adv.update(
        now=_ist_dt(26, 11, 0),
        expiry="2026-03-03",
        spot=25600.0,
        chain_rows=_chain_rows(),
        context=ctx,
        has_open_bkm=False,
    )
    assert out["signal"] == "ENTER_NOW"
    assert out["recommendation"]["strategy_type"] == "CALL_CREDIT_SPREAD"


def test_intraday_advisor_blocks_trade_on_fake_break_conflict_from_chart_plan(tmp_path: Path) -> None:
    adv = _advisor_with_config(tmp_path, signal_persistence_bars=1)
    ctx = _context(conflict=18.0, trend_conf=0.84, bias="BEARISH")
    ctx["structure"]["primary_pattern"] = "FAKE_BREAKDOWN_DOWN"
    ctx["structure"]["price_action_patterns"] = ["5m:FAKE_BREAKDOWN_DOWN"]
    ctx["structure"]["price_action_confirmation"] = "REVERSAL_CONFIRMED"
    out = adv.update(
        now=_ist_dt(26, 11, 0),
        expiry="2026-03-03",
        spot=25600.0,
        chain_rows=_chain_rows(),
        context=ctx,
        has_open_bkm=False,
    )
    assert out["signal"] == "WAIT"
    assert "REVERSAL_PATTERN_AGAINST_SETUP" in (out.get("reasons") or [])


def test_intraday_advisor_blocks_trade_when_confirmed_reversal_pattern_is_against_setup(tmp_path: Path) -> None:
    adv = _advisor_with_config(tmp_path, signal_persistence_bars=1)
    ctx = _context(conflict=18.0, trend_conf=0.82, bias="BULLISH")
    ctx["structure"]["price_action_bias"] = "BEARISH"
    ctx["structure"]["price_action_confirmation"] = "REVERSAL_CONFIRMED"
    ctx["structure"]["price_action_patterns"] = ["5m:FAKE_BREAKOUT_UP", "15m:DOUBLE_TOP_M_CONFIRMED"]
    ctx["structure"]["retest_status"] = "NONE"
    ctx["structure"]["retest_bias"] = "NEUTRAL"
    out = adv.update(
        now=_ist_dt(26, 11, 0),
        expiry="2026-03-03",
        spot=25600.0,
        chain_rows=_chain_rows(),
        context=ctx,
        has_open_bkm=False,
    )
    assert out["signal"] == "WAIT"
    assert "REVERSAL_PATTERN_AGAINST_SETUP" in (out.get("reasons") or [])


def test_intraday_advisor_requires_supportive_retest_in_high_volatility(tmp_path: Path) -> None:
    adv = _advisor_with_config(tmp_path, signal_persistence_bars=1, preferred_bias="BEARISH")
    ctx = _context(conflict=18.0, trend_conf=0.82, bias="BEARISH")
    ctx["trend"]["volatility_regime"] = "HIGH"
    ctx["structure"]["volatility_regime"] = "HIGH"
    ctx["structure"]["price_action_confirmation"] = "CANDLE_CONFIRMED"
    ctx["structure"]["retest_status"] = "NONE"
    ctx["structure"]["retest_bias"] = "NEUTRAL"
    out = adv.update(
        now=_ist_dt(26, 11, 0),
        expiry="2026-03-03",
        spot=25600.0,
        chain_rows=_chain_rows(),
        context=ctx,
        has_open_bkm=False,
    )
    assert out["signal"] == "WAIT"
    assert "HIGH_VOL_RETEST_REQUIRED" in (out.get("reasons") or [])


def test_intraday_advisor_blocks_high_vol_bullish_countertrend_put_spread_for_bearish_preference(tmp_path: Path) -> None:
    adv = _advisor_with_config(tmp_path, signal_persistence_bars=1, preferred_bias="BEARISH")
    ctx = _context(conflict=18.0, trend_conf=0.82, bias="BULLISH")
    ctx["trend"]["volatility_regime"] = "HIGH"
    ctx["structure"]["volatility_regime"] = "HIGH"
    out = adv.update(
        now=_ist_dt(26, 11, 0),
        expiry="2026-03-03",
        spot=25600.0,
        chain_rows=_chain_rows(),
        context=ctx,
        has_open_bkm=False,
    )
    assert out["signal"] == "WAIT"
    assert "HIGH_VOL_BULL_PUT_DISABLED" in (out.get("reasons") or [])


def test_intraday_advisor_can_reenable_high_vol_bull_put_for_bearish_preference(tmp_path: Path) -> None:
    adv = _advisor_with_config(
        tmp_path,
        signal_persistence_bars=1,
        preferred_bias="BEARISH",
        high_vol_bull_put_enabled_for_bearish_preference=True,
    )
    ctx = _context(conflict=18.0, trend_conf=0.90, bias="BULLISH")
    ctx["trend"]["volatility_regime"] = "HIGH"
    ctx["structure"]["volatility_regime"] = "HIGH"
    out = adv.update(
        now=_ist_dt(26, 11, 0),
        expiry="2026-03-03",
        spot=25600.0,
        chain_rows=_chain_rows(),
        context=ctx,
        has_open_bkm=False,
    )
    assert out["signal"] == "ENTER_NOW"
    assert out["recommendation"]["strategy_type"] == "PUT_CREDIT_SPREAD"


def test_intraday_advisor_blocks_directional_entry_when_retest_fails_against_setup(tmp_path: Path) -> None:
    adv = _advisor_with_config(tmp_path, signal_persistence_bars=1)
    ctx = _context(conflict=18.0, trend_conf=0.82, bias="BULLISH")
    ctx["structure"]["price_action_bias"] = "BULLISH"
    ctx["structure"]["price_action_confirmation"] = "CANDLE_CONFIRMED"
    ctx["structure"]["retest_status"] = "RETEST_FAILED"
    ctx["structure"]["retest_bias"] = "BEARISH"
    out = adv.update(
        now=_ist_dt(26, 11, 0),
        expiry="2026-03-03",
        spot=25600.0,
        chain_rows=_chain_rows(),
        context=ctx,
        has_open_bkm=False,
    )
    assert out["signal"] == "WAIT"
    assert "RETEST_FAILED_AGAINST_SETUP" in (out.get("reasons") or [])


def test_intraday_advisor_emits_entry_features_on_enter_now(tmp_path: Path) -> None:
    adv = _advisor_with_config(tmp_path, signal_persistence_bars=1)
    out = adv.update(
        now=_ist_dt(26, 11, 0),
        expiry="2026-03-03",
        spot=25600.0,
        chain_rows=_chain_rows(),
        context=_context(conflict=18.0, trend_conf=0.82, bias="BULLISH"),
        has_open_bkm=False,
    )
    assert out["signal"] == "ENTER_NOW"
    features = out.get("entry_features") if isinstance(out.get("entry_features"), dict) else {}
    assert features.get("strategy_type") == "PUT_CREDIT_SPREAD"
    assert features.get("price_action_bias") == "BULLISH"
    assert isinstance(out["recommendation"].get("entry_features"), dict)


def test_intraday_advisor_blocks_directional_entry_when_option_chain_is_strongly_against_setup(tmp_path: Path) -> None:
    adv = _advisor_with_config(tmp_path, signal_persistence_bars=1)
    ctx = _context(conflict=18.0, trend_conf=0.82, bias="BEARISH")
    ctx["option_chain"]["pcr_bias"] = "BULLISH"
    ctx["option_chain"]["pcr_total"] = 1.42
    ctx["option_chain"]["pcr_near_atm"] = 1.36
    ctx["option_chain"]["oi_build"]["bias"] = "BULLISH_SUPPORT"
    ctx["structure"]["pcr_unbalanced"] = True
    ctx["structure"]["pcr_unbalanced_side"] = "BULLISH"
    out = adv.update(
        now=_ist_dt(26, 11, 0),
        expiry="2026-03-03",
        spot=25600.0,
        chain_rows=_chain_rows(),
        context=ctx,
        has_open_bkm=False,
    )
    assert out["signal"] == "WAIT"
    assert "OPTION_CHAIN_AGAINST_SETUP" in (out.get("reasons") or [])


def test_intraday_advisor_blocks_directional_entry_when_swing_structure_is_against_setup(tmp_path: Path) -> None:
    adv = _advisor_with_config(tmp_path, signal_persistence_bars=1)
    ctx = _context(conflict=18.0, trend_conf=0.82, bias="BULLISH")
    ctx["structure"]["swing_structure_bias"] = "BEARISH"
    ctx["structure"]["swing_structure_label"] = "LH_LL_DOWNTREND"
    ctx["structure"]["swing_structure_confidence"] = 0.81
    out = adv.update(
        now=_ist_dt(26, 11, 0),
        expiry="2026-03-03",
        spot=25600.0,
        chain_rows=_chain_rows(),
        context=ctx,
        has_open_bkm=False,
    )
    assert out["signal"] == "WAIT"
    assert "SWING_STRUCTURE_AGAINST_SETUP" in (out.get("reasons") or [])
