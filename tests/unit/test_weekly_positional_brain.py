"""Tests for the weekly positional decision brain: entry direction/structure + the defense ladder."""
import market_ai.intraday_defined_risk.weekly_positional_brain as wb


def _entry(**m):
    base = {"atm_strike": 24000, "daily_ema20": 23800, "daily_ema50": 23700,
            "vol_regime": "RICH_SELL", "expected_move_pts": 300, "daily_atr": 150}
    base.update(m)
    return wb.plan_entry(base)


def test_entry_uptrend_sells_bull_put():
    p = _entry(daily_trend="BULLISH")
    assert p["action"] == "TRADE" and p["structure"] == "BULL_PUT_CREDIT_SPREAD"
    assert p["min_short_distance_pts"] == 450  # expected_move 300 + 1*ATR 150 (gap-safe)


def test_entry_downtrend_sells_bear_call():
    p = _entry(daily_trend="BEARISH", daily_ema20=24200, daily_ema50=24300)
    assert p["action"] == "TRADE" and p["structure"] == "BEAR_CALL_CREDIT_SPREAD"


def test_entry_neutral_sells_condor():
    p = _entry(daily_trend="NEUTRAL", daily_ema20=24010, daily_ema50=23990)
    assert p["action"] == "TRADE" and p["structure"] == "IRON_CONDOR"


def test_entry_stands_aside_on_thin_premium():
    assert _entry(daily_trend="BULLISH", vol_regime="CHEAP_BUY")["action"] == "STAND_ASIDE"


def test_broader_trend_does_not_fight_the_daily_ema():
    # label says bullish but price is below both daily EMAs -> neutral, don't fight the trend
    assert wb.broader_trend({"daily_trend": "BULLISH", "atm_strike": 24000,
                             "daily_ema20": 24200, "daily_ema50": 24300}) == "NEUTRAL"


# ── Defense ladder ────────────────────────────────────────────────────────────────────────────
POS = {"direction": "BULLISH", "structure": "BULL_PUT_CREDIT_SPREAD", "credit_rupees": 6000, "adjusted": False}


def _mgmt(**cur):
    base = {"mtm_rupees": 0, "max_short_delta": 0.15, "days_to_expiry": 7, "broader_trend": "BULLISH"}
    base.update(cur)
    return wb.evaluate_management(POS, base)["action"]


def test_take_profit_at_50pct():
    assert _mgmt(mtm_rupees=3000) == "TAKE_PROFIT"      # 50% of 6000
    assert _mgmt(mtm_rupees=2900) == "HOLD"             # below target


def test_stop_at_2x_credit():
    assert _mgmt(mtm_rupees=-12000) == "STOP_CLOSE"     # 2x of 6000
    assert _mgmt(mtm_rupees=-8000, max_short_delta=0.20) == "HOLD"


def test_gamma_cliff_exit():
    assert _mgmt(days_to_expiry=2) == "GAMMA_EXIT"


def test_thesis_broken_on_trend_flip():
    assert _mgmt(broader_trend="BEARISH") == "CLOSE_THESIS_BROKEN"


def test_short_tested_legs_into_condor_then_rolls_out():
    assert _mgmt(max_short_delta=0.34) == "LEG_INTO_CONDOR"
    condor = dict(POS, adjusted=True)
    r = wb.evaluate_management(condor, {"mtm_rupees": -2000, "max_short_delta": 0.34,
                                        "days_to_expiry": 6, "broader_trend": "BULLISH"})
    assert r["action"] == "ROLL_OUT"


def test_quiet_position_holds_for_theta():
    assert _mgmt(mtm_rupees=800, max_short_delta=0.18) == "HOLD"
