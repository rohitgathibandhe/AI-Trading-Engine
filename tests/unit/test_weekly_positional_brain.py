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
    assert p["min_short_distance_pts"] == 300  # expected_move floor (delta-placement handles the rest)


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


# ── Maturity: cumulative credit, adjustment cap, untested harvest ─────────────────────────────
def test_thresholds_use_cumulative_credit():
    # a leg-in added credit: total 9000. 50% target is now 4500, not 3000.
    pos = dict(POS, total_credit_rupees=9000, adjusted=True)
    r = wb.evaluate_management(pos, {"mtm_rupees": 3200, "max_short_delta": 0.15,
                                     "days_to_expiry": 6, "broader_trend": "BULLISH"})
    assert r["action"] == "HOLD"                      # 3200 < 50% of 9000
    r2 = wb.evaluate_management(pos, {"mtm_rupees": 4600, "max_short_delta": 0.15,
                                      "days_to_expiry": 6, "broader_trend": "BULLISH"})
    assert r2["action"] == "TAKE_PROFIT"              # 4600 >= 4500


def test_adjustment_cap_closes_instead_of_rolling_forever():
    pos = dict(POS, adjusted=True, adjustments=wb.MAX_ADJUSTMENTS)
    r = wb.evaluate_management(pos, {"mtm_rupees": -1000, "max_short_delta": 0.34,
                                     "days_to_expiry": 6, "broader_trend": "BULLISH"})
    assert r["action"] == "CLOSE_MAX_ADJUSTED"


def test_untested_side_is_harvested_when_near_worthless():
    pos = dict(POS, adjusted=True, adjustments=1)
    r = wb.evaluate_management(pos, {"mtm_rupees": 500, "max_short_delta": 0.20,
                                     "days_to_expiry": 6, "broader_trend": "BULLISH",
                                     "untested_decay_pct": 0.85})
    assert r["action"] == "HARVEST_UNTESTED"
