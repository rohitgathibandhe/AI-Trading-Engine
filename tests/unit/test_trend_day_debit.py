from datetime import time

from market_ai.intraday_defined_risk import strategy_selector as sel


def _choice(struct):
    c = sel.StrategyChoice(condition="CHOP", iv_regime="NORMAL", read=None, conviction=0.5)
    c.family, c.structures, c.rationale = sel.FAM_DIRECTIONAL_CREDIT, [struct], "x"
    return c


def test_with_trend_credit_becomes_debit_on_trend_day():
    md = {"trend_day": {"state": "TREND_DOWN"}, "seller_exit": True, "is_fade": True}
    c = sel._trend_day_structure(_choice("BEAR_CALL_CREDIT_SPREAD"), md, time(11, 0))
    assert c.structures == ["PUT_DEBIT_SPREAD"] and md.get("trend_day_debit")
    assert "seller_exit" not in md and "is_fade" not in md


def test_counter_trend_or_no_trend_day_unchanged():
    md = {"trend_day": {"state": "TREND_UP"}}
    assert sel._trend_day_structure(_choice("BEAR_CALL_CREDIT_SPREAD"), md, time(11, 0)).structures == ["BEAR_CALL_CREDIT_SPREAD"]
    md = {"trend_day": {"state": "NONE"}}
    assert sel._trend_day_structure(_choice("BULL_PUT_CREDIT_SPREAD"), md, time(11, 0)).structures == ["BULL_PUT_CREDIT_SPREAD"]


def test_respects_debit_blackout_unless_exempt(monkeypatch):
    monkeypatch.setattr(sel, "_DEBIT_BLACKOUT", (time(10, 0), time(10, 30)))
    md = {"trend_day": {"state": "TREND_UP"}}
    assert sel._trend_day_structure(_choice("BULL_PUT_CREDIT_SPREAD"), md, time(10, 10)).structures == ["BULL_PUT_CREDIT_SPREAD"]
    monkeypatch.setattr(sel, "_TREND_DAY_BLACKOUT_EXEMPT", True)
    assert sel._trend_day_structure(_choice("BULL_PUT_CREDIT_SPREAD"), dict(md), time(10, 10)).structures == ["CALL_DEBIT_SPREAD"]


def test_toggle_off(monkeypatch):
    monkeypatch.setattr(sel, "_TREND_DAY_BUY", False)
    md = {"trend_day": {"state": "TREND_UP"}}
    assert sel._trend_day_structure(_choice("BULL_PUT_CREDIT_SPREAD"), md, time(11, 0)).structures == ["BULL_PUT_CREDIT_SPREAD"]
