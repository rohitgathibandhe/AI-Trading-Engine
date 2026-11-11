from market_ai.intraday_defined_risk import strategy_selector as ss
from market_ai.intraday_defined_risk.strategy_selector import StrategyChoice


def _run(monkeypatch, env, dr):
    monkeypatch.setattr(ss, "_EDGE_ONLY", True)
    if env is None:
        monkeypatch.delenv("SEL_TWO_SIDED_DAYREAD", raising=False)
    else:
        monkeypatch.setenv("SEL_TWO_SIDED_DAYREAD", env)
    md = {"trend_day": {"state": "RANGE_TWO_SIDED", "day_read": dr}}
    c = StrategyChoice(condition="BREAKOUT_DOWN", family="X", structures=["PUT_DEBIT_SPREAD"], rationale="")
    ss._edge_only(c, md, None)
    return md


def test_two_sided_still_blocks_by_default(monkeypatch):
    md = _run(monkeypatch, None, {"efficiency": 0.61, "move_pct": -0.59, "side": -0.2})
    assert md.get("edge_only_two_sided") and "two_sided_dayread_keep" not in md


def test_two_sided_kept_when_read_agrees(monkeypatch):
    md = _run(monkeypatch, "0.55,0.4", {"efficiency": 0.61, "move_pct": -0.59, "side": -0.2})
    assert md.get("two_sided_dayread_keep") and not md.get("edge_only_two_sided")


def test_two_sided_blocked_when_read_disagrees(monkeypatch):
    md = _run(monkeypatch, "0.55,0.4", {"efficiency": 0.61, "move_pct": 0.59, "side": 0.2})
    assert md.get("edge_only_two_sided")


def _fade(monkeypatch, env, first="BULL_PUT_CREDIT_SPREAD", state="NONE"):
    monkeypatch.setattr(ss, "_EDGE_ONLY", True)
    monkeypatch.setenv("SEL_EDGE_FADE", env)
    md = {"trend_day": {"state": state}, "is_fade": True, "fade_stop_spot": 25330.0, "seller_exit": True}
    c = StrategyChoice(condition="RANGE_WIDE", family="X", structures=[first], rationale="")
    return ss._edge_only(c, md, None), md


def test_fade_off_by_default(monkeypatch):
    c, _ = _fade(monkeypatch, "0")
    assert c.structures == []


def test_fade_kept_as_matching_debit(monkeypatch):
    c, md = _fade(monkeypatch, "1")
    assert c.structures == ["CALL_DEBIT_SPREAD"] and "seller_exit" not in md and md["is_fade"]
    c, _ = _fade(monkeypatch, "1", first="BEAR_CALL_CREDIT_SPREAD")
    assert c.structures == ["PUT_DEBIT_SPREAD"]


def test_fade_not_on_trend_day(monkeypatch):
    c, _ = _fade(monkeypatch, "1", state="TREND_DOWN")
    assert c.structures == []
