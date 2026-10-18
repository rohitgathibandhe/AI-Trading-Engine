from datetime import time

from market_ai.intraday_defined_risk import strategy_selector as sel


def _c(cond, struct):
    c = sel.StrategyChoice(condition=cond, iv_regime="NORMAL", read=None, conviction=0.5)
    c.family, c.structures, c.rationale = sel.FAM_DIRECTIONAL_CREDIT, [struct], "x"
    return c


def test_edge_only_keeps_breakout_debits_and_converts_breakout_credits(monkeypatch):
    monkeypatch.setattr(sel, "_EDGE_ONLY", True)
    monkeypatch.setattr(sel, "_DEBIT_BLACKOUT", None)
    assert sel._edge_only(_c("BREAKOUT_UP", "CALL_DEBIT_SPREAD"), {}, time(11, 0)).structures == ["CALL_DEBIT_SPREAD"]
    md = {"seller_exit": True}
    assert sel._edge_only(_c("BREAKOUT_DOWN", "BEAR_CALL_CREDIT_SPREAD"), md, time(11, 0)).structures == ["PUT_DEBIT_SPREAD"]
    assert "seller_exit" not in md


def test_edge_only_stands_aside_elsewhere(monkeypatch):
    monkeypatch.setattr(sel, "_EDGE_ONLY", True)
    monkeypatch.setattr(sel, "_DEBIT_BLACKOUT", (time(10, 0), time(10, 30)))
    assert sel._edge_only(_c("CHOP", "BEAR_CALL_CREDIT_SPREAD"), {}, time(11, 0)).structures == []
    assert sel._edge_only(_c("BREAKOUT_UP", "BEAR_CALL_CREDIT_SPREAD"), {}, time(11, 0)).structures == []
    assert sel._edge_only(_c("BREAKOUT_UP", "CALL_DEBIT_SPREAD"), {}, time(10, 10)).structures == []


def test_off_by_default(monkeypatch):
    monkeypatch.setattr(sel, "_EDGE_ONLY", False)
    assert sel._edge_only(_c("CHOP", "BEAR_CALL_CREDIT_SPREAD"), {}, time(11, 0)).structures == ["BEAR_CALL_CREDIT_SPREAD"]
