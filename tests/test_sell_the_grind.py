from datetime import time

from market_ai.intraday_defined_risk import strategy_selector as ss


def _route(monkeypatch, sell_on: bool):
    monkeypatch.setattr(ss, "_CONFLUENCE_ONLY", True)
    monkeypatch.setattr(ss, "_NOMOM_SELL", sell_on)
    monkeypatch.setattr(ss, "_high_prob_confluence", lambda read, m: ("BEARISH", "HIGH-PROB BEARISH: test"))
    monkeypatch.setattr(ss, "_has_momentum", lambda m, d: (False, "NO momentum for a buy (rvol 0.6<1.05) — test"))
    meta: dict = {}
    choice = ss.select_strategy(meta, 22400.0, now_time=time(11, 30))
    return choice, meta


def test_bearish_without_momentum_sells_a_bear_call_with_seller_exit(monkeypatch):
    choice, meta = _route(monkeypatch, sell_on=True)
    assert choice.structures == ["BEAR_CALL_CREDIT_SPREAD"]
    assert meta.get("seller_exit") is True


def test_switch_off_restores_stand_aside(monkeypatch):
    choice, meta = _route(monkeypatch, sell_on=False)
    assert choice.structures == []
    assert not meta.get("seller_exit")
