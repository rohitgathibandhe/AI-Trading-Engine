from datetime import time

from market_ai.intraday_defined_risk import strategy_selector as ss
from market_ai.intraday_defined_risk.volatility_engine import RICH_SELL


def _mom(ok):
    return lambda m, d: (ok, "momentum OK" if ok else "NO momentum for a buy")


def test_fast_clean_move_buys(monkeypatch):
    monkeypatch.setattr(ss, "_has_momentum", _mom(True))
    side, _, det = ss.choose_buy_or_sell("BEARISH", {"trend_efficiency_ratio": 0.7}, context="trend", now_time=time(11, 0))
    assert side == "BUY" and det["buy_score"] >= det["sell_score"] + 2


def test_grind_sells(monkeypatch):
    monkeypatch.setattr(ss, "_has_momentum", _mom(False))
    side, _, _ = ss.choose_buy_or_sell("BEARISH", {"trend_efficiency_ratio": 0.4}, context="trend", now_time=time(11, 0))
    assert side == "SELL"


def test_seller_first_momentum_alone_is_not_enough_late_in_rich_chop(monkeypatch):
    monkeypatch.setattr(ss, "_has_momentum", _mom(True))
    side, _, _ = ss.choose_buy_or_sell("BEARISH", {"trend_efficiency_ratio": 0.2}, context="trend",
                                       vol_regime=RICH_SELL, now_time=time(14, 0))
    assert side == "SELL"


def test_2026_10_12_fades_would_have_sold(monkeypatch):
    # 12:06 / 13:07 fades: completed-bar rvol < 1 (no momentum), eff 0.22 / 0.00, weak 22,382 level
    monkeypatch.setattr(ss, "_has_momentum", _mom(False))
    for eff, t in ((0.2219, time(12, 6)), (0.0015, time(13, 7))):
        side, why, _ = ss.choose_buy_or_sell("BEARISH", {"trend_efficiency_ratio": eff, "distance_to_call_wall": 118},
                                             context="fade", now_time=t)
        assert side == "SELL", why


def test_strong_fade_with_momentum_still_buys(monkeypatch):
    monkeypatch.setattr(ss, "_has_momentum", _mom(True))
    side, _, _ = ss.choose_buy_or_sell("BEARISH", {"trend_efficiency_ratio": 0.6, "distance_to_call_wall": 200},
                                       context="fade", now_time=time(10, 52))
    assert side == "BUY"


def test_blackout_turns_a_buy_into_a_sell(monkeypatch):
    monkeypatch.setattr(ss, "_CONFLUENCE_ONLY", True)
    monkeypatch.setattr(ss, "_STRUCT_CHOOSER", True)
    monkeypatch.setattr(ss, "_DEBIT_BLACKOUT", (time(10, 0), time(10, 30)))   # live plist: SEL_DEBIT_BLACKOUT
    monkeypatch.setattr(ss, "_high_prob_confluence", lambda read, m: ("BEARISH", "HIGH-PROB BEARISH: test"))
    monkeypatch.setattr(ss, "_has_momentum", _mom(True))
    meta = {"trend_efficiency_ratio": 0.8}
    choice = ss.select_strategy(meta, 22400.0, now_time=time(10, 10))     # inside 10:00-10:30
    assert choice.structures == ["BEAR_CALL_CREDIT_SPREAD"] and meta["seller_exit"] is True
    assert meta["structure_choice"]["side"] == "SELL"


def test_morning_momentum_setup_buys_but_morning_grind_still_sells(monkeypatch):
    # 10-12 09:35-11:05: strong bearish trend, BUY beat SELL by Rs 2.5-4k — a morning momentum setup buys
    monkeypatch.setattr(ss, "_has_momentum", _mom(True))
    side, why, _ = ss.choose_buy_or_sell("BEARISH", {"trend_efficiency_ratio": 0.45}, context="trend", now_time=time(9, 50))
    assert side == "BUY", why
    # ...but without momentum the morning is still a sell (no-momentum = +2 sell outweighs +1 early)
    monkeypatch.setattr(ss, "_has_momentum", _mom(False))
    side, why, _ = ss.choose_buy_or_sell("BEARISH", {"trend_efficiency_ratio": 0.45}, context="trend", now_time=time(9, 50))
    assert side == "SELL", why


def _bull_route(monkeypatch, mom_ok, eff, t):
    monkeypatch.setattr(ss, "_CONFLUENCE_ONLY", True)
    monkeypatch.setattr(ss, "_STRUCT_CHOOSER", True)
    monkeypatch.setattr(ss, "_CONF_UP_STRUCT", "BULL_PUT")
    monkeypatch.setattr(ss, "_DEBIT_BLACKOUT", (time(10, 0), time(10, 30)))
    monkeypatch.setattr(ss, "_high_prob_confluence", lambda read, m: ("BULLISH", "HIGH-PROB BULLISH: test"))
    monkeypatch.setattr(ss, "_has_momentum", _mom(mom_ok))
    meta = {"trend_efficiency_ratio": eff}
    return ss.select_strategy(meta, 22560.0, now_time=t), meta


def test_2026_10_13_bullish_momentum_morning_buys_a_call_debit(monkeypatch):
    choice, meta = _bull_route(monkeypatch, True, 1.0, time(10, 40))
    assert choice.structures == ["CALL_DEBIT_SPREAD"] and meta["structure_choice"]["side"] == "BUY"


def test_bullish_grind_sells_a_bull_put_without_buy_fallback(monkeypatch):
    choice, meta = _bull_route(monkeypatch, False, 0.4, time(12, 0))
    assert choice.structures == ["BULL_PUT_CREDIT_SPREAD"] and meta["seller_exit"] is True


def test_bullish_sell_in_blackout_has_no_debit_fallback(monkeypatch):
    choice, _ = _bull_route(monkeypatch, True, 1.0, time(10, 10))
    assert choice.structures == ["BULL_PUT_CREDIT_SPREAD"]
