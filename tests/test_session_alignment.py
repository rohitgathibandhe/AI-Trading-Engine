from datetime import time

from market_ai.intraday_defined_risk import strategy_selector as ss


def _sel(monkeypatch, direction, orb, pvv):
    monkeypatch.setattr(ss, "_CONFLUENCE_ONLY", True)
    monkeypatch.setattr(ss, "_SESSION_ALIGN", True)
    monkeypatch.setattr(ss, "_RANGE_FADE", False)
    monkeypatch.setattr(ss, "_high_prob_confluence", lambda read, m: (direction, f"HIGH-PROB {direction}: test"))
    monkeypatch.setattr(ss, "_has_momentum", lambda m, d: (True, "momentum OK"))
    meta = {"opening_range_break_state": orb, "price_vs_vwap": pvv, "trend_efficiency_ratio": 0.89}
    return ss.select_strategy(meta, 22496.0, now_time=time(10, 45))


def test_2026_10_13_bearish_pullback_in_an_up_session_stands_aside(monkeypatch):
    choice = _sel(monkeypatch, "BEARISH", "UP", 21.0)
    assert choice.structures == [] and "pullback AGAINST the session" in choice.rationale


def test_bearish_read_on_a_down_session_still_trades(monkeypatch):
    assert _sel(monkeypatch, "BEARISH", "DOWN", -40.0).structures


def test_bullish_pullback_in_a_down_session_stands_aside(monkeypatch):
    assert _sel(monkeypatch, "BULLISH", "DOWN", -15.0).structures == []


def test_bearish_below_vwap_after_up_break_is_allowed(monkeypatch):
    # the up-break failed and price lost VWAP -> a genuine turn, not a pullback
    assert _sel(monkeypatch, "BEARISH", "UP", -12.0).structures
