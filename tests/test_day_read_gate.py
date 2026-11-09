from datetime import datetime, timedelta

from market_ai.intraday_defined_risk import strategy_selector as ss
from market_ai.intraday_defined_risk.strategy_selector import StrategyChoice
from market_ai.intraday_defined_risk.trend_day import day_read


def _bars(closes):
    t, out, prev = datetime(2026, 11, 9, 9, 15), [], closes[0]
    for c in closes:
        out.append({"t": t, "o": prev, "h": max(prev, c) + 1, "l": min(prev, c) - 1, "c": c})
        prev, t = c, t + timedelta(minutes=5)
    return out


def _choice(structure):
    return StrategyChoice(family="X", structures=[structure], rationale="")


def test_read_is_symmetric():
    up = day_read(_bars([25000 + 20 * i for i in range(12)]))
    down = day_read(_bars([25000 - 20 * i for i in range(12)]))
    assert up["move_pct"] > 0 > down["move_pct"]
    assert up["efficiency"] == down["efficiency"] and up["side"] == -down["side"]


def test_read_needs_three_bars():
    assert day_read(_bars([25000, 25100]))["efficiency"] == 0.0


def test_gate_off_by_default(monkeypatch):
    monkeypatch.delenv("SEL_DAY_READ_GATE", raising=False)
    assert ss._day_read_gate(_choice("PUT_DEBIT_SPREAD"), {}).structures == ["PUT_DEBIT_SPREAD"]


def test_gate_keeps_with_trend_and_blocks_against(monkeypatch):
    monkeypatch.setenv("SEL_DAY_READ_GATE", "0.5,0.3,0.5")
    md = {"trend_day": {"day_read": {"efficiency": 0.7, "move_pct": -0.5, "side": -0.8, "bars": 12}}}
    assert ss._day_read_gate(_choice("PUT_DEBIT_SPREAD"), md).structures == ["PUT_DEBIT_SPREAD"]
    blocked = ss._day_read_gate(_choice("CALL_DEBIT_SPREAD"), md)
    assert blocked.structures == [] and md["day_read_skip"]["bars"] == 12


def test_gate_blocks_a_flat_read(monkeypatch):
    monkeypatch.setenv("SEL_DAY_READ_GATE", "0.5,0.3,0.5")
    md = {"trend_day": {"day_read": {"efficiency": 0.2, "move_pct": -0.1, "side": -0.2, "bars": 12}}}
    assert ss._day_read_gate(_choice("PUT_DEBIT_SPREAD"), md).structures == []
