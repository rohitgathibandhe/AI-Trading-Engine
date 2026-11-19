from datetime import datetime
from types import SimpleNamespace as NS

from market_ai.intraday_defined_risk import execution as ex
from market_ai.intraday_defined_risk.data_models import StrategyType


def _run(monkeypatch, path, env="0.25,0"):
    monkeypatch.setenv("DEBIT_RANGE_LOCK", env)
    monkeypatch.setenv("DEBIT_CHART_STOP_BASIS", "SPOT")
    monkeypatch.setattr(ex, "DEBIT_CHART_STOP", False)
    monkeypatch.setattr(ex, "DEBIT_TRAIL_MODE", "SPOT")
    monkeypatch.setattr(ex, "DEBIT_SPOT_REVERSAL_PTS", 25.0)
    pos = NS(entry_credit_points=-90.0, lot_size=65, lots=3, entry_time=datetime(2026, 11, 18, 12, 40),
             structure=NS(strategy=StrategyType.PUT_DEBIT_SPREAD, width_points=250.0),
             metadata={"entry_day_range": 205.0, "entry_spot_ref": 25198.0})
    for spot, val in path:
        monkeypatch.setattr(ex, "_current_debit_value", lambda *a, _v=val: _v)
        monkeypatch.setattr(ex, "_guard_debit_mark", lambda p, v, *a: v)
        d = ex._evaluate_debit_exit(pos, NS(option_chain=NS(spot=spot), nifty_5m=NS(bars=[])), None,
                                    datetime(2026, 11, 18, 13, 30))
        if d.should_exit:
            return d.reason
    return "HOLD"


def test_locks_after_move_of_quarter_range(monkeypatch):
    # 11-18: +57 pts (> 0.25 x 205) then back through entry
    assert _run(monkeypatch, [(25170, 98), (25141, 101), (25237, 84)]) == "DEBIT_RANGE_LOCK"


def test_not_armed_below_threshold(monkeypatch):
    r = _run(monkeypatch, [(25170, 98), (25160, 99), (25237, 84)])
    assert r != "DEBIT_RANGE_LOCK"


def test_off_by_default(monkeypatch):
    r = _run(monkeypatch, [(25141, 101), (25237, 84)], env="")
    assert r != "DEBIT_RANGE_LOCK"
