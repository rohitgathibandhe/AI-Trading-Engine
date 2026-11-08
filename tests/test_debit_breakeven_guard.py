from datetime import datetime
from types import SimpleNamespace as NS

from market_ai.intraday_defined_risk import execution as ex
from market_ai.intraday_defined_risk.data_models import StrategyType


def _run(monkeypatch, path, arm="1000"):
    """path = [spread_value]; spot held flat so only the P&L path matters. Entry debit 69.25, 130 qty."""
    monkeypatch.setenv("DEBIT_BE_ARM_RS", arm)
    pos = NS(entry_credit_points=-69.25, lot_size=65, lots=2, entry_time=datetime(2026, 1, 13, 10, 0),
             structure=NS(strategy=StrategyType.CALL_DEBIT_SPREAD, width_points=200.0), metadata={"is_fade": False})
    for val in path:
        monkeypatch.setattr(ex, "_current_debit_value", lambda *a, _v=val: _v)
        d = ex._evaluate_debit_exit(pos, NS(option_chain=NS(spot=22600)), None, datetime(2026, 1, 13, 11, 0))
        if d.should_exit:
            return d.reason, d.pnl_rupees
    return "HOLD", None


def test_green_trade_falling_back_to_entry_is_closed_at_breakeven(monkeypatch):
    # +1,560 peak (69.25 -> 81.25), then back through entry: exit at ~0, not a full chart stop
    reason, pnl = _run(monkeypatch, [72.0, 81.25, 74.0, 69.0])
    assert reason == "DEBIT_BREAKEVEN" and -100 < pnl <= 0


def test_never_armed_trade_is_untouched(monkeypatch):
    # peak +390 < arm: the guard does nothing, the normal stops decide
    assert _run(monkeypatch, [72.0, 70.0, 66.0])[0] == "HOLD"


def test_off_by_default(monkeypatch):
    assert _run(monkeypatch, [81.25, 69.0], arm="0")[0] == "HOLD"
