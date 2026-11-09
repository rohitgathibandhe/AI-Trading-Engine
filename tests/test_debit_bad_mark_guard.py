from datetime import datetime
from types import SimpleNamespace as NS

from market_ai.intraday_defined_risk import execution as ex
from market_ai.intraday_defined_risk.data_models import StrategyType


def _pos():
    return NS(entry_credit_points=-90.625, lot_size=65, lots=3, entry_time=datetime(2026, 11, 9, 10, 45),
              structure=NS(strategy=StrategyType.PUT_DEBIT_SPREAD, width_points=250.0, legs=[]), metadata={})


def _pnls(monkeypatch, path):
    """path = [(spot, spread_value)]; returns the pnl of every poll."""
    pos, out = _pos(), []
    for spot, val in path:
        monkeypatch.setattr(ex, "_current_debit_value", lambda *a, _v=val: _v)
        d = ex._evaluate_debit_exit(pos, NS(option_chain=NS(spot=spot, find_quote=lambda *a: None)), None,
                                    datetime(2026, 11, 9, 11, 0))
        out.append(d.pnl_rupees)
    return pos, out


def test_one_poll_phantom_is_ignored(monkeypatch):
    monkeypatch.setenv("DEBIT_BAD_MARK_GUARD", "1")
    pos, pnl = _pnls(monkeypatch, [(24860, 90.0), (24862, 201.96), (24861, 89.0)])
    assert max(pnl) < 1000
    assert pos.metadata["debit_peak_pnl_rupees"] < 1000


def test_move_explained_by_spot_is_kept(monkeypatch):
    monkeypatch.setenv("DEBIT_BAD_MARK_GUARD", "1")
    _, pnl = _pnls(monkeypatch, [(24860, 90.0), (24760, 150.0)])
    assert round(pnl[1]) == round((150.0 - 90.625) * 195)


def test_a_persistent_jump_is_accepted_after_three_polls(monkeypatch):
    monkeypatch.setenv("DEBIT_BAD_MARK_GUARD", "1")
    _, pnl = _pnls(monkeypatch, [(24860, 90.0)] + [(24860, 40.0)] * 4)
    assert round(pnl[3]) == round((90.0 - 90.625) * 195)
    assert round(pnl[4]) == round((40.0 - 90.625) * 195)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("DEBIT_BAD_MARK_GUARD", raising=False)
    _, pnl = _pnls(monkeypatch, [(24860, 90.0), (24862, 201.96)])
    assert round(pnl[1]) == round((201.96 - 90.625) * 195)
