from datetime import datetime
from types import SimpleNamespace as NS

from market_ai.intraday_defined_risk import execution as ex
from market_ai.intraday_defined_risk.data_models import StrategyType


def _pos(strategy=StrategyType.CALL_DEBIT_SPREAD, is_fade=False):
    return NS(entry_credit_points=-69.25, lot_size=65, lots=2, entry_time=datetime(2026, 10, 13, 14, 10),
              structure=NS(strategy=strategy, width_points=200.0), metadata={"is_fade": is_fade})


def _run(monkeypatch, path):
    """path = [(spot, spread_value)]; returns the first exit reason (or HOLD)."""
    pos = _pos()
    for spot, val in path:
        monkeypatch.setattr(ex, "_current_debit_value", lambda *a, _v=val: _v)
        d = ex._evaluate_debit_exit(pos, NS(option_chain=NS(spot=spot)), None, datetime(2026, 10, 13, 14, 30))
        if d.should_exit:
            return d.reason, d.pnl_rupees
    return "HOLD", None


def test_runner_is_not_cut_by_a_premium_wiggle(monkeypatch):
    # the 10-13 shape: +962 peak, small give-back (Rs300 trail fired here), spot keeps making highs
    path = [(22580, 69.25), (22600, 76.6), (22592, 73.0), (22630, 85.0), (22680, 120.0), (22716, 136.5)]
    assert _run(monkeypatch, path)[0] == "HOLD"


def test_fader_is_cut_once_spot_retraces_25(monkeypatch):
    path = [(22580, 69.25), (22630, 85.0), (22604, 78.0)]
    reason, pnl = _run(monkeypatch, path)
    assert reason == "DEBIT_SPOT_REVERSAL" and pnl > 0


def test_fades_keep_their_own_exit_path(monkeypatch):
    pos = _pos(is_fade=True)
    monkeypatch.setattr(ex, "_current_debit_value", lambda *a: 85.0)
    ex._evaluate_debit_exit(pos, NS(option_chain=NS(spot=22630)), None, datetime(2026, 10, 13, 14, 30))
    assert "debit_fav_spot" not in pos.metadata
