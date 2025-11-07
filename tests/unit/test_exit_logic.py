import pytest

from market_ai.modules.strategies.monthly_strangle_with_weekly_hedge import _needs_exit

pytestmark = pytest.mark.unit


def _make_row(pnl: float, deployed: float = 1000.0) -> dict:
    qty = deployed / 100.0
    cost = 100.0
    buy_avg = cost + (-pnl / qty)
    return {
        "netQty": qty,
        "costPrice": cost,
        "buyAvg": buy_avg,
        "sellAvg": cost,
    }


def test_needs_exit_returns_reason_codes():
    tp_row = _make_row(pnl=300.0)
    sl_row = _make_row(pnl=-400.0)

    assert _needs_exit(tp_row, sl_pct=0.2, tp_pct=0.2) == "TAKE_PROFIT"
    assert _needs_exit(sl_row, sl_pct=0.2, tp_pct=0.2) == "STOP_LOSS"
