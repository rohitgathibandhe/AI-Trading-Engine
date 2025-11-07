import pytest

from market_ai.modules.strategies.monthly_strangle_with_weekly_hedge import _normalize_positions
from tests.mocks.dhan_mock import MockDhanWrapper

pytestmark = pytest.mark.unit


def test_normalize_positions_preserves_side_and_qty(sample_positions):
    rows = _normalize_positions(sample_positions)
    assert len(rows) == 4
    long_leg = next(r for r in rows if r["symbol"].startswith("NIFTY 11"))
    short_leg = next(r for r in rows if r["symbol"].startswith("NIFTY 25") and r["qty"] < 0)

    assert long_leg["side"] in {"BUY", "LONG"}
    assert long_leg["qty"] > 0
    assert short_leg["side"] in {"SELL", "SHORT"}
    assert short_leg["qty"] < 0


def test_mock_dhan_wrapper_injects_ltp(sample_positions, sample_ltp_map):
    wrapper = MockDhanWrapper(sample_positions, sample_ltp_map)
    enriched = wrapper.get_positions_live_with_ltp()
    ltp_values = {row["security_id"]: row["ltp"] for row in enriched}
    assert ltp_values[53023] == pytest.approx(2.15)
    assert ltp_values[52802] == pytest.approx(73.80)
