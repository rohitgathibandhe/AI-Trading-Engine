import pytest

from market_ai.modules.strategies.monthly_strangle_with_weekly_hedge import (
    LiveConfig,
    monitor_and_manage_positions,
    _normalize_positions,
)
from tests.mocks.dhan_mock import FakeDhanClient, MockDhanWrapper, sample_open_positions

pytestmark = pytest.mark.integration


def test_fake_dhan_client_order_flow():
    client = FakeDhanClient(open_positions=sample_open_positions())
    funds = client.get_funds()
    assert funds["available"] > 0

    resp = client.place_order(
        side="SELL",
        exchange_seg="NSE_FNO",
        security_id=53023,
        quantity=150,
        product_type="MARGIN",
    )
    order_id = resp["orderId"]
    assert any(o["orderId"] == order_id for o in client.orders)

    client.cancel_order(order_id)
    assert order_id not in client.orders


def test_fake_client_generates_order_when_sl_hit(monkeypatch):
    # run a single strategy tick (monitor_and_manage_positions) and ensure exit intent logs
    client = FakeDhanClient(open_positions=sample_open_positions())
    cfg = LiveConfig(sl_pct=0.01, tp_pct=1.0, warn_only=False, hedge_enable=False)
    # amplify loss on first long leg so _needs_exit triggers
    positions = client.get_positions_live()
    positions[0]["costPrice"] = 2.0  # cheaper than buyAvg so pnl negative
    monitor_and_manage_positions(client, cfg, positions, ingestor=None)
    assert client.order_log, "expected an exit intent when SL breached"


def test_strategy_can_use_mock_wrapper(sample_positions, sample_ltp_map):
    wrapper = MockDhanWrapper(sample_positions, sample_ltp_map)
    rows = wrapper.get_positions_live_with_ltp()
    prices = {row["security_id"]: row["ltp"] for row in rows}
    assert prices[53023] == pytest.approx(2.15)
    assert prices[52802] == pytest.approx(73.80)
