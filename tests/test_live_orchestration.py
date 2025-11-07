import sys
import unittest
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG_ROOT = ROOT / "data_engine"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from market_ai.modules.strategies.monthly_strangle_with_weekly_hedge import (  # noqa: E402
    LiveConfig,
    _submit_order,
    monitor_and_manage_positions,
)


class DummyDW:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, int, int]] = []

    def place_order(
        self,
        *,
        side: str,
        exchange_seg: str,
        security_id: int,
        quantity: int,
        product_type: str,
        order_type: str = "MARKET",
    ) -> None:
        self.calls.append((side, exchange_seg, security_id, quantity))


class StubIngestor:
    def __init__(self, monthly_df: pd.DataFrame) -> None:
        self._monthly = monthly_df

    def list_expiries(self) -> pd.Series:
        return pd.Series(["2025-11-25"])

    def fetch(self, expiry: str) -> pd.DataFrame:
        if expiry == "weekly":
            return pd.DataFrame()
        return self._monthly.copy()


def _make_positions() -> list[dict]:
    return [
        {
            "tradingSymbol": "NIFTY-2025-11-25-25000-CE",
            "exchangeSegment": "NSE_FNO",
            "positionType": "SHORT",
            "netQty": -75,
            "costPrice": 10.0,
            "sellAvg": 10.0,
            "securityId": 2001,
            "productType": "MARGIN",
        },
        {
            "tradingSymbol": "NIFTY-2025-11-25-25000-PE",
            "exchangeSegment": "NSE_FNO",
            "positionType": "SHORT",
            "netQty": -75,
            "costPrice": 12.0,
            "sellAvg": 12.0,
            "securityId": 2002,
            "productType": "MARGIN",
        },
    ]


def _make_monthly_df() -> pd.DataFrame:
    rows = [
        {
            "strike": 24800.0,
            "ltp_ce": 4.0,
            "ltp_pe": 18.0,
            "delta_ce": 0.15,
            "delta_pe": -0.30,
            "raw_call": {"securityId": 1901},
            "raw_put": {"securityId": 1902},
        },
        {
            "strike": 25000.0,
            "ltp_ce": 30.0,
            "ltp_pe": 36.0,
            "delta_ce": 0.45,
            "delta_pe": -0.45,
            "raw_call": {"securityId": 2001},
            "raw_put": {"securityId": 2002},
        },
        {
            "strike": 25200.0,
            "ltp_ce": 12.0,
            "ltp_pe": 6.0,
            "delta_ce": 0.20,
            "delta_pe": -0.20,
            "raw_call": {"securityId": 3001},
            "raw_put": {"securityId": 3002},
        },
    ]
    return pd.DataFrame(rows)


pytestmark = pytest.mark.integration


class LiveOrchestrationTests(unittest.TestCase):
    def test_submit_order_respects_warn_only(self):
        dw = DummyDW()
        cfg_warn = LiveConfig(warn_only=True)

        placed = _submit_order(
            dw,
            cfg_warn,
            side="SELL",
            exchange_seg="NSE_FNO",
            security_id=123,
            quantity=50,
            product_type="MARGIN",
            order_type="MARKET",
            price=10.0,
            context={"tag": "unit_test_warn"},
        )

        self.assertFalse(placed)
        self.assertEqual(dw.calls, [])

        cfg_live = LiveConfig(warn_only=False)
        placed_live = _submit_order(
            dw,
            cfg_live,
            side="BUY",
            exchange_seg="NSE_FNO",
            security_id=321,
            quantity=25,
            product_type="MARGIN",
            order_type="MARKET",
            price=5.0,
            context={"tag": "unit_test_live"},
        )
        self.assertTrue(placed_live)
        self.assertEqual(dw.calls[-1], ("BUY", "NSE_FNO", 321, 25))

    def test_monitor_rolls_without_orders_in_warn_only_mode(self):
        monthly_df = _make_monthly_df()
        ingestor = StubIngestor(monthly_df)
        positions = _make_positions()
        dw = DummyDW()
        cfg = LiveConfig(warn_only=True, hedge_enable=False)

        monitor_and_manage_positions(dw, cfg, positions, ingestor)

        self.assertEqual(dw.calls, [])

    def test_monitor_triggers_orders_when_live_mode(self):
        monthly_df = _make_monthly_df()
        ingestor = StubIngestor(monthly_df)
        positions = _make_positions()
        dw = DummyDW()
        cfg = LiveConfig(warn_only=False, hedge_enable=False)

        monitor_and_manage_positions(dw, cfg, positions, ingestor)

        self.assertGreaterEqual(len(dw.calls), 2)
        sides = {call[0] for call in dw.calls}
        self.assertTrue({"BUY", "SELL"}.issubset(sides))


if __name__ == "__main__":
    unittest.main()
