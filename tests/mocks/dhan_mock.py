"""Comprehensive mock helpers for Dhan integration tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from market_ai.modules.strategies.monthly_strangle_with_weekly_hedge import _normalize_positions

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def sample_open_positions() -> List[dict]:
    path = DATA_DIR / "sample_positions.json"
    return json.loads(path.read_text())


def sample_closed_today() -> List[dict]:
    path = DATA_DIR / "sample_closed_positions.json"
    if path.exists():
        return json.loads(path.read_text())
    return [
        {
            "positionType": "CLOSED",
            "tradingSymbol": "NIFTY 04 NOV 25000 CALL",
            "exchangeSegment": "NSE_FNO",
            "productType": "MARGIN",
            "buyAvg": 5.0,
            "sellAvg": 12.0,
            "costPrice": 12.0,
            "buyQty": 75,
            "sellQty": 75,
            "netQty": 0,
            "realizedProfit": 525.0,
            "unrealizedProfit": 0.0,
            "securityId": "40001",
        }
    ]


class FakeDhanClient:
    """In-memory mock of the broker client for tests."""

    def __init__(
        self,
        *,
        funds: Optional[dict] = None,
        open_positions: Optional[List[dict]] = None,
        closed_today: Optional[List[dict]] = None,
    ):
        self._funds = funds or {
            "available": 250000.0,
            "collateral": 100000.0,
            "utilized": 50000.0,
            "withdrawable": 225000.0,
        }
        self._open_positions = open_positions or sample_open_positions()
        self._closed_today = closed_today or sample_closed_today()
        self.order_log: List[Dict[str, Any]] = []
        self._order_seq = 1

    def get_funds(self) -> dict:
        return dict(self._funds)

    def get_positions_live(self) -> List[dict]:
        return json.loads(json.dumps(self._open_positions))

    def place_order(
        self,
        *,
        side: str,
        exchange_seg: str,
        security_id: int,
        quantity: int,
        product_type: str,
        order_type: str = "MARKET",
        price: Optional[float] = None,
    ) -> Dict[str, Any]:
        order_id = f"FAKE{self._order_seq:04d}"
        self._order_seq += 1
        intent = {
            "orderId": order_id,
            "side": side,
            "exchange_seg": exchange_seg,
            "security_id": security_id,
            "quantity": quantity,
            "product_type": product_type,
            "order_type": order_type,
            "price": price,
        }
        self.order_log.append(intent)
        return {"status": "success", "orderId": order_id}

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        self.order_log.append({"orderId": order_id, "action": "cancel"})
        return {"status": "success", "orderId": order_id}

    @property
    def orders(self) -> List[Dict[str, Any]]:
        return list(self.order_log)


class MockDhanHTTP:
    """Minimal stand-in for DhanHTTP that serves static payloads."""

    def __init__(self, positions: Optional[List[dict]] = None) -> None:
        self.positions = positions or sample_open_positions()

    def get(self, endpoint: str) -> dict:
        if endpoint.endswith("/positions") or endpoint == "/positions":
            return {"data": self.positions}
        raise ValueError(f"Unsupported endpoint in mock: {endpoint}")


class MockDhanWrapper:
    """Tiny wrapper exposing the subset of methods our tests exercise."""

    def __init__(self, positions: List[dict], ltp_map: Optional[Dict[Tuple[str, int], float]] = None) -> None:
        self.positions = positions
        self.ltp_map = ltp_map or {}

    def get_positions_live(self) -> List[dict]:
        rows = _normalize_positions(self.positions)
        for row in rows:
            raw = row.get("_raw") or {}
            if "security_id" not in row and raw.get("securityId") is not None:
                try:
                    row["security_id"] = int(raw["securityId"])
                except Exception:
                    row["security_id"] = raw["securityId"]
            if not row.get("exchange_seg") and raw.get("exchangeSegment"):
                row["exchange_seg"] = raw["exchangeSegment"]
        return rows

    def get_positions_live_with_ltp(self) -> List[dict]:
        rows = self.get_positions_live()
        for row in rows:
            seg = row.get("exchange_seg")
            sid = row.get("security_id")
            if seg and sid and (seg, sid) in self.ltp_map:
                row["ltp"] = self.ltp_map[(seg, sid)]
        return rows


class MockMarketAdapter:
    """Adapter with get_option_chain compatible with strategy backtests."""

    def __init__(self, option_chain_path: Optional[Path] = None) -> None:
        path = option_chain_path or (DATA_DIR / "mock_option_chain.json")
        self.option_chain = json.loads(path.read_text())

    def get_option_chain(self, underlying_id: int, expiry: str, as_of_date: Optional[str] = None) -> dict:
        return self.option_chain


def build_chain_dataframe(path: Optional[Path] = None) -> pd.DataFrame:
    """Convert the mock option chain JSON into the ingestor-friendly shape."""
    option_chain = json.loads((path or (DATA_DIR / "mock_option_chain.json")).read_text())
    rows = []
    for strike, legs in option_chain.items():
        strike_val = float(strike)
        ce = legs.get("ce") or {}
        pe = legs.get("pe") or {}
        rows.append(
            {
                "strike": strike_val,
                "ltp_ce": ce.get("last_price"),
                "ltp_pe": pe.get("last_price"),
                "delta_ce": (ce.get("greeks") or {}).get("delta"),
                "delta_pe": (pe.get("greeks") or {}).get("delta"),
                "raw_call": ce,
                "raw_put": pe,
                "underlying_ltp": ce.get("spot") or pe.get("spot"),
            }
        )
    return pd.DataFrame(rows)
