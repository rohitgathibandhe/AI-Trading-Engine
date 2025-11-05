# market_ai/modules/dhan_broker.py
from __future__ import annotations

import os, time, uuid, logging, requests
from typing import Optional, Dict, Any

LOG = logging.getLogger(__name__)
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

class DhanBroker:
    BASE = os.environ.get("DHAN_API_BASE", "https://api.dhan.co").rstrip("/")

    def __init__(self, client_id: Optional[str], access_token: Optional[str], lot_size: int = 50, timeout_s: int = 10):
        self.client_id  = client_id  or os.environ.get("DHAN_CLIENT_ID")
        self.token      = access_token or os.environ.get("DHAN_ACCESS_TOKEN")
        if not self.client_id or not self.token:
            raise RuntimeError("DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN missing")
        self.lot_size   = int(lot_size)
        self.timeout_s  = int(timeout_s)

    def _headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "access-token": self.token,
            "client-id": self.client_id,
        }

    def _place_order(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.BASE}/v2/orders"
        r = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout_s)
        try:
            js = r.json()
        except Exception:
            LOG.error("Non-JSON from orders: %s", r.text[:400]); r.raise_for_status()
            raise
        if r.status_code >= 400 or (isinstance(js, dict) and js.get("status") == "failed"):
            LOG.error("Order failed %s: %s", r.status_code, str(js)[:400])
            raise requests.HTTPError(f"{r.status_code}: {js}")
        return js

    def place(self, side: str, opt_type: str, strike: float, lots: int, price: float, expiry: str,
              underlying_scrip: int = 13, product_type: str = "MARGIN", validity: str = "DAY") -> Dict[str, Any]:
        qty = self.lot_size * int(lots)
        client_order_id = f"ALG-{int(time.time())}-{uuid.uuid4().hex[:8]}"
        payload = {
            "transactionType": "SELL" if side.upper() == "SELL" else "BUY",
            "exchangeSegment": "NSE_FNO",
            "productType": product_type,
            "orderType": "LIMIT",
            "validity": validity,
            "securityId": int(underlying_scrip),
            "instrument": "OPTIDX",
            "expiry": expiry,
            "optionType": opt_type.upper(),
            "strikePrice": float(strike),
            "quantity": int(qty),
            "price": float(price),
            "afterMarketOrder": False,
            "amoTime": None,
            "clientOrderId": client_order_id,
        }
        js = self._place_order(payload)
        LOG.info("LIVE ORDER placed: %s %s %d x%d @ %.2f -> %s",
                 side.upper(), opt_type.upper(), int(strike), qty, price, js.get("orderId"))
        return js

    def close(self, opt_type: str, strike: float, lots: int, price: float, expiry: str, **kw) -> Dict[str, Any]:
        return self.place("BUY", opt_type, strike, lots, price, expiry, **kw)
