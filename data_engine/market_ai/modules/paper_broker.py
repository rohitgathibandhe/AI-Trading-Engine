# market_ai/modules/paper_broker.py
from __future__ import annotations

import time
import uuid
import logging
from dataclasses import dataclass
from typing import Dict, Tuple, Any

LOG = logging.getLogger(__name__)
if not LOG.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

@dataclass
class Position:
    opt_type: str   # "CE" / "PE"
    strike: float
    expiry: str
    qty: int        # signed; negative = short
    avg_price: float

class PaperBroker:
    """
    Simple in-memory paper broker:
      - Accepts same method signatures as DhanBroker (extra kwargs ignored)
      - Tracks positions per (opt_type, strike, expiry)
      - Logs trades; returns a fake order payload
    """
    def __init__(self, lot_size: int = 50):
        self.lot_size = int(lot_size)
        # key: (opt_type, strike, expiry)
        self.positions: Dict[Tuple[str, float, str], Position] = {}

    def _order_resp(self, side: str, opt_type: str, strike: float, lots: int, price: float, expiry: str) -> Dict[str, Any]:
        return {
            "orderId": f"PAPER-{uuid.uuid4().hex[:10]}",
            "side": side,
            "opt_type": opt_type,
            "strike": int(round(strike)),
            "lots": int(lots),
            "qty": int(self.lot_size * lots),
            "price": float(price),
            "expiry": expiry,
            "ts": time.time(),
            "status": "FILLED",
        }

    def place(self, side: str, opt_type: str, strike: float, lots: int, price: float, expiry: str, **kwargs) -> Dict[str, Any]:
        """
        Paper place order. Extra kwargs (e.g., underlying_scrip) are accepted and ignored
        so this matches the live broker signature.
        """
        side = side.upper()
        opt_type = opt_type.upper()
        key = (opt_type, float(strike), expiry)

        qty = self.lot_size * int(lots)
        signed = -qty if side == "SELL" else qty  # SELL -> short

        pos = self.positions.get(key)
        if pos is None:
            self.positions[key] = Position(opt_type, float(strike), expiry, signed, float(price))
        else:
            # update avg price using weighted average
            new_qty = pos.qty + signed
            if new_qty == 0:
                # fully flattened
                pos.qty = 0
                pos.avg_price = 0.0
            else:
                # avg only if same direction; if netting, use simple update
                if (pos.qty >= 0 and signed >= 0) or (pos.qty <= 0 and signed <= 0):
                    pos.avg_price = (abs(pos.qty) * pos.avg_price + abs(signed) * float(price)) / abs(new_qty)
                pos.qty = new_qty
            self.positions[key] = pos

        LOG.info("PAPER %s %s %.0f @ %.2f x %d (lots) | pos=%d",
                 side, opt_type, float(strike), float(price), int(lots), self.positions[key].qty)

        return self._order_resp(side, opt_type, strike, lots, price, expiry)

    def close(self, opt_type: str, strike: float, lots: int, price: float, expiry: str, **kwargs) -> Dict[str, Any]:
        """
        Close short = BUY; close long = SELL. For our use we short first, so closing means BUY.
        """
        return self.place("BUY", opt_type, strike, lots, price, expiry, **kwargs)

    # Optional helpers
    def net_positions(self) -> Dict[str, Any]:
        out = []
        for (opt_type, strike, expiry), p in self.positions.items():
            if p.qty != 0:
                out.append({
                    "opt_type": opt_type,
                    "strike": strike,
                    "expiry": expiry,
                    "qty": p.qty,
                    "avg_price": p.avg_price
                })
        return {"positions": out}
