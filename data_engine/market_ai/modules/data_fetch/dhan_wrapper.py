# dhan_wrapper.py
from __future__ import annotations
import time, uuid, logging
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

log = logging.getLogger("DhanWrapper")
if not log.handlers:
    h = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s %(levelname)s [DhanWrapper] %(message)s")
    h.setFormatter(fmt)
    log.addHandler(h)
log.setLevel(logging.INFO)

# --- Your low-level API adapter (import your real dhan_api here) --------------
try:
    from dhan_api import DhanAPI  # your actual adapter
except Exception:
    class DhanAPI:
        def place_order(self, **kwargs) -> Dict[str, Any]:
            return {"status":"ok","order_id":str(uuid.uuid4()),"client_order_id":kwargs.get("client_order_id")}
        def modify_order(self, **kwargs) -> Dict[str, Any]:
            return {"status":"ok","order_id":kwargs.get("order_id")}
        def cancel_order(self, **kwargs) -> Dict[str, Any]:
            return {"status":"ok","order_id":kwargs.get("order_id")}
        def positions(self) -> List[Dict[str, Any]]:
            return []  # return your live positions from broker

# -----------------------------------------------------------------------------

class TransientError(Exception): pass
class FatalError(Exception): pass

class OrderSide:
    BUY = "BUY"
    SELL = "SELL"

class OrderType:
    MARKET = "MARKET"
    LIMIT = "LIMIT"

@dataclass
class Leg:
    symbol: str
    qty: int
    side: str
    order_type: str = OrderType.MARKET
    client_order_id: Optional[str] = None
    price: Optional[float] = None

@dataclass
class ReconciliationReport:
    fixed: bool
    notes: str

class DhanWrapper:
    """
    - Idempotent order placement via client_order_id (dedupe on retry)
    - Exponential backoff retries for transient failures
    - Reconciliation: compare intended legs vs broker positions; repair drift
    """
    def __init__(self, api: Optional[DhanAPI] = None, max_retries: int = 5):
        self.api = api or DhanAPI()
        self.max_retries = max_retries

    # inside DhanWrapper
    def get_lot_size(self, symbol: str | None) -> int:
        """
        Return lot size for the given trading symbol, using a master cache.
        Accepts full option symbol (e.g., NIFTY24OCT24000CE) or underlying key (NIFTY/BANKNIFTY/...).
        """
        if not symbol:
            return 0
        s = symbol.upper()
        # exact hit
        if s in getattr(self, "_lot_cache", {}):
            return int(self._lot_cache[s])
        # try by underlying key
        for key in ("BANKNIFTY","FINNIFTY","MIDCPNIFTY","NIFTY"):
            if key in s or s.startswith(key):
                if key in getattr(self, "_lot_cache", {}):
                    return int(self._lot_cache[key])
        # if master not loaded, indicate failure so agent fallback kicks in
        raise RuntimeError("lot size not found in master cache")

    # ---------- Low-level retry helper ----------
    def _retry(self, fn, label: str, **kwargs) -> Dict[str, Any]:
        backoff = [0.5, 1, 2, 4, 8][: self.max_retries]
        last_err = None
        for i, delay in enumerate(backoff, 1):
            try:
                return fn(**kwargs)
            except TransientError as e:
                last_err = e
                log.warning("%s transient error (attempt %d/%d): %s", label, i, len(backoff), e)
                time.sleep(delay)
            except Exception as e:
                # classify: if your API exposes error codes, detect here
                last_err = e
                log.warning("%s unexpected error (attempt %d/%d): %s", label, i, len(backoff), e)
                time.sleep(delay)
        raise FatalError(f"{label} failed after retries: {last_err}")

    # ---------- Public API ----------
    def place_order_idem(self, leg: Leg) -> Dict[str, Any]:
        if not leg.client_order_id:
            leg.client_order_id = f"{uuid.uuid4().hex}"
        payload = dict(
            symbol=leg.symbol, qty=leg.qty, side=leg.side,
            order_type=leg.order_type, price=leg.price,
            client_order_id=leg.client_order_id
        )
        resp = self._retry(self.api.place_order, "place_order", **payload)
        log.info("Placed %s %s x%d (cid=%s, oid=%s)", leg.side, leg.symbol, leg.qty,
                 leg.client_order_id, resp.get("order_id"))
        return resp

    def place_legs_idem(self, legs: List[Leg]) -> List[Dict[str, Any]]:
        resps = []
        for leg in legs:
            resps.append(self.place_order_idem(leg))
        # Immediate reconcile after batch to ensure shape is correct
        rep = self.reconcile_positions(legs)
        if not rep.fixed:
            log.warning("Post-place reconcile notes: %s", rep.notes)
        return resps

    def modify_order_idem(self, order_id: str, **kwargs) -> Dict[str, Any]:
        resp = self._retry(self.api.modify_order, "modify_order", order_id=order_id, **kwargs)
        log.info("Modified order %s", order_id)
        return resp

    def cancel_order_idem(self, order_id: str) -> Dict[str, Any]:
        resp = self._retry(self.api.cancel_order, "cancel_order", order_id=order_id)
        log.info("Cancelled order %s", order_id)
        return resp

    # ---------- Positions ----------
    def positions(self) -> List[Dict[str, Any]]:
        return self.api.positions()

    # ---------- Reconciliation ----------
    def reconcile_positions(self, intended_legs: List[Leg]) -> ReconciliationReport:
        """
        Compare intended short/long legs vs broker's live positions.
        Minimal logic:
          - Count net qty per symbol for intended vs broker
          - If broker short (negative) when intended long -> repair
          - If missing wings -> buy them
          - If extra shorts -> cover them
        """
        broker = self.positions()
        broker_map = self._aggregate(broker)  # symbol -> net_qty (short negative)
        intend_map = self._aggregate_intended(intended_legs)

        fix_actions: List[str] = []
        for sym, target_qty in intend_map.items():
            live_qty = broker_map.get(sym, 0)
            if live_qty == target_qty:
                continue
            diff = target_qty - live_qty
            side = OrderSide.BUY if diff > 0 else OrderSide.SELL
            qty = abs(diff)
            if qty == 0:
                continue
            fix_actions.append(f"{side} {sym} x{qty}")
            try:
                self.place_order_idem(Leg(symbol=sym, qty=qty, side=side))
            except Exception as e:
                log.error("Reconcile action failed for %s: %s", sym, e)
                return ReconciliationReport(False, f"failed:{sym}:{e}")

        return ReconciliationReport(True, ", ".join(fix_actions) if fix_actions else "in sync")

    # ---------- Helpers ----------
    @staticmethod
    def _aggregate(broker_positions: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Convert broker positions to a symbol -> net_qty map.
        Expect position rows like: {"symbol":"NIFTY24OCT24000CE","qty":50,"side":"SHORT"|"LONG"}
        Adjust per your real payload.
        """
        m: Dict[str, int] = {}
        for p in broker_positions:
            sym = p.get("symbol")
            qty = int(p.get("qty", 0))
            side = p.get("side", "LONG")
            signed = -qty if side.upper().startswith("SHORT") else qty
            m[sym] = m.get(sym, 0) + signed
        return m

    @staticmethod
    def _aggregate_intended(legs: List[Leg]) -> Dict[str, int]:
        m: Dict[str, int] = {}
        for l in legs:
            signed = -l.qty if l.side == OrderSide.SELL else l.qty
            m[l.symbol] = m.get(l.symbol, 0) + signed
        return m
