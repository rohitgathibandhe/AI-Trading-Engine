# -*- coding: utf-8 -*-
"""
Dhan wrapper (robust funds + LTP parsing, bulk LTP, and live positions).
No Streamlit dependencies.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from decimal import Decimal
from datetime import datetime, time as dtime
from enum import Enum
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo  # py3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore


# ---------------------------
# Enums expected by your agent
# ---------------------------

class Leg(Enum):
    SINGLE = "SINGLE"
    BULL_CALL = "BULL_CALL"
    BEAR_PUT = "BEAR_PUT"
    CUSTOM = "CUSTOM"


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


__all__ = [
    "DhanWrapper",
    "Leg",
    "OrderSide",
]

# ---- Public API compatibility aliases (used by older modules/tests) ----
# Some parts of the project call `get_ltp_many`, `get_positions_raw`, or `positions()`.
# Provide thin shims so those call sites don't break under refactors.


# ---------------------------
# Import Dhan SDK safely
# ---------------------------

def _import_from_dhan_sdk():
    last_err = None
    candidates = [
        ("dhan_sdk.dhan_http", "DhanHTTP"),
        ("dhan_sdk._market_feed", "MarketFeed"),
        ("dhan_sdk._option_chain", "OptionChain"),
        ("dhan_sdk._portfolio", "Portfolio"),
        # project-local fallbacks
        ("market_ai.dhan_sdk.dhan_http", "DhanHTTP"),
        ("market_ai.dhan_sdk._market_feed", "MarketFeed"),
        ("market_ai.dhan_sdk._option_chain", "OptionChain"),
        ("market_ai.dhan_sdk._portfolio", "Portfolio"),
    ]
    mods: Dict[str, Any] = {}
    for mod_name, symbol in candidates:
        try:
            mod = __import__(mod_name, fromlist=["*"])
            mods[symbol] = getattr(mod, symbol)
        except Exception as e:
            last_err = e
            continue
    needed = {"DhanHTTP", "MarketFeed", "OptionChain", "Portfolio"}
    if not needed.issubset(mods):
        raise ModuleNotFoundError(
            "Could not import dhan_sdk modules. Ensure 'dhan_sdk' is on PYTHONPATH. "
            f"Missing={needed - set(mods)}, last_error={last_err}"
        )
    return mods["DhanHTTP"], mods["MarketFeed"], mods["OptionChain"], mods["Portfolio"]


DhanHTTP, MarketFeed, OptionChain, Portfolio = _import_from_dhan_sdk()


@dataclass(frozen=True)
class DhanEndpoints:
    funds: str = os.getenv("DHAN_FUNDS_PATH", "/fundlimit")
    optionchain: str = os.getenv("DHAN_OPTIONCHAIN_PATH", "/optionchain")
    optionchain_expirylist: str = os.getenv("DHAN_OPTIONCHAIN_EXPIRYLIST_PATH", "/optionchain/expirylist")
    positions: str = os.getenv("DHAN_POSITIONS_PATH", "/positions")
    # positions are accessed via the Portfolio helper


def _ist_now() -> datetime:
    try:
        tz = ZoneInfo("Asia/Kolkata") if ZoneInfo else None
    except Exception:
        tz = None
    return datetime.now(tz) if tz else datetime.now()


def _is_india_market_open(now: Optional[datetime] = None) -> bool:
    if now is None:
        now = _ist_now()
    if now.weekday() >= 5:  # Sat/Sun
        return False
    start = dtime(9, 15)
    end = dtime(15, 30)
    return start <= now.time() <= end


class _DefaultLogger:
    def info(self, msg: str) -> None:
        print(msg)
    def warning(self, msg: str) -> None:
        print(msg)
    def exception(self, msg: str) -> None:
        print(msg)


class DhanWrapper:
    """
    Thin wrapper around dhan_sdk:
      - Robust funds parsing
      - Robust single LTP + bulk LTP
      - Live positions with multiple fallbacks
      - Simple background LTP poller
    """

    # --- Compatibility shims -------------------------------------------------
    def get_ltp_many(self, exchange_seg: str, security_ids: list[int]) -> Dict[tuple[str, int], Optional[float]]:
        """
        Backwards-compatible alias for `get_ltp_bulk`.
        Returns a dict {(seg, id): ltp or None}.
        """
        pairs: list[tuple[str, int]] = []
        for sid in security_ids or []:
            try:
                pairs.append((exchange_seg, int(sid)))
            except Exception:
                continue
        return self.get_ltp_bulk(pairs)

    def get_positions_raw(self) -> List[Dict[str, Any]]:
        """
        Return raw list from /v2/positions without normalization,
        matching older call sites that expect the untouched payload.
        """
        resp = self.http.get(self.endpoints.positions)
        if isinstance(resp, dict):
            data = resp.get("data") or resp.get("positions") or resp.get("netPositions")
            return data if isinstance(data, list) else []
        return resp if isinstance(resp, list) else []

    def positions(self) -> List[Dict[str, Any]]:
        """
        Alias for normalized live positions.
        """
        return self.get_positions_live()

    def __init__(
        self,
        dhan_client_id: Optional[str] = None,
        access_token: Optional[str] = None,
        logger: Optional[Any] = None,
        http_pool: Optional[dict] = None,
        disable_ssl: bool = False,
    ):
        self.client_id: str = (dhan_client_id or os.getenv("DHAN_CLIENT_ID", "")).strip()
        self.access_token: str = (access_token or os.getenv("DHAN_ACCESS_TOKEN", "")).strip()
        if not self.client_id or not self.access_token:
            raise TypeError(
                "DhanWrapper requires client_id and access_token. "
                "Pass them to the constructor or set DHAN_CLIENT_ID/DHAN_ACCESS_TOKEN."
            )

        self.log: Any = logger or _DefaultLogger()
        self.http: Any = DhanHTTP(
            client_id=self.client_id,
            access_token=self.access_token,
            disable_ssl=disable_ssl,
            pool=http_pool,
        )
        self.endpoints: DhanEndpoints = DhanEndpoints()

        # poller state
        self._stop: threading.Event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # -------- tiny helpers (used across LTP & positions) --------

    @staticmethod
    def _as_float(x: Any) -> Optional[float]:
        """Safe float() with None on failure; name used by get_ltp_once."""
        try:
            return float(x)
        except Exception:
            return None

    # keep older name as alias, some call sites may still reference it
    _to_float = _as_float

    @staticmethod
    def _pick(d: Dict[str, Any], *keys: str) -> Any:
        """Return the first non-empty value for any of the provided keys in dict d."""
        for k in keys:
            if k in d and d[k] not in (None, "", [], {}):
                return d[k]
        return None

    # -------- Funds --------

    def get_funds(self) -> Dict[str, Optional[float]]:
        """
        Returns a normalized dict with keys:
        - available, collateral, utilized, withdrawable
        """
        resp = self.http.get(self.endpoints.funds)

        payload: Any = resp
        if isinstance(resp, dict) and "data" in resp and resp.get("status") == "success":
            payload = resp["data"]

        out: Dict[str, Optional[float]] = {"available": None, "collateral": None, "utilized": None, "withdrawable": None}

        def _as_number(val: Any) -> Optional[float]:
            """Recursively parse a value as a float, handling nested dicts/lists and cleaning strings."""
            if val in (None, "", [], {}):
                return None

            # unwrap common nested structures (e.g. {'value': '12345'})
            if isinstance(val, dict):
                for key in ("value", "amount", "balance"):
                    if key in val:
                        return _as_number(val[key])
                # fall back to first numeric-looking value
                for v in val.values():
                    parsed = _as_number(v)
                    if parsed is not None:
                        return parsed
                return None

            if isinstance(val, (list, tuple)):
                for item in val:
                    parsed = _as_number(item)
                    if parsed is not None:
                        return parsed
                return None

            if isinstance(val, Decimal):
                return float(val)

            if isinstance(val, str):
                cleaned = re.sub(r"[^\d\.\-]", "", val).strip()
                if cleaned in ("", "-", ".", "-.", ".-", "--"):
                    return None
                val = cleaned

            return DhanWrapper._as_float(val)

        if isinstance(payload, dict):
            out["available"] = _as_number(self._pick(
                payload,
                "availableMargin",
                "available_margin",
                "availableBalance",
                "availabelBalance",
                "availableCash",
                "cashBalance",
                "availableCashBalance",
                "available",
            ))
            out["collateral"] = _as_number(self._pick(
                payload,
                "collateralMargin",
                "collateralAmount",
                "collateralValue",
                "collateral",
            ))
            out["utilized"] = _as_number(self._pick(
                payload,
                "utilised",
                "utilized",
                "utilizedMargin",
                "utilizedAmount",
                "usedMargin",
                "used_amount",
            ))
            out["withdrawable"] = _as_number(self._pick(
                payload,
                "withdrawableBalance",
                "withdrawableCashBalance",
                "withdrawableAmount",
                "withdrawableCash",
                "withdrawable",
            ))

        self.log.info(f"[Funds] parsed => {out}")
        return out

    # -------- Option Chain --------

    def get_expiry_list(self, underlying_security_id: int, underlying_seg: str) -> Dict[str, Any]:
        ctx = type("Ctx", (object,), {"get_dhan_http": lambda _: self.http})()
        oc = OptionChain(ctx)
        return oc.expiry_list(underlying_security_id, underlying_seg)

    def get_option_chain(self, underlying_security_id: int, underlying_seg: str, expiry: str) -> Dict[str, Any]:
        ctx = type("Ctx", (object,), {"get_dhan_http": lambda _: self.http})()
        oc = OptionChain(ctx)
        return oc.option_chain(underlying_security_id, underlying_seg, expiry)

    # -------- LTP (single + bulk) --------

    @staticmethod
    def _unwrap_to_segment(obj: Any, exchange_seg: str) -> Any:
        """
        Dhan sometimes nests {'data': {..., 'data': {...}}}.
        Keep unwrapping 'data' while it looks like a wrapper and we haven't
        reached the structure that contains the exchange segment key or a list.
        """
        seen = set()
        cur = obj
        while isinstance(cur, dict):
            if exchange_seg in cur or isinstance(cur.get(exchange_seg), (dict, list)):
                return cur
            oid = id(cur)
            if oid in seen:
                return cur
            seen.add(oid)
            nxt = cur.get("data")
            if isinstance(nxt, dict):
                cur = nxt
                continue
            if isinstance(nxt, list):
                return nxt
            break
        return cur

    def _extract_ltp_from_response(self, resp: Any, exchange_seg: str, security_id: int) -> Optional[float]:
        """
        Recursively search for the LTP value in a dhan_sdk LTP/quote response.
        """
        if isinstance(resp, dict) and resp.get("status") == "failure" and not resp.get("data"):
            return None

        def iter_nodes(obj: Any, visited: set[int]) -> Iterable[Dict[str, Any]]:
            if isinstance(obj, dict):
                oid = id(obj)
                if oid in visited:
                    return
                visited.add(oid)
                yield obj
                for v in obj.values():
                    yield from iter_nodes(v, visited)
            elif isinstance(obj, list):
                for item in obj:
                    yield from iter_nodes(item, visited)

        def match_ltp(node: Dict[str, Any]) -> Optional[float]:
            seg = node.get("exchangeSegment") or node.get("exchange_segment") or node.get("segment") or node.get("segmentName")
            sid = node.get("securityId") or node.get("security_id") or node.get("id") or node.get("instrumentToken")
            if seg and str(seg).upper() != str(exchange_seg).upper():
                return None
            if sid is not None and str(sid) != str(security_id):
                return None
            for key in ("last_price", "LTP", "ltp", "close", "closingPrice", "price", "lastPrice"):
                if key in node:
                    val = DhanWrapper._as_float(node[key])
                    if val is not None:
                        return val
            return None

        data = resp
        if isinstance(resp, dict):
            data = resp.get("data", resp)
        data = self._unwrap_to_segment(data, exchange_seg)

        if isinstance(data, dict) and exchange_seg in data:
            direct = data[exchange_seg]
            result = self._extract_ltp_from_response(direct, exchange_seg, security_id)
            if result is not None:
                return result
        if isinstance(data, dict):
            maybe = data.get(str(security_id)) or data.get(security_id)
            if isinstance(maybe, dict):
                for key in ("last_price", "LTP", "ltp"):
                    if key in maybe:
                        val = DhanWrapper._as_float(maybe[key])
                        if val is not None:
                            return val

        visited: set[int] = set()
        for node in iter_nodes(data, visited):
            if not isinstance(node, dict):
                continue
            ltp = match_ltp(node)
            if ltp is not None:
                return ltp
        return None

    def get_ltp_once(self, exchange_seg: str, security_id: int) -> Optional[float]:
        """
        Robustly parse different shapes Dhan may return for /marketfeed/ltp with fallbacks.
        """
        ctx = type("Ctx", (object,), {"get_dhan_http": lambda _: self.http})()
        mf = MarketFeed(ctx)
        payload = {exchange_seg: [security_id]}
        responses: list[Tuple[str, Any]] = []

        for source in ("ticker", "quote"):
            try:
                if source == "ticker":
                    resp = mf.ticker_data(payload)
                else:
                    resp = mf.quote_data(payload)
                responses.append((source, resp))
                ltp = self._extract_ltp_from_response(resp, exchange_seg, security_id)
                if ltp is not None:
                    return ltp
            except Exception as e:
                self.log.exception(f"[LTP] {source} fetch error: {e}")

        if responses:
            if all(isinstance(r, dict) and r.get("status") == "failure" for _, r in responses):
                return None
            last_source, last_resp = responses[-1]
            self.log.warning(f"[LTP] Could not parse LTP from {last_source} response: {last_resp}")
        return None

    def get_ltp_bulk(self, pairs: list[tuple[str, int]]) -> Dict[tuple[str, int], Optional[float]]:
        """
        Fetch LTP for many (exchange_seg, security_id) pairs at once.
        Returns a dict {(seg, id): ltp or None}.
        Works with both dict-of-dicts and list-of-dicts marketfeed shapes.
        """
        if not pairs:
            return {}

        # Build payload per segment
        payload: Dict[str, list[int]] = {}
        for seg, sid in pairs:
            if not seg or sid is None:
                continue
            try:
                sid = int(sid)
            except Exception:
                continue
            payload.setdefault(seg, []).append(sid)

        if not payload:
            return {}

        ctx = type("Ctx", (object,), {"get_dhan_http": lambda _: self.http})()
        mf = MarketFeed(ctx)

        out: Dict[tuple[str, int], Optional[float]] = {}
        try:
            resp = mf.ticker_data(payload)
            if isinstance(resp, dict) and resp.get("status") == "failure":
                msg = resp.get("remarks", {}).get("error_message")
                if msg:
                    self.log.debug(f"[LTP-bulk] status=failure remarks={msg}")
                return out
            data = resp.get("data", resp) if isinstance(resp, dict) else resp

            if isinstance(data, dict):
                for seg, node in data.items():
                    # dict-of-dicts: {'NSE_FNO': {'52802': {'last_price': ...}, ...}}
                    if isinstance(node, dict):
                        for k, v in node.items():
                            if not isinstance(v, dict):
                                continue
                            ltp = (
                                self._as_float(v.get("last_price"))
                                or self._as_float(v.get("LTP"))
                                or self._as_float(v.get("ltp"))
                            )
                            try:
                                out[(seg, int(k))] = ltp
                            except Exception:
                                pass
                    # list-of-dicts: {'NSE_FNO': [{'securityId': 52802, 'last_price': ...}, ...]}
                    elif isinstance(node, list):
                        for item in node:
                            if not isinstance(item, dict):
                                continue
                            sid = item.get("securityId") or item.get("security_id") or item.get("id")
                            ltp = (
                                self._as_float(item.get("last_price"))
                                or self._as_float(item.get("LTP"))
                                or self._as_float(item.get("ltp"))
                            )
                            try:
                                out[(seg, int(sid))] = ltp
                            except Exception:
                                pass
        except Exception as e:
            self.log.exception(f"[LTP-bulk] error: {e}")

        return out
    # -------- Live Positions (via official REST API) --------
    def get_positions_live(self) -> list[Dict[str, Any]]:
        """
        Call /v2/positions and normalize rows for the UI with:
          symbol, product, qty, avg_price, ltp, pnl, exchange_seg, security_id, side
        PnL rule:
          - if realizedProfit != 0 use realizedProfit
          - else BUY (qty>0):  (buyAvg - costPrice) * qty
               SELL (qty<0):  (costPrice - sellAvg) * abs(qty)
        """
        try:
            resp = self.http.get(self.endpoints.positions)
        except Exception as e:
            raise RuntimeError(f"Failed to fetch positions: {e}")

        payload: Any = resp
        if isinstance(resp, dict):
            payload = resp.get("data", resp.get("positions", resp.get("netPositions", [])))

        if not isinstance(payload, list):
            payload = []

        rows: list[Dict[str, Any]] = []

        for row in payload:
            if not isinstance(row, dict):
                continue

            symbol = self._pick(row, "tradingSymbol", "tradingsymbol", "symbol", "securityName", "displayName") or ""
            product = self._pick(row, "productType", "product", "product_type") or ""
            exchange_seg = self._pick(row, "exchangeSegment", "exchange_seg", "segment") or ""

            # ids & numbers
            security_id = self._pick(row, "securityId", "security_id", "id")
            try:
                security_id = int(security_id) if security_id is not None else None
            except Exception:
                security_id = None

            net_qty_val = self._as_float(self._pick(row, "netQty", "netQuantity", "qty", "netqty")) or 0.0
            qty_int = int(net_qty_val) if net_qty_val == int(net_qty_val) else int(net_qty_val)

            buy_qty = self._as_float(self._pick(row, "buyQty", "dayBuyQty", "carryForwardBuyQty"))
            sell_qty = self._as_float(self._pick(row, "sellQty", "daySellQty", "carryForwardSellQty"))

            cost_price = self._as_float(self._pick(row, "costPrice", "price"))
            buy_avg    = self._as_float(self._pick(row, "buyAvg", "buyAveragePrice"))
            sell_avg   = self._as_float(self._pick(row, "sellAvg", "sellAveragePrice"))
            realized   = self._as_float(self._pick(row, "realizedProfit", "realisedProfit"))
            unrealized = self._as_float(self._pick(row, "unrealizedProfit", "unrealisedProfit", "mtm"))
            ltp_value  = self._as_float(self._pick(row, "ltp", "lastPrice", "last_price", "closePrice"))

            if qty_int > 0:
                side = "BUY"
            elif qty_int < 0:
                side = "SELL"
            else:
                side = "FLAT"

            if qty_int != 0 and unrealized is not None:
                pnl = unrealized
            elif qty_int == 0 and realized is not None:
                pnl = realized
            else:
                pnl = realized if realized is not None else unrealized

            rows.append(
                {
                    "symbol": symbol,
                    "product": product,
                    "qty": qty_int,
                    "net_qty": qty_int,
                    "buy_qty": int(buy_qty) if buy_qty is not None else None,
                    "sell_qty": int(sell_qty) if sell_qty is not None else None,
                    "avg_price": cost_price,
                    "cost_price": cost_price,
                    "ltp": ltp_value,
                    "pnl": pnl,
                    "exchange_seg": exchange_seg,
                    "security_id": security_id,
                    "side": side,
                    "position_type": str(self._pick(row, "positionType", "side")).upper() if self._pick(row, "positionType", "side") else side,
                    "buy_avg": buy_avg,
                    "sell_avg": sell_avg,
                    "realized_profit": realized,
                    "unrealized_profit": unrealized,
                    "_raw": row,
                }
            )

        self.log.info(f"[Positions] normalized {len(rows)} rows (with side & pnl rule)")
        return rows

    def get_positions_live_with_ltp(self) -> list[Dict[str, Any]]:
        """
        Same as get_positions_live, but merges LTP via /marketfeed/ltp
        and fills 'ltp' and 'pnl' (pnl uses Dhan 'unrealizedProfit' when available).
        """
        rows = self.get_positions_live()

        # Build (seg,id) pairs for bulk LTP
        pairs: list[tuple[str, int]] = []
        for r in rows:
            seg = r.get("exchange_seg")
            sid = r.get("security_id")
            if isinstance(seg, str) and isinstance(sid, int):
                pairs.append((seg, sid))

        ltps = self.get_ltp_bulk(pairs) if pairs else {}

        for r in rows:
            seg = r.get("exchange_seg")
            sid = r.get("security_id")
            ltp_live = ltps.get((seg, sid)) if seg and sid else None

            raw = r.get("_raw", {})
            position_type = str(raw.get("positionType") or r.get("position_type") or r.get("side") or "").upper()
            net_qty = r.get("qty") or 0
            cost_price = self._as_float(raw.get("costPrice")) or self._as_float(r.get("avg_price"))
            buy_avg = self._as_float(raw.get("buyAvg"))
            sell_avg = self._as_float(raw.get("sellAvg"))
            buy_qty_f = self._as_float(raw.get("buyQty"))
            sell_qty_f = self._as_float(raw.get("sellQty"))
            buy_qty = int(buy_qty_f) if buy_qty_f is not None else None
            sell_qty = int(sell_qty_f) if sell_qty_f is not None else None
            unrealized = self._as_float(raw.get("unrealizedProfit"))
            realized = self._as_float(raw.get("realizedProfit"))

            if position_type == "LONG" or (isinstance(net_qty, int) and net_qty > 0):
                qty_disp = buy_qty if buy_qty is not None else net_qty
                ltp_val = buy_avg if buy_avg is not None else ltp_live if ltp_live is not None else cost_price
                ltp_val = self._as_float(ltp_val) or self._as_float(cost_price)
                pnl_val = unrealized if unrealized is not None else r.get("pnl") or 0.0
                r["qty"] = int(qty_disp or 0)
                r["ltp"] = ltp_val
                r["pnl"] = pnl_val
                r["avg_price"] = cost_price
            elif position_type == "SHORT" or (isinstance(net_qty, int) and net_qty < 0):
                qty_disp = -(sell_qty if sell_qty is not None else abs(net_qty))
                ltp_val = sell_avg if sell_avg is not None else ltp_live if ltp_live is not None else cost_price
                ltp_val = self._as_float(ltp_val) or self._as_float(cost_price)
                pnl_val = unrealized if unrealized is not None else r.get("pnl") or 0.0
                r["qty"] = int(qty_disp or 0)
                r["ltp"] = ltp_val
                r["pnl"] = pnl_val
                r["avg_price"] = cost_price
            else:
                qty_disp = sell_qty or buy_qty or net_qty
                ltp_val = ltp_live if ltp_live is not None else cost_price
                ltp_val = self._as_float(ltp_val) or self._as_float(cost_price)
                pnl_val = realized if realized is not None else r.get("pnl") or 0.0
                r["qty"] = int(qty_disp or 0)
                r["ltp"] = ltp_val
                r["pnl"] = pnl_val
                r["avg_price"] = cost_price

        self.log.info(f"[Positions] normalized {len(rows)} rows (with LTP merge)")
        return rows

    # -------- Background single-LTP poller (for NIFTY tile) --------

    def start_ltp_poller(
        self,
        exchange_seg: str,
        security_id: int,
        interval_sec: float,
        on_update: Callable[[float, datetime], None],
        poll_when_closed: bool = False,
    ) -> None:
        self.stop_ltp_poller()
        self._stop.clear()

        def _loop():
            while not self._stop.is_set():
                try:
                    if poll_when_closed or _is_india_market_open():
                        ltp = self.get_ltp_once(exchange_seg, security_id)
                        if ltp is not None:
                            on_update(ltp, _ist_now())
                except Exception as e:
                    self.log.exception(f"[LTP] poller: {e}")
                self._stop.wait(max(0.25, float(interval_sec)))

        self._thread = threading.Thread(target=_loop, daemon=True, name="DhanLTPPoller")
        self._thread.start()

    def stop_ltp_poller(self) -> None:
        """Stop the background LTP poller thread if running."""
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
