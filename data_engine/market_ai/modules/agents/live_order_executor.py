from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except Exception:
        return None


def _extract_order_id(resp: Any) -> Optional[str]:
    if not isinstance(resp, dict):
        return None
    for key in ("orderId", "order_id", "id"):
        val = resp.get(key)
        if val not in (None, ""):
            return str(val)
    data = resp.get("data")
    if isinstance(data, dict):
        for key in ("orderId", "order_id", "id"):
            val = data.get(key)
            if val not in (None, ""):
                return str(val)
    return None


def _order_filled(status_resp: Dict[str, Any]) -> bool:
    st = str(status_resp.get("orderStatus") or status_resp.get("status") or "").lower()
    if st in {"completed", "executed", "filled", "traded", "complete", "success"}:
        return True
    filled = _coerce_int(status_resp.get("filledQuantity") or status_resp.get("filledQty"))
    remaining = _coerce_int(status_resp.get("pendingQuantity") or status_resp.get("remainingQuantity"))
    if filled is not None and remaining is not None and filled > 0 and remaining == 0:
        return True
    return False


@dataclass
class LiveOrderExecutorConfig:
    enabled: bool = True
    fill_wait_sec: int = 20
    fill_poll_sec: float = 2.0
    max_retries: int = 2
    settle_delay_sec: float = 1.0
    verify_via_positions: bool = True

    @classmethod
    def from_settings(cls, settings: Dict[str, Any]) -> "LiveOrderExecutorConfig":
        return cls(
            enabled=bool(settings.get("live_order_verify_enabled", True)),
            fill_wait_sec=max(1, int(settings.get("live_order_fill_wait_sec", 20))),
            fill_poll_sec=max(0.2, float(settings.get("live_order_fill_poll_sec", 2.0))),
            max_retries=max(0, int(settings.get("live_order_retry_count", 2))),
            settle_delay_sec=max(0.0, float(settings.get("live_order_settle_delay_sec", 1.0))),
            verify_via_positions=bool(settings.get("live_order_verify_positions", True)),
        )


class LiveOrderExecutor:
    """
    Places live orders and verifies net position movement to reduce silent partial-fill risk.
    """

    def __init__(self, *, config: LiveOrderExecutorConfig, logger: Any = None) -> None:
        self.config = config
        self.log = logger

    def _log(self, level: str, msg: str, *args: Any) -> None:
        if self.log and hasattr(self.log, level):
            getattr(self.log, level)(msg, *args)

    def _get_net_qty(self, dw: Any, security_id: int) -> Optional[int]:
        total = 0
        matched = False
        rows = dw.get_positions_raw()
        for row in rows or []:
            try:
                sid = _coerce_int(row.get("securityId") or row.get("security_id"))
                if sid != int(security_id):
                    continue
                total += int(float(row.get("netQty") or row.get("netqty") or 0))
                matched = True
            except Exception:
                continue
        return total if matched else 0

    def _side_sign(self, side: str) -> int:
        return 1 if str(side or "").upper().startswith("B") else -1

    def _status_filled_qty(self, status_resp: Dict[str, Any]) -> Optional[int]:
        return _coerce_int(status_resp.get("filledQuantity") or status_resp.get("filledQty"))

    def place_and_verify(
        self,
        *,
        dw: Any,
        side: str,
        exchange_seg: str,
        security_id: int,
        quantity: int,
        product_type: str = "MARGIN",
        order_type: str = "MARKET",
        price: Optional[float] = None,
        client_order_prefix: str = "BKM",
    ) -> Dict[str, Any]:
        qty = max(0, int(quantity or 0))
        result: Dict[str, Any] = {
            "ok": False,
            "verified": False,
            "requested_qty": qty,
            "filled_qty_est": 0,
            "remaining_qty": qty,
            "attempts": 0,
            "order_ids": [],
            "pre_net_qty": None,
            "target_net_qty": None,
            "final_net_qty": None,
            "error": None,
        }
        if qty <= 0:
            result.update({"ok": True, "verified": True, "remaining_qty": 0})
            return result
        if not self.config.enabled:
            resp = dw.place_order(
                side=side,
                exchange_seg=exchange_seg,
                security_id=security_id,
                quantity=qty,
                product_type=product_type,
                order_type=order_type,
                price=price,
            )
            oid = _extract_order_id(resp)
            if oid:
                result["order_ids"] = [oid]
            result.update({"ok": True, "verified": False, "remaining_qty": 0})
            return result

        signed_total = self._side_sign(side) * qty
        pre_net: Optional[int] = None
        if self.config.verify_via_positions and hasattr(dw, "get_positions_raw"):
            try:
                pre_net = self._get_net_qty(dw, security_id)
            except Exception as exc:
                self._log("warning", "[ExecGuard] pre-net fetch failed sec_id=%s err=%s", security_id, exc)
        target_net = (pre_net + signed_total) if pre_net is not None else None
        result["pre_net_qty"] = pre_net
        result["target_net_qty"] = target_net

        remaining = qty
        current_net = pre_net
        last_status: Optional[Dict[str, Any]] = None
        last_error: Optional[str] = None
        for attempt in range(self.config.max_retries + 1):
            if remaining <= 0:
                break
            result["attempts"] = attempt + 1
            client_id = f"{client_order_prefix}-{uuid.uuid4().hex[:10]}"
            order_id: Optional[str] = None
            try:
                resp = dw.place_order(
                    side=str(side).upper(),
                    exchange_seg=exchange_seg,
                    security_id=int(security_id),
                    quantity=int(remaining),
                    product_type=product_type,
                    order_type=order_type,
                    price=price,
                    client_order_id=client_id,
                )
                order_id = _extract_order_id(resp)
                if order_id:
                    result["order_ids"].append(order_id)
            except Exception as exc:
                last_error = f"place_order_failed:{exc}"
                self._log(
                    "warning",
                    "[ExecGuard] place failed attempt=%s sec_id=%s qty=%s err=%s",
                    attempt + 1,
                    security_id,
                    remaining,
                    exc,
                )
                if attempt < self.config.max_retries:
                    time.sleep(self.config.fill_poll_sec)
                    continue
                break

            if self.config.settle_delay_sec > 0:
                time.sleep(self.config.settle_delay_sec)

            deadline = time.time() + float(self.config.fill_wait_sec)
            filled_by_status = False
            while True:
                if order_id and hasattr(dw, "order_status"):
                    try:
                        status_resp = dw.order_status(order_id)
                        if isinstance(status_resp, dict):
                            last_status = status_resp
                            if _order_filled(status_resp):
                                filled_by_status = True
                    except Exception as exc:
                        last_error = f"order_status_failed:{exc}"

                if target_net is not None and self.config.verify_via_positions and hasattr(dw, "get_positions_raw"):
                    try:
                        current_net = self._get_net_qty(dw, security_id)
                        result["final_net_qty"] = current_net
                        if current_net == target_net:
                            result["filled_qty_est"] = qty
                            result["remaining_qty"] = 0
                            result["ok"] = True
                            result["verified"] = True
                            return result
                    except Exception as exc:
                        last_error = f"positions_fetch_failed:{exc}"
                elif filled_by_status:
                    result["filled_qty_est"] = qty
                    result["remaining_qty"] = 0
                    result["ok"] = True
                    result["verified"] = True
                    return result

                if time.time() >= deadline:
                    break
                time.sleep(self.config.fill_poll_sec)

            # attempt deadline reached: compute remaining qty using current net delta if possible
            if target_net is not None and current_net is not None:
                diff = abs(int(target_net) - int(current_net))
                remaining = max(0, diff)
                result["remaining_qty"] = remaining
                result["filled_qty_est"] = max(0, qty - remaining)
            else:
                filled_qty = self._status_filled_qty(last_status or {})
                if filled_qty is not None:
                    remaining = max(0, remaining - filled_qty)
                    result["filled_qty_est"] = max(result["filled_qty_est"], qty - remaining)
                    result["remaining_qty"] = remaining
                elif filled_by_status:
                    result["filled_qty_est"] = qty
                    result["remaining_qty"] = 0
                    result["ok"] = True
                    result["verified"] = True
                    return result

            if remaining <= 0:
                result["ok"] = True
                result["verified"] = True
                return result

            self._log(
                "warning",
                "[ExecGuard] fill incomplete attempt=%s sec_id=%s requested=%s remaining=%s target_net=%s current_net=%s",
                attempt + 1,
                security_id,
                qty,
                remaining,
                target_net,
                current_net,
            )

        # Final refresh for reporting
        if target_net is not None and self.config.verify_via_positions and hasattr(dw, "get_positions_raw"):
            try:
                current_net = self._get_net_qty(dw, security_id)
                result["final_net_qty"] = current_net
                remaining = max(0, abs(int(target_net) - int(current_net)))
                result["remaining_qty"] = remaining
                result["filled_qty_est"] = max(0, qty - remaining)
                if remaining == 0:
                    result["ok"] = True
                    result["verified"] = True
                    return result
            except Exception as exc:
                last_error = f"positions_fetch_failed:{exc}"
        result["error"] = last_error or "fill_not_confirmed"
        return result
