from __future__ import annotations

from typing import Any, Dict, List

from market_ai.modules.agents.live_order_executor import LiveOrderExecutor, LiveOrderExecutorConfig


class _FakeDW:
    def __init__(self, *, fill_sequence: List[float]) -> None:
        self.fill_sequence = list(fill_sequence)
        self.order_seq = 0
        self.net_by_sec: Dict[int, int] = {}
        self.order_status_by_id: Dict[str, Dict[str, Any]] = {}
        self.place_calls: List[Dict[str, Any]] = []

    def place_order(self, **kwargs):
        self.place_calls.append(dict(kwargs))
        self.order_seq += 1
        order_id = f"O{self.order_seq}"
        sec_id = int(kwargs["security_id"])
        qty = int(kwargs["quantity"])
        side = str(kwargs["side"]).upper()
        sign = 1 if side.startswith("B") else -1
        fill_ratio = self.fill_sequence.pop(0) if self.fill_sequence else 1.0
        fill_qty = max(0, min(qty, int(round(qty * fill_ratio))))
        self.net_by_sec[sec_id] = int(self.net_by_sec.get(sec_id, 0)) + (sign * fill_qty)
        if fill_qty >= qty:
            self.order_status_by_id[order_id] = {
                "orderStatus": "COMPLETED",
                "filledQuantity": qty,
                "pendingQuantity": 0,
            }
        else:
            self.order_status_by_id[order_id] = {
                "orderStatus": "PENDING",
                "filledQuantity": fill_qty,
                "pendingQuantity": max(0, qty - fill_qty),
            }
        return {"orderId": order_id}

    def order_status(self, order_id: str):
        return dict(self.order_status_by_id.get(order_id) or {})

    def get_positions_raw(self):
        rows = []
        for sec_id, net in self.net_by_sec.items():
            rows.append({"securityId": sec_id, "netQty": net})
        return rows


def _exec(cfg: LiveOrderExecutorConfig | None = None) -> LiveOrderExecutor:
    return LiveOrderExecutor(
        config=cfg
        or LiveOrderExecutorConfig(
            enabled=True,
            fill_wait_sec=0,
            fill_poll_sec=0.0,
            max_retries=1,
            settle_delay_sec=0.0,
            verify_via_positions=True,
        ),
        logger=None,
    )


def test_place_and_verify_success_first_attempt() -> None:
    dw = _FakeDW(fill_sequence=[1.0])
    ex = _exec()
    out = ex.place_and_verify(
        dw=dw,
        side="SELL",
        exchange_seg="NSE_FNO",
        security_id=101,
        quantity=65,
    )
    assert out["ok"] is True
    assert out["verified"] is True
    assert out["attempts"] == 1
    assert out["remaining_qty"] == 0
    assert out["pre_net_qty"] == 0
    assert out["target_net_qty"] == -65
    assert out["final_net_qty"] == -65


def test_place_and_verify_retries_remaining_after_partial_fill() -> None:
    dw = _FakeDW(fill_sequence=[0.5, 1.0])
    ex = _exec()
    out = ex.place_and_verify(
        dw=dw,
        side="SELL",
        exchange_seg="NSE_FNO",
        security_id=102,
        quantity=100,
    )
    assert out["ok"] is True
    assert out["verified"] is True
    assert out["attempts"] == 2
    assert out["remaining_qty"] == 0
    assert out["filled_qty_est"] == 100
    assert out["final_net_qty"] == -100
    assert len(dw.place_calls) == 2
    assert int(dw.place_calls[0]["quantity"]) == 100
    assert int(dw.place_calls[1]["quantity"]) == 50


def test_place_and_verify_fails_when_fill_not_confirmed() -> None:
    dw = _FakeDW(fill_sequence=[0.0, 0.0])
    ex = _exec(
        LiveOrderExecutorConfig(
            enabled=True,
            fill_wait_sec=0,
            fill_poll_sec=0.0,
            max_retries=1,
            settle_delay_sec=0.0,
            verify_via_positions=True,
        )
    )
    out = ex.place_and_verify(
        dw=dw,
        side="BUY",
        exchange_seg="NSE_FNO",
        security_id=103,
        quantity=40,
    )
    assert out["ok"] is False
    assert out["verified"] is False
    assert out["attempts"] == 2
    assert out["remaining_qty"] == 40
    assert out["final_net_qty"] == 0
    assert out["error"] is not None

