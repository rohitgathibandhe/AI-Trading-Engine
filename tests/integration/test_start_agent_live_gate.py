from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import market_ai.start_agent as start_agent
from market_ai.modules.agents.live_gate import LiveGate, LiveGateConfig
from market_ai.modules.agents.live_order_executor import LiveOrderExecutor, LiveOrderExecutorConfig
from market_ai.modules.strategies.batman_bkm_monthly import (
    BatmanBKMConfig,
    BatmanBKMBasket,
    BatmanBKMStrategy,
    Leg,
)


class _DummyDW:
    def place_order(self, **kwargs):
        return {"ok": True, "order": kwargs}


class _ExecDW:
    def __init__(self, *, fill_sequence: list[float]) -> None:
        self.fill_sequence = list(fill_sequence)
        self.order_seq = 0
        self.net_by_sec: dict[int, int] = {}
        self.meta: dict[int, dict] = {}
        self.order_status_map: dict[str, dict] = {}

    def seed_leg(self, *, security_id: int, expiry: str, strike: float, option_type: str) -> None:
        self.meta[int(security_id)] = {
            "exchangeSegment": "NSE_FNO",
            "tradingSymbol": f"NIFTY {strike} {option_type}",
            "expiryDate": expiry,
            "strikePrice": strike,
            "optionType": option_type,
            "securityId": int(security_id),
        }

    def place_order(self, **kwargs):
        self.order_seq += 1
        order_id = f"O{self.order_seq}"
        sec_id = int(kwargs["security_id"])
        qty = int(kwargs["quantity"])
        side = str(kwargs["side"]).upper()
        sign = 1 if side.startswith("B") else -1
        ratio = self.fill_sequence.pop(0) if self.fill_sequence else 1.0
        fill_qty = max(0, min(qty, int(round(qty * ratio))))
        self.net_by_sec[sec_id] = int(self.net_by_sec.get(sec_id, 0)) + (sign * fill_qty)
        self.order_status_map[order_id] = {
            "orderStatus": "COMPLETED" if fill_qty == qty else "PENDING",
            "filledQuantity": fill_qty,
            "pendingQuantity": max(0, qty - fill_qty),
        }
        return {"orderId": order_id}

    def order_status(self, order_id: str):
        return dict(self.order_status_map.get(order_id) or {})

    def get_positions_raw(self):
        rows = []
        for sec_id, net in self.net_by_sec.items():
            base = dict(self.meta.get(sec_id) or {"securityId": sec_id})
            base["netQty"] = net
            rows.append(base)
        return rows


def _exec_guard() -> LiveOrderExecutor:
    return LiveOrderExecutor(
        config=LiveOrderExecutorConfig(
            enabled=True,
            fill_wait_sec=0,
            fill_poll_sec=0.0,
            max_retries=1,
            settle_delay_sec=0.0,
            verify_via_positions=True,
        ),
        logger=None,
    )


def test_failsafe_flattens_open_basket_and_locks_day(monkeypatch, tmp_path: Path) -> None:
    blotter_path = tmp_path / "trade_blotter.csv"
    summary_path = tmp_path / "trade_blotter_summary.json"
    strategy_state = tmp_path / "strategy_state.json"
    monkeypatch.setattr(start_agent, "TRADE_BLOTTER_PATH", blotter_path)
    monkeypatch.setattr(start_agent, "TRADE_BLOTTER_SUMMARY", summary_path)
    monkeypatch.setattr(start_agent, "STRATEGY_STATE_FILE", strategy_state)

    gate = LiveGate(
        config=LiveGateConfig(),
        status_path=tmp_path / "live_gate_status.json",
        sessions_path=tmp_path / "live_gate_sessions.jsonl",
        logger=None,
    )
    cfg = BatmanBKMConfig(lot_size=65, lot_multiplier=1)
    strategy = BatmanBKMStrategy(cfg)
    expiry = date(2026, 1, 29)
    strategy.basket = BatmanBKMBasket(
        expiry=expiry,
        legs=[
            Leg(option_type="CE", side="SELL", strike=22000, qty=65, entry=100.0, ltp=110.0, security_id="101", expiry=expiry.isoformat()),
            Leg(option_type="PE", side="BUY", strike=21000, qty=65, entry=90.0, ltp=95.0, security_id="102", expiry=expiry.isoformat()),
        ],
        net_credit=650.0,
        margin_required=1_000_000.0,
        credit_pct=0.065,
        entry_ts=datetime(2026, 1, 10, 10, 0),
        hedge_qty_call=2,
        hedge_qty_put=2,
    )

    closed_legs = start_agent._apply_live_gate_failsafe(
        live_gate=gate,
        reason="DATA_FAILSAFE_CONN_3_IN_600s",
        dw=_DummyDW(),
        bkm_strategy=strategy,
        trade_mode="paper",
    )

    assert closed_legs == 2
    assert strategy.basket is None
    snap = gate.snapshot()
    assert snap["status"] == "LOCKED"
    assert gate.should_block_entries() is True


def test_execute_bkm_open_live_rolls_back_on_partial_failure() -> None:
    expiry = date(2026, 1, 29)
    basket = BatmanBKMBasket(
        expiry=expiry,
        legs=[
            Leg(option_type="CE", side="SELL", strike=22000, qty=65, entry=100.0, ltp=100.0, security_id="1001", expiry=expiry.isoformat()),
            Leg(option_type="PE", side="SELL", strike=21000, qty=65, entry=110.0, ltp=110.0, security_id="1002", expiry=expiry.isoformat()),
        ],
        net_credit=700.0,
        margin_required=1_000_000.0,
        credit_pct=0.07,
        entry_ts=datetime(2026, 1, 10, 10, 0),
        hedge_qty_call=1,
        hedge_qty_put=1,
    )
    # open leg1 fills, open leg2 fails twice, rollback close for leg1 fills
    dw = _ExecDW(fill_sequence=[1.0, 0.0, 0.0, 1.0])
    dw.seed_leg(security_id=1001, expiry=expiry.isoformat(), strike=22000, option_type="CE")
    dw.seed_leg(security_id=1002, expiry=expiry.isoformat(), strike=21000, option_type="PE")

    out = start_agent._execute_bkm_open_live(dw=dw, basket=basket, live_order_executor=_exec_guard())

    assert out["ok"] is False
    assert out["opened_legs"] == 1
    assert out["planned_legs"] == 2
    assert int(out["rollback"]["submitted_close_legs"]) >= 1
    assert dw.net_by_sec.get(1001, 0) == 0
    assert dw.net_by_sec.get(1002, 0) == 0


def test_flatten_bkm_basket_keeps_state_on_close_execution_failure(monkeypatch, tmp_path: Path) -> None:
    blotter_path = tmp_path / "trade_blotter.csv"
    summary_path = tmp_path / "trade_blotter_summary.json"
    strategy_state = tmp_path / "strategy_state.json"
    monkeypatch.setattr(start_agent, "TRADE_BLOTTER_PATH", blotter_path)
    monkeypatch.setattr(start_agent, "TRADE_BLOTTER_SUMMARY", summary_path)
    monkeypatch.setattr(start_agent, "STRATEGY_STATE_FILE", strategy_state)

    cfg = BatmanBKMConfig(lot_size=65, lot_multiplier=1)
    strategy = BatmanBKMStrategy(cfg)
    expiry = date(2026, 1, 29)
    strategy.basket = BatmanBKMBasket(
        expiry=expiry,
        legs=[
            Leg(option_type="CE", side="SELL", strike=22000, qty=65, entry=100.0, ltp=110.0, security_id="2001", expiry=expiry.isoformat()),
            Leg(option_type="PE", side="BUY", strike=21000, qty=65, entry=90.0, ltp=95.0, security_id="2002", expiry=expiry.isoformat()),
        ],
        net_credit=650.0,
        margin_required=1_000_000.0,
        credit_pct=0.065,
        entry_ts=datetime(2026, 1, 10, 10, 0),
        hedge_qty_call=1,
        hedge_qty_put=1,
    )
    # CE close buy fills, PE close sell does not confirm
    dw = _ExecDW(fill_sequence=[1.0, 0.0, 0.0])
    dw.seed_leg(security_id=2001, expiry=expiry.isoformat(), strike=22000, option_type="CE")
    dw.seed_leg(security_id=2002, expiry=expiry.isoformat(), strike=21000, option_type="PE")
    dw.net_by_sec[2001] = -65
    dw.net_by_sec[2002] = 65

    out = start_agent._flatten_bkm_basket(
        dw=dw,
        bkm_strategy=strategy,
        trade_mode="live",
        reason="TEST_EXIT",
        live_order_executor=_exec_guard(),
    )

    assert out["ok"] is False
    assert out["closed_legs"] == 1
    assert strategy.basket is not None
