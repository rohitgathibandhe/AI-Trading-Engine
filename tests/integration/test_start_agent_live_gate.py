from __future__ import annotations

import json
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


class _DupRowDW:
    def get_positions_raw(self):
        return [
            {"securityId": 3001, "netQty": 65},
            {"securityId": 3001, "netQty": 130},
            {"securityId": 3002, "netQty": -65},
        ]


class _ResidualDW:
    def get_positions_raw(self):
        return [
            {"securityId": 4001, "netQty": 65},
        ]


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


def test_build_risk_config_propagates_daily_max_loss_from_settings() -> None:
    risk = start_agent._build_risk_config(
        {
            "max_intraday_loss": -10000.0,
            "intraday_target": 4000.0,
            "allow_carry_forward": True,
        }
    )

    assert risk.max_intraday_loss == -10000.0
    assert risk.daily_max_loss == -10000.0


def test_exec_guard_aggregates_duplicate_position_rows() -> None:
    assert _exec_guard()._get_net_qty(_DupRowDW(), 3001) == 195
    assert _exec_guard()._get_net_qty(_DupRowDW(), 3002) == -65


def test_flatten_bkm_basket_keeps_state_when_residual_broker_position_remains(
    monkeypatch, tmp_path: Path
) -> None:
    blotter_path = tmp_path / "trade_blotter.csv"
    summary_path = tmp_path / "trade_blotter_summary.json"
    strategy_state = tmp_path / "strategy_state.json"
    monkeypatch.setattr(start_agent, "TRADE_BLOTTER_PATH", blotter_path)
    monkeypatch.setattr(start_agent, "TRADE_BLOTTER_SUMMARY", summary_path)
    monkeypatch.setattr(start_agent, "STRATEGY_STATE_FILE", strategy_state)
    monkeypatch.setattr(
        start_agent,
        "_place_leg_order",
        lambda *args, **kwargs: {"ok": True, "verified": True},
    )

    cfg = BatmanBKMConfig(lot_size=65, lot_multiplier=1)
    strategy = BatmanBKMStrategy(cfg)
    expiry = date(2026, 1, 29)
    strategy.basket = BatmanBKMBasket(
        expiry=expiry,
        legs=[
            Leg(option_type="PE", side="BUY", strike=21000, qty=65, entry=90.0, ltp=95.0, security_id="4001", expiry=expiry.isoformat()),
        ],
        net_credit=650.0,
        margin_required=1_000_000.0,
        credit_pct=0.065,
        entry_ts=datetime(2026, 1, 10, 10, 0),
        hedge_qty_call=1,
        hedge_qty_put=1,
    )

    out = start_agent._flatten_bkm_basket(
        dw=_ResidualDW(),
        bkm_strategy=strategy,
        trade_mode="live",
        reason="TEST_EXIT",
        live_order_executor=_exec_guard(),
    )

    assert out["ok"] is False
    assert out["closed_legs"] == 0
    assert out["residual_positions"] == [
        {"security_id": 4001, "strike": 21000, "opt": "PE", "net_qty": 65}
    ]
    assert strategy.basket is not None


def test_set_intraday_ai_runtime_status_disabled_reflects_open_bkm(monkeypatch, tmp_path: Path) -> None:
    status_path = tmp_path / "intraday_ai_advisor_status.json"
    monkeypatch.setattr(start_agent, "INTRADAY_AI_ADVISOR_STATUS_FILE", status_path)

    out = start_agent._set_intraday_ai_runtime_status(
        now=datetime(2026, 3, 30, 10, 0),
        intraday_enabled=False,
        has_open_bkm=True,
        allow_parallel_with_bkm=False,
        bkm_expiry="2026-04-28",
    )

    assert out["status"] == "DISABLED"
    assert out["signal"] == "NO_TRADE"
    assert out["has_open_bkm"] is True
    assert out["expiry"] == "2026-04-28"
    assert "INTRADAY_AI_DISABLED" in out["reasons"]
    persisted = json.loads(status_path.read_text())
    assert persisted["status"] == "DISABLED"
    assert persisted["recommendation"]["headline"] == "Intraday AI is disabled."


def test_set_intraday_ai_runtime_status_parked_for_open_bkm(monkeypatch, tmp_path: Path) -> None:
    status_path = tmp_path / "intraday_ai_advisor_status.json"
    monkeypatch.setattr(start_agent, "INTRADAY_AI_ADVISOR_STATUS_FILE", status_path)

    out = start_agent._set_intraday_ai_runtime_status(
        now=datetime(2026, 3, 30, 10, 5),
        intraday_enabled=True,
        has_open_bkm=True,
        allow_parallel_with_bkm=False,
        bkm_expiry="2026-04-28",
    )

    assert out["status"] == "PARKED"
    assert out["signal"] == "NO_TRADE"
    assert out["reasons"] == ["OPEN_BKM_POSITION_PRESENT"]
    assert out["recommendation"]["signal_text"] == "No intraday trade while Batman is already open."
