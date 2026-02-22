from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import market_ai.start_agent as start_agent
from market_ai.modules.agents.live_gate import LiveGate, LiveGateConfig
from market_ai.modules.strategies.batman_bkm_monthly import (
    BatmanBKMConfig,
    BatmanBKMBasket,
    BatmanBKMStrategy,
    Leg,
)


class _DummyDW:
    def place_order(self, **kwargs):
        return {"ok": True, "order": kwargs}


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
