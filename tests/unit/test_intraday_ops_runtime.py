from __future__ import annotations

from datetime import datetime
from pathlib import Path

from market_ai.intraday_defined_risk.data_models import (
    AccountRiskLimits,
    AccountState,
    DecisionOutput,
    MarketSnapshot,
    OhlcvBar,
    OhlcvSeries,
    OptionType,
    OptionsChainSnapshot,
    OptionsContractQuote,
    RegimeLabel,
    StrategyType,
)
from market_ai.intraday_defined_risk.ops_runtime import (
    HealthStatus,
    OpsPaths,
    ReconciliationStatus,
    RuntimeConfig,
    RuntimeMode,
    evaluate_entry_gate,
    load_paper_position,
    load_runtime_config,
    open_position_from_decision,
    reconcile_positions,
    save_paper_position,
    set_runtime_mode,
)


def _paths(root: Path) -> OpsPaths:
    return OpsPaths(
        state_root=root,
        runtime_config=root / "runtime_config.json",
        runtime_state=root / "runtime_state.json",
        paper_state=root / "paper_state.json",
        shadow_decisions=root / "shadow_decisions.jsonl",
        paper_trades=root / "paper_trades.jsonl",
        operator_events=root / "operator_events.jsonl",
        reconciliation_status=root / "reconciliation_status.json",
        reconciliation_events=root / "reconciliation_events.jsonl",
        recovery_state=root / "recovery_state.json",
        emergency_flatten_events=root / "emergency_flatten_events.jsonl",
        shadow_report=root / "shadow_report.json",
        paper_report=root / "paper_report.json",
        operator_status_report=root / "operator_status_report.json",
        creds_path=root / "creds.json",
    )


def _snapshot(ts: datetime | None = None) -> MarketSnapshot:
    ts = ts or datetime(2026, 4, 3, 10, 0)
    bars_5m = [
        OhlcvBar(timestamp=ts.replace(hour=9, minute=15), open=22000.0, high=22020.0, low=21980.0, close=21990.0, volume=1000.0),
        OhlcvBar(timestamp=ts, open=21990.0, high=22010.0, low=21960.0, close=21970.0, volume=1200.0),
    ]
    bars_15m = [
        OhlcvBar(timestamp=ts.replace(hour=9, minute=15), open=22020.0, high=22030.0, low=21950.0, close=21970.0, volume=3000.0),
    ]
    quotes = [
        OptionsContractQuote(strike=22100.0, option_type=OptionType.CALL, bid=30.0, ask=31.0, ltp=30.5, delta=0.22),
        OptionsContractQuote(strike=22200.0, option_type=OptionType.CALL, bid=8.0, ask=9.0, ltp=8.5, delta=0.08),
    ]
    return MarketSnapshot(
        nifty_5m=OhlcvSeries(timeframe_minutes=5, bars=bars_5m),
        nifty_15m=OhlcvSeries(timeframe_minutes=15, bars=bars_15m),
        option_chain=OptionsChainSnapshot(timestamp=ts, expiry=ts.date(), spot=21970.0, quotes=quotes, margin_estimate_per_lot=10_000.0),
        risk_limits=AccountRiskLimits(max_risk_rupees_per_trade=10_000.0, max_margin_rupees=100_000.0, max_daily_loss_rupees=5_000.0),
        account_state=AccountState(realised_pnl_rupees=0.0, margin_used_rupees=0.0),
        live_vwap=22000.0,
        lot_size=65,
    )


def _decision() -> DecisionOutput:
    return DecisionOutput(
        action="TRADE",
        strategy=StrategyType.BEAR_CALL_CREDIT_SPREAD,
        regime=RegimeLabel.DOWN_TREND,
        rationale=["v83 bearish live candidate"],
        confidence_score=0.82,
        entry={"timestamp": "2026-04-03T10:00:00", "expected_credit_points": 22.0},
        stop_loss={"premium_multiple": 1.8},
        take_profit={"capture_pct": 0.65},
        time_exit="15:15",
        max_loss_rupees_per_lot=5_070.0,
        lots=1,
        legs=[
            {"action": "SELL", "option_type": "CALL", "strike": 22100.0, "bid": 30.0, "ask": 31.0, "ltp": 30.5, "delta": 0.22},
            {"action": "BUY", "option_type": "CALL", "strike": 22200.0, "bid": 8.0, "ask": 9.0, "ltp": 8.5, "delta": 0.08},
        ],
        metadata={
            "playbook": "SIDEWAYS_TO_BEARISH_REJECTION",
            "setup_direction": "BEARISH",
            "market_state": "TRANSITION",
            "tradability_class": "TRADABLE",
            "failure_type": "FAILED_RECLAIM",
            "bearish_trade_score": 7.2,
            "no_trade_score": 4.9,
            "structure_width_points": 100.0,
            "structure_credit_points": 22.0,
        },
    )


def _healthy() -> dict[str, object]:
    components = {
        name: {"status": HealthStatus.HEALTHY.value, "block_reasons": []}
        for name in [
            "broker_auth_health",
            "market_feed_health",
            "option_chain_health",
            "broker_position_sync_health",
            "state_store_health",
            "strategy_engine_health",
            "data_pipeline_health",
        ]
    }
    return {"status": HealthStatus.HEALTHY.value, "components": components, "block_reasons": []}


def test_default_runtime_mode_is_hard_disabled(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    config = load_runtime_config(paths)

    assert config.mode == RuntimeMode.LIVE_DISABLED
    assert config.live_arm is False


def test_shadow_mode_cannot_create_entry(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    config = RuntimeConfig(mode=RuntimeMode.SHADOW_LIVE)

    gate = evaluate_entry_gate(_decision(), _snapshot(), config=config, health=_healthy(), paths=paths)

    assert gate["allowed"] is False
    assert "MODE_NOT_PAPER_OR_MICRO" in gate["block_reasons"]


def test_micro_live_requires_manual_arm(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    config = RuntimeConfig(mode=RuntimeMode.MICRO_LIVE, live_arm=False)

    gate = evaluate_entry_gate(_decision(), _snapshot(), config=config, health=_healthy(), paths=paths)

    assert gate["allowed"] is False
    assert "MICRO_LIVE_NOT_ARMED" in gate["block_reasons"]


def test_paper_gate_passes_for_v83_bearish_candidate(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    config = RuntimeConfig(mode=RuntimeMode.PAPER_LIVE)

    gate = evaluate_entry_gate(_decision(), _snapshot(), config=config, health=_healthy(), paths=paths)

    assert gate["allowed"] is True
    assert gate["primary_block_reason"] == "NONE"


def test_paper_position_persists_without_broker_orphan_in_paper_mode(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    position = open_position_from_decision(_decision(), _snapshot())
    save_paper_position(position, paths=paths)

    restored = load_paper_position(paths)
    reconcile = reconcile_positions(paths=paths, mode=RuntimeMode.PAPER_LIVE, broker_positions=[])

    assert restored is not None
    assert restored.structure.strategy == StrategyType.BEAR_CALL_CREDIT_SPREAD
    assert reconcile["status"] == ReconciliationStatus.NO_POSITIONS.value


def test_micro_live_detects_broker_orphan_position(tmp_path: Path) -> None:
    paths = _paths(tmp_path)

    reconcile = reconcile_positions(
        paths=paths,
        mode=RuntimeMode.MICRO_LIVE,
        broker_positions=[{"side": "SELL", "strike": 22100, "option_type": "CALL"}],
    )

    assert reconcile["status"] == ReconciliationStatus.ORPHAN_POSITION.value
    assert reconcile["hard_lock"] is True
