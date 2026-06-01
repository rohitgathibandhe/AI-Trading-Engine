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
from market_ai.intraday_defined_risk.execution import build_no_trade_decision
from market_ai.intraday_defined_risk import ops_runtime
from market_ai.intraday_defined_risk.ops_runtime import (
    HealthStatus,
    OpsPaths,
    ReconciliationStatus,
    RuntimeConfig,
    RuntimeMode,
    build_near_trade_candidates_csv,
    build_paper_context_override_report,
    build_paper_live_report,
    build_paper_time_window_audit,
    build_paper_live_validation_report,
    build_unified_health,
    evaluate_entry_gate,
    evaluate_paper_context_override_gate,
    load_paper_position,
    load_runtime_config,
    open_position_from_decision,
    record_paper_entry,
    reconcile_positions,
    save_paper_position,
    set_runtime_mode,
    log_validation_decision,
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
        validation_decisions=root / "validation_decisions.jsonl",
        shadow_report=root / "shadow_report.json",
        paper_report=root / "paper_report.json",
        validation_report=root / "validation_report.json",
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


def _override_decision() -> DecisionOutput:
    decision = _decision()
    decision.metadata.update(
        {
            "playbook": "NO_TRADE",
            "paper_trade_attribution": "PAPER_CONTEXT_OVERRIDE",
            "paper_candidate_reason": "PAPER_CONTEXT_OVERRIDE",
            "market_state": "TRANSITION",
            "tradability_class": "TRADABLE",
            "failure_type": "FAILED_BREAKOUT",
            "setup_direction": "BEARISH",
            "bearish_trade_score": 6.8,
            "no_trade_score": 4.9,
        }
    )
    return decision


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


def test_paper_context_override_gate_passes_for_override_candidate(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    config = RuntimeConfig(
        mode=RuntimeMode.PAPER_LIVE,
        paper_override_bearish_score_threshold=6.0,
        paper_override_margin=1.0,
        paper_experiment_entry_end_hhmm="14:30",
    )

    gate = evaluate_paper_context_override_gate(_override_decision(), _snapshot(), config=config, health=_healthy(), paths=paths)

    assert gate["allowed"] is True
    assert gate["primary_block_reason"] == "NONE"
    assert gate["decision_origin"] == "PAPER_CONTEXT_OVERRIDE"


def test_research_dataset_staleness_is_degraded_not_paper_blocking(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.creds_path.write_text('{"client_id":"client","access_token":"token"}')
    paths.reconciliation_status.write_text('{"status":"NO_POSITIONS"}')

    monkeypatch.setattr(
        ops_runtime,
        "build_data_freshness_report",
        lambda **_kwargs: {
            "fresh": False,
            "as_of": "2026-04-22T09:35:00+05:30",
            "stale_reasons": ["RESEARCH_5M_DATASET_STALE", "RESEARCH_OPTION_CHAIN_STALE"],
            "checks": {},
            "collector_pid_status": {"alive": True},
        },
    )

    snapshot = _snapshot(datetime.now().replace(second=0, microsecond=0))

    health = build_unified_health(config=RuntimeConfig(mode=RuntimeMode.PAPER_LIVE), paths=paths, snapshot=snapshot)

    assert health["status"] == HealthStatus.DEGRADED.value
    assert health["components"]["data_pipeline_health"]["status"] == HealthStatus.DEGRADED.value
    gate = evaluate_entry_gate(_decision(), snapshot, config=RuntimeConfig(mode=RuntimeMode.PAPER_LIVE), health=health, paths=paths)
    assert "HEALTH_BLOCKED" not in gate["block_reasons"]
    assert "DATA_PIPELINE_HEALTH_BLOCKED" not in gate["block_reasons"]


def test_collector_staleness_does_not_block_paper_live_when_live_data_is_healthy(monkeypatch, tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.creds_path.write_text('{"client_id":"client","access_token":"token"}')
    paths.reconciliation_status.write_text('{"status":"NO_POSITIONS"}')

    monkeypatch.setattr(
        ops_runtime,
        "build_data_freshness_report",
        lambda **_kwargs: {
            "fresh": False,
            "as_of": "2026-05-15T10:05:00+05:30",
            "stale_reasons": ["COLLECTOR_HEARTBEAT_STALE", "COLLECTOR_PROCESS_NOT_RUNNING", "COLLECTOR_LOG_STALE"],
            "checks": {
                "structured_5m_dataset": {"fresh": True},
                "structured_option_chain_dataset": {"fresh": True},
            },
            "collector_pid_status": {"alive": False},
        },
    )

    snapshot = _snapshot(datetime(2026, 5, 15, 10, 5))

    health = build_unified_health(config=RuntimeConfig(mode=RuntimeMode.PAPER_LIVE), paths=paths, snapshot=snapshot)

    assert health["status"] == HealthStatus.DEGRADED.value
    assert health["components"]["data_pipeline_health"]["status"] == HealthStatus.DEGRADED.value
    gate = evaluate_entry_gate(_decision(), snapshot, config=RuntimeConfig(mode=RuntimeMode.PAPER_LIVE), health=health, paths=paths)
    assert "HEALTH_BLOCKED" not in gate["block_reasons"]
    assert "DATA_PIPELINE_HEALTH_BLOCKED" not in gate["block_reasons"]


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


def test_validation_report_counts_decisions_and_data_readiness(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    snapshot = _snapshot()
    no_trade = build_no_trade_decision(
        RegimeLabel.NO_TRADE.value,
        ["score too low"],
        extra_metadata={
            "playbook": "SIDEWAYS_TO_BEARISH_REJECTION",
            "market_state": "TRANSITION",
            "tradability_class": "TRADABLE",
            "failure_type": "FAILED_RECLAIM",
            "bearish_trade_score": 5.9,
            "no_trade_score": 4.7,
            "primary_block_reason": "SCORE_TOO_LOW",
        },
    )

    log_validation_decision(no_trade, snapshot=snapshot, paths=paths)
    log_validation_decision(_decision(), snapshot=snapshot, paths=paths, block_reason="NONE")

    report = build_paper_live_validation_report(paths, session_date=snapshot.timestamp.date().isoformat())

    assert report["total_decisions"] == 2
    assert report["trade_count"] == 1
    assert report["no_trade_count"] == 1
    assert report["top_no_trade_reasons"]["SCORE_TOO_LOW"] == 1
    assert report["data_availability_summary"]["ready_count"] == 2
    assert report["market_state_distribution"]["TRANSITION"] == 2


def test_validation_detects_repeated_no_trade_reason(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    snapshot = _snapshot()
    no_trade = build_no_trade_decision(
        RegimeLabel.NO_TRADE.value,
        ["not tradable"],
        extra_metadata={
            "market_state": "TRANSITION",
            "bearish_trade_score": 4.0,
            "no_trade_score": 6.0,
            "primary_block_reason": "NOT_TRADABLE",
        },
    )

    for _ in range(51):
        log_validation_decision(no_trade, snapshot=snapshot, paths=paths, block_reason="NOT_TRADABLE")

    report = build_paper_live_validation_report(paths, session_date=snapshot.timestamp.date().isoformat())

    assert report["anomalies_by_type"]["REPEATED_IDENTICAL_NO_TRADE_REASON"] >= 1


def test_override_reports_and_near_trade_candidates_are_built(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    snapshot = _snapshot()

    strict_no_trade = build_no_trade_decision(
        RegimeLabel.NO_TRADE.value,
        ["pattern mismatch"],
        extra_metadata={
            "playbook": "SIDEWAYS_TO_BEARISH_REJECTION",
            "market_state": "TRANSITION",
            "tradability_class": "TRADABLE",
            "failure_type": "FAILED_RECLAIM",
            "setup_subtype": "SUPPLY_ZONE_FAILED_RECLAIM",
            "bearish_trade_score": 7.1786,
            "no_trade_score": 0.0,
            "primary_block_reason": "SETUP_PATTERN_MISMATCH",
        },
    )
    override_trade = _override_decision()
    gate = {
        "allowed": True,
        "primary_block_reason": "NONE",
    }
    position = open_position_from_decision(override_trade, snapshot)
    record_paper_entry(position, decision=override_trade, snapshot=snapshot, gate=gate, paths=paths)
    ops_runtime._append_jsonl(
        paths.paper_trades,
        {
            "event": "PAPER_EXIT",
            "timestamp": snapshot.timestamp.isoformat(),
            "session_date": snapshot.timestamp.date().isoformat(),
            "entry_timestamp": snapshot.timestamp.isoformat(),
            "exit_timestamp": snapshot.timestamp.isoformat(),
            "exit_reason": "TIME_EXIT",
            "playbook": "NO_TRADE",
            "subtype": None,
            "paper_trade_attribution": "PAPER_CONTEXT_OVERRIDE",
            "paper_candidate_reason": "PAPER_CONTEXT_OVERRIDE",
            "strategy": StrategyType.BEAR_CALL_CREDIT_SPREAD.value,
            "realized_paper_pnl": 1250.0,
            "mfe_rupees": 1400.0,
            "mae_rupees": -250.0,
            "legs": override_trade.legs,
        },
    )

    log_validation_decision(
        strict_no_trade,
        snapshot=snapshot,
        paths=paths,
        paper_candidate={
            "paper_candidate_decision": "NO_TRADE",
            "paper_candidate_reason": "CREDIT_TOO_LOW",
            "mapped_strategy": StrategyType.BEAR_CALL_CREDIT_SPREAD.value,
            "spread_construction_status": "CREDIT_TOO_LOW",
            "would_create_paper_trade": False,
            "paper_experiment_window_allows": True,
        },
    )
    log_validation_decision(
        strict_no_trade,
        snapshot=snapshot,
        paths=paths,
        paper_candidate={
            "paper_candidate_decision": "NO_TRADE",
            "paper_candidate_reason": "CREDIT_TOO_LOW",
            "mapped_strategy": StrategyType.BEAR_CALL_CREDIT_SPREAD.value,
            "spread_construction_status": "CREDIT_TOO_LOW",
            "would_create_paper_trade": False,
            "paper_experiment_window_allows": True,
        },
    )
    log_validation_decision(
        strict_no_trade,
        snapshot=snapshot,
        paths=paths,
        paper_candidate={
            "paper_candidate_decision": "TRADE",
            "paper_candidate_reason": "PAPER_CONTEXT_OVERRIDE",
            "mapped_strategy": StrategyType.BEAR_CALL_CREDIT_SPREAD.value,
            "spread_construction_status": "PASSED",
            "would_create_paper_trade": True,
            "paper_experiment_window_allows": True,
        },
        actual_trade_created=True,
        decision_origin="PAPER_CONTEXT_OVERRIDE",
    )

    paper_report = build_paper_live_report(paths=paths)
    opportunities = build_near_trade_candidates_csv(paths=paths, config=RuntimeConfig(mode=RuntimeMode.PAPER_LIVE))
    time_window = build_paper_time_window_audit(paths=paths, config=RuntimeConfig(mode=RuntimeMode.PAPER_LIVE))
    override_report = build_paper_context_override_report(paths=paths, config=RuntimeConfig(mode=RuntimeMode.PAPER_LIVE))

    assert paper_report["attribution_summary"]["PAPER_CONTEXT_OVERRIDE"]["pnl_rupees"] == 1250.0
    assert len(opportunities) == 2
    assert paths.near_trade_candidates_csv.exists()
    assert time_window["paper_experiment_entry_end_hhmm"] == "14:30"
    assert override_report["paper_context_override_paper_trades"] == 1
    assert override_report["rejection_reasons_after_override"]["CREDIT_TOO_LOW"] == 2
