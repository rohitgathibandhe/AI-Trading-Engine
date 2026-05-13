from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from market_ai.intraday_defined_risk.data_models import (
    AccountRiskLimits,
    AccountState,
    DecisionOutput,
    MarketSnapshot,
    OhlcvBar,
    OhlcvSeries,
    OpenPosition,
    OptionType,
    OptionsChainSnapshot,
    OptionsContractQuote,
    RegimeLabel,
    RegimeState,
    StrategyLeg,
    StrategyType,
    TradeStructure,
)
from market_ai.intraday_defined_risk.execution import build_open_position, evaluate_exit, validate_entry_context
from market_ai.intraday_defined_risk.features import compute_pcr_trend
from market_ai.intraday_defined_risk.ops_runtime import HealthStatus, OpsPaths, RuntimeConfig, RuntimeMode, evaluate_entry_gate, save_runtime_state
from market_ai.intraday_defined_risk.strategy import is_tradable_bearish_rejection
from market_ai.intraday_defined_risk.strikes import select_best_structure


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
        paper_context_override_report=root / "paper_context_override_report.json",
        near_trade_candidates_csv=root / "near_trade_candidates.csv",
        paper_time_window_audit=root / "paper_time_window_audit.json",
        operator_status_report=root / "operator_status_report.json",
        creds_path=root / "creds.json",
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


def _bars(ts: datetime, closes_5m: list[float]) -> list[OhlcvBar]:
    bars: list[OhlcvBar] = []
    current = ts.replace(hour=9, minute=15, second=0, microsecond=0)
    prior_close = closes_5m[0]
    for close in closes_5m:
        open_price = prior_close
        high = max(open_price, close) + 6.0
        low = min(open_price, close) - 6.0
        bars.append(OhlcvBar(timestamp=current, open=open_price, high=high, low=low, close=close, volume=1000.0))
        prior_close = close
        current += timedelta(minutes=5)
    return bars


def _snapshot(
    ts: datetime,
    *,
    closes_5m: list[float],
    spot: float,
    quotes: list[OptionsContractQuote],
    live_vwap: float,
) -> MarketSnapshot:
    bars_5m = _bars(ts, closes_5m)
    bars_15m = [bars_5m[index] for index in range(0, len(bars_5m), 3) if index < len(bars_5m)]
    return MarketSnapshot(
        nifty_5m=OhlcvSeries(timeframe_minutes=5, bars=bars_5m),
        nifty_15m=OhlcvSeries(timeframe_minutes=15, bars=bars_15m[: max(1, len(bars_15m))]),
        option_chain=OptionsChainSnapshot(timestamp=ts, expiry=ts.date(), spot=spot, quotes=quotes, margin_estimate_per_lot=10_000.0),
        risk_limits=AccountRiskLimits(max_risk_rupees_per_trade=20_000.0, max_margin_rupees=200_000.0, max_daily_loss_rupees=10_000.0),
        account_state=AccountState(realised_pnl_rupees=0.0, margin_used_rupees=0.0),
        live_vwap=live_vwap,
        lot_size=65,
    )


def _gate_decision(*, ts: datetime, market_state: str, bearish_score: float, no_trade_score: float) -> DecisionOutput:
    return DecisionOutput(
        action="TRADE",
        strategy=StrategyType.BEAR_CALL_CREDIT_SPREAD,
        regime=RegimeLabel.DOWN_TREND,
        rationale=["bearish live candidate"],
        confidence_score=0.85,
        entry={"timestamp": ts.isoformat(), "expected_credit_points": 22.0},
        stop_loss={"premium_multiple": 2.0},
        take_profit={"capture_pct": 0.65},
        time_exit=ts.replace(hour=15, minute=15).isoformat(),
        max_loss_rupees_per_lot=5070.0,
        lots=1,
        legs=[
            {"action": "SELL", "option_type": "CALL", "strike": 22100.0, "bid": 30.0, "ask": 31.0, "ltp": 30.5, "delta": 0.22},
            {"action": "BUY", "option_type": "CALL", "strike": 22200.0, "bid": 8.0, "ask": 9.0, "ltp": 8.5, "delta": 0.08},
        ],
        metadata={
            "playbook": "SIDEWAYS_TO_BEARISH_REJECTION",
            "setup_direction": "BEARISH",
            "market_state": market_state,
            "tradability_class": "TRADABLE",
            "failure_type": "FAILED_RECLAIM",
            "bearish_trade_score": bearish_score,
            "no_trade_score": no_trade_score,
            "trade_score_margin_bearish": 0.2,
        },
    )


def _regime_state(*, playbook: str = "SIDEWAYS_TO_BEARISH_REJECTION", pcr_trend: str = "STABLE") -> RegimeState:
    return RegimeState(
        regime=RegimeLabel.DOWN_TREND,
        trend_15m="DOWN",
        execution_5m="DOWN",
        ema20_15m=22000.0,
        ema20_slope_15m=-5.0,
        rv30_pct=0.45,
        or_length_minutes=15,
        opening_range_high=22020.0,
        opening_range_low=21960.0,
        vwap=22000.0,
        confidence=0.85,
        reasons=["bearish context"],
        metadata={
            "playbook": playbook,
            "pcr_trend": pcr_trend,
            "bearish_upper_wick_fraction": 0.62,
            "bearish_close_location": 0.40,
            "bearish_candle_quality_score": 0.80,
        },
    )


def test_compute_pcr_trend_detects_direction() -> None:
    previous_quotes = [
        OptionsContractQuote(strike=22000.0, option_type=OptionType.CALL, bid=10.0, ask=11.0, ltp=10.5, oi=1000),
        OptionsContractQuote(strike=21900.0, option_type=OptionType.PUT, bid=12.0, ask=13.0, ltp=12.5, oi=1000),
    ]
    current_quotes = [
        OptionsContractQuote(strike=22000.0, option_type=OptionType.CALL, bid=10.0, ask=11.0, ltp=10.5, oi=1400),
        OptionsContractQuote(strike=21900.0, option_type=OptionType.PUT, bid=12.0, ask=13.0, ltp=12.5, oi=900),
    ]

    assert compute_pcr_trend(current_quotes, previous_quotes) == "FALLING"
    assert compute_pcr_trend(previous_quotes, current_quotes) == "RISING"


def test_validate_entry_context_blocks_falling_pcr_for_bearish_spread() -> None:
    ts = datetime(2026, 5, 11, 10, 30)
    snapshot = _snapshot(
        ts,
        closes_5m=[22010.0, 22000.0, 21990.0, 21980.0, 21970.0, 21960.0],
        spot=21960.0,
        live_vwap=22000.0,
        quotes=[
            OptionsContractQuote(strike=22100.0, option_type=OptionType.CALL, bid=20.0, ask=21.0, ltp=20.5, delta=0.22, oi=1200),
            OptionsContractQuote(strike=22200.0, option_type=OptionType.CALL, bid=7.0, ask=8.0, ltp=7.5, delta=0.08, oi=900),
        ],
    )

    allowed, reason = validate_entry_context(StrategyType.BEAR_CALL_CREDIT_SPREAD, snapshot, _regime_state(pcr_trend="FALLING"))
    assert allowed is False
    assert reason == "PCR_TREND_FALLING_BEARISH_NEGATION"

    gap_down_allowed, gap_down_reason = validate_entry_context(
        StrategyType.BEAR_CALL_CREDIT_SPREAD,
        snapshot,
        _regime_state(playbook="GAP_DOWN_BEARISH_CONTINUATION", pcr_trend="FALLING"),
    )
    assert gap_down_allowed is True
    assert gap_down_reason is None


def test_validate_entry_context_blocks_bearish_spread_above_vwap() -> None:
    ts = datetime(2026, 5, 11, 10, 35)
    snapshot = _snapshot(
        ts,
        closes_5m=[21970.0, 21990.0, 22010.0, 22030.0, 22050.0, 22060.0],
        spot=22060.0,
        live_vwap=22000.0,
        quotes=[
            OptionsContractQuote(strike=22150.0, option_type=OptionType.CALL, bid=18.0, ask=19.0, ltp=18.5, delta=0.21, oi=1000),
            OptionsContractQuote(strike=22250.0, option_type=OptionType.CALL, bid=7.0, ask=8.0, ltp=7.5, delta=0.09, oi=850),
        ],
    )

    allowed, reason = validate_entry_context(StrategyType.BEAR_CALL_CREDIT_SPREAD, snapshot, _regime_state())
    assert allowed is False
    assert reason == "PRICE_ABOVE_VWAP_AT_ENTRY"


def test_is_tradable_bearish_rejection_requires_real_wick() -> None:
    assert is_tradable_bearish_rejection(_regime_state()) is True
    weak = _regime_state()
    weak.metadata["bearish_upper_wick_fraction"] = 0.30
    assert is_tradable_bearish_rejection(weak) is False


def test_directional_balance_override_uses_relative_margin(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.creds_path.write_text('{"client_id":"client","access_token":"token"}')
    paths.reconciliation_status.write_text('{"status":"NO_POSITIONS"}')
    now = datetime.now().replace(hour=11, minute=45, second=0, microsecond=0)
    save_runtime_state(
        {
            "mode": RuntimeMode.PAPER_LIVE.value,
            "live_arm": False,
            "session_date": now.date().isoformat(),
            "daily_lock": {"active": False, "reason": None, "locked_at": None},
            "session_trade_count": 0,
            "trade_sequence": [],
            "realized_pnl_rupees": 0.0,
            "total_pnl_rupees": 0.0,
            "primary_block_reason": "NONE",
        },
        paths=paths,
    )
    snapshot = _snapshot(
        now,
        closes_5m=[22000.0, 21995.0, 21990.0, 21985.0, 21980.0, 21975.0],
        spot=21975.0,
        live_vwap=22010.0,
        quotes=[
            OptionsContractQuote(strike=22100.0, option_type=OptionType.CALL, bid=20.0, ask=21.0, ltp=20.5, delta=0.22, oi=1200),
            OptionsContractQuote(strike=22200.0, option_type=OptionType.CALL, bid=7.0, ask=8.0, ltp=7.5, delta=0.08, oi=800),
        ],
    )
    config = RuntimeConfig(mode=RuntimeMode.PAPER_LIVE, directional_balance_score_margin_override=0.08)

    gate = evaluate_entry_gate(
        _gate_decision(ts=now, market_state="DIRECTIONAL_BALANCE", bearish_score=5.35, no_trade_score=5.0),
        snapshot,
        config=config,
        health=_healthy(),
        paths=paths,
    )

    assert gate["allowed"] is False
    assert gate["primary_block_reason"] == "BEARISH_SCORE_MARGIN_FAIL"
    assert gate["score_margin_required"] == 0.4


def test_second_trade_requires_first_trade_profit_and_gap(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.creds_path.write_text('{"client_id":"client","access_token":"token"}')
    paths.reconciliation_status.write_text('{"status":"NO_POSITIONS"}')
    now = datetime.now().replace(hour=11, minute=15, second=0, microsecond=0)
    save_runtime_state(
        {
            "mode": RuntimeMode.PAPER_LIVE.value,
            "live_arm": False,
            "session_date": now.date().isoformat(),
            "daily_lock": {"active": False, "reason": None, "locked_at": None},
            "session_trade_count": 1,
            "trade_sequence": [
                {
                    "entry_timestamp": now.replace(hour=10, minute=30).isoformat(),
                    "exit_timestamp": now.replace(hour=11, minute=0).isoformat(),
                    "pnl_rupees": 1250.0,
                    "exit_reason": "TAKE_PROFIT",
                }
            ],
            "realized_pnl_rupees": 1250.0,
            "total_pnl_rupees": 1250.0,
            "primary_block_reason": "NONE",
        },
        paths=paths,
    )
    snapshot = _snapshot(
        now,
        closes_5m=[22000.0, 21995.0, 21990.0, 21985.0, 21980.0, 21975.0],
        spot=21975.0,
        live_vwap=22010.0,
        quotes=[
            OptionsContractQuote(strike=22100.0, option_type=OptionType.CALL, bid=20.0, ask=21.0, ltp=20.5, delta=0.22, oi=1200),
            OptionsContractQuote(strike=22200.0, option_type=OptionType.CALL, bid=7.0, ask=8.0, ltp=7.5, delta=0.08, oi=800),
        ],
    )

    gate = evaluate_entry_gate(
        _gate_decision(ts=now, market_state="TRANSITION", bearish_score=7.2, no_trade_score=5.0),
        snapshot,
        config=RuntimeConfig(mode=RuntimeMode.PAPER_LIVE),
        health=_healthy(),
        paths=paths,
    )

    assert gate["allowed"] is False
    assert gate["primary_block_reason"] == "SECOND_TRADE_MIN_GAP_NOT_MET"


def test_oi_wall_alignment_rejects_short_strike_below_wall() -> None:
    ts = datetime(2026, 5, 11, 11, 0)
    snapshot = _snapshot(
        ts,
        closes_5m=[22000.0, 21992.0, 21986.0, 21980.0, 21976.0, 21970.0],
        spot=21970.0,
        live_vwap=22000.0,
        quotes=[
            OptionsContractQuote(strike=22100.0, option_type=OptionType.CALL, bid=25.0, ask=26.0, ltp=25.5, delta=0.22, iv=18.0, oi=900),
            OptionsContractQuote(strike=22200.0, option_type=OptionType.CALL, bid=14.0, ask=15.0, ltp=14.5, delta=0.16, iv=18.0, oi=1400),
        ],
    )
    regime_state = RegimeState(
        regime=RegimeLabel.DOWN_TREND,
        trend_15m="DOWN",
        execution_5m="DOWN",
        ema20_15m=22000.0,
        ema20_slope_15m=-5.0,
        rv30_pct=0.30,
        or_length_minutes=15,
        opening_range_high=22020.0,
        opening_range_low=21960.0,
        vwap=22000.0,
        confidence=0.85,
        reasons=["bearish"],
        metadata={
            "playbook": "SIDEWAYS_TO_BEARISH_REJECTION",
            "setup_quality_score": 11.0,
            "current_call_wall": 22200.0,
        },
    )

    structure, reasons, report = select_best_structure(
        StrategyType.BEAR_CALL_CREDIT_SPREAD,
        snapshot,
        regime_state,
        setup_quality_score=11.0,
        playbook_tier="A",
    )

    assert structure is None
    assert any(candidate.get("canonical_rejection_reason") == "OI_WALL_BELOW_SHORT_STRIKE" for candidate in report["candidate_evaluations"])
    assert "No vertical spread satisfied liquidity, delta/distance, and credit-efficiency rules." in reasons


def _open_bear_call_position(ts: datetime, *, playbook: str, current_quotes: list[OptionsContractQuote], entry_credit_points: float = 20.0) -> tuple[OpenPosition, MarketSnapshot]:
    structure = TradeStructure(
        strategy=StrategyType.BEAR_CALL_CREDIT_SPREAD,
        legs=[
            StrategyLeg(action="SELL", option_type=OptionType.CALL, strike=current_quotes[0].strike, quote=current_quotes[0]),
            StrategyLeg(action="BUY", option_type=OptionType.CALL, strike=current_quotes[1].strike, quote=current_quotes[1]),
        ],
        credit_points=entry_credit_points,
        width_points=float(current_quotes[1].strike - current_quotes[0].strike),
        rationale=["test structure"],
        metadata={},
    )
    position = build_open_position(
        structure=structure,
        lots=1,
        lot_size=65,
        entry_time=ts - timedelta(minutes=45),
        entry_credit_points=entry_credit_points,
        max_loss_rupees_per_lot=5200.0,
        extra_metadata={"playbook": playbook},
    )
    snapshot = _snapshot(
        ts,
        closes_5m=[22120.0 - (index * 4.0) for index in range(110)],
        spot=21684.0,
        live_vwap=21950.0,
        quotes=current_quotes,
    )
    return position, snapshot


def test_conviction_exit_holds_past_standard_take_profit() -> None:
    ts = datetime(2026, 5, 11, 12, 0)
    position, snapshot = _open_bear_call_position(
        ts,
        playbook="SIDEWAYS_TO_BEARISH_REJECTION",
        current_quotes=[
            OptionsContractQuote(strike=22100.0, option_type=OptionType.CALL, bid=7.5, ask=8.5, ltp=8.0, delta=0.10, iv=22.0, oi=1200),
            OptionsContractQuote(strike=22200.0, option_type=OptionType.CALL, bid=2.5, ask=3.5, ltp=3.0, delta=0.04, iv=20.0, oi=800),
        ],
        entry_credit_points=20.0,
    )

    exit_decision = evaluate_exit(position, current_snapshot=snapshot, now=ts)

    assert exit_decision.should_exit is False
    assert position.metadata["exit_mode"] == "CONVICTION_TRAIL"


def test_gap_down_slow_decay_triggers_time_exit() -> None:
    ts = datetime(2026, 5, 11, 13, 5)
    position, snapshot = _open_bear_call_position(
        ts,
        playbook="GAP_DOWN_BEARISH_CONTINUATION",
        current_quotes=[
            OptionsContractQuote(strike=22100.0, option_type=OptionType.CALL, bid=15.0, ask=15.6, ltp=15.3, delta=0.21, iv=19.0, oi=1200),
            OptionsContractQuote(strike=22200.0, option_type=OptionType.CALL, bid=2.6, ask=3.4, ltp=3.0, delta=0.08, iv=17.0, oi=800),
        ],
        entry_credit_points=20.0,
    )
    position.metadata["exit_mode"] = "STANDARD_TRAIL"

    exit_decision = evaluate_exit(position, current_snapshot=snapshot, now=ts)

    assert exit_decision.should_exit is True
    assert exit_decision.reason == "GAP_DOWN_SLOW_DECAY_TIME_EXIT"
