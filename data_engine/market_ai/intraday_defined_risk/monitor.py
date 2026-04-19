from __future__ import annotations

from collections import Counter
from datetime import datetime
from importlib import import_module
from pathlib import Path
from time import sleep
from typing import Protocol

from .data_models import (
    AdaptiveParameters,
    DecisionOutput,
    MarketSnapshot,
    OpenPosition,
    RegimeLabel,
    StrategyType,
)
from .execution import build_no_trade_decision, build_open_position, build_trade_decision, evaluate_exit, validate_entry_time
from .learning import LearningStore
from .regime import classify_regime
from .risk import assess_trade_risk, kill_switch_triggered
from .strategy import (
    REGIME_TRADABILITY_LOW_EDGE,
    REGIME_TRADABILITY_NOT_TRADABLE,
    classify_regime_tradability,
    low_edge_tradability_filter_reason,
    playbook_tier,
    playbook_time_window,
    required_confidence_for_playbook,
    select_strategy,
)
from .strikes import select_best_structure, select_structure

ESTIMATED_ROUND_TRIP_COST_RUPEES_PER_LOT = 35.0
MIN_NET_EDGE_RUPEES = 500.0
MIN_NET_EDGE_COST_MULTIPLE = 3.0
PLAYBOOK_MIN_SAMPLES = 8.0
PLAYBOOK_BAD_PROFIT_FACTOR = 0.90
PLAYBOOK_GOOD_PROFIT_FACTOR = 1.20


def _canonical_strategy_rejection(reasons: list[str], *, setup_detected: bool) -> str:
    joined = " | ".join(reasons)
    if not setup_detected:
        return "SETUP_NOT_DETECTED"
    if "Tier" in joined:
        return "PLAYBOOK_TIER_BLOCKED"
    if "require time >=" in joined or "only allowed until" in joined or "allowed only after" in joined:
        return "TIME_WINDOW"
    if "below the required" in joined:
        return "SETUP_QUALITY_TOO_LOW"
    if "require the dedicated" in joined or "not yet printed a valid" in joined or "not balanced enough" in joined:
        return "SETUP_PATTERN_MISMATCH"
    if "No aligned regime/trigger pair available." in joined:
        return "NO_VALID_STRATEGY"
    return "SETUP_FILTERED"


def _initial_trade_funnel(
    snapshot: MarketSnapshot,
    regime_state: RegimeLabel | object,
    *,
    allowed_playbook_tiers: tuple[str, ...],
    experimental_policy: dict[str, object] | None = None,
) -> dict[str, object]:
    metadata = regime_state.metadata  # type: ignore[attr-defined]
    playbook = str(metadata.get("playbook") or "UNKNOWN")
    tradability, tradability_reason = classify_regime_tradability(regime_state)  # type: ignore[arg-type]
    start_time, end_time = playbook_time_window(
        regime_state,  # type: ignore[arg-type]
        experimental_policy=experimental_policy,
    )
    setup_detected = bool(
        metadata.get("bullish_entry_ready")
        or metadata.get("bearish_entry_ready")
        or metadata.get("range_entry_ready")
        or playbook not in {"NO_TRADE", "RANGE_NO_TRADE", "EVENT_DAY_NO_TRADE", "UNKNOWN"}
    )
    confidence_required = required_confidence_for_playbook(playbook)
    passed_setup_quality = bool(setup_detected and float(regime_state.confidence) >= confidence_required)  # type: ignore[attr-defined]
    passed_time_gating = bool(
        start_time is not None
        and end_time is not None
        and start_time <= snapshot.timestamp.time() <= end_time
    ) if start_time is not None and end_time is not None else False
    return {
        "date": snapshot.timestamp.date().isoformat(),
        "timestamp": snapshot.timestamp.isoformat(),
        "regime": regime_state.regime.value,  # type: ignore[attr-defined]
        "playbook": playbook,
        "playbook_tier": playbook_tier(playbook),
        "market_state": metadata.get("market_state"),
        "market_state_bias": metadata.get("market_state_bias"),
        "state_quality_score": round(float(metadata.get("state_quality_score") or 0.0), 4),
        "state_confidence_score": round(float(metadata.get("state_confidence_score") or 0.0), 4),
        "tradability_class": metadata.get("tradability_class"),
        "failure_type": metadata.get("failure_type"),
        "option_chain_pressure_state": metadata.get("option_chain_pressure_state"),
        "market_state_score": round(float(metadata.get("market_state_score") or 0.0), 4),
        "trend_quality_score": round(float(metadata.get("trend_quality_score") or 0.0), 4),
        "failure_score": round(float(metadata.get("failure_score") or 0.0), 4),
        "location_score": round(float(metadata.get("location_score") or 0.0), 4),
        "option_chain_pressure_score": round(float(metadata.get("option_chain_pressure_score") or 0.0), 4),
        "live_monetization_score": round(float(metadata.get("monetization_score") or 0.0), 4),
        "tradability_score": round(float(metadata.get("tradability_score") or 0.0), 4),
        "bearish_trade_score": round(float(metadata.get("bearish_trade_score") or 0.0), 4),
        "bullish_trade_score": round(float(metadata.get("bullish_trade_score") or 0.0), 4),
        "no_trade_score": round(float(metadata.get("no_trade_score") or 0.0), 4),
        "regime_tradability": tradability,
        "tradability_reason": tradability_reason,
        "setup_subtype": metadata.get("setup_subtype"),
        "bullish_shadow_subtype": metadata.get("bullish_shadow_subtype"),
        "bearish_family": metadata.get("bearish_family"),
        "bearish_subtype": metadata.get("bearish_subtype"),
        "condor_profile": metadata.get("condor_profile"),
        "setup_quality_score": round(float(metadata.get("setup_quality_score") or 0.0), 4),
        "setup_detected": setup_detected,
        "passed_setup_quality": passed_setup_quality,
        "passed_time_gating": passed_time_gating,
        "passed_playbook_tier": playbook_tier(playbook) in allowed_playbook_tiers,
        "passed_tradability_gate": tradability != REGIME_TRADABILITY_NOT_TRADABLE,
        "passed_spread_construction": False,
        "passed_liquidity": False,
        "passed_credit_width": False,
        "passed_delta_band": False,
        "passed_anchor_distance": False,
        "passed_net_edge": False,
        "selected_strategy": StrategyType.NO_TRADE.value,
        "monetization_score": 0.0,
        "final_trade_score": 0.0,
        "final_result": "REJECTED",
        "canonical_rejection_reason": None,
        "best_candidate": None,
        "best_failed_candidate": None,
        "candidate_evaluations": [],
        "experimental_policy_name": experimental_policy.get("name") if experimental_policy else None,
    }


class MarketDataProvider(Protocol):
    def current_snapshot(self) -> MarketSnapshot: ...

    def current_structure_quotes(self, position: OpenPosition) -> MarketSnapshot: ...


class BrokerExecutor(Protocol):
    def enter_trade(self, decision: DecisionOutput) -> dict[str, object]: ...

    def exit_trade(self, position: OpenPosition, reason: str) -> dict[str, object]: ...


class IntradayDefinedRiskAgent:
    def __init__(
        self,
        risk_limits: dict[str, float] | None = None,
        learning_store: LearningStore | None = None,
        parameters: AdaptiveParameters | None = None,
        *,
        allowed_playbook_tiers: tuple[str, ...] = ("A",),
        benchmark_mode: str = "strict",
        experimental_policy: dict[str, object] | None = None,
        use_regime_tradability_layer: bool = True,
        use_market_state_engine: bool = False,
    ) -> None:
        self.learning_store = learning_store or LearningStore()
        self.parameters = (parameters or self.learning_store.active_parameters()).clamped()
        self.allowed_playbook_tiers = tuple(sorted(set(allowed_playbook_tiers)))
        self.benchmark_mode = benchmark_mode
        self.experimental_policy = dict(experimental_policy or {})
        self.use_regime_tradability_layer = use_regime_tradability_layer
        self.use_market_state_engine = use_market_state_engine
        self.open_position: OpenPosition | None = None
        self._current_features: dict[str, object] = {}
        self._session_date = None
        self._session_trade_counts: Counter[str] = Counter()

    def evaluate(self, snapshot: MarketSnapshot) -> DecisionOutput:
        self._reset_session(snapshot.timestamp)
        regime_state = classify_regime(snapshot, self.parameters)
        regime_state.metadata["enable_market_state_gating"] = self.use_market_state_engine
        rationale = list(regime_state.reasons)
        playbook = str(regime_state.metadata.get("playbook") or "UNKNOWN")
        regime_tradability, tradability_reason = classify_regime_tradability(regime_state)
        self._current_features = {
            "rv30_pct": regime_state.rv30_pct,
            "trend_15m": regime_state.trend_15m,
            "execution_5m": regime_state.execution_5m,
            "or_length_minutes": regime_state.or_length_minutes,
            "playbook": playbook,
            "day_archetype": regime_state.metadata.get("day_archetype"),
            "bullish_entry_score": regime_state.metadata.get("bullish_entry_score", 0.0),
            "bearish_entry_score": regime_state.metadata.get("bearish_entry_score", 0.0),
            "bullish_setup": regime_state.metadata.get("bullish_setup"),
            "bearish_setup": regime_state.metadata.get("bearish_setup"),
            "smart_money_bias": regime_state.metadata.get("smart_money_bias"),
            "bullish_flow_score": regime_state.metadata.get("bullish_flow_score", 0.0),
            "bearish_flow_score": regime_state.metadata.get("bearish_flow_score", 0.0),
            "put_support_strike": regime_state.metadata.get("put_support_strike"),
            "call_resistance_strike": regime_state.metadata.get("call_resistance_strike"),
            "wall_migration_bias": regime_state.metadata.get("wall_migration_bias"),
            "put_wall_shift": regime_state.metadata.get("put_wall_shift", 0.0),
            "call_wall_shift": regime_state.metadata.get("call_wall_shift", 0.0),
            "bullish_planner_alignment": regime_state.metadata.get("bullish_planner_alignment", False),
            "bearish_planner_alignment": regime_state.metadata.get("bearish_planner_alignment", False),
            "setup_quality_score": regime_state.metadata.get("setup_quality_score", 0.0),
            "setup_direction": regime_state.metadata.get("setup_direction"),
            "setup_subtype": regime_state.metadata.get("setup_subtype"),
            "bullish_shadow_subtype": regime_state.metadata.get("bullish_shadow_subtype"),
            "bearish_family": regime_state.metadata.get("bearish_family"),
            "bearish_subtype": regime_state.metadata.get("bearish_subtype"),
            "condor_profile": regime_state.metadata.get("condor_profile"),
            "market_state": regime_state.metadata.get("market_state"),
            "market_state_bias": regime_state.metadata.get("market_state_bias"),
            "state_quality_score": regime_state.metadata.get("state_quality_score", 0.0),
            "state_confidence_score": regime_state.metadata.get("state_confidence_score", 0.0),
            "tradability_class": regime_state.metadata.get("tradability_class"),
            "failure_type": regime_state.metadata.get("failure_type"),
            "option_chain_pressure_state": regime_state.metadata.get("option_chain_pressure_state"),
            "market_state_score": regime_state.metadata.get("market_state_score", 0.0),
            "trend_quality_score": regime_state.metadata.get("trend_quality_score", 0.0),
            "failure_score": regime_state.metadata.get("failure_score", 0.0),
            "location_score": regime_state.metadata.get("location_score", 0.0),
            "option_chain_pressure_score": regime_state.metadata.get("option_chain_pressure_score", 0.0),
            "live_monetization_score": regime_state.metadata.get("monetization_score", 0.0),
            "tradability_score": regime_state.metadata.get("tradability_score", 0.0),
            "bearish_trade_score": regime_state.metadata.get("bearish_trade_score", 0.0),
            "bullish_trade_score": regime_state.metadata.get("bullish_trade_score", 0.0),
            "no_trade_score": regime_state.metadata.get("no_trade_score", 0.0),
            "playbook_tier": playbook_tier(playbook),
            "regime_tradability": regime_tradability,
            "benchmark_mode": self.benchmark_mode,
            "experimental_policy_name": self.experimental_policy.get("name"),
            "use_regime_tradability_layer": self.use_regime_tradability_layer,
            "use_market_state_engine": self.use_market_state_engine,
        }
        trade_funnel = _initial_trade_funnel(
            snapshot,
            regime_state,
            allowed_playbook_tiers=self.allowed_playbook_tiers,
            experimental_policy=self.experimental_policy,
        )

        try:
            snapshot.validate()
        except Exception as exc:  # noqa: BLE001
            decision = build_no_trade_decision(RegimeLabel.NO_TRADE.value, [f"Invalid snapshot: {exc}"])
            self.learning_store.log_decision(decision, self._current_features, session_date=snapshot.timestamp.date().isoformat())
            return decision

        kill_switch, kill_reasons = kill_switch_triggered(snapshot.account_state, snapshot.risk_limits)
        if kill_switch:
            decision = build_no_trade_decision(
                RegimeLabel.NO_TRADE.value,
                rationale + kill_reasons,
                confidence_score=regime_state.confidence,
                extra_metadata=dict(regime_state.metadata) | {"trade_funnel": trade_funnel | {"canonical_rejection_reason": "KILL_SWITCH"}},
            )
            self.learning_store.log_decision(decision, self._current_features, session_date=snapshot.timestamp.date().isoformat())
            return decision

        strategy, strategy_reasons = select_strategy(
            regime_state,
            snapshot.timestamp.time(),
            allowed_playbook_tiers=self.allowed_playbook_tiers,
            experimental_policy=self.experimental_policy,
        )
        trade_funnel["selected_strategy"] = strategy.value
        if strategy == StrategyType.NO_TRADE:
            trade_funnel["canonical_rejection_reason"] = _canonical_strategy_rejection(
                strategy_reasons,
                setup_detected=bool(trade_funnel["setup_detected"]),
            )
            decision = build_no_trade_decision(
                regime_state.regime.value,
                strategy_reasons,
                confidence_score=regime_state.confidence,
                extra_metadata=dict(regime_state.metadata) | {"trade_funnel": trade_funnel},
            )
            self.learning_store.log_decision(decision, self._current_features, session_date=snapshot.timestamp.date().isoformat())
            return decision
        if self.use_regime_tradability_layer:
            if regime_tradability == REGIME_TRADABILITY_NOT_TRADABLE:
                decision = build_no_trade_decision(
                    regime_state.regime.value,
                    strategy_reasons + [tradability_reason],
                    confidence_score=regime_state.confidence,
                    extra_metadata=dict(regime_state.metadata) | {"trade_funnel": trade_funnel | {
                        "passed_tradability_gate": False,
                        "canonical_rejection_reason": "STRUCTURALLY_NOT_MONETIZABLE",
                    }},
                )
                self.learning_store.log_decision(decision, self._current_features, session_date=snapshot.timestamp.date().isoformat())
                return decision
            if regime_tradability == REGIME_TRADABILITY_LOW_EDGE:
                low_edge_reason = low_edge_tradability_filter_reason(regime_state, self.parameters)
                if low_edge_reason:
                    decision = build_no_trade_decision(
                        regime_state.regime.value,
                        strategy_reasons + [tradability_reason, low_edge_reason],
                        confidence_score=regime_state.confidence,
                        extra_metadata=dict(regime_state.metadata) | {"trade_funnel": trade_funnel | {
                            "passed_tradability_gate": False,
                            "canonical_rejection_reason": "LOW_EDGE_FILTERED",
                        }},
                    )
                    self.learning_store.log_decision(decision, self._current_features, session_date=snapshot.timestamp.date().isoformat())
                    return decision
        else:
            trade_funnel["passed_tradability_gate"] = True
        playbook_stats = self.learning_store.playbook_summary(window=120)
        current_playbook_stats = playbook_stats.get(playbook)
        if current_playbook_stats is not None and current_playbook_stats["samples"] >= PLAYBOOK_MIN_SAMPLES:
            if current_playbook_stats["profit_factor"] < PLAYBOOK_BAD_PROFIT_FACTOR:
                decision = build_no_trade_decision(
                    regime_state.regime.value,
                    strategy_reasons + [f"Playbook {playbook} is underperforming on recent history; skipping deployment until learning recovers."],
                    confidence_score=regime_state.confidence,
                    extra_metadata=dict(regime_state.metadata) | {"trade_funnel": trade_funnel | {"canonical_rejection_reason": "PLAYBOOK_UNDERPERFORMING"}},
                )
                self.learning_store.log_decision(decision, self._current_features, session_date=snapshot.timestamp.date().isoformat())
                return decision
        if self._session_trade_counts[strategy.value] >= 1:
            decision = build_no_trade_decision(
                regime_state.regime.value,
                strategy_reasons + [f"{strategy.value} already traded once this session; skipping repeat entry."],
                confidence_score=regime_state.confidence,
                extra_metadata=dict(regime_state.metadata) | {"trade_funnel": trade_funnel | {"canonical_rejection_reason": "SESSION_CAP"}},
            )
            self.learning_store.log_decision(decision, self._current_features, session_date=snapshot.timestamp.date().isoformat())
            return decision

        entry_allowed, entry_reason = validate_entry_time(strategy, snapshot.timestamp)
        if not entry_allowed:
            decision = build_no_trade_decision(
                regime_state.regime.value,
                strategy_reasons + ([entry_reason] if entry_reason else []),
                confidence_score=regime_state.confidence,
                extra_metadata=dict(regime_state.metadata) | {"trade_funnel": trade_funnel | {"canonical_rejection_reason": "ENTRY_TIME_INVALID"}},
            )
            self.learning_store.log_decision(decision, self._current_features, session_date=snapshot.timestamp.date().isoformat())
            return decision

        structure, structure_reasons, structure_report = select_best_structure(
            strategy,
            snapshot,
            regime_state,
            self.parameters,
            setup_quality_score=float(regime_state.metadata.get("setup_quality_score") or 0.0),
            playbook_tier=playbook_tier(playbook),
        )
        trade_funnel.update({
            "passed_spread_construction": bool(structure_report.get("passed_spread_construction")),
            "passed_liquidity": bool(structure_report.get("passed_liquidity")),
            "passed_credit_width": bool(structure_report.get("passed_credit_width")),
            "passed_delta_band": bool(structure_report.get("passed_delta_band")),
            "passed_anchor_distance": bool(structure_report.get("passed_anchor_distance")),
            "monetization_score": float(structure_report.get("monetization_score") or 0.0),
            "final_trade_score": float(structure_report.get("final_trade_score") or 0.0),
            "best_candidate": structure_report.get("best_candidate"),
            "best_failed_candidate": structure_report.get("best_failed_candidate"),
            "candidate_evaluations": structure_report.get("candidate_evaluations") or [],
        })
        if not structure:
            decision = build_no_trade_decision(
                regime_state.regime.value,
                strategy_reasons + structure_reasons,
                confidence_score=regime_state.confidence,
                extra_metadata=dict(regime_state.metadata) | {"trade_funnel": trade_funnel | {
                    "canonical_rejection_reason": structure_report.get("canonical_rejection_reason") or "SPREAD_CONSTRUCTION_FAILED",
                }},
            )
            self.learning_store.log_decision(decision, self._current_features, session_date=snapshot.timestamp.date().isoformat())
            return decision

        self._current_features["credit_width_ratio"] = structure.credit_points / structure.width_points if structure.width_points > 0 else 0.0
        risk = assess_trade_risk(
            structure=structure,
            lot_size=snapshot.lot_size,
            risk_limits=snapshot.risk_limits,
            account_state=snapshot.account_state,
            margin_estimate_per_lot=structure.margin_estimate_per_lot,
        )
        if not risk.allowed:
            decision = build_no_trade_decision(
                regime_state.regime.value,
                strategy_reasons + structure_reasons + risk.reasons,
                confidence_score=regime_state.confidence,
                extra_metadata=dict(regime_state.metadata) | {"trade_funnel": trade_funnel | {"canonical_rejection_reason": "RISK_LIMIT"}},
            )
            self.learning_store.log_decision(decision, self._current_features, session_date=snapshot.timestamp.date().isoformat())
            return decision

        expected_credit_rupees = max(structure.credit_points - snapshot.slippage_points, 0.0) * snapshot.lot_size * risk.lots
        expected_round_trip_cost_rupees = ESTIMATED_ROUND_TRIP_COST_RUPEES_PER_LOT * risk.lots
        expected_net_edge_rupees = expected_credit_rupees - expected_round_trip_cost_rupees
        self._current_features["expected_credit_rupees"] = expected_credit_rupees
        self._current_features["expected_round_trip_cost_rupees"] = expected_round_trip_cost_rupees
        self._current_features["expected_net_edge_rupees"] = expected_net_edge_rupees
        playbook_min_edge = float(regime_state.metadata.get("minimum_net_edge_rupees") or 0.0)
        min_required_edge = max(MIN_NET_EDGE_RUPEES, playbook_min_edge, expected_round_trip_cost_rupees * MIN_NET_EDGE_COST_MULTIPLE)
        if expected_net_edge_rupees < min_required_edge:
            decision = build_no_trade_decision(
                regime_state.regime.value,
                strategy_reasons
                + structure_reasons
                + [f"Expected net edge {expected_net_edge_rupees:.2f} rupees is below the required post-cost threshold of {min_required_edge:.2f} rupees."],
                confidence_score=regime_state.confidence,
                extra_metadata=dict(regime_state.metadata) | {"trade_funnel": trade_funnel | {"canonical_rejection_reason": "NET_EDGE_TOO_LOW"}},
            )
            self.learning_store.log_decision(decision, self._current_features, session_date=snapshot.timestamp.date().isoformat())
            return decision

        confidence = min(
            1.0,
            regime_state.confidence + min(self._current_features["credit_width_ratio"] / 0.40, 0.20),
        )
        if current_playbook_stats is not None and current_playbook_stats["samples"] >= PLAYBOOK_MIN_SAMPLES and current_playbook_stats["profit_factor"] >= PLAYBOOK_GOOD_PROFIT_FACTOR:
            confidence = min(1.0, confidence + 0.03)
        trade_funnel["passed_net_edge"] = True
        trade_funnel["final_result"] = "EXECUTED"
        trade_funnel["canonical_rejection_reason"] = "NONE"
        decision = build_trade_decision(
            structure=structure,
            regime=regime_state.regime,
            rationale=strategy_reasons,
            confidence_score=confidence,
            entry_time=snapshot.timestamp,
            lots=risk.lots,
            lot_size=snapshot.lot_size,
            max_loss_rupees_per_lot=risk.max_loss_rupees_per_lot,
            slippage_points=snapshot.slippage_points,
            extra_metadata={
                "day_archetype": regime_state.metadata.get("day_archetype"),
                "playbook": playbook,
                "setup_subtype": regime_state.metadata.get("setup_subtype"),
                "bullish_shadow_subtype": regime_state.metadata.get("bullish_shadow_subtype"),
                "bearish_family": regime_state.metadata.get("bearish_family"),
                "bearish_subtype": regime_state.metadata.get("bearish_subtype"),
                "bullish_setup": regime_state.metadata.get("bullish_setup"),
                "bearish_setup": regime_state.metadata.get("bearish_setup"),
                "smart_money_bias": regime_state.metadata.get("smart_money_bias"),
                "condor_profile": regime_state.metadata.get("condor_profile"),
                "market_state": regime_state.metadata.get("market_state"),
                "market_state_bias": regime_state.metadata.get("market_state_bias"),
                "state_quality_score": regime_state.metadata.get("state_quality_score"),
                "state_confidence_score": regime_state.metadata.get("state_confidence_score"),
                "tradability_class": regime_state.metadata.get("tradability_class"),
                "failure_type": regime_state.metadata.get("failure_type"),
                "option_chain_pressure_state": regime_state.metadata.get("option_chain_pressure_state"),
                "market_state_score": regime_state.metadata.get("market_state_score"),
                "trend_quality_score": regime_state.metadata.get("trend_quality_score"),
                "failure_score": regime_state.metadata.get("failure_score"),
                "location_score": regime_state.metadata.get("location_score"),
                "option_chain_pressure_score": regime_state.metadata.get("option_chain_pressure_score"),
                "live_monetization_score": regime_state.metadata.get("monetization_score"),
                "tradability_score": regime_state.metadata.get("tradability_score"),
                "bearish_trade_score": regime_state.metadata.get("bearish_trade_score"),
                "bullish_trade_score": regime_state.metadata.get("bullish_trade_score"),
                "no_trade_score": regime_state.metadata.get("no_trade_score"),
                "trade_plan": regime_state.metadata.get("trade_plan"),
                "structure_signal": regime_state.metadata.get("structure_signal"),
                "fvg_context": regime_state.metadata.get("fvg_context"),
                "order_block_context": regime_state.metadata.get("order_block_context"),
                "plan_execution": regime_state.metadata.get("plan_execution"),
                "plan_invalidation_level": regime_state.metadata.get("plan_invalidation_level"),
                "plan_target_level": regime_state.metadata.get("plan_target_level"),
                "plan_thesis": regime_state.metadata.get("plan_thesis"),
                "setup_quality_score": regime_state.metadata.get("setup_quality_score"),
                "setup_direction": regime_state.metadata.get("setup_direction"),
                "playbook_tier": playbook_tier(playbook),
                "regime_tradability": regime_tradability,
                "experimental_policy_name": self.experimental_policy.get("name"),
                "trade_funnel": trade_funnel,
            },
        )
        self.learning_store.log_decision(decision, self._current_features, session_date=snapshot.timestamp.date().isoformat())
        return decision

    def start_position(self, snapshot: MarketSnapshot, decision: DecisionOutput) -> OpenPosition | None:
        self._reset_session(snapshot.timestamp)
        strategy = decision.strategy if isinstance(decision.strategy, StrategyType) else StrategyType(decision.strategy)
        regime_state = classify_regime(snapshot, self.parameters)
        structure, _, _ = select_best_structure(
            strategy,
            snapshot,
            regime_state,
            self.parameters,
            setup_quality_score=float(decision.metadata.get("setup_quality_score") or regime_state.metadata.get("setup_quality_score") or 0.0),
            playbook_tier=str(decision.metadata.get("playbook_tier") or playbook_tier(str(decision.metadata.get("playbook") or "UNKNOWN"))),
        )
        if not structure:
            return None
        open_position = build_open_position(
            structure=structure,
            lots=decision.lots,
            lot_size=snapshot.lot_size,
            entry_time=snapshot.timestamp,
            entry_credit_points=decision.entry["expected_credit_points"],
            max_loss_rupees_per_lot=decision.max_loss_rupees_per_lot,
            extra_metadata=decision.metadata,
        )
        self.open_position = open_position
        self._session_trade_counts[strategy.value] += 1
        return open_position

    def manage_position(self, snapshot: MarketSnapshot) -> DecisionOutput | None:
        self._reset_session(snapshot.timestamp)
        if not self.open_position:
            return None
        current_regime = classify_regime(snapshot, self.parameters)
        exit_decision = evaluate_exit(
            self.open_position,
            current_snapshot=snapshot,
            current_regime=current_regime,
            now=snapshot.timestamp,
        )
        if not exit_decision.should_exit:
            return None

        self.learning_store.log_outcome(
            session_date=snapshot.timestamp.date().isoformat(),
            strategy=self.open_position.structure.strategy.value,
            pnl_rupees=exit_decision.pnl_rupees,
            max_drawdown_rupees=max(-exit_decision.pnl_rupees, 0.0),
            margin_utilisation=min(
                snapshot.account_state.margin_used_rupees / snapshot.risk_limits.max_margin_rupees,
                1.0,
            ) if snapshot.risk_limits.max_margin_rupees > 0 else 1.0,
            features=self._current_features | {"exit_reason": exit_decision.reason},
        )
        self.open_position = None
        return build_no_trade_decision(
            RegimeLabel.NO_TRADE.value,
            current_regime.reasons + [f"Exited open position due to {exit_decision.reason}.", f"PnL {exit_decision.pnl_rupees:.2f} rupees."],
            confidence_score=current_regime.confidence,
            extra_metadata=dict(current_regime.metadata),
        )

    def _reset_session(self, timestamp: datetime) -> None:
        session_date = timestamp.date()
        if self._session_date != session_date:
            self._session_date = session_date
            self._session_trade_counts = Counter()


def load_object(path_spec: str):
    module_name, symbol_name = path_spec.split(":", 1)
    module = import_module(module_name)
    return getattr(module, symbol_name)


def run_live(config: dict[str, object]) -> None:
    provider_cls = load_object(str(config["provider_class"]))
    executor_cls = load_object(str(config["executor_class"]))
    provider = provider_cls(config)
    executor = executor_cls(config)
    learning_store = LearningStore(config.get("learning_db_path", "/tmp/intraday_defined_risk_learning.sqlite3"))
    agent = IntradayDefinedRiskAgent(learning_store=learning_store)
    poll_seconds = int(config.get("poll_seconds", 30))

    while True:
        snapshot = provider.current_snapshot()
        if agent.open_position:
            current_position = agent.open_position
            exit_decision = agent.manage_position(snapshot)
            if exit_decision is not None and current_position is not None:
                reason = exit_decision.rationale[-2] if len(exit_decision.rationale) >= 2 else "EXIT"
                executor.exit_trade(current_position, reason)
                print(exit_decision.to_json())
        else:
            decision = agent.evaluate(snapshot)
            print(decision.to_json())
            if decision.action == "TRADE":
                executor.enter_trade(decision)
                agent.start_position(snapshot, decision)
        sleep(poll_seconds)


def dump_decision(path: str | Path, decision: DecisionOutput) -> None:
    Path(path).write_text(decision.to_json() + "\n")
