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
from .strategy import select_strategy
from .strikes import select_structure

ESTIMATED_ROUND_TRIP_COST_RUPEES_PER_LOT = 35.0
MIN_NET_EDGE_RUPEES = 500.0
MIN_NET_EDGE_COST_MULTIPLE = 3.0
PLAYBOOK_MIN_SAMPLES = 8.0
PLAYBOOK_BAD_PROFIT_FACTOR = 0.90
PLAYBOOK_GOOD_PROFIT_FACTOR = 1.20


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
    ) -> None:
        self.learning_store = learning_store or LearningStore()
        self.parameters = (parameters or self.learning_store.active_parameters()).clamped()
        self.open_position: OpenPosition | None = None
        self._current_features: dict[str, object] = {}
        self._session_date = None
        self._session_trade_counts: Counter[str] = Counter()

    def evaluate(self, snapshot: MarketSnapshot) -> DecisionOutput:
        self._reset_session(snapshot.timestamp)
        regime_state = classify_regime(snapshot, self.parameters)
        rationale = list(regime_state.reasons)
        playbook = str(regime_state.metadata.get("playbook") or "UNKNOWN")
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
        }

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
                extra_metadata=dict(regime_state.metadata),
            )
            self.learning_store.log_decision(decision, self._current_features, session_date=snapshot.timestamp.date().isoformat())
            return decision

        strategy, strategy_reasons = select_strategy(regime_state, snapshot.timestamp.time())
        if strategy == StrategyType.NO_TRADE:
            decision = build_no_trade_decision(
                regime_state.regime.value,
                strategy_reasons,
                confidence_score=regime_state.confidence,
                extra_metadata=dict(regime_state.metadata),
            )
            self.learning_store.log_decision(decision, self._current_features, session_date=snapshot.timestamp.date().isoformat())
            return decision
        playbook_stats = self.learning_store.playbook_summary(window=120)
        current_playbook_stats = playbook_stats.get(playbook)
        if current_playbook_stats is not None and current_playbook_stats["samples"] >= PLAYBOOK_MIN_SAMPLES:
            if current_playbook_stats["profit_factor"] < PLAYBOOK_BAD_PROFIT_FACTOR:
                decision = build_no_trade_decision(
                    regime_state.regime.value,
                    strategy_reasons + [f"Playbook {playbook} is underperforming on recent history; skipping deployment until learning recovers."],
                    confidence_score=regime_state.confidence,
                    extra_metadata=dict(regime_state.metadata),
                )
                self.learning_store.log_decision(decision, self._current_features, session_date=snapshot.timestamp.date().isoformat())
                return decision
        if self._session_trade_counts[strategy.value] >= 1:
            decision = build_no_trade_decision(
                regime_state.regime.value,
                strategy_reasons + [f"{strategy.value} already traded once this session; skipping repeat entry."],
                confidence_score=regime_state.confidence,
                extra_metadata=dict(regime_state.metadata),
            )
            self.learning_store.log_decision(decision, self._current_features, session_date=snapshot.timestamp.date().isoformat())
            return decision

        entry_allowed, entry_reason = validate_entry_time(strategy, snapshot.timestamp)
        if not entry_allowed:
            decision = build_no_trade_decision(
                regime_state.regime.value,
                strategy_reasons + ([entry_reason] if entry_reason else []),
                confidence_score=regime_state.confidence,
                extra_metadata=dict(regime_state.metadata),
            )
            self.learning_store.log_decision(decision, self._current_features, session_date=snapshot.timestamp.date().isoformat())
            return decision

        structure, structure_reasons = select_structure(strategy, snapshot, regime_state, self.parameters)
        if not structure:
            decision = build_no_trade_decision(
                regime_state.regime.value,
                strategy_reasons + structure_reasons,
                confidence_score=regime_state.confidence,
                extra_metadata=dict(regime_state.metadata),
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
                extra_metadata=dict(regime_state.metadata),
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
                extra_metadata=dict(regime_state.metadata),
            )
            self.learning_store.log_decision(decision, self._current_features, session_date=snapshot.timestamp.date().isoformat())
            return decision

        confidence = min(
            1.0,
            regime_state.confidence + min(self._current_features["credit_width_ratio"] / 0.40, 0.20),
        )
        if current_playbook_stats is not None and current_playbook_stats["samples"] >= PLAYBOOK_MIN_SAMPLES and current_playbook_stats["profit_factor"] >= PLAYBOOK_GOOD_PROFIT_FACTOR:
            confidence = min(1.0, confidence + 0.03)
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
                "bullish_setup": regime_state.metadata.get("bullish_setup"),
                "bearish_setup": regime_state.metadata.get("bearish_setup"),
                "smart_money_bias": regime_state.metadata.get("smart_money_bias"),
                "trade_plan": regime_state.metadata.get("trade_plan"),
                "structure_signal": regime_state.metadata.get("structure_signal"),
                "fvg_context": regime_state.metadata.get("fvg_context"),
                "order_block_context": regime_state.metadata.get("order_block_context"),
                "plan_execution": regime_state.metadata.get("plan_execution"),
                "plan_invalidation_level": regime_state.metadata.get("plan_invalidation_level"),
                "plan_target_level": regime_state.metadata.get("plan_target_level"),
                "plan_thesis": regime_state.metadata.get("plan_thesis"),
            },
        )
        self.learning_store.log_decision(decision, self._current_features, session_date=snapshot.timestamp.date().isoformat())
        return decision

    def start_position(self, snapshot: MarketSnapshot, decision: DecisionOutput) -> OpenPosition | None:
        self._reset_session(snapshot.timestamp)
        strategy = decision.strategy if isinstance(decision.strategy, StrategyType) else StrategyType(decision.strategy)
        structure, _ = select_structure(strategy, snapshot, classify_regime(snapshot, self.parameters), self.parameters)
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
