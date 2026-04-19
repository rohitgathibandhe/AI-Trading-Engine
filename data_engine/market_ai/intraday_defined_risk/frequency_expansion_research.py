from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date, datetime, time
from pathlib import Path
from statistics import mean
from typing import Callable

from .backtest import DEFAULT_ENTRY_TIMES, load_backtest_dataset, _session_previous_closes
from .data_models import AccountRiskLimits, AccountState, AdaptiveParameters, MarketSnapshot, OhlcvSeries, OpenPosition, RegimeState, StrategyType
from .execution import TIME_EXIT, build_open_position, evaluate_exit, simulate_entry_credit
from .monitor import ESTIMATED_ROUND_TRIP_COST_RUPEES_PER_LOT, MIN_NET_EDGE_COST_MULTIPLE, MIN_NET_EDGE_RUPEES
from .regime import classify_regime
from .risk import assess_trade_risk
from .strikes import select_best_structure

LIVE_APPROVED_BEARISH_PLAYBOOKS = {
    "SIDEWAYS_TO_BEARISH_REJECTION",
    "GAP_DOWN_BEARISH_CONTINUATION",
    "GAP_UP_BEARISH_FAILURE",
}

MONETIZATION_REJECTION_REASONS = {
    "CREDIT_TOO_LOW",
    "WIDTH_TOO_LARGE",
    "DELTA_TOO_HIGH",
    "HEDGE_TOO_EXPENSIVE",
    "LIQUIDITY_BAD",
    "INVALIDATION_TOO_CLOSE",
    "SPREAD_CONSTRUCTION_FAILED",
    "NET_EDGE_TOO_LOW",
}

NO_TRADE_REASON_BUCKETS = {
    "TRUE_RANGE",
    "NOT_TRADABLE",
    "SCORE_TOO_LOW",
    "FAILURE_TYPE_NONE",
    "PLAYBOOK_NOT_FOUND",
    "MONETIZATION_FAIL",
    "TIME_WINDOW_FAIL",
    "OTHER",
}

DAY_BUCKETS = {
    "TRADED_DAY",
    "TRUE_NO_EDGE_DAY",
    "MISSED_BEARISH_DAY",
    "MISSED_BULLISH_DAY",
    "GOOD_SETUP_BAD_MONETIZATION_DAY",
    "CLASSIFICATION_BLIND_SPOT_DAY",
}


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: object) -> bool:
    return bool(value)


def _load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def _group_by_session_date(records: list[dict[str, object]], *, trade: bool = False) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        if trade:
            session_date = str(record.get("entry_timestamp") or record.get("timestamp") or "")[:10]
        else:
            session_date = str(record.get("session_date") or record.get("timestamp") or "")[:10]
        if session_date:
            grouped[session_date].append(record)
    return grouped


def _decision_sort_key(decision: dict[str, object]) -> tuple[float, float, float]:
    metadata = decision.get("metadata") or {}
    funnel = metadata.get("trade_funnel") or {}
    return (
        _as_float(metadata.get("setup_quality_score")),
        _as_float(metadata.get("bearish_trade_score")),
        _as_float(funnel.get("monetization_score")),
    )


def _map_no_trade_reason(decision: dict[str, object], params: AdaptiveParameters) -> str:
    metadata = decision.get("metadata") or {}
    funnel = metadata.get("trade_funnel") or {}
    market_state = str(metadata.get("market_state") or "")
    tradability = str(metadata.get("tradability_class") or "")
    failure_type = str(metadata.get("failure_type") or "")
    playbook = str(metadata.get("playbook") or "")
    canonical = str(funnel.get("canonical_rejection_reason") or "")
    setup_quality = _as_float(metadata.get("setup_quality_score"))
    bearish_trade_score = _as_float(metadata.get("bearish_trade_score"))
    no_trade_score = _as_float(metadata.get("no_trade_score"))
    if market_state == "TRUE_RANGE":
        return "TRUE_RANGE"
    if canonical in MONETIZATION_REJECTION_REASONS:
        return "MONETIZATION_FAIL"
    if canonical in {"TIME_WINDOW", "ENTRY_TIME_INVALID"}:
        return "TIME_WINDOW_FAIL"
    if canonical == "STRUCTURALLY_NOT_MONETIZABLE" or tradability == "NOT_TRADABLE":
        return "NOT_TRADABLE"
    if failure_type == "NONE" and (_as_bool(metadata.get("bearish_entry_ready")) or _as_bool(funnel.get("setup_detected"))):
        return "FAILURE_TYPE_NONE"
    if playbook in {"NO_TRADE", "RANGE_NO_TRADE", "EVENT_DAY_NO_TRADE"}:
        return "PLAYBOOK_NOT_FOUND"
    if canonical in {"SETUP_FILTERED", "SETUP_QUALITY_TOO_LOW"} or setup_quality < params.weak_setup_quality_threshold or bearish_trade_score < max(0.0, no_trade_score):
        return "SCORE_TOO_LOW"
    return "OTHER"


def _refined_no_trade_reason(decision: dict[str, object], params: AdaptiveParameters) -> str:
    primary = _map_no_trade_reason(decision, params)
    if primary != "OTHER":
        return primary
    metadata = decision.get("metadata") or {}
    funnel = metadata.get("trade_funnel") or {}
    canonical = str(funnel.get("canonical_rejection_reason") or "")
    market_state = str(metadata.get("market_state") or "")
    tradability = str(metadata.get("tradability_class") or "")
    setup_direction = str(metadata.get("setup_direction") or "")
    if market_state == "DIRECTIONAL_BALANCE":
        return "DIRECTIONAL_BALANCE_UNRESOLVED"
    if canonical == "NO_VALID_STRATEGY":
        return "NO_VALID_STRATEGY"
    if canonical == "SETUP_PATTERN_MISMATCH":
        return "PATTERN_MISMATCH"
    if canonical == "PLAYBOOK_TIER_BLOCKED":
        return "TIER_POLICY_BLOCKED"
    if canonical == "LOW_EDGE_FILTERED":
        return "LOW_EDGE_POLICY_BLOCKED"
    if tradability == "LOW_EDGE":
        return "LOW_EDGE_UNCONVINCING"
    if setup_direction == "BEARISH":
        return "BEARISH_CLASSIFICATION_GAP"
    if setup_direction == "BULLISH":
        return "BULLISH_CLASSIFICATION_GAP"
    if canonical == "SETUP_FILTERED":
        return "SETUP_FILTERED_OTHER"
    return "OTHER"


def _best_bearish_decision(decisions: list[dict[str, object]]) -> dict[str, object] | None:
    bearish = [
        decision
        for decision in decisions
        if _as_bool((decision.get("metadata") or {}).get("bearish_entry_ready"))
        or str((decision.get("metadata") or {}).get("setup_direction") or "") == "BEARISH"
    ]
    return max(bearish, key=_decision_sort_key) if bearish else None


def _best_bullish_decision(decisions: list[dict[str, object]]) -> dict[str, object] | None:
    bullish = [
        decision
        for decision in decisions
        if _as_bool((decision.get("metadata") or {}).get("bullish_entry_ready"))
        or str((decision.get("metadata") or {}).get("setup_direction") or "") == "BULLISH"
    ]
    return max(bullish, key=_decision_sort_key) if bullish else None


def _day_bucket(
    decisions: list[dict[str, object]],
    trades: list[dict[str, object]],
    params: AdaptiveParameters,
) -> tuple[str, str, dict[str, object] | None]:
    if trades:
        trade = trades[0]
        return "TRADED_DAY", "EXECUTED", {
            "playbook": trade.get("playbook"),
            "strategy": trade.get("strategy"),
            "pnl_rupees": trade.get("pnl_rupees"),
        }

    setup_decisions = [
        decision for decision in decisions if _as_bool(((decision.get("metadata") or {}).get("trade_funnel") or {}).get("setup_detected"))
    ]
    if not setup_decisions:
        return "TRUE_NO_EDGE_DAY", "PLAYBOOK_NOT_FOUND", None

    reason_counts = Counter(_map_no_trade_reason(decision, params) for decision in setup_decisions)
    best_bearish = _best_bearish_decision(setup_decisions)
    best_bullish = _best_bullish_decision(setup_decisions)
    dominant_reason = reason_counts.most_common(1)[0][0]

    if dominant_reason == "MONETIZATION_FAIL":
        return "GOOD_SETUP_BAD_MONETIZATION_DAY", dominant_reason, best_bearish or best_bullish

    if best_bearish is not None:
        metadata = best_bearish.get("metadata") or {}
        setup_quality = _as_float(metadata.get("setup_quality_score"))
        bearish_trade_score = _as_float(metadata.get("bearish_trade_score"))
        threshold = _as_float(metadata.get("bearish_trade_score_threshold"), 6.1)
        if setup_quality >= params.strong_setup_quality_threshold or bearish_trade_score >= threshold - 0.35:
            if dominant_reason in {"FAILURE_TYPE_NONE", "PLAYBOOK_NOT_FOUND", "OTHER"}:
                return "CLASSIFICATION_BLIND_SPOT_DAY", dominant_reason, best_bearish
            return "MISSED_BEARISH_DAY", dominant_reason, best_bearish

    if best_bullish is not None:
        metadata = best_bullish.get("metadata") or {}
        setup_quality = _as_float(metadata.get("setup_quality_score"))
        bullish_trade_score = _as_float(metadata.get("bullish_trade_score"))
        threshold = _as_float(metadata.get("bullish_trade_score_threshold"), 7.5)
        if setup_quality >= params.strong_setup_quality_threshold or bullish_trade_score >= threshold - 0.35:
            return "MISSED_BULLISH_DAY", dominant_reason, best_bullish

    if dominant_reason in {"TRUE_RANGE", "NOT_TRADABLE", "SCORE_TOO_LOW", "PLAYBOOK_NOT_FOUND"}:
        return "TRUE_NO_EDGE_DAY", dominant_reason, best_bearish or best_bullish
    return "CLASSIFICATION_BLIND_SPOT_DAY", dominant_reason, best_bearish or best_bullish


SubtypeDetector = Callable[[datetime, RegimeState], bool]


def _meta(regime_state: RegimeState) -> dict[str, object]:
    return regime_state.metadata


def _base_shadow_bearish_eligibility(ts: datetime, regime_state: RegimeState, params: AdaptiveParameters) -> bool:
    metadata = _meta(regime_state)
    playbook = str(metadata.get("playbook") or "")
    if playbook not in LIVE_APPROVED_BEARISH_PLAYBOOKS:
        return False
    if ts.time() >= time(14, 0):
        return False
    if str(metadata.get("market_state") or "") == "TRUE_RANGE":
        return False
    if not _as_bool(metadata.get("bearish_entry_ready")):
        return False
    if _as_float(metadata.get("setup_quality_score")) < params.strong_setup_quality_threshold:
        return False
    if _as_float(metadata.get("bearish_trade_score")) < 5.55:
        return False
    if _as_float(metadata.get("bearish_trade_score")) <= _as_float(metadata.get("no_trade_score")) + 0.25:
        return False
    if _as_float(metadata.get("bearish_location_live_score")) < 1.5:
        return False
    return True


def _is_midday_weak_reclaim_rejection(ts: datetime, regime_state: RegimeState) -> bool:
    metadata = _meta(regime_state)
    return (
        time(11, 30) <= ts.time() <= time(13, 0)
        and str(metadata.get("bearish_setup") or "") == "PULLBACK_REJECTION"
        and str(metadata.get("market_state") or "") in {"TRANSITION", "DIRECTIONAL_BALANCE"}
        and _as_bool(metadata.get("lower_high_confirmed"))
        and str(metadata.get("opening_range_break_state") or "") in {"DOWN", "FAILED_UP"}
        and _as_float(metadata.get("price_vs_vwap"), 999.0) < 0.0
        and _as_float(metadata.get("momentum_persistence_score")) <= 2.35
        and _as_float(metadata.get("bearish_candle_quality_score")) >= 3.0
    )


def _is_or_low_retest_failure(ts: datetime, regime_state: RegimeState) -> bool:
    metadata = _meta(regime_state)
    return (
        str(metadata.get("opening_range_break_state") or "") == "DOWN"
        and _as_float(metadata.get("price_vs_or_low"), 999.0) < 0.0
        and str(metadata.get("bearish_setup") or "") in {"PULLBACK_REJECTION", "GAP_CONTINUATION", "SHALLOW_CONTINUATION"}
        and (_as_bool(metadata.get("lower_high_confirmed")) or _as_bool(metadata.get("accepted_breakdown")))
        and _as_float(metadata.get("bearish_candle_quality_score")) >= 2.75
    )


def _is_post_breakdown_retest_reject(ts: datetime, regime_state: RegimeState) -> bool:
    metadata = _meta(regime_state)
    return (
        str(metadata.get("bearish_setup") or "") in {"PULLBACK_REJECTION", "SHALLOW_CONTINUATION", "GAP_CONTINUATION"}
        and (_as_bool(metadata.get("accepted_breakdown")) or str(metadata.get("structure_signal") or "") in {"BEARISH_BOS", "BEARISH_CHOCH"})
        and _as_bool(metadata.get("lower_high_confirmed"))
        and _as_float(metadata.get("trend_efficiency_ratio")) >= 0.28
        and _as_float(metadata.get("price_vs_vwap"), 999.0) < 0.0
    )


def _is_lower_high_after_balance(ts: datetime, regime_state: RegimeState) -> bool:
    metadata = _meta(regime_state)
    return (
        str(metadata.get("market_state") or "") == "DIRECTIONAL_BALANCE"
        and _as_bool(metadata.get("lower_high_confirmed"))
        and str(metadata.get("bearish_setup") or "") == "PULLBACK_REJECTION"
        and str(metadata.get("opening_range_break_state") or "") in {"DOWN", "FAILED_UP"}
        and _as_float(metadata.get("candle_overlap_ratio")) <= 0.58
        and _as_float(metadata.get("price_vs_vwap"), 999.0) < 0.0
    )


def _is_bearish_transition_rejection(ts: datetime, regime_state: RegimeState) -> bool:
    metadata = _meta(regime_state)
    return (
        str(metadata.get("market_state") or "") == "TRANSITION"
        and str(metadata.get("bearish_setup") or "") == "PULLBACK_REJECTION"
        and _as_bool(metadata.get("lower_high_confirmed"))
        and str(metadata.get("opening_range_break_state") or "") in {"DOWN", "FAILED_UP"}
        and _as_float(metadata.get("bearish_candle_quality_score")) >= 3.25
        and _as_float(metadata.get("price_vs_vwap"), 999.0) < 0.0
    )


SHADOW_BEARISH_SUBTYPES: dict[str, SubtypeDetector] = {
    "MIDDAY_WEAK_RECLAIM_REJECTION": _is_midday_weak_reclaim_rejection,
    "OR_LOW_RETEST_FAILURE": _is_or_low_retest_failure,
    "POST_BREAKDOWN_RETEST_REJECT": _is_post_breakdown_retest_reject,
    "LOWER_HIGH_AFTER_BALANCE": _is_lower_high_after_balance,
    "BEARISH_TRANSITION_REJECTION": _is_bearish_transition_rejection,
}


def _iter_snapshots(
    data_path: str | Path,
    *,
    start: str,
    end: str,
    risk_limits: AccountRiskLimits,
):
    dataset = load_backtest_dataset(data_path)
    bars_5m_by_day = dataset["bars_5m_by_day"]
    bars_15m_by_day = dataset["bars_15m_by_day"]
    chain_map = dataset["chain_map"]
    previous_session_closes = _session_previous_closes(dataset["bars_5m"])
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    timestamps = sorted(ts for ts in chain_map if start_dt <= ts <= end_dt)

    current_session_date: date | None = None
    previous_chain_snapshot = None
    current_day_bars_5m = []
    current_day_bars_15m = []
    session_bars_5m = []
    session_bars_15m = []
    session_5m_index = 0
    session_15m_index = 0

    for ts in timestamps:
        if current_session_date is None:
            current_session_date = ts.date()
        elif ts.date() != current_session_date:
            current_session_date = ts.date()
            previous_chain_snapshot = None
            current_day_bars_5m = []
            current_day_bars_15m = []
            session_bars_5m = []
            session_bars_15m = []
            session_5m_index = 0
            session_15m_index = 0

        if not current_day_bars_5m:
            current_day_bars_5m = bars_5m_by_day.get(current_session_date, [])
        if not current_day_bars_15m:
            current_day_bars_15m = bars_15m_by_day.get(current_session_date, [])

        while session_5m_index < len(current_day_bars_5m) and current_day_bars_5m[session_5m_index].timestamp <= ts:
            session_bars_5m.append(current_day_bars_5m[session_5m_index])
            session_5m_index += 1
        while session_15m_index < len(current_day_bars_15m) and current_day_bars_15m[session_15m_index].timestamp <= ts:
            session_bars_15m.append(current_day_bars_15m[session_15m_index])
            session_15m_index += 1
        if not session_bars_5m or not session_bars_15m:
            continue
        chain_snapshot = chain_map[ts]
        snapshot = MarketSnapshot(
            nifty_5m=OhlcvSeries(timeframe_minutes=5, bars=session_bars_5m),
            nifty_15m=OhlcvSeries(timeframe_minutes=15, bars=session_bars_15m),
            option_chain=chain_snapshot,
            risk_limits=risk_limits,
            account_state=AccountState(realised_pnl_rupees=0.0, margin_used_rupees=0.0),
            live_vwap=None,
            lot_size=65,
            slippage_points=0.5,
            previous_option_chain=previous_chain_snapshot,
            previous_session_close=previous_session_closes.get(ts.date()),
        )
        previous_chain_snapshot = chain_snapshot
        yield ts, snapshot


def _shadow_candidate_quality(regime_state: RegimeState, params: AdaptiveParameters) -> bool:
    metadata = regime_state.metadata
    return (
        _as_float(metadata.get("setup_quality_score")) >= params.strong_setup_quality_threshold
        and _as_float(metadata.get("bearish_trade_score")) >= 5.55
        and _as_float(metadata.get("bearish_trade_score")) > _as_float(metadata.get("no_trade_score")) + 0.25
        and _as_float(metadata.get("bearish_location_live_score")) >= 1.5
    )


def _close_shadow_position(
    position: OpenPosition,
    snapshot: MarketSnapshot,
    params: AdaptiveParameters,
    *,
    force_time_exit: bool = False,
) -> dict[str, object] | None:
    current_regime = classify_regime(snapshot, params)
    exit_now = snapshot.timestamp.replace(hour=15, minute=15, second=0, microsecond=0) if force_time_exit else snapshot.timestamp
    exit_decision = evaluate_exit(position, current_snapshot=snapshot, current_regime=current_regime, now=exit_now)
    if not exit_decision.should_exit:
        return None
    trade_cost_rupees = round(position.lots * ESTIMATED_ROUND_TRIP_COST_RUPEES_PER_LOT, 2)
    pnl = exit_decision.pnl_rupees - trade_cost_rupees
    return {
        "entry_timestamp": position.entry_time.isoformat(),
        "timestamp": exit_now.isoformat(),
        "playbook": position.metadata.get("playbook"),
        "shadow_subtype": position.metadata.get("shadow_subtype"),
        "market_state": position.metadata.get("market_state"),
        "failure_type": position.metadata.get("failure_type"),
        "exit_reason": exit_decision.reason,
        "gross_pnl_rupees": round(exit_decision.pnl_rupees, 2),
        "transaction_cost_rupees": trade_cost_rupees,
        "pnl_rupees": round(pnl, 2),
    }


def _open_shadow_position(snapshot: MarketSnapshot, regime_state: RegimeState, params: AdaptiveParameters, subtype: str) -> OpenPosition | None:
    structure, _, structure_report = select_best_structure(
        StrategyType.BEAR_CALL_CREDIT_SPREAD,
        snapshot,
        regime_state,
        params,
        setup_quality_score=_as_float(regime_state.metadata.get("setup_quality_score")),
        playbook_tier="B",
    )
    if structure is None:
        return None
    risk = assess_trade_risk(
        structure=structure,
        lot_size=snapshot.lot_size,
        risk_limits=snapshot.risk_limits,
        account_state=snapshot.account_state,
        margin_estimate_per_lot=structure.margin_estimate_per_lot,
    )
    if not risk.allowed:
        return None
    expected_credit_rupees = max(structure.credit_points - snapshot.slippage_points, 0.0) * snapshot.lot_size * risk.lots
    expected_round_trip_cost_rupees = ESTIMATED_ROUND_TRIP_COST_RUPEES_PER_LOT * risk.lots
    expected_net_edge_rupees = expected_credit_rupees - expected_round_trip_cost_rupees
    playbook_min_edge = _as_float(regime_state.metadata.get("minimum_net_edge_rupees"))
    min_required_edge = max(MIN_NET_EDGE_RUPEES, playbook_min_edge, expected_round_trip_cost_rupees * MIN_NET_EDGE_COST_MULTIPLE)
    if expected_net_edge_rupees < min_required_edge:
        return None
    entry_credit_points = simulate_entry_credit(structure, slippage_points=snapshot.slippage_points)
    return build_open_position(
        structure=structure,
        lots=risk.lots,
        lot_size=snapshot.lot_size,
        entry_time=snapshot.timestamp,
        entry_credit_points=entry_credit_points,
        max_loss_rupees_per_lot=risk.max_loss_rupees_per_lot,
        extra_metadata={
            "playbook": regime_state.metadata.get("playbook"),
            "shadow_subtype": subtype,
            "market_state": regime_state.metadata.get("market_state"),
            "failure_type": regime_state.metadata.get("failure_type"),
            "setup_quality_score": regime_state.metadata.get("setup_quality_score"),
            "structure_report": structure_report,
        },
    )


def _shadow_subtype_report(
    data_path: str | Path,
    *,
    start: str,
    end: str,
    live_full: dict[str, object],
    params: AdaptiveParameters,
) -> dict[str, object]:
    risk_limits = AccountRiskLimits(
        max_risk_rupees_per_trade=20000,
        max_margin_rupees=200000,
        max_daily_loss_rupees=10000,
    )
    live_trade_days = {str(trade.get("entry_timestamp") or "")[:10] for trade in live_full.get("trades", [])}
    per_subtype = {
        name: {
            "setups_detected": 0,
            "setups_passing_quality": 0,
            "trades": [],
            "trade_days": set(),
        }
        for name in SHADOW_BEARISH_SUBTYPES
    }
    open_positions: dict[str, OpenPosition | None] = {name: None for name in SHADOW_BEARISH_SUBTYPES}
    traded_days: dict[str, set[str]] = {name: set() for name in SHADOW_BEARISH_SUBTYPES}
    last_snapshots: dict[str, MarketSnapshot | None] = {name: None for name in SHADOW_BEARISH_SUBTYPES}
    current_day: date | None = None

    for ts, snapshot in _iter_snapshots(data_path, start=start, end=end, risk_limits=risk_limits):
        if current_day is None:
            current_day = ts.date()
        elif ts.date() != current_day:
            for subtype, position in open_positions.items():
                if position is not None and last_snapshots[subtype] is not None:
                    closed = _close_shadow_position(position, last_snapshots[subtype], params, force_time_exit=True)
                    if closed is not None:
                        per_subtype[subtype]["trades"].append(closed)
                    open_positions[subtype] = None
            current_day = ts.date()

        regime_state = classify_regime(snapshot, params)
        metadata = regime_state.metadata
        playbook = str(metadata.get("playbook") or "")
        for subtype, detector in SHADOW_BEARISH_SUBTYPES.items():
            if open_positions[subtype] is not None:
                closed = _close_shadow_position(open_positions[subtype], snapshot, params)
                if closed is not None:
                    per_subtype[subtype]["trades"].append(closed)
                    open_positions[subtype] = None
                last_snapshots[subtype] = snapshot
                continue
            if ts.strftime("%H:%M") not in DEFAULT_ENTRY_TIMES:
                last_snapshots[subtype] = snapshot
                continue
            if ts.date().isoformat() in traded_days[subtype]:
                last_snapshots[subtype] = snapshot
                continue
            if not _base_shadow_bearish_eligibility(ts, regime_state, params):
                last_snapshots[subtype] = snapshot
                continue
            if not detector(ts, regime_state):
                last_snapshots[subtype] = snapshot
                continue
            per_subtype[subtype]["setups_detected"] += 1
            if not _shadow_candidate_quality(regime_state, params):
                last_snapshots[subtype] = snapshot
                continue
            per_subtype[subtype]["setups_passing_quality"] += 1
            shadow_regime = replace(
                regime_state,
                metadata=dict(regime_state.metadata)
                | {
                    "bearish_family": "SHADOW_BEARISH_RESEARCH",
                    "bearish_subtype": subtype,
                    "setup_subtype": subtype,
                    "failure_type": subtype,
                    "playbook": playbook,
                },
            )
            position = _open_shadow_position(snapshot, shadow_regime, params, subtype)
            if position is None:
                last_snapshots[subtype] = snapshot
                continue
            open_positions[subtype] = position
            traded_days[subtype].add(ts.date().isoformat())
            per_subtype[subtype]["trade_days"].add(ts.date().isoformat())
            last_snapshots[subtype] = snapshot

    for subtype, position in open_positions.items():
        if position is not None and last_snapshots[subtype] is not None:
            closed = _close_shadow_position(position, last_snapshots[subtype], params, force_time_exit=True)
            if closed is not None:
                per_subtype[subtype]["trades"].append(closed)

    rows = []
    for subtype, payload in per_subtype.items():
        trades = payload["trades"]
        pnl_values = [float(trade["pnl_rupees"]) for trade in trades]
        wins = sum(1 for value in pnl_values if value > 0)
        losses = sum(1 for value in pnl_values if value < 0)
        overlap_days = sorted(payload["trade_days"] & live_trade_days)
        new_days = sorted(payload["trade_days"] - live_trade_days)
        rows.append(
            {
                "subtype": subtype,
                "setups_detected": payload["setups_detected"],
                "setups_passing_quality": payload["setups_passing_quality"],
                "executed_shadow_trades": len(trades),
                "wins": wins,
                "losses": losses,
                "realized_pnl_rupees": round(sum(pnl_values), 2),
                "average_pnl_rupees": round(mean(pnl_values), 2) if pnl_values else 0.0,
                "overlap_with_existing_live_trades": len(overlap_days),
                "overlap_days": overlap_days[:10],
                "new_valid_trade_days": len(new_days),
                "new_valid_days_examples": new_days[:10],
                "sample_trades": trades[:5],
                "all_trades": trades,
            }
        )
    rows.sort(key=lambda row: (row["realized_pnl_rupees"], row["new_valid_trade_days"], -row["losses"]), reverse=True)
    return {"rows": rows}


def _simulate_promotion_variant(
    live_full: dict[str, object],
    bearish_shadow_report: dict[str, object],
    *,
    selected_subtypes: tuple[str, ...],
) -> dict[str, object]:
    baseline_trades = list(live_full.get("trades", []))
    baseline_trade_days = {str(trade.get("entry_timestamp") or "")[:10] for trade in baseline_trades}
    baseline_pnl = round(sum(_as_float(trade.get("pnl_rupees")) for trade in baseline_trades), 2)
    subtype_rows = {
        str(row.get("subtype") or ""): row
        for row in bearish_shadow_report.get("rows", [])
    }
    overlap_days: set[str] = set()
    candidate_rows: list[dict[str, object]] = []
    for subtype in selected_subtypes:
        row = subtype_rows.get(subtype)
        if not row:
            continue
        for trade in row.get("all_trades", []):
            trade_day = str(trade.get("entry_timestamp") or "")[:10]
            if trade_day in baseline_trade_days:
                overlap_days.add(trade_day)
                continue
            candidate_rows.append(dict(trade) | {"shadow_subtype": subtype})

    best_by_day: dict[str, dict[str, object]] = {}
    for trade in sorted(
        candidate_rows,
        key=lambda item: (
            str(item.get("entry_timestamp") or ""),
            str(item.get("shadow_subtype") or ""),
        ),
    ):
        trade_day = str(trade.get("entry_timestamp") or "")[:10]
        if trade_day not in best_by_day:
            best_by_day[trade_day] = trade

    added_trades = [best_by_day[day] for day in sorted(best_by_day)]
    added_pnls = [_as_float(trade.get("pnl_rupees")) for trade in added_trades]
    strict_trades = len(baseline_trades) + len(added_trades)
    strict_pnl = round(baseline_pnl + sum(added_pnls), 2)
    added_wins = sum(1 for pnl in added_pnls if pnl > 0.0)
    added_losses = sum(1 for pnl in added_pnls if pnl < 0.0)
    return {
        "selected_subtypes": list(selected_subtypes),
        "strict_trades": strict_trades,
        "strict_pnl_rupees": strict_pnl,
        "baseline_trades": len(baseline_trades),
        "baseline_pnl_rupees": baseline_pnl,
        "added_trades": len(added_trades),
        "added_trade_days": [str(trade.get("entry_timestamp") or "")[:10] for trade in added_trades],
        "unique_added_trade_days": len(added_trades),
        "overlap_with_existing_live_trades": len(overlap_days),
        "overlap_days": sorted(overlap_days),
        "added_wins": added_wins,
        "added_losses": added_losses,
        "added_expectancy_rupees": round(mean(added_pnls), 2) if added_pnls else 0.0,
        "strict_expectancy_rupees": round(strict_pnl / strict_trades, 2) if strict_trades else 0.0,
        "low_quality_opportunity_flow_reopened": bool(added_losses > 0),
        "added_trade_samples": added_trades,
        "delta_vs_v83": {
            "trades": len(added_trades),
            "pnl_rupees": round(sum(added_pnls), 2),
        },
    }


def _promotion_experiment_report(
    live_full: dict[str, object],
    bearish_shadow_report: dict[str, object],
) -> dict[str, object]:
    variants = {
        "midday_only": ("MIDDAY_WEAK_RECLAIM_REJECTION",),
        "transition_only": ("BEARISH_TRANSITION_REJECTION",),
        "combined": ("MIDDAY_WEAK_RECLAIM_REJECTION", "BEARISH_TRANSITION_REJECTION"),
    }
    results = {
        name: _simulate_promotion_variant(
            live_full,
            bearish_shadow_report,
            selected_subtypes=subtypes,
        )
        for name, subtypes in variants.items()
    }
    if all(result["delta_vs_v83"]["pnl_rupees"] <= 0.0 for result in results.values()):
        recommendation = "KEEP_BOTH_SHADOW"
    else:
        best_name, best_result = max(
            results.items(),
            key=lambda item: (item[1]["delta_vs_v83"]["pnl_rupees"], -item[1]["added_losses"]),
        )
        if best_name == "midday_only":
            recommendation = "PROMOTE_SUBTYPE_1"
        elif best_name == "transition_only":
            recommendation = "PROMOTE_SUBTYPE_2"
        else:
            recommendation = "PROMOTE_COMBINED"
    return {
        "baseline": {
            "strict_trades": len(live_full.get("trades", [])),
            "strict_pnl_rupees": round(sum(_as_float(trade.get("pnl_rupees")) for trade in live_full.get("trades", [])), 2),
        },
        "variants": results,
        "recommendation": recommendation,
    }


def _directional_balance_mining(full: dict[str, object]) -> dict[str, object]:
    live_trade_days = {str(trade.get("entry_timestamp") or "")[:10] for trade in full.get("trades", [])}
    decisions = full.get("decisions", [])
    diagnostics = {
        "BALANCE_BELOW_VWAP_THEN_REJECT": [],
        "LOWER_HIGH_INSIDE_BALANCE": [],
        "OR_LOW_PRESSURE_BUILD": [],
        "REPEATED_FAILED_RECLAIM_INSIDE_BALANCE": [],
    }
    per_day_flags: dict[str, set[str]] = defaultdict(set)

    for decision in decisions:
        session_date = str(decision.get("session_date") or "")
        if session_date in live_trade_days:
            continue
        metadata = decision.get("metadata") or {}
        if str(metadata.get("market_state") or "") != "DIRECTIONAL_BALANCE":
            continue
        if _as_float(metadata.get("price_vs_vwap"), 999.0) < 0.0 and str(metadata.get("bearish_setup") or "") == "PULLBACK_REJECTION":
            diagnostics["BALANCE_BELOW_VWAP_THEN_REJECT"].append(session_date)
            per_day_flags[session_date].add("BALANCE_BELOW_VWAP_THEN_REJECT")
        if _as_bool(metadata.get("lower_high_confirmed")) and _as_float(metadata.get("candle_overlap_ratio")) <= 0.60:
            diagnostics["LOWER_HIGH_INSIDE_BALANCE"].append(session_date)
            per_day_flags[session_date].add("LOWER_HIGH_INSIDE_BALANCE")
        if str(metadata.get("opening_range_break_state") or "") in {"DOWN", "FAILED_UP"} and _as_float(metadata.get("below_vwap_last6")) >= 4.0:
            diagnostics["OR_LOW_PRESSURE_BUILD"].append(session_date)
            per_day_flags[session_date].add("OR_LOW_PRESSURE_BUILD")
        if _as_float(metadata.get("bars_below_vwap")) >= 6.0 and str(metadata.get("failure_type") or "") in {"FAILED_RECLAIM", "NONE"}:
            diagnostics["REPEATED_FAILED_RECLAIM_INSIDE_BALANCE"].append(session_date)
            per_day_flags[session_date].add("REPEATED_FAILED_RECLAIM_INSIDE_BALANCE")

    rows = []
    for name, dates in diagnostics.items():
        unique_dates = sorted(set(dates))
        rows.append(
            {
                "diagnostic": name,
                "count": len(unique_dates),
                "example_days": unique_dates[:10],
                "share_of_directional_balance_no_trade_days": round(len(unique_dates) / max(len(per_day_flags), 1), 4),
            }
        )
    rows.sort(key=lambda row: row["count"], reverse=True)
    return {
        "directional_balance_no_trade_days": len(per_day_flags),
        "rows": rows,
    }


def write_frequency_expansion_reports(
    *,
    data_path: str | Path,
    full_benchmark_path: str | Path,
    start: str,
    end: str,
    output_dir: str | Path,
    report_tag: str,
) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full = _load_json(full_benchmark_path)
    params = AdaptiveParameters().clamped()

    decisions = full.get("decisions", [])
    trades = full.get("trades", [])
    live_trade_days = {str(trade.get("entry_timestamp") or "")[:10] for trade in trades}
    decision_days = _group_by_session_date(decisions, trade=False)
    trade_days = _group_by_session_date(trades, trade=True)

    dataset = load_backtest_dataset(data_path)
    all_days = sorted(
        day.isoformat()
        for day in dataset["bars_5m_by_day"].keys()
        if start[:10] <= day.isoformat() <= end[:10]
    )

    daily_rows = []
    day_bucket_counts: Counter[str] = Counter()
    no_trade_reason_counts: Counter[str] = Counter()
    refined_no_trade_reason_counts: Counter[str] = Counter()
    bucket_examples: dict[str, list[dict[str, object]]] = defaultdict(list)
    no_trade_rows = []

    for session_date in all_days:
        day_decisions = decision_days.get(session_date, [])
        day_trades = trade_days.get(session_date, [])
        bucket, dominant_reason, exemplar = _day_bucket(day_decisions, day_trades, params)
        day_bucket_counts[bucket] += 1
        if bucket != "TRADED_DAY":
            metadata = exemplar.get("metadata") if isinstance(exemplar, dict) else None
            refined_reason = None
            if day_decisions:
                refined_reason = _refined_no_trade_reason(max(day_decisions, key=_decision_sort_key), params)
            no_trade_reason_counts[dominant_reason] += 1
            if refined_reason:
                refined_no_trade_reason_counts[refined_reason] += 1
            no_trade_rows.append(
                {
                    "date": session_date,
                    "dominant_block_reason": dominant_reason,
                    "refined_block_reason": refined_reason,
                    "bucket": bucket,
                    "best_playbook": (metadata or {}).get("playbook") if metadata is not None else None,
                    "best_setup": (metadata or {}).get("bearish_setup") if metadata is not None else None,
                }
            )
        if len(bucket_examples[bucket]) < 5:
            if bucket == "TRADED_DAY" and day_trades:
                bucket_examples[bucket].append({
                    "date": session_date,
                    "playbook": day_trades[0].get("playbook"),
                    "pnl_rupees": day_trades[0].get("pnl_rupees"),
                })
            elif exemplar is not None:
                metadata = exemplar.get("metadata") if isinstance(exemplar, dict) else None
                bucket_examples[bucket].append({
                    "date": session_date,
                    "dominant_block_reason": dominant_reason,
                    "playbook": (metadata or {}).get("playbook") if metadata is not None else exemplar.get("playbook"),
                    "setup_quality_score": _as_float((metadata or {}).get("setup_quality_score")),
                    "bearish_trade_score": _as_float((metadata or {}).get("bearish_trade_score")),
                    "bullish_trade_score": _as_float((metadata or {}).get("bullish_trade_score")),
                    "failure_type": (metadata or {}).get("failure_type") if metadata is not None else None,
                })
            else:
                bucket_examples[bucket].append({
                    "date": session_date,
                    "dominant_block_reason": dominant_reason,
                    "playbook": None,
                })
        daily_rows.append(
            {
                "date": session_date,
                "day_bucket": bucket,
                "dominant_block_reason": dominant_reason,
                "traded": bool(day_trades),
                "trade_playbook": day_trades[0].get("playbook") if day_trades else None,
                "trade_strategy": day_trades[0].get("strategy") if day_trades else None,
                "trade_pnl_rupees": round(_as_float(day_trades[0].get("pnl_rupees")), 2) if day_trades else 0.0,
                "trade_entry_timestamp": day_trades[0].get("entry_timestamp") if day_trades else None,
                "best_bearish_playbook": ((_best_bearish_decision(day_decisions) or {}).get("metadata") or {}).get("playbook") if day_decisions else None,
                "best_bearish_failure_type": ((_best_bearish_decision(day_decisions) or {}).get("metadata") or {}).get("failure_type") if day_decisions else None,
                "best_bearish_trade_score": round(_as_float((((_best_bearish_decision(day_decisions) or {}).get("metadata") or {}).get("bearish_trade_score"))), 4) if day_decisions else 0.0,
                "best_bullish_playbook": ((_best_bullish_decision(day_decisions) or {}).get("metadata") or {}).get("playbook") if day_decisions else None,
                "signals_seen": len(day_decisions),
            }
        )

    daily_csv = output_dir / f"intraday_v83_daily_results_{report_tag}.csv"
    with daily_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(daily_rows[0].keys()))
        writer.writeheader()
        writer.writerows(daily_rows)

    day_classification_report = {
        "counts": dict(day_bucket_counts),
        "examples": dict(bucket_examples),
    }
    day_classification_path = output_dir / f"intraday_day_classification_report_{report_tag}.json"
    day_classification_path.write_text(json.dumps(day_classification_report, indent=2, sort_keys=True) + "\n")

    no_trade_audit = {
        "counts": dict(no_trade_reason_counts),
        "refined_counts": dict(refined_no_trade_reason_counts),
        "largest_bucket": no_trade_reason_counts.most_common(1)[0][0] if no_trade_reason_counts else None,
        "largest_refined_bucket": refined_no_trade_reason_counts.most_common(1)[0][0] if refined_no_trade_reason_counts else None,
        "rows": no_trade_rows,
    }
    no_trade_audit_path = output_dir / f"intraday_no_trade_audit_report_{report_tag}.json"
    no_trade_audit_path.write_text(json.dumps(no_trade_audit, indent=2, sort_keys=True) + "\n")

    bearish_shadow_report = _shadow_subtype_report(
        data_path,
        start=start,
        end=end,
        live_full=full,
        params=params,
    )
    bearish_shadow_path = output_dir / f"intraday_bearish_shadow_subtype_report_{report_tag}.json"
    bearish_shadow_path.write_text(json.dumps(bearish_shadow_report, indent=2, sort_keys=True) + "\n")

    promotion_report = _promotion_experiment_report(full, bearish_shadow_report)
    promotion_report_path = output_dir / f"intraday_bearish_promotion_experiment_{report_tag}.json"
    promotion_report_path.write_text(json.dumps(promotion_report, indent=2, sort_keys=True) + "\n")

    profitable_shadow_days = {
        str(trade.get("entry_timestamp") or "")[:10]
        for row in bearish_shadow_report["rows"]
        for trade in row.get("all_trades", [])
        if _as_float(trade.get("pnl_rupees")) > 0.0
    }
    directional_balance_report = _directional_balance_mining(full)
    for row in directional_balance_report["rows"]:
        example_days = row["example_days"]
        matching_days = [
            day
            for day in sorted({
                str(decision.get("session_date") or "")
                for decision in decisions
                if str(decision.get("session_date") or "") in profitable_shadow_days
            })
            if day in set(
                str(decision.get("session_date") or "")
                for decision in decisions
                if str((decision.get("metadata") or {}).get("market_state") or "") == "DIRECTIONAL_BALANCE"
            )
        ]
        diagnostic_days = set(
            str(decision.get("session_date") or "")
            for decision in decisions
            if str(decision.get("session_date") or "") not in live_trade_days
            and str((decision.get("metadata") or {}).get("market_state") or "") == "DIRECTIONAL_BALANCE"
            and (
                (row["diagnostic"] == "BALANCE_BELOW_VWAP_THEN_REJECT" and _as_float((decision.get("metadata") or {}).get("price_vs_vwap"), 999.0) < 0.0 and str((decision.get("metadata") or {}).get("bearish_setup") or "") == "PULLBACK_REJECTION")
                or (row["diagnostic"] == "LOWER_HIGH_INSIDE_BALANCE" and _as_bool((decision.get("metadata") or {}).get("lower_high_confirmed")) and _as_float((decision.get("metadata") or {}).get("candle_overlap_ratio")) <= 0.60)
                or (row["diagnostic"] == "OR_LOW_PRESSURE_BUILD" and str((decision.get("metadata") or {}).get("opening_range_break_state") or "") in {"DOWN", "FAILED_UP"} and _as_float((decision.get("metadata") or {}).get("below_vwap_last6")) >= 4.0)
                or (row["diagnostic"] == "REPEATED_FAILED_RECLAIM_INSIDE_BALANCE" and _as_float((decision.get("metadata") or {}).get("bars_below_vwap")) >= 6.0 and str((decision.get("metadata") or {}).get("failure_type") or "") in {"FAILED_RECLAIM", "NONE"})
            )
        )
        profitable_resolution_days = sorted(diagnostic_days & profitable_shadow_days)
        row["profitable_bearish_resolution_days"] = len(profitable_resolution_days)
        row["profitable_bearish_resolution_rate"] = round(len(profitable_resolution_days) / max(len(diagnostic_days), 1), 4)
        row["profitable_resolution_examples"] = profitable_resolution_days[:10]
    directional_balance_path = output_dir / f"intraday_directional_balance_mining_report_{report_tag}.json"
    directional_balance_path.write_text(json.dumps(directional_balance_report, indent=2, sort_keys=True) + "\n")

    return {
        "daily_csv": str(daily_csv),
        "day_classification_report": str(day_classification_path),
        "no_trade_audit_report": str(no_trade_audit_path),
        "bearish_shadow_report": str(bearish_shadow_path),
        "directional_balance_report": str(directional_balance_path),
        "promotion_experiment_report": str(promotion_report_path),
    }
