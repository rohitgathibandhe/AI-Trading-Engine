from __future__ import annotations

import json
import os
from collections import Counter
from datetime import datetime, time as dtime, timedelta, timezone
from importlib import import_module
from pathlib import Path
from time import sleep
from typing import Protocol

from .data_models import (
    AdaptiveParameters,
    DecisionOutput,
    MarketSnapshot,
    OpenPosition,
    OptionType,
    RegimeLabel,
    RegimeState,
    StrategyLeg,
    StrategyType,
    TradeStructure,
)
from .execution import build_no_trade_decision, build_open_position, build_trade_decision, evaluate_exit, evaluate_hedge_opportunity, validate_entry_context, validate_entry_time
from .learning import LearningStore
from .ops_runtime import (
    RuntimeMode,
    build_operator_status_report,
    build_unified_health,
    evaluate_entry_gate,
    evaluate_paper_context_override_gate,
    load_paper_position,
    load_runtime_config,
    load_runtime_state,
    log_validation_decision,
    log_shadow_decision,
    manage_paper_position,
    open_position_from_decision,
    record_paper_entry,
    record_runtime_trade_entry,
    record_runtime_trade_exit,
    reconcile_positions,
    save_runtime_state,
)
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

_IST = timezone(timedelta(hours=5, minutes=30))
_MARKET_OPEN_TIME = dtime(9, 15)
_MARKET_CLOSE_TIME = dtime(15, 35)

ESTIMATED_ROUND_TRIP_COST_RUPEES_PER_LOT = 35.0
MIN_NET_EDGE_RUPEES = 500.0
MIN_NET_EDGE_COST_MULTIPLE = 3.0
PLAYBOOK_MIN_SAMPLES = 8.0
PLAYBOOK_BAD_PROFIT_FACTOR = 0.90
PLAYBOOK_GOOD_PROFIT_FACTOR = 1.20
STATE_ROOT = Path(__file__).resolve().parents[1] / "state"
AGENT_HEARTBEAT_PATH = STATE_ROOT / "intraday_v83_heartbeat.json"


def _write_v83_agent_heartbeat(*, runtime_config: object, phase: str) -> None:
    """Keep the shared UI watchdog heartbeat fresh for the v83 runner."""
    mode = getattr(runtime_config, "mode", None)
    mode_value = getattr(mode, "value", str(mode or "UNKNOWN"))
    payload = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "status": "RUNNING",
        "phase": phase,
        "pid": os.getpid(),
        "trade_mode": "paper" if mode_value == RuntimeMode.PAPER_LIVE.value else mode_value.lower(),
        "strategy_file": "intraday_defined_risk_v83",
        "v83_runtime_mode": mode_value,
    }
    try:
        AGENT_HEARTBEAT_PATH.parent.mkdir(parents=True, exist_ok=True)
        AGENT_HEARTBEAT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except Exception:
        # Heartbeat write failures are reported via state-store health; do not stop trade management.
        return


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


def _decision_block_reason(decision: DecisionOutput, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    metadata = decision.metadata if isinstance(decision.metadata, dict) else {}
    funnel = metadata.get("trade_funnel") if isinstance(metadata.get("trade_funnel"), dict) else {}
    for key in ("primary_block_reason", "canonical_rejection_reason", "rejection_reason"):
        value = metadata.get(key)
        if value:
            return str(value)
    value = funnel.get("canonical_rejection_reason") if isinstance(funnel, dict) else None
    if value:
        return str(value)
    return "NONE" if decision.action == "TRADE" else "NO_TRADE_DECISION"


def _update_runtime_decision_state(
    decision: DecisionOutput,
    *,
    runtime_config: object,
    snapshot: MarketSnapshot,
    block_reason: str | None = None,
) -> None:
    resolved = _decision_block_reason(decision, block_reason)
    runtime_state = load_runtime_state(config=runtime_config)
    runtime_state["mode"] = runtime_config.mode.value
    runtime_state["live_arm"] = runtime_config.live_arm
    runtime_state["primary_block_reason"] = resolved
    runtime_state["last_candidate"] = decision.to_dict()
    runtime_state["last_decision_at"] = snapshot.timestamp.isoformat()
    save_runtime_state(runtime_state)


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
        "bearish_shadow_subtype": metadata.get("bearish_shadow_subtype"),
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


def _shadow_only_failed_breakout_candidate(
    *,
    snapshot: MarketSnapshot,
    agent: "IntradayDefinedRiskAgent",
    decision: DecisionOutput,
) -> DecisionOutput | None:
    metadata = decision.metadata if isinstance(decision.metadata, dict) else {}
    if str(metadata.get("bearish_shadow_subtype") or "") != "BEARISH_FAILED_BREAKOUT_TRANSITION":
        return None
    regime_state = classify_regime(snapshot, agent.parameters)
    if str(regime_state.metadata.get("bearish_shadow_subtype") or "") != "BEARISH_FAILED_BREAKOUT_TRANSITION":
        return None
    entry_context_allowed, _ = validate_entry_context(StrategyType.BEAR_CALL_CREDIT_SPREAD, snapshot, regime_state)
    if not entry_context_allowed:
        return None
    structure, structure_reasons, structure_report = select_best_structure(
        StrategyType.BEAR_CALL_CREDIT_SPREAD,
        snapshot,
        regime_state,
        agent.parameters,
        setup_quality_score=float(regime_state.metadata.get("setup_quality_score") or 0.0),
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
        playbook=str(regime_state.metadata.get("playbook") or ""),
    )
    if not risk.allowed:
        return None
    expected_credit_rupees = max(structure.credit_points - snapshot.slippage_points, 0.0) * snapshot.lot_size * risk.lots
    expected_round_trip_cost_rupees = ESTIMATED_ROUND_TRIP_COST_RUPEES_PER_LOT * risk.lots
    expected_net_edge_rupees = expected_credit_rupees - expected_round_trip_cost_rupees
    playbook_min_edge = float(regime_state.metadata.get("minimum_net_edge_rupees") or 0.0)
    min_required_edge = max(MIN_NET_EDGE_RUPEES, playbook_min_edge, expected_round_trip_cost_rupees * MIN_NET_EDGE_COST_MULTIPLE)
    if expected_net_edge_rupees < min_required_edge:
        return None
    return build_trade_decision(
        structure=structure,
        regime=regime_state.regime,
        rationale=list(decision.rationale)
        + [
            "Shadow-only subtype BEARISH_FAILED_BREAKOUT_TRANSITION would relax the missing 5m trigger inside research mode only.",
        ]
        + structure_reasons,
        confidence_score=max(decision.confidence_score, regime_state.confidence),
        entry_time=snapshot.timestamp,
        lots=risk.lots,
        lot_size=snapshot.lot_size,
        max_loss_rupees_per_lot=risk.max_loss_rupees_per_lot,
        slippage_points=snapshot.slippage_points,
        extra_metadata={
            "day_archetype": regime_state.metadata.get("day_archetype"),
            "playbook": regime_state.metadata.get("playbook"),
            "setup_subtype": regime_state.metadata.get("setup_subtype"),
            "bullish_shadow_subtype": regime_state.metadata.get("bullish_shadow_subtype"),
            "bearish_shadow_subtype": regime_state.metadata.get("bearish_shadow_subtype"),
            "bearish_family": regime_state.metadata.get("bearish_family"),
            "bearish_subtype": regime_state.metadata.get("bearish_subtype"),
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
            "setup_quality_score": regime_state.metadata.get("setup_quality_score"),
            "setup_direction": regime_state.metadata.get("setup_direction"),
            "playbook_tier": "B",
            "regime_tradability": "TRADABLE",
            "shadow_only_subtype": "BEARISH_FAILED_BREAKOUT_TRANSITION",
            "shadow_only_reason": "RELAXED_5M_TRIGGER",
            "structure_report": structure_report,
        },
    )


def _structural_entry_quality_check(
    metadata: dict,
    direction: str,  # "BULLISH" or "BEARISH"
) -> tuple[bool, str]:
    """
    Gate both bullish and bearish paper overrides against structural quality:
    ORB break, BOS/CHoCH, impulse strength, option chain confirmation, OI pressure.
    Returns (passed, block_reason).
    """
    orb_state = str(metadata.get("opening_range_break_state") or "NONE")
    bullish_bos = bool(metadata.get("bullish_bos"))
    bullish_choch = bool(metadata.get("bullish_choch"))
    bearish_bos = bool(metadata.get("bearish_bos"))
    bearish_choch = bool(metadata.get("bearish_choch"))
    impulse = float(metadata.get("impulse_strength") or 0.0)
    price_vs_or_high = float(metadata.get("price_vs_or_high") or 0.0)
    price_vs_or_low = float(metadata.get("price_vs_or_low") or 0.0)
    price_vs_vwap = float(metadata.get("price_vs_vwap") or 0.0)
    bars_above = int(metadata.get("bars_above_vwap") or 0)
    bars_below = int(metadata.get("bars_below_vwap") or 0)
    smart_money = str(metadata.get("smart_money_bias") or "NEUTRAL").upper()
    oi_pressure = str(metadata.get("oi_pressure_bias") or "NEUTRAL").upper()
    option_chain_state = str(metadata.get("option_chain_pressure_state") or "NEUTRAL").upper()
    pcr = float(metadata.get("pcr_by_oi") or 1.0)
    call_wall_migration = str(metadata.get("call_wall_migration") or "NONE").upper()
    put_wall_migration = str(metadata.get("put_wall_migration") or "NONE").upper()
    open_space_up = float(metadata.get("open_space_up") or 0.0)
    open_space_down = float(metadata.get("open_space_down") or 0.0)
    overhead_call_pressure = float(metadata.get("overhead_call_pressure_score") or 0.0)
    downside_put_support = float(metadata.get("downside_put_support_score") or 0.0)
    higher_low = bool(metadata.get("higher_low_confirmed"))
    lower_high = bool(metadata.get("lower_high_confirmed"))

    if direction == "BULLISH":
        # 1. Require at least one structural confirmation: BOS, CHoCH, ORB breakout, or price clearly above OR high
        has_structure = bullish_bos or bullish_choch or orb_state == "BREAKOUT_UP" or price_vs_or_high > 30 or higher_low
        if not has_structure:
            return False, "BULL_NO_STRUCTURE_CONFIRMATION"

        # 2. Reject if price is below VWAP and most bars are below — market is not bullish enough
        if price_vs_vwap < -15 and bars_below > bars_above:
            return False, "BULL_PRICE_BELOW_VWAP"

        # 3. Reject if smart money is actively bearish (not just neutral)
        if smart_money == "BEARISH":
            return False, "BULL_SMART_MONEY_BEARISH"

        # 4. Reject if OI pressure is strongly bearish
        if oi_pressure == "BEARISH":
            return False, "BULL_OI_PRESSURE_BEARISH"

        # 5. Overhead call wall blocks entry UNLESS price has broken above OR high by meaningful margin
        if option_chain_state == "OVERHEAD_CALL_PRESSURE" and overhead_call_pressure > 1.5 and price_vs_or_high < 50:
            return False, "BULL_OVERHEAD_CALL_WALL_BLOCKING"

        # 6. Require meaningful open space above for put spread to be valid
        if open_space_up < 30:
            return False, "BULL_INSUFFICIENT_UPSIDE_ROOM"

        # 7. Require at least minimal impulse. Relax threshold when both BOS + CHoCH confirmed (doubly proven structure).
        _impulse_min = 1.0 if (bullish_bos and bullish_choch) else 1.5
        if impulse < _impulse_min:
            return False, "BULL_IMPULSE_TOO_WEAK"

        # 8. PCR sanity: very high PCR (> 1.6) signals heavy put loading = bearish positioning
        if pcr > 1.6:
            return False, "BULL_PCR_EXCESSIVE_PUT_LOADING"

        return True, "STRUCTURAL_GATE_PASSED"

    else:  # BEARISH
        # 1. Require at least one structural confirmation: BOS, CHoCH, ORB breakdown, or price clearly below OR low
        has_structure = bearish_bos or bearish_choch or orb_state == "BREAKDOWN" or price_vs_or_low < -30 or lower_high
        if not has_structure:
            return False, "BEAR_NO_STRUCTURE_CONFIRMATION"

        # 2. Reject if price is above VWAP and most bars are above — market is not bearish enough
        if price_vs_vwap > 15 and bars_above > bars_below:
            return False, "BEAR_PRICE_ABOVE_VWAP"

        # 3. Reject if smart money is actively bullish
        if smart_money == "BULLISH":
            return False, "BEAR_SMART_MONEY_BULLISH"

        # 4. Reject if OI pressure is strongly bullish
        if oi_pressure == "BULLISH":
            return False, "BEAR_OI_PRESSURE_BULLISH"

        # 5. Strong put support below blocks bearish entry UNLESS price has broken down through OR low
        if option_chain_state == "DOWNSIDE_PUT_SUPPORT" and downside_put_support > 1.5 and price_vs_or_low > -50:
            return False, "BEAR_PUT_WALL_BLOCKING"

        # 6. Require meaningful open space below for call spread to be valid
        if open_space_down < 30:
            return False, "BEAR_INSUFFICIENT_DOWNSIDE_ROOM"

        # 7. Require at least minimal impulse. Relax when both BOS + CHoCH confirmed.
        _impulse_min = 1.0 if (bearish_bos and bearish_choch) else 1.5
        if impulse < _impulse_min:
            return False, "BEAR_IMPULSE_TOO_WEAK"

        # 8. PCR sanity: very low PCR (< 0.5) signals heavy call loading = bullish positioning; wrong for bear trade
        if pcr < 0.5:
            return False, "BEAR_PCR_EXCESSIVE_CALL_LOADING"

        return True, "STRUCTURAL_GATE_PASSED"


def _paper_context_override_candidate(
    *,
    snapshot: MarketSnapshot,
    agent: "IntradayDefinedRiskAgent",
    decision: DecisionOutput,
    runtime_config: object,
) -> tuple[DecisionOutput | None, dict[str, object]]:
    metadata = decision.metadata if isinstance(decision.metadata, dict) else {}
    funnel = metadata.get("trade_funnel") if isinstance(metadata.get("trade_funnel"), dict) else {}
    market_state = str(metadata.get("market_state") or funnel.get("market_state") or "")
    tradability_class = str(metadata.get("tradability_class") or funnel.get("tradability_class") or metadata.get("regime_tradability") or "")
    failure_type = str(metadata.get("failure_type") or funnel.get("failure_type") or "")
    bearish_score = float(metadata.get("bearish_trade_score") or funnel.get("bearish_trade_score") or 0.0)
    bullish_score = float(metadata.get("bullish_trade_score") or funnel.get("bullish_trade_score") or 0.0)
    no_trade_score = float(metadata.get("no_trade_score") or funnel.get("no_trade_score") or 0.0)
    setup_direction = str(metadata.get("setup_direction") or funnel.get("setup_direction") or "").upper()
    state_bias = str(metadata.get("market_state_bias") or funnel.get("market_state_bias") or "").upper()
    v83_reason = _decision_block_reason(decision)
    info: dict[str, object] = {
        "paper_candidate_decision": "NO_TRADE",
        "paper_candidate_reason": "OVERRIDE_NOT_ELIGIBLE",
        "mapped_strategy": StrategyType.BEAR_CALL_CREDIT_SPREAD.value,
        "spread_construction_status": "NOT_ATTEMPTED",
        "would_create_paper_trade": False,
        "paper_experiment_window_allows": False,
    }

    if getattr(runtime_config, "mode", None) != RuntimeMode.PAPER_LIVE:
        info["paper_candidate_reason"] = "MODE_NOT_PAPER_LIVE"
        return None, info
    if decision.action == "TRADE":
        info.update({
            "paper_candidate_decision": "TRADE",
            "paper_candidate_reason": "V83_APPROVED",
            "spread_construction_status": "PASSED",
            "would_create_paper_trade": True,
        })
        return None, info

    start_time = getattr(runtime_config, "allowed_entry_start_hhmm", "09:30")
    end_time = getattr(runtime_config, "paper_experiment_entry_end_hhmm", "14:30")
    start_t = datetime.strptime(start_time, "%H:%M").time()
    end_t = datetime.strptime(end_time, "%H:%M").time()
    inside_window = start_t <= snapshot.timestamp.time() <= end_t
    info["paper_experiment_window_allows"] = inside_window

    # Route to bullish override path when market is clearly bullish
    _bullish_override_threshold = float(getattr(runtime_config, "paper_override_bullish_score_threshold",
                                                getattr(runtime_config, "paper_override_bearish_score_threshold", 5.0)))
    _bullish_override_margin = float(getattr(runtime_config, "paper_override_margin", 0.8))
    _is_bullish_override_candidate = (
        state_bias == "BULLISH"
        and market_state in {"TREND_UP", "TRANSITION", "DIRECTIONAL_BALANCE"}
        and bullish_score >= _bullish_override_threshold
        and bullish_score > no_trade_score + _bullish_override_margin
        and inside_window
    )
    if _is_bullish_override_candidate:
        return _build_bullish_paper_override(
            snapshot=snapshot, agent=agent, decision=decision,
            runtime_config=runtime_config, regime_state=None,
            bullish_score=bullish_score, no_trade_score=no_trade_score,
            v83_reason=v83_reason, info=info,
        )

    if market_state not in {"TRANSITION", "TREND_DOWN", "DIRECTIONAL_BALANCE"}:
        info["paper_candidate_reason"] = "MARKET_STATE_NOT_PAPER_OVERRIDE_APPROVED"
        return None, info
    if bearish_score < float(getattr(runtime_config, "paper_override_bearish_score_threshold", 5.0)):
        info["paper_candidate_reason"] = "PAPER_OVERRIDE_SCORE_TOO_LOW"
        return None, info
    if bearish_score <= no_trade_score + float(getattr(runtime_config, "paper_override_margin", 0.8)):
        info["paper_candidate_reason"] = "PAPER_OVERRIDE_MARGIN_FAIL"
        return None, info
    if not inside_window:
        info["paper_candidate_reason"] = "PAPER_EXPERIMENT_WINDOW_CLOSED"
        return None, info

    # Structural quality gate: require ORB break / BOS / CHoCH / option chain alignment
    _struct_passed, _struct_reason = _structural_entry_quality_check(metadata, "BEARISH")
    if not _struct_passed:
        info["paper_candidate_reason"] = _struct_reason
        return None, info

    regime_state = classify_regime(snapshot, agent.parameters)
    regime_metadata = regime_state.metadata if isinstance(regime_state.metadata, dict) else {}
    playbook = str(regime_metadata.get("playbook") or metadata.get("playbook") or funnel.get("playbook") or "UNKNOWN")
    entry_context_allowed, entry_context_reason = validate_entry_context(StrategyType.BEAR_CALL_CREDIT_SPREAD, snapshot, regime_state)
    if not entry_context_allowed:
        info["paper_candidate_reason"] = str(entry_context_reason or "ENTRY_CONTEXT_REJECTED")
        info["spread_construction_status"] = str(entry_context_reason or "ENTRY_CONTEXT_REJECTED")
        return None, info
    structure, structure_reasons, structure_report = select_best_structure(
        StrategyType.BEAR_CALL_CREDIT_SPREAD,
        snapshot,
        regime_state,
        agent.parameters,
        setup_quality_score=float(regime_metadata.get("setup_quality_score") or metadata.get("setup_quality_score") or funnel.get("setup_quality_score") or 0.0),
        playbook_tier=playbook_tier(playbook),
    )
    info["spread_construction_status"] = (
        "PASSED"
        if structure is not None
        else str(structure_report.get("canonical_rejection_reason") or "SPREAD_CONSTRUCTION_FAILED")
    )
    if structure is None:
        info["paper_candidate_reason"] = str(structure_report.get("canonical_rejection_reason") or "SPREAD_CONSTRUCTION_FAILED")
        return None, info

    risk = assess_trade_risk(
        structure=structure,
        lot_size=snapshot.lot_size,
        risk_limits=snapshot.risk_limits,
        account_state=snapshot.account_state,
        margin_estimate_per_lot=structure.margin_estimate_per_lot,
        playbook=str(regime_state.metadata.get("playbook") or ""),
    )
    if not risk.allowed:
        info["paper_candidate_reason"] = "RISK_LIMIT"
        info["spread_construction_status"] = "PASSED_RISK_BLOCKED"
        return None, info

    # Clamp lots to the ops-level max before building the decision so the override gate never hits LOTS_EXCEED_RUNTIME_LIMIT.
    ops_max_lots = int(getattr(getattr(runtime_config, "risk", None), "max_lots_per_trade", None) or 6)
    capped_lots = max(1, min(risk.lots, ops_max_lots))

    expected_credit_rupees = max(structure.credit_points - snapshot.slippage_points, 0.0) * snapshot.lot_size * capped_lots
    expected_round_trip_cost_rupees = ESTIMATED_ROUND_TRIP_COST_RUPEES_PER_LOT * risk.lots
    expected_net_edge_rupees = expected_credit_rupees - expected_round_trip_cost_rupees
    playbook_min_edge = float(regime_metadata.get("minimum_net_edge_rupees") or 0.0)
    min_required_edge = max(MIN_NET_EDGE_RUPEES, playbook_min_edge, expected_round_trip_cost_rupees * MIN_NET_EDGE_COST_MULTIPLE)
    if expected_net_edge_rupees < min_required_edge:
        info["paper_candidate_reason"] = "NET_EDGE_TOO_LOW"
        info["spread_construction_status"] = "PASSED_NET_EDGE_BLOCKED"
        return None, info

    info.update({
        "paper_candidate_decision": "TRADE",
        "paper_candidate_reason": "PAPER_CONTEXT_OVERRIDE",
        "would_create_paper_trade": True,
    })
    return build_trade_decision(
        structure=structure,
        regime=regime_state.regime,
        rationale=list(decision.rationale)
        + [
            f"Paper-only override converted strict v83 rejection {v83_reason} into a bearish candidate for learning.",
            "5m trigger requirements remain unchanged in v83 and MICRO_LIVE; this override exists only in PAPER_LIVE.",
        ]
        + structure_reasons,
        confidence_score=max(decision.confidence_score, regime_state.confidence),
        entry_time=snapshot.timestamp,
        lots=capped_lots,
        lot_size=snapshot.lot_size,
        max_loss_rupees_per_lot=risk.max_loss_rupees_per_lot,
        slippage_points=snapshot.slippage_points,
        extra_metadata={
            "day_archetype": regime_metadata.get("day_archetype"),
            "playbook": playbook,
            "setup_subtype": regime_metadata.get("setup_subtype") or metadata.get("setup_subtype"),
            "bullish_shadow_subtype": regime_metadata.get("bullish_shadow_subtype"),
            "bearish_shadow_subtype": regime_metadata.get("bearish_shadow_subtype"),
            "bearish_family": regime_metadata.get("bearish_family"),
            "bearish_subtype": regime_metadata.get("bearish_subtype"),
            "market_state": regime_metadata.get("market_state"),
            "market_state_bias": regime_metadata.get("market_state_bias"),
            "state_quality_score": regime_metadata.get("state_quality_score"),
            "state_confidence_score": regime_metadata.get("state_confidence_score"),
            "tradability_class": regime_metadata.get("tradability_class"),
            "failure_type": regime_metadata.get("failure_type"),
            "option_chain_pressure_state": regime_metadata.get("option_chain_pressure_state"),
            "market_state_score": regime_metadata.get("market_state_score"),
            "trend_quality_score": regime_metadata.get("trend_quality_score"),
            "failure_score": regime_metadata.get("failure_score"),
            "location_score": regime_metadata.get("location_score"),
            "option_chain_pressure_score": regime_metadata.get("option_chain_pressure_score"),
            "live_monetization_score": regime_metadata.get("monetization_score"),
            "tradability_score": regime_metadata.get("tradability_score"),
            "bearish_trade_score": regime_metadata.get("bearish_trade_score"),
            "bullish_trade_score": regime_metadata.get("bullish_trade_score"),
            "no_trade_score": regime_metadata.get("no_trade_score"),
            "setup_quality_score": regime_metadata.get("setup_quality_score"),
            "setup_direction": "BEARISH",
            "playbook_tier": "PAPER_EXPERIMENT",
            "regime_tradability": "TRADABLE",
            "paper_trade_attribution": "PAPER_CONTEXT_OVERRIDE",
            "paper_candidate_reason": "PAPER_CONTEXT_OVERRIDE",
            "v83_rejection_reason": v83_reason,
            "paper_override_source_playbook": metadata.get("playbook") or funnel.get("playbook"),
            "paper_override_source_subtype": metadata.get("setup_subtype") or metadata.get("bearish_subtype"),
            "paper_override_source_failure_type": failure_type,
            "paper_override_source_market_state": market_state,
            "structure_report": structure_report,
            "ml_bearish_entry_score": regime_metadata.get("bearish_entry_score"),
            "ml_bullish_entry_score": regime_metadata.get("bullish_entry_score"),
            "ml_india_vix": regime_metadata.get("india_vix"),
            "ml_session_phase": regime_metadata.get("session_phase"),
            "ml_vwap_reclaim": regime_metadata.get("vwap_reclaim"),
            "ml_banknifty_divergence": regime_metadata.get("banknifty_divergence"),
            "ml_days_to_monthly_expiry": regime_metadata.get("days_to_monthly_expiry"),
            "ml_bullish_momentum": regime_metadata.get("bullish_momentum_persistence"),
            "ml_bearish_momentum": regime_metadata.get("bearish_momentum_persistence"),
            "ml_rv30_pct": regime_state.rv30_pct,
            "ml_order_flow_imbalance": regime_metadata.get("order_flow_imbalance"),
        },
    ), info


def _build_bullish_paper_override(
    *,
    snapshot: "MarketSnapshot",
    agent: "IntradayDefinedRiskAgent",
    decision: "DecisionOutput",
    runtime_config: object,
    regime_state: object,
    bullish_score: float,
    no_trade_score: float,
    v83_reason: str,
    info: dict,
) -> "tuple[DecisionOutput | None, dict]":
    metadata = decision.metadata if isinstance(decision.metadata, dict) else {}
    funnel = metadata.get("trade_funnel") if isinstance(metadata.get("trade_funnel"), dict) else {}
    failure_type = str(metadata.get("failure_type") or funnel.get("failure_type") or "")
    market_state = str(metadata.get("market_state") or funnel.get("market_state") or "")
    playbook = str(metadata.get("playbook") or funnel.get("playbook") or "UNKNOWN")

    if regime_state is None:
        from .regime import classify_regime
        regime_state = classify_regime(snapshot, agent.parameters)
    regime_metadata = regime_state.metadata if isinstance(regime_state.metadata, dict) else {}

    # Structural quality gate: require ORB break / BOS / CHoCH / option chain alignment
    _combined_meta = {**metadata, **regime_metadata}
    _struct_passed, _struct_reason = _structural_entry_quality_check(_combined_meta, "BULLISH")
    if not _struct_passed:
        info["paper_candidate_reason"] = _struct_reason
        info["spread_construction_status"] = "STRUCTURAL_GATE_BLOCKED"
        return None, info

    entry_context_allowed, entry_context_reason = validate_entry_context(StrategyType.BULL_PUT_CREDIT_SPREAD, snapshot, regime_state)
    if not entry_context_allowed:
        info["paper_candidate_reason"] = str(entry_context_reason or "ENTRY_CONTEXT_REJECTED")
        info["spread_construction_status"] = str(entry_context_reason or "ENTRY_CONTEXT_REJECTED")
        return None, info

    structure, structure_reasons, structure_report = select_best_structure(
        StrategyType.BULL_PUT_CREDIT_SPREAD,
        snapshot,
        regime_state,
        agent.parameters,
        setup_quality_score=float(regime_metadata.get("setup_quality_score") or metadata.get("setup_quality_score") or funnel.get("setup_quality_score") or 0.0),
        playbook_tier=playbook_tier(playbook),
    )
    info["spread_construction_status"] = (
        "PASSED"
        if structure is not None
        else str(structure_report.get("canonical_rejection_reason") or "SPREAD_CONSTRUCTION_FAILED")
    )
    if structure is None:
        info["paper_candidate_reason"] = str(structure_report.get("canonical_rejection_reason") or "SPREAD_CONSTRUCTION_FAILED")
        return None, info

    risk = assess_trade_risk(
        structure=structure,
        lot_size=snapshot.lot_size,
        risk_limits=snapshot.risk_limits,
        account_state=snapshot.account_state,
        margin_estimate_per_lot=structure.margin_estimate_per_lot,
        playbook=str(regime_state.metadata.get("playbook") or ""),
    )
    if not risk.allowed:
        info["paper_candidate_reason"] = "RISK_LIMIT"
        info["spread_construction_status"] = "PASSED_RISK_BLOCKED"
        return None, info

    ops_max_lots = int(getattr(getattr(runtime_config, "risk", None), "max_lots_per_trade", None) or 6)
    capped_lots = max(1, min(risk.lots, ops_max_lots))

    expected_credit_rupees = max(structure.credit_points - snapshot.slippage_points, 0.0) * snapshot.lot_size * capped_lots
    expected_round_trip_cost_rupees = ESTIMATED_ROUND_TRIP_COST_RUPEES_PER_LOT * risk.lots
    expected_net_edge_rupees = expected_credit_rupees - expected_round_trip_cost_rupees
    playbook_min_edge = float(regime_metadata.get("minimum_net_edge_rupees") or 0.0)
    min_required_edge = max(MIN_NET_EDGE_RUPEES, playbook_min_edge, expected_round_trip_cost_rupees * MIN_NET_EDGE_COST_MULTIPLE)
    if expected_net_edge_rupees < min_required_edge:
        info["paper_candidate_reason"] = "NET_EDGE_TOO_LOW"
        info["spread_construction_status"] = "PASSED_NET_EDGE_BLOCKED"
        return None, info

    info.update({
        "paper_candidate_decision": "TRADE",
        "paper_candidate_reason": "PAPER_CONTEXT_OVERRIDE_BULLISH",
        "mapped_strategy": StrategyType.BULL_PUT_CREDIT_SPREAD.value,
        "would_create_paper_trade": True,
    })
    return build_trade_decision(
        structure=structure,
        regime=regime_state.regime,
        rationale=list(decision.rationale)
        + [
            f"Bullish paper override converted v83 rejection ({v83_reason}) into a bullish candidate for learning.",
            "5m trigger requirements remain unchanged in v83 and MICRO_LIVE; this override exists only in PAPER_LIVE.",
        ]
        + structure_reasons,
        confidence_score=max(decision.confidence_score, regime_state.confidence),
        entry_time=snapshot.timestamp,
        lots=capped_lots,
        lot_size=snapshot.lot_size,
        max_loss_rupees_per_lot=risk.max_loss_rupees_per_lot,
        slippage_points=snapshot.slippage_points,
        extra_metadata={
            "day_archetype": regime_metadata.get("day_archetype"),
            "playbook": playbook,
            "setup_subtype": regime_metadata.get("setup_subtype") or metadata.get("setup_subtype"),
            "bullish_shadow_subtype": regime_metadata.get("bullish_shadow_subtype"),
            "market_state": regime_metadata.get("market_state"),
            "market_state_bias": regime_metadata.get("market_state_bias"),
            "state_quality_score": regime_metadata.get("state_quality_score"),
            "tradability_class": regime_metadata.get("tradability_class"),
            "option_chain_pressure_state": regime_metadata.get("option_chain_pressure_state"),
            "bearish_trade_score": regime_metadata.get("bearish_trade_score"),
            "bullish_trade_score": regime_metadata.get("bullish_trade_score"),
            "no_trade_score": regime_metadata.get("no_trade_score"),
            "setup_quality_score": regime_metadata.get("setup_quality_score"),
            "setup_direction": "BULLISH",
            "playbook_tier": "PAPER_EXPERIMENT",
            "regime_tradability": "TRADABLE",
            "paper_trade_attribution": "PAPER_CONTEXT_OVERRIDE_BULLISH",
            "paper_candidate_reason": "PAPER_CONTEXT_OVERRIDE_BULLISH",
            "v83_rejection_reason": v83_reason,
            "paper_override_source_market_state": market_state,
            "structure_report": structure_report,
            "ml_bearish_entry_score": regime_metadata.get("bearish_entry_score"),
            "ml_bullish_entry_score": regime_metadata.get("bullish_entry_score"),
            "ml_india_vix": regime_metadata.get("india_vix"),
            "ml_session_phase": regime_metadata.get("session_phase"),
            "ml_vwap_reclaim": regime_metadata.get("vwap_reclaim"),
            "ml_banknifty_divergence": regime_metadata.get("banknifty_divergence"),
            "ml_days_to_monthly_expiry": regime_metadata.get("days_to_monthly_expiry"),
            "ml_bullish_momentum": regime_metadata.get("bullish_momentum_persistence"),
            "ml_bearish_momentum": regime_metadata.get("bearish_momentum_persistence"),
            "ml_rv30_pct": regime_state.rv30_pct,
            "ml_order_flow_imbalance": regime_metadata.get("order_flow_imbalance"),
        },
    ), info


class MarketDataProvider(Protocol):
    def current_snapshot(self) -> MarketSnapshot: ...

    def current_structure_quotes(self, position: OpenPosition) -> MarketSnapshot: ...


class BrokerExecutor(Protocol):
    def enter_trade(self, decision: DecisionOutput) -> dict[str, object]: ...

    def exit_trade(self, position: OpenPosition, reason: str) -> dict[str, object]: ...


def _broker_positions_from_executor(executor: object) -> list[dict[str, object]] | None:
    for attr in ("get_positions_raw", "get_positions_live", "positions"):
        fn = getattr(executor, attr, None)
        if not callable(fn):
            continue
        try:
            rows = fn()
        except Exception:
            return None
        if isinstance(rows, list):
            return [dict(row) for row in rows if isinstance(row, dict)]
    return None


class IntradayDefinedRiskAgent:
    def __init__(
        self,
        risk_limits: dict[str, float] | None = None,
        learning_store: LearningStore | None = None,
        parameters: AdaptiveParameters | None = None,
        *,
        allowed_playbook_tiers: tuple[str, ...] = ("A", "B"),
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
            "bearish_shadow_subtype": regime_state.metadata.get("bearish_shadow_subtype"),
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
        if self._session_trade_counts[strategy.value] >= 2:
            decision = build_no_trade_decision(
                regime_state.regime.value,
                strategy_reasons + [f"{strategy.value} already traded twice this session; skipping repeat entry."],
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

        context_allowed, context_reason = validate_entry_context(strategy, snapshot, regime_state)
        if not context_allowed:
            decision = build_no_trade_decision(
                regime_state.regime.value,
                strategy_reasons + [f"Entry context rejected: {context_reason}."],
                confidence_score=regime_state.confidence,
                extra_metadata=dict(regime_state.metadata) | {"trade_funnel": trade_funnel | {"canonical_rejection_reason": context_reason}},
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
            playbook=str(regime_state.metadata.get("playbook") or ""),
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
                "bearish_shadow_subtype": regime_state.metadata.get("bearish_shadow_subtype"),
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
                # Phase 8: ML feature snapshot stored at entry for win-probability training
                "ml_bearish_entry_score": regime_state.metadata.get("bearish_entry_score"),
                "ml_bullish_entry_score": regime_state.metadata.get("bullish_entry_score"),
                "ml_india_vix": regime_state.metadata.get("india_vix"),
                "ml_session_phase": regime_state.metadata.get("session_phase"),
                "ml_vwap_reclaim": regime_state.metadata.get("vwap_reclaim"),
                "ml_banknifty_divergence": regime_state.metadata.get("banknifty_divergence"),
                "ml_days_to_monthly_expiry": regime_state.metadata.get("days_to_monthly_expiry"),
                "ml_bullish_momentum": regime_state.metadata.get("bullish_momentum_persistence"),
                "ml_bearish_momentum": regime_state.metadata.get("bearish_momentum_persistence"),
                "ml_rv30_pct": regime_state.rv30_pct,
                "ml_order_flow_imbalance": regime_state.metadata.get("order_flow_imbalance"),
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

        # ── Condor partial close: close only the threatened side, keep the safe half ──
        if exit_decision.reason in {"CONDOR_CALL_SIDE_CLOSE", "CONDOR_PUT_SIDE_CLOSE"}:
            close_option_type = OptionType.CALL if exit_decision.reason == "CONDOR_CALL_SIDE_CLOSE" else OptionType.PUT
            keep_option_type = OptionType.PUT if close_option_type == OptionType.CALL else OptionType.CALL
            surviving_legs = [leg for leg in self.open_position.structure.legs if leg.option_type == keep_option_type]
            if surviving_legs:
                keep_strategy = StrategyType.BULL_PUT_CREDIT_SPREAD if keep_option_type == OptionType.PUT else StrategyType.BEAR_CALL_CREDIT_SPREAD
                threatened_side_cost = exit_decision.current_value_points * 0.5  # approx: each side is half
                surviving_credit = max(self.open_position.entry_credit_points - threatened_side_cost, 0.0)
                surviving_width = min(
                    (abs(surviving_legs[0].strike - surviving_legs[-1].strike) if len(surviving_legs) >= 2 else self.open_position.structure.width_points / 2.0),
                    self.open_position.structure.width_points,
                )
                new_structure = TradeStructure(
                    strategy=keep_strategy,
                    legs=surviving_legs,
                    credit_points=surviving_credit,
                    width_points=surviving_width,
                    rationale=[f"Condor partial close: {close_option_type.value} side closed due to delta breach. Keeping {keep_option_type.value} spread."],
                    metadata={**self.open_position.structure.metadata, "partial_close_from_condor": True},
                )
                old_meta = dict(self.open_position.metadata)
                old_meta["condor_partial_close_reason"] = exit_decision.reason
                old_meta["condor_partial_close_pnl_approx"] = round(exit_decision.pnl_rupees * 0.5, 2)
                self.open_position = OpenPosition(
                    structure=new_structure,
                    lots=self.open_position.lots,
                    lot_size=self.open_position.lot_size,
                    entry_time=self.open_position.entry_time,
                    entry_credit_points=surviving_credit,
                    target_value_points=surviving_credit * 0.35,
                    stop_value_points=surviving_credit * 2.0,
                    max_loss_rupees_per_lot=surviving_width * snapshot.lot_size,
                    take_profit_capture_pct=0.65,
                    metadata=old_meta,
                )
                return None  # Position adjusted, not closed — continue monitoring

        # ── Hedge opportunity: convert directional spread to Iron Condor ──
        if not exit_decision.should_exit:
            hedge = evaluate_hedge_opportunity(
                self.open_position,
                current_regime=current_regime,
                current_snapshot=snapshot,
                now=snapshot.timestamp,
            )
            if hedge["should_hedge"] and not self.open_position.metadata.get("hedge_attempted"):
                self.open_position.metadata["hedge_attempted"] = True
                self.open_position.metadata["hedge_reason"] = hedge["reason"]
                hedge_decision = self._execute_hedge(
                    hedge_side=str(hedge["hedge_side"]),
                    snapshot=snapshot,
                    current_regime=current_regime,
                )
                if hedge_decision is not None:
                    return hedge_decision
            return None

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
            extra_metadata=dict(current_regime.metadata) | {
                "exit_reason": exit_decision.reason,
                "exit_pnl_rupees": round(exit_decision.pnl_rupees, 2),
            },
        )

    def _execute_hedge(
        self,
        hedge_side: str,
        snapshot: MarketSnapshot,
        current_regime: RegimeState,
    ) -> DecisionOutput | None:
        """
        Converts an open directional credit spread into an Iron Condor by adding
        the complementary leg. Finds the optimal strike using select_structure(),
        then merges the new legs into the existing open position.

        Returns a DecisionOutput with action="TRADE" and is_hedge_addition=True
        so the runner layer places only the new hedge leg orders.
        Returns None if no valid hedge structure can be found.
        """
        if self.open_position is None:
            return None

        hedge_strategy = (
            StrategyType.BULL_PUT_CREDIT_SPREAD
            if hedge_side == "BULL_PUT"
            else StrategyType.BEAR_CALL_CREDIT_SPREAD
        )

        # Build a synthetic regime_state that directs strike selection for the hedge leg.
        # Use tighter width and lower edge target — this is insurance, not a primary trade.
        hedge_metadata = dict(current_regime.metadata)
        hedge_metadata["preferred_width_points"] = 75.0
        hedge_metadata["allowed_width_points"] = (75.0, 100.0)
        hedge_metadata["minimum_net_edge_rupees"] = 600.0
        hedge_metadata["playbook"] = (
            "SIDEWAYS_TO_BULLISH_RECLAIM" if hedge_side == "BULL_PUT" else "BEARISH_CONTINUATION"
        )
        hedge_metadata["bearish_entry_ready"] = True
        hedge_metadata["bullish_entry_ready"] = True
        hedge_regime = RegimeState(
            regime=current_regime.regime,
            trend_15m=current_regime.trend_15m,
            execution_5m=current_regime.execution_5m,
            ema20_15m=current_regime.ema20_15m,
            ema20_slope_15m=current_regime.ema20_slope_15m,
            rv30_pct=current_regime.rv30_pct,
            or_length_minutes=current_regime.or_length_minutes,
            opening_range_high=current_regime.opening_range_high,
            opening_range_low=current_regime.opening_range_low,
            vwap=current_regime.vwap,
            confidence=current_regime.confidence,
            reasons=current_regime.reasons,
            metadata=hedge_metadata,
        )

        hedge_structure, _, _ = select_best_structure(
            hedge_strategy,
            snapshot,
            hedge_regime,
            params=self.parameters,
            playbook_tier="A",
        )
        if hedge_structure is None:
            self.open_position.metadata["hedge_attempted"] = True
            self.open_position.metadata["hedge_failed_reason"] = "NO_VALID_STRUCTURE_FOUND"
            return None

        # Merge original legs + hedge legs into a single Iron Condor structure
        original = self.open_position
        merged_legs = list(original.structure.legs) + list(hedge_structure.legs)
        merged_credit = round(original.entry_credit_points + hedge_structure.credit_points, 4)
        merged_width = max(original.structure.width_points, hedge_structure.width_points)
        call_legs = [l for l in merged_legs if l.option_type == OptionType.CALL]
        put_legs = [l for l in merged_legs if l.option_type == OptionType.PUT]
        call_width = abs(call_legs[0].strike - call_legs[-1].strike) if len(call_legs) >= 2 else merged_width
        put_width = abs(put_legs[0].strike - put_legs[-1].strike) if len(put_legs) >= 2 else merged_width

        merged_structure = TradeStructure(
            strategy=StrategyType.IRON_CONDOR,
            legs=merged_legs,
            credit_points=merged_credit,
            width_points=merged_width,
            call_width_points=call_width,
            put_width_points=put_width,
            rationale=list(original.structure.rationale) + [
                f"Hedge leg added: {hedge_side} spread converted position to Iron Condor. "
                f"Original credit {original.entry_credit_points:.1f}pt + hedge {hedge_structure.credit_points:.1f}pt = {merged_credit:.1f}pt total."
            ],
            metadata={**original.structure.metadata, "hedge_addition": True, "hedge_side": hedge_side},
        )

        # Rebuild open position as the merged condor
        merged_max_loss = merged_width * snapshot.lot_size
        merged_meta = dict(original.metadata)
        merged_meta["hedge_executed"] = True
        merged_meta["hedge_side"] = hedge_side
        merged_meta["hedge_credit_pts"] = hedge_structure.credit_points
        self.open_position = OpenPosition(
            structure=merged_structure,
            lots=original.lots,
            lot_size=original.lot_size,
            entry_time=original.entry_time,
            entry_credit_points=merged_credit,
            target_value_points=merged_credit * 0.50,  # condor TP = 50% capture
            stop_value_points=merged_credit * 2.0,
            max_loss_rupees_per_lot=merged_max_loss,
            take_profit_capture_pct=0.50,
            metadata=merged_meta,
        )

        # Return a TRADE decision for only the new hedge legs so the runner places those orders
        return build_trade_decision(
            structure=hedge_structure,
            regime=current_regime.regime,
            rationale=[
                f"Hedge addition: converting {original.structure.strategy.value} to Iron Condor.",
                f"Market stalled with range_balance_score sufficient for range strategy.",
                f"Adding {hedge_side} spread to lock in premium on both sides.",
            ],
            confidence_score=current_regime.confidence,
            entry_time=snapshot.timestamp,
            lots=original.lots,
            lot_size=original.lot_size,
            max_loss_rupees_per_lot=hedge_structure.width_points * snapshot.lot_size,
            slippage_points=snapshot.slippage_points,
            extra_metadata={
                "is_hedge_addition": True,
                "hedge_side": hedge_side,
                "original_strategy": original.structure.strategy.value,
                "playbook": hedge_metadata["playbook"],
            },
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
    runtime_config = load_runtime_config()
    _initial_tiers = tuple(getattr(runtime_config, "allowed_playbook_tiers", ("A", "B")))
    agent = IntradayDefinedRiskAgent(learning_store=learning_store, allowed_playbook_tiers=_initial_tiers)
    if runtime_config.mode == RuntimeMode.PAPER_LIVE:
        agent.open_position = load_paper_position()
    poll_seconds = int(config.get("poll_seconds", 30))
    _ops_poll_counter = 0

    while True:
        _ops_poll_counter += 1
        runtime_config = load_runtime_config()
        # Keep agent's allowed tiers in sync with runtime config (takes effect next evaluate call).
        _tiers = tuple(getattr(runtime_config, "allowed_playbook_tiers", ("A", "B")))
        if agent.allowed_playbook_tiers != _tiers:
            agent.allowed_playbook_tiers = _tiers

        # ── After-hours fast-skip ──────────────────────────────────────────────
        # Between 15:35 and 09:15 IST there is nothing to trade. Sleep 5 minutes
        # instead of 30 seconds so we don't generate ~1 080 wasted NO_TRADE
        # decisions overnight (18 h × 60 decisions/h).
        _ist_now = datetime.now(_IST)
        if not (_MARKET_OPEN_TIME <= _ist_now.time() <= _MARKET_CLOSE_TIME):
            _write_v83_agent_heartbeat(runtime_config=runtime_config, phase="v83_after_hours")
            sleep(300)
            continue
        # ──────────────────────────────────────────────────────────────────────

        # Throttle the expensive status-report build to every 5th poll (~150 s).
        # On trade events / active positions the caller overrides this flag.
        _should_report = (_ops_poll_counter % 5 == 0)

        _write_v83_agent_heartbeat(runtime_config=runtime_config, phase="v83_run_live_loop")
        try:
            snapshot = provider.current_snapshot()
        except Exception as exc:  # noqa: BLE001
            _write_v83_agent_heartbeat(runtime_config=runtime_config, phase="v83_snapshot_unavailable")
            reason_code = str(getattr(exc, "reason_code", "") or "SNAPSHOT_UNAVAILABLE")
            diagnostics = getattr(exc, "diagnostics", None)
            data_status = dict(diagnostics) if isinstance(diagnostics, dict) else {
                "snapshot_status": "UNAVAILABLE",
                "reason": f"{type(exc).__name__}: {exc}",
            }

            # ── EOD force-exit when data is unavailable ──────────────────────
            # If it's past 15:15 and there's an open paper position, close it
            # at last-known value rather than leaving it orphaned overnight.
            _now_ist = datetime.now(_IST)
            _eod_force_exit_time = dtime(15, 15)
            if _now_ist.time() >= _eod_force_exit_time and runtime_config.mode == RuntimeMode.PAPER_LIVE:
                _stuck_position = load_paper_position()
                if _stuck_position is not None:
                    from .ops_runtime import save_paper_position, OpsPaths
                    _paths = OpsPaths()
                    try:
                        import json as _json
                        _state_raw = _paths.paper_state.read_text() if _paths.paper_state.exists() else "{}"
                        _state = _json.loads(_state_raw)
                    except Exception:
                        _state = {}
                    _mfe = float(_state.get("mfe_rupees") or 0.0)
                    _mae = float(_state.get("mae_rupees") or 0.0)
                    _exit_event = {
                        "event": "PAPER_EXIT",
                        "timestamp": _now_ist.isoformat(),
                        "session_date": _now_ist.date().isoformat(),
                        "entry_timestamp": _stuck_position.entry_time.isoformat(),
                        "exit_timestamp": _now_ist.isoformat(),
                        "exit_reason": "EOD_DATA_LOSS_FORCE_EXIT",
                        "playbook": _stuck_position.metadata.get("playbook"),
                        "paper_trade_attribution": _stuck_position.metadata.get("paper_trade_attribution"),
                        "paper_candidate_reason": _stuck_position.metadata.get("paper_candidate_reason"),
                        "strategy": _stuck_position.structure.strategy.value,
                        "realized_paper_pnl": None,
                        "mfe_rupees": round(_mfe, 2),
                        "mae_rupees": round(_mae, 2),
                        "note": "Force-closed at EOD: option chain data unavailable, PnL unknown.",
                    }
                    try:
                        with open(_paths.paper_trades, "a") as _f:
                            _f.write(_json.dumps(_exit_event) + "\n")
                        save_paper_position(None, paths=_paths, extra={"last_exit": _exit_event, "mfe_rupees": 0.0, "mae_rupees": 0.0})
                        agent.open_position = None
                    except Exception:
                        pass
            # ─────────────────────────────────────────────────────────────────

            runtime_state = load_runtime_state(config=runtime_config)
            runtime_state["mode"] = runtime_config.mode.value
            runtime_state["live_arm"] = runtime_config.live_arm
            runtime_state["primary_block_reason"] = reason_code
            runtime_state["last_decision_at"] = datetime.now().isoformat()
            runtime_state["last_candidate"] = {"error": f"{type(exc).__name__}: {exc}", "data_readiness": data_status}
            save_runtime_state(runtime_state)
            decision = build_no_trade_decision(
                RegimeLabel.NO_TRADE.value,
                [f"Live snapshot unavailable: {type(exc).__name__}: {exc}"],
                extra_metadata={
                    "runtime_mode": runtime_config.mode.value,
                    "primary_block_reason": reason_code,
                    "data_readiness": data_status,
                },
            )
            log_validation_decision(
                decision,
                block_reason=reason_code,
                data_status=data_status,
            )
            print(decision.to_json())
            if _should_report:
                build_operator_status_report()
            sleep(poll_seconds)
            continue
        health = build_unified_health(config=runtime_config, snapshot=snapshot)
        broker_positions = _broker_positions_from_executor(executor)
        reconcile_positions(mode=runtime_config.mode, broker_positions=broker_positions)
        if runtime_config.mode == RuntimeMode.LIVE_DISABLED:
            decision = build_no_trade_decision(
                RegimeLabel.NO_TRADE.value,
                ["Runtime mode is LIVE_DISABLED; no shadow, paper, or broker execution is allowed."],
                extra_metadata={"runtime_mode": runtime_config.mode.value, "primary_block_reason": "LIVE_DISABLED"},
            )
            log_validation_decision(decision, snapshot=snapshot, block_reason="LIVE_DISABLED")
            _update_runtime_decision_state(decision, runtime_config=runtime_config, snapshot=snapshot, block_reason="LIVE_DISABLED")
            print(decision.to_json())
            if _should_report:
                build_operator_status_report(snapshot=snapshot, broker_positions=broker_positions)
            sleep(poll_seconds)
            continue
        if runtime_config.mode in {RuntimeMode.RESEARCH}:
            decision = agent.evaluate(snapshot)
            log_validation_decision(decision, snapshot=snapshot, block_reason="RESEARCH_MODE")
            _update_runtime_decision_state(decision, runtime_config=runtime_config, snapshot=snapshot, block_reason="RESEARCH_MODE")
            print(decision.to_json())
            if _should_report:
                build_operator_status_report(snapshot=snapshot, broker_positions=broker_positions)
            sleep(poll_seconds)
            continue
        if runtime_config.mode == RuntimeMode.PAPER_LIVE:
            paper_position = manage_paper_position(snapshot)
            agent.open_position = paper_position
            if paper_position is not None:
                decision = build_no_trade_decision(
                    RegimeLabel.NO_TRADE.value,
                    ["Active paper structure is being managed; new entries are blocked until it exits."],
                    extra_metadata={
                        "runtime_mode": runtime_config.mode.value,
                        "primary_block_reason": "ACTIVE_STRUCTURE_EXISTS",
                        "playbook": paper_position.metadata.get("playbook"),
                        "setup_subtype": paper_position.metadata.get("setup_subtype") or paper_position.metadata.get("bearish_subtype"),
                    },
                )
                log_validation_decision(
                    decision,
                    snapshot=snapshot,
                    block_reason="ACTIVE_STRUCTURE_EXISTS",
                    event_type="POSITION_MANAGEMENT",
                )
                _update_runtime_decision_state(decision, runtime_config=runtime_config, snapshot=snapshot, block_reason="ACTIVE_STRUCTURE_EXISTS")
                build_operator_status_report(snapshot=snapshot, broker_positions=broker_positions)
                sleep(poll_seconds)
                continue

        if agent.open_position:
            current_position = agent.open_position
            position_decision = agent.manage_position(snapshot)
            if position_decision is not None and current_position is not None:
                if position_decision.action == "TRADE" and position_decision.metadata.get("is_hedge_addition"):
                    # Hedge addition: place the new complementary leg orders
                    if runtime_config.mode == RuntimeMode.MICRO_LIVE and runtime_config.live_arm:
                        executor.enter_trade(position_decision)
                        record_runtime_trade_entry(position_decision, snapshot, decision_origin="HEDGE_ADDITION")
                    elif runtime_config.mode == RuntimeMode.PAPER_LIVE:
                        record_paper_entry(
                            agent.open_position,  # already updated to merged condor
                            decision=position_decision,
                            snapshot=snapshot,
                            gate={"allowed": True, "primary_block_reason": "NONE"},
                        )
                    log_validation_decision(
                        position_decision,
                        snapshot=snapshot,
                        block_reason="NONE",
                        actual_trade_created=True,
                        event_type="HEDGE_ENTRY",
                    )
                    _update_runtime_decision_state(position_decision, runtime_config=runtime_config, snapshot=snapshot, block_reason="NONE")
                    print(position_decision.to_json())
                else:
                    # Normal exit decision
                    exit_decision = position_decision
                    reason = exit_decision.rationale[-2] if len(exit_decision.rationale) >= 2 else "EXIT"
                    if runtime_config.mode == RuntimeMode.MICRO_LIVE and runtime_config.live_arm:
                        executor.exit_trade(current_position, reason)
                        record_runtime_trade_exit(
                            current_position,
                            snapshot,
                            pnl_rupees=float(exit_decision.metadata.get("exit_pnl_rupees") or 0.0),
                            exit_reason=str(exit_decision.metadata.get("exit_reason") or reason),
                        )
                    log_validation_decision(exit_decision, snapshot=snapshot, block_reason="EXIT_DECISION", event_type="EXIT")
                    _update_runtime_decision_state(exit_decision, runtime_config=runtime_config, snapshot=snapshot, block_reason="EXIT_DECISION")
                    print(exit_decision.to_json())
        else:
            decision = agent.evaluate(snapshot)
            # Clamp lots to the operator cap *before* any gate check so that
            # LOTS_EXCEED_RUNTIME_LIMIT can never fire.  The risk engine already
            # computed a safe lot count; the runtime cap is a ceiling, not a veto.
            if decision.action == "TRADE":
                ops_max = int(getattr(getattr(runtime_config, "risk", None), "max_lots_per_trade", None) or 6)
                if decision.lots > ops_max:
                    decision.lots = ops_max
            print(decision.to_json())
            if decision.action == "TRADE":
                if runtime_config.mode == RuntimeMode.SHADOW_LIVE:
                    log_shadow_decision(decision, snapshot=snapshot)
                    log_validation_decision(decision, snapshot=snapshot, block_reason="SHADOW_WOULD_TRADE")
                    _update_runtime_decision_state(decision, runtime_config=runtime_config, snapshot=snapshot, block_reason="SHADOW_WOULD_TRADE")
                    runtime_state = load_runtime_state(config=runtime_config)
                    runtime_state["mode"] = runtime_config.mode.value
                    runtime_state["live_arm"] = runtime_config.live_arm
                    runtime_state["primary_block_reason"] = "SHADOW_WOULD_TRADE"
                    runtime_state["last_candidate"] = decision.to_dict()
                    runtime_state["last_decision_at"] = snapshot.timestamp.isoformat()
                    save_runtime_state(runtime_state)
                else:
                    gate = evaluate_entry_gate(decision, snapshot, config=runtime_config, health=health)
                    if gate["allowed"] and runtime_config.mode == RuntimeMode.PAPER_LIVE:
                        position = open_position_from_decision(decision, snapshot)
                        agent.open_position = position
                        record_paper_entry(position, decision=decision, snapshot=snapshot, gate=gate)
                        log_validation_decision(
                            decision,
                            snapshot=snapshot,
                            block_reason="NONE",
                            actual_trade_created=True,
                            decision_origin="V83_APPROVED",
                        )
                        _update_runtime_decision_state(decision, runtime_config=runtime_config, snapshot=snapshot, block_reason="NONE")
                    elif gate["allowed"] and runtime_config.mode == RuntimeMode.MICRO_LIVE and runtime_config.live_arm:
                        executor.enter_trade(decision)
                        agent.start_position(snapshot, decision)
                        record_runtime_trade_entry(decision, snapshot, decision_origin="V83_APPROVED")
                        log_validation_decision(
                            decision,
                            snapshot=snapshot,
                            block_reason="NONE",
                            actual_trade_created=True,
                            decision_origin="V83_APPROVED",
                        )
                        _update_runtime_decision_state(decision, runtime_config=runtime_config, snapshot=snapshot, block_reason="NONE")
                    else:
                        log_shadow_decision(decision, snapshot=snapshot, block_reason=str(gate.get("primary_block_reason") or "ENTRY_GATE_BLOCKED"))
                        log_validation_decision(
                            decision,
                            snapshot=snapshot,
                            block_reason=str(gate.get("primary_block_reason") or "ENTRY_GATE_BLOCKED"),
                        )
                        _update_runtime_decision_state(
                            decision,
                            runtime_config=runtime_config,
                            snapshot=snapshot,
                            block_reason=str(gate.get("primary_block_reason") or "ENTRY_GATE_BLOCKED"),
                        )
                        runtime_state = load_runtime_state(config=runtime_config)
                        runtime_state["mode"] = runtime_config.mode.value
                        runtime_state["live_arm"] = runtime_config.live_arm
                        runtime_state["primary_block_reason"] = gate.get("primary_block_reason")
                        runtime_state["last_candidate"] = decision.to_dict()
                        runtime_state["last_decision_at"] = snapshot.timestamp.isoformat()
                        save_runtime_state(runtime_state)
            elif runtime_config.mode == RuntimeMode.SHADOW_LIVE:
                log_shadow_decision(decision, snapshot=snapshot, block_reason="NO_TRADE_DECISION")
                log_validation_decision(decision, snapshot=snapshot)
                _update_runtime_decision_state(decision, runtime_config=runtime_config, snapshot=snapshot)
            else:
                paper_candidate = None
                if runtime_config.mode == RuntimeMode.PAPER_LIVE:
                    override_decision, override_info = _paper_context_override_candidate(
                        snapshot=snapshot,
                        agent=agent,
                        decision=decision,
                        runtime_config=runtime_config,
                    )
                    paper_candidate = override_info
                    if override_decision is not None:
                        override_gate = evaluate_paper_context_override_gate(
                            override_decision,
                            snapshot,
                            config=runtime_config,
                            health=health,
                        )
                        _override_label = str(override_decision.metadata.get("paper_trade_attribution") or "PAPER_CONTEXT_OVERRIDE") if isinstance(override_decision.metadata, dict) else "PAPER_CONTEXT_OVERRIDE"
                        paper_candidate["paper_candidate_reason"] = str(
                            _override_label if override_gate["allowed"] else override_gate.get("primary_block_reason") or "ENTRY_GATE_BLOCKED"
                        )
                        paper_candidate["paper_experiment_window_allows"] = bool(override_gate.get("inside_paper_experiment_window"))
                        if not override_gate["allowed"]:
                            paper_candidate["paper_candidate_decision"] = "NO_TRADE"
                            paper_candidate["would_create_paper_trade"] = False
                            paper_candidate["spread_construction_status"] = (
                                str(paper_candidate.get("spread_construction_status") or "PASSED")
                                if not str(override_gate.get("primary_block_reason") or "").startswith("PAPER_")
                                else str(override_gate.get("primary_block_reason") or "ENTRY_GATE_BLOCKED")
                            )
                        else:
                            position = open_position_from_decision(override_decision, snapshot)
                            agent.open_position = position
                            record_paper_entry(position, decision=override_decision, snapshot=snapshot, gate=override_gate)
                            log_validation_decision(
                                decision,
                                snapshot=snapshot,
                                paper_candidate=paper_candidate,
                                actual_trade_created=True,
                                decision_origin=_override_label,
                            )
                            _update_runtime_decision_state(
                                override_decision,
                                runtime_config=runtime_config,
                                snapshot=snapshot,
                                block_reason="NONE",
                            )
                            build_operator_status_report(snapshot=snapshot, broker_positions=broker_positions)
                            sleep(poll_seconds)
                            continue
                shadow_candidate = _shadow_only_failed_breakout_candidate(
                    snapshot=snapshot,
                    agent=agent,
                    decision=decision,
                )
                if shadow_candidate is not None:
                    log_shadow_decision(
                        shadow_candidate,
                        snapshot=snapshot,
                        block_reason="SHADOW_ONLY_RESEARCH_SUBTYPE",
                        force_would_trade=True,
                    )
                log_validation_decision(decision, snapshot=snapshot, paper_candidate=paper_candidate)
                _update_runtime_decision_state(decision, runtime_config=runtime_config, snapshot=snapshot)
        # Always build the status report when a position is active so the UI
        # reflects real-time P&L; otherwise throttle to every 5th poll.
        if _should_report or agent.open_position:
            build_operator_status_report(snapshot=snapshot, broker_positions=broker_positions)
        sleep(poll_seconds)


def dump_decision(path: str | Path, decision: DecisionOutput) -> None:
    Path(path).write_text(decision.to_json() + "\n")
