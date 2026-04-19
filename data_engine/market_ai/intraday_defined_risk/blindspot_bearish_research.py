from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import date, datetime, time
from pathlib import Path
from statistics import mean
from typing import Callable

from .backtest import DEFAULT_ENTRY_TIMES
from .data_models import AccountRiskLimits, AdaptiveParameters, MarketSnapshot, OpenPosition, RegimeState
from .frequency_expansion_research import (
    _as_bool,
    _as_float,
    _close_shadow_position,
    _decision_sort_key,
    _group_by_session_date,
    _iter_snapshots,
    _load_json,
    _open_shadow_position,
    _refined_no_trade_reason,
)
from .regime import classify_regime


BLIND_SPOT_REASONS = {"PATTERN_MISMATCH", "NO_VALID_STRATEGY"}


def _metadata(decision: dict[str, object]) -> dict[str, object]:
    return decision.get("metadata") or {}


def _hidden_bearish_family(metadata: dict[str, object]) -> str:
    market_state = str(metadata.get("market_state") or "")
    playbook = str(metadata.get("playbook") or "")
    bearish_setup = str(metadata.get("bearish_setup") or "")
    failure_type = str(metadata.get("failure_type") or "")
    opening_range_break = str(metadata.get("opening_range_break_state") or "")

    if playbook == "BEARISH_FAILED_RECLAIM" or bearish_setup == "FAILED_RECLAIM" or failure_type == "FAILED_RECLAIM":
        return "weak reclaim rejection"
    if opening_range_break == "DOWN" and _as_float(metadata.get("below_vwap_last6")) >= 4.0:
        return "OR-low pressure build"
    if market_state == "DIRECTIONAL_BALANCE" and opening_range_break in {"DOWN", "FAILED_UP"}:
        return "directional balance breakdown"
    if _as_bool(metadata.get("accepted_breakdown")):
        return "post-breakdown retest reject"
    if _as_bool(metadata.get("lower_high_confirmed")):
        return "lower-high rejection"
    if failure_type == "FAILED_BOUNCE" or _as_bool(metadata.get("failed_bounce")):
        return "failed bounce variant"
    if market_state == "TRANSITION" and bearish_setup == "PULLBACK_REJECTION":
        return "transition rejection"
    if str(metadata.get("structure_signal") or "") == "BEARISH_BOS":
        return "post-breakdown retest reject"
    return "other bearish family"


def _quality_pass(regime_state: RegimeState, params: AdaptiveParameters) -> bool:
    metadata = regime_state.metadata
    setup_quality = _as_float(metadata.get("setup_quality_score"))
    bearish_score = _as_float(metadata.get("bearish_trade_score"))
    no_trade_score = _as_float(metadata.get("no_trade_score"))
    return (
        setup_quality >= params.strong_setup_quality_threshold
        and bearish_score >= 5.45
        and bearish_score > no_trade_score + 0.15
        and _as_float(metadata.get("bearish_location_live_score")) >= 1.25
    )


def _base_blindspot_eligibility(ts: datetime, regime_state: RegimeState, params: AdaptiveParameters) -> bool:
    metadata = regime_state.metadata
    playbook = str(metadata.get("playbook") or "")
    return (
        ts.time() < time(14, 0)
        and str(metadata.get("market_state") or "") != "TRUE_RANGE"
        and playbook in {
            "BEARISH_CONTINUATION",
            "BEARISH_FAILED_RECLAIM",
            "HIGH_CONFLUENCE_BEARISH_CONTINUATION",
            "SIDEWAYS_TO_BEARISH_REJECTION",
        }
        and (
            _as_bool(metadata.get("bearish_entry_ready"))
            or str(metadata.get("setup_direction") or "") == "BEARISH"
            or str(metadata.get("bearish_setup") or "") in {"PULLBACK_REJECTION", "FAILED_RECLAIM", "HIGH_CONFLUENCE_CONTINUATION"}
        )
        and _quality_pass(regime_state, params)
    )


BlindspotDetector = Callable[[datetime, RegimeState], bool]


def _is_or_failed_up_lower_high_reject(ts: datetime, regime_state: RegimeState) -> bool:
    metadata = regime_state.metadata
    return (
        str(metadata.get("playbook") or "") in {"BEARISH_CONTINUATION", "HIGH_CONFLUENCE_BEARISH_CONTINUATION"}
        and str(metadata.get("market_state") or "") in {"TRANSITION", "DIRECTIONAL_BALANCE"}
        and (
            str(metadata.get("opening_range_break_state") or "") == "FAILED_UP"
            or str(metadata.get("failure_type") or "") == "FAILED_BREAKOUT"
        )
        and _as_float(metadata.get("price_vs_vwap"), 999.0) < 0.0
        and (_as_bool(metadata.get("lower_high_confirmed")) or _as_float(metadata.get("bearish_candle_quality_score")) >= 3.25)
    )


def _is_accepted_breakdown_retest_reject(ts: datetime, regime_state: RegimeState) -> bool:
    metadata = regime_state.metadata
    return (
        str(metadata.get("playbook") or "") in {"BEARISH_CONTINUATION", "HIGH_CONFLUENCE_BEARISH_CONTINUATION"}
        and str(metadata.get("bearish_setup") or "") in {"PULLBACK_REJECTION", "HIGH_CONFLUENCE_CONTINUATION"}
        and str(metadata.get("market_state") or "") in {"TRANSITION", "DIRECTIONAL_BALANCE", "TREND_DOWN"}
        and _as_float(metadata.get("price_vs_vwap"), 999.0) < 0.0
        and (
            _as_bool(metadata.get("accepted_breakdown"))
            or str(metadata.get("failure_type") or "") == "ACCEPTED_BREAKDOWN"
            or str(metadata.get("structure_signal") or "") == "BEARISH_BOS"
        )
        and str(metadata.get("opening_range_break_state") or "") in {"DOWN", "FAILED_UP", "NONE"}
        and (_as_bool(metadata.get("lower_high_confirmed")) or _as_float(metadata.get("bearish_candle_quality_score")) >= 3.0)
    )


def _is_directional_balance_or_low_pressure_build(ts: datetime, regime_state: RegimeState) -> bool:
    metadata = regime_state.metadata
    return (
        str(metadata.get("market_state") or "") == "DIRECTIONAL_BALANCE"
        and str(metadata.get("opening_range_break_state") or "") == "DOWN"
        and _as_float(metadata.get("below_vwap_last6")) >= 4.0
        and _as_float(metadata.get("price_vs_vwap"), 999.0) < 0.0
        and (
            _as_bool(metadata.get("lower_high_confirmed"))
            or _as_bool(metadata.get("accepted_breakdown"))
            or _as_float(metadata.get("bearish_candle_quality_score")) >= 3.2
        )
        and str(metadata.get("failure_type") or "") not in {
            "FAILED_BOUNCE",
            "BEARISH_REJECTION_TRANSITION",
            "MIDDAY_WEAK_RECLAIM_REJECTION",
            "BEARISH_TRANSITION_REJECTION",
        }
    )


NEW_BLINDSPOT_BEARISH_SUBTYPES: dict[str, BlindspotDetector] = {
    "OR_FAILED_UP_LOWER_HIGH_REJECT": _is_or_failed_up_lower_high_reject,
    "ACCEPTED_BREAKDOWN_RETEST_REJECT": _is_accepted_breakdown_retest_reject,
    "DIRECTIONAL_BALANCE_OR_LOW_PRESSURE_BUILD": _is_directional_balance_or_low_pressure_build,
}


def _blindspot_day_rows(full: dict[str, object], params: AdaptiveParameters) -> list[dict[str, object]]:
    decisions = full.get("decisions", [])
    trades = full.get("trades", [])
    live_trade_days = {str(trade.get("entry_timestamp") or "")[:10] for trade in trades}
    decision_days = _group_by_session_date(decisions, trade=False)
    rows: list[dict[str, object]] = []
    for session_date in sorted(decision_days):
        if session_date in live_trade_days:
            continue
        day_decisions = decision_days.get(session_date, [])
        bearish_decisions = [
            decision for decision in day_decisions
            if _as_bool(_metadata(decision).get("bearish_entry_ready"))
            or str(_metadata(decision).get("setup_direction") or "") == "BEARISH"
        ]
        if not bearish_decisions:
            continue
        best = max(bearish_decisions, key=_decision_sort_key)
        refined_reason = _refined_no_trade_reason(best, params)
        if refined_reason not in BLIND_SPOT_REASONS:
            continue
        metadata = _metadata(best)
        rows.append(
            {
                "date": session_date,
                "refined_block_reason": refined_reason,
                "hidden_bearish_family": _hidden_bearish_family(metadata),
                "playbook": metadata.get("playbook"),
                "bearish_setup": metadata.get("bearish_setup"),
                "market_state": metadata.get("market_state"),
                "failure_type": metadata.get("failure_type"),
                "setup_quality_score": round(_as_float(metadata.get("setup_quality_score")), 4),
                "bearish_trade_score": round(_as_float(metadata.get("bearish_trade_score")), 4),
                "no_trade_score": round(_as_float(metadata.get("no_trade_score")), 4),
                "opening_range_break_state": metadata.get("opening_range_break_state"),
                "price_vs_vwap": round(_as_float(metadata.get("price_vs_vwap")), 4),
                "lower_high_confirmed": _as_bool(metadata.get("lower_high_confirmed")),
                "accepted_breakdown": _as_bool(metadata.get("accepted_breakdown")),
                "structure_signal": metadata.get("structure_signal"),
            }
        )
    return rows


def _close_positions_for_day(
    per_subtype: dict[str, dict[str, object]],
    open_positions: dict[str, OpenPosition | None],
    last_snapshots: dict[str, MarketSnapshot | None],
    params: AdaptiveParameters,
) -> None:
    for subtype, position in open_positions.items():
        if position is None or last_snapshots[subtype] is None:
            continue
        closed = _close_shadow_position(position, last_snapshots[subtype], params, force_time_exit=True)
        if closed is not None:
            per_subtype[subtype]["trades"].append(closed)
        open_positions[subtype] = None


def _blindspot_shadow_report(
    data_path: str | Path,
    *,
    start: str,
    end: str,
    live_full: dict[str, object],
    blindspot_rows: list[dict[str, object]],
    params: AdaptiveParameters,
) -> dict[str, object]:
    blindspot_dates = {str(row["date"]) for row in blindspot_rows}
    live_trade_days = {str(trade.get("entry_timestamp") or "")[:10] for trade in live_full.get("trades", [])}
    risk_limits = AccountRiskLimits(
        max_risk_rupees_per_trade=20000,
        max_margin_rupees=200000,
        max_daily_loss_rupees=10000,
    )
    per_subtype: dict[str, dict[str, object]] = {
        name: {"setups_detected": 0, "setups_passing_quality": 0, "trades": [], "trade_days": set()}
        for name in NEW_BLINDSPOT_BEARISH_SUBTYPES
    }
    open_positions: dict[str, OpenPosition | None] = {name: None for name in NEW_BLINDSPOT_BEARISH_SUBTYPES}
    last_snapshots: dict[str, MarketSnapshot | None] = {name: None for name in NEW_BLINDSPOT_BEARISH_SUBTYPES}
    traded_days: dict[str, set[str]] = {name: set() for name in NEW_BLINDSPOT_BEARISH_SUBTYPES}
    current_day: date | None = None

    for ts, snapshot in _iter_snapshots(data_path, start=start, end=end, risk_limits=risk_limits):
        if current_day is None:
            current_day = ts.date()
        elif ts.date() != current_day:
            _close_positions_for_day(per_subtype, open_positions, last_snapshots, params)
            current_day = ts.date()
        session_date = ts.date().isoformat()
        if session_date not in blindspot_dates:
            continue
        regime_state = classify_regime(snapshot, params)
        for subtype, detector in NEW_BLINDSPOT_BEARISH_SUBTYPES.items():
            if open_positions[subtype] is not None:
                closed = _close_shadow_position(open_positions[subtype], snapshot, params)
                if closed is not None:
                    per_subtype[subtype]["trades"].append(closed)
                    open_positions[subtype] = None
                last_snapshots[subtype] = snapshot
                continue
            if ts.strftime("%H:%M") not in DEFAULT_ENTRY_TIMES or session_date in traded_days[subtype]:
                last_snapshots[subtype] = snapshot
                continue
            if not _base_blindspot_eligibility(ts, regime_state, params):
                last_snapshots[subtype] = snapshot
                continue
            if not detector(ts, regime_state):
                last_snapshots[subtype] = snapshot
                continue
            per_subtype[subtype]["setups_detected"] += 1
            if not _quality_pass(regime_state, params):
                last_snapshots[subtype] = snapshot
                continue
            per_subtype[subtype]["setups_passing_quality"] += 1
            shadow_regime = replace(
                regime_state,
                metadata=dict(regime_state.metadata)
                | {
                    "bearish_family": "BLINDSPOT_BEARISH_RESEARCH",
                    "bearish_subtype": subtype,
                    "setup_subtype": subtype,
                    "failure_type": subtype,
                },
            )
            position = _open_shadow_position(snapshot, shadow_regime, params, subtype)
            if position is None:
                last_snapshots[subtype] = snapshot
                continue
            open_positions[subtype] = position
            traded_days[subtype].add(session_date)
            per_subtype[subtype]["trade_days"].add(session_date)
            last_snapshots[subtype] = snapshot

    _close_positions_for_day(per_subtype, open_positions, last_snapshots, params)

    rows = []
    for subtype, payload in per_subtype.items():
        trades = payload["trades"]
        pnl_values = [_as_float(trade.get("pnl_rupees")) for trade in trades]
        wins = sum(1 for value in pnl_values if value > 0.0)
        losses = sum(1 for value in pnl_values if value < 0.0)
        trade_days = set(payload["trade_days"])
        overlap_days = sorted(trade_days & live_trade_days)
        new_days = sorted(trade_days - live_trade_days)
        rows.append(
            {
                "subtype": subtype,
                "detection_conditions": _subtype_conditions(subtype),
                "setups_detected": payload["setups_detected"],
                "setups_passing_quality": payload["setups_passing_quality"],
                "shadow_trades": len(trades),
                "wins": wins,
                "losses": losses,
                "realized_pnl_rupees": round(sum(pnl_values), 2),
                "average_pnl_rupees": round(mean(pnl_values), 2) if pnl_values else 0.0,
                "unique_added_trade_days": len(new_days),
                "new_trade_days": new_days,
                "overlap_with_current_live_trades": len(overlap_days),
                "overlap_days": overlap_days,
                "improves_frequency_without_weak_flow": bool(new_days and losses == 0 and sum(pnl_values) > 0.0),
                "sample_trades": trades[:8],
                "all_trades": trades,
            }
        )
    rows.sort(key=lambda row: (row["realized_pnl_rupees"], row["unique_added_trade_days"], -row["losses"]), reverse=True)
    return {
        "rows": rows,
        "recommended_next_test": _recommend_subtype(rows),
    }


def _subtype_conditions(subtype: str) -> list[str]:
    mapping = {
        "OR_FAILED_UP_LOWER_HIGH_REJECT": [
            "playbook in BEARISH_CONTINUATION/HIGH_CONFLUENCE_BEARISH_CONTINUATION",
            "market_state in TRANSITION/DIRECTIONAL_BALANCE",
            "opening range failed upward or failure_type == FAILED_BREAKOUT",
            "price below VWAP",
            "lower-high confirmed or bearish candle quality >= 3.25",
        ],
        "ACCEPTED_BREAKDOWN_RETEST_REJECT": [
            "playbook in BEARISH_CONTINUATION/HIGH_CONFLUENCE_BEARISH_CONTINUATION",
            "bearish_setup in PULLBACK_REJECTION/HIGH_CONFLUENCE_CONTINUATION",
            "market_state in TRANSITION/DIRECTIONAL_BALANCE/TREND_DOWN",
            "price below VWAP",
            "accepted breakdown or bearish BOS",
            "lower-high confirmed or bearish candle quality >= 3.0",
        ],
        "DIRECTIONAL_BALANCE_OR_LOW_PRESSURE_BUILD": [
            "market_state == DIRECTIONAL_BALANCE",
            "opening range break state == DOWN",
            "at least four of last six bars below VWAP",
            "price below VWAP",
            "lower-high confirmed, accepted breakdown, or bearish candle quality >= 3.2",
            "not already captured by failed-bounce/rejection-transition subtypes",
        ],
    }
    return mapping.get(subtype, [])


def _recommend_subtype(rows: list[dict[str, object]]) -> str | None:
    valid = [
        row for row in rows
        if int(row.get("shadow_trades") or 0) > 0
        and float(row.get("realized_pnl_rupees") or 0.0) > 0.0
        and int(row.get("losses") or 0) == 0
        and int(row.get("unique_added_trade_days") or 0) > 0
    ]
    if valid:
        return max(valid, key=lambda row: (row["unique_added_trade_days"], row["average_pnl_rupees"]))["subtype"]
    positive = [
        row for row in rows
        if int(row.get("shadow_trades") or 0) > 0 and float(row.get("realized_pnl_rupees") or 0.0) > 0.0
    ]
    if positive:
        return max(positive, key=lambda row: (row["realized_pnl_rupees"], -row["losses"]))["subtype"]
    return None


def _directional_balance_followup(full: dict[str, object], blindspot_rows: list[dict[str, object]]) -> dict[str, object]:
    decisions = full.get("decisions", [])
    live_trade_days = {str(trade.get("entry_timestamp") or "")[:10] for trade in full.get("trades", [])}
    excluded_failures = {
        "FAILED_BOUNCE",
        "BEARISH_REJECTION_TRANSITION",
        "MIDDAY_WEAK_RECLAIM_REJECTION",
        "BEARISH_TRANSITION_REJECTION",
    }
    diagnostics: dict[str, set[str]] = defaultdict(set)
    for decision in decisions:
        session_date = str(decision.get("session_date") or "")
        if not session_date or session_date in live_trade_days:
            continue
        metadata = _metadata(decision)
        if str(metadata.get("market_state") or "") != "DIRECTIONAL_BALANCE":
            continue
        if str(metadata.get("failure_type") or "") in excluded_failures:
            continue
        if str(metadata.get("opening_range_break_state") or "") in {"DOWN", "FAILED_UP"} and _as_float(metadata.get("below_vwap_last6")) >= 4.0:
            diagnostics["DB_OR_LOW_PRESSURE_WITHOUT_FAILURE_LABEL"].add(session_date)
        if _as_bool(metadata.get("lower_high_confirmed")) and _as_float(metadata.get("price_vs_vwap"), 999.0) < 0.0:
            diagnostics["DB_LOWER_HIGH_BELOW_VWAP"].add(session_date)
        if _as_float(metadata.get("bars_below_vwap")) >= 6.0 and str(metadata.get("bearish_setup") or "") == "PULLBACK_REJECTION":
            diagnostics["DB_REPEATED_RECLAIM_FAILURE_PRESSURE"].add(session_date)
    rows = []
    blindspot_dates = {str(row["date"]) for row in blindspot_rows}
    for name, dates in diagnostics.items():
        rows.append(
            {
                "diagnostic": name,
                "days": len(dates),
                "blindspot_overlap_days": len(set(dates) & blindspot_dates),
                "examples": sorted(dates)[:10],
            }
        )
    rows.sort(key=lambda row: (row["blindspot_overlap_days"], row["days"]), reverse=True)
    return {
        "excluded_failure_labels": sorted(excluded_failures),
        "rows": rows,
        "recommended_shadow_candidate": "DIRECTIONAL_BALANCE_OR_LOW_PRESSURE_BUILD" if rows else None,
    }


def write_blindspot_bearish_research_reports(
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
    blindspot_rows = _blindspot_day_rows(full, params)

    family_counts = Counter(row["hidden_bearish_family"] for row in blindspot_rows)
    block_counts = Counter(row["refined_block_reason"] for row in blindspot_rows)
    classification_report = {
        "qualifying_reasons": sorted(BLIND_SPOT_REASONS),
        "days": len(blindspot_rows),
        "counts_by_hidden_bearish_family": dict(family_counts),
        "counts_by_block_reason": dict(block_counts),
        "examples_by_family": {
            family: [row for row in blindspot_rows if row["hidden_bearish_family"] == family][:5]
            for family in sorted(family_counts)
        },
    }

    table_path = output_dir / f"intraday_bearish_blindspot_day_table_{report_tag}.csv"
    fieldnames = list(blindspot_rows[0].keys()) if blindspot_rows else [
        "date", "refined_block_reason", "hidden_bearish_family", "playbook"
    ]
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(blindspot_rows)

    classification_path = output_dir / f"intraday_bearish_blindspot_classification_{report_tag}.json"
    classification_path.write_text(json.dumps(classification_report, indent=2, sort_keys=True) + "\n")

    shadow_report = _blindspot_shadow_report(
        data_path,
        start=start,
        end=end,
        live_full=full,
        blindspot_rows=blindspot_rows,
        params=params,
    )
    shadow_path = output_dir / f"intraday_bearish_blindspot_shadow_subtypes_{report_tag}.json"
    shadow_path.write_text(json.dumps(shadow_report, indent=2, sort_keys=True) + "\n")

    directional_report = _directional_balance_followup(full, blindspot_rows)
    directional_path = output_dir / f"intraday_directional_balance_blindspot_followup_{report_tag}.json"
    directional_path.write_text(json.dumps(directional_report, indent=2, sort_keys=True) + "\n")

    return {
        "blindspot_classification_report": str(classification_path),
        "blindspot_day_table": str(table_path),
        "shadow_subtype_report": str(shadow_path),
        "directional_balance_followup_report": str(directional_path),
    }
