from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from datetime import date, datetime, time
from pathlib import Path
from statistics import mean
from typing import Mapping

from .backtest import DEFAULT_ENTRY_TIMES
from .data_models import AccountRiskLimits, AdaptiveParameters, MarketSnapshot, OpenPosition, RegimeState
from .frequency_expansion_research import _close_shadow_position, _iter_snapshots, _open_shadow_position
from .regime import classify_regime

STATE_ROOT = Path(__file__).resolve().parents[1] / "state"
SHADOW_SUBTYPE = "BEARISH_FAILED_BREAKOUT_TRANSITION"
DEFAULT_REFERENCE_TIMESTAMP = "2026-04-23T10:37:20.399931"


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_bool(value: object) -> bool:
    return bool(value)


def _shadow_match_reasons(
    metadata: Mapping[str, object],
    *,
    execution_5m: str | None = None,
    params: AdaptiveParameters | None = None,
) -> list[str]:
    params = (params or AdaptiveParameters()).clamped()
    reasons: list[str] = []

    if str(metadata.get("market_state") or "") != "TRANSITION":
        reasons.append("market_state_not_transition")
    if str(metadata.get("failure_type") or "") != "FAILED_BREAKOUT":
        reasons.append("failure_type_not_failed_breakout")
    if str(metadata.get("tradability_class") or "") != "TRADABLE":
        reasons.append("tradability_not_tradable")
    if str(metadata.get("opening_range_break_state") or "") != "FAILED_UP":
        reasons.append("opening_range_break_not_failed_up")

    five_min = str(execution_5m or metadata.get("execution_5m") or "")
    if five_min in {"UP_CONFIRMED", "RANGE_CONFIRMED"}:
        reasons.append("execution_trigger_still_bullish_or_range")

    bearish_score = _as_float(metadata.get("bearish_trade_score"))
    threshold = _as_float(metadata.get("bearish_trade_score_threshold"), params.bearish_trade_score_threshold)
    if bearish_score < threshold - params.failed_breakout_transition_shadow_buffer:
        reasons.append("bearish_trade_score_below_near_threshold_buffer")

    setup_quality = _as_float(metadata.get("setup_quality_score"))
    if setup_quality < params.strong_setup_quality_threshold:
        reasons.append("setup_quality_below_strong_threshold")

    structure_signal = str(metadata.get("structure_signal") or "")
    if structure_signal not in {"BALANCED", "BEARISH_BOS", "BEARISH_CHOCH"}:
        reasons.append("structure_not_bearish_transition_ready")

    lower_high_confirmed = _as_bool(metadata.get("lower_high_confirmed"))
    bearish_candle_quality = _as_float(metadata.get("bearish_candle_quality_score"))
    if not lower_high_confirmed and bearish_candle_quality < 3.0:
        reasons.append("no_lower_high_or_candle_rejection_confirmation")

    if _as_float(metadata.get("trend_efficiency_ratio")) < 0.28:
        reasons.append("trend_efficiency_too_low")

    momentum_persistence = _as_float(metadata.get("momentum_persistence_score"))
    if momentum_persistence < 0.5 or momentum_persistence > 3.8:
        reasons.append("momentum_persistence_outside_transition_band")

    rejection_checks = 0
    for key in ("price_vs_vwap", "price_vs_ema20", "price_vs_or_high"):
        if _as_float(metadata.get(key), 999.0) <= -5.0:
            rejection_checks += 1
    if rejection_checks == 0:
        reasons.append("acceptance_not_lost_below_vwap_ema20_or_breakout_zone")

    return reasons


def matches_bearish_failed_breakout_transition_shadow(
    metadata: Mapping[str, object],
    *,
    execution_5m: str | None = None,
    params: AdaptiveParameters | None = None,
) -> bool:
    return not _shadow_match_reasons(metadata, execution_5m=execution_5m, params=params)


def _quality_gate(metadata: Mapping[str, object], params: AdaptiveParameters) -> tuple[bool, str | None]:
    bearish_score = _as_float(metadata.get("bearish_trade_score"))
    no_trade_score = _as_float(metadata.get("no_trade_score"))
    if bearish_score <= no_trade_score + 0.05:
        return False, "score_margin_too_small"
    if _as_float(metadata.get("bearish_location_live_score")) < 1.5:
        return False, "location_score_too_low"
    return True, None


def _load_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text())


def _load_jsonl(path: str | Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _shadow_report_rows(
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
    open_position: OpenPosition | None = None
    last_snapshot: MarketSnapshot | None = None
    current_day: date | None = None
    traded_days: set[str] = set()
    setup_examples: list[dict[str, object]] = []
    entry_fail_reasons: Counter[str] = Counter()
    trades: list[dict[str, object]] = []
    setups_detected = 0
    setups_passing_quality = 0

    for ts, snapshot in _iter_snapshots(data_path, start=start, end=end, risk_limits=risk_limits):
        session_date = ts.date().isoformat()
        if current_day is None:
            current_day = ts.date()
        elif ts.date() != current_day:
            if open_position is not None and last_snapshot is not None:
                closed = _close_shadow_position(open_position, last_snapshot, params, force_time_exit=True)
                if closed is not None:
                    trades.append(closed)
                open_position = None
            current_day = ts.date()

        regime_state = classify_regime(snapshot, params)
        metadata = regime_state.metadata
        last_snapshot = snapshot

        if open_position is not None:
            closed = _close_shadow_position(open_position, snapshot, params)
            if closed is not None:
                trades.append(closed)
                open_position = None
            continue

        if ts.strftime("%H:%M") not in DEFAULT_ENTRY_TIMES:
            continue
        if ts.time() >= time(14, 0):
            continue
        if session_date in traded_days:
            continue
        if not matches_bearish_failed_breakout_transition_shadow(metadata, execution_5m=regime_state.execution_5m, params=params):
            continue

        setups_detected += 1
        if len(setup_examples) < 12:
            setup_examples.append(
                {
                    "timestamp": ts.isoformat(),
                    "playbook": metadata.get("playbook"),
                    "failure_type": metadata.get("failure_type"),
                    "market_state": metadata.get("market_state"),
                    "tradability_class": metadata.get("tradability_class"),
                    "bearish_trade_score": round(_as_float(metadata.get("bearish_trade_score")), 4),
                    "bearish_trade_score_threshold": round(_as_float(metadata.get("bearish_trade_score_threshold")), 4),
                    "setup_quality_score": round(_as_float(metadata.get("setup_quality_score")), 4),
                    "price_vs_vwap": round(_as_float(metadata.get("price_vs_vwap")), 4),
                    "price_vs_ema20": round(_as_float(metadata.get("price_vs_ema20")), 4),
                    "price_vs_or_high": round(_as_float(metadata.get("price_vs_or_high")), 4),
                    "execution_5m": regime_state.execution_5m,
                }
            )

        quality_ok, quality_reason = _quality_gate(metadata, params)
        if not quality_ok:
            entry_fail_reasons[str(quality_reason or "QUALITY_FAIL")] += 1
            continue
        setups_passing_quality += 1

        shadow_regime = replace(
            regime_state,
            metadata=dict(metadata)
            | {
                "bearish_shadow_subtype": SHADOW_SUBTYPE,
                "playbook": str(metadata.get("playbook") or "NO_TRADE"),
                "shadow_entry_mode": "RELAXED_5M_TRIGGER",
            },
        )
        position = _open_shadow_position(snapshot, shadow_regime, params, SHADOW_SUBTYPE)
        if position is None:
            entry_fail_reasons["STRUCTURE_OR_NET_EDGE_FAIL"] += 1
            continue
        open_position = position
        traded_days.add(session_date)

    if open_position is not None and last_snapshot is not None:
        closed = _close_shadow_position(open_position, last_snapshot, params, force_time_exit=True)
        if closed is not None:
            trades.append(closed)

    pnl_values = [float(trade.get("pnl_rupees") or 0.0) for trade in trades]
    wins = sum(1 for pnl in pnl_values if pnl > 0.0)
    losses = sum(1 for pnl in pnl_values if pnl < 0.0)
    trade_days = {str(trade.get("entry_timestamp") or "")[:10] for trade in trades}
    overlap_days = sorted(trade_days & live_trade_days)
    unique_added_days = sorted(trade_days - live_trade_days)
    win_rate = (wins / len(trades)) if trades else 0.0
    average_pnl = round(mean(pnl_values), 2) if pnl_values else 0.0
    realized_pnl = round(sum(pnl_values), 2)
    low_quality_flow_reopened = bool(trades and (average_pnl <= 0.0 or win_rate < 0.55))

    return {
        "subtype": SHADOW_SUBTYPE,
        "setups_detected": setups_detected,
        "setups_passing_quality": setups_passing_quality,
        "shadow_trades": len(trades),
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 4),
        "realized_pnl_rupees": realized_pnl,
        "average_pnl_rupees": average_pnl,
        "overlap_with_v83_live_trades": len(overlap_days),
        "overlap_days": overlap_days,
        "unique_added_trade_days": len(unique_added_days),
        "unique_added_days": unique_added_days,
        "low_quality_flow_reopened": low_quality_flow_reopened,
        "top_entry_fail_reasons": entry_fail_reasons.most_common(10),
        "sample_setups": setup_examples,
        "trades": trades,
    }


def build_reference_case_diagnostic(
    *,
    reference_timestamp: str = DEFAULT_REFERENCE_TIMESTAMP,
    runner_log_path: str | Path = STATE_ROOT / "intraday_v83_runner.log",
    params: AdaptiveParameters | None = None,
) -> dict[str, object]:
    params = (params or AdaptiveParameters()).clamped()
    for payload in _load_jsonl(runner_log_path):
        if str(payload.get("entry", {}).get("timestamp") or "") == reference_timestamp:
            continue
        if str(payload.get("metadata", {}).get("trade_funnel", {}).get("timestamp") or "") == reference_timestamp:
            metadata = dict(payload.get("metadata") or {})
            rationale = list(payload.get("rationale") or [])
            execution_5m = None
            if any("5m execution trigger is not confirmed" in line for line in rationale):
                execution_5m = "UNCONFIRMED"
            matches = matches_bearish_failed_breakout_transition_shadow(metadata, execution_5m=execution_5m, params=params)
            return {
                "reference_timestamp": reference_timestamp,
                "captured_by_shadow_subtype": matches,
                "shadow_subtype": SHADOW_SUBTYPE if matches else None,
                "shadow_rejection_reasons": _shadow_match_reasons(metadata, execution_5m=execution_5m, params=params),
                "decision": {
                    "playbook": metadata.get("playbook"),
                    "market_state": metadata.get("market_state"),
                    "tradability_class": metadata.get("tradability_class"),
                    "failure_type": metadata.get("failure_type"),
                    "bearish_trade_score": round(_as_float(metadata.get("bearish_trade_score")), 4),
                    "bearish_trade_score_threshold": round(_as_float(metadata.get("bearish_trade_score_threshold")), 4),
                    "score_gap_to_threshold": round(_as_float(metadata.get("bearish_trade_score_threshold")) - _as_float(metadata.get("bearish_trade_score")), 4),
                    "setup_quality_score": round(_as_float(metadata.get("setup_quality_score")), 4),
                    "price_vs_vwap": round(_as_float(metadata.get("price_vs_vwap")), 4),
                    "price_vs_ema20": round(_as_float(metadata.get("price_vs_ema20")), 4),
                    "price_vs_or_high": round(_as_float(metadata.get("price_vs_or_high")), 4),
                    "opening_range_break_state": metadata.get("opening_range_break_state"),
                    "lower_high_confirmed": _as_bool(metadata.get("lower_high_confirmed")),
                    "bearish_candle_quality_score": round(_as_float(metadata.get("bearish_candle_quality_score")), 4),
                    "trend_efficiency_ratio": round(_as_float(metadata.get("trend_efficiency_ratio")), 4),
                    "momentum_persistence_score": round(_as_float(metadata.get("momentum_persistence_score")), 4),
                    "execution_5m_inferred": execution_5m,
                },
                "rationale": rationale,
            }
    return {
        "reference_timestamp": reference_timestamp,
        "captured_by_shadow_subtype": False,
        "shadow_subtype": None,
        "shadow_rejection_reasons": ["reference_case_not_found_in_runner_log"],
        "decision": None,
        "rationale": [],
    }


def run_bearish_failed_breakout_transition_shadow_research(
    *,
    data_path: str | Path,
    full_benchmark_path: str | Path,
    start: str,
    end: str,
    params: AdaptiveParameters | None = None,
    report_date: str | None = None,
    write: bool = True,
    output_dir: str | Path | None = None,
    reference_timestamp: str = DEFAULT_REFERENCE_TIMESTAMP,
) -> dict[str, object]:
    params = (params or AdaptiveParameters()).clamped()
    report_date = report_date or datetime.now().date().isoformat()
    output_root = Path(output_dir) if output_dir is not None else STATE_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    live_full = _load_json(full_benchmark_path)
    summary = _shadow_report_rows(
        data_path,
        start=start,
        end=end,
        live_full=live_full,
        params=params,
    )
    reference_case = build_reference_case_diagnostic(
        reference_timestamp=reference_timestamp,
        params=params,
    )

    if summary["shadow_trades"] >= 3 and summary["realized_pnl_rupees"] > 0.0 and summary["losses"] == 0 and summary["unique_added_trade_days"] >= 2:
        recommendation = "RUN_NARROW_PROMOTION_EXPERIMENT"
    elif summary["shadow_trades"] > 0 and summary["realized_pnl_rupees"] > 0.0:
        recommendation = "KEEP_SHADOW"
    else:
        recommendation = "REJECT"

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_path": str(data_path),
        "full_benchmark_path": str(full_benchmark_path),
        "start": start,
        "end": end,
        "subtype": SHADOW_SUBTYPE,
        "summary": summary,
        "reference_case": reference_case,
        "recommendation": recommendation,
    }

    if write:
        report_path = output_root / f"intraday_bearish_failed_breakout_transition_shadow_{report_date}_v83.json"
        diagnostic_path = output_root / f"intraday_bearish_failed_breakout_transition_reference_case_{report_date}_v83.json"
        report_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        diagnostic_path.write_text(json.dumps(reference_case, indent=2, sort_keys=True) + "\n")
        payload["report_path"] = str(report_path)
        payload["diagnostic_path"] = str(diagnostic_path)
    return payload
