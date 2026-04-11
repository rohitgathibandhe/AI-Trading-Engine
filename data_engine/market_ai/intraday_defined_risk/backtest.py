from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from statistics import mean, median

from .data_models import (
    AdaptiveParameters,
    AccountRiskLimits,
    AccountState,
    MarketSnapshot,
    OhlcvBar,
    OhlcvSeries,
    OptionType,
    OptionsChainSnapshot,
    OptionsContractQuote,
)
from .execution import evaluate_exit
from .regime import classify_regime
from .strikes import select_structure
from .learning import LearningStore
from .monitor import IntradayDefinedRiskAgent


def _default_entry_times() -> set[str]:
    values: set[str] = set()
    hour = 9
    minute = 30
    while (hour, minute) <= (14, 0):
        values.add(f"{hour:02d}:{minute:02d}")
        minute += 5
        if minute >= 60:
            hour += 1
            minute -= 60
    return values


DEFAULT_ENTRY_TIMES = _default_entry_times()

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


def _build_funnel_report(decisions: list[dict[str, object]], trades: list[dict[str, object]]) -> dict[str, object]:
    setups_detected = 0
    setups_passed_setup_quality = 0
    setups_by_playbook: Counter[str] = Counter()
    rejected_by_reason: Counter[str] = Counter()
    rejected_by_playbook: Counter[str] = Counter()
    rejected_time = 0
    rejected_spread = 0
    rejected_credit = 0
    rejected_liquidity = 0
    rejected_delta = 0
    rejected_anchor = 0
    rejected_hedge = 0
    monthly: dict[str, dict[str, object]] = defaultdict(lambda: {
        "setups_detected": 0,
        "setups_passed_setup_quality": 0,
        "trades_executed": 0,
        "rejected_time_window": 0,
        "rejected_spread_construction": 0,
        "rejected_credit_width": 0,
        "rejected_liquidity": 0,
        "rejected_delta": 0,
        "rejected_invalidation_distance": 0,
        "rejected_hedge_cost": 0,
        "setups_by_playbook": Counter(),
        "missed_opportunities_by_playbook": Counter(),
    })
    for decision in decisions:
        funnel = ((decision.get("metadata") or {}).get("trade_funnel") or {})
        playbook = str(funnel.get("playbook") or (decision.get("metadata") or {}).get("playbook") or "UNKNOWN")
        month = str(funnel.get("date") or decision.get("session_date") or "")[:7]
        if funnel.get("setup_detected"):
            setups_detected += 1
            setups_by_playbook[playbook] += 1
            monthly[month]["setups_detected"] += 1
            monthly[month]["setups_by_playbook"][playbook] += 1
            if funnel.get("passed_setup_quality"):
                setups_passed_setup_quality += 1
                monthly[month]["setups_passed_setup_quality"] += 1
        if funnel.get("final_result") == "EXECUTED":
            monthly[month]["trades_executed"] += 1
            continue
        reason = str(funnel.get("canonical_rejection_reason") or "UNKNOWN")
        if funnel.get("setup_detected"):
            rejected_by_reason[reason] += 1
            rejected_by_playbook[playbook] += 1
            if reason == "TIME_WINDOW":
                rejected_time += 1
                monthly[month]["rejected_time_window"] += 1
            if reason in MONETIZATION_REJECTION_REASONS:
                rejected_spread += 1
                monthly[month]["rejected_spread_construction"] += 1
                monthly[month]["missed_opportunities_by_playbook"][playbook] += 1
            if reason in {"CREDIT_TOO_LOW", "NET_EDGE_TOO_LOW"}:
                rejected_credit += 1
                monthly[month]["rejected_credit_width"] += 1
            if reason == "LIQUIDITY_BAD":
                rejected_liquidity += 1
                monthly[month]["rejected_liquidity"] += 1
            if reason == "DELTA_TOO_HIGH":
                rejected_delta += 1
                monthly[month]["rejected_delta"] += 1
            if reason == "INVALIDATION_TOO_CLOSE":
                rejected_anchor += 1
                monthly[month]["rejected_invalidation_distance"] += 1
            if reason == "HEDGE_TOO_EXPENSIVE":
                rejected_hedge += 1
                monthly[month]["rejected_hedge_cost"] += 1

    pnl_by_playbook: dict[str, list[float]] = defaultdict(list)
    wins_by_playbook: Counter[str] = Counter()
    losses_by_playbook: Counter[str] = Counter()
    for trade in trades:
        playbook = str(trade.get("playbook") or "UNKNOWN")
        pnl = float(trade.get("pnl_rupees") or 0.0)
        pnl_by_playbook[playbook].append(pnl)
        if pnl > 0:
            wins_by_playbook[playbook] += 1
        elif pnl < 0:
            losses_by_playbook[playbook] += 1

    monthly_summary = {}
    for month, payload in monthly.items():
        monthly_summary[month] = {
            "setups_detected": payload["setups_detected"],
            "setups_passed_setup_quality": payload["setups_passed_setup_quality"],
            "trades_executed": payload["trades_executed"],
            "rejected_time_window": payload["rejected_time_window"],
            "rejected_spread_construction": payload["rejected_spread_construction"],
            "rejected_credit_width": payload["rejected_credit_width"],
            "rejected_liquidity": payload["rejected_liquidity"],
            "rejected_delta": payload["rejected_delta"],
            "rejected_invalidation_distance": payload["rejected_invalidation_distance"],
            "rejected_hedge_cost": payload["rejected_hedge_cost"],
            "setups_by_playbook": dict(payload["setups_by_playbook"]),
            "missed_opportunities_by_playbook": dict(payload["missed_opportunities_by_playbook"]),
        }

    return {
        "setups_detected": setups_detected,
        "setups_passed_setup_quality": setups_passed_setup_quality,
        "setups_by_playbook": dict(setups_by_playbook),
        "rejected_time_window": rejected_time,
        "rejected_spread_construction": rejected_spread,
        "rejected_credit_width": rejected_credit,
        "rejected_liquidity": rejected_liquidity,
        "rejected_delta": rejected_delta,
        "rejected_invalidation_distance": rejected_anchor,
        "rejected_hedge_cost": rejected_hedge,
        "trades_executed": sum(1 for decision in decisions if ((decision.get("metadata") or {}).get("trade_funnel") or {}).get("final_result") == "EXECUTED"),
        "top_rejection_reasons": rejected_by_reason.most_common(10),
        "blocked_by_spread_construction_by_playbook": [
            (playbook, count)
            for playbook, count in rejected_by_playbook.most_common()
            if count > 0
        ][:10],
        "win_loss_by_playbook": {
            playbook: {"wins": wins_by_playbook.get(playbook, 0), "losses": losses_by_playbook.get(playbook, 0)}
            for playbook in sorted(set(wins_by_playbook) | set(losses_by_playbook) | set(pnl_by_playbook))
        },
        "average_pnl_by_playbook": {
            playbook: round(mean(values), 2) if values else 0.0
            for playbook, values in pnl_by_playbook.items()
        },
        "missed_opportunity_leaderboard": [
            (playbook, count)
            for playbook, count in sorted(
                (
                    (playbook, sum(month_payload["missed_opportunities_by_playbook"].get(playbook, 0) for month_payload in monthly_summary.values()))
                    for playbook in setups_by_playbook
                ),
                key=lambda item: item[1],
                reverse=True,
            )
            if count > 0
        ][:10],
        "monthly_summary": monthly_summary,
    }


def run_backtest(
    data_path: str | Path,
    start: str | None = None,
    end: str | None = None,
    risk_limits: AccountRiskLimits | None = None,
    learning_db_path: str | Path = "/tmp/intraday_defined_risk_backtest.sqlite3",
    parameters: AdaptiveParameters | None = None,
    reset_learning_db: bool = True,
    entry_times: set[str] | None = None,
    round_trip_cost_rupees_per_lot: float = 0.0,
    *,
    allowed_playbook_tiers: tuple[str, ...] = ("A",),
    include_opportunity_benchmark: bool = True,
) -> dict[str, object]:
    if reset_learning_db:
        Path(learning_db_path).unlink(missing_ok=True)
    dataset = load_backtest_dataset(data_path)
    bars_5m = dataset["bars_5m"]
    bars_15m = dataset["bars_15m"]
    bars_5m_by_day = dataset["bars_5m_by_day"]
    bars_15m_by_day = dataset["bars_15m_by_day"]
    chain_map = dataset["chain_map"]
    previous_session_closes = _session_previous_closes(bars_5m)

    start_dt = datetime.fromisoformat(start) if start else None
    end_dt = datetime.fromisoformat(end) if end else None
    timestamps = sorted(ts for ts in chain_map if (start_dt is None or ts >= start_dt) and (end_dt is None or ts <= end_dt))

    risk_limits = risk_limits or AccountRiskLimits(
        max_risk_rupees_per_trade=20000,
        max_margin_rupees=200000,
        max_daily_loss_rupees=10000,
    )
    account_state = AccountState(realised_pnl_rupees=0.0, margin_used_rupees=0.0)
    total_realized_pnl_rupees = 0.0
    agent = IntradayDefinedRiskAgent(
        learning_store=LearningStore(learning_db_path),
        parameters=parameters,
        allowed_playbook_tiers=allowed_playbook_tiers,
        benchmark_mode="opportunity" if set(allowed_playbook_tiers) == {"A", "B"} else "strict",
    )
    entry_times = entry_times or DEFAULT_ENTRY_TIMES

    decisions: list[dict[str, object]] = []
    trades: list[dict[str, object]] = []
    wins = 0
    losses = 0
    flats = 0
    equity_curve = [0.0]
    signal_counts: Counter[str] = Counter()
    trade_counts: Counter[str] = Counter()
    pnl_by_strategy: dict[str, float] = defaultdict(float)
    margin_util_samples: list[float] = []
    last_snapshot: MarketSnapshot | None = None
    current_session_date: date | None = None
    total_transaction_costs_rupees = 0.0
    previous_chain_snapshot = None
    current_day_bars_5m: list[OhlcvBar] = []
    current_day_bars_15m: list[OhlcvBar] = []
    session_bars_5m: list[OhlcvBar] = []
    session_bars_15m: list[OhlcvBar] = []
    session_5m_index = 0
    session_15m_index = 0

    def close_open_position(snapshot: MarketSnapshot, *, force_time_exit: bool = False) -> None:
        nonlocal account_state, total_realized_pnl_rupees, wins, losses, flats, last_snapshot, total_transaction_costs_rupees
        if not agent.open_position:
            return
        position = agent.open_position
        current_strategy = position.structure.strategy.value
        current_regime = classify_regime(snapshot, agent.parameters)
        exit_now = snapshot.timestamp.replace(hour=15, minute=15, second=0, microsecond=0) if force_time_exit else snapshot.timestamp
        exit_decision = evaluate_exit(
            position,
            current_snapshot=snapshot,
            current_regime=current_regime,
            now=exit_now,
        )
        if not exit_decision.should_exit:
            return
        gross_pnl_fragment = exit_decision.pnl_rupees
        trade_cost_rupees = round(position.lots * round_trip_cost_rupees_per_lot, 2)
        pnl_fragment = gross_pnl_fragment - trade_cost_rupees
        total_transaction_costs_rupees += trade_cost_rupees
        total_realized_pnl_rupees += pnl_fragment
        account_state = AccountState(
            realised_pnl_rupees=account_state.realised_pnl_rupees + pnl_fragment,
            margin_used_rupees=0.0,
        )
        agent.learning_store.log_outcome(
            session_date=snapshot.timestamp.date().isoformat(),
            strategy=current_strategy,
            pnl_rupees=pnl_fragment,
            max_drawdown_rupees=max(-pnl_fragment, 0.0),
            margin_utilisation=min(
                snapshot.account_state.margin_used_rupees / snapshot.risk_limits.max_margin_rupees,
                1.0,
            ) if snapshot.risk_limits.max_margin_rupees > 0 else 1.0,
            features=agent._current_features | {"exit_reason": exit_decision.reason},
        )
        agent.open_position = None
        if pnl_fragment > 0:
            wins += 1
        elif pnl_fragment < 0:
            losses += 1
        else:
            flats += 1
        trade_counts[current_strategy] += 1
        pnl_by_strategy[current_strategy] += pnl_fragment
        equity_curve.append(total_realized_pnl_rupees)
        trades.append(
            {
                "entry_timestamp": position.entry_time.isoformat(),
                "timestamp": exit_now.isoformat(),
                "strategy": current_strategy,
                "day_archetype": position.metadata.get("day_archetype"),
                "playbook": position.metadata.get("playbook"),
                "bullish_setup": position.metadata.get("bullish_setup"),
                "bearish_setup": position.metadata.get("bearish_setup"),
                "smart_money_bias": position.metadata.get("smart_money_bias"),
                "structure_width_points": position.metadata.get("structure_width_points"),
                "structure_credit_points": position.metadata.get("structure_credit_points"),
                "exit_reason": exit_decision.reason,
                "gross_pnl_rupees": gross_pnl_fragment,
                "transaction_cost_rupees": trade_cost_rupees,
                "pnl_rupees": pnl_fragment,
            }
        )

    for ts in timestamps:
        if current_session_date is None:
            current_session_date = ts.date()
        elif ts.date() != current_session_date:
            if agent.open_position and last_snapshot is not None:
                close_open_position(last_snapshot, force_time_exit=True)
            account_state = AccountState(realised_pnl_rupees=0.0, margin_used_rupees=0.0)
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
            account_state=account_state,
            live_vwap=None,
            lot_size=65,
            slippage_points=0.5,
            previous_option_chain=previous_chain_snapshot,
            previous_session_close=previous_session_closes.get(ts.date()),
        )

        if agent.open_position:
            close_open_position(snapshot)
            last_snapshot = snapshot
            previous_chain_snapshot = chain_snapshot
            continue

        if ts.strftime("%H:%M") not in entry_times:
            last_snapshot = snapshot
            previous_chain_snapshot = chain_snapshot
            continue

        decision = agent.evaluate(snapshot)
        decision_record = decision.to_dict()
        decision_record["timestamp"] = ts.isoformat()
        decision_record["session_date"] = ts.date().isoformat()
        decisions.append(decision_record)
        signal_counts[decision.strategy.value] += 1
        if decision.action == "TRADE":
            agent.start_position(snapshot, decision)
            margin_used = snapshot.option_chain.margin_estimate_per_lot or decision.max_loss_rupees_per_lot
            margin_util_samples.append(
                min(margin_used / risk_limits.max_margin_rupees, 1.0) if risk_limits.max_margin_rupees > 0 else 1.0
            )
            account_state = AccountState(
                realised_pnl_rupees=account_state.realised_pnl_rupees,
                margin_used_rupees=margin_used,
            )
        last_snapshot = snapshot
        previous_chain_snapshot = chain_snapshot

    if agent.open_position and last_snapshot is not None:
        close_open_position(last_snapshot, force_time_exit=True)

    gross_profit = round(sum(max(trade["pnl_rupees"], 0.0) for trade in trades), 2)
    gross_loss = round(abs(sum(min(trade["pnl_rupees"], 0.0) for trade in trades)), 2)
    profit_factor = round(gross_profit / gross_loss, 4) if gross_loss > 0 else round(gross_profit, 4)
    max_drawdown = round(_max_drawdown(equity_curve), 2)
    max_margin_util = round(max(margin_util_samples), 4) if margin_util_samples else 0.0
    median_margin_util = round(median(margin_util_samples), 4) if margin_util_samples else 0.0
    drawdown_ratio = max_drawdown / max(gross_profit, 1.0)
    objective = round(profit_factor - (0.5 * drawdown_ratio) - (0.3 * median_margin_util), 4)
    summary = {
        "start": timestamps[0].isoformat() if timestamps else None,
        "end": timestamps[-1].isoformat() if timestamps else None,
        "signals": len(decisions),
        "trades": len(trades),
        "wins": wins,
        "losses": losses,
        "flats": flats,
        "realized_pnl_rupees": round(total_realized_pnl_rupees, 2),
        "transaction_costs_rupees": round(total_transaction_costs_rupees, 2),
        "win_rate": round(wins / len(trades), 4) if trades else 0.0,
        "gross_profit_rupees": gross_profit,
        "gross_loss_rupees": gross_loss,
        "profit_factor": profit_factor,
        "max_drawdown_rupees": max_drawdown,
        "max_margin_utilisation": max_margin_util,
        "median_margin_utilisation": median_margin_util,
        "objective_score": objective,
        "signal_counts_by_strategy": dict(signal_counts),
        "trade_counts_by_strategy": dict(trade_counts),
        "pnl_by_strategy": {key: round(value, 2) for key, value in pnl_by_strategy.items()},
        "trade_counts_by_archetype": _trade_counts_by_key(trades, "day_archetype"),
        "pnl_by_archetype": _trade_pnl_by_key(trades, "day_archetype"),
        "trade_counts_by_playbook": _trade_counts_by_key(trades, "playbook"),
        "pnl_by_playbook": _trade_pnl_by_key(trades, "playbook"),
        "parameters": parameters.clamped().__dict__ if parameters is not None else None,
        "entry_times": sorted(entry_times),
        "round_trip_cost_rupees_per_lot": round_trip_cost_rupees_per_lot,
        "benchmark_mode": "opportunity" if set(allowed_playbook_tiers) == {"A", "B"} else "strict",
        "allowed_playbook_tiers": list(allowed_playbook_tiers),
    }
    funnel_report = _build_funnel_report(decisions, trades)
    summary["trade_funnel"] = funnel_report
    opportunity_result = None
    if include_opportunity_benchmark and set(allowed_playbook_tiers) == {"A"}:
        opportunity_result = run_backtest(
            data_path,
            start=start,
            end=end,
            risk_limits=risk_limits,
            learning_db_path=f"{learning_db_path}.opportunity",
            parameters=parameters,
            reset_learning_db=True,
            entry_times=entry_times,
            round_trip_cost_rupees_per_lot=round_trip_cost_rupees_per_lot,
            allowed_playbook_tiers=("A", "B"),
            include_opportunity_benchmark=False,
        )
        summary["opportunity_benchmark"] = opportunity_result["summary"]
        summary["opportunity_gap"] = {
            "trade_delta": opportunity_result["summary"]["trades"] - summary["trades"],
            "pnl_delta_rupees": round(
                opportunity_result["summary"]["realized_pnl_rupees"] - summary["realized_pnl_rupees"],
                2,
            ),
        }
    return {
        "summary": summary,
        "decisions": decisions,
        "trades": trades,
        "opportunity": opportunity_result,
    }


def load_backtest_dataset(data_path: str | Path) -> dict[str, object]:
    root = Path(data_path)
    bars_5m = load_ohlcv_csv(root / "nifty_5m.csv", timeframe_minutes=5)
    bars_15m = load_ohlcv_csv(root / "nifty_15m.csv", timeframe_minutes=15)
    chain_map = load_option_chain_csv(root / "options_chain.csv")
    return {
        "bars_5m": bars_5m,
        "bars_15m": bars_15m,
        "bars_5m_by_day": _group_bars_by_day(bars_5m),
        "bars_15m_by_day": _group_bars_by_day(bars_15m),
        "chain_map": chain_map,
    }


def load_ohlcv_csv(path: str | Path, timeframe_minutes: int) -> list[OhlcvBar]:
    bars: list[OhlcvBar] = []
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            bars.append(
                OhlcvBar(
                    timestamp=datetime.fromisoformat(row["timestamp"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row.get("volume", 0.0)),
                )
            )
    return bars


def load_option_chain_csv(path: str | Path) -> dict[datetime, OptionsChainSnapshot]:
    grouped: dict[datetime, list[OptionsContractQuote]] = defaultdict(list)
    expiries: dict[datetime, date] = {}
    spots: dict[datetime, float] = {}
    margins: dict[datetime, float | None] = {}
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ts = datetime.fromisoformat(row["timestamp"])
            expiries[ts] = date.fromisoformat(row["expiry"])
            spots[ts] = float(row["spot"])
            margin_value = row.get("margin_estimate_per_lot")
            margins[ts] = float(margin_value) if margin_value not in {None, ""} else None
            grouped[ts].append(
                OptionsContractQuote(
                    strike=float(row["strike"]),
                    option_type=OptionType(row["option_type"]),
                    bid=float(row["bid"]),
                    ask=float(row["ask"]),
                    ltp=float(row["ltp"]),
                    delta=float(row["delta"]) if row.get("delta") not in {None, ""} else None,
                    iv=float(row["iv"]) if row.get("iv") not in {None, ""} else None,
                    oi=int(float(row["oi"])) if row.get("oi") not in {None, ""} else None,
                    symbol=row.get("symbol") or None,
                )
            )
    return {
        ts: OptionsChainSnapshot(
            timestamp=ts,
            expiry=expiries[ts],
            spot=spots[ts],
            quotes=quotes,
            margin_estimate_per_lot=margins[ts],
        )
        for ts, quotes in grouped.items()
    }


def _group_bars_by_day(bars: list[OhlcvBar]) -> dict[date, list[OhlcvBar]]:
    grouped: dict[date, list[OhlcvBar]] = defaultdict(list)
    for bar in bars:
        grouped[bar.timestamp.date()].append(bar)
    return grouped


def _max_drawdown(equity_curve: list[float]) -> float:
    peak = equity_curve[0] if equity_curve else 0.0
    drawdown = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return drawdown


def _trade_counts_by_key(trades: list[dict[str, object]], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for trade in trades:
        counts[str(trade.get(key) or "UNKNOWN")] += 1
    return dict(counts)


def _trade_pnl_by_key(trades: list[dict[str, object]], key: str) -> dict[str, float]:
    pnl: defaultdict[str, float] = defaultdict(float)
    for trade in trades:
        pnl[str(trade.get(key) or "UNKNOWN")] += float(trade.get("pnl_rupees") or 0.0)
    return {name: round(value, 2) for name, value in pnl.items()}


def _session_previous_closes(bars_5m: list[OhlcvBar]) -> dict[date, float]:
    closes_by_day: dict[date, float] = {}
    ordered_days: list[date] = []
    for bar in bars_5m:
        session_day = bar.timestamp.date()
        if session_day not in closes_by_day:
            ordered_days.append(session_day)
        closes_by_day[session_day] = bar.close
    previous_map: dict[date, float] = {}
    prev_close: float | None = None
    for session_day in ordered_days:
        if prev_close is not None:
            previous_map[session_day] = prev_close
        prev_close = closes_by_day[session_day]
    return previous_map
