from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, time
from pathlib import Path

from .data_models import (
    DecisionOutput,
    ExitDecision,
    MarketSnapshot,
    OpenPosition,
    OptionType,
    RegimeLabel,
    RegimeState,
    StrategyLeg,
    StrategyType,
    TradeStructure,
)
from .features import compute_vwap, last_n_closes_above, last_n_closes_below, latest_pivot_high, latest_pivot_low, session_bars
from .features import bullish_reversal_structure, bearish_reversal_structure, closes, ema, ema_value


_STATE_ROOT = Path(__file__).resolve().parents[1] / "state"


def _read_current_iv_rank() -> float | None:
    """Read the most recent IV rank percentile from iv_history.csv.

    Format: date,avg_iv,iv_rank  (written daily by watchdog after 15:30).
    Returns None when file is missing or has fewer than 2 rows (no history yet).
    """
    iv_path = _STATE_ROOT / "iv_history.csv"
    if not iv_path.exists():
        return None
    try:
        rows = [line.strip() for line in iv_path.read_text().splitlines() if line.strip()]
        for row in reversed(rows):
            parts = row.split(",")
            if len(parts) >= 3:
                try:
                    return float(parts[2])
                except ValueError:
                    continue
    except Exception:
        pass
    return None


TIME_EXIT = time(15, 15)
LATEST_DIRECTIONAL_ENTRY = time(14, 30)
DIRECTIONAL_TP_CAPTURE = 0.65
CONDOR_TP_CAPTURE = 0.50
CONVICTION_TP_CAPTURE = 0.80
DIRECTIONAL_DELTA_SL = 0.40
BULLISH_PLAYBOOK_DELTA_SL = 0.50
CONDOR_DELTA_SL = 0.25
PREMIUM_SL_MULTIPLIER = 2.0
DAILY_PROFIT_TRAIL_ARM_RUPEES = 5000.0
DAILY_PROFIT_TRAIL_GIVEBACK_RUPEES = 1500.0
BULLISH_PLAYBOOK_PROFIT_TRAIL_ARM_RUPEES = 6500.0
BULLISH_PLAYBOOK_PROFIT_TRAIL_GIVEBACK_RUPEES = 2000.0
PROFIT_TRAIL_CAPTURE_ARM = 0.35
PROFIT_TRAIL_GIVEBACK_POINTS = 6.0


def validate_entry_time(strategy: StrategyType, now: datetime) -> tuple[bool, str | None]:
    if now.time() < time(9, 15):
        return False, "Entries are not allowed before 09:15 IST."
    if strategy in {StrategyType.BEAR_CALL_CREDIT_SPREAD, StrategyType.BULL_PUT_CREDIT_SPREAD} and now.time() < time(9, 30):
        return False, "Directional entries prefer time >= 09:30 IST."
    if strategy in {StrategyType.BEAR_CALL_CREDIT_SPREAD, StrategyType.BULL_PUT_CREDIT_SPREAD} and now.time() > LATEST_DIRECTIONAL_ENTRY:
        return False, "Directional entries are blocked after 14:30 IST to avoid low-quality late-session deployment."
    if strategy in {StrategyType.IRON_CONDOR, StrategyType.SHORT_STRANGLE, StrategyType.SHORT_STRADDLE} and now.time() < time(10, 0):
        return False, "Iron Condor / Short Strangle / Short Straddle entries are allowed only after 10:00 IST."
    if now.time() >= TIME_EXIT:
        return False, "No new entries are allowed after the 15:15 IST flattening cut-off."
    return True, None


def validate_entry_context(
    strategy: StrategyType,
    snapshot: MarketSnapshot,
    regime_state: RegimeState,
) -> tuple[bool, str | None]:
    if strategy != StrategyType.BEAR_CALL_CREDIT_SPREAD:
        return True, None
    metadata = regime_state.metadata if isinstance(regime_state.metadata, dict) else {}
    playbook = str(metadata.get("playbook") or "UNKNOWN")
    pcr_trend = str(metadata.get("pcr_trend") or "UNKNOWN")
    if playbook != "GAP_DOWN_BEARISH_CONTINUATION" and pcr_trend == "FALLING":
        return False, "PCR_TREND_FALLING_BEARISH_NEGATION"

    bars = session_bars(snapshot.nifty_5m)
    if not bars:
        return True, None
    entry_candle = bars[-1]
    vwap = snapshot.live_vwap
    if vwap is None:
        vwap = regime_state.vwap
    if vwap is None:
        vwap = compute_vwap(bars)
    if (
        playbook != "GAP_DOWN_BEARISH_CONTINUATION"
        and vwap is not None
        and vwap > 0
        and entry_candle.close > (vwap * 1.002)
    ):
        return False, "PRICE_ABOVE_VWAP_AT_ENTRY"
    return True, None


def simulate_entry_credit(structure: TradeStructure, slippage_points: float) -> float:
    return max(structure.credit_points - slippage_points, 0.0)


def compute_structure_spread_pts(structure, option_chain) -> float:
    """Sum of half bid-ask spreads across all legs — actual market impact for limit orders.

    Returns 0.0 when bid/ask unavailable (falls back gracefully; caller uses fixed slippage).
    """
    total = 0.0
    for leg in structure.legs:
        quote = option_chain.find_quote(leg.strike, leg.option_type)
        if quote is not None and quote.ask > 0 and quote.bid > 0:
            total += (quote.ask - quote.bid) / 2.0
    return round(total, 2)


def build_open_position(
    structure: TradeStructure,
    lots: int,
    lot_size: int,
    entry_time: datetime,
    entry_credit_points: float,
    max_loss_rupees_per_lot: float,
    extra_metadata: dict[str, object] | None = None,
) -> OpenPosition:
    tp_capture = DIRECTIONAL_TP_CAPTURE if structure.strategy not in {StrategyType.IRON_CONDOR, StrategyType.SHORT_STRANGLE, StrategyType.SHORT_STRADDLE} else CONDOR_TP_CAPTURE
    # IV rank-based TP scaling: high IV → hold longer (more premium to decay, wider targets);
    # compressed IV → exit sooner (less edge available, favour quick capture).
    _iv_rank = _read_current_iv_rank()
    if _iv_rank is not None:
        if _iv_rank > 65:
            tp_capture = min(tp_capture * 1.15, 0.85)   # e.g. 65% → ~75%
        elif _iv_rank < 30:
            tp_capture = max(tp_capture * 0.85, 0.45)   # e.g. 65% → ~55%
    target_value = entry_credit_points * (1.0 - tp_capture)
    stop_value = entry_credit_points * PREMIUM_SL_MULTIPLIER
    metadata = {
        "time_exit": entry_time.replace(hour=15, minute=15, second=0, microsecond=0).isoformat(),
        "session_profit_peak_rupees": 0.0,
        "exit_mode": "STANDARD_TRAIL",
        "iv_rank_at_entry": _iv_rank,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return OpenPosition(
        structure=structure,
        lots=lots,
        lot_size=lot_size,
        entry_time=entry_time,
        entry_credit_points=entry_credit_points,
        target_value_points=target_value,
        stop_value_points=stop_value,
        max_loss_rupees_per_lot=max_loss_rupees_per_lot,
        take_profit_capture_pct=tp_capture,
        metadata=metadata,
    )


def build_trade_decision(
    structure: TradeStructure,
    regime: RegimeLabel | str,
    rationale: list[str],
    confidence_score: float,
    entry_time: datetime,
    lots: int,
    lot_size: int,
    max_loss_rupees_per_lot: float,
    slippage_points: float,
    extra_metadata: dict[str, object] | None = None,
) -> DecisionOutput:
    regime_label = regime if isinstance(regime, RegimeLabel) else RegimeLabel(regime)
    entry_credit_points = simulate_entry_credit(structure, slippage_points=slippage_points)
    tp_capture = DIRECTIONAL_TP_CAPTURE if structure.strategy not in {StrategyType.IRON_CONDOR, StrategyType.SHORT_STRANGLE, StrategyType.SHORT_STRADDLE} else CONDOR_TP_CAPTURE
    delta_sl = DIRECTIONAL_DELTA_SL if structure.strategy not in {StrategyType.IRON_CONDOR, StrategyType.SHORT_STRANGLE, StrategyType.SHORT_STRADDLE} else CONDOR_DELTA_SL
    playbook = str(extra_metadata.get("playbook")) if extra_metadata else ""
    if structure.strategy == StrategyType.BULL_PUT_CREDIT_SPREAD and playbook in {
        "OPEN_DRIVE_BULLISH",
        "HIGH_CONFLUENCE_BULLISH_CONTINUATION",
        "EARLY_BALANCE_BULLISH_RECLAIM",
        "SIDEWAYS_TO_BULLISH_RECLAIM",
        "GAP_UP_BULLISH_CONTINUATION",
        "GAP_DOWN_BULLISH_RECOVERY",
    }:
        delta_sl = BULLISH_PLAYBOOK_DELTA_SL
    legs = [_serialize_leg(leg, lots=lots, lot_size=lot_size) for leg in structure.legs]
    metadata = {
        "structure_credit_points": round(structure.credit_points, 4),
        "structure_width_points": round(structure.width_points, 4),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return DecisionOutput(
        action="TRADE",
        strategy=structure.strategy,
        regime=regime_label,
        rationale=rationale + structure.rationale,
        confidence_score=confidence_score,
        entry={
            "timestamp": entry_time.isoformat(),
            "order_type": "LIMIT",
            "reference_price": "MID",
            "expected_credit_points": round(entry_credit_points, 4),
            "slippage_points": slippage_points,
        },
        stop_loss={
            "premium_multiple": PREMIUM_SL_MULTIPLIER,
            "spread_value_points": round(entry_credit_points * PREMIUM_SL_MULTIPLIER, 4),
            "short_delta_abs": delta_sl,
        },
        take_profit={
            "credit_capture_pct": tp_capture,
            "spread_value_points": round(entry_credit_points * (1.0 - tp_capture), 4),
        },
        time_exit=entry_time.replace(hour=15, minute=15, second=0, microsecond=0).isoformat(),
        max_loss_rupees_per_lot=round(max_loss_rupees_per_lot, 2),
        lots=lots,
        legs=legs,
        metadata=metadata,
    )


def build_no_trade_decision(
    regime: RegimeLabel | str,
    rationale: list[str],
    *,
    confidence_score: float = 0.0,
    extra_metadata: dict[str, object] | None = None,
) -> DecisionOutput:
    try:
        regime_label = regime if isinstance(regime, RegimeLabel) else RegimeLabel(regime)
    except ValueError:
        regime_label = RegimeLabel.NO_TRADE
    metadata: dict[str, object] = {}
    if extra_metadata:
        metadata.update(extra_metadata)
    return DecisionOutput(
        action="NO_TRADE",
        strategy=StrategyType.NO_TRADE,
        regime=regime_label,
        rationale=rationale,
        confidence_score=confidence_score,
        entry={},
        stop_loss={},
        take_profit={},
        time_exit="",
        max_loss_rupees_per_lot=0.0,
        lots=0,
        legs=[],
        metadata=metadata,
    )


def mark_to_market_value_points(position: OpenPosition, quotes_by_leg: list[StrategyLeg]) -> float:
    liability = 0.0
    for leg in quotes_by_leg:
        mark = leg.quote.mid_price or leg.quote.ltp
        if mark <= 0:
            continue
        if leg.action == "SELL":
            liability += mark
        else:
            liability -= mark
    return max(liability, 0.0)


def evaluate_exit(
    position: OpenPosition,
    current_structure: TradeStructure | None = None,
    *,
    current_snapshot: MarketSnapshot | None = None,
    current_regime: RegimeState | None = None,
    now: datetime,
) -> ExitDecision:
    current_value_points, current_legs = _current_position_mark(position, current_snapshot, current_structure)
    if now.time() >= TIME_EXIT:
        liability = current_value_points if current_value_points is not None else position.stop_value_points
        pnl_rupees = (position.entry_credit_points - liability) * position.lot_size * position.lots
        return ExitDecision(True, "TIME_EXIT", liability, pnl_rupees)

    entry_time = position.entry_time
    if entry_time.tzinfo is None and now.tzinfo is not None:
        entry_time = entry_time.replace(tzinfo=now.tzinfo)
    elif entry_time.tzinfo is not None and now.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=None)
    elapsed_minutes = max(int((now - entry_time).total_seconds() // 60), 0)

    quick_invalidation_reason = _intrabar_regime_invalidation_reason(position, current_snapshot, now=now)
    if quick_invalidation_reason:
        liability = current_value_points if current_value_points is not None else position.stop_value_points
        pnl_rupees = (position.entry_credit_points - liability) * position.lot_size * position.lots
        return ExitDecision(True, quick_invalidation_reason, liability, pnl_rupees)

    invalidation_reason = _regime_invalidation_reason(position, current_snapshot, current_regime, minutes_since_entry=elapsed_minutes)
    if invalidation_reason:
        liability = current_value_points if current_value_points is not None else position.stop_value_points
        pnl_rupees = (position.entry_credit_points - liability) * position.lot_size * position.lots
        return ExitDecision(True, invalidation_reason, liability, pnl_rupees)

    # IV expansion exit: a ≥20% jump in average chain IV after entry signals a regime shift —
    # the market is repricing uncertainty, which works against short-premium positions because
    # vega losses accelerate faster than theta accrues. Gate: wait 15 min to avoid early noise.
    if elapsed_minutes >= 15 and current_snapshot is not None:
        entry_avg_iv = position.metadata.get("avg_chain_iv")
        if entry_avg_iv is not None:
            current_ivs = [
                float(q.iv)
                for q in current_snapshot.option_chain.quotes
                if q.iv is not None and q.ltp > 0
            ]
            if current_ivs:
                current_avg_iv = sum(current_ivs) / len(current_ivs)
                if current_avg_iv >= float(entry_avg_iv) * 1.20:
                    liability = current_value_points if current_value_points is not None else position.stop_value_points
                    pnl_rupees = (position.entry_credit_points - liability) * position.lot_size * position.lots
                    return ExitDecision(True, "IV_EXPANSION_EXIT", liability, pnl_rupees)

    if current_value_points is None:
        return ExitDecision(False, "MISSING_QUOTES", 0.0, 0.0)

    pnl_rupees = (position.entry_credit_points - current_value_points) * position.lot_size * position.lots
    profit_trail_reason = _session_profit_trail_reason(position, current_snapshot, pnl_rupees)
    if profit_trail_reason:
        return ExitDecision(True, profit_trail_reason, current_value_points, pnl_rupees)
    structure_trail_reason = _structure_profit_trail_reason(position, current_snapshot, current_value_points)
    if structure_trail_reason:
        return ExitDecision(True, structure_trail_reason, current_value_points, pnl_rupees)
    mfe_trail_reason = _mfe_profit_trail_reason(position, current_value_points, elapsed_minutes)
    if mfe_trail_reason:
        return ExitDecision(True, mfe_trail_reason, current_value_points, pnl_rupees)
    effective_target_capture = position.take_profit_capture_pct
    if current_snapshot is not None:
        current_capture_pct = max(position.entry_credit_points - current_value_points, 0.0) / max(position.entry_credit_points, 0.01)
        if should_use_conviction_exit(
            position,
            current_snapshot,
            minutes_since_entry=elapsed_minutes,
            current_legs=current_legs,
            current_regime=current_regime,
        ):
            effective_target_capture = max(effective_target_capture, CONVICTION_TP_CAPTURE)
            position.metadata["exit_mode"] = "CONVICTION_TRAIL"
        elif position.metadata.get("playbook") == "GAP_DOWN_BEARISH_CONTINUATION":
            if current_capture_pct >= 0.70 and now.time() < time(12, 0):
                effective_target_capture = max(effective_target_capture, CONVICTION_TP_CAPTURE)
                position.metadata["exit_mode"] = "GAP_DOWN_FAST_DECAY_TRAIL"
            else:
                position.metadata["exit_mode"] = "STANDARD_TRAIL"
            if current_capture_pct < 0.40 and now.time() >= time(13, 0):
                return ExitDecision(True, "GAP_DOWN_SLOW_DECAY_TIME_EXIT", current_value_points, pnl_rupees)
        else:
            position.metadata["exit_mode"] = "STANDARD_TRAIL"
    effective_target_value_points = position.entry_credit_points * (1.0 - effective_target_capture)
    if current_value_points <= effective_target_value_points:
        return ExitDecision(True, "TAKE_PROFIT", current_value_points, pnl_rupees)
    if current_value_points >= position.stop_value_points:
        return ExitDecision(True, "PREMIUM_STOP", current_value_points, pnl_rupees)

    # Theta-aware profit protection: professional rule — don't let a good trade turn bad.
    # After 90 min with 60% captured, gamma starts working against the seller faster than
    # theta works for them. Close the trade rather than risk giving it back.
    current_capture_pct = max(position.entry_credit_points - current_value_points, 0.0) / max(position.entry_credit_points, 0.01)
    if current_capture_pct >= 0.60 and elapsed_minutes >= 90:
        return ExitDecision(True, "THETA_TARGET_HIT", current_value_points, pnl_rupees)
    # Afternoon protection: graduated thresholds — as the close approaches, a smaller
    # captured gain is worth locking in because holding through last-hour volatility
    # risks giving it back. After 14:00 we only need 25% captured; after 13:30 we need 40%.
    if current_capture_pct >= 0.25 and now.time() >= time(14, 0):
        return ExitDecision(True, "AFTERNOON_PROFIT_LOCK", current_value_points, pnl_rupees)
    if current_capture_pct >= 0.40 and now.time() >= time(13, 30):
        return ExitDecision(True, "AFTERNOON_PROFIT_LOCK", current_value_points, pnl_rupees)
    # Delta-decay exit: when the short put/call delta drops to near-zero the spread has
    # extracted most of its theta. Exit cleanly rather than holding to TIME_EXIT.
    # Gate: at least 60 min held so entry-day high-delta situations don't trigger this.
    if elapsed_minutes >= 60 and position.structure.strategy in {StrategyType.BULL_PUT_CREDIT_SPREAD, StrategyType.BEAR_CALL_CREDIT_SPREAD}:
        short_deltas_decay = [abs(leg.quote.delta) for leg in current_legs if leg.action == "SELL" and leg.quote.delta is not None]
        if short_deltas_decay and max(short_deltas_decay) <= 0.08:
            return ExitDecision(True, "DELTA_DECAY_EXIT", current_value_points, pnl_rupees)

    # Condor partial close: when one side is threatened, close just that side rather
    # than exiting the full condor. This preserves the safe half which still has
    # theta working in its favour. Trigger is 0.25 delta (before full DELTA_STOP at 0.25).
    CONDOR_PARTIAL_CLOSE_DELTA = 0.23
    if position.structure.strategy == StrategyType.IRON_CONDOR and current_legs:
        call_short_deltas = [
            abs(leg.quote.delta)
            for leg in current_legs
            if leg.action == "SELL" and leg.option_type == OptionType.CALL and leg.quote.delta is not None
        ]
        put_short_deltas = [
            abs(leg.quote.delta)
            for leg in current_legs
            if leg.action == "SELL" and leg.option_type == OptionType.PUT and leg.quote.delta is not None
        ]
        if call_short_deltas and max(call_short_deltas) >= CONDOR_PARTIAL_CLOSE_DELTA:
            return ExitDecision(True, "CONDOR_CALL_SIDE_CLOSE", current_value_points, pnl_rupees)
        if put_short_deltas and max(put_short_deltas) >= CONDOR_PARTIAL_CLOSE_DELTA:
            return ExitDecision(True, "CONDOR_PUT_SIDE_CLOSE", current_value_points, pnl_rupees)

    short_delta_limit = DIRECTIONAL_DELTA_SL if position.structure.strategy not in {StrategyType.IRON_CONDOR, StrategyType.SHORT_STRANGLE, StrategyType.SHORT_STRADDLE} else CONDOR_DELTA_SL
    selection_mode = position.structure.metadata.get("selection_mode")
    if position.structure.strategy == StrategyType.BEAR_CALL_CREDIT_SPREAD and selection_mode == "STRUCTURE_DISTANCE_FALLBACK":
        short_delta_limit = 1.01
    if position.structure.strategy == StrategyType.BULL_PUT_CREDIT_SPREAD and position.metadata.get("playbook") in {
        "OPEN_DRIVE_BULLISH",
        "HIGH_CONFLUENCE_BULLISH_CONTINUATION",
        "EARLY_BALANCE_BULLISH_RECLAIM",
        "SIDEWAYS_TO_BULLISH_RECLAIM",
        "GAP_UP_BULLISH_CONTINUATION",
        "GAP_DOWN_BULLISH_RECOVERY",
        "RANGE_NO_TRADE",
        "EARLY_STRUCTURE_BULLISH",
    }:
        short_delta_limit = BULLISH_PLAYBOOK_DELTA_SL
    short_deltas = [abs(leg.quote.delta) for leg in current_legs if leg.action == "SELL" and leg.quote.delta is not None]
    if short_deltas and max(short_deltas) >= short_delta_limit:
        # Minimum hold time before DELTA_STOP:
        # - Directional spreads: 10 min (was 3 — delta noise in first few candles is not a signal)
        # - Theta strategies (condor, strangle, straddle): 20 min — position must breathe
        #   before delta noise from the first few candles triggers an exit
        _is_theta_strategy = position.structure.strategy in {
            StrategyType.IRON_CONDOR, StrategyType.SHORT_STRANGLE, StrategyType.SHORT_STRADDLE
        }
        _max_profit_r = position.entry_credit_points * position.lot_size * position.lots
        _min_hold = 20 if _is_theta_strategy else 10
        if elapsed_minutes < _min_hold:
            pass  # hold
        elif (
            position.structure.strategy == StrategyType.BULL_PUT_CREDIT_SPREAD
            and position.metadata.get("playbook") in {
                "OPEN_DRIVE_BULLISH",
                "HIGH_CONFLUENCE_BULLISH_CONTINUATION",
                "EARLY_BALANCE_BULLISH_RECLAIM",
                "SIDEWAYS_TO_BULLISH_RECLAIM",
                "GAP_UP_BULLISH_CONTINUATION",
                "GAP_DOWN_BULLISH_RECOVERY",
                "RANGE_NO_TRADE",
                "EARLY_STRUCTURE_BULLISH",
                "SCORE_DRIVEN_BULL",
            }
            and pnl_rupees > -(_max_profit_r * 0.25)  # hold unless lost >25% of max credit
        ):
            return ExitDecision(False, "HOLD", current_value_points, pnl_rupees)
        else:
            return ExitDecision(True, "DELTA_STOP", current_value_points, pnl_rupees)

    return ExitDecision(False, "HOLD", current_value_points, pnl_rupees)


def _current_position_mark(
    position: OpenPosition,
    current_snapshot: MarketSnapshot | None,
    current_structure: TradeStructure | None,
) -> tuple[float | None, list[StrategyLeg]]:
    if current_snapshot is not None:
        repriced_legs: list[StrategyLeg] = []
        for leg in position.structure.legs:
            quote = current_snapshot.option_chain.find_quote(leg.strike, leg.option_type)
            if quote is None:
                return None, []
            repriced_legs.append(
                StrategyLeg(
                    action=leg.action,
                    option_type=leg.option_type,
                    strike=leg.strike,
                    quote=quote,
                )
            )
        return mark_to_market_value_points(position, repriced_legs), repriced_legs

    if current_structure is not None:
        return current_structure.credit_points, current_structure.legs
    return None, []


def _regime_invalidation_reason(
    position: OpenPosition,
    current_snapshot: MarketSnapshot | None,
    current_regime: RegimeState | None,
    *,
    minutes_since_entry: int = 0,
) -> str | None:
    if current_snapshot is None:
        return None
    bars = session_bars(current_snapshot.nifty_5m)
    if len(bars) < 2:
        return None
    vwap = current_snapshot.live_vwap
    if vwap is None and current_regime is not None:
        vwap = current_regime.vwap
    if vwap is None:
        vwap = compute_vwap(bars)
    if vwap is None or vwap <= 0:
        return None

    spot = current_snapshot.option_chain.spot
    strategy = position.structure.strategy
    # Closed-bar checks (n=2) use pre-existing bar history and can fire immediately
    # after entry if the condition was already true before we entered. Require at least
    # one complete 5-minute candle (5 minutes) to have formed since entry before
    # triggering these — this prevents the thesis being invalidated by stale bars.
    closed_bar_check_allowed = minutes_since_entry >= 5
    if strategy == StrategyType.BULL_PUT_CREDIT_SPREAD:
        if closed_bar_check_allowed and spot < vwap and last_n_closes_below(vwap, bars, n=2):
            return "VWAP_INVALIDATION"
    elif strategy == StrategyType.BEAR_CALL_CREDIT_SPREAD:
        ema20_5m = ema_value(closes(bars), period=20)
        if closed_bar_check_allowed and (
            ema20_5m is not None
            and last_n_closes_above(ema20_5m, bars, n=2)
            and (vwap is None or last_n_closes_above(vwap, bars, n=2))
        ):
            return "EMA20_INVALIDATION"
        if closed_bar_check_allowed and (
            bullish_reversal_structure(bars)
            and ema20_5m is not None
            and bars[-1].close > ema20_5m
            and (vwap is None or last_n_closes_above(vwap, bars, n=2))
        ):
            return "REVERSAL_STRUCTURE"
        if closed_bar_check_allowed and spot > vwap and last_n_closes_above(vwap, bars, n=2):
            return "VWAP_INVALIDATION"
    elif strategy in {StrategyType.IRON_CONDOR, StrategyType.SHORT_STRANGLE, StrategyType.SHORT_STRADDLE}:
        # Regime change is a live signal, not a bar check — fire immediately.
        if current_regime is not None and current_regime.regime != RegimeLabel.RANGE:
            return "RANGE_INVALIDATION"
    return None


def should_use_conviction_exit(
    position: OpenPosition,
    snapshot: MarketSnapshot,
    *,
    minutes_since_entry: int,
    current_legs: list[StrategyLeg] | None = None,
    current_regime: RegimeState | None = None,
) -> bool:
    if position.structure.strategy != StrategyType.BEAR_CALL_CREDIT_SPREAD or minutes_since_entry < 30:
        return False
    bars = session_bars(snapshot.nifty_5m)
    closes_5m = closes(bars)
    spot = snapshot.option_chain.spot
    if spot <= 0:
        return False
    ema20_slope = 0.0
    ema50_slope = 0.0
    ema_spacing = 0.0
    if current_regime is not None and isinstance(current_regime.metadata, dict):
        ema20_slope = float(current_regime.metadata.get("ema20_slope_5m") or 0.0) / spot
        ema50_slope = float(current_regime.metadata.get("ema50_slope_5m") or 0.0) / spot
        ema_spacing = abs(float(current_regime.metadata.get("ema_distance_pct_5m") or 0.0)) / 100.0
    elif len(closes_5m) >= 100:
        ema20_values = ema(closes_5m, period=20)
        ema50_values = ema(closes_5m, period=50)
        ema100_values = ema(closes_5m, period=100)
        if min(len(ema20_values), len(ema50_values), len(ema100_values)) < 4:
            return False
        ema20_slope = (ema20_values[-1] - ema20_values[-4]) / spot
        ema50_slope = (ema50_values[-1] - ema50_values[-4]) / spot
        ema_spacing = ((abs(ema20_values[-1] - ema50_values[-1]) + abs(ema50_values[-1] - ema100_values[-1])) / spot)
    else:
        return False
    vwap = snapshot.live_vwap if snapshot.live_vwap is not None else compute_vwap(bars)
    if vwap is None or vwap <= 0:
        return False
    live_legs = current_legs or []
    short_deltas = [abs(leg.quote.delta) for leg in live_legs if leg.action == "SELL" and leg.quote.delta is not None]
    max_short_delta = max(short_deltas, default=0.0)
    return bool(
        ema20_slope < -0.0003
        and ema50_slope < -0.0002
        and ema_spacing > 0.005
        and spot < vwap
        and max_short_delta < 0.15
    )


def _intrabar_regime_invalidation_reason(
    position: OpenPosition,
    current_snapshot: MarketSnapshot | None,
    *,
    now: datetime,
) -> str | None:
    if current_snapshot is None or position.structure.strategy != StrategyType.BEAR_CALL_CREDIT_SPREAD:
        return None
    last_check_raw = position.metadata.get("last_intrabar_regime_check_at")
    if last_check_raw:
        try:
            last_check = datetime.fromisoformat(str(last_check_raw))
        except ValueError:
            last_check = None
        if last_check is not None:
            if last_check.tzinfo is None and now.tzinfo is not None:
                last_check = last_check.replace(tzinfo=now.tzinfo)
            elif last_check.tzinfo is not None and now.tzinfo is None:
                last_check = last_check.replace(tzinfo=None)
        if last_check is not None and (now - last_check).total_seconds() < 120:
            return None
    position.metadata["last_intrabar_regime_check_at"] = now.isoformat()
    bars = session_bars(current_snapshot.nifty_5m)
    if len(bars) < 4:
        return None
    vwap = current_snapshot.live_vwap if current_snapshot.live_vwap is not None else compute_vwap(bars)
    ema20_5m = ema_value(closes(bars), period=20)
    last = bars[-1]
    if (
        vwap is not None
        and ema20_5m is not None
        and current_snapshot.option_chain.spot > vwap
        and last.close > ema20_5m
        and last.close > last.open
        and bullish_reversal_structure(bars[-6:])
    ):
        return "INTRABAR_REGIME_INVALIDATION"
    return None


def _session_profit_trail_reason(
    position: OpenPosition,
    current_snapshot: MarketSnapshot | None,
    open_pnl_rupees: float,
) -> str | None:
    if current_snapshot is None:
        return None
    if position.metadata.get("playbook") == "GAP_DOWN_BEARISH_CONTINUATION":
        return None
    session_realized = current_snapshot.account_state.realised_pnl_rupees
    combined_pnl = session_realized + open_pnl_rupees
    prior_peak = float(position.metadata.get("session_profit_peak_rupees", 0.0))
    peak = max(prior_peak, combined_pnl)
    position.metadata["session_profit_peak_rupees"] = peak
    trail_arm = DAILY_PROFIT_TRAIL_ARM_RUPEES
    trail_giveback = DAILY_PROFIT_TRAIL_GIVEBACK_RUPEES
    if (
        position.structure.strategy == StrategyType.BULL_PUT_CREDIT_SPREAD
        and position.metadata.get("playbook") in {"SIDEWAYS_TO_BULLISH_RECLAIM", "HIGH_CONFLUENCE_BULLISH_CONTINUATION"}
    ):
        trail_arm = BULLISH_PLAYBOOK_PROFIT_TRAIL_ARM_RUPEES
        trail_giveback = BULLISH_PLAYBOOK_PROFIT_TRAIL_GIVEBACK_RUPEES
    if peak < trail_arm:
        return None
    trailing_floor = max(trail_arm, peak - trail_giveback)
    if combined_pnl <= trailing_floor:
        return "DAILY_PROFIT_TRAIL"
    return None


def _structure_profit_trail_reason(
    position: OpenPosition,
    current_snapshot: MarketSnapshot | None,
    current_value_points: float,
) -> str | None:
    if current_snapshot is None or position.entry_credit_points <= 0:
        return None
    capture_pct = max(position.entry_credit_points - current_value_points, 0.0) / position.entry_credit_points
    if capture_pct < PROFIT_TRAIL_CAPTURE_ARM:
        return None
    best_value_points = float(position.metadata.get("best_value_points", position.entry_credit_points))
    best_value_points = min(best_value_points, current_value_points)
    position.metadata["best_value_points"] = best_value_points
    if current_value_points <= best_value_points:
        return None

    bars = session_bars(current_snapshot.nifty_5m)
    if len(bars) < 6:
        return None
    last_close = bars[-1].close
    ema20_5m = ema_value(closes(bars), period=20)
    giveback_triggered = current_value_points >= (best_value_points + PROFIT_TRAIL_GIVEBACK_POINTS)
    if not giveback_triggered:
        return None

    if position.structure.strategy == StrategyType.BULL_PUT_CREDIT_SPREAD:
        pivot_low = latest_pivot_low(bars, lookback=10)
        structure_broken = (
            (ema20_5m is not None and last_n_closes_below(ema20_5m, bars, n=2))
            or (pivot_low is not None and last_close < pivot_low)
        )
        if structure_broken:
            return "PROFIT_TRAIL_STRUCTURE"
    elif position.structure.strategy == StrategyType.BEAR_CALL_CREDIT_SPREAD:
        pivot_high = latest_pivot_high(bars, lookback=10)
        structure_broken = (
            (ema20_5m is not None and last_n_closes_above(ema20_5m, bars, n=2))
            or (pivot_high is not None and last_close > pivot_high)
        )
        if structure_broken:
            return "PROFIT_TRAIL_STRUCTURE"
    return None


def _mfe_profit_trail_reason(
    position: OpenPosition,
    current_value_points: float,
    elapsed_minutes: int,
) -> str | None:
    """Proportional MFE trail: once 55%+ of premium is captured, never give back more than 40% of that peak.

    Complements the absolute-point structure trail: a 100-pt credit and a 40-pt credit both
    get protected proportionally rather than by a single fixed giveback.  Gate: 20 min minimum
    hold so the trade can breathe past initial noise.
    """
    if position.entry_credit_points <= 0 or elapsed_minutes < 20:
        return None
    current_capture = max(0.0, position.entry_credit_points - current_value_points)
    current_capture_pct = current_capture / position.entry_credit_points
    best_pct = float(position.metadata.get("mfe_capture_pct", 0.0))
    best_pct = max(best_pct, current_capture_pct)
    position.metadata["mfe_capture_pct"] = round(best_pct, 4)
    if best_pct < 0.55:
        return None
    if current_capture_pct < best_pct * 0.60:
        return "MFE_TRAIL_EXIT"
    return None


def evaluate_hedge_opportunity(
    position: OpenPosition,
    current_regime: RegimeState,
    current_snapshot: MarketSnapshot,
    now: datetime,
) -> dict:
    """
    Detects when a directional credit spread should be converted to an Iron Condor
    by adding the opposite leg. This fires when:
      - We hold a bear call spread and the market has stalled/turned range-bound
        (regime → RANGE, price recovering, but short call still safely OTM)
      - We hold a bull put spread and the market has stalled/turned range-bound
        (regime → RANGE, price pulling back, but short put still safely OTM)

    Returns a dict with:
      should_hedge: bool
      reason: str
      hedge_side: 'BULL_PUT' | 'BEAR_CALL' | None  (which spread to ADD)
    """
    result = {"should_hedge": False, "reason": "NO_HEDGE", "hedge_side": None}
    strategy = position.structure.strategy
    if strategy not in {StrategyType.BEAR_CALL_CREDIT_SPREAD, StrategyType.BULL_PUT_CREDIT_SPREAD}:
        return result

    # Already too late in session to open a new leg
    if now.time() >= time(13, 0):
        return result

    # Must be in position long enough to have a clear picture
    entry_time = position.entry_time
    if entry_time.tzinfo is None and now.tzinfo is not None:
        entry_time = entry_time.replace(tzinfo=now.tzinfo)
    elif entry_time.tzinfo is not None and now.tzinfo is None:
        entry_time = entry_time.replace(tzinfo=None)
    elapsed_minutes = max(int((now - entry_time).total_seconds() // 60), 0)
    if elapsed_minutes < 20:
        return result

    # Must have already captured some profit (position is working)
    bars = session_bars(current_snapshot.nifty_5m)
    if not bars:
        return result
    spot = current_snapshot.option_chain.spot
    vwap = current_snapshot.live_vwap or compute_vwap(bars)
    ema20_5m = ema_value(closes(bars), period=20)

    regime = current_regime.regime
    metadata = current_regime.metadata if isinstance(current_regime.metadata, dict) else {}
    range_balance_score = float(metadata.get("range_balance_score") or 0.0)

    if strategy == StrategyType.BEAR_CALL_CREDIT_SPREAD:
        # Market stalled: regime shifted to RANGE and price is holding below our short call
        short_call_strike = min(
            (leg.strike for leg in position.structure.legs if leg.action == "SELL" and leg.option_type == OptionType.CALL),
            default=None,
        )
        if short_call_strike is None:
            return result
        safe_margin = (short_call_strike - spot) / max(spot, 1.0)
        market_range_bound = (
            regime == RegimeLabel.RANGE
            or (vwap is not None and ema20_5m is not None and abs(spot - vwap) / max(spot, 1.0) <= 0.0020)
        )
        if (
            market_range_bound
            and safe_margin >= 0.008  # short call at least 0.8% above spot = safely OTM
            and range_balance_score >= 2.5
        ):
            result["should_hedge"] = True
            result["reason"] = "BEAR_CALL_MARKET_STALLED_ADD_BULL_PUT"
            result["hedge_side"] = "BULL_PUT"
            return result

    elif strategy == StrategyType.BULL_PUT_CREDIT_SPREAD:
        short_put_strike = max(
            (leg.strike for leg in position.structure.legs if leg.action == "SELL" and leg.option_type == OptionType.PUT),
            default=None,
        )
        if short_put_strike is None:
            return result
        safe_margin = (spot - short_put_strike) / max(spot, 1.0)
        market_range_bound = (
            regime == RegimeLabel.RANGE
            or (vwap is not None and ema20_5m is not None and abs(spot - vwap) / max(spot, 1.0) <= 0.0020)
        )
        if (
            market_range_bound
            and safe_margin >= 0.008
            and range_balance_score >= 2.5
        ):
            result["should_hedge"] = True
            result["reason"] = "BULL_PUT_MARKET_STALLED_ADD_BEAR_CALL"
            result["hedge_side"] = "BEAR_CALL"
            return result

    return result


def _serialize_leg(leg: StrategyLeg, lots: int, lot_size: int) -> dict[str, object]:
    payload = asdict(leg.quote)
    payload["option_type"] = leg.option_type.value
    payload["action"] = leg.action
    payload["strike"] = leg.strike
    payload["quantity"] = lots * lot_size
    return payload
