from __future__ import annotations

import os
from dataclasses import asdict
from datetime import datetime, time, timedelta, timezone
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
DAILY_PROFIT_TRAIL_ARM_RUPEES = 1200.0   # was 5000 — modest paper-lot winners (Rs800-1200)
DAILY_PROFIT_TRAIL_GIVEBACK_RUPEES = 500.0  # never armed, so winners gave everything back
BULLISH_PLAYBOOK_PROFIT_TRAIL_ARM_RUPEES = 6500.0
BULLISH_PLAYBOOK_PROFIT_TRAIL_GIVEBACK_RUPEES = 2000.0
PROFIT_TRAIL_CAPTURE_ARM = 0.35
PROFIT_TRAIL_GIVEBACK_POINTS = 6.0

_IST = timezone(timedelta(hours=5, minutes=30))  # India has no DST — fixed offset is exact


def _to_ist_naive(dt: datetime) -> datetime:
    """Normalize a datetime to IST wall-clock, tz-naive.

    Aware datetimes are CONVERTED to IST (astimezone) before stripping tzinfo, not
    relabeled — so a UTC-aware value (e.g. a position restored from JSON after a
    restart) does not skew elapsed-time math by 5.5h and bypass/suppress the
    min-hold stops. Naive datetimes are assumed to already be IST wall-clock.
    """
    if dt.tzinfo is not None:
        return dt.astimezone(_IST).replace(tzinfo=None)
    return dt


def _elapsed_minutes(now: datetime, entry_time: datetime) -> int:
    return max(int((_to_ist_naive(now) - _to_ist_naive(entry_time)).total_seconds() // 60), 0)


def validate_entry_time(strategy: StrategyType, now: datetime) -> tuple[bool, str | None]:
    if now.time() < time(9, 15):
        return False, "Entries are not allowed before 09:15 IST."
    if strategy in {StrategyType.BEAR_CALL_CREDIT_SPREAD, StrategyType.BULL_PUT_CREDIT_SPREAD} and now.time() < time(9, 30):
        return False, "Directional entries prefer time >= 09:30 IST."
    if strategy in {StrategyType.BEAR_CALL_CREDIT_SPREAD, StrategyType.BULL_PUT_CREDIT_SPREAD} and now.time() > LATEST_DIRECTIONAL_ENTRY:
        return False, "Directional entries are blocked after 14:30 IST to avoid low-quality late-session deployment."
    if strategy in {StrategyType.IRON_CONDOR, StrategyType.IRON_FLY, StrategyType.SHORT_STRANGLE, StrategyType.SHORT_STRADDLE} and now.time() < time(10, 0):
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
    if structure.metadata.get("is_debit"):
        # Debit spread: credit_points is stored NEGATIVE (=-debit). Slippage worsens
        # the fill (pay more), so the effective entry stays negative. No 0-clamp.
        return structure.credit_points - slippage_points
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


# Real per-lot fees for a Nifty options round trip on a discount broker (brokerage + STT +
# exchange txn + GST + stamp, entry AND exit). Conservative flat estimate.
FEES_PER_LOT_ROUNDTRIP_RUPEES = 25.0
# Never cost a structure below this per lot — guards against a missing/locked quote reading the
# spread as ~0 and letting a thin trade through under-costed.
MIN_ROUND_TRIP_COST_PER_LOT_RUPEES = 25.0


def estimate_round_trip_cost_rupees(structure, option_chain, lots: int, lot_size: int) -> float:
    """STRUCTURE-AWARE round-trip cost, replacing a flat per-lot magic number.

    Sums the ACTUAL per-leg bid/ask spreads (both sides = entry + exit) and adds fees, so a 4-leg
    fly correctly costs ~2x a 2-leg vertical AND an illiquid leg (wide spread) is costed for what it
    really is instead of a blind constant. On liquid Nifty index strikes the spreads are razor-thin
    (measured ~0.4pt round-trip on a fly, ~0.15pt on a vertical), so this is close to the old flat
    number there — its value is adaptivity: it self-adjusts to real liquidity and never silently
    under-costs a thin/illiquid leg. Falls back to the fee floor when bid/ask is unavailable.
    """
    half_spread_pts = compute_structure_spread_pts(structure, option_chain)   # one side, sum of legs
    round_trip_pts = 2.0 * half_spread_pts                                    # entry + exit
    spread_cost = round_trip_pts * lot_size * lots
    fees = FEES_PER_LOT_ROUNDTRIP_RUPEES * lots
    return round(max(spread_cost + fees, MIN_ROUND_TRIP_COST_PER_LOT_RUPEES * lots), 2)


def build_open_position(
    structure: TradeStructure,
    lots: int,
    lot_size: int,
    entry_time: datetime,
    entry_credit_points: float,
    max_loss_rupees_per_lot: float,
    extra_metadata: dict[str, object] | None = None,
) -> OpenPosition:
    tp_capture = DIRECTIONAL_TP_CAPTURE if structure.strategy not in {StrategyType.IRON_CONDOR, StrategyType.IRON_FLY, StrategyType.SHORT_STRANGLE, StrategyType.SHORT_STRADDLE} else CONDOR_TP_CAPTURE
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
    tp_capture = DIRECTIONAL_TP_CAPTURE if structure.strategy not in {StrategyType.IRON_CONDOR, StrategyType.IRON_FLY, StrategyType.SHORT_STRANGLE, StrategyType.SHORT_STRADDLE} else CONDOR_TP_CAPTURE
    delta_sl = DIRECTIONAL_DELTA_SL if structure.strategy not in {StrategyType.IRON_CONDOR, StrategyType.IRON_FLY, StrategyType.SHORT_STRANGLE, StrategyType.SHORT_STRADDLE} else CONDOR_DELTA_SL
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
    # Fallback entry prices by (strike, option_type). When a live quote is stale/
    # absent (mid_price None and ltp<=0), we fall back to the entry price for that
    # leg instead of DROPPING it. Dropping a long (BUY) hedge leg overstated the
    # liability by the hedge's worth and fired premature PREMIUM_STOPs (e.g. a
    # deep-OTM long put quoting 0 near expiry made a 14pt spread mark at 16pt).
    entry_price_by_key: dict[tuple[float, object], float] = {}
    for leg in position.structure.legs:
        entry_price_by_key[(leg.strike, leg.option_type)] = float(
            leg.quote.mid_price or leg.quote.ltp or 0.0
        )
    liability = 0.0
    for leg in quotes_by_leg:
        mark = leg.quote.mid_price or leg.quote.ltp or 0.0
        if mark <= 0:
            mark = entry_price_by_key.get((leg.strike, leg.option_type), 0.0)
        mark = max(float(mark), 0.0)
        if leg.action == "SELL":
            liability += mark
        else:
            liability -= mark
    return max(liability, 0.0)


# ── Debit-spread exit engine (mirror of the credit logic) ───────────────────
DEBIT_TP_CAPTURE = 0.60       # take profit at 60% of max profit
DEBIT_STOP_FRAC = 0.50        # stop when 50% of the debit is lost
DEBIT_MIN_HOLD = 10           # min minutes before the stop can fire (noise guard)
# A neutral theta structure is entered when trend_efficiency < this (low follow-through = range-like).
# Its RANGE_INVALIDATION exit must use the SAME bar, so it only fires when real follow-through returns
# — mirrors the selector's _BUY_MIN_EFFICIENCY so entry and exit agree on what "not a range" means.
_THETA_INVALIDATION_MIN_EFF = 0.50
# ...and the breakout must PERSIST this many consecutive cycles (~30s each) before the exit fires, so a
# one-cycle head-fake across the 0.50 line can't evict a theta trade that then reverts. See the exit.
_THETA_INVALIDATION_MIN_STREAK = 2
# A credit spread's momentum exits (EMA20/VWAP invalidation) only fire once the SHORT leg is genuinely
# tested — |delta| >= this. Below it the short is safe and the trade holds for theta (seller discipline).
_CREDIT_EXIT_MIN_SHORT_DELTA = float(os.environ.get("CREDIT_EXIT_MIN_SHORT_DELTA", "0.30") or 0.30)
DEBIT_TRAIL_ARM = 0.40        # once 40% of max profit is captured, trail
DEBIT_TRAIL_GIVEBACK = 0.35   # exit if the trade gives back 35% of its peak profit
# Rs-denominated debit trail: arm once Rs X in profit; never give back more than Rs Y of peak.
# The %-of-max trail (40%) requires a ~190pt spot move — too far for intraday debit spreads;
# this absolute trail protects real gains regardless of spread width.
DEBIT_RS_TRAIL_ARM = float(os.environ.get('DEBIT_RS_TRAIL_ARM', '600') or 600.0)
DEBIT_RS_TRAIL_GIVEBACK = float(os.environ.get('DEBIT_RS_TRAIL_GIVEBACK', '300') or 300.0)


def _current_debit_value(
    position: OpenPosition,
    current_snapshot: MarketSnapshot | None,
    current_structure: TradeStructure | None,
) -> float | None:
    """Current sellable value of a debit spread in points = long_mark - short_mark."""
    if current_snapshot is None:
        return None
    # If EITHER leg is missing from the chain window, return None (caller marks at
    # breakeven) rather than guessing. An intrinsic-value fallback was tried and
    # REJECTED — it mismatched a real-priced leg (with time value) against an
    # intrinsic-only leg, blowing up P&L by ~-Rs420k on the narrow backtest chain.
    long_mark = short_mark = None
    for leg in position.structure.legs:
        quote = current_snapshot.option_chain.find_quote(leg.strike, leg.option_type)
        if quote is None:
            return None
        mark = quote.mid_price or quote.ltp
        if mark is None or mark <= 0:
            return None
        mark = max(float(mark), 0.0)
        if leg.action == "BUY":
            long_mark = mark
        else:
            short_mark = mark
    if long_mark is None or short_mark is None:
        return None
    return max(long_mark - short_mark, 0.0)


_FADE_MIN_HOLD_MIN = 3   # small breathe so a rejection wick's own noise doesn't insta-stop the fade

# SEL_CHOP patience gate: if the trade hasn't shown Rs200 of profit within 45 minutes, the
# chop regime didn't deliver the expected move — exit cheap rather than bleed to TIME_EXIT.
# The Sep-29 -Rs3,146 (held 335 min, TIME_EXIT) and Aug-12 -Rs2,291 (171 min) were both
# trades that peaked far below Rs200 then slowly bled. Good SEL_CHOP winners (Sep-22
# +Rs1,677, Sep-28 +Rs1,219) showed profit well within 45 min.
_CHOP_PATIENCE_MINUTES = int(os.environ.get("SEL_CHOP_PATIENCE_MIN", "45") or 45)
_CHOP_PATIENCE_MIN_MFE = float(os.environ.get("SEL_CHOP_PATIENCE_MFE", "200") or 200.0)


def _evaluate_chop_patience_exit(
    position: OpenPosition,
    current_snapshot: MarketSnapshot | None,
    current_structure: TradeStructure | None,
    now: datetime,
) -> ExitDecision | None:
    """Exit a SEL_CHOP trade that hasn't shown Rs200 profit after 45 minutes.

    Applies to all strategy types (credit spreads, debit spreads, iron flies).
    Returns None when the patience check passes or the playbook is not SEL_CHOP.
    """
    if position.metadata.get("playbook") != "SEL_CHOP":
        return None
    elapsed = _elapsed_minutes(now, position.entry_time)
    if elapsed < _CHOP_PATIENCE_MINUTES:
        return None
    if position.structure.strategy in {StrategyType.CALL_DEBIT_SPREAD, StrategyType.PUT_DEBIT_SPREAD}:
        entry_debit = abs(position.entry_credit_points)
        value = _current_debit_value(position, current_snapshot, current_structure)
        if value is None:
            value = entry_debit
        pnl = (value - entry_debit) * position.lot_size * position.lots
    else:
        value, _ = _current_position_mark(position, current_snapshot, current_structure)
        if value is None:
            return None
        pnl = (position.entry_credit_points - value) * position.lot_size * position.lots
    peak = max(float(position.metadata.get("chop_peak_pnl_rupees") or 0.0), pnl)
    position.metadata["chop_peak_pnl_rupees"] = peak
    if peak < _CHOP_PATIENCE_MIN_MFE:
        return ExitDecision(True, "CHOP_PATIENCE_EXIT", value, pnl)
    return None



def _evaluate_fade_exit(
    position: OpenPosition,
    current_snapshot: MarketSnapshot | None,
    current_structure: TradeStructure | None,
    now: datetime,
) -> ExitDecision:
    """Range-fade scalp exit — SPOT-based, not option-value based. Take the swing at the target (into
    the range interior) or cut cheap at the stop (the faded level broke the wrong way). A PUT_DEBIT is
    a FADE-SHORT (profit as spot falls); a CALL_DEBIT is a FADE-LONG (profit as spot rises)."""
    md = position.metadata or {}
    target = md.get("fade_target_spot")
    stop = md.get("fade_stop_spot")
    entry_debit = abs(position.entry_credit_points)
    value = _current_debit_value(position, current_snapshot, current_structure)
    if value is None:
        value = entry_debit
    pnl = (value - entry_debit) * position.lot_size * position.lots
    if now.time() >= TIME_EXIT:
        return ExitDecision(True, "FADE_TIME_EXIT", value, pnl)
    spot = float(current_snapshot.option_chain.spot) if current_snapshot else None
    if spot is None or target is None or stop is None:
        return ExitDecision(False, "HOLD", value, pnl)
    if _elapsed_minutes(now, position.entry_time) < _FADE_MIN_HOLD_MIN:
        return ExitDecision(False, "HOLD", value, pnl)
    is_short = position.structure.strategy == StrategyType.PUT_DEBIT_SPREAD
    if is_short:                                   # fade-short: target is BELOW, stop is ABOVE the level
        if spot <= float(target):
            return ExitDecision(True, "FADE_TARGET", value, pnl)
        if spot >= float(stop):
            return ExitDecision(True, "FADE_STOP", value, pnl)
    else:                                          # fade-long: target ABOVE, stop BELOW
        if spot >= float(target):
            return ExitDecision(True, "FADE_TARGET", value, pnl)
        if spot <= float(stop):
            return ExitDecision(True, "FADE_STOP", value, pnl)
    return ExitDecision(False, "HOLD", value, pnl)


def _evaluate_debit_exit(
    position: OpenPosition,
    current_snapshot: MarketSnapshot | None,
    current_structure: TradeStructure | None,
    now: datetime,
) -> ExitDecision:
    """Debit spread P&L is the mirror of a credit spread: you pay a debit and profit
    when the spread's value RISES. pnl = (current_value - entry_debit) * size.
    Exits: profit target (% of max profit), hard stop (% of debit lost), profit
    trail once armed, and the 15:15 flatten."""
    entry_debit = abs(position.entry_credit_points)  # stored negative
    width = position.structure.width_points
    max_profit = max(width - entry_debit, 0.0)
    value = _current_debit_value(position, current_snapshot, current_structure)
    if value is None:
        value = entry_debit  # no data → mark at breakeven
    pnl = (value - entry_debit) * position.lot_size * position.lots
    profit_captured = value - entry_debit

    if now.time() >= TIME_EXIT:
        return ExitDecision(True, "TIME_EXIT", value, pnl)

    elapsed_minutes = _elapsed_minutes(now, position.entry_time)

    # Profit target: captured >= 60% of max profit
    if max_profit > 0 and profit_captured >= DEBIT_TP_CAPTURE * max_profit:
        return ExitDecision(True, "DEBIT_TARGET", value, pnl)

    # Hard stop: lost >= 50% of the debit (after a short breathe)
    if elapsed_minutes >= DEBIT_MIN_HOLD and value <= entry_debit * (1.0 - DEBIT_STOP_FRAC):
        return ExitDecision(True, "DEBIT_STOP", value, pnl)

    # Rs-denominated trail: arm at Rs 600 profit; floor = max(peak - Rs 500, 0).
    # The %-of-max trail below requires ~190pt spot move to arm — effectively never fires
    # intraday. This absolute guard locks in real gains at any debit width.
    peak_pnl_rs = max(float(position.metadata.get('debit_peak_pnl_rupees') or 0.0), pnl)
    position.metadata['debit_peak_pnl_rupees'] = peak_pnl_rs
    if peak_pnl_rs >= DEBIT_RS_TRAIL_ARM:
        floor_pnl_rs = max(peak_pnl_rs - DEBIT_RS_TRAIL_GIVEBACK, 0.0)
        if pnl <= floor_pnl_rs:
            return ExitDecision(True, 'DEBIT_RS_TRAIL', value, pnl)

    # Profit trail: once armed at 40% of max profit, exit on a 35% giveback of peak
    peak = max(float(position.metadata.get("debit_peak_value") or entry_debit), value)
    position.metadata["debit_peak_value"] = peak
    peak_profit = peak - entry_debit
    if max_profit > 0 and peak_profit >= DEBIT_TRAIL_ARM * max_profit:
        floor_profit = peak_profit * (1.0 - DEBIT_TRAIL_GIVEBACK)
        if profit_captured <= floor_profit:
            return ExitDecision(True, "DEBIT_PROFIT_TRAIL", value, pnl)

    return ExitDecision(False, "HOLD", value, pnl)


# REVERSAL-AT-WALL exit (default OFF until validated): the mirror of the entry override. A held
# DIRECTIONAL debit riding INTO a wall that has flipped to a reversal-confluence AGAINST it should get
# out — don't ride a bearish put-debit down into a support that's screaming "bounce" (the 23,021 case),
# and don't ride a bullish call-debit up into a resistance that's screaming "rejection". Requires the SAME
# confluence as the entry override: the wall building hard AND CVD confirming the turn AND the confirmed
# W/M chart pattern AT the level. Reuses the entry thresholds so entry and exit read the wall identically.
_REVERSAL_EXIT = os.environ.get("SEL_REVERSAL_EXIT", "0") == "1"
_REVERSAL_EXIT_OI = float(os.environ.get("SEL_FADE_OVERRIDE_OI", "15000") or 15000.0)
_REVERSAL_EXIT_CVD = float(os.environ.get("SEL_FADE_OVERRIDE_CVD", "0") or 0.0)
_REVERSAL_EXIT_EDGE = float(os.environ.get("SEL_FADE_EDGE_PTS", "15") or 15.0)
_REVERSAL_EXIT_MIN_HOLD = float(os.environ.get("SEL_REVERSAL_EXIT_MIN_HOLD", "3") or 3.0)


def _evaluate_reversal_exit(
    position: OpenPosition,
    current_regime: RegimeState | None,
    now: datetime,
) -> ExitDecision | None:
    """Exit a directional debit when a reversal-confluence forms AGAINST it AT the wall. Returns an
    ExitDecision only when it should fire; None otherwise (so the caller falls through to normal exits).
    Fails closed on missing regime/CVD — the guard only acts on positive confirmation, never on absence."""
    if not _REVERSAL_EXIT or current_regime is None:
        return None
    m = current_regime.metadata or {}
    cvd = m.get("cvd_slope")
    if cvd is None:
        return None
    if _elapsed_minutes(now, position.entry_time) < _REVERSAL_EXIT_MIN_HOLD:
        return None  # let it breathe — don't knife out on the entry bar
    cvd = float(cvd)
    strat = position.structure.strategy
    from .strategy_matrix import _m_pattern_at_resistance, _w_pattern_at_support

    def _f(key: str) -> float:
        try:
            return float(m.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0

    if strat == StrategyType.PUT_DEBIT_SPREAD:  # bearish position — threat is a bounce UP off support
        support = _f("support_5m") or _f("support_15m") or _f("current_put_wall")
        spot = _f("spot_price") or _f("spot")
        at_wall = support > 0 and spot > 0 and abs(spot - support) <= _REVERSAL_EXIT_EDGE
        oi_build = _f("put_support_oi_change") >= _REVERSAL_EXIT_OI
        cvd_turn = cvd >= _REVERSAL_EXIT_CVD
        if at_wall and oi_build and cvd_turn and _w_pattern_at_support(m):
            return ExitDecision(True, "REVERSAL_AT_WALL_EXIT", 0.0, 0.0)
    elif strat == StrategyType.CALL_DEBIT_SPREAD:  # bullish position — threat is rejection DOWN off resistance
        resist = _f("resistance_5m") or _f("resistance_15m") or _f("current_call_wall")
        spot = _f("spot_price") or _f("spot")
        at_wall = resist > 0 and spot > 0 and abs(spot - resist) <= _REVERSAL_EXIT_EDGE
        oi_build = _f("call_resistance_oi_change") >= _REVERSAL_EXIT_OI
        cvd_turn = cvd <= -_REVERSAL_EXIT_CVD
        if at_wall and oi_build and cvd_turn and _m_pattern_at_resistance(m):
            return ExitDecision(True, "REVERSAL_AT_WALL_EXIT", 0.0, 0.0)
    return None


def evaluate_exit(
    position: OpenPosition,
    current_structure: TradeStructure | None = None,
    *,
    current_snapshot: MarketSnapshot | None = None,
    current_regime: RegimeState | None = None,
    now: datetime,
) -> ExitDecision:
    if position.structure.strategy in {StrategyType.CALL_DEBIT_SPREAD, StrategyType.PUT_DEBIT_SPREAD}:
        # REVERSAL-AT-WALL guard FIRST: if the wall has flipped against the position, get out before the
        # normal target/stop — this is the "should have exited at 23,021" discipline (data-driven exit).
        _rev = _evaluate_reversal_exit(position, current_regime, now)
        if _rev is not None:
            # price the exit with the real mark so P&L is correct, keeping the reversal reason.
            _mark = _evaluate_debit_exit(position, current_snapshot, current_structure, now)
            return ExitDecision(True, "REVERSAL_AT_WALL_EXIT", _mark.current_value_points, _mark.pnl_rupees)
        # RANGE-FADE positions are scalps, not fat-tail rides: exit on a SPOT target (the swing) or a
        # SPOT stop just beyond the faded level. This is the make-or-break discipline — a break of the
        # level cuts cheap. Falls through to the normal debit exit only if the fade exit says HOLD.
        if position.metadata.get("is_fade"):
            _fade = _evaluate_fade_exit(position, current_snapshot, current_structure, now)
            if _fade.should_exit:
                return _fade
        return _evaluate_debit_exit(position, current_snapshot, current_structure, now)
    # SEL_CHOP patience: must show Rs200 profit within 45 min or exit cheap.
    _patience = _evaluate_chop_patience_exit(position, current_snapshot, current_structure, now)
    if _patience is not None:
        return _patience
    current_value_points, current_legs = _current_position_mark(position, current_snapshot, current_structure)
    if now.time() >= TIME_EXIT:
        liability = current_value_points if current_value_points is not None else position.stop_value_points
        pnl_rupees = (position.entry_credit_points - liability) * position.lot_size * position.lots
        return ExitDecision(True, "TIME_EXIT", liability, pnl_rupees)

    elapsed_minutes = _elapsed_minutes(now, position.entry_time)

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
    # theta working in its favour. Trigger is 0.23 delta, just below the 0.25
    # CONDOR_DELTA_SL. Gated by the same 20-min min-hold as the theta-strategy
    # DELTA_STOP below — otherwise first-candle delta noise (0.20→0.23 in a minute)
    # closed a side before the position had any time to breathe.
    CONDOR_PARTIAL_CLOSE_DELTA = 0.23
    CONDOR_PARTIAL_CLOSE_MIN_HOLD = 20
    if (
        position.structure.strategy == StrategyType.IRON_CONDOR
        and current_legs
        and elapsed_minutes >= CONDOR_PARTIAL_CLOSE_MIN_HOLD
    ):
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

    short_delta_limit = DIRECTIONAL_DELTA_SL if position.structure.strategy not in {StrategyType.IRON_CONDOR, StrategyType.IRON_FLY, StrategyType.SHORT_STRANGLE, StrategyType.SHORT_STRADDLE} else CONDOR_DELTA_SL
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
            StrategyType.IRON_CONDOR, StrategyType.IRON_FLY, StrategyType.SHORT_STRANGLE, StrategyType.SHORT_STRADDLE
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


def _credit_short_tested(position, snapshot, min_delta: float) -> bool:
    """True if a credit spread's SHORT leg is genuinely tested (|delta| >= min_delta). While False, the
    short is safe and the seller should HOLD for theta rather than cut on a momentum blip. Missing
    delta -> treat as tested (fail safe: don't trap a position we can't measure)."""
    try:
        legs = position.structure.legs
    except AttributeError:
        return True
    saw_short = False
    for leg in legs:
        if getattr(leg, "action", "") != "SELL":
            continue
        saw_short = True
        try:
            q = snapshot.option_chain.find_quote(leg.strike, leg.option_type)
        except Exception:  # noqa: BLE001
            return True
        d = getattr(q, "delta", None) if q is not None else None
        if d is None:
            return True
        if abs(float(d)) >= min_delta:
            return True
    return False if saw_short else True


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
    # SELLER DISCIPLINE: a credit spread is SOLD to collect theta — it must be HELD while its short is
    # safe, and defended/exited only when the short is genuinely threatened. Firing a momentum
    # invalidation (EMA20/VWAP) at t+5min while the short is far OTM is a scalper's cut, not a seller's:
    # it captures ~zero decay (2026-09-04: bear-call cut at 5.1min for a scrap; 08-05/08-31 the same cut
    # cost -1,160/-2,428). Gate these exits on the short leg actually being tested (|delta| >= threshold);
    # while the short is safe, hold for theta. Defined-risk wings cap the interim.
    short_tested = _credit_short_tested(position, current_snapshot, _CREDIT_EXIT_MIN_SHORT_DELTA)
    if strategy == StrategyType.BULL_PUT_CREDIT_SPREAD:
        if closed_bar_check_allowed and short_tested and spot < vwap and last_n_closes_below(vwap, bars, n=2):
            return "VWAP_INVALIDATION"
    elif strategy == StrategyType.BEAR_CALL_CREDIT_SPREAD:
        ema20_5m = ema_value(closes(bars), period=20)
        if closed_bar_check_allowed and short_tested and (
            ema20_5m is not None
            and last_n_closes_above(ema20_5m, bars, n=2)
            and (vwap is None or last_n_closes_above(vwap, bars, n=2))
        ):
            return "EMA20_INVALIDATION"
        if closed_bar_check_allowed and short_tested and (
            bullish_reversal_structure(bars)
            and ema20_5m is not None
            and bars[-1].close > ema20_5m
            and (vwap is None or last_n_closes_above(vwap, bars, n=2))
        ):
            return "REVERSAL_STRUCTURE"
        if closed_bar_check_allowed and short_tested and spot > vwap and last_n_closes_above(vwap, bars, n=2):
            return "VWAP_INVALIDATION"
    elif strategy == StrategyType.PUT_DEBIT_SPREAD:
        # Bearish thesis: entered expecting the market to fall. Invalidate when market proves
        # bullish by reclaiming both VWAP and EMA20 for 2 consecutive bars — the same signal
        # BEAR_CALL credit uses, mirrored for the debit side. Gate: 15 min hold so early noise
        # (the position just entered) doesn't fire before the trade can breathe.
        ema20_5m = ema_value(closes(bars), period=20)
        if closed_bar_check_allowed and minutes_since_entry >= 15:
            vwap_reclaimed = spot > (vwap or 0) and last_n_closes_above(vwap, bars, n=2)
            ema_reclaimed = ema20_5m is not None and last_n_closes_above(ema20_5m, bars, n=2)
            if vwap_reclaimed and ema_reclaimed:
                return "BULLISH_RECLAIM_INVALIDATION"
    elif strategy == StrategyType.CALL_DEBIT_SPREAD:
        # Bullish thesis: entered expecting the market to rise. Invalidate when market proves
        # bearish by losing both VWAP and EMA20 for 2 consecutive bars.
        ema20_5m = ema_value(closes(bars), period=20)
        if closed_bar_check_allowed and minutes_since_entry >= 15:
            vwap_lost = spot < (vwap or float('inf')) and last_n_closes_below(vwap, bars, n=2)
            ema_lost = ema20_5m is not None and last_n_closes_below(ema20_5m, bars, n=2)
            if vwap_lost and ema_lost:
                return "BEARISH_BREAKDOWN_INVALIDATION"
    elif strategy in {StrategyType.IRON_CONDOR, StrategyType.IRON_FLY, StrategyType.SHORT_STRANGLE, StrategyType.SHORT_STRADDLE}:
        # Two reasons these flies died at 0.6-1.2 min on 2026-08-06 (mfe never left 0, then decayed
        # to +812/+770 by close — ~Rs 1,644 handed back, seen in exit_shadow_toclose):
        #   (a) NO min-hold: every other invalidation path waits one candle; this one fired instantly.
        #   (b) CLASSIFIER MISMATCH: the fly is *entered* on a low-efficiency BREAKOUT_DOWN that the
        #       SELECTOR reads as range-like ("the tape retraces what it gives, sell neutral premium",
        #       trend_efficiency < 0.50). But the EXIT here used a DIFFERENT test — classify_regime !=
        #       RANGE — which is TRUE the instant it enters, so the fly was dead on arrival. A min-hold
        #       alone only delays that to +5 min; it does not reconcile the two views.
        # Fix: gate on one candle AND on the SAME signal the entry used. Only invalidate when genuine
        # follow-through has RETURNED (trend_efficiency >= the entry threshold) — i.e. a real breakout
        # the neutral structure can't hold. While the low-efficiency chop that justified the fly
        # persists, keep collecting decay. Missing efficiency -> treat as high (allow exit) so a data
        # gap never traps a position. Defined-risk wings cap the interim either way.
        _eff = 1.0
        if current_regime is not None:
            try:
                _eff = float((current_regime.metadata or {}).get("trend_efficiency_ratio", 1.0))
            except (TypeError, ValueError):
                _eff = 1.0
        _breaking = bool(
            closed_bar_check_allowed and current_regime is not None
            and current_regime.regime != RegimeLabel.RANGE
            and _eff >= _THETA_INVALIDATION_MIN_EFF
        )
        # PERSISTENCE: even with the efficiency gate, RANGE_INVALIDATION still handed back winners on
        # 2026-08-10 (condor +182 vs +741 held; fly -221 at 5.9min vs +1,206 held) because a HEAD-FAKE
        # breakout crosses trend_efficiency 0.50 for a single cycle, trips the exit, then reverts. A
        # real breakout PERSISTS across consecutive reads; a fake does not. So require the break to
        # hold for _THETA_INVALIDATION_MIN_STREAK consecutive cycles before evicting. The streak lives
        # on position.metadata (round-trips through save/load); it resets the moment the tape calms, so
        # a genuine sustained trend still exits within ~a minute while a one-candle fake no longer can.
        _streak = int(position.metadata.get("_range_inval_streak") or 0)
        _streak = _streak + 1 if _breaking else 0
        position.metadata["_range_inval_streak"] = _streak
        if _streak >= _THETA_INVALIDATION_MIN_STREAK:
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
    # Arm at 40% capture (was 55%): a credit spread up 40% that reverses used to
    # give it all back into a loss. Once armed, exit if capture falls below 55%
    # of the peak — "don't let a green trade go red".
    if best_pct < 0.40:
        return None
    if current_capture_pct < best_pct * 0.55:
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
    elapsed_minutes = _elapsed_minutes(now, position.entry_time)
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
