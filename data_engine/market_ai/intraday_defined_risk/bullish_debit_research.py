from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, time
from math import sqrt
from pathlib import Path
from statistics import mean
import json

from .backtest import DEFAULT_ENTRY_TIMES, _session_previous_closes, load_backtest_dataset, run_backtest
from .data_models import (
    AccountRiskLimits,
    AccountState,
    AdaptiveParameters,
    MarketSnapshot,
    OhlcvBar,
    OhlcvSeries,
    OptionType,
    OptionsChainSnapshot,
    OptionsContractQuote,
    StrategyLeg,
    StrategyType,
    TradeStructure,
)
from .features import bearish_reversal_structure, closes, compute_vwap, ema_value, last_n_closes_below, recent_higher_lows, session_bars
from .regime import classify_regime
from .risk import assess_trade_risk


LATE_SESSION_BULLISH_SUBTYPE = "LATE_SESSION_BULLISH_RECLAIM"
CALL_DEBIT_WIDTHS = (50.0, 75.0, 100.0)
LONG_CALL_DELTA_BAND = (0.45, 0.55)
SHORT_CALL_DELTA_BAND = (0.20, 0.30)
MIN_REWARD_RISK = 1.20
MAX_DEBIT_TO_RANGE_RATIO = 0.35
MIN_TIME_REMAINING_HOURS = 1.0
PROFIT_TAKE_SHARE_OF_MAX_PROFIT = 0.55
TIME_EXIT = time(15, 15)

EXPANSION_PLAYBOOKS = (
    "OPEN_DRIVE_BULLISH",
    "GAP_UP_BULLISH_CONTINUATION",
    "HIGH_CONFLUENCE_BULLISH_CONTINUATION",
)
EXPANSION_CALL_DEBIT_WIDTHS = (50.0, 75.0, 100.0)
EXPANSION_LONG_CALL_DELTA_BAND = (0.50, 0.60)
EXPANSION_SHORT_CALL_DELTA_BAND = (0.25, 0.35)
EXPANSION_MIN_REWARD_RISK = 1.20
EXPANSION_MAX_DEBIT_TO_RANGE_RATIO = 0.40
EXPANSION_MAX_INTRADAY_MOVE_PCT = 0.012
EXPANSION_MAX_VWAP_DISTANCE_PCT = 0.0075
EXPANSION_MIN_TIME_REMAINING_HOURS = 2.0
EXPANSION_PROFIT_TAKE_SHARE_OF_MAX_PROFIT = 0.70
EXPANSION_MOMENTUM_LOSS_BARS = 3
EXPANSION_RESISTANCE_BUFFER_POINTS = 35.0


@dataclass(frozen=True)
class DebitCandidate:
    timestamp: datetime
    long_strike: float
    short_strike: float
    long_delta_abs: float
    short_delta_abs: float
    width_points: float
    debit_points: float
    max_profit_points: float
    reward_risk: float
    break_even_distance_points: float
    expected_move_points: float
    intraday_range_points: float
    hours_remaining: float
    max_spread_ratio: float
    stop_pain: float
    score: float
    valid: bool
    canonical_rejection_reason: str


@dataclass
class OpenDebitTrade:
    entry_timestamp: datetime
    entry_date: date
    long_strike: float
    short_strike: float
    entry_debit_points: float
    width_points: float
    max_profit_points: float
    target_value_points: float
    lots: int
    lot_size: int
    playbook: str
    subtype: str
    long_delta_abs: float
    short_delta_abs: float
    score: float
    highest_spot_seen: float = 0.0
    bars_without_new_high: int = 0


def _hours_remaining(timestamp: datetime) -> float:
    exit_ts = timestamp.replace(hour=15, minute=15, second=0, microsecond=0)
    return max((exit_ts - timestamp).total_seconds() / 3600.0, 0.0)


def _session_open(snapshot: MarketSnapshot) -> float:
    bars = session_bars(snapshot.nifty_5m)
    return bars[0].open


def _average_chain_iv(snapshot: MarketSnapshot) -> float | None:
    ivs = [float(quote.iv) for quote in snapshot.option_chain.quotes if quote.iv is not None and quote.ltp > 0]
    if not ivs:
        return None
    return sum(ivs) / len(ivs)


def _session_range_points(snapshot: MarketSnapshot) -> float:
    bars = session_bars(snapshot.nifty_5m)
    return max(bar.high for bar in bars) - min(bar.low for bar in bars)


def _expected_intraday_move_points(snapshot: MarketSnapshot, avg_iv: float | None) -> float:
    session_range = max(_session_range_points(snapshot), 1.0)
    if avg_iv is None or avg_iv <= 0:
        return max(session_range * 0.80, 30.0)
    daily_expected_move = snapshot.option_chain.spot * (avg_iv / 100.0) / sqrt(252.0)
    iv_move = daily_expected_move * sqrt(max(_hours_remaining(snapshot.timestamp), 0.25) / 6.25)
    return max(iv_move, session_range * 0.80, 30.0)


def _liquid_call_quotes(snapshot: MarketSnapshot, params: AdaptiveParameters) -> list[OptionsContractQuote]:
    quotes: list[OptionsContractQuote] = []
    for quote in snapshot.option_chain.quotes:
        if quote.option_type != OptionType.CALL:
            continue
        if quote.mid_price is None or quote.ltp <= 0:
            continue
        if quote.spread_ratio > params.liquidity_spread_ratio_cap:
            continue
        if quote.delta is None:
            continue
        quotes.append(quote)
    return sorted(quotes, key=lambda quote: quote.strike)


def _rejection_reason(
    *,
    passed_liquidity: bool,
    passed_debit: bool,
    passed_reward_risk: bool,
    passed_width_vs_move: bool,
    passed_time: bool,
    passed_extension: bool,
    passed_delta: bool,
) -> str:
    if not passed_debit:
        return "DEBIT_TOO_HIGH"
    if not passed_reward_risk:
        return "RR_TOO_LOW"
    if not passed_width_vs_move or not passed_time:
        return "TIME_INSUFFICIENT"
    if not passed_extension:
        return "STRUCTURE_ALREADY_EXTENDED"
    if not passed_delta:
        return "DELTA_MISMATCH"
    if not passed_liquidity:
        return "LIQUIDITY_BAD"
    return "NONE"


def _expansion_rejection_reason(
    *,
    passed_liquidity: bool,
    passed_delta: bool,
    passed_debit: bool,
    passed_reward_risk: bool,
    passed_width_vs_move: bool,
    passed_time: bool,
    passed_extension: bool,
    passed_vwap_distance: bool,
    passed_resistance: bool,
) -> str:
    if not passed_debit:
        return "DEBIT_TOO_HIGH"
    if not passed_delta:
        return "DELTA_MISMATCH"
    if not passed_reward_risk:
        return "RR_TOO_LOW"
    if not passed_width_vs_move:
        return "EXPECTED_MOVE_TOO_SMALL"
    if not passed_time:
        return "TIME_INSUFFICIENT"
    if not passed_extension:
        return "MOVE_ALREADY_EXTENDED"
    if not passed_vwap_distance:
        return "PRICE_TOO_FAR_FROM_VWAP"
    if not passed_resistance:
        return "RESISTANCE_TOO_CLOSE"
    if not passed_liquidity:
        return "LIQUIDITY_BAD"
    return "NONE"


def _score_candidate(
    *,
    reward_risk: float,
    break_even_distance_points: float,
    expected_move_points: float,
    long_delta_abs: float,
    short_delta_abs: float,
    intraday_range_points: float,
    debit_points: float,
    liquidity_quality: float,
) -> float:
    move_room = max(0.0, 1.0 - (break_even_distance_points / max(expected_move_points, 1.0)))
    delta_quality = max(0.0, 1.0 - abs(long_delta_abs - 0.50) / 0.05) + max(0.0, 1.0 - abs(short_delta_abs - 0.25) / 0.05)
    debit_efficiency = max(0.0, 1.0 - (debit_points / max(intraday_range_points * MAX_DEBIT_TO_RANGE_RATIO, 1.0)))
    score = (
        3.0 * reward_risk
        + 2.0 * move_room
        + 1.5 * delta_quality
        + 1.5 * debit_efficiency
        + 1.0 * liquidity_quality
    )
    return round(score, 4)


def _expansion_score_candidate(
    *,
    reward_risk: float,
    break_even_distance_points: float,
    resistance_distance_points: float | None,
    expected_move_points: float,
    long_delta_abs: float,
    short_delta_abs: float,
    intraday_range_points: float,
    debit_points: float,
    liquidity_quality: float,
) -> float:
    move_room = max(0.0, 1.0 - (break_even_distance_points / max(expected_move_points, 1.0)))
    resistance_room = (
        max(0.0, min((resistance_distance_points or expected_move_points) / max(expected_move_points, 1.0), 1.0))
        if resistance_distance_points is not None
        else 1.0
    )
    delta_quality = max(0.0, 1.0 - abs(long_delta_abs - 0.55) / 0.05) + max(0.0, 1.0 - abs(short_delta_abs - 0.30) / 0.05)
    debit_efficiency = max(0.0, 1.0 - (debit_points / max(intraday_range_points * EXPANSION_MAX_DEBIT_TO_RANGE_RATIO, 1.0)))
    score = (
        3.0 * reward_risk
        + 2.0 * move_room
        + 2.0 * resistance_room
        + 1.5 * delta_quality
        + 1.0 * debit_efficiency
        + 1.0 * liquidity_quality
    )
    return round(score, 4)


def _late_session_bullish_ready(snapshot: MarketSnapshot, params: AdaptiveParameters) -> tuple[object, bool, list[str]]:
    regime_state = classify_regime(snapshot, params)
    metadata = regime_state.metadata
    reasons: list[str] = []
    if metadata.get("bullish_shadow_subtype") != LATE_SESSION_BULLISH_SUBTYPE:
        reasons.append("Subtype is not LATE_SESSION_BULLISH_RECLAIM.")
        return regime_state, False, reasons
    bars_5m = session_bars(snapshot.nifty_5m)
    vwap = regime_state.vwap or compute_vwap(bars_5m)
    if vwap is None or snapshot.option_chain.spot <= vwap:
        reasons.append("Spot is not above VWAP.")
    if regime_state.ema20_15m is None or snapshot.option_chain.spot <= regime_state.ema20_15m:
        reasons.append("Spot is not above 15m EMA20.")
    if regime_state.ema20_slope_15m is None or regime_state.ema20_slope_15m <= 0:
        reasons.append("15m EMA20 slope is not positive.")
    if not recent_higher_lows(bars_5m, needed=2):
        reasons.append("Higher-low structure is not confirmed.")
    if str(metadata.get("structure_signal") or "UNKNOWN") in {"BEARISH_BOS", "BEARISH_CHOCH"}:
        reasons.append("Bearish structure conflict is active.")
    return regime_state, not reasons, reasons


def _expansion_bullish_ready(snapshot: MarketSnapshot, params: AdaptiveParameters) -> tuple[object, bool, list[str]]:
    regime_state = classify_regime(snapshot, params)
    metadata = regime_state.metadata
    playbook = str(metadata.get("playbook") or "UNKNOWN")
    reasons: list[str] = []
    if playbook not in EXPANSION_PLAYBOOKS:
        reasons.append("Playbook is not a bullish expansion regime.")
        return regime_state, False, reasons
    bars_5m = session_bars(snapshot.nifty_5m)
    vwap = regime_state.vwap or compute_vwap(bars_5m)
    if vwap is None or snapshot.option_chain.spot <= vwap:
        reasons.append("Spot is not above VWAP.")
    if regime_state.ema20_15m is None or snapshot.option_chain.spot <= regime_state.ema20_15m:
        reasons.append("Spot is not above 15m EMA20.")
    if regime_state.ema20_slope_15m is None or regime_state.ema20_slope_15m <= 0:
        reasons.append("15m EMA20 slope is not positive.")
    if not bool(metadata.get("bullish_entry_ready")):
        reasons.append("Bullish entry is not ready.")
    if str(metadata.get("structure_signal") or "UNKNOWN") not in {"BULLISH_BOS", "BULLISH_CHOCH"}:
        reasons.append("Bullish structure expansion is not confirmed.")
    if str(metadata.get("smart_money_bias") or "UNKNOWN") == "BEARISH":
        reasons.append("Bearish structure conflict is active.")
    resistance_candidates = [
        float(value)
        for value in (
            metadata.get("resistance_5m"),
            metadata.get("resistance_15m"),
            metadata.get("call_resistance_strike"),
        )
        if value is not None and float(value) > snapshot.option_chain.spot
    ]
    if resistance_candidates and min(level - snapshot.option_chain.spot for level in resistance_candidates) <= EXPANSION_RESISTANCE_BUFFER_POINTS:
        reasons.append("Immediate resistance overhead is too close.")
    return regime_state, not reasons, reasons


def _select_call_debit_candidates(
    snapshot: MarketSnapshot,
    regime_state: object,
    params: AdaptiveParameters,
) -> tuple[DebitCandidate | None, list[DebitCandidate]]:
    call_quotes = _liquid_call_quotes(snapshot, params)
    quote_by_strike = {quote.strike: quote for quote in call_quotes}
    spot = snapshot.option_chain.spot
    vwap = getattr(regime_state, "vwap")
    ema20_15m = getattr(regime_state, "ema20_15m")
    expected_move_points = _expected_intraday_move_points(snapshot, _average_chain_iv(snapshot))
    intraday_range_points = max(_session_range_points(snapshot), 1.0)
    hours_remaining = _hours_remaining(snapshot.timestamp)
    anchor = max(value for value in (spot, vwap or spot, ema20_15m or spot))

    candidates: list[DebitCandidate] = []
    for long_quote in call_quotes:
        long_delta_abs = abs(long_quote.delta or 0.0)
        if not (LONG_CALL_DELTA_BAND[0] <= long_delta_abs <= LONG_CALL_DELTA_BAND[1]):
            continue
        for width in CALL_DEBIT_WIDTHS:
            short_quote = quote_by_strike.get(long_quote.strike + width)
            if short_quote is None:
                continue
            short_delta_abs = abs(short_quote.delta or 0.0)
            long_mid = long_quote.mid_price
            short_mid = short_quote.mid_price
            if long_mid is None or short_mid is None:
                continue
            debit_points = long_mid - short_mid
            max_profit_points = width - debit_points
            reward_risk = max_profit_points / max(debit_points, 0.01)
            break_even = long_quote.strike + debit_points
            break_even_distance_points = max(break_even - spot, 0.0)
            liquidity_quality = max(
                0.0,
                1.0 - (max(long_quote.spread_ratio, short_quote.spread_ratio) / max(params.liquidity_spread_ratio_cap, 0.01)),
            )
            extension_distance = max(spot - max(vwap or spot, ema20_15m or spot), 0.0)
            passed_liquidity = max(long_quote.spread_ratio, short_quote.spread_ratio) <= params.liquidity_spread_ratio_cap
            passed_delta = SHORT_CALL_DELTA_BAND[0] <= short_delta_abs <= SHORT_CALL_DELTA_BAND[1]
            passed_debit = debit_points > 0 and debit_points <= (intraday_range_points * MAX_DEBIT_TO_RANGE_RATIO)
            passed_reward_risk = reward_risk >= MIN_REWARD_RISK
            passed_width_vs_move = width <= expected_move_points
            passed_time = hours_remaining >= MIN_TIME_REMAINING_HOURS
            passed_extension = not (
                extension_distance >= (intraday_range_points * 0.45)
                and break_even_distance_points >= (expected_move_points * 0.35)
            )
            stop_pain = break_even_distance_points / max(expected_move_points, 1.0)
            score = _score_candidate(
                reward_risk=reward_risk,
                break_even_distance_points=break_even_distance_points,
                expected_move_points=expected_move_points,
                long_delta_abs=long_delta_abs,
                short_delta_abs=short_delta_abs,
                intraday_range_points=intraday_range_points,
                debit_points=debit_points,
                liquidity_quality=liquidity_quality,
            )
            valid = all(
                (
                    passed_liquidity,
                    passed_delta,
                    passed_debit,
                    passed_reward_risk,
                    passed_width_vs_move,
                    passed_time,
                    passed_extension,
                    max_profit_points > 0,
                )
            )
            candidates.append(
                DebitCandidate(
                    timestamp=snapshot.timestamp,
                    long_strike=long_quote.strike,
                    short_strike=short_quote.strike,
                    long_delta_abs=round(long_delta_abs, 4),
                    short_delta_abs=round(short_delta_abs, 4),
                    width_points=width,
                    debit_points=round(debit_points, 4),
                    max_profit_points=round(max_profit_points, 4),
                    reward_risk=round(reward_risk, 4),
                    break_even_distance_points=round(break_even_distance_points, 4),
                    expected_move_points=round(expected_move_points, 4),
                    intraday_range_points=round(intraday_range_points, 4),
                    hours_remaining=round(hours_remaining, 4),
                    max_spread_ratio=round(max(long_quote.spread_ratio, short_quote.spread_ratio), 4),
                    stop_pain=round(stop_pain, 4),
                    score=score,
                    valid=valid,
                    canonical_rejection_reason=_rejection_reason(
                        passed_liquidity=passed_liquidity,
                        passed_debit=passed_debit,
                        passed_reward_risk=passed_reward_risk,
                        passed_width_vs_move=passed_width_vs_move,
                        passed_time=passed_time,
                        passed_extension=passed_extension,
                        passed_delta=passed_delta,
                    ) if not valid else "NONE",
                )
            )
    valid_candidates = [candidate for candidate in candidates if candidate.valid]
    best_candidate = max(valid_candidates, key=lambda candidate: candidate.score) if valid_candidates else None
    return best_candidate, candidates


def _call_debit_value_points(snapshot: MarketSnapshot, trade: OpenDebitTrade) -> float | None:
    long_quote = snapshot.option_chain.find_quote(trade.long_strike, OptionType.CALL)
    short_quote = snapshot.option_chain.find_quote(trade.short_strike, OptionType.CALL)
    if long_quote is None or short_quote is None or long_quote.mid_price is None or short_quote.mid_price is None:
        return None
    return max((long_quote.mid_price - short_quote.mid_price) - snapshot.slippage_points, 0.0)


def _expansion_call_debit_exit_reason(snapshot: MarketSnapshot, regime_state: object, trade: OpenDebitTrade) -> tuple[str | None, float | None]:
    current_value = _call_debit_value_points(snapshot, trade)
    if current_value is None:
        return None, None
    bars = session_bars(snapshot.nifty_5m)
    vwap = getattr(regime_state, "vwap") or compute_vwap(bars)
    ema20_5m = ema_value(closes(bars), period=20)
    structure_signal = str(getattr(regime_state, "metadata").get("structure_signal") or "UNKNOWN")
    latest_high = bars[-1].high
    if latest_high > trade.highest_spot_seen + 0.5:
        trade.highest_spot_seen = latest_high
        trade.bars_without_new_high = 0
    else:
        trade.bars_without_new_high += 1
    if snapshot.timestamp.time() >= TIME_EXIT:
        return "TIME_EXIT", current_value
    if vwap is not None and snapshot.option_chain.spot < vwap and last_n_closes_below(vwap, bars, n=2):
        return "VWAP_INVALIDATION", current_value
    if ema20_5m is not None and last_n_closes_below(ema20_5m, bars, n=2):
        return "EMA20_INVALIDATION", current_value
    if bearish_reversal_structure(bars) and structure_signal in {"BEARISH_BOS", "BEARISH_CHOCH"}:
        return "STRUCTURE_REVERSAL", current_value
    if trade.bars_without_new_high >= EXPANSION_MOMENTUM_LOSS_BARS:
        return "MOMENTUM_LOSS", current_value
    if current_value >= trade.target_value_points:
        return "TAKE_PROFIT", current_value
    return None, current_value


def _select_expansion_call_debit_candidates(
    snapshot: MarketSnapshot,
    regime_state: object,
    params: AdaptiveParameters,
) -> tuple[DebitCandidate | None, list[DebitCandidate]]:
    call_quotes = _liquid_call_quotes(snapshot, params)
    quote_by_strike = {quote.strike: quote for quote in call_quotes}
    spot = snapshot.option_chain.spot
    vwap = getattr(regime_state, "vwap")
    avg_iv = _average_chain_iv(snapshot)
    expected_move_points = _expected_intraday_move_points(snapshot, avg_iv)
    intraday_range_points = max(_session_range_points(snapshot), 1.0)
    hours_remaining = _hours_remaining(snapshot.timestamp)
    previous_close = snapshot.previous_session_close or _session_open(snapshot)
    intraday_move_pct = max((spot - previous_close) / max(previous_close, 1.0), 0.0)
    resistance_candidates = [
        float(value)
        for value in (
            getattr(regime_state, "metadata").get("resistance_5m"),
            getattr(regime_state, "metadata").get("resistance_15m"),
            getattr(regime_state, "metadata").get("call_resistance_strike"),
        )
        if value is not None and float(value) > spot
    ]
    nearest_resistance_distance = min((level - spot for level in resistance_candidates), default=None)
    vwap_distance_pct = max((spot - vwap) / max(spot, 1.0), 0.0) if vwap is not None else float("inf")

    candidates: list[DebitCandidate] = []
    for long_quote in call_quotes:
        long_delta_abs = abs(long_quote.delta or 0.0)
        if not (EXPANSION_LONG_CALL_DELTA_BAND[0] <= long_delta_abs <= EXPANSION_LONG_CALL_DELTA_BAND[1]):
            continue
        for width in EXPANSION_CALL_DEBIT_WIDTHS:
            short_quote = quote_by_strike.get(long_quote.strike + width)
            if short_quote is None:
                continue
            short_delta_abs = abs(short_quote.delta or 0.0)
            long_mid = long_quote.mid_price
            short_mid = short_quote.mid_price
            if long_mid is None or short_mid is None:
                continue
            debit_points = long_mid - short_mid
            max_profit_points = width - debit_points
            reward_risk = max_profit_points / max(debit_points, 0.01)
            break_even = long_quote.strike + debit_points
            break_even_distance_points = max(break_even - spot, 0.0)
            liquidity_quality = max(
                0.0,
                1.0 - (max(long_quote.spread_ratio, short_quote.spread_ratio) / max(params.liquidity_spread_ratio_cap, 0.01)),
            )
            passed_liquidity = max(long_quote.spread_ratio, short_quote.spread_ratio) <= params.liquidity_spread_ratio_cap
            passed_delta = EXPANSION_SHORT_CALL_DELTA_BAND[0] <= short_delta_abs <= EXPANSION_SHORT_CALL_DELTA_BAND[1]
            passed_debit = debit_points > 0 and debit_points <= (intraday_range_points * EXPANSION_MAX_DEBIT_TO_RANGE_RATIO)
            passed_reward_risk = reward_risk >= EXPANSION_MIN_REWARD_RISK
            passed_width_vs_move = width <= expected_move_points
            passed_time = hours_remaining >= EXPANSION_MIN_TIME_REMAINING_HOURS
            passed_extension = intraday_move_pct <= EXPANSION_MAX_INTRADAY_MOVE_PCT
            passed_vwap_distance = vwap_distance_pct <= EXPANSION_MAX_VWAP_DISTANCE_PCT
            passed_resistance = nearest_resistance_distance is None or nearest_resistance_distance > break_even_distance_points + EXPANSION_RESISTANCE_BUFFER_POINTS
            stop_pain = break_even_distance_points / max(expected_move_points, 1.0)
            score = _expansion_score_candidate(
                reward_risk=reward_risk,
                break_even_distance_points=break_even_distance_points,
                resistance_distance_points=nearest_resistance_distance,
                expected_move_points=expected_move_points,
                long_delta_abs=long_delta_abs,
                short_delta_abs=short_delta_abs,
                intraday_range_points=intraday_range_points,
                debit_points=debit_points,
                liquidity_quality=liquidity_quality,
            )
            valid = all(
                (
                    passed_liquidity,
                    passed_delta,
                    passed_debit,
                    passed_reward_risk,
                    passed_width_vs_move,
                    passed_time,
                    passed_extension,
                    passed_vwap_distance,
                    passed_resistance,
                    max_profit_points > 0,
                )
            )
            candidates.append(
                DebitCandidate(
                    timestamp=snapshot.timestamp,
                    long_strike=long_quote.strike,
                    short_strike=short_quote.strike,
                    long_delta_abs=round(long_delta_abs, 4),
                    short_delta_abs=round(short_delta_abs, 4),
                    width_points=width,
                    debit_points=round(debit_points, 4),
                    max_profit_points=round(max_profit_points, 4),
                    reward_risk=round(reward_risk, 4),
                    break_even_distance_points=round(break_even_distance_points, 4),
                    expected_move_points=round(expected_move_points, 4),
                    intraday_range_points=round(intraday_range_points, 4),
                    hours_remaining=round(hours_remaining, 4),
                    max_spread_ratio=round(max(long_quote.spread_ratio, short_quote.spread_ratio), 4),
                    stop_pain=round(stop_pain, 4),
                    score=score,
                    valid=valid,
                    canonical_rejection_reason=_expansion_rejection_reason(
                        passed_liquidity=passed_liquidity,
                        passed_delta=passed_delta,
                        passed_debit=passed_debit,
                        passed_reward_risk=passed_reward_risk,
                        passed_width_vs_move=passed_width_vs_move,
                        passed_time=passed_time,
                        passed_extension=passed_extension,
                        passed_vwap_distance=passed_vwap_distance,
                        passed_resistance=passed_resistance,
                    ) if not valid else "NONE",
                )
            )
    valid_candidates = [candidate for candidate in candidates if candidate.valid]
    best_candidate = max(valid_candidates, key=lambda candidate: candidate.score) if valid_candidates else None
    return best_candidate, candidates


def _call_debit_exit_reason(snapshot: MarketSnapshot, regime_state: object, trade: OpenDebitTrade) -> tuple[str | None, float | None]:
    current_value = _call_debit_value_points(snapshot, trade)
    if current_value is None:
        return None, None
    bars = session_bars(snapshot.nifty_5m)
    vwap = getattr(regime_state, "vwap") or compute_vwap(bars)
    ema20_5m = ema_value(closes(bars), period=20)
    structure_signal = str(getattr(regime_state, "metadata").get("structure_signal") or "UNKNOWN")
    if snapshot.timestamp.time() >= TIME_EXIT:
        return "TIME_EXIT", current_value
    if vwap is not None and snapshot.option_chain.spot < vwap and last_n_closes_below(vwap, bars, n=2):
        return "VWAP_INVALIDATION", current_value
    if ema20_5m is not None and last_n_closes_below(ema20_5m, bars, n=2):
        return "EMA20_INVALIDATION", current_value
    if bearish_reversal_structure(bars) and structure_signal in {"BEARISH_BOS", "BEARISH_CHOCH"}:
        return "STRUCTURE_REVERSAL", current_value
    if current_value >= trade.target_value_points:
        return "TAKE_PROFIT", current_value
    return None, current_value


def _build_snapshot(
    ts: datetime,
    *,
    session_bars_5m: list[OhlcvBar],
    session_bars_15m: list[OhlcvBar],
    chain_snapshot: OptionsChainSnapshot,
    risk_limits: AccountRiskLimits,
    account_state: AccountState,
    previous_chain_snapshot: OptionsChainSnapshot | None,
    previous_session_close: float | None,
) -> MarketSnapshot:
    return MarketSnapshot(
        nifty_5m=OhlcvSeries(timeframe_minutes=5, bars=session_bars_5m),
        nifty_15m=OhlcvSeries(timeframe_minutes=15, bars=session_bars_15m),
        option_chain=chain_snapshot,
        risk_limits=risk_limits,
        account_state=account_state,
        live_vwap=None,
        lot_size=65,
        slippage_points=0.5,
        previous_option_chain=previous_chain_snapshot,
        previous_session_close=previous_session_close,
    )


def _extract_credit_model_report(shadow_result: dict[str, object]) -> dict[str, object]:
    decisions = shadow_result["decisions"]
    trades = shadow_result["trades"]
    subtype_decisions = [
        decision
        for decision in decisions
        if str(((decision.get("metadata") or {}).get("trade_funnel") or {}).get("bullish_shadow_subtype") or (decision.get("metadata") or {}).get("bullish_shadow_subtype") or "") == LATE_SESSION_BULLISH_SUBTYPE
    ]
    subtype_trades = [
        trade
        for trade in trades
        if str(trade.get("bullish_shadow_subtype") or "") == LATE_SESSION_BULLISH_SUBTYPE
        and str(trade.get("strategy") or "") == StrategyType.BULL_PUT_CREDIT_SPREAD.value
    ]
    rejection_reasons = Counter(
        str((((decision.get("metadata") or {}).get("trade_funnel") or {}).get("canonical_rejection_reason")) or "UNKNOWN")
        for decision in subtype_decisions
        if str((((decision.get("metadata") or {}).get("trade_funnel") or {}).get("final_result")) or "REJECTED") != "EXECUTED"
    )
    pnls = [float(trade.get("pnl_rupees") or 0.0) for trade in subtype_trades]
    return {
        "strategy": StrategyType.BULL_PUT_CREDIT_SPREAD.value,
        "setups_detected": sum(
            1
            for decision in subtype_decisions
            if bool((((decision.get("metadata") or {}).get("trade_funnel") or {}).get("setup_detected")))
        ),
        "setups_passing_quality": sum(
            1
            for decision in subtype_decisions
            if bool((((decision.get("metadata") or {}).get("trade_funnel") or {}).get("passed_setup_quality")))
        ),
        "executed_trades": len(subtype_trades),
        "wins": sum(1 for pnl in pnls if pnl > 0),
        "losses": sum(1 for pnl in pnls if pnl < 0),
        "realized_pnl_rupees": round(sum(pnls), 2),
        "average_pnl_rupees": round(mean(pnls), 2) if pnls else 0.0,
        "max_loss_rupees": round(min(pnls), 2) if pnls else 0.0,
        "expectancy_rupees": round(mean(pnls), 2) if pnls else 0.0,
        "win_rate": round(sum(1 for pnl in pnls if pnl > 0) / len(pnls), 4) if pnls else 0.0,
        "rejection_reasons": rejection_reasons.most_common(10),
        "trades": subtype_trades,
        "decision_timestamps": [
            str(((decision.get("metadata") or {}).get("trade_funnel") or {}).get("timestamp") or decision.get("timestamp"))
            for decision in subtype_decisions
            if bool((((decision.get("metadata") or {}).get("trade_funnel") or {}).get("setup_detected")))
            and bool((((decision.get("metadata") or {}).get("trade_funnel") or {}).get("passed_setup_quality")))
        ],
    }


def _recommendation(credit_report: dict[str, object], debit_report: dict[str, object]) -> str:
    debit_pnl = float(debit_report["realized_pnl_rupees"])
    credit_pnl = float(credit_report["realized_pnl_rupees"])
    debit_wins = int(debit_report["wins"])
    debit_losses = int(debit_report["losses"])
    if debit_report["executed_trades"] and debit_pnl > credit_pnl and debit_losses == 0:
        return "promote_to_tier_b_shadow_only"
    if debit_report["executed_trades"] and debit_pnl > 0:
        return "keep_shadow"
    if debit_report["setups_detected"] and debit_pnl <= 0:
        return "reject"
    return "keep_shadow"


def run_late_session_bullish_debit_shadow_study(
    *,
    data_path: str | Path,
    start: str,
    end: str,
    round_trip_cost_rupees_per_lot: float,
    parameters: AdaptiveParameters | None = None,
) -> dict[str, object]:
    params = (parameters or AdaptiveParameters()).clamped()
    credit_shadow = run_backtest(
        data_path,
        start=start,
        end=end,
        learning_db_path="/tmp/intraday_defined_risk_credit_shadow.sqlite3",
        parameters=params,
        reset_learning_db=True,
        round_trip_cost_rupees_per_lot=round_trip_cost_rupees_per_lot,
        allowed_playbook_tiers=("A", "B"),
        include_opportunity_benchmark=False,
    )
    credit_report = _extract_credit_model_report(credit_shadow)
    candidate_timestamps = {datetime.fromisoformat(ts) for ts in credit_report["decision_timestamps"]}

    dataset = load_backtest_dataset(data_path)
    bars_5m_by_day = dataset["bars_5m_by_day"]
    bars_15m_by_day = dataset["bars_15m_by_day"]
    chain_map = dataset["chain_map"]
    previous_session_closes = _session_previous_closes(dataset["bars_5m"])
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    timestamps = sorted(ts for ts in chain_map if start_dt <= ts <= end_dt)

    risk_limits = AccountRiskLimits(
        max_risk_rupees_per_trade=20000.0,
        max_margin_rupees=200000.0,
        max_daily_loss_rupees=10000.0,
    )
    account_state = AccountState(realised_pnl_rupees=0.0, margin_used_rupees=0.0)
    current_session_date: date | None = None
    current_day_bars_5m: list[OhlcvBar] = []
    current_day_bars_15m: list[OhlcvBar] = []
    session_bars_5m: list[OhlcvBar] = []
    session_bars_15m: list[OhlcvBar] = []
    session_5m_index = 0
    session_15m_index = 0
    previous_chain_snapshot = None
    last_snapshot: MarketSnapshot | None = None
    open_trade: OpenDebitTrade | None = None
    session_trade_taken = False

    diagnostics: list[dict[str, object]] = []
    executed_trades: list[dict[str, object]] = []
    candidate_stats: list[dict[str, object]] = []

    def close_trade(snapshot: MarketSnapshot, reason: str) -> None:
        nonlocal account_state, open_trade
        if open_trade is None:
            return
        current_value = _call_debit_value_points(snapshot, open_trade)
        if current_value is None:
            current_value = 0.0
        pnl_rupees = ((current_value - open_trade.entry_debit_points) * open_trade.lot_size * open_trade.lots) - (
            round_trip_cost_rupees_per_lot * open_trade.lots
        )
        executed_trades.append(
            {
                "entry_timestamp": open_trade.entry_timestamp.isoformat(),
                "timestamp": snapshot.timestamp.isoformat(),
                "strategy": StrategyType.CALL_DEBIT_SPREAD.value,
                "bullish_shadow_subtype": open_trade.subtype,
                "playbook": open_trade.playbook,
                "long_strike": open_trade.long_strike,
                "short_strike": open_trade.short_strike,
                "entry_debit_points": round(open_trade.entry_debit_points, 4),
                "exit_value_points": round(current_value, 4),
                "width_points": open_trade.width_points,
                "exit_reason": reason,
                "pnl_rupees": round(pnl_rupees, 2),
                "lots": open_trade.lots,
                "score": open_trade.score,
            }
        )
        account_state = AccountState(realised_pnl_rupees=account_state.realised_pnl_rupees + pnl_rupees, margin_used_rupees=0.0)
        open_trade = None

    for ts in timestamps:
        if current_session_date is None:
            current_session_date = ts.date()
        elif ts.date() != current_session_date:
            if open_trade is not None and last_snapshot is not None:
                close_trade(last_snapshot, "TIME_EXIT")
            current_session_date = ts.date()
            previous_chain_snapshot = None
            current_day_bars_5m = []
            current_day_bars_15m = []
            session_bars_5m = []
            session_bars_15m = []
            session_5m_index = 0
            session_15m_index = 0
            account_state = AccountState(realised_pnl_rupees=0.0, margin_used_rupees=0.0)
            session_trade_taken = False

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
        snapshot = _build_snapshot(
            ts,
            session_bars_5m=session_bars_5m,
            session_bars_15m=session_bars_15m,
            chain_snapshot=chain_snapshot,
            risk_limits=risk_limits,
            account_state=account_state,
            previous_chain_snapshot=previous_chain_snapshot,
            previous_session_close=previous_session_closes.get(ts.date()),
        )
        regime_state, eligible, eligibility_reasons = _late_session_bullish_ready(snapshot, params)

        if open_trade is not None:
            reason, current_value = _call_debit_exit_reason(snapshot, regime_state, open_trade)
            if reason is not None and current_value is not None:
                close_trade(snapshot, reason)
            last_snapshot = snapshot
            previous_chain_snapshot = chain_snapshot
            continue

        if ts.strftime("%H:%M") not in DEFAULT_ENTRY_TIMES or session_trade_taken:
            last_snapshot = snapshot
            previous_chain_snapshot = chain_snapshot
            continue

        if ts not in candidate_timestamps:
            last_snapshot = snapshot
            previous_chain_snapshot = chain_snapshot
            continue

        if not eligible:
            last_snapshot = snapshot
            previous_chain_snapshot = chain_snapshot
            continue

        best_candidate, candidate_pool = _select_call_debit_candidates(snapshot, regime_state, params)
        candidate_stats.extend(asdict(candidate) for candidate in candidate_pool)
        if best_candidate is None:
            best_failed = max(candidate_pool, key=lambda candidate: candidate.score) if candidate_pool else None
            diagnostics.append(
                {
                    "timestamp": ts.isoformat(),
                    "reason": best_failed.canonical_rejection_reason if best_failed else "NO_VALID_CANDIDATE",
                    "best_candidate": asdict(best_failed) if best_failed else None,
                }
            )
            last_snapshot = snapshot
            previous_chain_snapshot = chain_snapshot
            continue

        structure = TradeStructure(
            strategy=StrategyType.CALL_DEBIT_SPREAD,
            legs=[
                StrategyLeg(
                    action="BUY",
                    option_type=OptionType.CALL,
                    strike=best_candidate.long_strike,
                    quote=snapshot.option_chain.find_quote(best_candidate.long_strike, OptionType.CALL),  # type: ignore[arg-type]
                ),
                StrategyLeg(
                    action="SELL",
                    option_type=OptionType.CALL,
                    strike=best_candidate.short_strike,
                    quote=snapshot.option_chain.find_quote(best_candidate.short_strike, OptionType.CALL),  # type: ignore[arg-type]
                ),
            ],
            credit_points=0.0,
            width_points=best_candidate.width_points,
            call_width_points=best_candidate.width_points,
            metadata={
                "debit_points": best_candidate.debit_points,
                "reward_risk": best_candidate.reward_risk,
                "break_even_distance_points": best_candidate.break_even_distance_points,
            },
        )
        risk = assess_trade_risk(
            structure=structure,
            lot_size=snapshot.lot_size,
            risk_limits=risk_limits,
            account_state=account_state,
            margin_estimate_per_lot=best_candidate.debit_points * snapshot.lot_size,
        )
        if not risk.allowed or risk.lots < 1:
            diagnostics.append(
                {
                    "timestamp": ts.isoformat(),
                    "reason": "RISK_LIMIT",
                    "best_candidate": asdict(best_candidate),
                }
            )
            last_snapshot = snapshot
            previous_chain_snapshot = chain_snapshot
            continue

        open_trade = OpenDebitTrade(
            entry_timestamp=ts,
            entry_date=ts.date(),
            long_strike=best_candidate.long_strike,
            short_strike=best_candidate.short_strike,
            entry_debit_points=round(best_candidate.debit_points + snapshot.slippage_points, 4),
            width_points=best_candidate.width_points,
            max_profit_points=round(best_candidate.max_profit_points, 4),
            target_value_points=round(
                best_candidate.debit_points + (best_candidate.max_profit_points * PROFIT_TAKE_SHARE_OF_MAX_PROFIT),
                4,
            ),
            lots=risk.lots,
            lot_size=snapshot.lot_size,
            playbook=str(getattr(regime_state, "metadata").get("playbook") or "UNKNOWN"),
            subtype=LATE_SESSION_BULLISH_SUBTYPE,
            long_delta_abs=best_candidate.long_delta_abs,
            short_delta_abs=best_candidate.short_delta_abs,
            score=best_candidate.score,
        )
        session_trade_taken = True
        account_state = AccountState(
            realised_pnl_rupees=account_state.realised_pnl_rupees,
            margin_used_rupees=best_candidate.debit_points * snapshot.lot_size * risk.lots,
        )
        last_snapshot = snapshot
        previous_chain_snapshot = chain_snapshot

    if open_trade is not None and last_snapshot is not None:
        close_trade(last_snapshot, "TIME_EXIT")

    pnls = [float(trade["pnl_rupees"]) for trade in executed_trades]
    winning_trades = [trade for trade in executed_trades if float(trade["pnl_rupees"]) > 0]
    valid_candidates = [candidate for candidate in candidate_stats if candidate["valid"]]
    best_width = None
    if valid_candidates:
        by_width: dict[float, list[float]] = defaultdict(list)
        for candidate in valid_candidates:
            by_width[float(candidate["width_points"])].append(float(candidate["score"]))
        best_width = max(by_width.items(), key=lambda item: mean(item[1]))[0]
    optimal_entry_timing = None
    if executed_trades:
        by_time: dict[str, list[float]] = defaultdict(list)
        for trade in executed_trades:
            by_time[trade["entry_timestamp"][11:16]].append(float(trade["pnl_rupees"]))
        optimal_entry_timing = max(by_time.items(), key=lambda item: mean(item[1]))[0]

    debit_report = {
        "strategy": StrategyType.CALL_DEBIT_SPREAD.value,
        "setups_detected": len(candidate_timestamps),
        "setups_passing_quality": len(candidate_timestamps),
        "executed_trades": len(executed_trades),
        "wins": sum(1 for pnl in pnls if pnl > 0),
        "losses": sum(1 for pnl in pnls if pnl < 0),
        "realized_pnl_rupees": round(sum(pnls), 2),
        "average_pnl_rupees": round(mean(pnls), 2) if pnls else 0.0,
        "max_loss_rupees": round(min(pnls), 2) if pnls else 0.0,
        "expectancy_rupees": round(mean(pnls), 2) if pnls else 0.0,
        "win_rate": round(sum(1 for pnl in pnls if pnl > 0) / len(pnls), 4) if pnls else 0.0,
        "rejection_reasons": Counter(item["reason"] for item in diagnostics).most_common(10),
        "best_configuration": {
            "best_width_points": best_width,
            "average_long_delta_abs": round(mean(float(trade["long_delta_abs"]) for trade in winning_trades), 4) if winning_trades else round(mean(float(candidate["long_delta_abs"]) for candidate in valid_candidates), 4) if valid_candidates else None,
            "average_short_delta_abs": round(mean(float(trade["short_delta_abs"]) for trade in winning_trades), 4) if winning_trades else round(mean(float(candidate["short_delta_abs"]) for candidate in valid_candidates), 4) if valid_candidates else None,
            "optimal_entry_timing": optimal_entry_timing,
        },
        "rejected_setups": diagnostics,
        "trades": executed_trades,
    }

    report = {
        "activation_scope": LATE_SESSION_BULLISH_SUBTYPE,
        "credit_model": credit_report,
        "debit_model": debit_report,
        "conversion_report": {
            "setups_detected": len(candidate_timestamps),
            "setups_passing_quality": len(candidate_timestamps),
            "credit_model_trades": credit_report["executed_trades"],
            "debit_model_trades": debit_report["executed_trades"],
        },
        "performance_comparison": {
            "credit_model": {
                "win_rate": credit_report["win_rate"],
                "average_pnl_rupees": credit_report["average_pnl_rupees"],
                "max_loss_rupees": credit_report["max_loss_rupees"],
                "expectancy_rupees": credit_report["expectancy_rupees"],
                "realized_pnl_rupees": credit_report["realized_pnl_rupees"],
            },
            "debit_model": {
                "win_rate": debit_report["win_rate"],
                "average_pnl_rupees": debit_report["average_pnl_rupees"],
                "max_loss_rupees": debit_report["max_loss_rupees"],
                "expectancy_rupees": debit_report["expectancy_rupees"],
                "realized_pnl_rupees": debit_report["realized_pnl_rupees"],
            },
        },
        "structural_diagnosis": {
            "credit_model_rejections": credit_report["rejection_reasons"],
            "debit_model_rejections": debit_report["rejection_reasons"],
        },
        "best_configuration": debit_report["best_configuration"],
        "recommendation": _recommendation(credit_report, debit_report),
    }
    return report


def write_late_session_bullish_debit_shadow_study(
    output_path: str | Path,
    *,
    data_path: str | Path,
    start: str,
    end: str,
    round_trip_cost_rupees_per_lot: float,
    parameters: AdaptiveParameters | None = None,
) -> dict[str, object]:
    report = run_late_session_bullish_debit_shadow_study(
        data_path=data_path,
        start=start,
        end=end,
        round_trip_cost_rupees_per_lot=round_trip_cost_rupees_per_lot,
        parameters=parameters,
    )
    Path(output_path).write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    return report


def _extract_expansion_setup_universe(shadow_result: dict[str, object]) -> dict[str, object]:
    decisions = shadow_result["decisions"]
    filtered_decisions = []
    for decision in decisions:
        metadata = decision.get("metadata") or {}
        funnel = metadata.get("trade_funnel") or {}
        playbook = str(funnel.get("playbook") or metadata.get("playbook") or "UNKNOWN")
        if playbook not in EXPANSION_PLAYBOOKS:
            continue
        if not bool(funnel.get("setup_detected")):
            continue
        filtered_decisions.append(decision)

    timestamps = [
        str(((decision.get("metadata") or {}).get("trade_funnel") or {}).get("timestamp") or decision.get("timestamp"))
        for decision in filtered_decisions
        if bool((((decision.get("metadata") or {}).get("trade_funnel") or {}).get("passed_setup_quality")))
    ]
    return {
        "decisions": filtered_decisions,
        "timestamps": timestamps,
        "setups_detected": len(filtered_decisions),
        "setups_passing_quality": sum(
            1
            for decision in filtered_decisions
            if bool((((decision.get("metadata") or {}).get("trade_funnel") or {}).get("passed_setup_quality")))
        ),
    }


def _playbook_rollup(
    *,
    decisions: list[dict[str, object]],
    trades: list[dict[str, object]],
    rejections: list[dict[str, object]],
    all_candidates: list[dict[str, object]],
) -> dict[str, object]:
    setups_by_playbook: Counter[str] = Counter()
    passed_quality_by_playbook: Counter[str] = Counter()
    executed_by_playbook: Counter[str] = Counter()
    wins_by_playbook: Counter[str] = Counter()
    losses_by_playbook: Counter[str] = Counter()
    pnl_by_playbook: defaultdict[str, list[float]] = defaultdict(list)
    rejections_by_playbook: defaultdict[str, Counter[str]] = defaultdict(Counter)
    candidates_by_playbook: defaultdict[str, list[dict[str, object]]] = defaultdict(list)

    for decision in decisions:
        funnel = ((decision.get("metadata") or {}).get("trade_funnel") or {})
        playbook = str(funnel.get("playbook") or (decision.get("metadata") or {}).get("playbook") or "UNKNOWN")
        setups_by_playbook[playbook] += 1
        if funnel.get("passed_setup_quality"):
            passed_quality_by_playbook[playbook] += 1

    for trade in trades:
        playbook = str(trade.get("playbook") or "UNKNOWN")
        pnl = float(trade.get("pnl_rupees") or 0.0)
        executed_by_playbook[playbook] += 1
        pnl_by_playbook[playbook].append(pnl)
        if pnl > 0:
            wins_by_playbook[playbook] += 1
        elif pnl < 0:
            losses_by_playbook[playbook] += 1

    for rejection in rejections:
        playbook = str(rejection.get("playbook") or "UNKNOWN")
        rejections_by_playbook[playbook][str(rejection.get("reason") or "UNKNOWN")] += 1

    for candidate in all_candidates:
        candidates_by_playbook[str(candidate.get("playbook") or "UNKNOWN")].append(candidate)

    report: dict[str, object] = {}
    for playbook in EXPANSION_PLAYBOOKS:
        pnls = pnl_by_playbook.get(playbook, [])
        candidate_pool = candidates_by_playbook.get(playbook, [])
        best_candidate = max(candidate_pool, key=lambda item: float(item.get("score") or 0.0)) if candidate_pool else None
        report[playbook] = {
            "setups_detected": setups_by_playbook.get(playbook, 0),
            "setups_passing_quality": passed_quality_by_playbook.get(playbook, 0),
            "executed_trades": executed_by_playbook.get(playbook, 0),
            "wins": wins_by_playbook.get(playbook, 0),
            "losses": losses_by_playbook.get(playbook, 0),
            "realized_pnl_rupees": round(sum(pnls), 2),
            "average_pnl_rupees": round(mean(pnls), 2) if pnls else 0.0,
            "expectancy_rupees": round(mean(pnls), 2) if pnls else 0.0,
            "max_loss_rupees": round(min(pnls), 2) if pnls else 0.0,
            "win_rate": round(sum(1 for pnl in pnls if pnl > 0) / len(pnls), 4) if pnls else 0.0,
            "rejection_reasons": rejections_by_playbook.get(playbook, Counter()).most_common(10),
            "best_candidate_attempt": best_candidate,
        }
    return report


def _expansion_recommendation(report: dict[str, object]) -> str:
    playbook_rollup = report["by_playbook"]
    promotable = []
    for playbook, payload in playbook_rollup.items():
        if int(payload["executed_trades"]) < 3:
            continue
        if float(payload["realized_pnl_rupees"]) <= 0:
            continue
        if int(payload["losses"]) > 0:
            continue
        promotable.append(playbook)
    if promotable:
        return f"promote_to_tier_a_trial:{','.join(promotable)}"
    if sum(int(payload["executed_trades"]) for payload in playbook_rollup.values()) > 0:
        return "keep_shadow"
    return "reject"


def run_bullish_expansion_debit_shadow_study(
    *,
    data_path: str | Path,
    start: str,
    end: str,
    round_trip_cost_rupees_per_lot: float,
    parameters: AdaptiveParameters | None = None,
) -> dict[str, object]:
    params = (parameters or AdaptiveParameters()).clamped()
    shadow_result = run_backtest(
        data_path,
        start=start,
        end=end,
        learning_db_path="/tmp/intraday_defined_risk_bullish_expansion_shadow.sqlite3",
        parameters=params,
        reset_learning_db=True,
        round_trip_cost_rupees_per_lot=round_trip_cost_rupees_per_lot,
        allowed_playbook_tiers=("A", "B"),
        include_opportunity_benchmark=False,
    )
    setup_universe = _extract_expansion_setup_universe(shadow_result)
    candidate_timestamps = {datetime.fromisoformat(ts) for ts in setup_universe["timestamps"]}

    dataset = load_backtest_dataset(data_path)
    bars_5m_by_day = dataset["bars_5m_by_day"]
    bars_15m_by_day = dataset["bars_15m_by_day"]
    chain_map = dataset["chain_map"]
    previous_session_closes = _session_previous_closes(dataset["bars_5m"])
    start_dt = datetime.fromisoformat(start)
    end_dt = datetime.fromisoformat(end)
    timestamps = sorted(ts for ts in chain_map if start_dt <= ts <= end_dt)

    risk_limits = AccountRiskLimits(
        max_risk_rupees_per_trade=20000.0,
        max_margin_rupees=200000.0,
        max_daily_loss_rupees=10000.0,
    )
    account_state = AccountState(realised_pnl_rupees=0.0, margin_used_rupees=0.0)
    current_session_date: date | None = None
    current_day_bars_5m: list[OhlcvBar] = []
    current_day_bars_15m: list[OhlcvBar] = []
    session_bars_5m: list[OhlcvBar] = []
    session_bars_15m: list[OhlcvBar] = []
    session_5m_index = 0
    session_15m_index = 0
    previous_chain_snapshot = None
    last_snapshot: MarketSnapshot | None = None
    open_trade: OpenDebitTrade | None = None
    session_trade_taken = False

    diagnostics: list[dict[str, object]] = []
    executed_trades: list[dict[str, object]] = []
    candidate_stats: list[dict[str, object]] = []

    def close_trade(snapshot: MarketSnapshot, reason: str) -> None:
        nonlocal account_state, open_trade
        if open_trade is None:
            return
        current_value = _call_debit_value_points(snapshot, open_trade)
        if current_value is None:
            current_value = 0.0
        pnl_rupees = ((current_value - open_trade.entry_debit_points) * open_trade.lot_size * open_trade.lots) - (
            round_trip_cost_rupees_per_lot * open_trade.lots
        )
        executed_trades.append(
            {
                "entry_timestamp": open_trade.entry_timestamp.isoformat(),
                "timestamp": snapshot.timestamp.isoformat(),
                "strategy": StrategyType.CALL_DEBIT_SPREAD.value,
                "playbook": open_trade.playbook,
                "entry_debit_points": round(open_trade.entry_debit_points, 4),
                "exit_value_points": round(current_value, 4),
                "width_points": open_trade.width_points,
                "long_strike": open_trade.long_strike,
                "short_strike": open_trade.short_strike,
                "exit_reason": reason,
                "pnl_rupees": round(pnl_rupees, 2),
                "lots": open_trade.lots,
                "long_delta_abs": open_trade.long_delta_abs,
                "short_delta_abs": open_trade.short_delta_abs,
                "score": open_trade.score,
            }
        )
        account_state = AccountState(realised_pnl_rupees=account_state.realised_pnl_rupees + pnl_rupees, margin_used_rupees=0.0)
        open_trade = None

    for ts in timestamps:
        if current_session_date is None:
            current_session_date = ts.date()
        elif ts.date() != current_session_date:
            if open_trade is not None and last_snapshot is not None:
                close_trade(last_snapshot, "TIME_EXIT")
            current_session_date = ts.date()
            previous_chain_snapshot = None
            current_day_bars_5m = []
            current_day_bars_15m = []
            session_bars_5m = []
            session_bars_15m = []
            session_5m_index = 0
            session_15m_index = 0
            account_state = AccountState(realised_pnl_rupees=0.0, margin_used_rupees=0.0)
            session_trade_taken = False

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
        snapshot = _build_snapshot(
            ts,
            session_bars_5m=session_bars_5m,
            session_bars_15m=session_bars_15m,
            chain_snapshot=chain_snapshot,
            risk_limits=risk_limits,
            account_state=account_state,
            previous_chain_snapshot=previous_chain_snapshot,
            previous_session_close=previous_session_closes.get(ts.date()),
        )
        regime_state, eligible, eligibility_reasons = _expansion_bullish_ready(snapshot, params)

        if open_trade is not None:
            reason, current_value = _expansion_call_debit_exit_reason(snapshot, regime_state, open_trade)
            if reason is not None and current_value is not None:
                close_trade(snapshot, reason)
            last_snapshot = snapshot
            previous_chain_snapshot = chain_snapshot
            continue

        if ts.strftime("%H:%M") not in DEFAULT_ENTRY_TIMES or session_trade_taken:
            last_snapshot = snapshot
            previous_chain_snapshot = chain_snapshot
            continue
        if ts not in candidate_timestamps:
            last_snapshot = snapshot
            previous_chain_snapshot = chain_snapshot
            continue
        if not eligible:
            diagnostics.append(
                {
                    "timestamp": ts.isoformat(),
                    "playbook": str(regime_state.metadata.get("playbook") or "UNKNOWN"),
                    "reason": eligibility_reasons[0] if eligibility_reasons else "ELIGIBILITY_FAILED",
                    "best_candidate": None,
                }
            )
            last_snapshot = snapshot
            previous_chain_snapshot = chain_snapshot
            continue

        best_candidate, candidate_pool = _select_expansion_call_debit_candidates(snapshot, regime_state, params)
        playbook = str(regime_state.metadata.get("playbook") or "UNKNOWN")
        for candidate in candidate_pool:
            record = asdict(candidate)
            record["playbook"] = playbook
            candidate_stats.append(record)
        if best_candidate is None:
            best_failed = max(candidate_pool, key=lambda candidate: candidate.score) if candidate_pool else None
            diagnostics.append(
                {
                    "timestamp": ts.isoformat(),
                    "playbook": playbook,
                    "reason": best_failed.canonical_rejection_reason if best_failed else "NO_VALID_CANDIDATE",
                    "best_candidate": asdict(best_failed) if best_failed else None,
                }
            )
            last_snapshot = snapshot
            previous_chain_snapshot = chain_snapshot
            continue

        structure = TradeStructure(
            strategy=StrategyType.CALL_DEBIT_SPREAD,
            legs=[
                StrategyLeg(
                    action="BUY",
                    option_type=OptionType.CALL,
                    strike=best_candidate.long_strike,
                    quote=snapshot.option_chain.find_quote(best_candidate.long_strike, OptionType.CALL),  # type: ignore[arg-type]
                ),
                StrategyLeg(
                    action="SELL",
                    option_type=OptionType.CALL,
                    strike=best_candidate.short_strike,
                    quote=snapshot.option_chain.find_quote(best_candidate.short_strike, OptionType.CALL),  # type: ignore[arg-type]
                ),
            ],
            credit_points=0.0,
            width_points=best_candidate.width_points,
            call_width_points=best_candidate.width_points,
            metadata={
                "debit_points": best_candidate.debit_points,
                "reward_risk": best_candidate.reward_risk,
                "break_even_distance_points": best_candidate.break_even_distance_points,
            },
        )
        risk = assess_trade_risk(
            structure=structure,
            lot_size=snapshot.lot_size,
            risk_limits=risk_limits,
            account_state=account_state,
            margin_estimate_per_lot=best_candidate.debit_points * snapshot.lot_size,
        )
        if not risk.allowed or risk.lots < 1:
            diagnostics.append(
                {
                    "timestamp": ts.isoformat(),
                    "playbook": playbook,
                    "reason": "RISK_LIMIT",
                    "best_candidate": asdict(best_candidate),
                }
            )
            last_snapshot = snapshot
            previous_chain_snapshot = chain_snapshot
            continue

        open_trade = OpenDebitTrade(
            entry_timestamp=ts,
            entry_date=ts.date(),
            long_strike=best_candidate.long_strike,
            short_strike=best_candidate.short_strike,
            entry_debit_points=round(best_candidate.debit_points + snapshot.slippage_points, 4),
            width_points=best_candidate.width_points,
            max_profit_points=round(best_candidate.max_profit_points, 4),
            target_value_points=round(
                best_candidate.debit_points + (best_candidate.max_profit_points * EXPANSION_PROFIT_TAKE_SHARE_OF_MAX_PROFIT),
                4,
            ),
            lots=risk.lots,
            lot_size=snapshot.lot_size,
            playbook=playbook,
            subtype=playbook,
            long_delta_abs=best_candidate.long_delta_abs,
            short_delta_abs=best_candidate.short_delta_abs,
            score=best_candidate.score,
            highest_spot_seen=snapshot.option_chain.spot,
        )
        session_trade_taken = True
        account_state = AccountState(
            realised_pnl_rupees=account_state.realised_pnl_rupees,
            margin_used_rupees=best_candidate.debit_points * snapshot.lot_size * risk.lots,
        )
        last_snapshot = snapshot
        previous_chain_snapshot = chain_snapshot

    if open_trade is not None and last_snapshot is not None:
        close_trade(last_snapshot, "TIME_EXIT")

    pnls = [float(trade["pnl_rupees"]) for trade in executed_trades]
    valid_candidates = [candidate for candidate in candidate_stats if candidate["valid"]]
    scoring_pool = valid_candidates or candidate_stats
    width_scores: defaultdict[float, list[float]] = defaultdict(list)
    delta_pairs: defaultdict[str, list[float]] = defaultdict(list)
    for candidate in scoring_pool:
        width_scores[float(candidate["width_points"])].append(float(candidate["score"]))
        delta_label = f"L{float(candidate['long_delta_abs']):.2f}-S{float(candidate['short_delta_abs']):.2f}"
        delta_pairs[delta_label].append(float(candidate["score"]))
    best_width = max(width_scores.items(), key=lambda item: mean(item[1]))[0] if width_scores else None
    best_delta_pair = max(delta_pairs.items(), key=lambda item: mean(item[1]))[0] if delta_pairs else None
    optimal_entry_timing = None
    if executed_trades:
        by_time: dict[str, list[float]] = defaultdict(list)
        for trade in executed_trades:
            by_time[trade["entry_timestamp"][11:16]].append(float(trade["pnl_rupees"]))
        optimal_entry_timing = max(by_time.items(), key=lambda item: mean(item[1]))[0]

    report = {
        "activation_scope": list(EXPANSION_PLAYBOOKS),
        "no_trade_baseline": {
            "executed_trades": 0,
            "realized_pnl_rupees": 0.0,
            "win_rate": 0.0,
            "average_pnl_rupees": 0.0,
            "expectancy_rupees": 0.0,
            "max_loss_rupees": 0.0,
        },
        "conversion_report": {
            "setups_detected": setup_universe["setups_detected"],
            "setups_passing_quality": setup_universe["setups_passing_quality"],
            "debit_trades_executed": len(executed_trades),
        },
        "performance": {
            "win_rate": round(sum(1 for pnl in pnls if pnl > 0) / len(pnls), 4) if pnls else 0.0,
            "average_pnl_rupees": round(mean(pnls), 2) if pnls else 0.0,
            "expectancy_rupees": round(mean(pnls), 2) if pnls else 0.0,
            "max_loss_rupees": round(min(pnls), 2) if pnls else 0.0,
            "realized_pnl_rupees": round(sum(pnls), 2),
            "wins": sum(1 for pnl in pnls if pnl > 0),
            "losses": sum(1 for pnl in pnls if pnl < 0),
        },
        "structural_diagnostics": {
            "rejection_reasons": Counter(item["reason"] for item in diagnostics).most_common(10),
            "best_performing_width_points": best_width,
            "best_performing_delta_pair": best_delta_pair,
            "optimal_entry_timing": optimal_entry_timing,
        },
        "by_playbook": _playbook_rollup(
            decisions=setup_universe["decisions"],
            trades=executed_trades,
            rejections=diagnostics,
            all_candidates=candidate_stats,
        ),
        "rejected_setups": diagnostics,
        "trades": executed_trades,
    }
    report["recommendation"] = _expansion_recommendation(report)
    return report


def write_bullish_expansion_debit_shadow_study(
    output_path: str | Path,
    *,
    data_path: str | Path,
    start: str,
    end: str,
    round_trip_cost_rupees_per_lot: float,
    parameters: AdaptiveParameters | None = None,
) -> dict[str, object]:
    report = run_bullish_expansion_debit_shadow_study(
        data_path=data_path,
        start=start,
        end=end,
        round_trip_cost_rupees_per_lot=round_trip_cost_rupees_per_lot,
        parameters=parameters,
    )
    Path(output_path).write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    return report
