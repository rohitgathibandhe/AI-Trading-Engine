from __future__ import annotations

from math import inf

from .data_models import (
    AdaptiveParameters,
    MarketSnapshot,
    OptionType,
    OptionsContractQuote,
    RegimeState,
    StrategyLeg,
    StrategyType,
    TradeStructure,
)


DIRECTIONAL_WIDTH_CHOICES = (50.0, 100.0, 150.0)


def select_structure(
    strategy: StrategyType,
    snapshot: MarketSnapshot,
    regime_state: RegimeState,
    params: AdaptiveParameters | None = None,
) -> tuple[TradeStructure | None, list[str]]:
    params = (params or AdaptiveParameters()).clamped()
    if strategy == StrategyType.BEAR_CALL_CREDIT_SPREAD:
        return _select_vertical(
            strategy=strategy,
            snapshot=snapshot,
            regime_state=regime_state,
            option_type=OptionType.CALL,
            short_delta_band=(0.18, 0.35),
            long_delta_band=(0.05, 0.20),
            min_credit_ratio=max(0.22, min(params.directional_credit_width_ratio, 0.28)),
            allow_distance_fallback_when_deltas_present=True,
            min_short_strike=regime_state.metadata.get("min_short_call_strike"),
        )
    if strategy == StrategyType.BULL_PUT_CREDIT_SPREAD:
        playbook = regime_state.metadata.get("playbook")
        bullish_setup = regime_state.metadata.get("bullish_setup")
        bullish_confluence_score = float(regime_state.metadata.get("bullish_confluence_score") or 0.0)
        bullish_support_quality = float(regime_state.metadata.get("bullish_support_quality_score") or 0.0)
        smart_money_bias = regime_state.metadata.get("smart_money_bias")
        short_delta_band = (0.18, 0.25)
        long_delta_band = (0.05, 0.10)
        min_credit_ratio = params.directional_credit_width_ratio
        if (
            playbook == "SIDEWAYS_TO_BULLISH_RECLAIM"
            and bullish_setup == "VWAP_HOLD_HIGHER_LOW"
            and bullish_confluence_score >= 11.0
            and bullish_support_quality >= 4.0
            and smart_money_bias == "BULLISH"
        ):
            min_credit_ratio = min(min_credit_ratio, 0.30)
        elif (
            playbook == "SIDEWAYS_TO_BULLISH_RECLAIM"
            and bullish_setup == "SHALLOW_CONTINUATION"
            and bullish_confluence_score >= 11.0
            and bullish_support_quality >= 4.5
            and smart_money_bias == "BULLISH"
        ):
            min_credit_ratio = min(min_credit_ratio, 0.32)
        if (
            playbook == "SIDEWAYS_TO_BULLISH_RECLAIM"
            and regime_state.metadata.get("bullish_setup") == "VWAP_HOLD_HIGHER_LOW"
            and bool(regime_state.metadata.get("early_sideways_bullish_ready"))
            and regime_state.metadata.get("smart_money_bias") == "BULLISH"
        ):
            short_delta_band = (0.10, 0.25)
            long_delta_band = (0.0, 0.10)
        return _select_vertical(
            strategy=strategy,
            snapshot=snapshot,
            regime_state=regime_state,
            option_type=OptionType.PUT,
            short_delta_band=short_delta_band,
            long_delta_band=long_delta_band,
            min_credit_ratio=min_credit_ratio,
            max_short_strike=regime_state.metadata.get("max_short_put_strike"),
        )
    if strategy == StrategyType.IRON_CONDOR:
        return _select_condor(snapshot=snapshot, regime_state=regime_state, params=params)
    return None, ["No allowed structure for the requested strategy."]


def _liquid_otm_quotes(
    quotes: list[OptionsContractQuote],
    option_type: OptionType,
    spot: float,
) -> list[OptionsContractQuote]:
    filtered: list[OptionsContractQuote] = []
    for quote in quotes:
        if quote.option_type != option_type:
            continue
        if quote.ltp <= 0 or quote.mid_price is None or quote.spread_ratio > 0.15:
            continue
        if option_type == OptionType.CALL and quote.strike <= spot:
            continue
        if option_type == OptionType.PUT and quote.strike >= spot:
            continue
        filtered.append(quote)
    return sorted(filtered, key=lambda quote: quote.strike)


def _select_vertical(
    strategy: StrategyType,
    snapshot: MarketSnapshot,
    regime_state: RegimeState,
    option_type: OptionType,
    short_delta_band: tuple[float, float],
    long_delta_band: tuple[float, float],
    min_credit_ratio: float,
    *,
    allow_distance_fallback_when_deltas_present: bool = True,
    min_short_strike: float | None = None,
    max_short_strike: float | None = None,
    candidate_quotes: list[OptionsContractQuote] | None = None,
) -> tuple[TradeStructure | None, list[str]]:
    quotes = candidate_quotes or _liquid_otm_quotes(snapshot.option_chain.quotes, option_type=option_type, spot=snapshot.option_chain.spot)
    if len(quotes) < 2:
        return None, ["No liquid OTM quotes available for vertical selection."]

    preferred_width = regime_state.metadata.get("preferred_width_points")
    allowed_widths = regime_state.metadata.get("allowed_width_points")
    target_short_put_buffer = regime_state.metadata.get("target_short_put_buffer_points")
    rv_points = snapshot.option_chain.spot * regime_state.rv30_pct / 100.0
    has_deltas = any(quote.delta is not None for quote in quotes)
    best_delta_pair: tuple[OptionsContractQuote, OptionsContractQuote] | None = None
    best_delta_score = -inf
    best_distance_pair: tuple[OptionsContractQuote, OptionsContractQuote] | None = None
    best_distance_score = -inf
    reasons: list[str] = []

    for short_quote in quotes:
        if min_short_strike is not None and short_quote.strike < min_short_strike:
            continue
        if max_short_strike is not None and short_quote.strike > max_short_strike:
            continue
        for long_quote in quotes:
            if option_type == OptionType.CALL and long_quote.strike <= short_quote.strike:
                continue
            if option_type == OptionType.PUT and long_quote.strike >= short_quote.strike:
                continue

            width = abs(long_quote.strike - short_quote.strike)
            if width <= 0:
                continue
            if allowed_widths is not None and width not in {float(value) for value in allowed_widths}:
                continue

            short_mid = short_quote.mid_price
            long_mid = long_quote.mid_price
            if short_mid is None or long_mid is None:
                continue
            credit = short_mid - long_mid
            if credit <= 0:
                continue
            if credit < min_credit_ratio * width:
                continue

            if has_deltas and short_quote.delta is not None and long_quote.delta is not None:
                short_abs = abs(short_quote.delta)
                long_abs = abs(long_quote.delta)
                if short_delta_band[0] <= short_abs <= short_delta_band[1] and long_delta_band[0] <= long_abs <= long_delta_band[1]:
                    short_target = sum(short_delta_band) / 2.0
                    long_target = sum(long_delta_band) / 2.0
                    delta_distance = abs(short_abs - short_target) + abs(long_abs - long_target)
                    score = (credit / width) - delta_distance - short_quote.spread_ratio - long_quote.spread_ratio
                    if option_type == OptionType.PUT and max_short_strike is not None and target_short_put_buffer is not None:
                        buffer_distance = max_short_strike - short_quote.strike
                        score -= abs(buffer_distance - target_short_put_buffer) / 100.0
                    if preferred_width is not None:
                        score -= abs(width - float(preferred_width)) / 100.0
                    if score > best_delta_score:
                        best_delta_score = score
                        best_delta_pair = (short_quote, long_quote)

            if option_type == OptionType.CALL and (short_quote.strike - snapshot.option_chain.spot) < rv_points:
                continue
            if option_type == OptionType.PUT and (snapshot.option_chain.spot - short_quote.strike) < rv_points:
                continue
            if width not in DIRECTIONAL_WIDTH_CHOICES:
                continue
            distance_score = (credit / width) - short_quote.spread_ratio - long_quote.spread_ratio
            if option_type == OptionType.PUT and max_short_strike is not None and target_short_put_buffer is not None:
                buffer_distance = max_short_strike - short_quote.strike
                distance_score -= abs(buffer_distance - target_short_put_buffer) / 100.0
            if preferred_width is not None:
                distance_score -= abs(width - float(preferred_width)) / 100.0
            if distance_score > best_distance_score:
                best_distance_score = distance_score
                best_distance_pair = (short_quote, long_quote)

    best_pair = best_delta_pair
    if best_pair is None and (not has_deltas or allow_distance_fallback_when_deltas_present):
        best_pair = best_distance_pair
    if not best_pair:
        if has_deltas and not allow_distance_fallback_when_deltas_present:
            reasons.append("Deltas are available, but no vertical spread matched the stricter delta bands; refusing RV-distance fallback.")
        else:
            reasons.append("No vertical spread satisfied liquidity, delta/distance, and credit-efficiency rules.")
        return None, reasons

    short_quote, long_quote = best_pair
    credit = (short_quote.mid_price or 0.0) - (long_quote.mid_price or 0.0)
    width = abs(long_quote.strike - short_quote.strike)
    selection_mode = "DELTA" if best_delta_pair is not None else ("STRUCTURE_DISTANCE_FALLBACK" if has_deltas else "RV_DISTANCE")
    structure_reasons = [
        f"Selected short {option_type.value} strike {short_quote.strike} and protective wing {long_quote.strike}.",
        f"Net credit {credit:.2f} points on width {width:.2f} points.",
    ]
    derived_margin = max((width - credit) * snapshot.lot_size, 0.0)
    return TradeStructure(
        strategy=strategy,
        legs=[
            StrategyLeg(action="SELL", option_type=option_type, strike=short_quote.strike, quote=short_quote),
            StrategyLeg(action="BUY", option_type=option_type, strike=long_quote.strike, quote=long_quote),
        ],
        credit_points=credit,
        width_points=width,
        call_width_points=width if option_type == OptionType.CALL else 0.0,
        put_width_points=width if option_type == OptionType.PUT else 0.0,
        margin_estimate_per_lot=snapshot.option_chain.margin_estimate_per_lot or derived_margin,
        rationale=structure_reasons,
        metadata={
            "selection_mode": selection_mode,
            "preferred_width_points": preferred_width,
            "margin_source": "BROKER_ESTIMATE" if snapshot.option_chain.margin_estimate_per_lot is not None else "MAX_LOSS_PROXY",
        },
    ), structure_reasons


def _shortlist_condor_quotes(
    snapshot: MarketSnapshot,
    regime_state: RegimeState,
    option_type: OptionType,
    short_delta_band: tuple[float, float],
    *,
    max_candidates: int = 6,
) -> list[OptionsContractQuote]:
    quotes = _liquid_otm_quotes(snapshot.option_chain.quotes, option_type=option_type, spot=snapshot.option_chain.spot)
    if not quotes:
        return []
    target_delta = sum(short_delta_band) / 2.0
    rv_points = snapshot.option_chain.spot * regime_state.rv30_pct / 100.0
    target_distance = max(40.0, min(120.0, rv_points * 1.2))

    def _score(quote: OptionsContractQuote) -> float:
        distance = abs(quote.strike - snapshot.option_chain.spot)
        delta_penalty = abs(abs(quote.delta) - target_delta) if quote.delta is not None else 0.20
        distance_penalty = abs(distance - target_distance) / max(target_distance, 1.0)
        return delta_penalty + distance_penalty + quote.spread_ratio

    return sorted(quotes, key=_score)[:max_candidates]


def _select_condor(
    snapshot: MarketSnapshot,
    regime_state: RegimeState,
    params: AdaptiveParameters,
) -> tuple[TradeStructure | None, list[str]]:
    playbook = str(regime_state.metadata.get("playbook") or "UNKNOWN")
    range_condor_mode = playbook == "RANGE_BALANCED_CONDOR"
    short_delta_band = (0.12, 0.20) if range_condor_mode else (0.10, 0.15)
    long_delta_band = (0.05, 0.10) if range_condor_mode else (0.05, 0.08)
    call_candidates = _shortlist_condor_quotes(snapshot, regime_state, OptionType.CALL, short_delta_band)
    put_candidates = _shortlist_condor_quotes(snapshot, regime_state, OptionType.PUT, short_delta_band)
    call_side, call_reasons = _select_vertical(
        strategy=StrategyType.BEAR_CALL_CREDIT_SPREAD,
        snapshot=snapshot,
        regime_state=regime_state,
        option_type=OptionType.CALL,
        short_delta_band=short_delta_band,
        long_delta_band=long_delta_band,
        min_credit_ratio=0.0,
        candidate_quotes=call_candidates,
    )
    put_side, put_reasons = _select_vertical(
        strategy=StrategyType.BULL_PUT_CREDIT_SPREAD,
        snapshot=snapshot,
        regime_state=regime_state,
        option_type=OptionType.PUT,
        short_delta_band=short_delta_band,
        long_delta_band=long_delta_band,
        min_credit_ratio=0.0,
        candidate_quotes=put_candidates,
    )
    reasons = call_reasons + put_reasons
    if not call_side or not put_side:
        reasons.append("Condor construction requires both call and put spreads.")
        return None, reasons

    total_credit = call_side.credit_points + put_side.credit_points
    max_width = max(call_side.width_points, put_side.width_points)
    condor_credit_ratio = float(regime_state.metadata.get("range_condor_credit_ratio") or params.condor_credit_width_ratio)
    condor_credit_ratio = min(condor_credit_ratio, params.condor_credit_width_ratio) if range_condor_mode else params.condor_credit_width_ratio
    if total_credit < condor_credit_ratio * max_width:
        reasons.append("Condor total credit does not meet the required credit-efficiency threshold.")
        return None, reasons

    derived_margin = max(
        max(call_side.width_points - total_credit, 0.0),
        max(put_side.width_points - total_credit, 0.0),
    ) * snapshot.lot_size
    return TradeStructure(
        strategy=StrategyType.IRON_CONDOR,
        legs=call_side.legs + put_side.legs,
        credit_points=total_credit,
        width_points=max_width,
        call_width_points=call_side.width_points,
        put_width_points=put_side.width_points,
        margin_estimate_per_lot=snapshot.option_chain.margin_estimate_per_lot or derived_margin,
        rationale=[
            f"Selected iron condor with total credit {total_credit:.2f} points.",
            f"Call width {call_side.width_points:.2f} points and put width {put_side.width_points:.2f} points.",
        ],
        metadata={
            "call_selection_mode": call_side.metadata.get("selection_mode"),
            "put_selection_mode": put_side.metadata.get("selection_mode"),
            "condor_credit_ratio": condor_credit_ratio,
            "margin_source": "BROKER_ESTIMATE" if snapshot.option_chain.margin_estimate_per_lot is not None else "MAX_LOSS_PROXY",
        },
    ), [
        f"Iron condor total credit {total_credit:.2f} points satisfies the {condor_credit_ratio:.2f} credit/width rule.",
    ]
