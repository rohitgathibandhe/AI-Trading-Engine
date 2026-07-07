from __future__ import annotations

from dataclasses import replace
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


DIRECTIONAL_WIDTH_CHOICES = (50.0, 75.0, 100.0, 150.0)
CANDIDATE_REJECTION_PRIORITY = (
    "LIQUIDITY_BAD",
    "OI_WALL_BELOW_SHORT_STRIKE",
    "DELTA_TOO_HIGH",
    "INVALIDATION_TOO_CLOSE",
    "CREDIT_TOO_LOW_IV_ADJUSTED",
    "CREDIT_TOO_LOW",
    "HEDGE_TOO_EXPENSIVE",
    "WIDTH_TOO_LARGE",
)


def select_structure(
    strategy: StrategyType,
    snapshot: MarketSnapshot,
    regime_state: RegimeState,
    params: AdaptiveParameters | None = None,
) -> tuple[TradeStructure | None, list[str]]:
    structure, reasons, _ = select_best_structure(
        strategy,
        snapshot,
        regime_state,
        params=params,
    )
    return structure, reasons


def select_best_structure(
    strategy: StrategyType,
    snapshot: MarketSnapshot,
    regime_state: RegimeState,
    params: AdaptiveParameters | None = None,
    *,
    setup_quality_score: float | None = None,
    playbook_tier: str = "A",
) -> tuple[TradeStructure | None, list[str], dict[str, object]]:
    params = (params or AdaptiveParameters()).clamped()
    setup_quality = float(
        setup_quality_score
        if setup_quality_score is not None
        else regime_state.metadata.get("setup_quality_score") or 0.0
    )
    if strategy == StrategyType.BEAR_CALL_CREDIT_SPREAD:
        return _evaluate_vertical_candidates(
            strategy=strategy,
            snapshot=snapshot,
            regime_state=regime_state,
            option_type=OptionType.CALL,
            short_delta_band=(0.18, 0.25),
            long_delta_band=(0.05, 0.15),
            params=params,
            setup_quality_score=setup_quality,
            playbook_tier=playbook_tier,
            allow_distance_fallback_when_deltas_present=False,
            min_short_strike=regime_state.metadata.get("min_short_call_strike"),
        )
    if strategy == StrategyType.BULL_PUT_CREDIT_SPREAD:
        playbook = regime_state.metadata.get("playbook")
        short_delta_band = (0.18, 0.25)
        long_delta_band = (0.05, 0.10)
        if (
            playbook == "SIDEWAYS_TO_BULLISH_RECLAIM"
            and regime_state.metadata.get("bullish_setup") == "VWAP_HOLD_HIGHER_LOW"
            and bool(regime_state.metadata.get("early_sideways_bullish_ready"))
            and regime_state.metadata.get("smart_money_bias") == "BULLISH"
        ):
            short_delta_band = (0.10, 0.25)
            long_delta_band = (0.0, 0.10)
        return _evaluate_vertical_candidates(
            strategy=strategy,
            snapshot=snapshot,
            regime_state=regime_state,
            option_type=OptionType.PUT,
            short_delta_band=short_delta_band,
            long_delta_band=long_delta_band,
            params=params,
            setup_quality_score=setup_quality,
            playbook_tier=playbook_tier,
            allow_distance_fallback_when_deltas_present=False,
            max_short_strike=regime_state.metadata.get("max_short_put_strike"),
        )
    if strategy == StrategyType.IRON_CONDOR:
        return _select_condor_candidates(
            snapshot=snapshot,
            regime_state=regime_state,
            params=params,
            setup_quality_score=setup_quality,
            playbook_tier=playbook_tier,
        )
    if strategy == StrategyType.SHORT_STRANGLE:
        return _select_strangle_candidates(
            snapshot=snapshot,
            regime_state=regime_state,
            params=params,
            setup_quality_score=setup_quality,
            playbook_tier=playbook_tier,
        )
    if strategy == StrategyType.SHORT_STRADDLE:
        return _select_straddle_candidates(
            snapshot=snapshot,
            regime_state=regime_state,
            params=params,
            setup_quality_score=setup_quality,
            playbook_tier=playbook_tier,
        )
    report = {
        "strategy": strategy.value,
        "setup_quality_score": round(setup_quality, 4),
        "monetization_score": 0.0,
        "final_trade_score": 0.0,
        "passed_spread_construction": False,
        "passed_liquidity": False,
        "passed_credit_width": False,
        "passed_delta_band": False,
        "passed_anchor_distance": False,
        "canonical_rejection_reason": "UNSUPPORTED_STRATEGY",
        "best_candidate": None,
        "best_failed_candidate": None,
        "candidate_evaluations": [],
    }
    return None, ["No allowed structure for the requested strategy."], report


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


def _effective_directional_credit_ratio(
    regime_state: RegimeState,
    params: AdaptiveParameters,
    *,
    setup_quality_score: float,
    playbook_tier: str,
) -> float:
    playbook = str(regime_state.metadata.get("playbook") or "UNKNOWN")
    base_ratio = params.directional_credit_width_ratio
    if (
        playbook in {"SIDEWAYS_TO_BULLISH_RECLAIM", "SIDEWAYS_TO_BEARISH_REJECTION", "GAP_DOWN_BEARISH_CONTINUATION", "GAP_UP_BEARISH_FAILURE"}
        and setup_quality_score >= params.strong_setup_quality_threshold
    ):
        return min(base_ratio, params.relaxed_directional_credit_width_ratio)
    if playbook_tier != "A" or setup_quality_score <= params.weak_setup_quality_threshold:
        return max(base_ratio, params.strict_directional_credit_width_ratio)
    return base_ratio


def _effective_hedge_cost_ratio_cap(
    params: AdaptiveParameters,
    *,
    setup_quality_score: float,
    playbook_tier: str,
) -> float:
    if setup_quality_score >= params.strong_setup_quality_threshold:
        return params.relaxed_hedge_cost_ratio_cap
    if playbook_tier != "A" or setup_quality_score <= params.weak_setup_quality_threshold:
        return params.strict_hedge_cost_ratio_cap
    return params.hedge_cost_ratio_cap


def _iv_adjusted_credit_ratio(
    required_credit_ratio: float,
    avg_chain_iv: float | None,
) -> tuple[float, bool]:
    if avg_chain_iv is None:
        return required_credit_ratio, False
    if float(avg_chain_iv) <= 15.0:
        return required_credit_ratio * 0.72, True
    if float(avg_chain_iv) <= 18.0:
        return required_credit_ratio * 0.88, True
    if float(avg_chain_iv) >= 22.0:
        return required_credit_ratio * 1.15, True
    return required_credit_ratio, False


def _width_preference_bonus(
    width_points: float,
    avg_chain_iv: float | None,
) -> float:
    if avg_chain_iv is None:
        return 0.0
    if float(avg_chain_iv) > 22.0 and width_points == 75.0:
        return 0.35
    if float(avg_chain_iv) < 15.0 and width_points == 50.0:
        return 0.20
    return 0.0


def _candidate_selection_mode(
    *,
    has_deltas: bool,
    delta_pass: bool,
    fallback_pass: bool,
) -> str:
    if has_deltas and delta_pass:
        return "DELTA"
    if fallback_pass:
        return "STRUCTURE_DISTANCE_FALLBACK" if has_deltas else "RV_DISTANCE"
    return "NONE"


def _candidate_rejection_reason(flags: dict[str, bool]) -> str:
    for reason in CANDIDATE_REJECTION_PRIORITY:
        if not flags.get(reason, True):
            return reason
    return "NONE"


def _score_candidate(
    *,
    credit_points: float,
    width_points: float,
    short_delta_abs: float | None,
    short_delta_band: tuple[float, float],
    liquidity_quality: float,
    credit_ratio: float,
    required_credit_ratio: float,
    anchor_distance_points: float | None,
    target_anchor_buffer_points: float | None,
    hedge_cost_ratio: float,
    theta_efficiency: float,
    slippage_penalty: float,
    stop_pain: float,
) -> float:
    short_target = sum(short_delta_band) / 2.0
    short_half_band = max((short_delta_band[1] - short_delta_band[0]) / 2.0, 0.01)
    if short_delta_abs is None:
        delta_score = 0.55
    else:
        delta_score = max(0.0, 1.0 - (abs(short_delta_abs - short_target) / short_half_band))
    credit_score = min(1.5, credit_ratio / max(required_credit_ratio, 0.01))
    if anchor_distance_points is None:
        anchor_score = 0.5
    else:
        target_anchor = max(target_anchor_buffer_points or 0.0, 1.0)
        anchor_score = min(1.5, max(anchor_distance_points, 0.0) / target_anchor)
    hedge_score = max(0.0, 1.0 - (hedge_cost_ratio / 0.40))
    theta_score = min(1.5, theta_efficiency / 0.20)
    stop_pain_penalty = min(1.0, stop_pain)
    score = (
        3.0 * credit_score
        + 2.0 * delta_score
        + 1.5 * liquidity_quality
        + 1.5 * anchor_score
        + 1.0 * hedge_score
        + 1.0 * theta_score
        - 1.5 * slippage_penalty
        - 1.0 * stop_pain_penalty
        + min(0.6, credit_points / max(width_points, 1.0))
    )
    return round(score, 4)


def _evaluate_vertical_candidates(
    strategy: StrategyType,
    snapshot: MarketSnapshot,
    regime_state: RegimeState,
    option_type: OptionType,
    short_delta_band: tuple[float, float],
    long_delta_band: tuple[float, float],
    params: AdaptiveParameters,
    *,
    setup_quality_score: float,
    playbook_tier: str,
    allow_distance_fallback_when_deltas_present: bool = True,
    allow_wider_width_fallback: bool = True,
    min_short_strike: float | None = None,
    max_short_strike: float | None = None,
    candidate_quotes: list[OptionsContractQuote] | None = None,
) -> tuple[TradeStructure | None, list[str], dict[str, object]]:
    quotes = candidate_quotes or _liquid_otm_quotes(snapshot.option_chain.quotes, option_type=option_type, spot=snapshot.option_chain.spot)
    playbook = str(regime_state.metadata.get("playbook") or "UNKNOWN")
    if len(quotes) < 2:
        report = {
            "strategy": strategy.value,
            "playbook": playbook,
            "setup_quality_score": round(setup_quality_score, 4),
            "monetization_score": 0.0,
            "final_trade_score": 0.0,
            "passed_spread_construction": False,
            "passed_liquidity": False,
            "passed_credit_width": False,
            "passed_delta_band": False,
            "passed_anchor_distance": False,
            "canonical_rejection_reason": "LIQUIDITY_BAD",
            "best_candidate": None,
            "best_failed_candidate": None,
            "candidate_evaluations": [],
        }
        return None, ["No liquid OTM quotes available for vertical selection."], report

    quote_by_strike = {quote.strike: quote for quote in quotes}
    preferred_width = regime_state.metadata.get("preferred_width_points")
    preferred_widths = tuple(float(value) for value in (regime_state.metadata.get("allowed_width_points") or ()))
    target_short_put_buffer = regime_state.metadata.get("target_short_put_buffer_points")
    base_required_credit_ratio = _effective_directional_credit_ratio(
        regime_state,
        params,
        setup_quality_score=setup_quality_score,
        playbook_tier=playbook_tier,
    )
    avg_chain_iv = (
        float(regime_state.metadata.get("avg_chain_iv"))
        if regime_state.metadata.get("avg_chain_iv") is not None
        else None
    )
    required_credit_ratio, iv_adjusted_credit_floor_active = _iv_adjusted_credit_ratio(
        base_required_credit_ratio,
        avg_chain_iv,
    )
    max_hedge_cost_ratio = _effective_hedge_cost_ratio_cap(
        params,
        setup_quality_score=setup_quality_score,
        playbook_tier=playbook_tier,
    )
    call_wall_strike = None
    if option_type == OptionType.CALL:
        raw_call_wall = regime_state.metadata.get("current_call_wall") or regime_state.metadata.get("call_resistance_strike")
        call_wall_strike = float(raw_call_wall) if raw_call_wall is not None else None
    rv_points = snapshot.option_chain.spot * regime_state.rv30_pct / 100.0
    hours_remaining = max(
        (
            snapshot.timestamp.replace(hour=15, minute=15, second=0, microsecond=0)
            - snapshot.timestamp
        ).total_seconds()
        / 3600.0,
        0.5,
    )
    has_deltas = any(quote.delta is not None for quote in quotes)
    candidate_evaluations: list[dict[str, object]] = []

    for requested_width in params.candidate_widths:
        width_seen = False
        for short_quote in quotes:
            if min_short_strike is not None and short_quote.strike < min_short_strike:
                continue
            if max_short_strike is not None and short_quote.strike > max_short_strike:
                continue
            if option_type == OptionType.CALL and call_wall_strike is not None and short_quote.strike < call_wall_strike:
                width_seen = True
                candidate_evaluations.append(
                    {
                        "requested_width_points": requested_width,
                        "width_points": requested_width,
                        "short_strike": short_quote.strike,
                        "long_strike": None,
                        "call_wall_strike": call_wall_strike,
                        "valid": False,
                        "selection_mode": "NONE",
                        "canonical_rejection_reason": "OI_WALL_BELOW_SHORT_STRIKE",
                        "rejection_detail": "Bear-call short strike must be at or above the nearby call wall.",
                    }
                )
                continue
            long_strike = short_quote.strike + requested_width if option_type == OptionType.CALL else short_quote.strike - requested_width
            long_quote = quote_by_strike.get(long_strike)
            if long_quote is None:
                continue
            width_seen = True
            short_mid = short_quote.mid_price
            long_mid = long_quote.mid_price
            if short_mid is None or long_mid is None:
                candidate_evaluations.append(
                    {
                        "requested_width_points": requested_width,
                        "width_points": requested_width,
                        "short_strike": short_quote.strike,
                        "long_strike": long_quote.strike,
                        "valid": False,
                        "selection_mode": "NONE",
                        "canonical_rejection_reason": "LIQUIDITY_BAD",
                        "rejection_detail": "Missing reliable mid-price for one or both legs.",
                    }
                )
                continue
            credit = short_mid - long_mid
            width = abs(long_quote.strike - short_quote.strike)
            short_spread_ratio = short_quote.spread_ratio
            long_spread_ratio = long_quote.spread_ratio
            avg_spread_ratio = (short_spread_ratio + long_spread_ratio) / 2.0
            passed_liquidity = bool(
                short_spread_ratio <= params.liquidity_spread_ratio_cap
                and long_spread_ratio <= params.liquidity_spread_ratio_cap
            )
            credit_ratio = credit / width if width > 0 else 0.0
            passed_credit_width = credit > 0 and credit_ratio >= required_credit_ratio
            short_abs = abs(short_quote.delta) if short_quote.delta is not None else None
            long_abs = abs(long_quote.delta) if long_quote.delta is not None else None
            passed_delta = True
            if has_deltas:
                passed_delta = bool(
                    short_abs is not None
                    and long_abs is not None
                    and short_delta_band[0] <= short_abs <= short_delta_band[1]
                    and long_delta_band[0] <= long_abs <= long_delta_band[1]
                )
            anchor_level = max_short_strike if option_type == OptionType.PUT else min_short_strike
            if option_type == OptionType.PUT:
                anchor_distance = (max_short_strike - short_quote.strike) if max_short_strike is not None else None
            else:
                anchor_distance = (short_quote.strike - min_short_strike) if min_short_strike is not None else None
            fallback_pass = False
            if option_type == OptionType.CALL:
                fallback_pass = (short_quote.strike - snapshot.option_chain.spot) >= rv_points and width in DIRECTIONAL_WIDTH_CHOICES
            else:
                fallback_pass = (snapshot.option_chain.spot - short_quote.strike) >= rv_points and width in DIRECTIONAL_WIDTH_CHOICES
            passed_anchor_distance = bool(
                (anchor_distance is None or anchor_distance >= params.minimum_anchor_distance_points)
                and (anchor_distance is None or anchor_distance >= 0)
            )
            hedge_cost_ratio = long_mid / max(short_mid, 0.01)
            passed_hedge_cost = bool(hedge_cost_ratio <= max_hedge_cost_ratio)
            selection_mode = _candidate_selection_mode(
                has_deltas=has_deltas,
                delta_pass=passed_delta,
                fallback_pass=passed_anchor_distance and fallback_pass and allow_distance_fallback_when_deltas_present,
            )
            valid = bool(
                passed_liquidity
                and passed_credit_width
                and passed_hedge_cost
                and passed_anchor_distance
                and selection_mode != "NONE"
            )
            theta_efficiency = credit_ratio * max(hours_remaining / 4.0, 0.5)
            slippage_penalty = snapshot.slippage_points / max(credit, 0.50)
            stop_pain = ((width - credit) / max(width, 1.0)) * (short_abs or 0.20)
            liquidity_quality = max(0.0, 1.0 - (avg_spread_ratio / max(params.liquidity_spread_ratio_cap, 0.01)))
            monetization_score = _score_candidate(
                credit_points=credit,
                width_points=width,
                short_delta_abs=short_abs,
                short_delta_band=short_delta_band,
                liquidity_quality=liquidity_quality,
                credit_ratio=credit_ratio,
                required_credit_ratio=required_credit_ratio,
                anchor_distance_points=anchor_distance,
                target_anchor_buffer_points=target_short_put_buffer,
                hedge_cost_ratio=hedge_cost_ratio,
                theta_efficiency=theta_efficiency,
                slippage_penalty=slippage_penalty,
                stop_pain=stop_pain,
            )
            preferred_width_penalty = (
                params.width_preference_penalty
                if preferred_widths and requested_width not in preferred_widths
                else 0.0
            )
            width_preference_bonus = _width_preference_bonus(width, avg_chain_iv)
            monetization_score = round(monetization_score - preferred_width_penalty + width_preference_bonus, 4)
            rejection_flags = {
                "LIQUIDITY_BAD": passed_liquidity,
                "OI_WALL_BELOW_SHORT_STRIKE": True,
                "DELTA_TOO_HIGH": passed_delta or (selection_mode == "STRUCTURE_DISTANCE_FALLBACK"),
                "INVALIDATION_TOO_CLOSE": passed_anchor_distance,
                "CREDIT_TOO_LOW_IV_ADJUSTED": (passed_credit_width or not iv_adjusted_credit_floor_active),
                "CREDIT_TOO_LOW": passed_credit_width,
                "HEDGE_TOO_EXPENSIVE": passed_hedge_cost,
                "WIDTH_TOO_LARGE": True,
            }
            candidate_evaluations.append(
                {
                    "requested_width_points": requested_width,
                    "width_points": width,
                    "preferred_width_penalty": round(preferred_width_penalty, 4),
                    "width_preference_bonus": round(width_preference_bonus, 4),
                    "playbook_preferred_widths": preferred_widths,
                    "short_strike": short_quote.strike,
                    "long_strike": long_quote.strike,
                    "call_wall_strike": call_wall_strike,
                    "anchor_level": anchor_level,
                    "anchor_distance_points": anchor_distance,
                    "target_anchor_buffer_points": target_short_put_buffer,
                    "short_delta_abs": short_abs,
                    "long_delta_abs": long_abs,
                    "credit_points": round(credit, 4),
                    "credit_width_ratio": round(credit_ratio, 4),
                    "base_required_credit_width_ratio": round(base_required_credit_ratio, 4),
                    "required_credit_width_ratio": round(required_credit_ratio, 4),
                    "iv_adjusted_credit_floor_active": iv_adjusted_credit_floor_active,
                    "avg_chain_iv": round(avg_chain_iv, 4) if avg_chain_iv is not None else None,
                    "max_hedge_cost_ratio": round(max_hedge_cost_ratio, 4),
                    "hedge_cost_points": round(long_mid, 4),
                    "hedge_cost_ratio": round(hedge_cost_ratio, 4),
                    "liquidity_quality": round(liquidity_quality, 4),
                    "avg_spread_ratio": round(avg_spread_ratio, 4),
                    "theta_efficiency": round(theta_efficiency, 4),
                    "slippage_penalty": round(slippage_penalty, 4),
                    "expected_stop_pain": round(stop_pain, 4),
                    "selection_mode": selection_mode,
                    "passed_liquidity": passed_liquidity,
                    "passed_credit_width": passed_credit_width,
                    "passed_delta_band": passed_delta,
                    "passed_anchor_distance": passed_anchor_distance,
                    "passed_hedge_cost": passed_hedge_cost,
                    "valid": valid,
                    "monetization_score": monetization_score,
                    "canonical_rejection_reason": "NONE" if valid else _candidate_rejection_reason(rejection_flags),
                    "rejection_detail": "" if valid else "Candidate failed monetization filters.",
                }
            )
        if not width_seen:
            candidate_evaluations.append(
                {
                    "requested_width_points": requested_width,
                    "width_points": requested_width,
                    "valid": False,
                    "selection_mode": "NONE",
                    "canonical_rejection_reason": "WIDTH_TOO_LARGE",
                    "rejection_detail": "No liquid pair exists for this requested width.",
                }
            )

    valid_candidates = [candidate for candidate in candidate_evaluations if candidate.get("valid")]
    best_candidate = max(valid_candidates, key=lambda candidate: (float(candidate.get("monetization_score") or 0.0), -abs(float(candidate.get("width_points") or 0.0) - float(preferred_width or candidate.get("width_points") or 0.0)))) if valid_candidates else None
    failed_candidates = [candidate for candidate in candidate_evaluations if not candidate.get("valid")]
    best_failed_candidate = max(failed_candidates, key=lambda candidate: float(candidate.get("monetization_score") or -inf)) if failed_candidates else None
    passed_liquidity = any(bool(candidate.get("passed_liquidity")) for candidate in candidate_evaluations)
    passed_credit_width = any(bool(candidate.get("passed_credit_width")) for candidate in candidate_evaluations)
    passed_delta_band = any(bool(candidate.get("passed_delta_band")) for candidate in candidate_evaluations)
    passed_anchor_distance = any(bool(candidate.get("passed_anchor_distance")) for candidate in candidate_evaluations)
    report = {
        "strategy": strategy.value,
        "playbook": playbook,
        "setup_quality_score": round(setup_quality_score, 4),
        "monetization_score": round(float(best_candidate.get("monetization_score") or 0.0) if best_candidate else float(best_failed_candidate.get("monetization_score") or 0.0) if best_failed_candidate else 0.0, 4),
        "final_trade_score": round(setup_quality_score + (float(best_candidate.get("monetization_score") or 0.0) if best_candidate else 0.0), 4),
        "passed_spread_construction": best_candidate is not None,
        "passed_liquidity": passed_liquidity,
        "passed_credit_width": passed_credit_width,
        "passed_delta_band": passed_delta_band,
        "passed_anchor_distance": passed_anchor_distance,
        "canonical_rejection_reason": "NONE" if best_candidate else str(best_failed_candidate.get("canonical_rejection_reason") or "WIDTH_TOO_LARGE") if best_failed_candidate else "WIDTH_TOO_LARGE",
        "best_candidate": best_candidate,
        "best_failed_candidate": best_failed_candidate,
        "candidate_evaluations": candidate_evaluations,
        "narrower_width_would_pass": bool(
            best_failed_candidate is not None
            and any(
                candidate.get("valid")
                and float(candidate.get("width_points") or 0.0) < float(best_failed_candidate.get("width_points") or 0.0)
                for candidate in candidate_evaluations
            )
        ),
    }
    if best_candidate is None and allow_wider_width_fallback:
        meaningful_failures = [
            candidate
            for candidate in failed_candidates
            if candidate.get("canonical_rejection_reason") != "WIDTH_TOO_LARGE"
        ]
        if meaningful_failures and all(
            str(candidate.get("canonical_rejection_reason") or "") == "INVALIDATION_TOO_CLOSE"
            for candidate in meaningful_failures
        ):
            fallback_width = min(max(params.candidate_widths) * 1.5, 200.0)
            if fallback_width not in params.candidate_widths:
                fallback_params = replace(params, candidate_widths=(fallback_width,))
                fallback_structure, fallback_reasons, fallback_report = _evaluate_vertical_candidates(
                    strategy=strategy,
                    snapshot=snapshot,
                    regime_state=regime_state,
                    option_type=option_type,
                    short_delta_band=short_delta_band,
                    long_delta_band=long_delta_band,
                    params=fallback_params,
                    setup_quality_score=setup_quality_score,
                    playbook_tier=playbook_tier,
                    allow_distance_fallback_when_deltas_present=allow_distance_fallback_when_deltas_present,
                    allow_wider_width_fallback=False,
                    min_short_strike=min_short_strike,
                    max_short_strike=max_short_strike,
                    candidate_quotes=quotes,
                )
                for candidate in fallback_report.get("candidate_evaluations") or []:
                    if isinstance(candidate, dict):
                        candidate["fallback_width_retry"] = True
                        candidate_evaluations.append(candidate)
                if fallback_structure is not None:
                    fallback_structure.metadata["fallback_width_retry"] = True
                    fallback_structure.metadata["fallback_from_invalidations"] = True
                    fallback_report["candidate_evaluations"] = candidate_evaluations
                    fallback_report["used_fallback_width_retry"] = True
                    return (
                        fallback_structure,
                        ["All primary candidates failed because invalidation was too close; retried with one wider fallback width."]
                        + fallback_reasons,
                        fallback_report,
                    )
                report["candidate_evaluations"] = candidate_evaluations
                report["used_fallback_width_retry"] = True
                report["canonical_rejection_reason"] = str(
                    fallback_report.get("canonical_rejection_reason") or report["canonical_rejection_reason"]
                )
    if best_candidate is None:
        return None, ["No vertical spread satisfied liquidity, delta/distance, and credit-efficiency rules."], report

    short_quote = quote_by_strike[float(best_candidate["short_strike"])]
    long_quote = quote_by_strike[float(best_candidate["long_strike"])]
    credit = float(best_candidate["credit_points"])
    width = float(best_candidate["width_points"])
    structure_reasons = [
        f"Selected short {option_type.value} strike {short_quote.strike} and protective wing {long_quote.strike}.",
        f"Net credit {credit:.2f} points on width {width:.2f} points.",
        f"Monetization score {float(best_candidate['monetization_score']):.2f} with setup quality {setup_quality_score:.2f}.",
    ]
    derived_margin = max((width - credit) * snapshot.lot_size, 0.0)
    structure = TradeStructure(
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
            "selection_mode": str(best_candidate["selection_mode"]),
            "preferred_width_points": preferred_width,
            "preferred_width_penalty": best_candidate.get("preferred_width_penalty"),
            "width_preference_bonus": best_candidate.get("width_preference_bonus"),
            "margin_source": "BROKER_ESTIMATE" if snapshot.option_chain.margin_estimate_per_lot is not None else "MAX_LOSS_PROXY",
            "monetization_score": float(best_candidate["monetization_score"]),
            "setup_quality_score": setup_quality_score,
            "final_trade_score": round(setup_quality_score + float(best_candidate["monetization_score"]), 4),
            "short_delta_abs": best_candidate.get("short_delta_abs"),
            "credit_width_ratio": best_candidate.get("credit_width_ratio"),
            "anchor_distance_points": best_candidate.get("anchor_distance_points"),
            "avg_chain_iv": best_candidate.get("avg_chain_iv"),
        },
    )
    return structure, structure_reasons, report


def _select_condor_candidates(
    snapshot: MarketSnapshot,
    regime_state: RegimeState,
    params: AdaptiveParameters,
    *,
    setup_quality_score: float,
    playbook_tier: str,
) -> tuple[TradeStructure | None, list[str], dict[str, object]]:
    playbook = str(regime_state.metadata.get("playbook") or "UNKNOWN")
    delta_profiles = (
        ("08_14", (0.08, 0.14), (0.03, 0.08)),
        ("10_16", (0.10, 0.16), (0.04, 0.09)),
        ("12_20", (0.12, 0.20), (0.05, 0.10)),
    )
    preferred_delta_label = str(regime_state.metadata.get("condor_short_delta_band_bias") or "12_20")
    shape_bias = str(regime_state.metadata.get("condor_shape_bias") or "SYMMETRIC")
    compression_score = float(regime_state.metadata.get("range_compression_score") or 0.0)
    avg_chain_iv = regime_state.metadata.get("avg_chain_iv")
    call_by_strike = {quote.strike: quote for quote in _liquid_otm_quotes(snapshot.option_chain.quotes, OptionType.CALL, snapshot.option_chain.spot)}
    put_by_strike = {quote.strike: quote for quote in _liquid_otm_quotes(snapshot.option_chain.quotes, OptionType.PUT, snapshot.option_chain.spot)}
    if not call_by_strike or not put_by_strike:
        report = {
            "strategy": StrategyType.IRON_CONDOR.value,
            "playbook": playbook,
            "setup_quality_score": round(setup_quality_score, 4),
            "monetization_score": 0.0,
            "final_trade_score": 0.0,
            "passed_spread_construction": False,
            "passed_liquidity": False,
            "passed_credit_width": False,
            "passed_delta_band": False,
            "passed_anchor_distance": False,
            "canonical_rejection_reason": "LIQUIDITY_BAD",
            "best_candidate": None,
            "best_failed_candidate": None,
            "candidate_evaluations": [],
        }
        return None, ["Iron condor candidate search could not find liquid call and put wings."], report

    candidate_evaluations: list[dict[str, object]] = []
    condor_credit_ratio = params.condor_credit_width_ratio
    if setup_quality_score >= params.strong_setup_quality_threshold:
        condor_credit_ratio = max(0.14, condor_credit_ratio - 0.05)
    elif playbook_tier != "A":
        condor_credit_ratio = min(0.40, condor_credit_ratio + 0.03)
    if compression_score >= 0.75:
        condor_credit_ratio = max(0.12, condor_credit_ratio - 0.03)
    if avg_chain_iv is not None and float(avg_chain_iv) >= 22.0:
        condor_credit_ratio = max(0.12, condor_credit_ratio - 0.02)
    if avg_chain_iv is not None and float(avg_chain_iv) <= 15.0:
        condor_credit_ratio = min(0.40, condor_credit_ratio + 0.03)
    max_hedge_cost_ratio = _effective_hedge_cost_ratio_cap(
        params,
        setup_quality_score=setup_quality_score,
        playbook_tier=playbook_tier,
    )
    hours_remaining = max(
        (
            snapshot.timestamp.replace(hour=15, minute=15, second=0, microsecond=0)
            - snapshot.timestamp
        ).total_seconds()
        / 3600.0,
        0.5,
    )

    for delta_band_label, short_delta_band, long_delta_band in delta_profiles:
        call_candidates = _shortlist_condor_quotes(snapshot, regime_state, OptionType.CALL, short_delta_band)
        put_candidates = _shortlist_condor_quotes(snapshot, regime_state, OptionType.PUT, short_delta_band)
        for call_width in params.candidate_widths:
            for put_width in params.candidate_widths:
                if compression_score < 0.25 and min(call_width, put_width) < 75.0:
                    continue
                if compression_score >= 0.75 and avg_chain_iv is not None and float(avg_chain_iv) >= 22.0 and max(call_width, put_width) > 100.0:
                    continue
                shape_label = (
                    "SYMMETRIC"
                    if call_width == put_width
                    else ("ASYMMETRIC_CALL_WIDER" if call_width > put_width else "ASYMMETRIC_PUT_WIDER")
                )
                width_seen = False
                for short_call in call_candidates:
                    long_call = call_by_strike.get(short_call.strike + call_width)
                    if long_call is None:
                        continue
                    for short_put in put_candidates:
                        long_put = put_by_strike.get(short_put.strike - put_width)
                        if long_put is None:
                            continue
                        width_seen = True
                        call_credit = (short_call.mid_price or 0.0) - (long_call.mid_price or 0.0)
                        put_credit = (short_put.mid_price or 0.0) - (long_put.mid_price or 0.0)
                        total_credit = call_credit + put_credit
                        max_width = max(call_width, put_width)
                        avg_spread_ratio = (
                            short_call.spread_ratio
                            + long_call.spread_ratio
                            + short_put.spread_ratio
                            + long_put.spread_ratio
                        ) / 4.0
                        passed_liquidity = bool(avg_spread_ratio <= params.liquidity_spread_ratio_cap)
                        passed_delta = all(
                            delta is not None
                            for delta in (short_call.delta, long_call.delta, short_put.delta, long_put.delta)
                        ) and (
                            short_delta_band[0] <= abs(short_call.delta) <= short_delta_band[1]
                            and short_delta_band[0] <= abs(short_put.delta) <= short_delta_band[1]
                            and long_delta_band[0] <= abs(long_call.delta) <= long_delta_band[1]
                            and long_delta_band[0] <= abs(long_put.delta) <= long_delta_band[1]
                        )
                        credit_ratio = total_credit / max_width if max_width > 0 else 0.0
                        passed_credit = total_credit > 0 and credit_ratio >= condor_credit_ratio
                        hedge_cost_ratio = ((long_call.mid_price or 0.0) + (long_put.mid_price or 0.0)) / max((short_call.mid_price or 0.01) + (short_put.mid_price or 0.01), 0.01)
                        passed_hedge_cost = hedge_cost_ratio <= max_hedge_cost_ratio
                        theta_efficiency = credit_ratio * max(hours_remaining / 4.0, 0.5)
                        slippage_penalty = (snapshot.slippage_points * 2.0) / max(total_credit, 0.50)
                        liquidity_quality = max(0.0, 1.0 - (avg_spread_ratio / max(params.liquidity_spread_ratio_cap, 0.01)))
                        monetization_score = _score_candidate(
                            credit_points=total_credit,
                            width_points=max_width,
                            short_delta_abs=max(abs(short_call.delta or 0.0), abs(short_put.delta or 0.0)),
                            short_delta_band=short_delta_band,
                            liquidity_quality=liquidity_quality,
                            credit_ratio=credit_ratio,
                            required_credit_ratio=condor_credit_ratio,
                            anchor_distance_points=None,
                            target_anchor_buffer_points=None,
                            hedge_cost_ratio=hedge_cost_ratio,
                            theta_efficiency=theta_efficiency,
                            slippage_penalty=slippage_penalty,
                            stop_pain=((max_width - total_credit) / max(max_width, 1.0)) * max(abs(short_call.delta or 0.0), abs(short_put.delta or 0.0), 0.1),
                        )
                        if delta_band_label != preferred_delta_label:
                            monetization_score = round(monetization_score - 0.25, 4)
                        if (
                            (shape_bias == "SYMMETRIC" and shape_label != "SYMMETRIC")
                            or (shape_bias == "CALL_WIDER" and shape_label != "ASYMMETRIC_CALL_WIDER")
                            or (shape_bias == "PUT_WIDER" and shape_label != "ASYMMETRIC_PUT_WIDER")
                        ):
                            monetization_score = round(monetization_score - 0.20, 4)
                        rejection_flags = {
                            "LIQUIDITY_BAD": passed_liquidity,
                            "DELTA_TOO_HIGH": passed_delta,
                            "INVALIDATION_TOO_CLOSE": True,
                            "CREDIT_TOO_LOW": passed_credit,
                            "HEDGE_TOO_EXPENSIVE": passed_hedge_cost,
                            "WIDTH_TOO_LARGE": True,
                        }
                        valid = bool(passed_liquidity and passed_delta and passed_credit and passed_hedge_cost)
                        candidate_evaluations.append(
                            {
                                "requested_width_points": max_width,
                                "width_points": max_width,
                                "call_width_points": call_width,
                                "put_width_points": put_width,
                                "shape_label": shape_label,
                                "delta_band_label": delta_band_label,
                                "short_call_strike": short_call.strike,
                                "long_call_strike": long_call.strike,
                                "short_put_strike": short_put.strike,
                                "long_put_strike": long_put.strike,
                                "credit_points": round(total_credit, 4),
                                "credit_width_ratio": round(credit_ratio, 4),
                                "required_credit_width_ratio": round(condor_credit_ratio, 4),
                                "max_hedge_cost_ratio": round(max_hedge_cost_ratio, 4),
                                "avg_spread_ratio": round(avg_spread_ratio, 4),
                                "selection_mode": "CONDOR",
                                "condor_profile": regime_state.metadata.get("condor_profile"),
                                "passed_liquidity": passed_liquidity,
                                "passed_credit_width": passed_credit,
                                "passed_delta_band": passed_delta,
                                "passed_anchor_distance": True,
                                "passed_hedge_cost": passed_hedge_cost,
                                "valid": valid,
                                "monetization_score": monetization_score,
                                "canonical_rejection_reason": "NONE" if valid else _candidate_rejection_reason(rejection_flags),
                                "rejection_detail": "" if valid else "Condor candidate failed monetization filters.",
                            }
                        )
                if not width_seen:
                    candidate_evaluations.append(
                        {
                            "requested_width_points": max(call_width, put_width),
                            "width_points": max(call_width, put_width),
                            "call_width_points": call_width,
                            "put_width_points": put_width,
                            "shape_label": shape_label,
                            "delta_band_label": delta_band_label,
                            "valid": False,
                            "selection_mode": "CONDOR",
                            "condor_profile": regime_state.metadata.get("condor_profile"),
                            "canonical_rejection_reason": "WIDTH_TOO_LARGE",
                            "rejection_detail": "No liquid condor wings exist for this width pair.",
                        }
                    )

    valid_candidates = [candidate for candidate in candidate_evaluations if candidate.get("valid")]
    best_candidate = max(valid_candidates, key=lambda candidate: float(candidate.get("monetization_score") or 0.0)) if valid_candidates else None
    failed_candidates = [candidate for candidate in candidate_evaluations if not candidate.get("valid")]
    best_failed_candidate = max(failed_candidates, key=lambda candidate: float(candidate.get("monetization_score") or -inf)) if failed_candidates else None
    report = {
        "strategy": StrategyType.IRON_CONDOR.value,
        "playbook": playbook,
        "setup_quality_score": round(setup_quality_score, 4),
        "monetization_score": round(float(best_candidate.get("monetization_score") or 0.0) if best_candidate else float(best_failed_candidate.get("monetization_score") or 0.0) if best_failed_candidate else 0.0, 4),
        "final_trade_score": round(setup_quality_score + (float(best_candidate.get("monetization_score") or 0.0) if best_candidate else 0.0), 4),
        "passed_spread_construction": best_candidate is not None,
        "passed_liquidity": any(bool(candidate.get("passed_liquidity")) for candidate in candidate_evaluations),
        "passed_credit_width": any(bool(candidate.get("passed_credit_width")) for candidate in candidate_evaluations),
        "passed_delta_band": any(bool(candidate.get("passed_delta_band")) for candidate in candidate_evaluations),
        "passed_anchor_distance": True,
        "canonical_rejection_reason": "NONE" if best_candidate else str(best_failed_candidate.get("canonical_rejection_reason") or "WIDTH_TOO_LARGE") if best_failed_candidate else "WIDTH_TOO_LARGE",
        "best_candidate": best_candidate,
        "best_failed_candidate": best_failed_candidate,
        "candidate_evaluations": candidate_evaluations,
    }
    if best_candidate is None:
        return None, ["Condor optimizer could not find a valid symmetric or asymmetric defined-risk structure."], report

    short_call = call_by_strike[float(best_candidate["short_call_strike"])]
    long_call = call_by_strike[float(best_candidate["long_call_strike"])]
    short_put = put_by_strike[float(best_candidate["short_put_strike"])]
    long_put = put_by_strike[float(best_candidate["long_put_strike"])]
    total_credit = float(best_candidate["credit_points"])
    max_width = float(best_candidate["width_points"])
    call_width = float(best_candidate["call_width_points"])
    put_width = float(best_candidate["put_width_points"])
    rationale = [
        f"Selected {best_candidate['shape_label'].lower()} iron condor with call strikes {short_call.strike}/{long_call.strike} and put strikes {short_put.strike}/{long_put.strike}.",
        f"Total credit {total_credit:.2f} points on call width {call_width:.2f} and put width {put_width:.2f}.",
        f"Delta band {best_candidate['delta_band_label']} produced monetization score {float(best_candidate['monetization_score']):.2f}.",
    ]
    derived_margin = max((max_width - total_credit) * snapshot.lot_size, 0.0)
    structure = TradeStructure(
        strategy=StrategyType.IRON_CONDOR,
        legs=[
            StrategyLeg(action="SELL", option_type=OptionType.CALL, strike=short_call.strike, quote=short_call),
            StrategyLeg(action="BUY", option_type=OptionType.CALL, strike=long_call.strike, quote=long_call),
            StrategyLeg(action="SELL", option_type=OptionType.PUT, strike=short_put.strike, quote=short_put),
            StrategyLeg(action="BUY", option_type=OptionType.PUT, strike=long_put.strike, quote=long_put),
        ],
        credit_points=total_credit,
        width_points=max_width,
        call_width_points=call_width,
        put_width_points=put_width,
        margin_estimate_per_lot=snapshot.option_chain.margin_estimate_per_lot or derived_margin,
        rationale=rationale,
        metadata={
            "selection_mode": "CONDOR",
            "condor_profile": regime_state.metadata.get("condor_profile"),
            "condor_shape": best_candidate.get("shape_label"),
            "condor_delta_band_label": best_candidate.get("delta_band_label"),
            "margin_source": "BROKER_ESTIMATE" if snapshot.option_chain.margin_estimate_per_lot is not None else "MAX_LOSS_PROXY",
            "monetization_score": float(best_candidate["monetization_score"]),
            "setup_quality_score": setup_quality_score,
            "final_trade_score": round(setup_quality_score + float(best_candidate["monetization_score"]), 4),
            "credit_width_ratio": best_candidate.get("credit_width_ratio"),
        },
    )
    return structure, rationale, report


def _select_strangle_candidates(
    snapshot: MarketSnapshot,
    regime_state: RegimeState,
    params: AdaptiveParameters,
    *,
    setup_quality_score: float,
    playbook_tier: str,
) -> tuple[TradeStructure | None, list[str], dict[str, object]]:
    """Select a short strangle: naked short OTM call + naked short OTM put.

    Used when the market is range-bound but IV is elevated enough that condor
    wing premiums are expensive relative to the short credit (hedge cost too high).
    Risk is capped practically at 2x credit (PREMIUM_SL_MULTIPLIER), sized through
    compute_max_loss_rupees_per_lot accordingly.
    """
    spot = snapshot.option_chain.spot
    avg_chain_iv = float(regime_state.metadata.get("avg_chain_iv") or 0.0)
    call_quotes = _liquid_otm_quotes(snapshot.option_chain.quotes, OptionType.CALL, spot)
    put_quotes = _liquid_otm_quotes(snapshot.option_chain.quotes, OptionType.PUT, spot)
    call_delta_band = (0.15, 0.25)   # target 0.20-0.25 OTM — professional standard
    put_delta_band = (0.15, 0.25)

    _empty_report: dict[str, object] = {
        "strategy": StrategyType.SHORT_STRANGLE.value,
        "setup_quality_score": round(setup_quality_score, 4),
        "monetization_score": 0.0,
        "final_trade_score": 0.0,
        "passed_spread_construction": False,
        "passed_liquidity": False,
        "passed_credit_width": False,
        "passed_delta_band": False,
        "passed_anchor_distance": True,
        "canonical_rejection_reason": "LIQUIDITY_BAD",
        "best_candidate": None,
        "best_failed_candidate": None,
        "candidate_evaluations": [],
    }

    if not call_quotes or not put_quotes:
        return None, ["Short strangle: no liquid OTM calls or puts found."], _empty_report

    min_total_credit = max(60.0, spot * 0.0025)  # at least 60pts or 0.25% of spot

    candidate_evaluations: list[dict[str, object]] = []
    best_valid: dict[str, object] | None = None
    best_failed: dict[str, object] | None = None

    for short_call in call_quotes:
        if short_call.delta is None:
            continue
        call_delta = abs(short_call.delta)
        if not (call_delta_band[0] <= call_delta <= call_delta_band[1]):
            continue
        for short_put in put_quotes:
            if short_put.delta is None:
                continue
            put_delta = abs(short_put.delta)
            if not (put_delta_band[0] <= put_delta <= put_delta_band[1]):
                continue
            call_credit = short_call.mid_price or short_call.ltp or 0.0
            put_credit = short_put.mid_price or short_put.ltp or 0.0
            total_credit = call_credit + put_credit
            width_points = short_call.strike - short_put.strike
            avg_spread_ratio = (short_call.spread_ratio + short_put.spread_ratio) / 2.0

            passed_liquidity = avg_spread_ratio <= params.liquidity_spread_ratio_cap
            passed_credit = total_credit >= min_total_credit
            passed_delta = True  # already filtered above
            valid = passed_liquidity and passed_credit

            # Score: reward higher credit and tighter spread; penalise extreme deltas
            delta_balance = 1.0 - abs(call_delta - put_delta) / max(call_delta + put_delta, 0.01)
            liq_quality = max(0.0, 1.0 - avg_spread_ratio / max(params.liquidity_spread_ratio_cap, 0.01))
            monetization_score = round(
                (total_credit / max(spot * 0.01, 1.0)) * 10.0
                + delta_balance * 0.5
                + liq_quality * 0.3
                - (0.2 if avg_chain_iv < 18.0 else 0.0),
                4,
            )

            rejection_flags = {
                "LIQUIDITY_BAD": passed_liquidity,
                "CREDIT_TOO_LOW": passed_credit,
                "DELTA_TOO_HIGH": passed_delta,
            }
            entry = {
                "short_call_strike": short_call.strike,
                "short_put_strike": short_put.strike,
                "call_credit": round(call_credit, 4),
                "put_credit": round(put_credit, 4),
                "total_credit": round(total_credit, 4),
                "width_points": round(width_points, 4),
                "call_delta": round(call_delta, 4),
                "put_delta": round(put_delta, 4),
                "avg_spread_ratio": round(avg_spread_ratio, 4),
                "valid": valid,
                "monetization_score": monetization_score,
                "canonical_rejection_reason": "NONE" if valid else _candidate_rejection_reason(rejection_flags),
            }
            candidate_evaluations.append(entry)
            if valid:
                if best_valid is None or monetization_score > float(best_valid.get("monetization_score") or 0.0):
                    best_valid = entry
            else:
                if best_failed is None or monetization_score > float(best_failed.get("monetization_score") or 0.0):
                    best_failed = entry

    report: dict[str, object] = {
        "strategy": StrategyType.SHORT_STRANGLE.value,
        "setup_quality_score": round(setup_quality_score, 4),
        "monetization_score": round(float(best_valid.get("monetization_score") or 0.0) if best_valid else float(best_failed.get("monetization_score") or 0.0) if best_failed else 0.0, 4),
        "final_trade_score": round(setup_quality_score + (float(best_valid.get("monetization_score") or 0.0) if best_valid else 0.0), 4),
        "passed_spread_construction": best_valid is not None,
        "passed_liquidity": any(bool(c.get("passed_liquidity", True)) for c in candidate_evaluations),
        "passed_credit_width": any(bool(c.get("passed_credit", True)) for c in candidate_evaluations),
        "passed_delta_band": True,
        "passed_anchor_distance": True,
        "canonical_rejection_reason": "NONE" if best_valid else str(best_failed.get("canonical_rejection_reason") or "CREDIT_TOO_LOW") if best_failed else "LIQUIDITY_BAD",
        "best_candidate": best_valid,
        "best_failed_candidate": best_failed,
        "candidate_evaluations": candidate_evaluations,
    }

    if best_valid is None:
        return None, ["Short strangle: no candidate met credit and liquidity thresholds."], report

    short_call_q = next(q for q in call_quotes if q.strike == best_valid["short_call_strike"])
    short_put_q = next(q for q in put_quotes if q.strike == best_valid["short_put_strike"])
    total_credit = float(best_valid["total_credit"])
    width_points = float(best_valid["width_points"])
    rationale = [
        f"Short strangle: sell {short_call_q.strike} call + sell {short_put_q.strike} put.",
        f"Total credit {total_credit:.2f} pts, width {width_points:.0f} pts, avg IV {avg_chain_iv:.1f}%.",
        f"Deltas: call {best_valid['call_delta']:.3f}, put {best_valid['put_delta']:.3f}. Stop at 2x credit.",
    ]
    structure = TradeStructure(
        strategy=StrategyType.SHORT_STRANGLE,
        legs=[
            StrategyLeg(action="SELL", option_type=OptionType.CALL, strike=short_call_q.strike, quote=short_call_q),
            StrategyLeg(action="SELL", option_type=OptionType.PUT, strike=short_put_q.strike, quote=short_put_q),
        ],
        credit_points=total_credit,
        width_points=width_points,
        call_width_points=0.0,
        put_width_points=0.0,
        margin_estimate_per_lot=snapshot.option_chain.margin_estimate_per_lot,
        rationale=rationale,
        metadata={
            "selection_mode": "STRANGLE",
            "short_call_strike": short_call_q.strike,
            "short_put_strike": short_put_q.strike,
            "avg_chain_iv": avg_chain_iv,
            "monetization_score": best_valid["monetization_score"],
        },
    )
    return structure, rationale, report


def _select_straddle_candidates(
    snapshot: MarketSnapshot,
    regime_state: RegimeState,
    params: AdaptiveParameters,
    *,
    setup_quality_score: float,
    playbook_tier: str,
) -> tuple[TradeStructure | None, list[str], dict[str, object]]:
    """Select a short straddle: sell ATM call + sell ATM put.

    Fires on extreme-IV balanced range days where selling both ATM options
    captures maximum premium. Risk is managed via 2x-credit stop (same as strangle).
    """
    spot = snapshot.option_chain.spot
    avg_chain_iv = float(regime_state.metadata.get("avg_chain_iv") or 0.0)
    all_quotes = snapshot.option_chain.quotes

    _empty_report: dict[str, object] = {
        "strategy": StrategyType.SHORT_STRADDLE.value,
        "setup_quality_score": round(setup_quality_score, 4),
        "monetization_score": 0.0,
        "final_trade_score": 0.0,
        "passed_spread_construction": False,
        "passed_liquidity": False,
        "passed_credit_width": False,
        "passed_delta_band": False,
        "passed_anchor_distance": True,
        "canonical_rejection_reason": "LIQUIDITY_BAD",
        "best_candidate": None,
        "best_failed_candidate": None,
        "candidate_evaluations": [],
    }

    call_quotes = [q for q in all_quotes if q.option_type == OptionType.CALL and q.strike >= spot]
    put_quotes = [q for q in all_quotes if q.option_type == OptionType.PUT and q.strike <= spot]
    if not call_quotes or not put_quotes:
        return None, ["Short straddle: no call or put quotes near ATM."], _empty_report

    # Find nearest ATM call and put (closest strike to spot on each side)
    atm_call = min(call_quotes, key=lambda q: abs(q.strike - spot))
    atm_put = min(put_quotes, key=lambda q: abs(q.strike - spot))

    call_credit = atm_call.mid_price or atm_call.ltp or 0.0
    put_credit = atm_put.mid_price or atm_put.ltp or 0.0
    total_credit = call_credit + put_credit
    min_total_credit = max(120.0, spot * 0.005)  # straddle needs at least 0.5% of spot

    avg_spread_ratio = ((atm_call.spread_ratio or 0.0) + (atm_put.spread_ratio or 0.0)) / 2.0
    passed_liquidity = avg_spread_ratio <= params.liquidity_spread_ratio_cap
    passed_credit = total_credit >= min_total_credit

    if not passed_liquidity or not passed_credit:
        reason = "LIQUIDITY_BAD" if not passed_liquidity else "CREDIT_TOO_LOW"
        failed_entry: dict[str, object] = {
            "short_call_strike": atm_call.strike,
            "short_put_strike": atm_put.strike,
            "call_credit": round(call_credit, 4),
            "put_credit": round(put_credit, 4),
            "total_credit": round(total_credit, 4),
            "avg_spread_ratio": round(avg_spread_ratio, 4),
            "canonical_rejection_reason": reason,
        }
        _empty_report["canonical_rejection_reason"] = reason
        _empty_report["best_failed_candidate"] = failed_entry
        return None, [f"Short straddle: {reason} (credit={total_credit:.1f}, spread_ratio={avg_spread_ratio:.3f})."], _empty_report

    monetization_score = round(
        (total_credit / max(spot * 0.01, 1.0)) * 10.0
        + max(0.0, 1.0 - avg_spread_ratio / max(params.liquidity_spread_ratio_cap, 0.01)) * 0.5,
        4,
    )
    best_valid: dict[str, object] = {
        "short_call_strike": atm_call.strike,
        "short_put_strike": atm_put.strike,
        "call_credit": round(call_credit, 4),
        "put_credit": round(put_credit, 4),
        "total_credit": round(total_credit, 4),
        "avg_spread_ratio": round(avg_spread_ratio, 4),
        "monetization_score": monetization_score,
        "canonical_rejection_reason": "NONE",
    }
    report: dict[str, object] = {
        "strategy": StrategyType.SHORT_STRADDLE.value,
        "setup_quality_score": round(setup_quality_score, 4),
        "monetization_score": monetization_score,
        "final_trade_score": round(setup_quality_score + monetization_score, 4),
        "passed_spread_construction": True,
        "passed_liquidity": True,
        "passed_credit_width": True,
        "passed_delta_band": True,
        "passed_anchor_distance": True,
        "canonical_rejection_reason": "NONE",
        "best_candidate": best_valid,
        "best_failed_candidate": None,
        "candidate_evaluations": [best_valid],
    }
    rationale = [
        f"Short straddle: sell {atm_call.strike} call + sell {atm_put.strike} put (ATM).",
        f"Total credit {total_credit:.2f} pts, avg IV {avg_chain_iv:.1f}%. Stop at 2x credit.",
        f"ATM strikes chosen for maximum premium on extreme-IV balanced session.",
    ]
    structure = TradeStructure(
        strategy=StrategyType.SHORT_STRADDLE,
        legs=[
            StrategyLeg(action="SELL", option_type=OptionType.CALL, strike=atm_call.strike, quote=atm_call),
            StrategyLeg(action="SELL", option_type=OptionType.PUT, strike=atm_put.strike, quote=atm_put),
        ],
        credit_points=total_credit,
        width_points=atm_call.strike - atm_put.strike,
        call_width_points=0.0,
        put_width_points=0.0,
        margin_estimate_per_lot=snapshot.option_chain.margin_estimate_per_lot,
        rationale=rationale,
        metadata={
            "selection_mode": "STRADDLE",
            "short_call_strike": atm_call.strike,
            "short_put_strike": atm_put.strike,
            "avg_chain_iv": avg_chain_iv,
            "monetization_score": monetization_score,
        },
    )
    return structure, rationale, report


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
