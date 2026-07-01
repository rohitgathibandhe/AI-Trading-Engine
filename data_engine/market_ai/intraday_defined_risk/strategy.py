from __future__ import annotations

from datetime import time

from .data_models import RegimeLabel, RegimeState, StrategyType


OPEN_DRIVE_BULLISH_START = time(9, 45)
OPEN_DRIVE_BULLISH_END = time(11, 15)
GAP_UP_BULLISH_START = time(9, 45)
GAP_UP_BULLISH_END = time(11, 30)
GAP_DOWN_BULLISH_RECOVERY_START = time(10, 15)
GAP_DOWN_BULLISH_RECOVERY_END = time(13, 0)
SIDEWAYS_BULLISH_RECLAIM_START = time(11, 30)
SIDEWAYS_BULLISH_RECLAIM_END = time(13, 30)
EARLY_BALANCE_BULLISH_START = time(10, 10)
EARLY_BALANCE_BULLISH_END = time(11, 15)
HIGH_CONFLUENCE_BULLISH_START = time(10, 0)
HIGH_CONFLUENCE_BULLISH_END = time(11, 15)
AFTERNOON_TREND_BULLISH_START = time(13, 0)
AFTERNOON_TREND_BULLISH_END = time(14, 0)
BEAR_CALL_START = time(13, 0)
EARLY_BEAR_CALL_START = time(10, 30)
EARLY_BALANCE_BEARISH_START = time(10, 10)
EARLY_BALANCE_BEARISH_END = time(11, 15)
HIGH_CONFLUENCE_BEARISH_START = time(10, 0)
HIGH_CONFLUENCE_BEARISH_END = time(11, 15)
SIDEWAYS_BEARISH_REJECTION_START = time(11, 30)
SIDEWAYS_BEARISH_REJECTION_END = time(14, 30)
GAP_BEARISH_START = time(9, 45)
GAP_BEARISH_END = time(13, 0)
GAP_DOWN_BEARISH_CONTINUATION_END = time(12, 0)
GAP_UP_BEARISH_FAILURE_END = time(11, 15)
RANGE_CONDOR_START = time(11, 30)
RANGE_CONDOR_END = time(13, 30)
RANGE_STRANGLE_START = time(11, 30)
RANGE_STRANGLE_END = time(13, 30)
# Wednesday = Day 1 of new weekly series (post-Tuesday expiry).
# IC can start earlier on Wednesday — morning range is well-defined after the
# previous day's expiry flush, and 7DTE options carry the most weekly premium.
RANGE_CONDOR_POST_EXPIRY_START = time(10, 45)   # 45 min earlier than normal
RANGE_CONDOR_POST_EXPIRY_END = time(13, 30)      # same close — no extra risk
OPEN_DRIVE_BULLISH_MIN_CONFIDENCE = 0.64
GAP_UP_BULLISH_MIN_CONFIDENCE = 0.68
GAP_DOWN_BULLISH_RECOVERY_MIN_CONFIDENCE = 0.70
SIDEWAYS_BULLISH_RECLAIM_MIN_CONFIDENCE = 0.66
EARLY_BALANCE_BULLISH_MIN_CONFIDENCE = 0.72
HIGH_CONFLUENCE_BULLISH_MIN_CONFIDENCE = 0.88
AFTERNOON_TREND_BULLISH_MIN_CONFIDENCE = 0.72
RANGE_CONDOR_MIN_CONFIDENCE = 0.55
CONFIDENCE_EPSILON = 1e-6
BEAR_CALL_MIN_CONFIDENCE = 0.62
REGIME_TRADABILITY_TRADABLE = "TRADABLE"
REGIME_TRADABILITY_SPREAD = REGIME_TRADABILITY_TRADABLE
REGIME_TRADABILITY_LOW_EDGE = "LOW_EDGE"
REGIME_TRADABILITY_NOT_TRADABLE = "NOT_TRADABLE"

BEARISH_CONTEXT_FAILURE_TYPES = {
    "FAILED_BREAKOUT",
    "FAILED_RECLAIM",
    "FAILED_BOUNCE",
    "BEARISH_REJECTION_TRANSITION",
    "ACCEPTED_BREAKDOWN",
}
ENABLE_BEAR_CALL_ACTIVE = False
ENABLE_EARLY_BALANCE_BEARISH_ACTIVE = True
ENABLE_SIDEWAYS_BEARISH_REJECTION_ACTIVE = True
ENABLE_GAP_UP_BEARISH_FAILURE_ACTIVE = True
ENABLE_GAP_DOWN_BEARISH_CONTINUATION_ACTIVE = True
ENABLE_HIGH_CONFLUENCE_BEARISH_ACTIVE = True
ENABLE_OPEN_DRIVE_BULLISH_ACTIVE = False
ENABLE_GAP_UP_BULLISH_ACTIVE = False
ENABLE_GAP_DOWN_BULLISH_RECOVERY_ACTIVE = False
ENABLE_SIDEWAYS_BULLISH_RECLAIM_ACTIVE = True
ENABLE_EARLY_BALANCE_BULLISH_ACTIVE = True
ENABLE_HIGH_CONFLUENCE_BULLISH_ACTIVE = True
ENABLE_AFTERNOON_TREND_BULLISH_ACTIVE = False
ENABLE_RANGE_CONDOR_ACTIVE = True

# Short straddle: extreme IV + confirmed range day → sell ATM call + put.
# Only fires when IV is very elevated and the session is genuinely balanced.
SHORT_STRADDLE_IV_FLOOR = 28.0
SHORT_STRADDLE_RANGE_BALANCE_MIN = 3.0
SHORT_STRADDLE_START = time(11, 30)
SHORT_STRADDLE_END = time(13, 30)

TIER_A_PLAYBOOKS = {
    # Core bearish spread playbooks — live-validated
    "SIDEWAYS_TO_BEARISH_REJECTION",
    "GAP_DOWN_BEARISH_CONTINUATION",
    "GAP_UP_BEARISH_FAILURE",
    "RANGE_BALANCED_CONDOR",
    "EARLY_BALANCE_BEARISH_FAILED_RECLAIM",
    "HIGH_CONFLUENCE_BEARISH_CONTINUATION",
    "BEARISH_FAILED_RECLAIM",
    "BEARISH_CONTINUATION",
    # Live-enabled bullish spread playbooks — promoted from Tier B
    "GAP_DOWN_BULLISH_RECOVERY",
    "SIDEWAYS_TO_BULLISH_RECLAIM",
    "EARLY_BALANCE_BULLISH_RECLAIM",
    "HIGH_CONFLUENCE_BULLISH_CONTINUATION",
    "AFTERNOON_TREND_HOLD_BULLISH",
}

TIER_B_PLAYBOOKS = {
    # Playbooks still in research / not yet enabled live
    "OPEN_DRIVE_BULLISH",
    "GAP_UP_BULLISH_CONTINUATION",
}

STABLE_SPREAD_PLAYBOOKS = {
    "GAP_DOWN_BEARISH_CONTINUATION",
    "SIDEWAYS_TO_BEARISH_REJECTION",
    "GAP_UP_BEARISH_FAILURE",
    "EARLY_BALANCE_BEARISH_FAILED_RECLAIM",
    "HIGH_CONFLUENCE_BEARISH_CONTINUATION",
    "BEARISH_FAILED_RECLAIM",
    "BEARISH_CONTINUATION",
}

NOT_TRADABLE_EXPANSION_PLAYBOOKS = {
    # These bullish playbooks need directional options, not spreads
    "OPEN_DRIVE_BULLISH",
    "GAP_UP_BULLISH_CONTINUATION",
}


def playbook_tier(playbook: str) -> str:
    if playbook in TIER_A_PLAYBOOKS:
        return "A"
    if playbook in TIER_B_PLAYBOOKS:
        return "B"
    return "C"


def classify_regime_tradability(regime_state: RegimeState) -> tuple[str, str]:
    metadata = regime_state.metadata
    explicit_class = str(metadata.get("tradability_class") or "")
    explicit_reason = str(metadata.get("tradability_reason") or "")
    if explicit_class in {REGIME_TRADABILITY_TRADABLE, REGIME_TRADABILITY_LOW_EDGE, REGIME_TRADABILITY_NOT_TRADABLE}:
        return explicit_class, explicit_reason or f"Context layer classified the setup as {explicit_class}."

    playbook = str(metadata.get("playbook") or "UNKNOWN")
    bullish_shadow_subtype = str(metadata.get("bullish_shadow_subtype") or "")
    bearish_subtype = str(metadata.get("bearish_subtype") or "")
    tier = playbook_tier(playbook)

    if bullish_shadow_subtype == "LATE_SESSION_BULLISH_RECLAIM":
        return (
            REGIME_TRADABILITY_LOW_EDGE,
            "Late-session bullish reclaim shows only a small edge under current spread monetization and must remain low-edge.",
        )
    if bearish_subtype == "GAP_DOWN_FAILED_BOUNCE":
        return (
            REGIME_TRADABILITY_TRADABLE,
            "Gap-down failed bounce is a validated early-session bearish spread subtype.",
        )
    if bearish_subtype == "BEARISH_REJECTION_TRANSITION":
        return (
            REGIME_TRADABILITY_TRADABLE,
            "Transition-state bearish rejection is a validated live bearish subtype when rejection holds below the trigger zone.",
        )
    if playbook in STABLE_SPREAD_PLAYBOOKS:
        return (
            REGIME_TRADABILITY_TRADABLE,
            f"{playbook} is a validated spread-monetizable playbook.",
        )
    if playbook in NOT_TRADABLE_EXPANSION_PLAYBOOKS:
        return (
            REGIME_TRADABILITY_NOT_TRADABLE,
            f"{playbook} is structurally not monetizable with the current spread framework.",
        )
    if tier == "A":
        return (
            REGIME_TRADABILITY_TRADABLE,
            f"{playbook} remains live-eligible under the validated Tier A stack.",
        )
    if regime_state.regime == RegimeLabel.UP_TREND:
        return (
            REGIME_TRADABILITY_NOT_TRADABLE,
            "Bullish regimes remain structurally unproven for current spread monetization.",
        )
    if tier == "B":
        return (
            REGIME_TRADABILITY_NOT_TRADABLE,
            f"{playbook} is still a research-only playbook without proven live spread edge.",
        )
    return (
        REGIME_TRADABILITY_LOW_EDGE,
        f"{playbook} does not yet have robust spread expectancy and should stay behind stricter deployment filters.",
    )


def low_edge_tradability_filter_reason(
    regime_state: RegimeState,
    params: AdaptiveParameters,
) -> str | None:
    tradability, _ = classify_regime_tradability(regime_state)
    if tradability != REGIME_TRADABILITY_LOW_EDGE:
        return None
    setup_quality_score = float(regime_state.metadata.get("setup_quality_score") or 0.0)
    bullish_trade_score = float(regime_state.metadata.get("bullish_trade_score") or 0.0)
    no_trade_score = float(regime_state.metadata.get("no_trade_score") or 0.0)
    required_confidence = required_confidence_for_playbook(str(regime_state.metadata.get("playbook") or "UNKNOWN")) + 0.03
    if setup_quality_score < params.strong_setup_quality_threshold:
        return (
            f"Low-edge regime requires setup quality >= {params.strong_setup_quality_threshold:.2f}; "
            f"current quality is {setup_quality_score:.2f}."
        )
    if bullish_trade_score <= no_trade_score + params.bullish_trade_margin:
        return (
            f"Low-edge regime requires bullish trade score > no-trade score by {params.bullish_trade_margin:.2f}; "
            f"current spread is {bullish_trade_score - no_trade_score:.2f}."
        )
    if regime_state.confidence + CONFIDENCE_EPSILON < required_confidence:
        return (
            f"Low-edge regime requires conviction >= {required_confidence:.2f}; "
            f"current conviction is {regime_state.confidence:.2f}."
        )
    return None


def _gap_down_failed_bounce_end_time(
    regime_state: RegimeState,
    experimental_policy: dict[str, object] | None,
) -> time:
    if not experimental_policy:
        return GAP_DOWN_BEARISH_CONTINUATION_END
    subtype = str(regime_state.metadata.get("bearish_subtype") or "")
    if subtype != "GAP_DOWN_FAILED_BOUNCE":
        return GAP_DOWN_BEARISH_CONTINUATION_END
    override = experimental_policy.get("gap_down_failed_bounce_end_time")
    if isinstance(override, time):
        return override
    if isinstance(override, str):
        return time.fromisoformat(override)
    return GAP_DOWN_BEARISH_CONTINUATION_END


def required_confidence_for_playbook(playbook: str) -> float:
    if playbook == "OPEN_DRIVE_BULLISH":
        return OPEN_DRIVE_BULLISH_MIN_CONFIDENCE
    if playbook == "HIGH_CONFLUENCE_BULLISH_CONTINUATION":
        return HIGH_CONFLUENCE_BULLISH_MIN_CONFIDENCE
    if playbook == "GAP_UP_BULLISH_CONTINUATION":
        return GAP_UP_BULLISH_MIN_CONFIDENCE
    if playbook == "GAP_DOWN_BULLISH_RECOVERY":
        return GAP_DOWN_BULLISH_RECOVERY_MIN_CONFIDENCE
    if playbook == "EARLY_BALANCE_BULLISH_RECLAIM":
        return EARLY_BALANCE_BULLISH_MIN_CONFIDENCE
    if playbook == "AFTERNOON_TREND_HOLD_BULLISH":
        return AFTERNOON_TREND_BULLISH_MIN_CONFIDENCE
    if playbook == "RANGE_BALANCED_CONDOR":
        return RANGE_CONDOR_MIN_CONFIDENCE
    if playbook in {
        "EARLY_BALANCE_BEARISH_FAILED_RECLAIM",
        "SIDEWAYS_TO_BEARISH_REJECTION",
        "GAP_UP_BEARISH_FAILURE",
        "GAP_DOWN_BEARISH_CONTINUATION",
        "HIGH_CONFLUENCE_BEARISH_CONTINUATION",
        "BEARISH_FAILED_RECLAIM",
        "BEARISH_CONTINUATION",
    }:
        return BEAR_CALL_MIN_CONFIDENCE
    return SIDEWAYS_BULLISH_RECLAIM_MIN_CONFIDENCE


def playbook_time_window(
    regime_state: RegimeState,
    *,
    experimental_policy: dict[str, object] | None = None,
) -> tuple[time | None, time | None]:
    playbook = str(regime_state.metadata.get("playbook") or "UNKNOWN")
    bullish_setup = regime_state.metadata.get("bullish_setup")
    if playbook == "OPEN_DRIVE_BULLISH":
        return OPEN_DRIVE_BULLISH_START, OPEN_DRIVE_BULLISH_END
    if playbook == "HIGH_CONFLUENCE_BULLISH_CONTINUATION":
        return HIGH_CONFLUENCE_BULLISH_START, HIGH_CONFLUENCE_BULLISH_END
    if playbook == "GAP_UP_BULLISH_CONTINUATION":
        return GAP_UP_BULLISH_START, GAP_UP_BULLISH_END
    if playbook == "GAP_DOWN_BULLISH_RECOVERY":
        return GAP_DOWN_BULLISH_RECOVERY_START, GAP_DOWN_BULLISH_RECOVERY_END
    if playbook == "EARLY_BALANCE_BULLISH_RECLAIM":
        return EARLY_BALANCE_BULLISH_START, EARLY_BALANCE_BULLISH_END
    if playbook == "AFTERNOON_TREND_HOLD_BULLISH":
        return AFTERNOON_TREND_BULLISH_START, AFTERNOON_TREND_BULLISH_END
    if playbook == "SIDEWAYS_TO_BULLISH_RECLAIM":
        start = (
            time(10, 30)
            if bullish_setup == "VWAP_HOLD_HIGHER_LOW" and bool(regime_state.metadata.get("early_sideways_bullish_ready"))
            else SIDEWAYS_BULLISH_RECLAIM_START
        )
        return start, SIDEWAYS_BULLISH_RECLAIM_END
    if playbook == "EARLY_BALANCE_BEARISH_FAILED_RECLAIM":
        return EARLY_BALANCE_BEARISH_START, EARLY_BALANCE_BEARISH_END
    if playbook == "SIDEWAYS_TO_BEARISH_REJECTION":
        return SIDEWAYS_BEARISH_REJECTION_START, SIDEWAYS_BEARISH_REJECTION_END
    if playbook == "GAP_UP_BEARISH_FAILURE":
        return GAP_BEARISH_START, GAP_UP_BEARISH_FAILURE_END
    if playbook == "GAP_DOWN_BEARISH_CONTINUATION":
        return GAP_BEARISH_START, _gap_down_failed_bounce_end_time(regime_state, experimental_policy)
    if playbook == "HIGH_CONFLUENCE_BEARISH_CONTINUATION":
        return HIGH_CONFLUENCE_BEARISH_START, HIGH_CONFLUENCE_BEARISH_END
    if playbook == "RANGE_BALANCED_CONDOR":
        return RANGE_CONDOR_START, RANGE_CONDOR_END
    return None, None


def is_tradable_bearish_rejection(regime_state: RegimeState) -> bool:
    metadata = regime_state.metadata
    candle_pattern = str(metadata.get("bearish_candle_pattern") or "")
    close_location = float(metadata.get("bearish_close_location") or 1.0)
    rejection_quality_score = float(metadata.get("bearish_candle_quality_score") or 0.0)

    # BEARISH_EXPANSION candles have near-zero upper wicks by definition — the body
    # spans from open (near the high) to close (near the low), giving strong downside
    # momentum. The upper_wick_fraction gate is meaningless for them; use close
    # location and quality score instead.
    if candle_pattern == "BEARISH_EXPANSION":
        return bool(close_location <= 0.45 and rejection_quality_score >= 1.5)

    upper_wick_fraction = float(metadata.get("bearish_upper_wick_fraction") or 0.0)
    return bool(
        upper_wick_fraction >= 0.50
        and close_location <= 0.45
        and rejection_quality_score >= 0.55
    )


def select_strategy(
    regime_state: RegimeState,
    now_time: time,
    *,
    allowed_playbook_tiers: tuple[str, ...] = ("A", "B"),
    experimental_policy: dict[str, object] | None = None,
) -> tuple[StrategyType, list[str]]:
    reasons = list(regime_state.reasons)
    metadata = regime_state.metadata
    context_layer_active = any(
        key in metadata
        for key in (
            "tradability_class",
            "bearish_trade_score",
            "bullish_trade_score",
            "no_trade_score",
            "failure_type",
        )
    )
    day_archetype = str(regime_state.metadata.get("day_archetype") or "UNCLASSIFIED")
    market_state = str(metadata.get("market_state") or "TRUE_RANGE")
    tradability, tradability_reason = classify_regime_tradability(regime_state)
    failure_type = str(metadata.get("failure_type") or "NONE")
    bearish_trade_score = float(metadata.get("bearish_trade_score") or 0.0)
    bullish_trade_score = float(metadata.get("bullish_trade_score") or 0.0)
    no_trade_score = float(metadata.get("no_trade_score") or 0.0)
    bearish_failure_score = float(metadata.get("bearish_failure_score") or 0.0)
    bullish_failure_score = float(metadata.get("bullish_failure_score") or 0.0)
    bearish_location_live_score = float(metadata.get("bearish_location_live_score") or 0.0)
    open_space_up = float(metadata.get("open_space_up") or 0.0)
    overhead_call_pressure_score = float(metadata.get("overhead_call_pressure_score") or 0.0)
    bearish_margin = float(metadata.get("trade_score_margin_bearish") or 1.5)
    bullish_margin = float(metadata.get("trade_score_margin_bullish") or 1.8)
    bearish_threshold = float(metadata.get("bearish_trade_score_threshold") or 6.5)
    bullish_threshold = float(metadata.get("bullish_trade_score_threshold") or 7.0)
    if now_time < time(9, 15):
        reasons.append("Pre-market entry is not allowed.")
        return StrategyType.NO_TRADE, reasons
    if context_layer_active and now_time >= time(14, 30):
        reasons.append("No new entries are allowed after 14:30 IST under the context-aware late-session penalty.")
        return StrategyType.NO_TRADE, reasons
    if (
        bool(metadata.get("enable_market_state_gating"))
        and market_state == "TRUE_RANGE"
        and str(metadata.get("playbook") or "UNKNOWN") != "RANGE_BALANCED_CONDOR"
    ):
        reasons.append("Market state engine classifies the session as TRUE_RANGE; directional deployment is blocked outside strict condor criteria.")
        return StrategyType.NO_TRADE, reasons
    if context_layer_active and tradability == REGIME_TRADABILITY_NOT_TRADABLE and str(metadata.get("playbook") or "UNKNOWN") != "RANGE_BALANCED_CONDOR":
        reasons.append(tradability_reason)
        return StrategyType.NO_TRADE, reasons

    # Multi-timeframe daily bias gate: block strongly counter-trend intraday trades
    daily_trend = str(metadata.get("daily_trend") or "NEUTRAL")
    daily_bias_score = int(metadata.get("daily_bias_score") or 0)
    if regime_state.regime == RegimeLabel.DOWN_TREND and daily_trend == "STRONGLY_BULLISH":
        reasons.append(
            f"Daily trend is STRONGLY_BULLISH (bias_score={daily_bias_score}); short-side entry is blocked against the higher-timeframe trend."
        )
        return StrategyType.NO_TRADE, reasons
    if regime_state.regime == RegimeLabel.UP_TREND and daily_trend == "STRONGLY_BEARISH":
        reasons.append(
            f"Daily trend is STRONGLY_BEARISH (bias_score={daily_bias_score}); long-side entry is blocked against the higher-timeframe trend."
        )
        return StrategyType.NO_TRADE, reasons

    if regime_state.regime == RegimeLabel.DOWN_TREND:
        bearish_setup = metadata.get("bearish_setup")
        bearish_ready = bool(metadata.get("bearish_entry_ready"))
        playbook = str(metadata.get("playbook") or "UNKNOWN")
        if not bearish_ready:
            reasons.append(
                "Bearish downtrend detected, but price has not yet printed a valid bearish pullback rejection, failed reclaim, or shallow continuation setup."
            )
            return StrategyType.NO_TRADE, reasons
        # OI conflict gate: if institutions are positioned bullish, require meaningfully higher score.
        # Three signals: smart_money_bias (OI flow), wall_migration_bias (call/put wall drift), PCR.
        # 2+ signals opposing → add a surcharge to the score threshold.
        # All 3 opposing (hard conflict: smart money BULLISH + PCR < 0.70) → hard block.
        _oi_threshold_surcharge = 0.0
        if context_layer_active:
            smart_money_bias = str(metadata.get("smart_money_bias") or "NEUTRAL")
            wall_migration_bias = str(metadata.get("wall_migration_bias") or "NEUTRAL")
            pcr_val = metadata.get("pcr_by_oi")
            _pcr_conflict = pcr_val is not None and float(pcr_val) < 0.70
            oi_conflict_count = sum([
                smart_money_bias == "BULLISH",
                wall_migration_bias == "BULLISH",
                _pcr_conflict,
            ])
            if oi_conflict_count >= 2:
                # Hard block: institutional flow + PCR both say bullish — don't fight them
                if smart_money_bias == "BULLISH" and _pcr_conflict:
                    reasons.append(
                        f"Hard OI conflict: smart_money=BULLISH and PCR={pcr_val:.2f} (<0.70). "
                        "Institutions are positioned bullish; bear spread entry blocked."
                    )
                    return StrategyType.NO_TRADE, reasons
                # Soft block: 2/3 signals oppose → raise score threshold by 0.5
                _oi_threshold_surcharge = 0.5
        if context_layer_active and failure_type not in BEARISH_CONTEXT_FAILURE_TYPES and bearish_trade_score < 5.5:
            reasons.append(
                f"Bearish routing requires a recognized failure/acceptance context; current failure type is {failure_type}."
            )
            return StrategyType.NO_TRADE, reasons
        if context_layer_active and bearish_failure_score < 1.2:
            reasons.append(
                f"Bearish failure score {bearish_failure_score:.2f} is below the required 1.20."
            )
            return StrategyType.NO_TRADE, reasons
        if context_layer_active and bearish_location_live_score < 0.8:
            reasons.append(
                f"Bearish location score {bearish_location_live_score:.2f} is below the required 0.80."
            )
            return StrategyType.NO_TRADE, reasons
        _effective_bearish_threshold = bearish_threshold + _oi_threshold_surcharge
        if context_layer_active and bearish_trade_score < _effective_bearish_threshold:
            _surcharge_note = f" (+{_oi_threshold_surcharge:.1f} OI-conflict surcharge)" if _oi_threshold_surcharge > 0 else ""
            reasons.append(
                f"Bearish trade score {bearish_trade_score:.2f} is below the required {_effective_bearish_threshold:.2f}{_surcharge_note}."
            )
            return StrategyType.NO_TRADE, reasons
        if context_layer_active and bearish_trade_score <= no_trade_score + bearish_margin:
            reasons.append(
                f"Bearish trade score {bearish_trade_score:.2f} does not exceed no-trade score {no_trade_score:.2f} by the required {bearish_margin:.2f}."
            )
            return StrategyType.NO_TRADE, reasons
        required_bear_confidence = BEAR_CALL_MIN_CONFIDENCE + (0.05 if now_time >= time(14, 0) else 0.0)
        if regime_state.confidence + CONFIDENCE_EPSILON < required_bear_confidence:
            reasons.append(
                f"Bearish setup detected, but conviction {regime_state.confidence:.2f} is below the required {required_bear_confidence:.2f}."
            )
            return StrategyType.NO_TRADE, reasons
        if day_archetype not in {"OPEN_DRIVE_BEARISH", "EARLY_BALANCE_TO_BEARISH", "SIDEWAYS_TO_BEARISH", "GAP_UP_FAILURE", "GAP_DOWN_CONTINUATION", "HIGH_CONFLUENCE_BEARISH", "TREND_BEARISH"}:
            reasons.append(
                f"Bearish regime is generic {day_archetype}; only gap/open-drive/sideways/trend-bearish archetypes are eligible for downside deployment."
            )
            return StrategyType.NO_TRADE, reasons
        if day_archetype == "OPEN_DRIVE_BEARISH" and bearish_setup != "TIGHT_BREAKDOWN":
            reasons.append("Open-drive bearish sessions require a tight breakdown setup, not a generic continuation.")
            return StrategyType.NO_TRADE, reasons
        if day_archetype == "EARLY_BALANCE_TO_BEARISH" and (bearish_setup != "FAILED_RECLAIM" or playbook != "EARLY_BALANCE_BEARISH_FAILED_RECLAIM"):
            reasons.append("Early-balance bearish sessions require the dedicated failed-reclaim playbook before deployment.")
            return StrategyType.NO_TRADE, reasons
        if day_archetype == "SIDEWAYS_TO_BEARISH" and (bearish_setup != "PULLBACK_REJECTION" or playbook != "SIDEWAYS_TO_BEARISH_REJECTION"):
            reasons.append("Sideways-to-bearish sessions require the dedicated sideways bearish rejection playbook before deployment.")
            return StrategyType.NO_TRADE, reasons
        if playbook == "SIDEWAYS_TO_BEARISH_REJECTION" and not is_tradable_bearish_rejection(regime_state):
            reasons.append(
                "Sideways-to-bearish rejection requires a high-quality candle: either a BEARISH_EXPANSION (close in lower 45%, quality >= 1.5) "
                "or a rejection wick candle (upper wick >= 50%, close in lower 45%, quality >= 0.55)."
            )
            return StrategyType.NO_TRADE, reasons
        if day_archetype == "GAP_UP_FAILURE" and (bearish_setup != "GAP_FAILURE" or playbook != "GAP_UP_BEARISH_FAILURE"):
            reasons.append("Gap-up bearish sessions require the dedicated gap-failure playbook before deployment.")
            return StrategyType.NO_TRADE, reasons
        if day_archetype == "GAP_DOWN_CONTINUATION" and (bearish_setup != "GAP_CONTINUATION" or playbook != "GAP_DOWN_BEARISH_CONTINUATION"):
            reasons.append("Gap-down bearish sessions require the dedicated gap-continuation playbook before deployment.")
            return StrategyType.NO_TRADE, reasons
        if day_archetype == "HIGH_CONFLUENCE_BEARISH" and (bearish_setup != "HIGH_CONFLUENCE_CONTINUATION" or playbook != "HIGH_CONFLUENCE_BEARISH_CONTINUATION"):
            reasons.append("High-confluence bearish sessions require the dedicated continuation playbook before deployment.")
            return StrategyType.NO_TRADE, reasons
        tier = playbook_tier(playbook)
        if tier not in allowed_playbook_tiers:
            reasons.append(
                f"Playbook {playbook} is Tier {tier}; current benchmark mode allows only tiers {', '.join(allowed_playbook_tiers)}."
            )
            return StrategyType.NO_TRADE, reasons
        if day_archetype == "OPEN_DRIVE_BEARISH":
            required_bear_start = EARLY_BEAR_CALL_START
            required_bear_end = time(14, 0)
        elif day_archetype == "EARLY_BALANCE_TO_BEARISH":
            required_bear_start = EARLY_BALANCE_BEARISH_START
            required_bear_end = EARLY_BALANCE_BEARISH_END
        elif day_archetype == "SIDEWAYS_TO_BEARISH":
            required_bear_start = SIDEWAYS_BEARISH_REJECTION_START
            required_bear_end = SIDEWAYS_BEARISH_REJECTION_END
        elif day_archetype == "GAP_UP_FAILURE":
            required_bear_start = GAP_BEARISH_START
            required_bear_end = GAP_UP_BEARISH_FAILURE_END
        elif day_archetype == "HIGH_CONFLUENCE_BEARISH":
            required_bear_start = HIGH_CONFLUENCE_BEARISH_START
            required_bear_end = HIGH_CONFLUENCE_BEARISH_END
        elif day_archetype == "GAP_DOWN_CONTINUATION":
            required_bear_start = GAP_BEARISH_START
            required_bear_end = _gap_down_failed_bounce_end_time(regime_state, experimental_policy)
        else:
            required_bear_start = GAP_BEARISH_START
            required_bear_end = GAP_BEARISH_END
        if now_time > required_bear_end:
            reasons.append(
                f"Bear Call Credit Spreads for {day_archetype} are only allowed until {required_bear_end.strftime('%H:%M')} IST."
            )
            return StrategyType.NO_TRADE, reasons
        if now_time >= required_bear_start:
            reasons.append(
                f"15m TrendDown + bearish {(bearish_setup or 'TREND_FOLLOW')} confirmation on {day_archetype} -> Bear Call Credit Spread."
            )
            return StrategyType.BEAR_CALL_CREDIT_SPREAD, reasons
        reasons.append(
            f"Bear Call Credit Spreads require time >= {required_bear_start.strftime('%H:%M')} IST for the current {day_archetype} archetype."
        )
        return StrategyType.NO_TRADE, reasons

    if regime_state.regime == RegimeLabel.UP_TREND:
        bullish_setup = regime_state.metadata.get("bullish_setup")
        playbook = str(regime_state.metadata.get("playbook") or "UNKNOWN")
        if not regime_state.metadata.get("bullish_entry_ready"):
            reasons.append(
                "Bullish uptrend detected, but price has not yet printed a valid pullback reclaim or shallow continuation setup."
            )
            return StrategyType.NO_TRADE, reasons
        if context_layer_active and tradability == REGIME_TRADABILITY_NOT_TRADABLE:
            reasons.append(tradability_reason)
            return StrategyType.NO_TRADE, reasons
        if context_layer_active and bullish_failure_score < 1.0 and failure_type not in {"ACCEPTED_BREAKOUT", "NONE"}:
            reasons.append(
                f"Bullish context is too conflicted; failure score is {bullish_failure_score:.2f} with failure type {failure_type}."
            )
            return StrategyType.NO_TRADE, reasons
        if context_layer_active and open_space_up < 120.0:
            reasons.append(
                f"Bullish open space is too small at {open_space_up:.2f} points."
            )
            return StrategyType.NO_TRADE, reasons
        if context_layer_active and overhead_call_pressure_score >= 1.5:
            reasons.append(
                f"Overhead call-wall pressure {overhead_call_pressure_score:.2f} is too high for bullish deployment."
            )
            return StrategyType.NO_TRADE, reasons
        if context_layer_active and bullish_trade_score < bullish_threshold:
            reasons.append(
                f"Bullish trade score {bullish_trade_score:.2f} is below the required {bullish_threshold:.2f}."
            )
            return StrategyType.NO_TRADE, reasons
        if context_layer_active and bullish_trade_score <= no_trade_score + bullish_margin:
            reasons.append(
                f"Bullish trade score {bullish_trade_score:.2f} does not exceed no-trade score {no_trade_score:.2f} by the required {bullish_margin:.2f}."
            )
            return StrategyType.NO_TRADE, reasons
        if playbook == "OPEN_DRIVE_BULLISH":
            required_confidence = OPEN_DRIVE_BULLISH_MIN_CONFIDENCE
        elif playbook == "HIGH_CONFLUENCE_BULLISH_CONTINUATION":
            required_confidence = HIGH_CONFLUENCE_BULLISH_MIN_CONFIDENCE
        elif playbook == "GAP_UP_BULLISH_CONTINUATION":
            required_confidence = GAP_UP_BULLISH_MIN_CONFIDENCE
        elif playbook == "GAP_DOWN_BULLISH_RECOVERY":
            required_confidence = GAP_DOWN_BULLISH_RECOVERY_MIN_CONFIDENCE
        elif playbook == "EARLY_BALANCE_BULLISH_RECLAIM":
            required_confidence = EARLY_BALANCE_BULLISH_MIN_CONFIDENCE
        elif playbook == "AFTERNOON_TREND_HOLD_BULLISH":
            required_confidence = AFTERNOON_TREND_BULLISH_MIN_CONFIDENCE
        else:
            required_confidence = SIDEWAYS_BULLISH_RECLAIM_MIN_CONFIDENCE
        required_bull_confidence = required_confidence + (0.05 if now_time >= time(14, 0) else 0.0)
        if regime_state.confidence + CONFIDENCE_EPSILON < required_bull_confidence:
            reasons.append(
                f"Bullish setup detected, but conviction {regime_state.confidence:.2f} is below the required {required_bull_confidence:.2f}."
            )
            return StrategyType.NO_TRADE, reasons
        if playbook not in {"OPEN_DRIVE_BULLISH", "HIGH_CONFLUENCE_BULLISH_CONTINUATION", "SIDEWAYS_TO_BULLISH_RECLAIM", "EARLY_BALANCE_BULLISH_RECLAIM", "GAP_UP_BULLISH_CONTINUATION", "GAP_DOWN_BULLISH_RECOVERY", "AFTERNOON_TREND_HOLD_BULLISH"}:
            reasons.append(
                f"Bullish regime is generic {day_archetype} with playbook {playbook}; only dedicated gap/open-drive/sideways bullish playbooks are eligible for upside deployment."
            )
            return StrategyType.NO_TRADE, reasons
        if playbook == "HIGH_CONFLUENCE_BULLISH_CONTINUATION" and bullish_setup != "HIGH_CONFLUENCE_CONTINUATION":
            reasons.append("High-confluence bullish sessions require the dedicated continuation setup before deployment.")
            return StrategyType.NO_TRADE, reasons
        if playbook == "OPEN_DRIVE_BULLISH" and bullish_setup not in {"OPEN_DRIVE_CONTINUATION", "OPEN_DRIVE_RECLAIM"}:
            reasons.append("Open-drive bullish sessions require an open-drive continuation or reclaim setup before deployment.")
            return StrategyType.NO_TRADE, reasons
        if playbook == "GAP_UP_BULLISH_CONTINUATION" and bullish_setup != "GAP_CONTINUATION":
            reasons.append("Gap-up bullish sessions require the dedicated gap-continuation setup before deployment.")
            return StrategyType.NO_TRADE, reasons
        if playbook == "GAP_DOWN_BULLISH_RECOVERY" and bullish_setup != "GAP_RECOVERY":
            reasons.append("Gap-down bullish sessions require the dedicated gap-recovery setup before deployment.")
            return StrategyType.NO_TRADE, reasons
        if playbook == "AFTERNOON_TREND_HOLD_BULLISH" and bullish_setup != "AFTERNOON_TREND_HOLD":
            reasons.append("Afternoon bullish trend-hold sessions require the dedicated late-session trend-hold setup before deployment.")
            return StrategyType.NO_TRADE, reasons
        if playbook == "EARLY_BALANCE_BULLISH_RECLAIM" and bullish_setup != "EARLY_BALANCE_RECLAIM":
            reasons.append("Early balance bullish sessions require the dedicated early-balance reclaim setup before deployment.")
            return StrategyType.NO_TRADE, reasons
        if playbook == "SIDEWAYS_TO_BULLISH_RECLAIM" and bullish_setup not in {"PULLBACK_RECLAIM", "SHALLOW_CONTINUATION", "VWAP_HOLD_HIGHER_LOW"}:
            reasons.append("Sideways-to-bullish sessions require a pullback reclaim, VWAP hold higher-low, or shallow continuation setup before deployment.")
            return StrategyType.NO_TRADE, reasons
        tier = playbook_tier(playbook)
        if tier not in allowed_playbook_tiers:
            reasons.append(
                f"Playbook {playbook} is Tier {tier}; current benchmark mode allows only tiers {', '.join(allowed_playbook_tiers)}."
            )
            return StrategyType.NO_TRADE, reasons
        if playbook == "OPEN_DRIVE_BULLISH":
            required_bull_start = OPEN_DRIVE_BULLISH_START
        elif playbook == "HIGH_CONFLUENCE_BULLISH_CONTINUATION":
            required_bull_start = HIGH_CONFLUENCE_BULLISH_START
        elif playbook == "GAP_UP_BULLISH_CONTINUATION":
            required_bull_start = GAP_UP_BULLISH_START
        elif playbook == "GAP_DOWN_BULLISH_RECOVERY":
            required_bull_start = GAP_DOWN_BULLISH_RECOVERY_START
        elif playbook == "EARLY_BALANCE_BULLISH_RECLAIM":
            required_bull_start = EARLY_BALANCE_BULLISH_START
        elif playbook == "AFTERNOON_TREND_HOLD_BULLISH":
            required_bull_start = AFTERNOON_TREND_BULLISH_START
        else:
            required_bull_start = (
                time(10, 30)
                if bullish_setup == "VWAP_HOLD_HIGHER_LOW" and bool(regime_state.metadata.get("early_sideways_bullish_ready"))
                else SIDEWAYS_BULLISH_RECLAIM_START
            )
        if playbook == "OPEN_DRIVE_BULLISH":
            required_bull_end = OPEN_DRIVE_BULLISH_END
        elif playbook == "HIGH_CONFLUENCE_BULLISH_CONTINUATION":
            required_bull_end = HIGH_CONFLUENCE_BULLISH_END
        elif playbook == "GAP_UP_BULLISH_CONTINUATION":
            required_bull_end = GAP_UP_BULLISH_END
        elif playbook == "GAP_DOWN_BULLISH_RECOVERY":
            required_bull_end = GAP_DOWN_BULLISH_RECOVERY_END
        elif playbook == "EARLY_BALANCE_BULLISH_RECLAIM":
            required_bull_end = EARLY_BALANCE_BULLISH_END
        elif playbook == "AFTERNOON_TREND_HOLD_BULLISH":
            required_bull_end = AFTERNOON_TREND_BULLISH_END
        else:
            required_bull_end = SIDEWAYS_BULLISH_RECLAIM_END
        if now_time > required_bull_end:
            reasons.append(
                f"Bull Put Credit Spreads for {playbook} are only allowed until {required_bull_end.strftime('%H:%M')} IST to avoid chasing late continuation."
            )
            return StrategyType.NO_TRADE, reasons
        if now_time >= required_bull_start:
            reasons.append(
                f"15m TrendUp + bullish {bullish_setup or 'CONTINUATION'} confirmation on {playbook} -> Bull Put Credit Spread."
            )
            return StrategyType.BULL_PUT_CREDIT_SPREAD, reasons
        reasons.append(
            f"Bull Put Credit Spreads require time >= {required_bull_start.strftime('%H:%M')} IST for the current {playbook} playbook."
        )
        return StrategyType.NO_TRADE, reasons

    if regime_state.regime == RegimeLabel.RANGE:
        playbook = str(regime_state.metadata.get("playbook") or "UNKNOWN")
        range_ready = bool(regime_state.metadata.get("range_entry_ready"))
        range_balance_score = float(regime_state.metadata.get("range_balance_score") or 0.0)
        if context_layer_active and tradability == REGIME_TRADABILITY_NOT_TRADABLE:
            reasons.append(tradability_reason)
            return StrategyType.NO_TRADE, reasons
        if playbook != "RANGE_BALANCED_CONDOR" or not range_ready:
            reasons.append("Range regime detected, but the session is not balanced enough for the dedicated condor playbook.")
            return StrategyType.NO_TRADE, reasons
        required_range_confidence = RANGE_CONDOR_MIN_CONFIDENCE + (0.05 if now_time >= time(14, 0) else 0.0)
        if regime_state.confidence + CONFIDENCE_EPSILON < required_range_confidence:
            reasons.append(
                f"Range playbook detected, but conviction {regime_state.confidence:.2f} is below the required {required_range_confidence:.2f}."
            )
            return StrategyType.NO_TRADE, reasons
        if not ENABLE_RANGE_CONDOR_ACTIVE:
            reasons.append(f"Dedicated range condor playbook remains in shadow until it proves edge on honest data. Balance score {range_balance_score:.2f}.")
            return StrategyType.NO_TRADE, reasons
        if playbook_tier(playbook) not in allowed_playbook_tiers:
            reasons.append(
                f"Playbook {playbook} is Tier {playbook_tier(playbook)}; current benchmark mode allows only tiers {', '.join(allowed_playbook_tiers)}."
            )
            return StrategyType.NO_TRADE, reasons
        # Wednesday (Day 1 of new weekly series, post-Tuesday expiry):
        # 7DTE options carry maximum weekly premium — best IC deployment day.
        # Start 45 min earlier than normal, same RV30 limit.
        is_post_expiry_day = bool(regime_state.metadata.get("is_post_expiry_day"))
        is_expiry_day = bool(regime_state.metadata.get("is_expiry_day"))
        _rv30_limit = 0.20
        _condor_start = RANGE_CONDOR_POST_EXPIRY_START if is_post_expiry_day else RANGE_CONDOR_START
        _condor_end = RANGE_CONDOR_POST_EXPIRY_END if is_post_expiry_day else RANGE_CONDOR_END
        if is_expiry_day:
            # Tuesday expiry day: same-day expiry chain → extreme gamma → no IC or strangle
            reasons.append("Expiry day (Tuesday): Iron Condor / Strangle blocked — same-day expiry gamma risk is unacceptable.")
            return StrategyType.NO_TRADE, reasons
        # Extreme IV path: rv30 > 0.30 + avg_iv >= straddle floor → sell ATM straddle
        _straddle_rv30_limit = 0.30
        avg_chain_iv = float(regime_state.metadata.get("avg_chain_iv") or 0.0)
        range_balance_score = float(regime_state.metadata.get("range_balance_score") or 0.0)
        if regime_state.rv30_pct > _straddle_rv30_limit and avg_chain_iv >= SHORT_STRADDLE_IV_FLOOR and range_balance_score >= SHORT_STRADDLE_RANGE_BALANCE_MIN:
            if is_expiry_day:
                reasons.append("Expiry day: Short Straddle blocked — same-day ATM gamma risk is unacceptable.")
                return StrategyType.NO_TRADE, reasons
            if now_time > SHORT_STRADDLE_END:
                reasons.append(f"Short Straddle entries avoided after {SHORT_STRADDLE_END.strftime('%H:%M')} IST.")
                return StrategyType.NO_TRADE, reasons
            if now_time >= SHORT_STRADDLE_START:
                reasons.append(
                    f"Range day with extreme IV (rv30={regime_state.rv30_pct:.3f}, avg_iv={avg_chain_iv:.1f}%): "
                    "sell ATM straddle to maximise premium capture on balanced session."
                )
                return StrategyType.SHORT_STRADDLE, reasons
            reasons.append(f"Short Straddle allowed only after {SHORT_STRADDLE_START.strftime('%H:%M')} IST.")
            return StrategyType.NO_TRADE, reasons
        # Elevated IV path: rv30 0.20–0.30 makes condor wings expensive → use short strangle instead
        _strangle_rv30_limit = 0.30
        if _rv30_limit < regime_state.rv30_pct <= _strangle_rv30_limit and avg_chain_iv >= 18.0:
            if now_time > RANGE_STRANGLE_END:
                reasons.append(f"Short Strangle entries avoided after {RANGE_STRANGLE_END.strftime('%H:%M')} IST.")
                return StrategyType.NO_TRADE, reasons
            if now_time >= RANGE_STRANGLE_START:
                reasons.append(
                    f"Range day with elevated IV (rv30={regime_state.rv30_pct:.3f}, avg_iv={avg_chain_iv:.1f}%): "
                    "condor wings too expensive → Short Strangle."
                )
                return StrategyType.SHORT_STRANGLE, reasons
            reasons.append(f"Short Strangle allowed only after {RANGE_STRANGLE_START.strftime('%H:%M')} IST.")
            return StrategyType.NO_TRADE, reasons
        if regime_state.rv30_pct > _rv30_limit:
            reasons.append(f"Iron Condor requires low-volatility range (rv30={regime_state.rv30_pct:.3f} > {_rv30_limit}).")
            return StrategyType.NO_TRADE, reasons
        if now_time > _condor_end:
            reasons.append(f"Iron Condor entries are avoided after {_condor_end.strftime('%H:%M')} IST.")
            return StrategyType.NO_TRADE, reasons
        if now_time >= _condor_start:
            weekly_note = (
                f" [Day-1 weekly series: 7DTE, wider strikes, credit_ratio={regime_state.metadata.get('range_condor_credit_ratio', 0.16):.2f}]"
                if is_post_expiry_day else ""
            )
            reasons.append(f"Neutral 15m regime + balanced range playbook ({range_balance_score:.2f}) -> Iron Condor{weekly_note}.")
            return StrategyType.IRON_CONDOR, reasons
        reasons.append(f"Iron Condor is allowed only after {_condor_start.strftime('%H:%M')} IST.")
        return StrategyType.NO_TRADE, reasons

    reasons.append("No aligned regime/trigger pair available.")
    return StrategyType.NO_TRADE, reasons
