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
SIDEWAYS_BEARISH_REJECTION_END = time(12, 45)
GAP_BEARISH_START = time(9, 45)
GAP_BEARISH_END = time(13, 0)
GAP_DOWN_BEARISH_CONTINUATION_END = time(12, 0)
GAP_UP_BEARISH_FAILURE_END = time(11, 15)
RANGE_CONDOR_START = time(11, 30)
RANGE_CONDOR_END = time(13, 30)
OPEN_DRIVE_BULLISH_MIN_CONFIDENCE = 0.64
GAP_UP_BULLISH_MIN_CONFIDENCE = 0.68
GAP_DOWN_BULLISH_RECOVERY_MIN_CONFIDENCE = 0.70
SIDEWAYS_BULLISH_RECLAIM_MIN_CONFIDENCE = 0.66
EARLY_BALANCE_BULLISH_MIN_CONFIDENCE = 0.72
HIGH_CONFLUENCE_BULLISH_MIN_CONFIDENCE = 0.88
AFTERNOON_TREND_BULLISH_MIN_CONFIDENCE = 0.72
RANGE_CONDOR_MIN_CONFIDENCE = 0.55
CONFIDENCE_EPSILON = 1e-6
BEAR_CALL_MIN_CONFIDENCE = 0.85
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

TIER_A_PLAYBOOKS = {
    "SIDEWAYS_TO_BEARISH_REJECTION",
    "GAP_DOWN_BEARISH_CONTINUATION",
    "GAP_UP_BEARISH_FAILURE",
    "RANGE_BALANCED_CONDOR",
}

TIER_B_PLAYBOOKS = {
    "SIDEWAYS_TO_BULLISH_RECLAIM",
    "OPEN_DRIVE_BULLISH",
    "HIGH_CONFLUENCE_BULLISH_CONTINUATION",
    "GAP_UP_BULLISH_CONTINUATION",
    "GAP_DOWN_BULLISH_RECOVERY",
    "EARLY_BALANCE_BULLISH_RECLAIM",
    "AFTERNOON_TREND_HOLD_BULLISH",
    "EARLY_BALANCE_BEARISH_FAILED_RECLAIM",
    "HIGH_CONFLUENCE_BEARISH_CONTINUATION",
    "BEARISH_FAILED_RECLAIM",
    "BEARISH_CONTINUATION",
}


def playbook_tier(playbook: str) -> str:
    if playbook in TIER_A_PLAYBOOKS:
        return "A"
    if playbook in TIER_B_PLAYBOOKS:
        return "B"
    return "C"


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


def playbook_time_window(regime_state: RegimeState) -> tuple[time | None, time | None]:
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
        return GAP_BEARISH_START, GAP_DOWN_BEARISH_CONTINUATION_END
    if playbook == "HIGH_CONFLUENCE_BEARISH_CONTINUATION":
        return HIGH_CONFLUENCE_BEARISH_START, HIGH_CONFLUENCE_BEARISH_END
    if playbook == "RANGE_BALANCED_CONDOR":
        return RANGE_CONDOR_START, RANGE_CONDOR_END
    return None, None


def select_strategy(
    regime_state: RegimeState,
    now_time: time,
    *,
    allowed_playbook_tiers: tuple[str, ...] = ("A", "B"),
) -> tuple[StrategyType, list[str]]:
    reasons = list(regime_state.reasons)
    day_archetype = str(regime_state.metadata.get("day_archetype") or "UNCLASSIFIED")
    if now_time < time(9, 15):
        reasons.append("Pre-market entry is not allowed.")
        return StrategyType.NO_TRADE, reasons

    if regime_state.regime == RegimeLabel.DOWN_TREND:
        metadata = regime_state.metadata
        bearish_setup = metadata.get("bearish_setup")
        bearish_ready = bool(metadata.get("bearish_entry_ready"))
        playbook = str(metadata.get("playbook") or "UNKNOWN")
        if not bearish_ready:
            reasons.append(
                "Bearish downtrend detected, but price has not yet printed a valid bearish pullback rejection, failed reclaim, or shallow continuation setup."
            )
            return StrategyType.NO_TRADE, reasons
        if regime_state.confidence + CONFIDENCE_EPSILON < BEAR_CALL_MIN_CONFIDENCE:
            reasons.append(
                f"Bearish setup detected, but conviction {regime_state.confidence:.2f} is below the required {BEAR_CALL_MIN_CONFIDENCE:.2f}."
            )
            return StrategyType.NO_TRADE, reasons
        if day_archetype not in {"OPEN_DRIVE_BEARISH", "EARLY_BALANCE_TO_BEARISH", "SIDEWAYS_TO_BEARISH", "GAP_UP_FAILURE", "GAP_DOWN_CONTINUATION", "HIGH_CONFLUENCE_BEARISH"}:
            reasons.append(
                f"Bearish regime is generic {day_archetype}; only gap/open-drive/sideways bearish archetypes are eligible for downside deployment."
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
            required_bear_end = GAP_DOWN_BEARISH_CONTINUATION_END
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
        if regime_state.confidence + CONFIDENCE_EPSILON < required_confidence:
            reasons.append(
                f"Bullish setup detected, but conviction {regime_state.confidence:.2f} is below the required {required_confidence:.2f}."
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
        if playbook != "RANGE_BALANCED_CONDOR" or not range_ready:
            reasons.append("Range regime detected, but the session is not balanced enough for the dedicated condor playbook.")
            return StrategyType.NO_TRADE, reasons
        if regime_state.confidence + CONFIDENCE_EPSILON < RANGE_CONDOR_MIN_CONFIDENCE:
            reasons.append(
                f"Range playbook detected, but conviction {regime_state.confidence:.2f} is below the required {RANGE_CONDOR_MIN_CONFIDENCE:.2f}."
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
        if regime_state.rv30_pct > 0.20:
            reasons.append("Iron Condor requires a cleaner low-volatility range regime.")
            return StrategyType.NO_TRADE, reasons
        if now_time > RANGE_CONDOR_END:
            reasons.append(f"Iron Condor entries are avoided after {RANGE_CONDOR_END.strftime('%H:%M')} IST to prevent late-session compression traps.")
            return StrategyType.NO_TRADE, reasons
        if now_time >= RANGE_CONDOR_START:
            reasons.append(f"Neutral 15m regime + balanced range playbook ({range_balance_score:.2f}) -> Iron Condor.")
            return StrategyType.IRON_CONDOR, reasons
        reasons.append(f"Iron Condor is allowed only after {RANGE_CONDOR_START.strftime('%H:%M')} IST.")
        return StrategyType.NO_TRADE, reasons

    reasons.append("No aligned regime/trigger pair available.")
    return StrategyType.NO_TRADE, reasons
