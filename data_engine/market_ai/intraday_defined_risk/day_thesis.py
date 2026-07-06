"""
Day Thesis Engine — one clear market view formed at 10:00-10:15 AM,
anchoring the entire session's strategy selection.

A professional trader doesn't score the market every 10 minutes.
They read it once — multi-timeframe, options flow, key levels — form
a thesis, and then execute from that thesis until it's invalidated.

Decision hierarchy:
  1. Daily structure: where is price in the bigger picture?
  2. First-hour behavior: did the market open trending or ranging?
  3. Institutional positioning: where are OI walls, PCR, smart money?
  4. Premium environment: is IV worth selling?

→ ONE answer: TREND_UP / TREND_DOWN / RANGE_PREMIUM / UNCERTAIN
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_STATE_ROOT = Path(__file__).parents[2] / "state"
_THESIS_PATH = _STATE_ROOT / "day_thesis.json"

# How much spot must move beyond a key level before the thesis is invalidated
_RANGE_THESIS_INVALIDATION_PTS = 60.0   # price closed above resistance by 60 pts → not range anymore
_TREND_THESIS_INVALIDATION_PTS = 100.0  # price retraced more than 100 pts → trend lost


# ──────────────────────────────────────────────────────────────────────────────
# Core computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_day_thesis(
    spot: float,
    trend_15m: str,
    execution_5m: str,
    daily_tf: dict[str, Any],
    option_context: dict[str, Any],
    atm_straddle_pts: float,
    premarket_bias: str,
    smart_money_bias: str,
    open_space_up: float,
    open_space_down: float,
    now: datetime,
) -> dict[str, Any]:
    """
    Form the day thesis. Called once after the opening range is established
    (10:00–10:30 AM). Returns a dict that's persisted for the session.
    """
    # ── Signal aggregation ───────────────────────────────────────────────────
    trend_up_score = 0.0
    trend_dn_score = 0.0
    range_score = 0.0
    signals: list[str] = []

    # 15m trend: strongest single signal
    if trend_15m == "TREND_UP":
        trend_up_score += 2.0
        signals.append("15m TREND_UP (+2 trend-up)")
    elif trend_15m == "TREND_DOWN":
        trend_dn_score += 2.0
        signals.append("15m TREND_DOWN (+2 trend-dn)")
    else:
        range_score += 1.0
        signals.append("15m NEUTRAL (+1 range)")

    # 5m execution: confirmed direction
    if execution_5m == "UP_CONFIRMED":
        trend_up_score += 1.0
        signals.append("5m UP_CONFIRMED (+1 trend-up)")
    elif execution_5m == "DOWN_CONFIRMED":
        trend_dn_score += 1.0
        signals.append("5m DOWN_CONFIRMED (+1 trend-dn)")
    elif execution_5m == "RANGE_CONFIRMED":
        range_score += 1.5
        signals.append("5m RANGE_CONFIRMED (+1.5 range)")
    else:
        range_score += 0.5
        signals.append("5m UNCONFIRMED (+0.5 range)")

    # HTF resistance proximity: price near a wall = range trade
    daily_resistance = daily_tf.get("daily_swing_resistance")
    weekly_high = daily_tf.get("weekly_high")
    daily_support = daily_tf.get("daily_swing_support")
    weekly_low = daily_tf.get("weekly_low")

    # Nearest structural resistance above spot
    htf_resistance_levels = [
        l for l in [daily_resistance, weekly_high, option_context.get("nearest_call_wall")]
        if isinstance(l, (int, float)) and float(l) > spot
    ]
    htf_support_levels = [
        l for l in [daily_support, weekly_low, option_context.get("nearest_put_wall")]
        if isinstance(l, (int, float)) and float(l) < spot
    ]
    key_resistance = min((float(l) for l in htf_resistance_levels), default=None)
    key_support = max((float(l) for l in htf_support_levels), default=None)

    dist_to_res = (key_resistance - spot) if key_resistance else 999.0
    dist_to_sup = (spot - key_support) if key_support else 999.0

    if dist_to_res < 100.0:
        range_score += 2.0
        signals.append(f"HTF resistance {key_resistance:.0f} only {dist_to_res:.0f}pts away (+2 range)")
    elif dist_to_res < 200.0:
        range_score += 0.8
        signals.append(f"HTF resistance {key_resistance:.0f} is {dist_to_res:.0f}pts away (+0.8 range)")
    elif dist_to_res >= 200.0 and trend_15m == "TREND_UP":
        trend_up_score += 0.5
        signals.append(f"HTF resistance {key_resistance:.0f} far ({dist_to_res:.0f}pts) — trend has room (+0.5 trend-up)")

    if dist_to_sup < 100.0:
        range_score += 2.0
        signals.append(f"HTF support {key_support:.0f} only {dist_to_sup:.0f}pts away (+2 range)")
    elif dist_to_sup >= 200.0 and trend_15m == "TREND_DOWN":
        trend_dn_score += 0.5
        signals.append(f"HTF support {key_support:.0f} far — downtrend has room (+0.5 trend-dn)")

    # IV / premium environment: elevated IV = premium selling opportunity
    iv_regime = "NORMAL"
    if atm_straddle_pts >= 120.0:
        iv_regime = "HIGH"
        range_score += 1.0
        signals.append(f"ATM straddle {atm_straddle_pts:.0f}pts (HIGH IV, +1 range)")
    elif atm_straddle_pts >= 80.0:
        iv_regime = "MODERATE"
        range_score += 0.5
        signals.append(f"ATM straddle {atm_straddle_pts:.0f}pts (MODERATE IV, +0.5 range)")
    elif atm_straddle_pts < 50.0:
        iv_regime = "LOW"
        signals.append(f"ATM straddle {atm_straddle_pts:.0f}pts (LOW IV, no premium edge)")

    # Pre-market bias (gap + PCR computed earlier)
    if premarket_bias == "TRENDING":
        trend_up_score += 0.5
        trend_dn_score += 0.5
        signals.append("Pre-market bias TRENDING (+0.5 each direction)")
    elif premarket_bias == "RANGE":
        range_score += 0.8
        signals.append("Pre-market bias RANGE (+0.8 range)")

    # Smart money / institutional flow
    if smart_money_bias == "BULLISH":
        trend_up_score += 0.5
        signals.append("Smart money BULLISH (+0.5 trend-up)")
    elif smart_money_bias == "BEARISH":
        trend_dn_score += 0.5
        signals.append("Smart money BEARISH (+0.5 trend-dn)")
    else:
        range_score += 0.3
        signals.append("Smart money NEUTRAL (+0.3 range)")

    # ── Thesis determination ─────────────────────────────────────────────────
    # A trend day requires strong multi-timeframe alignment.
    # A range day just needs price containment + premium opportunity.
    # Uncertainty means stay out.

    max_trend = max(trend_up_score, trend_dn_score)
    leading = "TREND_UP" if trend_up_score >= trend_dn_score else "TREND_DOWN"
    leading_score = max_trend

    MIN_TREND_SCORE = 3.5
    MIN_RANGE_SCORE = 3.0
    MIN_RANGE_PREMIUM = 60.0   # straddle must be worth selling

    if leading_score >= MIN_TREND_SCORE and leading_score > range_score + 0.5:
        day_type = leading
    elif range_score >= MIN_RANGE_SCORE and atm_straddle_pts >= MIN_RANGE_PREMIUM:
        day_type = "RANGE_PREMIUM"
    else:
        day_type = "UNCERTAIN"

    # Conviction = how much the winning category leads
    if day_type in ("TREND_UP", "TREND_DOWN"):
        conviction = min(0.95, 0.50 + (leading_score - MIN_TREND_SCORE) * 0.15)
    elif day_type == "RANGE_PREMIUM":
        conviction = min(0.90, 0.50 + (range_score - MIN_RANGE_SCORE) * 0.12)
    else:
        conviction = 0.30

    # ── Strategy selection ───────────────────────────────────────────────────
    if day_type == "TREND_UP":
        preferred_strategy = "BULL_PUT_SPREAD"
        reasoning = (
            f"Trend up: 15m={trend_15m}, 5m={execution_5m}, resistance {dist_to_res:.0f}pts away — "
            f"directional bull spread."
        )
    elif day_type == "TREND_DOWN":
        preferred_strategy = "BEAR_CALL_SPREAD"
        reasoning = (
            f"Trend down: 15m={trend_15m}, 5m={execution_5m}, support {dist_to_sup:.0f}pts away — "
            f"directional bear spread."
        )
    elif day_type == "RANGE_PREMIUM":
        if atm_straddle_pts >= 100.0 and open_space_up >= 80.0 and open_space_down >= 80.0:
            preferred_strategy = "SHORT_STRADDLE"
            reasoning = (
                f"Range: spot {dist_to_res:.0f}pts from HTF resistance {key_resistance:.0f}, "
                f"ATM straddle {atm_straddle_pts:.0f}pts (rich premium), 15m neutral — "
                f"sell ATM straddle."
            )
        elif atm_straddle_pts >= 70.0:
            preferred_strategy = "SHORT_STRANGLE"
            reasoning = (
                f"Range: spot {dist_to_res:.0f}pts from resistance {key_resistance:.0f if key_resistance else 'N/A'}, "
                f"ATM straddle {atm_straddle_pts:.0f}pts — sell OTM strangle."
            )
        else:
            preferred_strategy = "IRON_CONDOR"
            reasoning = (
                f"Range: price contained, ATM straddle {atm_straddle_pts:.0f}pts (low IV) — "
                f"iron condor for defined-risk theta."
            )
    else:
        preferred_strategy = "NO_TRADE"
        reasoning = (
            f"Unclear: trend_score={leading_score:.1f}, range_score={range_score:.1f}, "
            f"straddle={atm_straddle_pts:.0f}pts — no edge, stay out."
        )

    return {
        "session_date": now.strftime("%Y-%m-%d"),
        "formed_at": now.isoformat(),
        "formed_at_price": round(spot, 2),
        "day_type": day_type,
        "preferred_strategy": preferred_strategy,
        "conviction": round(conviction, 3),
        "key_resistance": round(key_resistance, 2) if key_resistance else None,
        "key_support": round(key_support, 2) if key_support else None,
        "iv_regime": iv_regime,
        "atm_straddle_pts": round(atm_straddle_pts, 2),
        "reasoning": reasoning,
        "trend_up_score": round(trend_up_score, 2),
        "trend_dn_score": round(trend_dn_score, 2),
        "range_score": round(range_score, 2),
        "signals": signals,
        "invalidated": False,
        "invalidation_reason": "",
    }


# ──────────────────────────────────────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────────────────────────────────────

def load_day_thesis(session_date: str | None = None, *, path: Path | None = None) -> dict[str, Any] | None:
    """Load today's thesis from the state file. Returns None if stale or missing."""
    p = path or _THESIS_PATH
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except Exception:
        return None
    today = session_date or datetime.now().strftime("%Y-%m-%d")
    if data.get("session_date") != today:
        return None
    return data


def save_day_thesis(thesis: dict[str, Any], *, path: Path | None = None) -> None:
    """Persist the thesis to the state file."""
    p = path or _THESIS_PATH
    try:
        p.write_text(json.dumps(thesis, indent=2))
    except Exception as exc:
        logger.warning("Failed to save day_thesis: %s", exc)


def check_invalidation(
    thesis: dict[str, Any],
    spot: float,
    trend_15m: str,
    execution_5m: str,
) -> dict[str, Any]:
    """
    Check whether the session thesis is still valid.
    Invalidation means the market has moved decisively AGAINST the thesis.
    """
    if thesis.get("invalidated"):
        return thesis

    day_type = thesis.get("day_type", "UNCERTAIN")
    key_resistance = thesis.get("key_resistance")

    if day_type == "RANGE_PREMIUM":
        # Invalidate if price broke cleanly above resistance (trend emerging)
        if (
            key_resistance
            and spot > key_resistance + _RANGE_THESIS_INVALIDATION_PTS
            and trend_15m == "TREND_UP"
            and execution_5m == "UP_CONFIRMED"
        ):
            thesis = dict(thesis)
            thesis["invalidated"] = True
            thesis["invalidation_reason"] = (
                f"Price {spot:.0f} broke above resistance {key_resistance:.0f} by "
                f"{spot - key_resistance:.0f}pts with 15m TREND_UP — thesis shifted to TREND_UP."
            )
            logger.info("Day thesis invalidated: %s", thesis["invalidation_reason"])

    elif day_type == "TREND_UP":
        # Invalidate if 15m trend reversed
        if trend_15m == "TREND_DOWN" and execution_5m == "DOWN_CONFIRMED":
            thesis = dict(thesis)
            thesis["invalidated"] = True
            thesis["invalidation_reason"] = "15m turned TREND_DOWN — bullish thesis invalidated."
            logger.info("Day thesis invalidated: %s", thesis["invalidation_reason"])

    elif day_type == "TREND_DOWN":
        if trend_15m == "TREND_UP" and execution_5m == "UP_CONFIRMED":
            thesis = dict(thesis)
            thesis["invalidated"] = True
            thesis["invalidation_reason"] = "15m turned TREND_UP — bearish thesis invalidated."
            logger.info("Day thesis invalidated: %s", thesis["invalidation_reason"])

    return thesis


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point for classify_regime()
# ──────────────────────────────────────────────────────────────────────────────

_THESIS_FORM_TIME = time(10, 0)   # earliest time to form the thesis
_THESIS_FORM_END = time(10, 45)   # after this, too late to form a fresh thesis


def get_or_form_day_thesis(
    spot: float,
    trend_15m: str,
    execution_5m: str,
    daily_tf: dict[str, Any],
    option_context: dict[str, Any],
    atm_straddle_pts: float,
    premarket_bias: str,
    smart_money_bias: str,
    open_space_up: float,
    open_space_down: float,
    now: datetime,
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    """
    Load today's thesis if it exists and is still valid.
    Otherwise form a new one (only between 10:00 and 10:45 AM).
    Always checks for invalidation before returning.
    """
    session_date = now.strftime("%Y-%m-%d")
    thesis = load_day_thesis(session_date, path=path)

    if thesis is None:
        # No thesis yet for today
        if not (_THESIS_FORM_TIME <= now.time() <= _THESIS_FORM_END):
            # Too early or too late to form — return empty
            return {
                "session_date": session_date,
                "day_type": "UNFORMED",
                "preferred_strategy": "UNKNOWN",
                "conviction": 0.0,
                "reasoning": f"Thesis not formed yet (time={now.time().strftime('%H:%M')}, window=10:00-10:45).",
                "invalidated": False,
            }
        # Form fresh
        if execution_5m == "UNCONFIRMED" and trend_15m == "NEUTRAL":
            # Market hasn't established direction yet — too early to commit
            return {
                "session_date": session_date,
                "day_type": "UNFORMED",
                "preferred_strategy": "UNKNOWN",
                "conviction": 0.0,
                "reasoning": "Market direction unestablished at thesis formation time — waiting.",
                "invalidated": False,
            }
        thesis = compute_day_thesis(
            spot=spot,
            trend_15m=trend_15m,
            execution_5m=execution_5m,
            daily_tf=daily_tf,
            option_context=option_context,
            atm_straddle_pts=atm_straddle_pts,
            premarket_bias=premarket_bias,
            smart_money_bias=smart_money_bias,
            open_space_up=open_space_up,
            open_space_down=open_space_down,
            now=now,
        )
        save_day_thesis(thesis, path=path)
        logger.info(
            "Day thesis formed at %s: type=%s conviction=%.2f — %s",
            now.strftime("%H:%M"),
            thesis["day_type"],
            thesis["conviction"],
            thesis["reasoning"],
        )

    # Check if still valid
    thesis = check_invalidation(thesis, spot, trend_15m, execution_5m)
    if thesis.get("invalidated"):
        save_day_thesis(thesis, path=path)  # persist the invalidation flag

    return thesis
