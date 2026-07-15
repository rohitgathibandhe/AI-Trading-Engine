"""
Strategy Selector — condition-first strategy choice.

The agent's failure was structural: it owns one tool (credit spreads) and forces
it onto every market. A real trader does the opposite — reads the market, names
the condition, then picks the strategy family that FITS that condition, and only
then worries about entry timing.

This module is that missing brain. It does NOT pick strikes or place trades; it
answers one question every cycle: *given what the market is doing right now, what
KIND of trade (if any) belongs here?* The output is a StrategyChoice — a family
(directional-debit, premium-selling, long-vol, stand-aside …), the concrete
structures that express it, and the reasoning.

Condition taxonomy (from signals the agent already computes):
  STRONG_TREND_UP / DOWN  — 15m + 5m confirmed, momentum, room to the next wall
  BREAKOUT_UP / DOWN      — opening-range / structure break with follow-through
  RANGE_WIDE              — two-sided walls far apart, balanced tape, rich premium
  RANGE_TIGHT             — compressed, low premium, walls close
  HIGH_VOL_UNDIRECTED     — elevated IV/RV but no direction (event/whipsaw risk)
  CHOP / TRANSITION       — uncommitted tape → stand aside (the -Rs73k killer)

IV regime (rich vs cheap premium) selects SELLING vs BUYING within a condition.

NOTE: several target structures (directional debit spreads, long options) are not
yet executable in strikes.py/execution.py — this selector NAMES them so the build
order is explicit. Credit spreads / condor / strangle ARE executable today.
"""

from __future__ import annotations

import os
from datetime import time
from dataclasses import dataclass, field
from typing import Any

from .trade_planner import assess_market, MarketRead, _f

# Research toggle: stand aside from RANGE_WIDE condors. The ordering-robustness
# test showed condor P&L is noise (~0 median, sign-flips across orderings) while
# PUT_DEBIT is the robust edge. This flag lets us validate whether cutting the
# noisy condor (and freeing the slot for the robust edge) helps across orderings.
_NO_CONDOR = os.environ.get("SEL_NO_CONDOR") == "1"

# Selection-skill filter (VALIDATED, default ON): the agent's entry SCORE does not separate
# put-debit winners from losers (7.15 vs 7.09). The one signal that DOES: option_chain_pressure_state.
# Trades opened under OVERHEAD_CALL_PRESSURE (capped/grinding tape, no clean down-leg) lose
# -1,333/trade over 29 trades (34% win, -38.7k total) vs BALANCED_WALLS +3,563/trade (61% win).
# Skipping them is ordering-robust: median 141,523 -> ~165,900, worst-case 91,368 -> 123,678,
# EVERY grid up +10-38k, PF up on all 10. NOTE: signal mined in-sample on the dense 7mo set —
# mechanism is sound but confirm on forward/live data. Set SEL_SKIP_OVERHEAD_PRESSURE=0 to disable.
_SKIP_OVERHEAD_PRESSURE = os.environ.get("SEL_SKIP_OVERHEAD_PRESSURE", "1") == "1"


def _sel_env_float(name):
    v = os.environ.get(name)
    try:
        return float(v) if v not in (None, "") else None
    except ValueError:
        return None


# Anti-chase (VALIDATED, default 0.4): live 2026-07-15 the agent entered a put-debit AFTER a sharp
# recent drop (last hour -0.58%, price -136 vs VWAP) — it bought the EXHAUSTION of a ~100pt impulse
# right before the tape went flat, and bled theta. Skip the debit when the last-hour drop already
# exceeds the threshold (the impulse is spent → no continuation for the debit). Ordering-robust on
# the dense 7mo: median 170,665 -> 186,848 (+16k), floor held (147,821 -> 147,961), PF up on ALL 10
# grids (2.15-4.09). Mechanism sound (don't chase); confirm on forward/live. SEL_ANTICHASE_HOUR_DROP=0 disables.
_ac_env = os.environ.get("SEL_ANTICHASE_HOUR_DROP")
_ANTICHASE_HOUR_DROP = 0.4 if _ac_env is None else (_sel_env_float("SEL_ANTICHASE_HOUR_DROP") or None)

# Research toggle: which structure to deploy on STRONG_TREND_UP / BREAKOUT_UP.
# Baseline "BULL_PUT" earns ~0 across orderings (no up-side edge). Values:
#   BULL_PUT   — with-trend credit spread (current default)
#   CALL_DEBIT — buy a call-debit with the up-move (symmetric to the put-debit engine)
#   ASIDE      — stand aside on up-moves entirely
_UP_STRUCT = os.environ.get("SEL_UP_STRUCT", "BULL_PUT").upper()

# Research toggle: blackout window for the PUT_DEBIT engine. Leak analysis showed
# down-trend debit entries in 10:00-10:29 bleed -Rs36.7k over 20 trades / 12 distinct
# days (open-drive exhaustion → entries buy the bottom right before the bounce), while
# 9:30 and 10:30+ are net positive. Format "HH:MM-HH:MM"; empty = no blackout.
def _parse_window(s: str):
    try:
        a, b = s.split("-")
        ah, am = map(int, a.split(":")); bh, bm = map(int, b.split(":"))
        return time(ah, am), time(bh, bm)
    except Exception:
        return None
_DEBIT_BLACKOUT = _parse_window(os.environ.get("SEL_DEBIT_BLACKOUT", ""))

# Strategy MODE. Default = the validated directional config (put-debit engine).
#   SELLER = always-positioned DEFINED-RISK premium seller, condition-matched:
#            bullish -> bull-put credit, bearish -> bear-call credit, sideways -> iron condor.
#   HYBRID = buy the sharp down-move (proven put-debit) but SELL premium everywhere else.
# This is the user's vision (never idle; sell defined-risk premium matched to the read).
# Gated so it can be validated on the ordering-robust harness before it drives live.
_SELLER_MODE = os.environ.get("SEL_MODE", "").upper()


def _seller_choice(choice, condition, now_time, *, hybrid: bool = False):
    """Defined-risk, always-positioned premium seller: match a credit structure to the
    market condition. hybrid=True keeps the proven PUT-DEBIT on sharp down-moves."""
    if condition in (STRONG_TREND_UP, BREAKOUT_UP):
        choice.family, choice.structures = FAM_DIRECTIONAL_CREDIT, ["BULL_PUT_CREDIT_SPREAD"]
        choice.rationale = f"{condition}: SELL bull-put credit (defined risk) — theta pays if price holds up or sideways."
    elif condition in (STRONG_TREND_DOWN, BREAKOUT_DOWN):
        if hybrid:
            choice.family, choice.structures = FAM_DIRECTIONAL_DEBIT, ["PUT_DEBIT_SPREAD"]
            choice.rationale = f"{condition}: BUY put-debit (proven engine — Nifty falls sharp)."
        else:
            choice.family, choice.structures = FAM_DIRECTIONAL_CREDIT, ["BEAR_CALL_CREDIT_SPREAD"]
            choice.rationale = f"{condition}: SELL bear-call credit (defined risk) — theta pays if price stays below."
    elif condition in (RANGE_WIDE, RANGE_TIGHT, HIGH_VOL_UNDIRECTED):
        choice.family, choice.structures = FAM_PREMIUM_SELL, ["IRON_CONDOR"]
        choice.rationale = f"{condition}: SELL iron condor (defined risk) — theta pays if price stays in the range."
    else:  # CHOP
        choice.family, choice.structures = FAM_STAND_ASIDE, []
        choice.rationale = "CHOP: uncommitted tape — even a seller stands aside (whipsaw risk on both wings)."
    choice.executable_today = bool(choice.structures) and all(s in _EXECUTABLE for s in choice.structures)
    return choice
from .volatility_engine import assess_vol, RICH_SELL, CHEAP_BUY

# ── Condition labels ────────────────────────────────────────────────────────
STRONG_TREND_UP = "STRONG_TREND_UP"
STRONG_TREND_DOWN = "STRONG_TREND_DOWN"
BREAKOUT_UP = "BREAKOUT_UP"
BREAKOUT_DOWN = "BREAKOUT_DOWN"
RANGE_WIDE = "RANGE_WIDE"
RANGE_TIGHT = "RANGE_TIGHT"
HIGH_VOL_UNDIRECTED = "HIGH_VOL_UNDIRECTED"
CHOP = "CHOP"

# ── Strategy families (the "what to deploy") ────────────────────────────────
FAM_DIRECTIONAL_DEBIT = "DIRECTIONAL_DEBIT"      # buy a spread in the trend's direction (positive skew)
FAM_DIRECTIONAL_CREDIT = "DIRECTIONAL_CREDIT"    # sell a spread on the far side (with-trend, theta)
FAM_PREMIUM_SELL = "PREMIUM_SELL"                # condor / strangle inside a range
FAM_LONG_VOL = "LONG_VOL"                        # long straddle/strangle for an expected expansion
FAM_STAND_ASIDE = "STAND_ASIDE"

# IV regimes
IV_RICH = "RICH"     # sell premium is favoured
IV_CHEAP = "CHEAP"   # buy premium is favoured
IV_NORMAL = "NORMAL"

# Executable today (Stage 2 added the directional debit spreads).
_EXECUTABLE = {"BEAR_CALL_CREDIT_SPREAD", "BULL_PUT_CREDIT_SPREAD", "IRON_CONDOR",
               "SHORT_STRANGLE", "SHORT_STRADDLE", "CALL_DEBIT_SPREAD", "PUT_DEBIT_SPREAD"}


@dataclass
class StrategyChoice:
    condition: str = CHOP
    iv_regime: str = IV_NORMAL
    family: str = FAM_STAND_ASIDE
    structures: list[str] = field(default_factory=list)  # concrete structures, best-first
    executable_today: bool = False
    conviction: float = 0.0
    rationale: str = ""
    read: MarketRead | None = None
    vol_regime: str = "NEUTRAL"          # RICH_SELL / CHEAP_BUY / NEUTRAL (volatility engine)
    vol_notes: list[str] = field(default_factory=list)


def _iv_regime(metadata: dict[str, Any]) -> str:
    """Rich vs cheap premium from India VIX + realized vol + IV rank."""
    vix = _f(metadata, "india_vix", 0.0)
    ivr = _f(metadata, "iv_rank_at_entry", _f(metadata, "iv_rank", 50.0))
    if vix >= 16 or ivr >= 65:
        return IV_RICH
    if (0 < vix < 11) or ivr <= 30:
        return IV_CHEAP
    return IV_NORMAL


def classify_condition(read: MarketRead, metadata: dict[str, Any]) -> str:
    """Name the market condition from the read + structure signals."""
    m = metadata
    orb = str(m.get("opening_range_break_state") or "NONE")
    trend_up_q = _f(m, "bullish_trend_quality_score", 0.0)
    trend_dn_q = _f(m, "bearish_trend_quality_score", 0.0)
    rv = _f(m, "rv30_pct", 0.0)
    vix = _f(m, "india_vix", 0.0)

    # Breakout: OR break with directional consensus
    if orb == "UP" and read.bias == "BULLISH":
        return BREAKOUT_UP
    if orb in {"DOWN", "FAILED_UP"} and read.bias == "BEARISH":
        return BREAKOUT_DOWN

    # Strong trend: high trend-quality + directional bias + conviction
    if read.bias == "BULLISH" and trend_up_q >= 4.0 and read.conviction >= 0.60:
        return STRONG_TREND_UP
    if read.bias == "BEARISH" and trend_dn_q >= 4.0 and read.conviction >= 0.60:
        return STRONG_TREND_DOWN

    # Range: two-sided walls, balanced tape
    if read.bias == "NEUTRAL" and read.call_wall is not None and read.put_wall is not None:
        width = read.call_wall - read.put_wall
        if width >= 200:
            return RANGE_WIDE
        if width > 0:
            return RANGE_TIGHT

    # High vol but no direction = event / whipsaw risk
    if (vix >= 16 or rv >= 1.2) and read.bias == "NEUTRAL":
        return HIGH_VOL_UNDIRECTED

    # Everything else = uncommitted tape
    return CHOP


def select_strategy(metadata: dict[str, Any], spot: float, now_time=None) -> StrategyChoice:
    """Read the market, name the condition, and choose the strategy family + structures.

    now_time (datetime.time, optional): used for time-of-day rules learned from the
    retrospective — e.g. condors only after the range forms (mid-day)."""
    read = assess_market(metadata, spot)
    condition = classify_condition(read, metadata)
    iv = _iv_regime(metadata)
    # Volatility engine FIRST — trade vol, not just direction. This tells us whether
    # premium is rich (sell) or cheap (buy), which the pros decide before direction.
    vol = assess_vol(metadata)
    choice = StrategyChoice(condition=condition, iv_regime=iv, read=read, conviction=read.conviction)
    choice.vol_regime = vol.regime
    choice.vol_notes = vol.notes

    # Defined-risk SELLER / HYBRID modes — always-positioned premium selling matched to
    # the condition. Validated on the ordering-robust harness before it drives live.
    if _SELLER_MODE == "SELLER":
        return _seller_choice(choice, condition, now_time, hybrid=False)
    if _SELLER_MODE == "HYBRID":
        return _seller_choice(choice, condition, now_time, hybrid=True)

    if condition in (STRONG_TREND_UP, BREAKOUT_UP):
        # Nifty UP-moves grind and chop — a call-debit needs a clean continued rally
        # and gets whipsawed (7mo: CALL_DEBIT -Rs30.9k). A with-trend BULL-PUT profits
        # on up OR sideways via theta, which fits the grindy character far better.
        # NOTE: ordering-robust test shows BULL_PUT earns ~0 here — no up-side edge yet.
        # _UP_STRUCT lets us test alternatives under the same robust harness.
        if _UP_STRUCT == "CALL_DEBIT":
            choice.family, choice.structures = FAM_DIRECTIONAL_DEBIT, ["CALL_DEBIT_SPREAD"]
            choice.rationale = f"{condition}: buy CALL DEBIT with the up-move (testing a symmetric up-side engine)."
        elif _UP_STRUCT == "ASIDE":
            choice.family, choice.structures = FAM_STAND_ASIDE, []
            choice.rationale = f"{condition}: stand aside on up-moves (no validated up-side edge)."
        else:
            choice.family, choice.structures = FAM_DIRECTIONAL_CREDIT, ["BULL_PUT_CREDIT_SPREAD"]
            choice.rationale = f"{condition}: with-trend bull-put (Nifty up-moves grind — theta fits better than a debit needing a clean rally)."

    elif condition in (STRONG_TREND_DOWN, BREAKOUT_DOWN):
        # Nifty DOWN-moves are sharp — a PUT-DEBIT's positive skew captures them
        # (7mo: PUT_DEBIT +Rs40.3k). This is the core edge.
        _in_blackout = (_DEBIT_BLACKOUT is not None and now_time is not None
                        and _DEBIT_BLACKOUT[0] <= now_time < _DEBIT_BLACKOUT[1])
        if _in_blackout:
            choice.family, choice.structures = FAM_STAND_ASIDE, []
            choice.rationale = f"{condition} but in the {_DEBIT_BLACKOUT[0].strftime('%H:%M')}-{_DEBIT_BLACKOUT[1].strftime('%H:%M')} blackout — post-open-drive exhaustion zone (debit entries here leak); wait for a fresh leg."
        elif _SKIP_OVERHEAD_PRESSURE and str(metadata.get("option_chain_pressure_state")) == "OVERHEAD_CALL_PRESSURE":
            choice.family, choice.structures = FAM_STAND_ASIDE, []
            choice.rationale = f"{condition} but OVERHEAD_CALL_PRESSURE — capped/grinding tape, put-debit has no clean down-leg here (34% win, -1,333/trade); stand aside."
        elif _ANTICHASE_HOUR_DROP is not None and float(metadata.get("last_hour_change_pct") or 0.0) <= -_ANTICHASE_HOUR_DROP:
            choice.family, choice.structures = FAM_STAND_ASIDE, []
            choice.rationale = f"{condition} but last hour already fell {float(metadata.get('last_hour_change_pct') or 0.0):.2f}% — chasing an extended move (impulse likely spent); stand aside."
        else:
            choice.family, choice.structures = FAM_DIRECTIONAL_DEBIT, ["PUT_DEBIT_SPREAD"]
            choice.rationale = f"{condition}: buy PUT DEBIT with the down-move (Nifty falls sharply — +skew capped-loss captures it)."

    elif condition == RANGE_WIDE:
        # Retrospective lesson: morning condors LOSE (-Rs1,151/24tr @10-12) because the
        # day's range hasn't formed yet; mid-day condors win (+Rs7.6k @70%). Only sell
        # the range once it's established (>= 11:30).
        _too_early = now_time is not None and now_time < time(11, 30)
        if _NO_CONDOR:
            choice.family, choice.structures = FAM_STAND_ASIDE, []
            choice.rationale = "RANGE_WIDE but condor disabled (SEL_NO_CONDOR) — condor P&L is order-noise; stand aside, keep the slot for the robust directional edge."
        elif _too_early:
            choice.family, choice.structures = FAM_STAND_ASIDE, []
            choice.rationale = "RANGE_WIDE but before 11:30 — range not yet formed (morning condors lose); wait."
        elif iv in (IV_RICH, IV_NORMAL):
            # NOTE: the volatility engine is ADVISORY here for now — its rich/cheap read
            # is attached to the decision (choice.vol_regime/vol_notes) but does NOT yet
            # gate the trade. A "stand aside when vol cheap" rule was tested and REJECTED
            # (7mo: +104,858 -> +84,945) — the VRP thresholds need calibration for Nifty
            # before vol drives trades. Kept computing so it can be calibrated on real data.
            choice.family = FAM_PREMIUM_SELL
            choice.structures = ["IRON_CONDOR", "SHORT_STRANGLE"]
            _rich = " (vol RICH)" if vol.regime == RICH_SELL else (" (vol CHEAP — watch)" if vol.regime == CHEAP_BUY else "")
            choice.rationale = f"RANGE_WIDE / mid-day{_rich}: two-sided walls, balanced tape — sell premium inside the formed range."
        else:
            choice.family, choice.structures = FAM_STAND_ASIDE, []
            choice.rationale = "RANGE_WIDE but IV CHEAP: premium too thin to sell; stand aside."

    elif condition == RANGE_TIGHT:
        choice.family, choice.structures = FAM_STAND_ASIDE, []
        choice.rationale = "RANGE_TIGHT: walls close, premium thin, no room — stand aside."

    elif condition == HIGH_VOL_UNDIRECTED:
        # Elevated vol, no direction: either sell rich premium (if truly rangebound)
        # or stay out (whipsaw). Default to stand-aside — this is where credit spreads died.
        choice.family, choice.structures = FAM_STAND_ASIDE, []
        choice.rationale = "HIGH_VOL_UNDIRECTED: elevated vol without direction = whipsaw risk — stand aside."

    else:  # CHOP
        choice.family, choice.structures = FAM_STAND_ASIDE, []
        choice.rationale = f"CHOP: uncommitted tape (bias {read.bias}, conviction {read.conviction:.2f}) — stand aside. This condition was the -Rs73k killer."

    # NOTE: a vol rule "skip debit when IV 12-16" was calibrated from post-hoc buckets
    # (that bucket showed -Rs18,807/25% win) and REJECTED on validation: applying it made
    # 7mo WORSE (+84,945 -> +8,877). Reason = PATH DEPENDENCE — in a one-position system,
    # cutting trades frees the slot for other (worse) trades, so bucket edges don't
    # transfer. Lesson recorded; vol stays advisory until a rule survives full re-sim.
    choice.executable_today = bool(choice.structures) and all(s in _EXECUTABLE for s in choice.structures)
    return choice
