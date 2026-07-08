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

from dataclasses import dataclass, field
from typing import Any

from .trade_planner import assess_market, MarketRead, _f

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

# Executable today vs needs-build (so the build order is explicit and honest)
_EXECUTABLE = {"BEAR_CALL_CREDIT_SPREAD", "BULL_PUT_CREDIT_SPREAD", "IRON_CONDOR", "SHORT_STRANGLE", "SHORT_STRADDLE"}


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


def select_strategy(metadata: dict[str, Any], spot: float) -> StrategyChoice:
    """Read the market, name the condition, and choose the strategy family + structures."""
    read = assess_market(metadata, spot)
    condition = classify_condition(read, metadata)
    iv = _iv_regime(metadata)
    choice = StrategyChoice(condition=condition, iv_regime=iv, read=read, conviction=read.conviction)

    if condition in (STRONG_TREND_UP, BREAKOUT_UP):
        # Trend up: buy a call debit (positive skew) in cheap IV, or sell a bull-put
        # (with-trend theta) when premium is rich.
        if iv == IV_CHEAP:
            choice.family, choice.structures = FAM_DIRECTIONAL_DEBIT, ["CALL_DEBIT_SPREAD"]
        else:
            choice.family, choice.structures = FAM_DIRECTIONAL_CREDIT, ["BULL_PUT_CREDIT_SPREAD"]
        choice.rationale = f"{condition} / IV {iv}: trade WITH the up-move; {'buy call debit (cheap premium, +skew)' if iv==IV_CHEAP else 'sell bull-put below support (rich premium, theta)'}."

    elif condition in (STRONG_TREND_DOWN, BREAKOUT_DOWN):
        if iv == IV_CHEAP:
            choice.family, choice.structures = FAM_DIRECTIONAL_DEBIT, ["PUT_DEBIT_SPREAD"]
        else:
            choice.family, choice.structures = FAM_DIRECTIONAL_CREDIT, ["BEAR_CALL_CREDIT_SPREAD"]
        choice.rationale = f"{condition} / IV {iv}: trade WITH the down-move; {'buy put debit (cheap premium, +skew)' if iv==IV_CHEAP else 'sell bear-call above resistance (rich premium, theta)'}."

    elif condition == RANGE_WIDE:
        # Sell premium into a defined range only when it's rich.
        if iv in (IV_RICH, IV_NORMAL):
            choice.family = FAM_PREMIUM_SELL
            choice.structures = ["IRON_CONDOR", "SHORT_STRANGLE"]
            choice.rationale = f"RANGE_WIDE / IV {iv}: two-sided walls, balanced tape — sell premium inside the range."
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

    choice.executable_today = bool(choice.structures) and all(s in _EXECUTABLE for s in choice.structures)
    return choice
