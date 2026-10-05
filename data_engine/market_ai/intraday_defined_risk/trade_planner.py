"""
Trade Planner — market-first reasoning layer.

The agent already computes every institutional signal (OI walls + their velocity,
smart-money bias, PCR + PCR trend, FII writing/unwinding, VWAP, order flow, opening
range, gamma/pin, multi-timeframe S/R). Historically those signals fed a rigid
cascade of pre-coded playbooks, each guarded by ~20 independent veto gates — so the
agent "force-fit" a coded strategy instead of deciding one.

This module inverts that: it READS the market into a coherent view, then DECIDES a
trade plan from what the market is actually doing — direction, structure, strikes,
entry trigger, stop, target. Structures are chosen to fit the read, not the other
way round:

  - Clear directional lean, defined-risk  -> credit spread away from the wall
  - Strong directional lean + rich premium -> naked sell with a structural stop
                                              (only when allow_naked and conviction high)
  - Two-sided institutional walls          -> iron condor / strangle inside the range
  - No edge                                -> NO_TRADE (and say why)

It is pure, deterministic, and fast, so it runs every cycle live AND replays through
the backtest harness for validation. No per-cycle LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Day-type / bias vocabulary ──────────────────────────────────────────────
TREND_UP = "TRENDING_UP"
TREND_DOWN = "TRENDING_DOWN"
RANGE_BOUND = "RANGE_BOUND"
VOLATILE = "VOLATILE"
UNCLEAR = "UNCLEAR"

BULLISH = "BULLISH"
BEARISH = "BEARISH"
NEUTRAL = "NEUTRAL"

# ── Structures the planner may choose ───────────────────────────────────────
BEAR_CALL_SPREAD = "BEAR_CALL_CREDIT_SPREAD"
BULL_PUT_SPREAD = "BULL_PUT_CREDIT_SPREAD"
IRON_CONDOR = "IRON_CONDOR"
SHORT_STRANGLE = "SHORT_STRANGLE"
NAKED_CALL = "NAKED_CALL"
NAKED_PUT = "NAKED_PUT"
NO_STRUCTURE = "NONE"

# Conviction thresholds
_MIN_DIRECTIONAL_CONVICTION = 0.62   # below this, no directional trade
_MIN_RANGE_CONVICTION = 0.60
_NAKED_CONVICTION = 0.80             # naked selling requires very strong read
_MIN_SIGNAL_CONSENSUS = 4            # at least N aligned signals (robustness)


def _has_bear_trigger(m: dict) -> bool:
    """A professional shorts on a TRIGGER, not just a bearish lean: a failed
    reclaim, a confirmed breakdown, a rejection at resistance, or an OR breakdown."""
    return bool(
        m.get("failed_reclaim")
        or m.get("accepted_breakdown")
        or m.get("bearish_rejection_transition")
        or m.get("lower_high_confirmed")
        or str(m.get("opening_range_break_state") or "") in {"DOWN", "FAILED_UP"}
    )


def _has_bull_trigger(m: dict) -> bool:
    """Go long premium-selling on a trigger: confirmed breakout, higher-low
    reclaim, failed breakdown, or an OR breakout up."""
    return bool(
        m.get("accepted_breakout")
        or m.get("higher_low_confirmed")
        or m.get("failed_breakdown")
        or str(m.get("opening_range_break_state") or "") in {"UP", "FAILED_DOWN"}
    )


@dataclass
class MarketRead:
    """A professional's read of the session, derived from live signals."""
    day_type: str = UNCLEAR
    bias: str = NEUTRAL
    conviction: float = 0.0            # 0..1
    who_controls: str = "BALANCED"     # BULLS / BEARS / BALANCED
    oi_narrative: str = ""             # human-readable OI story
    support: float | None = None
    resistance: float | None = None
    put_wall: float | None = None
    call_wall: float | None = None
    bull_signals: int = 0
    bear_signals: int = 0
    range_signals: int = 0
    signal_detail: dict[str, str] = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class TradePlan:
    should_trade: bool = False
    direction: str = NEUTRAL
    structure: str = NO_STRUCTURE
    short_strike: float | None = None
    long_strike: float | None = None
    width_points: float | None = None
    entry_trigger: str = ""            # what must be true to enter
    stop_loss: str = ""                # structural stop description
    target: str = ""                   # profit objective
    conviction: float = 0.0
    is_naked: bool = False
    rationale: list[str] = field(default_factory=list)
    read: MarketRead | None = None


def _f(meta: dict, key: str, default: float = 0.0) -> float:
    v = meta.get(key)
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


# ── Step 1: read the market ─────────────────────────────────────────────────
def assess_market(metadata: dict[str, Any], spot: float) -> MarketRead:
    """Synthesize the full signal set into a coherent market read."""
    m = metadata
    detail: dict[str, str] = {}
    bull = bear = rng = 0

    # Price vs VWAP — who is in control intraday
    pvv = _f(m, "price_vs_vwap", 0.0)
    if pvv > 8:
        bull += 1; detail["vwap"] = "above (bull)"
    elif pvv < -8:
        bear += 1; detail["vwap"] = "below (bear)"
    else:
        rng += 1; detail["vwap"] = "at (balanced)"

    # Opening-range break
    orb = str(m.get("opening_range_break_state") or "NONE")
    if orb == "UP":
        bull += 1; detail["or_break"] = "broke up"
    elif orb in {"DOWN", "FAILED_UP"}:
        bear += 1; detail["or_break"] = "broke down / failed up"
    else:
        rng += 1; detail["or_break"] = "inside range"

    # Order-flow imbalance (aggression)
    ofi = _f(m, "order_flow_imbalance", 0.0)
    if ofi >= 0.15:
        bull += 1; detail["order_flow"] = "call-side aggression"
    elif ofi <= -0.15:
        bear += 1; detail["order_flow"] = "put-side aggression"
    else:
        detail["order_flow"] = "balanced"

    # Smart-money / institutional OI bias
    smb = str(m.get("smart_money_bias") or "NEUTRAL")
    if smb == "BULLISH":
        bull += 1; detail["smart_money"] = "bullish OI flow"
    elif smb == "BEARISH":
        bear += 1; detail["smart_money"] = "bearish OI flow"
    else:
        detail["smart_money"] = "neutral"

    # FII positioning (T-1)
    fii = str(m.get("fii_bias") or "NEUTRAL")
    if fii in {"BULLISH", "STRONG_BULLISH"}:
        bull += 1; detail["fii"] = "net long"
    elif fii in {"BEARISH", "STRONG_BEARISH"}:
        bear += 1; detail["fii"] = "net short"
    else:
        detail["fii"] = "neutral"

    # PCR + PCR trend — put writing vs call writing
    pcr = _f(m, "pcr_by_oi", 1.0)
    pcr_trend = str(m.get("pcr_trend") or "UNKNOWN")
    if pcr >= 1.15 or pcr_trend == "RISING":
        bull += 1; detail["pcr"] = f"put writing (pcr {pcr:.2f}, {pcr_trend.lower()})"
    elif pcr <= 0.80 or pcr_trend == "FALLING":
        bear += 1; detail["pcr"] = f"call writing / put unwind (pcr {pcr:.2f}, {pcr_trend.lower()})"
    else:
        rng += 1; detail["pcr"] = f"balanced (pcr {pcr:.2f})"

    # Walls (prefer multi-session, fall back to current)
    call_wall = m.get("multi_session_call_wall") or m.get("current_call_wall")
    put_wall = m.get("multi_session_put_wall") or m.get("current_put_wall")
    call_wall = float(call_wall) if call_wall is not None else None
    put_wall = float(put_wall) if put_wall is not None else None

    # Wall migration = OI building or unwinding at the walls
    call_v = _f(m, "call_wall_oi_velocity", 0.0)
    put_v = _f(m, "put_wall_oi_velocity", 0.0)
    wm = str(m.get("wall_migration_bias") or "NEUTRAL")
    if wm == "BULLISH":
        bull += 1; detail["wall_migration"] = "walls shifting up"
    elif wm == "BEARISH":
        bear += 1; detail["wall_migration"] = "walls shifting down"

    # 15m trend quality
    btq = _f(m, "bullish_trend_quality_score", 0.0)
    xtq = _f(m, "bearish_trend_quality_score", 0.0)
    if btq >= 3.0 and btq > xtq:
        bull += 1; detail["trend_quality"] = "bullish structure"
    elif xtq >= 3.0 and xtq > btq:
        bear += 1; detail["trend_quality"] = "bearish structure"
    else:
        rng += 1; detail["trend_quality"] = "no clear trend"

    total = bull + bear + rng

    # OI narrative — the professional's one-liner on positioning
    narrative_bits = []
    if put_wall is not None:
        pw_word = "writing" if (put_v > 0 or pcr_trend == "RISING") else ("unwinding" if put_v < 0 else "holding")
        narrative_bits.append(f"puts {pw_word} at {put_wall:.0f}")
    if call_wall is not None:
        cw_word = "writing" if (call_v > 0 or pcr_trend == "FALLING") else ("unwinding" if call_v < 0 else "holding")
        narrative_bits.append(f"calls {cw_word} at {call_wall:.0f}")
    oi_narrative = "; ".join(narrative_bits) or "no dominant OI wall"

    # Decide bias + conviction from consensus
    lead = max(bull, bear, rng)
    if bull == lead and bull - bear >= 2:
        bias = BULLISH; who = "BULLS"; strong = bull
    elif bear == lead and bear - bull >= 2:
        bias = BEARISH; who = "BEARS"; strong = bear
    elif rng >= max(bull, bear) and abs(bull - bear) <= 1:
        bias = NEUTRAL; who = "BALANCED"; strong = rng
    else:
        bias = NEUTRAL; who = "BALANCED"; strong = rng

    conviction = round(min(0.95, 0.35 + 0.11 * (strong - 2)), 3) if total else 0.0
    conviction = max(0.0, conviction)

    # Day type
    if bias == BULLISH and (btq >= 3.0 or orb == "UP"):
        day_type = TREND_UP
    elif bias == BEARISH and (xtq >= 3.0 or orb in {"DOWN", "FAILED_UP"}):
        day_type = TREND_DOWN
    elif bias == NEUTRAL and call_wall is not None and put_wall is not None and (call_wall - put_wall) >= 150:
        day_type = RANGE_BOUND
    elif _f(m, "india_vix", 0.0) >= 20 or _f(m, "rv30_pct", 0.0) >= 1.2:
        day_type = VOLATILE
    else:
        day_type = UNCLEAR if bias == NEUTRAL else (TREND_UP if bias == BULLISH else TREND_DOWN)

    # Levels
    resistance = m.get("daily_swing_resistance") or (call_wall if call_wall else None)
    support = m.get("daily_swing_support") or (put_wall if put_wall else None)
    resistance = float(resistance) if resistance is not None else None
    support = float(support) if support is not None else None

    reasoning = (
        f"{day_type} / {bias} (conviction {conviction:.2f}, {who} in control). "
        f"Signals {bull}B/{bear}Be/{rng}R. OI: {oi_narrative}."
    )

    return MarketRead(
        day_type=day_type, bias=bias, conviction=conviction, who_controls=who,
        oi_narrative=oi_narrative, support=support, resistance=resistance,
        put_wall=put_wall, call_wall=call_wall,
        bull_signals=bull, bear_signals=bear, range_signals=rng,
        signal_detail=detail, reasoning=reasoning,
    )


# ── Step 2: form the trade plan from the read ───────────────────────────────
def form_trade_plan(
    read: MarketRead,
    metadata: dict[str, Any],
    spot: float,
    *,
    allow_naked: bool = False,
) -> TradePlan:
    """Turn the market read into a concrete, executable plan."""
    plan = TradePlan(read=read, conviction=read.conviction)
    consensus = max(read.bull_signals, read.bear_signals, read.range_signals)

    # ── BEARISH: sell call premium above resistance ─────────────────────────
    if (
        read.bias == BEARISH
        and read.conviction >= _MIN_DIRECTIONAL_CONVICTION
        and read.bear_signals >= _MIN_SIGNAL_CONSENSUS
        and _has_bear_trigger(metadata)
    ):
        anchor = read.resistance or read.call_wall or (spot + 100)
        short = max(anchor, spot + 60)
        naked = allow_naked and read.conviction >= _NAKED_CONVICTION and read.call_wall is not None
        plan.should_trade = True
        plan.direction = BEARISH
        plan.structure = NAKED_CALL if naked else BEAR_CALL_SPREAD
        plan.is_naked = naked
        plan.short_strike = short
        plan.long_strike = None if naked else short + 100
        plan.width_points = None if naked else 100.0
        plan.entry_trigger = f"price rejects {read.resistance or read.call_wall:.0f} / stays below VWAP" if (read.resistance or read.call_wall) else "price stays below VWAP"
        plan.stop_loss = f"spot closes above {short:.0f} (short strike) on 5m" if naked else "short delta > 0.40 or spot breaks above resistance"
        plan.target = "50-65% of credit captured, or theta decay into close"
        plan.rationale = [
            f"Bears in control ({read.bear_signals} signals). {read.oi_narrative}.",
            f"Sell call premium above {short:.0f} — market is capped by resistance/call wall.",
            "Naked short call (high conviction + defined stop)." if naked else "Defined-risk bear-call spread (100pt wing).",
        ]
        return plan

    # ── BULLISH: sell put premium below support ─────────────────────────────
    if (
        read.bias == BULLISH
        and read.conviction >= _MIN_DIRECTIONAL_CONVICTION
        and read.bull_signals >= _MIN_SIGNAL_CONSENSUS
        and _has_bull_trigger(metadata)
    ):
        anchor = read.support or read.put_wall or (spot - 100)
        short = min(anchor, spot - 60)
        naked = allow_naked and read.conviction >= _NAKED_CONVICTION and read.put_wall is not None
        plan.should_trade = True
        plan.direction = BULLISH
        plan.structure = NAKED_PUT if naked else BULL_PUT_SPREAD
        plan.is_naked = naked
        plan.short_strike = short
        plan.long_strike = None if naked else short - 100
        plan.width_points = None if naked else 100.0
        plan.entry_trigger = f"price holds above {read.support or read.put_wall:.0f} / reclaims VWAP" if (read.support or read.put_wall) else "price holds above VWAP"
        plan.stop_loss = f"spot closes below {short:.0f} (short strike) on 5m" if naked else "short delta > 0.40 or spot breaks below support"
        plan.target = "50-65% of credit captured, or theta decay into close"
        plan.rationale = [
            f"Bulls in control ({read.bull_signals} signals). {read.oi_narrative}.",
            f"Sell put premium below {short:.0f} — market is supported by put wall/support.",
            "Naked short put (high conviction + defined stop)." if naked else "Defined-risk bull-put spread (100pt wing).",
        ]
        return plan

    # ── RANGE: sell both sides inside the institutional walls ───────────────
    if (
        read.bias == NEUTRAL
        and read.conviction >= _MIN_RANGE_CONVICTION
        and read.call_wall is not None
        and read.put_wall is not None
        and (read.call_wall - read.put_wall) >= 150
    ):
        straddle = _f(metadata, "atm_straddle_price", 0.0)
        rich_premium = straddle >= 100 or _f(metadata, "india_vix", 0.0) >= 15
        plan.should_trade = True
        plan.direction = NEUTRAL
        plan.structure = SHORT_STRANGLE if (allow_naked and rich_premium) else IRON_CONDOR
        plan.is_naked = plan.structure == SHORT_STRANGLE
        plan.short_strike = None  # two-sided; handled by condor anchors
        plan.width_points = 100.0
        plan.entry_trigger = f"price oscillating between {read.put_wall:.0f} and {read.call_wall:.0f}"
        plan.stop_loss = "either short side delta > 0.30, or spot breaks a wall on 5m"
        plan.target = "theta decay; close at 50% of combined credit"
        plan.rationale = [
            f"Balanced tape, institutional walls at {read.put_wall:.0f}/{read.call_wall:.0f} ({read.call_wall - read.put_wall:.0f}pt range). {read.oi_narrative}.",
            f"Sell both sides inside the range — {'strangle (rich premium)' if plan.is_naked else 'defined-risk condor'}.",
        ]
        return plan

    # ── No edge ─────────────────────────────────────────────────────────────
    plan.should_trade = False
    plan.structure = NO_STRUCTURE
    plan.rationale = [
        f"No tradeable edge: {read.reasoning}",
        f"Consensus {consensus} signals, conviction {read.conviction:.2f} — below thresholds. Stay flat.",
    ]
    return plan


def plan_trade(metadata: dict[str, Any], spot: float, *, allow_naked: bool = False) -> TradePlan:
    """Convenience: read the market and form a plan in one call."""
    read = assess_market(metadata, spot)
    return form_trade_plan(read, metadata, spot, allow_naked=allow_naked)
