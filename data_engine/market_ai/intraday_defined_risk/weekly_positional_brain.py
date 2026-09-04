"""WEEKLY POSITIONAL BRAIN — the decision core for defined-risk weekly option selling.

Two pure decisions, encoding the design + the researched management framework:

  plan_entry(m)              -> what to sell this week (or stand aside), from the BROADER (daily/weekly)
                                trend + support/resistance + premium edge. Direction first, then structure.
  evaluate_management(pos,c) -> the DEFENSE LADDER: how to react as the week unfolds. This is where a
                                positional seller actually makes money — management, not entry.

Management framework (sourced): close winners at 50% of credit (tastytrade's 4,872-trade study — the
last 50% of profit takes ~70% of the time and carries the most gamma); cap losers at 2x credit; exit
before the final-days gamma cliff; and when a short is tested, walk a ladder that ALWAYS adjusts for a
CREDIT and ALWAYS stays defined-risk:

  Rung 1  short tested (delta ~>= 0.30)      -> LEG INTO A CONDOR: sell the opposite-side spread for a
                                               credit (the 'roll the untested side' mechanic) — offsets
                                               the tested loss and widens its breakeven, no new capital.
  Rung 2  short breached, time left, trend ok-> ROLL OUT to next week's expiry at same/lower short, for
                                               a net credit only.
  Rung 3  thesis broken / loss >= 2x / no    -> CLOSE. Take the defined loss. Never roll to a debit,
          credit available to roll             never fight a flipped daily trend, never go naked.

Everything here is PURE + deterministic so it can be unit-tested and shadow-run before a rupee is real.
The executor (live chain, strikes, orders, credit-availability for a roll) wraps these decisions.
"""
from __future__ import annotations

import os
from typing import Any

BULLISH, BEARISH, NEUTRAL = "BULLISH", "BEARISH", "NEUTRAL"


def _f(m: dict, k: str, d: float = 0.0) -> float:
    try:
        v = m.get(k)
        return float(v) if v is not None else d
    except (TypeError, ValueError):
        return d


def _s(m: dict, k: str, d: str = "") -> str:
    v = m.get(k)
    return str(v).upper() if v is not None else d


def _env_f(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


# ── Tunables (env-overridable) ───────────────────────────────────────────────────────────────
PROFIT_TARGET_FRAC   = _env_f("WK_PROFIT_TARGET_FRAC", 0.50)   # close at 50% of the credit
STOP_LOSS_MULT       = _env_f("WK_STOP_LOSS_MULT", 2.0)        # close if loss reaches 2x the credit
GAMMA_CLIFF_DAYS     = _env_f("WK_GAMMA_CLIFF_DAYS", 2.0)      # exit this many trading days before expiry
DEFENSE_SHORT_DELTA  = _env_f("WK_DEFENSE_SHORT_DELTA", 0.30)  # a short leg tested when |delta| >= this
MIN_CREDIT_RATIO     = _env_f("WK_MIN_CREDIT_RATIO", 0.10)     # net credit / wing width floor (worth the risk)
MAX_ADJUSTMENTS      = int(_env_f("WK_MAX_ADJUSTMENTS", 2))    # after this many rolls, stop defending -> close
UNTESTED_HARVEST_PCT = _env_f("WK_UNTESTED_HARVEST_PCT", 0.80) # close the untested side once it has decayed this far
# Gap-safety comes from FOUR layers — trend-alignment (primary), the 0.18-delta short (~82% OTM), this
# expected-move floor, and the defined-risk wing. The short is placed BY DELTA (see the executor); this
# is only the MINIMUM distance so it can't sit too close. A full extra ATR on top double-counted with
# delta and pushed the short so far OTM it collected no premium (2026-09-03: credit ratio 0.01).
GAP_ATR_BUFFER       = _env_f("WK_GAP_ATR_BUFFER", 0.0)        # short beyond expected_move + this * daily ATR


# ── Broader trend (the HIGHER timeframe read — daily/weekly, not intraday) ────────────────────
def broader_trend(m: dict[str, Any]) -> str:
    """The weekly bias comes from the DAILY/WEEKLY structure: daily trend label, daily EMA20/50
    alignment, and price vs the weekly range. 'Understand the broader trend before choosing the
    strategy.' Bullish only above the daily EMAs; bearish only below; else neutral."""
    dt = _s(m, "daily_trend")
    if dt in (BULLISH, BEARISH):
        base = dt
    else:
        base = NEUTRAL
    spot = _f(m, "atm_strike") or _f(m, "spot_price") or _f(m, "spot")
    e20, e50 = _f(m, "daily_ema20"), _f(m, "daily_ema50")
    ema_vote = NEUTRAL
    if spot > 0 and e20 > 0:
        above20 = spot >= e20
        above50 = spot >= e50 if e50 > 0 else above20
        if above20 and above50:
            ema_vote = BULLISH
        elif not above20 and not above50:
            ema_vote = BEARISH
    # The daily EMA is the trend filter: the label must agree with it, else neutral (don't fight it).
    if base == NEUTRAL:
        return ema_vote
    if ema_vote in (NEUTRAL, base):
        return base
    return NEUTRAL


def _premium_ok(m: dict[str, Any]) -> bool:
    """Only sell when premium carries a genuine edge: VRP positive (RICH_SELL) or at least not cheap.
    A thin weekly credit is not worth the overnight gap risk."""
    vr = _s(m, "vol_regime")
    if vr == "RICH_SELL":
        return True
    if vr == "CHEAP_BUY":
        return False
    return _f(m, "atm_chain_iv", 0.0) > 0  # NEUTRAL vol: allow, the credit-ratio gate still applies downstream


# ── Entry: what to sell this week (or stand aside) ────────────────────────────────────────────
def plan_entry(m: dict[str, Any]) -> dict[str, Any]:
    """Direction first (broader trend), then the defined-risk structure that fits it. Bullish ->
    bull-put below support; bearish -> bear-call above resistance; neutral -> iron condor. Stand aside
    only when premium has no edge. Strikes/credit are finalized by the executor against the live chain;
    this returns the intent + the gap-safe strike guidance."""
    if not _premium_ok(m):
        return {"action": "STAND_ASIDE", "why": "premium has no selling edge this week (VRP not positive)"}

    bt = broader_trend(m)
    structure = {BULLISH: "BULL_PUT_CREDIT_SPREAD",
                 BEARISH: "BEAR_CALL_CREDIT_SPREAD",
                 NEUTRAL: "IRON_CONDOR"}[bt]

    # Gap-safe short distance: beyond the weekly expected move PLUS a daily-ATR buffer, so a typical
    # overnight gap doesn't breach the short. The executor snaps to the nearest OI wall past this.
    expected_move = _f(m, "expected_move_pts")
    atr = _f(m, "daily_atr")
    min_short_distance = expected_move + GAP_ATR_BUFFER * atr

    return {
        "action": "TRADE",
        "structure": structure,
        "direction": bt,
        "hold": "EXPIRY",                       # positional: hold for weekly theta, managed by the ladder
        "min_short_distance_pts": round(min_short_distance, 0),
        "min_credit_ratio": MIN_CREDIT_RATIO,   # executor rejects the week if the live credit can't clear this
        "why": (f"broader trend {bt} -> sell {structure} "
                f"{'below support' if bt==BULLISH else 'above resistance' if bt==BEARISH else 'both walls'}; "
                f"short >= {min_short_distance:.0f}pts out (expected move {expected_move:.0f} + {GAP_ATR_BUFFER}xATR "
                f"{atr:.0f}) to survive a gap."),
    }


# ── Management: the defense ladder ────────────────────────────────────────────────────────────
def _trend_flipped(direction: str, current_bias: str) -> bool:
    """The daily thesis is broken when the current broader trend is the OPPOSITE of the position."""
    return (direction == BULLISH and current_bias == BEARISH) or (direction == BEARISH and current_bias == BULLISH)


def evaluate_management(position: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Walk the mature defense ladder. Returns {action, reason, tested_side?}. Actions:
        HOLD, TAKE_PROFIT, STOP_CLOSE, GAMMA_EXIT, CLOSE_THESIS_BROKEN,
        HARVEST_UNTESTED, LEG_INTO_CONDOR, ROLL_OUT, CLOSE_MAX_ADJUSTED.

    Maturity: thresholds use the CUMULATIVE credit (initial + every adjustment credit), so each defensive
    roll that collects credit correctly widens the profit target and the loss cap. Adjustments are
    CAPPED (after MAX_ADJUSTMENTS rolls, stop defending and close), and a condor's untested side is
    harvested once it decays to near-worthless (locks that credit, frees the risk).

    position: {direction, structure, credit_rupees, total_credit_rupees?, adjusted?, adjustments?}
    current : {mtm_rupees, max_short_delta, days_to_expiry, broader_trend,
               tested_side('CALL'/'PUT')?, untested_decay_pct?}
    """
    credit = _f(position, "total_credit_rupees") or _f(position, "credit_rupees")   # CUMULATIVE
    direction = _s(position, "direction")
    already_condor = bool(position.get("adjusted")) or _s(position, "structure") == "IRON_CONDOR"
    adjustments = int(position.get("adjustments") or 0)
    mtm = _f(current, "mtm_rupees")
    short_delta = abs(_f(current, "max_short_delta"))
    dte = _f(current, "days_to_expiry")
    cur_bias = _s(current, "broader_trend")
    untested_decay = _f(current, "untested_decay_pct")

    # 1) WINNER — bank at 50% of the CUMULATIVE credit (the last 50% isn't worth the gamma/time).
    if credit > 0 and mtm >= PROFIT_TARGET_FRAC * credit:
        return {"action": "TAKE_PROFIT", "reason": f"hit {PROFIT_TARGET_FRAC:.0%} of credit ({mtm:+,.0f} of {credit:,.0f})"}

    # 2) GAMMA CLIFF — never hold a weekly into the final days.
    if dte <= GAMMA_CLIFF_DAYS:
        return {"action": "GAMMA_EXIT", "reason": f"{dte:.0f} trading days to expiry — off before the gamma cliff"}

    # 3) HARD STOP — loss reached 2x the CUMULATIVE credit.
    if credit > 0 and mtm <= -STOP_LOSS_MULT * credit:
        return {"action": "STOP_CLOSE", "reason": f"loss hit {STOP_LOSS_MULT:.0f}x credit ({mtm:+,.0f} vs {credit:,.0f})"}

    # 4) THESIS BROKEN — the daily trend flipped against the position: don't fight it, take the loss.
    if _trend_flipped(direction, cur_bias):
        return {"action": "CLOSE_THESIS_BROKEN", "reason": f"daily trend flipped to {cur_bias} against a {direction} position"}

    # 5) HARVEST the untested side of a condor once it is near-worthless — lock the credit, drop the risk.
    if already_condor and untested_decay >= UNTESTED_HARVEST_PCT:
        return {"action": "HARVEST_UNTESTED",
                "reason": f"untested side decayed {untested_decay:.0%} (>= {UNTESTED_HARVEST_PCT:.0%}) — buy it back cheap, lock the credit"}

    # 6) SHORT TESTED — the credit-only, defined-risk defense, now CAPPED.
    if short_delta >= DEFENSE_SHORT_DELTA:
        if adjustments >= MAX_ADJUSTMENTS:
            return {"action": "CLOSE_MAX_ADJUSTED",
                    "reason": f"short tested (delta {short_delta:.2f}) after {adjustments} adjustments — stop "
                              f"defending, take the defined loss (don't roll forever)"}
        if not already_condor:
            return {"action": "LEG_INTO_CONDOR", "tested_side": _s(current, "tested_side"),
                    "reason": f"short tested (delta {short_delta:.2f}) — sell the opposite spread for a credit "
                              f"(widen the tested breakeven, stay defined-risk)"}
        return {"action": "ROLL_OUT", "tested_side": _s(current, "tested_side"),
                "reason": f"short tested (delta {short_delta:.2f}), adjustment {adjustments+1}/{MAX_ADJUSTMENTS} — "
                          f"roll the tested spread out to next week for a NET CREDIT ONLY (else close)"}

    # 7) Otherwise let theta work.
    return {"action": "HOLD", "reason": f"thesis intact (short delta {short_delta:.2f}, mtm {mtm:+,.0f}) — hold for decay"}
