"""Strategy precondition matrix — the "desk brain".

For each option structure the agent can trade, this encodes:
  1. PRECONDITION — the market-data signals that must hold for the structure to have an edge
     (grounded in fields the agent already computes: IV/vol regime, dealer gamma, pin, walls,
     trend quality/efficiency, chain pressure). This is what was missing: the structures existed
     but nothing said WHEN each one works.
  2. FIT SCORE — how well the current tape matches the structure (higher = better fit), so the
     selector can rank the whole menu and pick the best-fit instead of collapsing to one trade.
  3. HOLDING DOCTRINE — grounded in the THETA SIGN, which is the user's key point:
        BUYING premium (debit)  -> theta works AGAINST you  -> HOLD SHORT (close same day / on target)
        SELLING premium (credit)-> theta works FOR you       -> HOLD while the thesis holds (swing / to expiry)
     So the recommended holding period is a property of the structure, refined by the tape.

The IV/vol regime is the FIRST gate, the way a desk decides: RICH premium => SELL structures;
CHEAP premium => BUY structures; NEUTRAL => lean sell (collect theta) but smaller.

This module is pure/metadata-driven and side-effect free so it can be unit-tested and validated
on the shadow book BEFORE it drives any live selection. It does NOT execute or size trades.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# ── Holding doctrine ───────────────────────────────────────────────────────────────────────
HOLD_INTRADAY = "INTRADAY"   # close same day — debit / theta-negative: time is the enemy
HOLD_SWING = "SWING"         # hold 1-few days while the thesis holds — credit / theta-positive
HOLD_EXPIRY = "EXPIRY"       # hold to expiry unless the range/level is breached — defined-risk seller in a range

# Directional families (for readability / grouping)
BULLISH, BEARISH, NEUTRAL = "BULLISH", "BEARISH", "NEUTRAL"


def _f(m: dict[str, Any], k: str, d: float = 0.0) -> float:
    try:
        v = m.get(k)
        return float(v) if v not in (None, "") else d
    except (TypeError, ValueError):
        return d


def _s(m: dict[str, Any], k: str, d: str = "") -> str:
    return str(m.get(k) or d).upper()


def _bias(m: dict[str, Any]) -> str:
    """Net directional lean from the strongest available signals."""
    for k in ("thesis_net_bias", "setup_direction", "smart_money_bias", "oi_pressure_bias"):
        v = _s(m, k)
        if v in (BULLISH, BEARISH):
            return v
    bq, sq = _f(m, "bullish_trend_quality_score"), _f(m, "bearish_trend_quality_score")
    if bq - sq >= 1.5:
        return BULLISH
    if sq - bq >= 1.5:
        return BEARISH
    return NEUTRAL


def _iv_rich(m: dict[str, Any]) -> bool:
    return _s(m, "vol_regime") == "RICH_SELL"


def _iv_cheap(m: dict[str, Any]) -> bool:
    return _s(m, "vol_regime") == "CHEAP_BUY"


def _pinned(m: dict[str, Any]) -> bool:
    """Dealer long-gamma pin — realized vol suppressed, price glued to a strike."""
    return _s(m, "gex_regime") == "LONG_GAMMA" and bool(m.get("pin_risk_active"))


def _ranging(m: dict[str, Any]) -> bool:
    """Balanced walls / low trend efficiency = two-sided range."""
    return (
        _s(m, "option_chain_pressure_state") == "BALANCED_WALLS"
        or _f(m, "trend_efficiency_ratio", 1.0) < 0.30
        or _f(m, "range_balance_score") >= 1.5
    )


def _wall_width_pts(m: dict[str, Any]) -> float:
    put_w, call_w = _f(m, "put_support_strike"), _f(m, "call_resistance_strike")
    return (call_w - put_w) if (put_w and call_w and call_w > put_w) else 0.0


@dataclass
class StratSpec:
    name: str                                  # StrategyType value (or IRON_FLY, not-yet-built)
    family: str                                # BULLISH / BEARISH / NEUTRAL
    is_credit: bool                            # sell premium (theta+) vs buy (theta-)
    defined_risk: bool                         # capped loss (charter) vs naked
    default_hold: str                          # HOLD_INTRADAY / SWING / EXPIRY
    precondition: Callable[[dict], tuple[bool, float, str]]
    built: bool = True                         # has a strike-builder + execution wired today


# ── Preconditions ──────────────────────────────────────────────────────────────────────────
# Each returns (ok, score, why). ok = the structure is eligible; score = fit quality; why = reason.

def _pc_put_debit(m):  # BEAR PUT SPREAD (buy) — the validated down-edge
    ok = (
        _bias(m) == BEARISH
        and not _iv_rich(m)                                   # buying: don't overpay rich premium
        and _f(m, "bearish_trend_quality_score") >= 4.0
        and _f(m, "trend_efficiency_ratio", 0.0) >= 0.40      # a REAL down-move, not a fake
        and not _pinned(m)
    )
    score = _f(m, "bearish_trend_quality_score") + 3 * _f(m, "trend_efficiency_ratio")
    return ok, score, "bearish + real down-move + premium not rich (buy the sharp fall)"


def _pc_call_debit(m):  # BULL CALL SPREAD (buy)
    ok = (
        _bias(m) == BULLISH
        and not _iv_rich(m)
        and _f(m, "bullish_trend_quality_score") >= 4.0
        and _f(m, "trend_efficiency_ratio", 0.0) >= 0.45      # stricter: Nifty up-moves grind
        and not _pinned(m)
    )
    score = _f(m, "bullish_trend_quality_score") + 3 * _f(m, "trend_efficiency_ratio") - 1.0  # de-prioritised: weak up-edge
    return ok, score, "bullish + clean rally + premium not rich (Nifty up-moves grind — needs efficiency)"


def _pc_bull_put(m):  # BULL PUT SPREAD (sell) — theta with an up/sideways lean, support below
    ok = (
        _bias(m) in (BULLISH, NEUTRAL)
        and not _iv_cheap(m)                                 # seller-first: sell unless premium is cheap
        and _f(m, "put_support_strike") > 0                 # a put wall to lean on
        and _s(m, "opening_range_break_state") != "DOWN"    # not actively breaking down
    )
    score = 4.0 + _f(m, "bullish_option_chain_pressure_score")
    return ok, score, "up/neutral + rich IV + put-wall support below (sell puts, collect theta)"


def _pc_bear_call(m):  # BEAR CALL SPREAD (sell) — theta with a down/sideways lean, resistance above
    ok = (
        _bias(m) in (BEARISH, NEUTRAL)
        and not _iv_cheap(m)                                 # seller-first: sell unless premium is cheap
        and (_f(m, "call_resistance_strike") > 0 or _f(m, "overhead_call_pressure_score") >= 1.5)
        and _s(m, "opening_range_break_state") != "UP"
    )
    score = 4.0 + _f(m, "overhead_call_pressure_score") + _f(m, "bearish_option_chain_pressure_score")
    return ok, score, "down/neutral + rich IV + overhead call resistance (sell calls, collect theta)"


def _pc_iron_condor(m):  # SHORT IRON CONDOR (sell) — neutral, WIDE two-sided range, rich IV
    ok = (
        _bias(m) == NEUTRAL
        and _ranging(m)
        and not _iv_cheap(m)                                 # sell unless premium is cheap
        and _wall_width_pts(m) >= 150                        # enough room between walls
        and not _pinned(m)                                   # a tight PIN => iron fly fits better
    )
    score = 3.0 + _f(m, "range_balance_score") + (_wall_width_pts(m) / 100.0)
    return ok, score, "neutral wide range + rich/neutral IV + walls apart (sell condor, harvest range)"


def _pc_iron_fly(m):  # SHORT IRON BUTTERFLY (sell) — a tight PIN, max premium at the ATM. NOT built yet.
    spot_to_pin = abs(_f(m, "spot_to_pin_pts", 999))
    ok = (
        _bias(m) == NEUTRAL
        and _pinned(m)
        and not _iv_cheap(m)
        and spot_to_pin <= 40                               # price glued to the pin/ATM
    )
    score = 5.0 + _f(m, "gamma_concentration") * 10 - spot_to_pin / 40.0
    return ok, score, "tight long-gamma PIN at the ATM + rich/neutral IV (sell the pin, max theta)"


def _pc_short_strangle(m):  # SHORT STRANGLE (sell, naked-ish) — range but wider premium
    ok = (
        _bias(m) == NEUTRAL
        and _ranging(m)
        and _iv_rich(m)                                      # only when premium is genuinely rich (naked risk)
    )
    score = 2.0 + _f(m, "range_balance_score")               # below condor: undefined risk
    return ok, score, "neutral range + RICH IV (wider premium than condor, but naked — needs margin room)"


def _pc_short_straddle(m):  # SHORT STRADDLE (sell, naked) — strong pin, very rich IV. Highest risk.
    ok = (
        _bias(m) == NEUTRAL
        and _pinned(m)
        and _iv_rich(m)
        and abs(_f(m, "spot_to_pin_pts", 999)) <= 25
    )
    score = 1.5 + _f(m, "gamma_concentration") * 8          # lowest priority: naked, defended by iron fly
    return ok, score, "very tight pin + RICH IV (max premium, but naked — iron fly is the safer expression)"


# ── The matrix ─────────────────────────────────────────────────────────────────────────────
SPECS: list[StratSpec] = [
    StratSpec("PUT_DEBIT_SPREAD",        BEARISH, is_credit=False, defined_risk=True,  default_hold=HOLD_INTRADAY, precondition=_pc_put_debit),
    StratSpec("CALL_DEBIT_SPREAD",       BULLISH, is_credit=False, defined_risk=True,  default_hold=HOLD_INTRADAY, precondition=_pc_call_debit),
    StratSpec("BULL_PUT_CREDIT_SPREAD",  BULLISH, is_credit=True,  defined_risk=True,  default_hold=HOLD_SWING,    precondition=_pc_bull_put),
    StratSpec("BEAR_CALL_CREDIT_SPREAD", BEARISH, is_credit=True,  defined_risk=True,  default_hold=HOLD_SWING,    precondition=_pc_bear_call),
    StratSpec("IRON_CONDOR",             NEUTRAL, is_credit=True,  defined_risk=True,  default_hold=HOLD_EXPIRY,   precondition=_pc_iron_condor),
    StratSpec("IRON_FLY",                NEUTRAL, is_credit=True,  defined_risk=True,  default_hold=HOLD_EXPIRY,   precondition=_pc_iron_fly),
    StratSpec("SHORT_STRANGLE",          NEUTRAL, is_credit=True,  defined_risk=False, default_hold=HOLD_EXPIRY,   precondition=_pc_short_strangle),
    StratSpec("SHORT_STRADDLE",          NEUTRAL, is_credit=True,  defined_risk=False, default_hold=HOLD_EXPIRY,   precondition=_pc_short_straddle),
]


def recommend_hold(spec: StratSpec, m: dict[str, Any]) -> str:
    """Refine the holding doctrine from the tape. Sellers hold longer when the range is stable and
    IV is elevated (theta keeps paying); buyers always close intraday (theta bleeds them)."""
    if not spec.is_credit:
        return HOLD_INTRADAY                                 # debit: never hold — theta is the enemy
    # Credit/seller: hold for theta while the thesis holds. A pinned long-gamma range with rich IV is
    # the strongest 'keep holding' signal; a wide-but-choppy range is a shorter swing.
    if _pinned(m) or _wall_width_pts(m) >= 250:
        return HOLD_EXPIRY
    return spec.default_hold


def evaluate_matrix(m: dict[str, Any]) -> list[dict[str, Any]]:
    """Score every structure's precondition against the tape; return the eligible ones, best-first."""
    out = []
    for spec in SPECS:
        ok, score, why = spec.precondition(m)
        if ok:
            out.append({
                "strategy": spec.name, "family": spec.family, "is_credit": spec.is_credit,
                "defined_risk": spec.defined_risk, "built": spec.built,
                "hold": recommend_hold(spec, m), "score": round(score, 3), "why": why,
            })
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def select_from_matrix(m: dict[str, Any], *, defined_risk_only: bool = True,
                       built_only: bool = True, seller_first: bool = True,
                       strong_buy_score: float = 9.0, buy_beat_margin: float = 2.0) -> dict[str, Any]:
    """Pick the best-fit structure from the whole menu. Filters to defined-risk / built structures by
    default (the charter). Returns a stand-aside verdict when nothing fits.

    SELLER-FIRST (default): the agent's primary skill is option SELLING. Default to the best-fit
    CREDIT structure; only choose a DEBIT (buying) when the buy signal is STRONG (score >=
    strong_buy_score) AND clearly beats the best credit (by >= buy_beat_margin). i.e. "sell premium
    unless the agent gets a strong signal to buy." Set seller_first=False for a symmetric ranking.
    """
    ranked = evaluate_matrix(m)
    if defined_risk_only:
        ranked = [r for r in ranked if r["defined_risk"]]
    if built_only:
        ranked = [r for r in ranked if r["built"]]
    if not ranked:
        return {"strategy": "STAND_ASIDE", "hold": None, "score": 0.0,
                "why": "no structure's preconditions are met on this tape", "alternatives": []}

    if seller_first:
        credits = [r for r in ranked if r["is_credit"]]
        debits = [r for r in ranked if not r["is_credit"]]
        best_credit = credits[0] if credits else None
        best_debit = debits[0] if debits else None
        if best_credit is None:
            best = best_debit                      # nothing to sell — a strong buy stands on its own
        elif (best_debit is not None and best_debit["score"] >= strong_buy_score
              and best_debit["score"] >= best_credit["score"] + buy_beat_margin):
            best = best_debit                      # STRONG, clearly-superior buy signal — override to buy
            best["why"] = "STRONG buy signal overrides seller-first — " + best["why"]
        else:
            best = best_credit                     # default stance: SELL premium, collect theta
    else:
        best = ranked[0]

    best["alternatives"] = [r for r in ranked if r["strategy"] != best["strategy"]][:3]
    return best
