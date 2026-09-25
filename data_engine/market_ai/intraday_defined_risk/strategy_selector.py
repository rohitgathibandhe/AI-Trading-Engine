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
# VWAP-SELLER routing (validated on real option fills, 2026-09-15): read the regime the VWAP/OR way and
# SELL with it — TREND -> directional credit with the trend, RANGE -> iron-fly. Selling the regime read
# made +5,475 on the shadow fills; buying it lost -2,015. Env-gated (default on) so it forward-tests on
# paper. Only overrides on a CONFIDENT trend; RANGE falls through to the guarded range/fly logic.
_VWAP_SELLER = os.environ.get("SEL_VWAP_SELLER", "1") == "1"
_VWAP_TREND_EFF = float(os.environ.get("SEL_VWAP_TREND_EFF", "0.35") or 0.35)  # efficiency for a confident trend

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

# Selection signal (VALIDATED, default ON): skip put-debit when smart_money_bias==BULLISH — buying
# puts against institutional bullish OI flow (screen: 6 trades, 0% win, -15.2k). Ordering-robust
# (dense 7mo, on top of overhead+anti-chase): median 186,848 -> 192,058, floor 147,961 -> 157,194
# (+9k), variance tighter (swing 60k->50k). Small sample (6) but 0% win + clean mechanism + robust.
# SEL_SKIP_SMB_BULLISH=0 disables.
_SKIP_SMB_BULLISH = os.environ.get("SEL_SKIP_SMB_BULLISH", "1") == "1"
# state_confidence floor: screen showed low-third loses, but no trade falls below 0.55/0.65 so a
# safe threshold can't bind — NOT actionable, left env-gated off. SEL_MIN_STATE_CONF to research.
_MIN_STATE_CONF = _sel_env_float("SEL_MIN_STATE_CONF")

# ── SELLER-FIRST (default ON) ────────────────────────────────────────────────────────────────
# Until now a down-read mapped unconditionally to PUT_DEBIT — the selector never COMPARED
# structures, it looked one up. That made the agent a permanent option BUYER on every bearish day,
# paying theta for hours. The live shadow book (real fills, same tape) says that is the wrong
# default: over 7 days put_debit is the ONLY net-negative structure (-416) while every
# theta-COLLECTING structure is green (bear_call +780, short_strangle +3,022, iron_fly +5,529,
# short_straddle +10,984).
#
# Measured on the agent's own trade (2026-07-23): spot made a LOWER LOW at 14:51 (23,814) than at
# 13:15 (23,829) and the position was worth LESS. The move went further in our favour and we earned
# less — that is theta, and no exit rule can fix it. It is a structure problem.
#
# So: SELL the bearish view by default, and BUY the debit only on a genuinely efficient trend,
# because a debit needs follow-through to beat its own decay. trend_efficiency_ratio measures
# exactly that follow-through, and it separates the live record cleanly:
#     eff = 1.00  -> put_debit +2,535 / +922 / +240   (3/3 green, BEATS bear_call's +2,047)
#     eff ~0.1-0.2-> put_debit -1,946 / -2,287 / -491 / +611  (net -4,113 vs bear_call -1,267)
# Verified NOT lookahead: shadow preconditions are computed as-of entry time, and the live agent
# already carries trend_efficiency_ratio in its entry metadata (0.0269 at today's 09:48 entry).
#
# HONEST CAVEAT: n=7 live days and the eff values are bimodal (~1.0 or ~0.15), so any threshold in
# (0.2, 1.0) fits this sample equally — 0.50 is the midpoint, not a tuned optimum. This is a
# DEFAULT-FLIP justified by theta mechanics + the user's seller-first doctrine, to be confirmed
# forward like everything else. SEL_SELLER_FIRST=0 restores the old buy-always behaviour.
_SELLER_FIRST = os.environ.get("SEL_SELLER_FIRST", "1") == "1"
_BUY_MIN_EFFICIENCY = _sel_env_float("SEL_BUY_MIN_EFFICIENCY")
if _BUY_MIN_EFFICIENCY is None:
    _BUY_MIN_EFFICIENCY = 0.50

# ── SELL THE CHOP / RANGE (default ON) ────────────────────────────────────────────────────────
# The classifier funnels every moderate-conviction / mild-bias day into CHOP, which stood aside — so
# the agent did NOTHING on the majority of sessions (531 CHOP + 259 RANGE_WIDE cycles over 4 days,
# all stand-aside). Those are premium-selling days: on the exact days the agent stood aside, the
# neutral sellers made iron_fly +4,053 / strangle +2,723 / straddle +8,029 (live shadow book). The
# old "CHOP = -Rs73k killer" warning was about DIRECTIONAL credit spreads whipsawing in chop — a
# NEUTRAL seller is the opposite trade and the forward record says it profits. So route a chop/range
# with ADEQUATE premium to a neutral seller (defined-risk fly first) instead of standing aside.
# Guards keep it honest: premium rich/normal (not cheap) and the range formed (after 11:00 — morning
# flies get run over before the range sets). Stays NEUTRAL so the -73k directional failure can't
# recur; the RANGE_INVALIDATION exit cuts it if the chop breaks into a trend. n is small —
# SEL_SELL_CHOP=0 disables. Ties to [[project_ordering_robustness]].
_SELL_CHOP = os.environ.get("SEL_SELL_CHOP", "1") == "1"
_CHOP_SELL_AFTER = time(11, 0)   # let the range form before selling the ATM
_CHOP_DRIFT_MIN_CONV = _sel_env_float("SEL_CHOP_DRIFT_MIN_CONV") or 0.40   # min lean to sell a with-drift spread in chop
_CHOP_DRIFT_AFTER = time(10, 0)   # with-drift spread can enter earlier than the neutral fly (it leans with an established bias, needs less range-formation)
# EXHAUSTION guard for with-drift entries. The drift is BACKWARD-looking, so fading a move that is
# already extended AND stalling at its extreme sells into a reversal — the single most repeated loss:
# bear-calls entered at the low on 08-05/08-12/08-31 (-1,160 / -2,291 / -2,428), each right before the
# bounce. Skip the directional lean when the move is both extended and has stalled.
_DRIFT_EXHAUST_MOVE_PTS = _sel_env_float("SEL_DRIFT_EXHAUST_MOVE_PTS") or 70.0   # ~0.3% of Nifty = a big directional move
_DRIFT_STALL_PCT = _sel_env_float("SEL_DRIFT_STALL_PCT") or 0.15                 # last-hour move smaller than this (in drift dir) = stalled
# A neutral fly needs a RANGE. Don't sell one into a tape that is moving efficiently — it gets
# whipsawed out in minutes (09-01: two flies sold at trend_eff 0.70/0.55, both cut on RANGE_INVALIDATION
# in 5.6 min, net churn). Mirrors the exit's _THETA_INVALIDATION_MIN_EFF so entry and exit agree on the
# range/trend boundary: sell the fly only when efficiency is BELOW this; above it the tape is trending.
_RANGE_FLY_MAX_EFF = _sel_env_float("SEL_RANGE_FLY_MAX_EFF") or 0.50

# ── ORB-GATED TREND (default ON) ──────────────────────────────────────────────────────────────
# Verification 2026-08 found the agent calls STRONG_TREND on choppy days: it fired STRONG_TREND_DOWN
# at 09:48 on 2026-07-23 (a -0.9% gap, accepted_breakout=False) — before the 30-min opening range
# even formed — and the day was flat chop. Across 07-16/21/23 it called trends with
# accepted_breakout=False; those are exactly the days it should have SOLD premium, not chased a
# direction. The BREAKOUT branch already requires an accepted breakout; the STRONG_TREND branch did
# NOT. Grounded in ORB research (8yr Nifty ORB backtest + Zerodha options ORB): only call a trend
# once the 30-min range has formed, the breakout is ACCEPTED (price closed beyond and held), and the
# range is wide enough to matter (tiny ranges = fakeouts). OI is a SOFT confirm only — the wall
# velocity fields are frequently 0, so they can veto a clearly-defended break but never gate on
# absence. An un-confirmed "trend" falls through to CHOP/RANGE -> sell the neutral fly.
# SEL_ORB_GATE_TREND=0 restores the old un-gated trend classification.
_ORB_GATE_TREND = os.environ.get("SEL_ORB_GATE_TREND", "1") == "1"
_ORB_MIN_RANGE_PTS = _sel_env_float("SEL_ORB_MIN_RANGE_PTS")
if _ORB_MIN_RANGE_PTS is None:
    _ORB_MIN_RANGE_PTS = 40.0        # skip tiny opening ranges — they break falsely (ORB backtest)
_ORB_FORMED_AFTER = time(9, 45)      # first two 15-min candles = the opening range
_ORB_EFF_CONFIRM = _sel_env_float("SEL_ORB_EFF_CONFIRM") or 0.50   # a move this efficient IS confirmed even w/o an accepted OR break

# ── ZONE CONFLUENCE — the big-player range trade (default ON) ──────────────────────────────────
# Root cause (5-whys, 2026-08): the agent has only two modes — trend-follow (needs a confirmed
# breakout) and premium-sell (needs rich IV). In a low-vol rangebound tape NEITHER fires, so it
# stands aside on the majority of days (501 chop stand-asides / 3 days). A desk does the third thing:
# it positions AROUND demand and supply zones — buy where institutions accumulate (support / put
# wall, put writers defending), sell where they distribute (resistance / call wall). This mode fires
# when price is AT a zone with OI + chart-pattern confluence, and positions WITH the zone as a
# DEFINED-RISK spread anchored to the wall: at demand -> bull-put below support; at supply -> bear-call
# above resistance. It needs neither rich IV nor a breakout — the edge is the OI-defended zone holding,
# which is exactly the structural, institutional read the user asked for. All signals already exist.
# SEL_ZONE_CONFLUENCE=0 disables.
_ZONE_CONFLUENCE = os.environ.get("SEL_ZONE_CONFLUENCE", "1") == "1"
_ZONE_EDGE_FRAC = _sel_env_float("SEL_ZONE_EDGE_FRAC") or 0.33   # price within this fraction of a wall = "at the zone"
_ZONE_MIN_CONFLUENCE = int(_sel_env_float("SEL_ZONE_MIN_CONFLUENCE") or 2)   # confluences beyond location+level


def _levels_near(m: dict, price: float, tol: float, side: str) -> list[str]:
    """Higher-timeframe structure levels within `tol` of price — the multi-timeframe stack that turns
    an OI wall into a zone a desk actually defends. DEMAND uses supports, SUPPLY uses resistances;
    daily EMAs / VWAP bands / weekly extremes / the round number count on the right side."""
    if side == "DEMAND":
        keys = ("support_15m", "daily_ema20", "daily_ema50", "weekly_low", "vwap_lower_1", "vwap_lower_2")
    else:
        keys = ("resistance_15m", "daily_ema20", "daily_ema50", "weekly_high", "vwap_upper_1", "vwap_upper_2")
    hits = []
    for k in keys:
        v = _f(m, k, 0.0)
        if v > 0 and abs(v - price) <= tol:
            hits.append(k)
    if abs(round(price / 100.0) * 100.0 - price) <= tol:     # a round 100-level at price
        hits.append("round_100")
    return hits


def _zone_confluence(read, m: dict, spot: float) -> str | None:
    """Seasoned-trader demand/supply read. A zone is valid only when the OI wall STACKS with
    higher-timeframe price levels (multi-TF confluence), order flow confirms, and it does not fight
    the higher-timeframe trend. Writes zone_confluence_score / _why to metadata for conviction + audit.
      DEMAND: price at the put-wall zone + >=1 HTF support at price + (bullish hold OR OI holding) +
              daily bias not bearish + flow not bearish.
      SUPPLY: the mirror at the call-wall zone.
    Returns 'DEMAND' / 'SUPPLY' / None."""
    if not _ZONE_CONFLUENCE or spot <= 0:
        return None
    cw, pw = _f(m, "current_call_wall", 0.0), _f(m, "current_put_wall", 0.0)
    if cw <= 0 or pw <= 0 or cw <= pw:
        return None
    rng = cw - pw
    tol = max(rng * 0.15, spot * 0.0025)                     # zone thickness: ~0.25% of spot or 15% of the wall range
    daily_bias = _f(m, "daily_bias_score", 0.0)
    smb = str(m.get("smart_money_bias") or "NEUTRAL").upper()
    mig = str(m.get("wall_migration_bias") or "NEUTRAL").upper()
    first_test = bool(m.get("at_first_test_of_level"))

    def _decide(side: str) -> str | None:
        levels = _levels_near(m, spot, tol, side)
        if not levels:                                       # MULTI-TF REQUIREMENT: no HTF level here -> not a zone
            return None
        if side == "DEMAND":
            pattern = str(m.get("bullish_candle_pattern") or "NONE") != "NONE" or first_test
            oi_holding = _f(m, "put_wall_shift", 0.0) >= 0 and mig != "BEARISH"     # supports not being pulled
            trend_ok = daily_bias >= 0 and smb != "BEARISH"                          # HTF permits longs (no knife-catch)
        else:
            pattern = str(m.get("bearish_candle_pattern") or "NONE") != "NONE" or first_test
            oi_holding = _f(m, "call_wall_shift", 0.0) <= 0 and mig != "BULLISH"     # resistance not lifting
            trend_ok = daily_bias <= 0 and smb != "BULLISH"
        if not trend_ok:                                     # never trade a zone against the higher-TF trend
            return None
        conf = len(levels) + (1 if pattern else 0) + (1 if oi_holding else 0)
        if conf < 1 + _ZONE_MIN_CONFLUENCE:                  # location+HTF level is the floor; need real stack on top
            return None
        m["zone_confluence_score"] = conf
        m["zone_confluence_why"] = f"HTF levels {levels}; pattern={pattern}; oi_holding={oi_holding}; daily_bias={daily_bias:g}"
        return side

    if spot <= pw + rng * _ZONE_EDGE_FRAC:                   # at the demand edge
        d = _decide("DEMAND")
        if d:
            return d
    if spot >= cw - rng * _ZONE_EDGE_FRAC:                   # at the supply edge
        return _decide("SUPPLY")
    return None


def _orb_confirms_trend(m: dict, now_time, direction: str) -> bool:
    """True only if a STRONG_TREND in `direction` ('UP'/'DOWN') is confirmed by an accepted
    opening-range breakout — the ORB discipline. Fixes 'trend called on unconfirmed early momentum'."""
    if not _ORB_GATE_TREND:
        return True
    # (a) the 30-min opening range must have formed
    if now_time is not None and now_time < _ORB_FORMED_AFTER:
        return False
    # (b) CONFIRMATION: an accepted OR-breakout, OR the move actually FOLLOWING THROUGH (realized
    # trend_efficiency). accepted_breakout ALONE is too strict — 2026-08-04 a real -1.04% down move
    # had accepted_breakout=False ALL day, so the gate blocked a genuine trend: the agent missed the
    # put-debit (+2,985) and would have sold a losing fly (-3,431). Realized efficiency is the honest
    # confirmation the move is real (NOT lookahead — it is efficiency up to NOW), and it is exactly
    # what distinguishes 08-04 (eff -> 1.0) from 07-23's fake early trend (eff 0.04 at the 09:48 entry).
    # Direction-appropriate: a DOWN trend is confirmed by accepted_breakdown, not the upside flag.
    _accepted = bool(m.get("accepted_breakout")) if direction == "UP" else bool(m.get("accepted_breakdown"))
    if not (_accepted or _trend_efficiency(m) >= _ORB_EFF_CONFIRM):
        return False
    # (c) if the confirmation is an accepted OR-break, its direction must match; when confirmation is
    # efficiency-based, a trend can develop without a clean OR break, so we trust the read's bias.
    if _accepted:
        orb = str(m.get("opening_range_break_state") or "NONE")
        if direction == "UP" and orb != "UP":
            return False
        if direction == "DOWN" and orb not in ("DOWN", "FAILED_UP"):
            return False
    # (d) opening range wide enough to be meaningful
    orh, orl = _f(m, "or_high", 0.0), _f(m, "or_low", 0.0)
    if orh > 0 and orl > 0 and (orh - orl) < _ORB_MIN_RANGE_PTS:
        return False
    return True
    # NOTE: OI confirmation (veto a break where the wall is still BUILDING = writers defending) is a
    # sound refinement but deferred — the wall-velocity scale (0..30, ~25% of readings > 0.5) needs a
    # calibrated threshold before it gates, or it over-blocks real trends. Follow-up, validated first.


def _trend_efficiency(metadata: dict) -> float:
    """Follow-through of the move: |net displacement| / total path travelled. ~1.0 = clean trend
    (a debit's move keeps extending), ~0.1 = chop (the tape retraces everything it gives)."""
    for k in ("trend_efficiency_ratio", "trend_efficiency", "efficiency_ratio"):
        v = metadata.get(k)
        if isinstance(v, (int, float)):
            return float(v)
    return 0.0

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

# Selection filter (NEW 2026-07-21, default ON): VETO a directional TREND/BREAKOUT debit when the
# tape is a PINNED LONG-GAMMA range. Retrospective on today's live loss: the agent selected
# STRONG_TREND_DOWN / BREAKOUT_DOWN (put-debit) at 0.78 conviction on a 0.29%-range day, while its OWN
# signals all said "range, no trend": gex_regime=LONG_GAMMA, pin_risk_active=True,
# trend_efficiency_ratio=0.13, inside_opening_range=True, opening_range_break_state=NONE. A directional
# debit needs follow-through; in a dealer-long-gamma pin the tape is suppressed / mean-reverting, so it
# just bleeds (today: -84.5 on the booked leg). Require all THREE independent range signals to agree so
# this never fires on a real trend. Mechanism is sound (long gamma => dealers dampen realized vol);
# one strong live example so far — confirm as forward data accumulates. SEL_PIN_VETO=0 disables;
# SEL_PIN_VETO_EFF_MAX tunes the trend-efficiency ceiling.
_PIN_VETO = os.environ.get("SEL_PIN_VETO", "1") == "1"
_PIN_VETO_EFF_MAX = _sel_env_float("SEL_PIN_VETO_EFF_MAX") or 0.40

# Selection filter (NEW 2026-07-21, default ON): a fresh opening-range break must be ACCEPTED — held
# past a retest — to be classified as a BREAKOUT, not a fakeout that immediately reverts. Today 11:35:
# opening_range_break_state=DOWN but accepted_breakout=False and trend_efficiency_ratio=0.0049 — a fake
# breakdown that a long-gamma pin pulled straight back, so the put-debit had no follow-through. An
# un-accepted break falls through to STRONG_TREND (if trend-quality is high) or CHOP, where the other
# gates apply. Complements the pin veto (this also catches fakeouts on normal, non-pinned days).
# FAILED_UP (a rejected upside break) is left as-is — it's itself a rejection signal. SEL_REQUIRE_ACCEPTED_BREAKOUT=0 disables.
_REQUIRE_ACCEPTED_BREAKOUT = os.environ.get("SEL_REQUIRE_ACCEPTED_BREAKOUT", "1") == "1"

# Core stance (NEW 2026-07-21, default ON): the agent's primary skill is OPTION SELLING, not buying.
# A pinned long-gamma range (what _pinned_range_veto detects) is the BEST premium-sell condition —
# dealers suppress realized vol, so a defined-risk iron condor at the pin harvests theta reliably.
# So on that detection, instead of standing aside, SELL an iron condor at the pin — UNLESS premium is
# genuinely cheap (vol regime CHEAP_BUY = implied under realized), in which case there's nothing worth
# selling and we stand aside. This is deliberately independent of SEL_NO_CONDOR (which disabled the
# generic sell-everywhere condor); this is the TARGETED sell only on days the greeks say vol is pinned.
# Runs paper-on-live so the cost math is proven forward before it ever earns real money. SEL_PIN_SELL=0
# reverts to stand-aside.
_PIN_SELL = os.environ.get("SEL_PIN_SELL", "1") == "1"


def _pinned_range_veto(m: dict[str, Any]) -> tuple[bool, str]:
    """True when the tape is a pinned long-gamma, low-efficiency range — where a directional debit
    bleeds. All three signals must agree, so it won't veto a genuine trend."""
    if not _PIN_VETO:
        return False, ""
    gex = str(m.get("gex_regime") or "").upper()
    pinned = bool(m.get("pin_risk_active"))
    eff = _f(m, "trend_efficiency_ratio", 1.0)  # missing -> treat as high -> no veto
    if gex == "LONG_GAMMA" and pinned and eff < _PIN_VETO_EFF_MAX:
        return True, (
            f"pinned long-gamma range (gex=LONG_GAMMA, pin_risk_active, trend_efficiency "
            f"{eff:.2f}<{_PIN_VETO_EFF_MAX:.2f}) — dealers suppress the range; a directional debit "
            f"needs follow-through it won't get. Stand aside."
        )
    return False, ""


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
ZONE_DEMAND = "ZONE_DEMAND"      # price at a demand zone (support/put-wall) — position long with it
ZONE_SUPPLY = "ZONE_SUPPLY"      # price at a supply zone (resistance/call-wall) — position short with it
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
_EXECUTABLE = {"BEAR_CALL_CREDIT_SPREAD", "BULL_PUT_CREDIT_SPREAD", "IRON_CONDOR", "IRON_FLY",
               "SHORT_STRANGLE", "SHORT_STRADDLE", "CALL_DEBIT_SPREAD", "PUT_DEBIT_SPREAD"}
# HARD BAN (desk conviction): the operator does not trade IRON_FLY (nor the naked ATM SHORT_STRADDLE) —
# a good strategy in theory, but no conviction in it, so it must never fire. Removed from the executable
# set so any path that ever selects one is stood aside. Env SEL_BANNED_STRUCTURES to change the list.
_BANNED_STRUCTURES = {s.strip().upper() for s in
                      os.environ.get("SEL_BANNED_STRUCTURES", "IRON_FLY,SHORT_STRADDLE").split(",") if s.strip()}
_EXECUTABLE = _EXECUTABLE - _BANNED_STRUCTURES


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


def classify_condition(read: MarketRead, metadata: dict[str, Any], now_time=None, spot: float = 0.0) -> str:
    """Name the market condition from the read + structure signals."""
    m = metadata
    orb = str(m.get("opening_range_break_state") or "NONE")
    trend_up_q = _f(m, "bullish_trend_quality_score", 0.0)
    trend_dn_q = _f(m, "bearish_trend_quality_score", 0.0)
    rv = _f(m, "rv30_pct", 0.0)
    vix = _f(m, "india_vix", 0.0)

    # Breakout: OR break with directional consensus — but a fresh UP/DOWN break must be ACCEPTED
    # (held past retest), not a fakeout that immediately reverts. FAILED_UP is a rejection signal and
    # is exempt. An un-accepted fresh break falls through to STRONG_TREND / CHOP below.
    # Direction-appropriate acceptance: a DOWN break is confirmed by accepted_breakdown, an UP break by
    # accepted_breakout. Checking only accepted_breakout for BOTH threw away 2026-08-12's clean ORB
    # breakdown (accepted_breakdown=True, accepted_breakout=False) — it fell to CHOP and the agent sold
    # a losing -2,291 with-drift bear-call instead of trading the confirmed downtrend (put-debit +570).
    accepted_up = bool(m.get("accepted_breakout")) or not _REQUIRE_ACCEPTED_BREAKOUT
    accepted_dn = bool(m.get("accepted_breakdown")) or not _REQUIRE_ACCEPTED_BREAKOUT
    if orb == "UP" and read.bias == "BULLISH" and accepted_up:
        return BREAKOUT_UP
    if orb == "DOWN" and read.bias == "BEARISH" and accepted_dn:
        return BREAKOUT_DOWN
    if orb == "FAILED_UP" and read.bias == "BEARISH":
        return BREAKOUT_DOWN

    # Strong trend: high trend-quality + directional bias + conviction, AND confirmed by an accepted
    # opening-range breakout (ORB gate) — otherwise early/gap momentum masquerades as a trend and the
    # day is really chop (2026-07-23: STRONG_TREND_DOWN at 09:48, accepted_breakout=False, flat day).
    if read.bias == "BULLISH" and trend_up_q >= 4.0 and read.conviction >= 0.60 and _orb_confirms_trend(m, now_time, "UP"):
        return STRONG_TREND_UP
    if read.bias == "BEARISH" and trend_dn_q >= 4.0 and read.conviction >= 0.60 and _orb_confirms_trend(m, now_time, "DOWN"):
        return STRONG_TREND_DOWN

    # ZONE CONFLUENCE — no accepted breakout, so it's not a trend. Is price AT a demand/supply zone
    # with OI + pattern confluence? That is the big-player range trade (buy demand, sell supply),
    # checked BEFORE the generic range/chop buckets so a zone read wins over "just stand aside".
    zone = _zone_confluence(read, m, spot)
    if zone == "DEMAND":
        return ZONE_DEMAND
    if zone == "SUPPLY":
        return ZONE_SUPPLY

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


def _vwap_regime(read, m: dict[str, Any]) -> str:
    """The validated VWAP/OR regime read. A CONFIDENT trend needs realized efficiency AND an accepted
    opening-range break in the bias direction; everything else is RANGE (handled by the range logic)."""
    eff = _trend_efficiency(m)
    if eff >= _VWAP_TREND_EFF and read.bias == "BULLISH" and bool(m.get("accepted_breakout")):
        return "TREND_UP"
    if eff >= _VWAP_TREND_EFF and read.bias == "BEARISH" and bool(m.get("accepted_breakdown")):
        return "TREND_DOWN"
    return "RANGE"


# ── HIGH-PROBABILITY CONFLUENCE GATE (default ON) ──────────────────────────────────────────────
# The desk mandate: intraday takes ONLY high-probability directional trades — PRICE ACTION (higher-
# timeframe S/R, M/W reversal patterns, opening-range-break acceptance) that is CONFIRMED by the
# OPTION CHAIN (smart-money OI bias, change-in-OI at the call/put walls, PCR trend, pressure scores),
# with the move showing real follow-through (trend_efficiency) and the 20-EMA not fighting it. When
# those DON'T align, there is NO high-probability trade -> STAND ASIDE (the seat stays empty).
# This makes stand-aside the default and removes the premium-selling paths (chop/range -> iron fly)
# that produced the whipsaw churn. Env SEL_CONFLUENCE_ONLY=0 restores the legacy condition cascade.
_CONFLUENCE_ONLY = os.environ.get("SEL_CONFLUENCE_ONLY", "1") == "1"
_CONF_MIN_EFF = _sel_env_float("SEL_CONF_MIN_EFF") or 0.35
_CONF_STRONG_EFF = _sel_env_float("SEL_CONF_STRONG_EFF") or 0.55   # eff alone confirms a clean trend
# Up-side is stood aside by default: on the shadow book every bullish structure was ~breakeven-to-
# negative (Nifty intraday has no validated up-side edge — [[project_ordering_robustness]]). Set
# SEL_CONF_ALLOW_BULLISH=1 to also trade a CALL_DEBIT on high-conviction up days.
_CONF_ALLOW_BULLISH = os.environ.get("SEL_CONF_ALLOW_BULLISH", "0") == "1"
# Structure for a BULLISH confluence day. BUYING up (CALL_DEBIT) is validated-negative — up-moves
# grind, a long debit whipsaws (real data: -16k / 16 tr). SELLING up (BULL_PUT credit) fits the grind
# (theta, wins on up OR sideways) and the synthetic dense set was BIASED AGAINST selling (5x-wide
# spreads), so it deserves a re-test on the REAL live-capture data. Default BULL_PUT.
_CONF_UP_STRUCT = os.environ.get("SEL_CONF_UP_STRUCT", "BULL_PUT").upper()
# When ON, require the option chain to POSITIVELY CONFIRM the direction (change-in-OI / smart-money /
# pressure agreeing), not merely not-oppose. Live the chain computes real values ~98% of polls (the
# shadow book showed UNKNOWN only because the replay doesn't thread the previous chain), so this is a
# real filter live. Default OFF until the dense-dataset backtest shows it sharpens selection.
_CONF_REQUIRE_CHAIN = os.environ.get("SEL_CONF_REQUIRE_CHAIN", "0") == "1"


def _chain_confirms(m: dict[str, Any], direction: str) -> bool:
    """Positive option-chain agreement: smart-money / OI-pressure bias equals the direction, OR a clear
    same-side pressure tilt (change-in-OI at the walls). Used only when SEL_CONF_REQUIRE_CHAIN is on."""
    votes = [str(m.get(k, "")).upper() for k in ("smart_money_bias", "oi_pressure_bias")]
    if direction in votes:
        return True
    bull = _f(m, "bullish_option_chain_pressure_score", 0.0)
    bear = _f(m, "bearish_option_chain_pressure_score", 0.0)
    return (bear - bull) >= 1.0 if direction == "BEARISH" else (bull - bear) >= 1.0


def _chain_opposes(m: dict[str, Any], direction: str) -> bool:
    """Veto only when the option chain CONTRADICTS the direction (smart-money/OI/thesis explicitly the
    other way, or a strong opposite pressure tilt). smart_money_bias is UNKNOWN most days, so we require
    the chain not to FIGHT us rather than to positively confirm — else the gate never fires."""
    opp = "BULLISH" if direction == "BEARISH" else "BEARISH"
    votes = [str(m.get(k, "")).upper() for k in ("smart_money_bias", "oi_pressure_bias", "thesis_net_bias")]
    if opp in votes and direction not in votes:
        return True
    bull = _f(m, "bullish_option_chain_pressure_score", 0.0)
    bear = _f(m, "bearish_option_chain_pressure_score", 0.0)
    return (bull - bear) >= 1.5 if direction == "BEARISH" else (bear - bull) >= 1.5


def _high_prob_confluence(read, m: dict[str, Any]) -> tuple[str | None, str]:
    """(direction, why). Fires on a high-conviction DIRECTIONAL setup built from the signals that
    actually populate: directional bias + real follow-through (trend_efficiency) + an opening-range
    break in that direction OR a strong clean trend, with the option chain not fighting it. BEARISH ->
    buy a PUT-DEBIT (the validated convex edge); BULLISH stands aside by default. None -> stand aside.

    Calibrated on the shadow book (scripts/confluence_gate_backtest.py): the price-action FLAGS the
    first version used (accepted_breakout/breakdown, M/W patterns) were DEAD — 0-2 firings in 32 days —
    so the gate never traded. Bias+efficiency+ORB are populated and select the high-conviction days;
    routing those to a put-debit made +2,787 over 10 days (vs -1,241 for the bear-call credit)."""
    eff = _trend_efficiency(m)
    orb = str(m.get("opening_range_break_state") or "").upper()
    bias = getattr(read, "bias", "NEUTRAL")
    if eff < _CONF_MIN_EFF:
        return None, f"no follow-through (efficiency {eff:.2f} < {_CONF_MIN_EFF:.2f}) — chop, stand aside"
    _bear_confirm = (orb == "DOWN") or eff >= _CONF_STRONG_EFF
    _bull_confirm = (orb == "UP") or eff >= _CONF_STRONG_EFF
    if _CONF_REQUIRE_CHAIN:  # stricter: chain must positively AGREE, not merely not-oppose
        _bear_confirm = _bear_confirm and _chain_confirms(m, "BEARISH")
        _bull_confirm = _bull_confirm and _chain_confirms(m, "BULLISH")
    if bias == "BEARISH" and _bear_confirm and not _chain_opposes(m, "BEARISH"):
        return "BEARISH", (f"HIGH-PROB BEARISH: bias down + follow-through eff{eff:.2f} + "
                           f"{'opening-range break DOWN' if orb == 'DOWN' else 'strong clean trend'} "
                           f"(chain not fighting) — BUY a put-debit (convex: small loss, big win).")
    if bias == "BULLISH" and _bull_confirm and not _chain_opposes(m, "BULLISH"):
        if _CONF_ALLOW_BULLISH:
            return "BULLISH", (f"HIGH-PROB BULLISH: bias up + follow-through eff{eff:.2f} — BUY a call-debit.")
        return None, ("bullish setup but intraday UP-SIDE has no validated edge (every up-structure "
                      "~breakeven-to-negative on the record) — stand aside.")
    return None, (f"no confluence (eff {eff:.2f}, orb {orb}, bias {bias}) — stand aside")


# ── RANGE-FADE (mean reversion — the 80% engine) ───────────────────────────────────────────────
# Most sessions are sideways-but-volatile: price oscillates between S/R and each swing is worth
# 20-30+ pts. FADE the edges — short a confirmed M-top rejection AT resistance, long a confirmed
# W-bottom rejection AT support — targeting the opposite edge. Validated live (2026-09-23): the
# M-pattern fired with spot ON resistance 23,388 / 6-11pt under the call wall 23,400, wick 1.1-1.37.
# The ONE discipline that keeps this from being a falling knife: only fade a CONFIRMED rejection AT a
# real level (not blindly), with a tight stop just beyond it — if the level breaks (the 20% trend
# day) you're out cheap. Env-gated (SEL_RANGE_FADE, default OFF until the fade-exit is built+proven).
_RANGE_FADE = os.environ.get("SEL_RANGE_FADE", "0") == "1"
_FADE_EDGE_PTS = _sel_env_float("SEL_FADE_EDGE_PTS") or 15.0      # spot must be within this of the level
_FADE_MAX_EFF = _sel_env_float("SEL_FADE_MAX_EFF") or 0.55         # don't fade a strong trend (it'll break)
_FADE_TARGET_PTS = _sel_env_float("SEL_FADE_TARGET_PTS") or 30.0   # take the swing into the range interior
_FADE_STOP_PTS = _sel_env_float("SEL_FADE_STOP_PTS") or 12.0       # cut fast if the level breaks the wrong way


def _range_fade_signal(read, m: dict[str, Any], spot: float) -> tuple[str | None, str]:
    """(direction, why). FADE_SHORT at a confirmed M-top on resistance; FADE_LONG at a confirmed
    W-bottom on support; else None. Only in a NON-trending tape (eff < _FADE_MAX_EFF) — fading a real
    trend is how range-traders blow up. Proximity to a REAL level + the rejection wick are the gate."""
    from .strategy_matrix import _m_pattern_at_resistance, _w_pattern_at_support
    eff = _trend_efficiency(m)
    if eff >= _FADE_MAX_EFF:
        return None, f"tape is trending (eff {eff:.2f}) — don't fade, it'll break the level"
    resist = _f(m, "resistance_5m") or _f(m, "resistance_15m") or _f(m, "current_call_wall")
    support = _f(m, "support_5m") or _f(m, "support_15m") or _f(m, "current_put_wall")
    at_resist = resist > 0 and abs(spot - resist) <= _FADE_EDGE_PTS
    at_support = support > 0 and abs(spot - support) <= _FADE_EDGE_PTS
    if at_resist and _m_pattern_at_resistance(m):
        return "FADE_SHORT", (f"RANGE-FADE SHORT: M-top rejection AT resistance {resist:.0f} "
                              f"(spot {spot:.0f}, eff {eff:.2f}) — short into the range, target the low.")
    if at_support and _w_pattern_at_support(m):
        return "FADE_LONG", (f"RANGE-FADE LONG: W-bottom rejection AT support {support:.0f} "
                             f"(spot {spot:.0f}, eff {eff:.2f}) — long into the range, target the high.")
    return None, (f"no fade setup (eff {eff:.2f}, at_resist {at_resist}, at_support {at_support})")


def select_strategy(metadata: dict[str, Any], spot: float, now_time=None) -> StrategyChoice:
    """Read the market, name the condition, and choose the strategy family + structures.

    now_time (datetime.time, optional): used for time-of-day rules learned from the
    retrospective — e.g. condors only after the range forms (mid-day)."""
    read = assess_market(metadata, spot)
    condition = classify_condition(read, metadata, now_time, spot)
    iv = _iv_regime(metadata)
    # Volatility engine FIRST — trade vol, not just direction. This tells us whether
    # premium is rich (sell) or cheap (buy), which the pros decide before direction.
    vol = assess_vol(metadata)
    choice = StrategyChoice(condition=condition, iv_regime=iv, read=read, conviction=read.conviction)
    choice.vol_regime = vol.regime
    choice.vol_notes = vol.notes

    # HIGH-PROBABILITY CONFLUENCE GATE (default ON): intraday trades ONLY when price action and the
    # option chain agree on a directional, high-probability setup. No confluence -> NO TRADE. This is
    # the top-priority decision: it short-circuits the entire legacy condition cascade (incl. every
    # premium-selling iron-fly/condor path), so the seat stays empty on uncommitted days. Seller-first:
    # a confluence direction sells the WITH-TREND defined-risk credit spread (bull-put / bear-call).
    if _CONFLUENCE_ONLY:
        _dir, _why = _high_prob_confluence(read, metadata)
        # The validated STRUCTURE for a high-conviction directional leg is a DEBIT, not a credit spread:
        # on the shadow book the bear-call CREDIT lost -1,241 (71% win, capped winner / big loser) while
        # the PUT-DEBIT made +2,787 (50% win, convex — small losses, big wins). So confluence -> BUY the
        # directional debit. Debits leak in the post-open-drive window, so honour the debit blackout.
        _bull_is_debit = _CONF_UP_STRUCT == "CALL_DEBIT"
        # The debit blackout only applies to DEBIT entries (post-open-drive exhaustion). A BEARISH day
        # always buys a put-debit; a BULLISH day buys a call-debit only if SEL_CONF_UP_STRUCT=CALL_DEBIT.
        _is_debit_entry = (_dir == "BEARISH") or (_dir == "BULLISH" and _bull_is_debit)
        _in_blackout = (_DEBIT_BLACKOUT is not None and now_time is not None
                        and _DEBIT_BLACKOUT[0] <= now_time < _DEBIT_BLACKOUT[1])
        if _is_debit_entry and _in_blackout:
            choice.family, choice.structures = FAM_STAND_ASIDE, []
            choice.rationale = (f"{_dir} confluence but inside the {_DEBIT_BLACKOUT[0].strftime('%H:%M')}-"
                                f"{_DEBIT_BLACKOUT[1].strftime('%H:%M')} debit blackout (post-open-drive "
                                f"exhaustion — debit entries leak here) — wait for a fresh leg.")
        elif _dir == "BEARISH":
            choice.family, choice.structures = FAM_DIRECTIONAL_DEBIT, ["PUT_DEBIT_SPREAD"]
            choice.rationale = _why
        elif _dir == "BULLISH" and _bull_is_debit:
            choice.family, choice.structures = FAM_DIRECTIONAL_DEBIT, ["CALL_DEBIT_SPREAD"]
            choice.rationale = _why
        elif _dir == "BULLISH":
            choice.family, choice.structures = FAM_DIRECTIONAL_CREDIT, ["BULL_PUT_CREDIT_SPREAD"]
            # Strong bullish conviction -> sell the put CLOSER (higher delta) for real credit even at low
            # VIX; strikes.py reads this flag. Far-OTM would collect peanuts and get rejected as thin.
            metadata["conf_bull_sell_closer"] = True
            choice.rationale = _why + " [SELL a bull-put CLOSER (higher-delta short) — theta fits the grind, real credit at low VIX]"
        else:
            metadata.pop("conf_bull_sell_closer", None)
            # No directional confluence. Before standing aside, try a RANGE FADE (the 80% engine):
            # fade a confirmed M/W rejection AT a real S/R edge in a non-trending tape. Uses a cheap
            # directional debit with a FADE exit (tight target + stop just beyond the level — set here
            # so execution can manage it as a scalp, not a fat-tail hold). Env-gated (default off).
            _fdir, _fwhy = _range_fade_signal(read, metadata, spot) if _RANGE_FADE else (None, "")
            if _fdir == "FADE_SHORT":
                choice.family, choice.structures = FAM_DIRECTIONAL_DEBIT, ["PUT_DEBIT_SPREAD"]
                metadata["is_fade"] = True
                metadata["fade_entry_spot"] = spot
                metadata["fade_target_spot"] = spot - _FADE_TARGET_PTS   # target the range interior (down)
                metadata["fade_stop_spot"] = spot + _FADE_STOP_PTS       # cut if it breaks back UP through the level
                choice.rationale = _fwhy
            elif _fdir == "FADE_LONG":
                choice.family, choice.structures = FAM_DIRECTIONAL_DEBIT, ["CALL_DEBIT_SPREAD"]
                metadata["is_fade"] = True
                metadata["fade_entry_spot"] = spot
                metadata["fade_target_spot"] = spot + _FADE_TARGET_PTS
                metadata["fade_stop_spot"] = spot - _FADE_STOP_PTS
                choice.rationale = _fwhy
            else:
                metadata.pop("is_fade", None)
                choice.family, choice.structures = FAM_STAND_ASIDE, []
                choice.rationale = "NO high-probability confluence — stand aside (no trade is a position). " + _why
        choice.executable_today = bool(choice.structures) and all(s in _EXECUTABLE for s in choice.structures)
        return choice

    # VWAP-SELLER override: on a CONFIDENT trend, SELL a directional credit WITH the trend (the validated
    # seller edge). RANGE falls through to the existing guarded range/fly logic. Env-gated forward test.
    if _VWAP_SELLER and vol.regime != CHEAP_BUY:
        _reg = _vwap_regime(read, metadata)
        if _reg == "TREND_UP":
            choice.family, choice.structures = FAM_DIRECTIONAL_CREDIT, ["BULL_PUT_CREDIT_SPREAD"]
            choice.rationale = ("VWAP-seller: TREND_UP (efficient + accepted OR break) — SELL a bull-put "
                                "WITH the trend (validated: selling the regime read made +5,475 on shadow fills).")
            choice.executable_today = all(s in _EXECUTABLE for s in choice.structures)
            return choice
        if _reg == "TREND_DOWN":
            choice.family, choice.structures = FAM_DIRECTIONAL_CREDIT, ["BEAR_CALL_CREDIT_SPREAD"]
            choice.rationale = ("VWAP-seller: TREND_DOWN (efficient + accepted OR break) — SELL a bear-call "
                                "WITH the trend (validated seller edge).")
            choice.executable_today = all(s in _EXECUTABLE for s in choice.structures)
            return choice
        # RANGE -> fall through to the range/chop logic (routes to IRON_FLY with its guards).

    # Pinned long-gamma range detected (see _pinned_range_veto): DON'T buy a directional debit into it.
    # Think like a SELLER — this is the best premium-harvest condition, so SELL a defined-risk iron
    # condor at the pin (unless premium is cheap, then stand aside). Applied before every mode.
    if condition in (STRONG_TREND_UP, STRONG_TREND_DOWN, BREAKOUT_UP, BREAKOUT_DOWN):
        _pv, _pv_why = _pinned_range_veto(metadata)
        if _pv:
            if _PIN_SELL and vol.regime != CHEAP_BUY:
                # Tight pin (spot glued to the ATM) → IRON FLY: max premium at the pin, defined risk.
                # Wider range → IRON CONDOR. Fly is best-first with condor as the build fallback.
                _spot_to_pin = abs(_f(metadata, "spot_to_pin_pts", 999.0))
                if _spot_to_pin <= 40.0:
                    choice.structures = ["IRON_FLY", "IRON_CONDOR"]
                    _lbl = f"IRON FLY at the pin (spot {_spot_to_pin:.0f}pt from ATM)"
                else:
                    choice.structures = ["IRON_CONDOR"]
                    _lbl = "IRON CONDOR across the range"
                choice.family = FAM_PREMIUM_SELL
                choice.rationale = (
                    f"PINNED LONG-GAMMA range → SELL defined-risk {_lbl} — harvest the suppressed "
                    f"realized vol (vol={vol.regime}). [was {condition}; a directional debit would "
                    f"bleed here: {_pv_why}]"
                )
            else:
                choice.family, choice.structures = FAM_STAND_ASIDE, []
                _why = "premium too cheap to sell (implied<realized)" if vol.regime == CHEAP_BUY else "pin-sell OFF"
                choice.rationale = f"VETOED {condition} — stand aside ({_why}): {_pv_why}"
            return choice

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
        elif _SKIP_SMB_BULLISH and str(metadata.get("smart_money_bias")) == "BULLISH":
            choice.family, choice.structures = FAM_STAND_ASIDE, []
            choice.rationale = f"{condition} but smart_money_bias BULLISH — buying puts against institutional flow (0% win in screen); stand aside."
        elif _MIN_STATE_CONF is not None and float(metadata.get("state_confidence_score") or metadata.get("state_confidence") or 1.0) < _MIN_STATE_CONF:
            choice.family, choice.structures = FAM_STAND_ASIDE, []
            choice.rationale = f"{condition} but state_confidence below {_MIN_STATE_CONF} — low-conviction read (loses in screen); stand aside."
        else:
            _eff = _trend_efficiency(metadata)
            if _SELLER_FIRST and _eff < _BUY_MIN_EFFICIENCY:
                # Low efficiency means the tape has NO DIRECTION — so the right seller here is
                # NEUTRAL, not bearish. A bear-call is still a directional bet, and on the live
                # record it is the wrong one: across the four low-efficiency days the directional
                # seller made -1,267 while the neutral fly made +5,783 (fly won 3 of 4, and on the
                # pin day by +4,920). This is not curve-fitting — it is the same signal read
                # honestly: trend_efficiency measures directional follow-through, so when it is
                # absent we should stop taking a directional position at all, not merely flip which
                # side of it we are on. Fly first, bear-call as the build fallback.
                choice.family = FAM_PREMIUM_SELL
                choice.structures = ["IRON_FLY", "BEAR_CALL_CREDIT_SPREAD"]
                choice.rationale = (
                    f"{condition} but trend_efficiency {_eff:.2f} < {_BUY_MIN_EFFICIENCY:.2f} — the tape "
                    f"retraces what it gives, so there is no direction to buy AND none to sell against. "
                    f"SELL NEUTRAL premium (iron fly at the ATM, bear-call fallback) and collect the decay."
                )
            else:
                choice.family, choice.structures = FAM_DIRECTIONAL_DEBIT, ["PUT_DEBIT_SPREAD"]
                choice.rationale = (
                    f"{condition}: buy PUT DEBIT with the down-move (trend_efficiency {_eff:.2f} >= "
                    f"{_BUY_MIN_EFFICIENCY:.2f} — clean follow-through, the one regime where the debit "
                    f"out-earns selling the move)."
                )

    elif condition == ZONE_DEMAND:
        # Multi-TF DEMAND zone: OI put wall stacked with higher-timeframe supports, a bullish hold,
        # flow not against, higher-TF trend permitting. Position WITH the institutions — SELL a
        # defined-risk BULL-PUT below the zone (strike selection anchors the short put to support).
        # Conviction scales with the confluence stack (a desk sizes up the higher-confluence zone).
        _cf = int(_f(metadata, "zone_confluence_score", 3))
        choice.conviction = min(0.90, 0.55 + 0.07 * _cf)
        choice.family, choice.structures = FAM_DIRECTIONAL_CREDIT, ["BULL_PUT_CREDIT_SPREAD"]
        choice.rationale = (f"ZONE_DEMAND (confluence {_cf}): OI put wall + HTF supports + bullish hold — "
                            f"SELL a bull-put below the zone. [{metadata.get('zone_confluence_why','')}]")

    elif condition == ZONE_SUPPLY:
        # Multi-TF SUPPLY zone: OI call wall stacked with higher-timeframe resistances, a bearish
        # rejection, flow not against, higher-TF trend permitting — SELL a bear-call above the zone.
        _cf = int(_f(metadata, "zone_confluence_score", 3))
        choice.conviction = min(0.90, 0.55 + 0.07 * _cf)
        choice.family, choice.structures = FAM_DIRECTIONAL_CREDIT, ["BEAR_CALL_CREDIT_SPREAD"]
        choice.rationale = (f"ZONE_SUPPLY (confluence {_cf}): OI call wall + HTF resistances + bearish rejection — "
                            f"SELL a bear-call above the zone. [{metadata.get('zone_confluence_why','')}]")

    elif condition == RANGE_WIDE:
        # Retrospective lesson: morning condors LOSE (-Rs1,151/24tr @10-12) because the
        # day's range hasn't formed yet; mid-day condors win (+Rs7.6k @70%). Only sell
        # the range once it's established (>= 11:30).
        _too_early = now_time is not None and now_time < time(11, 30)
        if _too_early:
            choice.family, choice.structures = FAM_STAND_ASIDE, []
            choice.rationale = "RANGE_WIDE but before 11:30 — range not yet formed (morning condors lose); wait."
        elif (iv in (IV_RICH, IV_NORMAL) or vol.regime == RICH_SELL) and _trend_efficiency(metadata) < _RANGE_FLY_MAX_EFF:
            # SEL_NO_CONDOR drops the CONDOR specifically (its P&L is order-noise on the ordering-robust
            # harness) — but that should NOT mean standing aside on a formed range, which cost real
            # money: on range/stand-aside days the live shadow made strangle +2,723 / fly +4,053. So
            # sell the range with a NEUTRAL structure the record likes (fly first, strangle fallback),
            # only adding the condor when it is not disabled.
            # VRP reconciliation: the crude iv gate calls VIX<11 "cheap", but that is an ABSOLUTE-level
            # proxy — the real question is IV vs REALIZED. The volatility engine measures exactly that,
            # and RICH_SELL means IV > realized (a genuine selling edge) even at low absolute VIX. On
            # 2026-08-11 (VIX 10.25 -> iv=CHEAP) the tape pinned a tight range all day and iron_fly made
            # +3,398 in the shadow, yet the agent stood aside every cycle on "IV CHEAP". Trusting the
            # VRP read is also what the prior 7mo sweep supports: adding a stand-aside-when-cheap rule
            # made it WORSE (+104,858 -> +84,945), i.e. trading low-vol days was the better policy.
            choice.family = FAM_PREMIUM_SELL
            choice.structures = ["IRON_FLY", "IRON_CONDOR"] if _NO_CONDOR else ["IRON_CONDOR", "IRON_FLY"]
            _rich = " (vol RICH)" if vol.regime == RICH_SELL else (" (vol CHEAP — watch)" if vol.regime == CHEAP_BUY else "")
            choice.rationale = f"RANGE_WIDE / mid-day{_rich}: two-sided walls, balanced tape — sell NEUTRAL premium inside the formed range{' (condor off: fly/strangle)' if _NO_CONDOR else ''}."
        elif iv in (IV_RICH, IV_NORMAL) or vol.regime == RICH_SELL:
            # Premium is sellable but the tape is MOVING (trend_efficiency high) — a neutral fly gets
            # whipsawed out in minutes (09-01 churn). A range structure needs an actual range; don't
            # sell one into a trending tape. Stand aside rather than churn.
            choice.family, choice.structures = FAM_STAND_ASIDE, []
            choice.rationale = (f"RANGE_WIDE walls but trend_efficiency {_trend_efficiency(metadata):.2f} "
                                f">= {_RANGE_FLY_MAX_EFF:.2f} — the tape is MOVING, not ranging; a fly would be "
                                f"whipsawed (the 09-01 5.6-min churn). Stand aside.")
        else:
            choice.family, choice.structures = FAM_STAND_ASIDE, []
            choice.rationale = "RANGE_WIDE but premium thin on BOTH reads (iv cheap AND vol not rich) — stand aside."

    elif condition == RANGE_TIGHT:
        choice.family, choice.structures = FAM_STAND_ASIDE, []
        choice.rationale = "RANGE_TIGHT: walls close, premium thin, no room — stand aside."

    elif condition == HIGH_VOL_UNDIRECTED:
        # Elevated vol, no direction: either sell rich premium (if truly rangebound)
        # or stay out (whipsaw). Default to stand-aside — this is where credit spreads died.
        choice.family, choice.structures = FAM_STAND_ASIDE, []
        choice.rationale = "HIGH_VOL_UNDIRECTED: elevated vol without direction = whipsaw risk — stand aside."

    else:  # CHOP
        # Premium is sellable if EITHER the iv-level read is rich/normal OR the VRP engine says
        # RICH_SELL (IV > realized). VIX<11 alone ("iv cheap") must not veto a genuine selling edge —
        # it cost a +3,398 iron_fly on 2026-08-11. See the matching RANGE_WIDE gate above.
        _prem_ok = iv in (IV_RICH, IV_NORMAL) or vol.regime == RICH_SELL
        _range_formed = now_time is None or now_time >= _CHOP_SELL_AFTER
        _drift_ok = now_time is None or now_time >= _CHOP_DRIFT_AFTER
        _lean = read.bias in ("BULLISH", "BEARISH") and read.conviction >= _CHOP_DRIFT_MIN_CONV
        # When the directional vertical can't be BUILT (e.g. an overhead wall forces the short too
        # far OTM to earn its width), fall back to a defined-risk NEUTRAL condor that harvests the
        # SAME premium — but only under the neutral path's own preconditions (premium sellable AND the
        # range has formed), so we never sell a neutral structure into an unformed range. The executor
        # walks this ranked list and only reaches the condor if the vertical fails to construct.
        _neutral_fallback = ["IRON_CONDOR"] if (_prem_ok and _range_formed) else []
        # Exhaustion: is the drift move already extended AND stalled at its extreme? A bearish drift is
        # stalled when the last hour is no longer falling; a bullish drift when it is no longer rising.
        _sess_move = abs(_f(metadata, "session_move_pts", 0.0))
        _last_hr = _f(metadata, "last_hour_change_pct", 0.0)
        _stalled = (_last_hr >= -_DRIFT_STALL_PCT) if read.bias == "BEARISH" else (_last_hr <= _DRIFT_STALL_PCT)
        _exhausted = _lean and _sess_move >= _DRIFT_EXHAUST_MOVE_PTS and _stalled
        if _SELL_CHOP and _exhausted:
            # Don't fade an exhausted move: extended + stalled at the extreme = reversal risk, not drift.
            choice.family, choice.structures = FAM_STAND_ASIDE, []
            choice.rationale = (
                f"CHOP {read.bias} drift but the move is EXHAUSTED (session {_sess_move:.0f}pts, last hour "
                f"{_last_hr:+.2f}% — extended and stalled at the extreme). Fading it sells into a reversal "
                f"(the repeated -1,160/-2,291/-2,428 bear-call-at-the-low loss). Stand aside."
            )
        elif _SELL_CHOP and _lean and _drift_ok and read.bias == "BULLISH":
            # Mild UP-DRIFT (not a confirmed trend) — SELL a WITH-DRIFT bull-put. It leans with the
            # bias AND collects theta (wins on up OR sideways), sells an OTM short (less IV-sensitive
            # than the ATM fly, so it works when ATM premium is thin), and is a high-win-rate seller.
            # The shadow book shows bull_put +1,072/+1,084/+630 on exactly the mild-drift days the
            # agent stood aside on — this converts those MISSES into trades and lifts the win rate.
            choice.family, choice.structures = FAM_DIRECTIONAL_CREDIT, ["BULL_PUT_CREDIT_SPREAD"] + _neutral_fallback
            choice.rationale = (f"CHOP but mild BULLISH drift (conviction {read.conviction:.2f}) — SELL a "
                                f"with-drift bull-put (theta + lean; high-win seller), not stand aside."
                                + (" Neutral condor fallback if the vertical can't be built." if _neutral_fallback else ""))
        elif _SELL_CHOP and _lean and _drift_ok and read.bias == "BEARISH":
            choice.family, choice.structures = FAM_DIRECTIONAL_CREDIT, ["BEAR_CALL_CREDIT_SPREAD"] + _neutral_fallback
            choice.rationale = (f"CHOP but mild BEARISH drift (conviction {read.conviction:.2f}) — SELL a "
                                f"with-drift bear-call (theta + lean), not stand aside."
                                + (" Neutral condor fallback if the vertical can't be built." if _neutral_fallback else ""))
        elif _SELL_CHOP and _prem_ok and _range_formed:
            # Genuinely NEUTRAL chop with sellable premium — neutral defined-risk fly, collect theta.
            choice.family = FAM_PREMIUM_SELL
            choice.structures = ["IRON_FLY", "IRON_CONDOR"]
            choice.rationale = (
                f"CHOP (bias {read.bias}, conviction {read.conviction:.2f}) but premium is sellable "
                f"and the range has formed — SELL NEUTRAL (defined-risk fly). NEUTRAL, not directional: "
                f"the -Rs73k killer was directional spreads here."
            )
        else:
            choice.family, choice.structures = FAM_STAND_ASIDE, []
            _why = "premium too cheap to sell" if not _prem_ok else ("range not yet formed (< 11:00)" if not _range_formed else "chop-selling disabled")
            choice.rationale = f"CHOP: uncommitted tape (bias {read.bias}, conviction {read.conviction:.2f}) — stand aside ({_why})."

    # NOTE: a vol rule "skip debit when IV 12-16" was calibrated from post-hoc buckets
    # (that bucket showed -Rs18,807/25% win) and REJECTED on validation: applying it made
    # 7mo WORSE (+84,945 -> +8,877). Reason = PATH DEPENDENCE — in a one-position system,
    # cutting trades frees the slot for other (worse) trades, so bucket edges don't
    # transfer. Lesson recorded; vol stays advisory until a rule survives full re-sim.

    # CONDOR only on/near expiry. A non-expiry condor carries little theta, has wide strikes, and gets
    # run over — the iron-FLY beats it on range days (+5,475 vs +1,060 on the shadow fills), and the
    # user flagged low confidence in non-expiry condors. Off expiry, swap any IRON_CONDOR -> IRON_FLY.
    if choice.structures and not bool(metadata.get("is_expiry_day")):
        _swapped, _seen = [], set()
        for s in choice.structures:
            s2 = "IRON_FLY" if s == "IRON_CONDOR" else s
            if s2 not in _seen:
                _seen.add(s2); _swapped.append(s2)
        choice.structures = _swapped

    choice.executable_today = bool(choice.structures) and all(s in _EXECUTABLE for s in choice.structures)
    return choice
