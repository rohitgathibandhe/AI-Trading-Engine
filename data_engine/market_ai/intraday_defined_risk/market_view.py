"""
MarketView — the intelligence core that reads market data and derives the
optimal strategy from first principles.

A professional trader doesn't ask "which of my pre-built playbooks fits today?"
They ask: "What is the market doing, and what is the best trade for that?"

This module answers the first question by counting independent directional
signals and producing a structured MarketView. classify_regime() then uses
MarketView as the primary routing mechanism — score and signal count drive
the decision, not `execution_5m == "UP_CONFIRMED"` as a hard gate.

Key design choices:
- Uses a fixed base threshold (4.0) rather than the adaptive params threshold,
  which can be raised too high by EOD learning after losses.
- Requires signal count >= MIN_SIGNALS as a robustness check (not just a score
  from a thin data slice).
- execution_5m and trend_15m are score CONTRIBUTORS, not binary gates.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Minimum number of independent bullish/bearish signals required to route
MIN_BULL_SIGNALS = 5   # out of ~10 checked
MIN_BEAR_SIGNALS = 5
MIN_RANGE_SIGNALS = 4  # out of ~7 checked

# Fixed entry threshold — NOT the adaptive params threshold, which EOD learning
# can raise too high after losses. The adaptive threshold should inform sizing,
# not block valid entries with strong multi-signal consensus.
BULL_SCORE_THRESHOLD = 4.0
BEAR_SCORE_THRESHOLD = 4.0
SCORE_MARGIN = 0.6   # bull/bear score must exceed no_trade_score by this much


@dataclass
class MarketView:
    direction: str = "NEUTRAL"   # "BULL" | "BEAR" | "RANGE" | "NEUTRAL"
    strength: float = 0.0        # raw bull or bear score from metadata
    signal_count: int = 0        # number of independent signals that aligned
    total_signals: int = 0       # total signals checked
    archetype: str = "UNCLASSIFIED"
    playbook: str = "NO_TRADE"
    reasons: list[str] = field(default_factory=list)
    signal_detail: dict[str, str] = field(default_factory=dict)
    strike_guidance: dict = field(default_factory=dict)
    directed: bool = False       # True means MarketView drove the regime decision


def _vwap_balance(meta: dict) -> tuple[str, float]:
    bav = int(meta.get("bars_above_vwap") or 0)
    bbv = int(meta.get("bars_below_vwap") or 0)
    total = max(bav + bbv, 1)
    ratio = bav / total
    if ratio >= 0.75:
        return "BULL", ratio
    if ratio <= 0.25:
        return "BEAR", 1.0 - ratio
    return "NEUTRAL", max(ratio, 1.0 - ratio)


def count_directional_signals(meta: dict, spot: float) -> tuple[int, int, int, dict[str, str]]:
    """
    Count independent bull/bear/range signals from metadata.
    Returns (bull_count, bear_count, range_count, signal_detail).

    Each signal is binary (+1 bull, +1 bear, or neutral) — the SCORE already
    incorporates weighted signal combination; this count is for robustness
    (prevents a single very-high-weight signal from overriding 9 opposing ones).
    """
    bull = bear = rng = 0
    detail: dict[str, str] = {}

    # 1. VWAP balance
    vwap_dir, _ = _vwap_balance(meta)
    if vwap_dir == "BULL":
        bull += 1; detail["vwap"] = "BULL"
    elif vwap_dir == "BEAR":
        bear += 1; detail["vwap"] = "BEAR"
    else:
        detail["vwap"] = "NEUTRAL"; rng += 1

    # 2. Opening range break
    or_state = str(meta.get("opening_range_break_state") or "NONE")
    if or_state == "UP":
        bull += 1; detail["or_break"] = "BULL"
    elif or_state in {"DOWN", "FAILED_UP"}:
        bear += 1; detail["or_break"] = "BEAR"
    else:
        detail["or_break"] = "NEUTRAL"; rng += 1

    # 3. Confirmed breakout / breakdown
    if meta.get("accepted_breakout"):
        bull += 1; detail["confirmed_direction"] = "BULL"
    elif meta.get("accepted_breakdown"):
        bear += 1; detail["confirmed_direction"] = "BEAR"
    else:
        detail["confirmed_direction"] = "NEUTRAL"

    # 4. Order flow imbalance
    ofi = float(meta.get("order_flow_imbalance") or 0.0)
    if ofi >= 0.1:
        bull += 1; detail["ofi"] = "BULL"
    elif ofi <= -0.1:
        bear += 1; detail["ofi"] = "BEAR"
    else:
        detail["ofi"] = "NEUTRAL"; rng += 1

    # 5. FII / institutional flow
    fii = str(meta.get("fii_bias") or "NEUTRAL")
    if fii in {"BULLISH", "STRONG_BULLISH"}:
        bull += 1; detail["fii"] = "BULL"
    elif fii in {"BEARISH", "STRONG_BEARISH"}:
        bear += 1; detail["fii"] = "BEAR"
    else:
        detail["fii"] = "NEUTRAL"; rng += 1

    # 6. Smart money / OI flow
    smb = str(meta.get("smart_money_bias") or "NEUTRAL")
    if smb == "BULLISH":
        bull += 1; detail["smart_money"] = "BULL"
    elif smb == "BEARISH":
        bear += 1; detail["smart_money"] = "BEAR"
    else:
        detail["smart_money"] = "NEUTRAL"; rng += 1

    # 7. Put wall proximity (floor support for bull)
    pw = meta.get("multi_session_put_wall")
    if pw:
        d = spot - float(pw)
        if d >= 40:
            bull += 1; detail["put_wall"] = "BULL"
        elif d < 0:
            bear += 1; detail["put_wall"] = "BEAR"
        else:
            detail["put_wall"] = "NEUTRAL"
    else:
        detail["put_wall"] = "ABSENT"

    # 8. Call wall proximity (ceiling for bear)
    cw = meta.get("multi_session_call_wall")
    if cw:
        d = float(cw) - spot
        if d <= 30:
            bear += 1; detail["call_wall"] = "BEAR"
        elif d >= 80:
            bull += 1; detail["call_wall"] = "BULL"
        else:
            detail["call_wall"] = "NEUTRAL"; rng += 1
    else:
        detail["call_wall"] = "ABSENT"

    # 9. Bullish vs bearish trend quality (pre-computed composite)
    btq = float(meta.get("bullish_trend_quality_score") or 0.0)
    bbtq = float(meta.get("bearish_trend_quality_score") or 0.0)
    if btq >= 3.0 and btq > bbtq:
        bull += 1; detail["trend_quality"] = "BULL"
    elif bbtq >= 3.0 and bbtq > btq:
        bear += 1; detail["trend_quality"] = "BEAR"
    else:
        detail["trend_quality"] = "NEUTRAL"; rng += 1

    # 10. execution_5m (5-minute price action confirmation — score contributor, not gate)
    exec5m = str(meta.get("execution_5m_view") or "UNCONFIRMED")
    if exec5m == "UP_CONFIRMED":
        bull += 1; detail["execution_5m"] = "BULL"
    elif exec5m == "DOWN_CONFIRMED":
        bear += 1; detail["execution_5m"] = "BEAR"
    elif exec5m == "RANGE_CONFIRMED":
        rng += 1; detail["execution_5m"] = "RANGE"
    else:
        detail["execution_5m"] = "NEUTRAL"

    return bull, bear, rng, detail


def _infer_bull_playbook(meta: dict, bull_signals: int) -> tuple[str, str]:
    """Return (archetype, playbook) label based on what signals drove the bull view."""
    or_up = str(meta.get("opening_range_break_state") or "") == "UP"
    or_failed_down = str(meta.get("opening_range_break_state") or "") == "FAILED_DOWN"
    gap_up = bool(meta.get("gap_up"))
    gap_down = bool(meta.get("gap_down"))

    if gap_up and or_up:
        return "GAP_UP_CONTINUATION", "GAP_UP_BULLISH_CONTINUATION"
    if gap_down and bool(meta.get("accepted_breakout")):
        return "GAP_DOWN_RECOVERY", "GAP_DOWN_BULLISH_RECOVERY"
    if or_up and bool(meta.get("accepted_breakout")):
        return "TREND_BULLISH", "HIGH_CONFLUENCE_BULLISH_CONTINUATION"
    if or_failed_down:
        return "TREND_BULLISH", "GAP_DOWN_BULLISH_RECOVERY"
    if bull_signals >= 8:
        return "STRONG_BULL_CONSENSUS", "HIGH_CONFLUENCE_BULLISH_CONTINUATION"
    return "MARKET_VIEW_BULL", "MARKET_VIEW_BULL"


def _infer_bear_playbook(meta: dict, bear_signals: int) -> tuple[str, str]:
    or_down = str(meta.get("opening_range_break_state") or "") == "DOWN"
    or_failed_up = str(meta.get("opening_range_break_state") or "") == "FAILED_UP"
    gap_down = bool(meta.get("gap_down"))
    gap_up = bool(meta.get("gap_up"))

    if gap_down and or_down:
        return "GAP_DOWN_CONTINUATION", "GAP_DOWN_BEARISH_CONTINUATION"
    if gap_up and or_failed_up:
        return "GAP_UP_FAILURE", "GAP_UP_BEARISH_FAILURE"
    if or_failed_up or or_down:
        return "SIDEWAYS_TO_BEARISH", "SIDEWAYS_TO_BEARISH_REJECTION"
    if bear_signals >= 8:
        return "STRONG_BEAR_CONSENSUS", "HIGH_CONFLUENCE_BEARISH_CONTINUATION"
    return "MARKET_VIEW_BEAR", "MARKET_VIEW_BEAR"


def _strike_guidance_bull(meta: dict, spot: float) -> dict:
    pw = meta.get("multi_session_put_wall")
    anchor = (float(pw) - 25.0) if pw else (spot - 100.0)
    return {
        "max_short_put_strike": anchor,
        "allowed_width_points": (75.0, 100.0),
        "preferred_width_points": 75.0,
        "minimum_net_edge_rupees": 750.0,
    }


def _strike_guidance_bear(meta: dict, spot: float) -> dict:
    cw = meta.get("multi_session_call_wall")
    anchor = (float(cw) + 25.0) if cw else (spot + 100.0)
    return {
        "min_short_call_strike": anchor,
        "allowed_width_points": (75.0, 100.0),
        "preferred_width_points": 75.0,
        "minimum_net_edge_rupees": 750.0,
    }


def derive_market_view(
    metadata: dict,
    spot: float,
    bull_score: float,
    bear_score: float,
    no_trade_score: float,
    range_score: float = 0.0,
) -> MarketView:
    """
    Derive a structured market view from computed signal scores.

    This is the intelligence core. It counts independent directional signals
    and combines them with the existing composite scores (bull_score, bear_score)
    to produce a clear BULL / BEAR / RANGE / NEUTRAL view.

    The fixed threshold (BULL_SCORE_THRESHOLD = 4.0) prevents the adaptive
    learning system from raising thresholds high enough to block valid setups.
    Adaptive learning should adjust SIZING and CONFIDENCE, not entry gates.
    """
    # Write execution_5m into metadata so count_directional_signals can read it
    # (execution_5m is stored as a regime_state attribute, not a metadata key;
    # the caller should set "execution_5m_view" before calling this function)

    bull_sig, bear_sig, range_sig, detail = count_directional_signals(metadata, spot)
    total = bull_sig + bear_sig + range_sig

    reasons: list[str] = []

    bull_edge = bull_score - max(no_trade_score + SCORE_MARGIN, BULL_SCORE_THRESHOLD)
    bear_edge = bear_score - max(no_trade_score + SCORE_MARGIN, BEAR_SCORE_THRESHOLD)

    if (
        bull_edge > 0
        and bull_sig >= MIN_BULL_SIGNALS
        and bull_edge >= bear_edge
    ):
        archetype, playbook = _infer_bull_playbook(metadata, bull_sig)
        reasons.append(
            f"MarketView: BULL ({bull_sig}/{total} signals, score {bull_score:.2f}). "
            f"Signals: {', '.join(k for k, v in detail.items() if v == 'BULL')}."
        )
        return MarketView(
            direction="BULL",
            strength=bull_score,
            signal_count=bull_sig,
            total_signals=total,
            archetype=archetype,
            playbook=playbook,
            reasons=reasons,
            signal_detail=detail,
            strike_guidance=_strike_guidance_bull(metadata, spot),
            directed=True,
        )

    if (
        bear_edge > 0
        and bear_sig >= MIN_BEAR_SIGNALS
        and bear_edge > bull_edge
    ):
        archetype, playbook = _infer_bear_playbook(metadata, bear_sig)
        reasons.append(
            f"MarketView: BEAR ({bear_sig}/{total} signals, score {bear_score:.2f}). "
            f"Signals: {', '.join(k for k, v in detail.items() if v == 'BEAR')}."
        )
        return MarketView(
            direction="BEAR",
            strength=bear_score,
            signal_count=bear_sig,
            total_signals=total,
            archetype=archetype,
            playbook=playbook,
            reasons=reasons,
            signal_detail=detail,
            strike_guidance=_strike_guidance_bear(metadata, spot),
            directed=True,
        )

    # Range check
    cw = metadata.get("multi_session_call_wall")
    pw = metadata.get("multi_session_put_wall")
    if cw and pw:
        wall_spread = float(cw) - float(pw)
        d_cw = float(cw) - spot
        d_pw = spot - float(pw)
        oi_sandwich = wall_spread >= 75 and d_cw >= 40 and d_pw >= 40
    else:
        oi_sandwich = False
    ofi = abs(float(metadata.get("order_flow_imbalance") or 0.0))
    if (range_score >= 2.5 or oi_sandwich) and ofi < 0.45:
        reasons.append(
            f"MarketView: RANGE (range_score={range_score:.1f}, oi_sandwich={oi_sandwich}). "
            f"Neutral signals dominate."
        )
        return MarketView(
            direction="RANGE",
            strength=range_score,
            signal_count=range_sig,
            total_signals=total,
            archetype="RANGE_BALANCED",
            playbook="RANGE_BALANCED_CONDOR",
            reasons=reasons,
            signal_detail=detail,
            directed=True,
        )

    reasons.append(
        f"MarketView: NEUTRAL — insufficient consensus "
        f"(bull={bull_sig}/{total} score={bull_score:.2f}, "
        f"bear={bear_sig}/{total} score={bear_score:.2f})."
    )
    return MarketView(
        direction="NEUTRAL",
        strength=max(bull_score, bear_score),
        signal_count=max(bull_sig, bear_sig),
        total_signals=total,
        reasons=reasons,
        signal_detail=detail,
        directed=False,
    )
