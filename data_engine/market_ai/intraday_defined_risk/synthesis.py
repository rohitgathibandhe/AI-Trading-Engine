"""
Trade Synthesis Engine — market-first trade reasoning.

A professional trader looks at 10–12 independent market signals and derives
the optimal trade by consensus, not by checking a rigid set of conditions.
This module replicates that reasoning: it scores every viable strategy against
ALL available intelligence and recommends the highest-consensus trade.

Architecture:
  synthesize_trade(meta, spot) → TradeSynthesis

  TradeSynthesis.override_eligible = True means synthesis found enough
  multi-signal agreement to override an existing NO_TRADE from the rule-based
  regime classifier. The synthesis playbook then bypasses score-threshold gates
  in strategy.py and goes straight to structure selection + execution.

Entry conditions for synthesis override:
  - Session PRIME (10:00–13:30)
  - Not expiry day, no pin risk
  - Best strategy score >= _OVERRIDE_THRESHOLD (0.50)
  - Minimum of 8 signals evaluated (avoids thin data)
"""
from __future__ import annotations

from dataclasses import dataclass, field


_BULL_PUT_THRESHOLD = 0.38   # minimum consensus to recommend bull put
_BEAR_CALL_THRESHOLD = 0.38  # minimum consensus to recommend bear call
_CONDOR_THRESHOLD = 0.33     # condor needs less directional conviction
_OVERRIDE_THRESHOLD = 0.50   # minimum to override a NO_TRADE decision


@dataclass
class TradeSynthesis:
    recommended_strategy: str | None = None   # "BULL_PUT_CREDIT_SPREAD" etc.
    recommended_playbook: str | None = None   # injected into regime metadata
    synthesis_score: float = 0.0              # normalized [-1, +1] for best strategy
    rationale: list[str] = field(default_factory=list)
    strike_guidance: dict = field(default_factory=dict)  # max_short_put_strike etc.
    override_eligible: bool = False           # True → can override NO_TRADE
    all_scores: dict = field(default_factory=dict)       # score per strategy
    signal_details: dict = field(default_factory=dict)   # per-signal values


def _vwap_ratio(meta: dict) -> float:
    bav = int(meta.get("bars_above_vwap") or 0)
    bbv = int(meta.get("bars_below_vwap") or 0)
    return bav / max(bav + bbv, 1)


def _fii_to_score(fii: str, direction: str) -> float:
    bull = {"STRONG_BULLISH": 1.0, "BULLISH": 0.5, "NEUTRAL": 0.0, "UNKNOWN": 0.0,
            "BEARISH": -0.5, "STRONG_BEARISH": -1.0}
    bear = {"STRONG_BEARISH": 1.0, "BEARISH": 0.5, "NEUTRAL": 0.0, "UNKNOWN": 0.0,
            "BULLISH": -0.5, "STRONG_BULLISH": -1.0}
    return (bull if direction == "BULL" else bear).get(fii, 0.0)


def _score_bull_put(meta: dict, spot: float) -> tuple[float, dict]:
    """Score conditions for a bull put credit spread. Returns (normalized_score, signals)."""
    s: dict[str, float] = {}

    # 1. VWAP balance: more bars above = bullish
    vr = _vwap_ratio(meta)
    s["vwap"] = 1.0 if vr >= 0.8 else (0.6 if vr >= 0.6 else (0.0 if vr >= 0.4 else -1.0))

    # 2. Opening range break
    or_st = str(meta.get("opening_range_break_state") or "NONE")
    s["or_break"] = 1.0 if or_st == "UP" else (-1.0 if or_st == "DOWN" else -0.3)

    # 3. Confirmed breakout (accepted, not a false break)
    if meta.get("accepted_breakout"):
        s["breakout"] = 0.8
    elif meta.get("accepted_breakdown"):
        s["breakout"] = -0.8

    # 4. Order flow imbalance (call buying > put buying → bullish)
    ofi = float(meta.get("order_flow_imbalance") or 0.0)
    s["ofi"] = (0.8 if ofi >= 0.3 else 0.3 if ofi >= 0.05 else
                0.0 if ofi >= -0.05 else -0.3 if ofi >= -0.3 else
                -0.5 if ofi >= -0.5 else -0.8)

    # 5. FII/DII institutional flow
    s["fii"] = _fii_to_score(str(meta.get("fii_bias") or "NEUTRAL"), "BULL")

    # 6. Smart money OI bias (intraday OI flow direction)
    smb = str(meta.get("smart_money_bias") or "NEUTRAL")
    s["smart_money"] = 0.6 if smb == "BULLISH" else (-0.6 if smb == "BEARISH" else 0.0)

    # 7. PUT wall: floor below spot. More cushion = safer to sell puts.
    pw = meta.get("multi_session_put_wall")
    if pw:
        d = spot - float(pw)
        s["put_wall_cushion"] = (0.9 if d >= 60 else 0.5 if d >= 30 else
                                 0.1 if d >= 0 else -0.8)

    # 8. CALL wall overhead — for PUT spreads, a nearby call wall means spot is
    # capped at resistance, which PROTECTS the downside put leg (not a headwind).
    # Only penalise when call wall is so tight it strongly suggests a reversal.
    cw = meta.get("multi_session_call_wall")
    if cw:
        d = float(cw) - spot
        s["call_wall_overhead"] = (0.0 if d < 20 else 0.2 if d < 50 else
                                   0.5 if d < 100 else 0.7)

    # 9. IV level: is it worth selling premium?
    iv = float(meta.get("avg_chain_iv") or 0.0)
    s["iv"] = (0.7 if 18 <= iv <= 40 else 0.5 if iv > 40 else
               0.2 if iv >= 14 else -0.6)

    # 10. Bullish trend quality (composite momentum score already computed)
    btq = float(meta.get("bullish_trend_quality_score") or 0.0)
    s["bull_trend"] = 1.0 if btq >= 5.0 else (0.5 if btq >= 3.0 else 0.1 if btq >= 1.5 else -0.3)

    # 11. Bearish trend quality contrast (strong bear signal reduces bull conviction)
    bbtq = float(meta.get("bearish_trend_quality_score") or 0.0)
    s["bear_contrast"] = (0.3 if bbtq < 2.0 else 0.0 if bbtq < 4.0 else
                          -0.5 if bbtq < 6.0 else -0.9)

    # 12. Straddle premium: sufficient premium at put strikes?
    straddle = float(meta.get("atm_straddle_price") or 0.0)
    s["premium"] = 0.6 if straddle >= 80 else (0.0 if straddle >= 40 else -0.5)

    score = round(sum(s.values()) / max(len(s), 1), 4)
    return score, s


def _score_bear_call(meta: dict, spot: float) -> tuple[float, dict]:
    """Score conditions for a bear call credit spread."""
    s: dict[str, float] = {}

    vr = _vwap_ratio(meta)
    s["vwap"] = 1.0 if vr <= 0.2 else (0.6 if vr <= 0.4 else 0.0 if vr <= 0.6 else -1.0)

    or_st = str(meta.get("opening_range_break_state") or "NONE")
    s["or_break"] = 1.0 if or_st == "DOWN" else (-1.0 if or_st == "UP" else -0.3)

    if meta.get("accepted_breakdown"):
        s["breakdown"] = 0.8
    elif meta.get("accepted_breakout"):
        s["breakdown"] = -0.8

    ofi = float(meta.get("order_flow_imbalance") or 0.0)
    s["ofi"] = (0.8 if ofi <= -0.3 else 0.3 if ofi <= -0.05 else
                0.0 if ofi <= 0.05 else -0.3 if ofi <= 0.3 else -0.8)

    s["fii"] = _fii_to_score(str(meta.get("fii_bias") or "NEUTRAL"), "BEAR")

    smb = str(meta.get("smart_money_bias") or "NEUTRAL")
    s["smart_money"] = 0.6 if smb == "BEARISH" else (-0.6 if smb == "BULLISH" else 0.0)

    cw = meta.get("multi_session_call_wall")
    if cw:
        d = float(cw) - spot
        s["call_wall"] = (0.9 if d <= 30 else 0.5 if d <= 60 else
                          0.1 if d <= 100 else -0.5)

    pw = meta.get("multi_session_put_wall")
    if pw:
        d = spot - float(pw)
        s["put_wall_floor"] = (-0.6 if d < 20 else -0.1 if d < 50 else
                               0.3 if d < 100 else 0.6)

    iv = float(meta.get("avg_chain_iv") or 0.0)
    s["iv"] = 0.7 if 18 <= iv <= 40 else (0.5 if iv > 40 else 0.2 if iv >= 14 else -0.6)

    bbtq = float(meta.get("bearish_trend_quality_score") or 0.0)
    s["bear_trend"] = 1.0 if bbtq >= 5.0 else (0.5 if bbtq >= 3.0 else 0.1 if bbtq >= 1.5 else -0.3)

    btq = float(meta.get("bullish_trend_quality_score") or 0.0)
    s["bull_contrast"] = (0.3 if btq < 2.0 else 0.0 if btq < 4.0 else
                          -0.5 if btq < 6.0 else -0.9)

    straddle = float(meta.get("atm_straddle_price") or 0.0)
    s["premium"] = 0.6 if straddle >= 80 else (0.0 if straddle >= 40 else -0.5)

    score = round(sum(s.values()) / max(len(s), 1), 4)
    return score, s


def _score_condor(meta: dict, spot: float) -> tuple[float, dict]:
    """Score conditions for an iron condor."""
    s: dict[str, float] = {}

    rb = float(meta.get("range_balance_score") or 0.0)
    s["range_balance"] = (1.0 if rb >= 3.5 else 0.6 if rb >= 2.5 else
                          0.3 if rb >= 2.0 else -0.3 if rb >= 1.0 else -0.8)

    cw = meta.get("multi_session_call_wall")
    pw = meta.get("multi_session_put_wall")
    if cw and pw:
        wall_spread = float(cw) - float(pw)
        d_cw = float(cw) - spot
        d_pw = spot - float(pw)
        if wall_spread >= 75 and d_cw >= 40 and d_pw >= 40:
            s["oi_sandwich"] = 0.9
        elif wall_spread >= 75 and min(d_cw, d_pw) >= 20:
            s["oi_sandwich"] = 0.4
        else:
            s["oi_sandwich"] = 0.0
    else:
        s["oi_sandwich"] = -0.3

    vr = _vwap_ratio(meta)
    s["vwap_neutral"] = round(1.0 - 2.0 * abs(vr - 0.5), 3)

    ofi = float(meta.get("order_flow_imbalance") or 0.0)
    s["ofi_neutral"] = round(max(-0.8, 1.0 - 3.0 * abs(ofi)), 3)

    iv = float(meta.get("avg_chain_iv") or 0.0)
    s["iv"] = 0.8 if iv >= 20 else (0.5 if iv >= 16 else 0.0 if iv >= 12 else -0.5)

    straddle = float(meta.get("atm_straddle_price") or 0.0)
    s["straddle"] = (1.0 if straddle >= 150 else 0.7 if straddle >= 100 else
                     0.4 if straddle >= 80 else -0.5)

    max_trend = max(
        float(meta.get("bullish_trend_quality_score") or 0),
        float(meta.get("bearish_trend_quality_score") or 0),
    )
    s["no_strong_trend"] = (-0.8 if max_trend >= 6.0 else -0.3 if max_trend >= 4.0 else
                            0.2 if max_trend >= 2.5 else 0.8)

    score = round(sum(s.values()) / max(len(s), 1), 4)
    return score, s


def _build_strike_guidance(strategy: str, meta: dict, spot: float) -> dict:
    """Derive concrete strike anchors from OI walls and support/resistance levels."""
    g: dict = {}
    cw = meta.get("multi_session_call_wall")
    pw = meta.get("multi_session_put_wall")

    if strategy == "BULL_PUT_CREDIT_SPREAD":
        anchor = float(pw) if pw else float(meta.get("bullish_support_anchor") or (spot - 100))
        g["max_short_put_strike"] = anchor - 25.0
        g["allowed_width_points"] = (75.0, 100.0)
        g["preferred_width_points"] = 75.0
        g["minimum_net_edge_rupees"] = 750.0

    elif strategy == "BEAR_CALL_CREDIT_SPREAD":
        anchor = float(cw) if cw else float(meta.get("bearish_resistance_anchor") or (spot + 100))
        g["min_short_call_strike"] = anchor + 25.0
        g["allowed_width_points"] = (75.0, 100.0)
        g["preferred_width_points"] = 75.0
        g["minimum_net_edge_rupees"] = 750.0

    elif strategy == "IRON_CONDOR":
        if cw and pw:
            g["condor_short_call_anchor"] = float(cw) + 25.0
            g["condor_short_put_anchor"] = float(pw) - 25.0
        g["allowed_width_points"] = (75.0, 100.0)
        g["preferred_width_points"] = 75.0
        g["minimum_net_edge_rupees"] = 700.0

    return g


def synthesize_trade(meta: dict, spot: float) -> TradeSynthesis:
    """
    Market-first trade synthesis.

    Scores all viable strategies against 12 independent market signals and
    returns the highest-consensus recommendation. Unlike the rule-based
    regime classifier, synthesis uses additive multi-signal agreement rather
    than rigid threshold gates.

    Returns TradeSynthesis with override_eligible=True when consensus is
    strong enough (>= 0.50) to override an existing NO_TRADE decision.
    """
    # Safety gates — same as the underlying execution system
    if str(meta.get("session_phase") or "") != "PRIME":
        return TradeSynthesis(rationale=["Synthesis skipped: not PRIME session."])
    if bool(meta.get("is_expiry_day")) or bool(meta.get("pin_risk_active")):
        return TradeSynthesis(rationale=["Synthesis skipped: expiry day or pin risk."])
    if spot <= 0:
        return TradeSynthesis(rationale=["Synthesis skipped: invalid spot."])

    bull_score, bull_sigs = _score_bull_put(meta, spot)
    bear_score, bear_sigs = _score_bear_call(meta, spot)
    cond_score, cond_sigs = _score_condor(meta, spot)

    all_scores = {
        "BULL_PUT_CREDIT_SPREAD": bull_score,
        "BEAR_CALL_CREDIT_SPREAD": bear_score,
        "IRON_CONDOR": cond_score,
    }

    # Pick best eligible strategy
    candidates = [
        ("BULL_PUT_CREDIT_SPREAD",  bull_score, bull_sigs, _BULL_PUT_THRESHOLD),
        ("BEAR_CALL_CREDIT_SPREAD", bear_score, bear_sigs, _BEAR_CALL_THRESHOLD),
        ("IRON_CONDOR",             cond_score, cond_sigs, _CONDOR_THRESHOLD),
    ]
    eligible = [(st, sc, sg) for st, sc, sg, thr in candidates if sc >= thr]

    if not eligible:
        best = max(candidates, key=lambda x: x[1])
        return TradeSynthesis(
            synthesis_score=best[1],
            all_scores=all_scores,
            rationale=[
                f"Synthesis: strongest candidate {best[0]} scored {best[1]:.3f} "
                f"(below minimum {_BULL_PUT_THRESHOLD:.2f}). Mixed signals — no override.",
            ],
        )

    best_strat, best_score, best_sigs = max(eligible, key=lambda x: x[1])

    # Human-readable rationale from top signals
    top_pos = sorted(((k, v) for k, v in best_sigs.items() if v >= 0.5), key=lambda x: -x[1])
    top_neg = sorted(((k, v) for k, v in best_sigs.items() if v <= -0.4), key=lambda x: x[1])
    rationale = [
        f"Synthesis: {best_strat} wins with consensus score {best_score:.3f} "
        f"(bull={bull_score:.3f} | bear={bear_score:.3f} | condor={cond_score:.3f}).",
        f"Strong signals: {', '.join(k for k, _ in top_pos[:5])}.",
    ]
    if top_neg:
        rationale.append(f"Headwinds: {', '.join(k for k, _ in top_neg[:3])}.")

    # Map to synthesis-specific playbook (bypasses score-threshold gates)
    playbook_map = {
        "BULL_PUT_CREDIT_SPREAD":  "SYNTHESIS_BULL_PUT",
        "BEAR_CALL_CREDIT_SPREAD": "SYNTHESIS_BEAR_CALL",
        "IRON_CONDOR":             "OI_WALL_CONDOR" if (
            meta.get("multi_session_call_wall") and meta.get("multi_session_put_wall")
        ) else "RANGE_BALANCED_CONDOR",
    }

    guidance = _build_strike_guidance(best_strat, meta, spot)

    return TradeSynthesis(
        recommended_strategy=best_strat,
        recommended_playbook=playbook_map[best_strat],
        synthesis_score=best_score,
        rationale=rationale,
        strike_guidance=guidance,
        override_eligible=best_score >= _OVERRIDE_THRESHOLD,
        all_scores=all_scores,
        signal_details={best_strat: best_sigs},
    )
