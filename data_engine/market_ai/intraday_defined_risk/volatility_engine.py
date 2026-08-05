"""
Volatility Engine — trade VOL, not direction (the professional's core edge).

The structural edge in options is the variance risk premium: implied vol tends to be
priced richer than realized vol turns out to be. Pros SELL premium when it is rich and
BUY when it is cheap, then let direction/structure be secondary. Our agent had every
raw ingredient (broker IV via avg_chain_iv, India VIX, realized range, ATR, IV history)
but never COMPARED them, so it guessed direction instead of trading vol.

This module answers, every cycle: is volatility RICH or CHEAP right now?
  - Implied vol  = avg_chain_iv (broker per-option IV, annualized %) | India VIX fallback
  - Realized vol = annualized from daily ATR:  (ATR/spot) * sqrt(252) * 100
  - VRP          = implied - realized  (+ => premium rich to SELL; - => cheap to BUY)
  - IV rank      = percentile of today's implied vol vs its own history (sell high, buy low)
  -> VolRegime: RICH_SELL / CHEAP_BUY / NEUTRAL  + a confidence and human notes.

The selector consults this FIRST (sell vs buy premium), then picks direction/structure.
Deterministic, evidence-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import sqrt
from typing import Any

RICH_SELL = "RICH_SELL"
CHEAP_BUY = "CHEAP_BUY"
NEUTRAL = "NEUTRAL"

_TRADING_DAYS = 252
_VRP_RICH = 3.0    # implied richer than realized by >3 vol points → sell edge
_VRP_CHEAP = -2.0  # realized exceeds implied → buying/long-vol favoured
_IV_RANK_HIGH = 0.65
_IV_RANK_LOW = 0.30


def _f(m: dict, k: str, d: float = 0.0) -> float:
    v = m.get(k)
    try:
        return float(v) if v is not None else d
    except (TypeError, ValueError):
        return d


@dataclass
class VolState:
    implied_vol: float = 0.0        # annualized %
    realized_vol: float = 0.0       # annualized %
    vrp: float = 0.0                # implied - realized
    iv_rank: float | None = None    # 0..1 percentile vs history
    regime: str = NEUTRAL
    confidence: float = 0.0
    notes: list[str] = field(default_factory=list)


def compute_iv_rank(current_iv: float, history_ivs: list[float]) -> float | None:
    hist = [h for h in history_ivs if h and h > 0]
    if len(hist) < 10 or current_iv <= 0:
        return None
    below = sum(1 for h in hist if h <= current_iv)
    return round(below / len(hist), 3)


def assess_vol(metadata: dict[str, Any], *, iv_history: list[float] | None = None) -> VolState:
    st = VolState()
    spot = _f(metadata, "nifty_spot", _f(metadata, "atm_strike"))

    # Implied vol: broker chain IV first, India VIX fallback
    implied = _f(metadata, "avg_chain_iv")
    if implied <= 0:
        implied = _f(metadata, "india_vix")
    st.implied_vol = round(implied, 2)

    # Realized vol annualized from daily ATR (or a straddle-implied fallback for realized move)
    atr = _f(metadata, "daily_atr")
    if atr > 0 and spot > 0:
        st.realized_vol = round((atr / spot) * sqrt(_TRADING_DAYS) * 100.0, 2)

    if st.implied_vol > 0 and st.realized_vol > 0:
        st.vrp = round(st.implied_vol - st.realized_vol, 2)

    if iv_history:
        st.iv_rank = compute_iv_rank(st.implied_vol, iv_history)

    # ── decide the regime ───────────────────────────────────────────────────
    rich_votes = cheap_votes = 0
    if st.vrp >= _VRP_RICH:
        rich_votes += 1; st.notes.append(f"IV {st.implied_vol:.0f}% vs realized {st.realized_vol:.0f}% → +{st.vrp:.0f} VRP (premium rich)")
    elif st.vrp <= _VRP_CHEAP:
        cheap_votes += 1; st.notes.append(f"realized {st.realized_vol:.0f}% > IV {st.implied_vol:.0f}% → {st.vrp:.0f} VRP (premium cheap / underpricing move)")
    else:
        st.notes.append(f"VRP {st.vrp:+.0f} (implied~realized)")

    if st.iv_rank is not None:
        if st.iv_rank >= _IV_RANK_HIGH:
            rich_votes += 1; st.notes.append(f"IV rank {st.iv_rank:.0%} (high vs its history → sell)")
        elif st.iv_rank <= _IV_RANK_LOW:
            cheap_votes += 1; st.notes.append(f"IV rank {st.iv_rank:.0%} (low vs its history → buy)")

    if rich_votes > cheap_votes:
        st.regime = RICH_SELL
    elif cheap_votes > rich_votes:
        st.regime = CHEAP_BUY
    else:
        st.regime = NEUTRAL
    st.confidence = round(min(1.0, 0.4 + 0.3 * abs(rich_votes - cheap_votes)), 2)
    return st
