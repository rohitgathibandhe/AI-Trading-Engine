"""Dealer Gamma Exposure (GEX) — the pin-vs-trend regime read the pros use intraday.

Options dealers hedge their books delta-neutral. WHEN they are net LONG gamma they
buy dips / sell rips to stay hedged — which SUPPRESSES realised vol (price pins,
mean-reverts to big-OI strikes). When they are net SHORT gamma they must chase —
buy strength / sell weakness — which AMPLIFIES moves (trends, accelerations). The
"zero-gamma" flip level is the spot where the book crosses from one regime to the
other: above it tends to pin, below it tends to trend.

Nifty chains give us OI + IV per strike but not gamma, so we compute gamma from
Black-Scholes (spot, strike, IV, time-to-expiry). The absolute rupee scale of GEX
is convention-dependent and irrelevant here — what matters is the SIGN (long vs
short gamma) and the flip level, both of which are scale-invariant.

IMPORTANT: this module only COMPUTES the read. Whether (and which way) GEX should
tilt a decision is a question for the ordering-robust backtest, not an assumption —
the sign convention below is the textbook one but must be validated before it gates
a single trade (same discipline as the condor/blackout work).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log, sqrt, exp, pi
from typing import Any, Iterable

LONG_GAMMA = "LONG_GAMMA"     # dealers dampen — expect pin / mean-reversion
SHORT_GAMMA = "SHORT_GAMMA"   # dealers chase — expect trend / acceleration
NEUTRAL_GAMMA = "NEUTRAL_GAMMA"

_TRADING_DAYS = 252.0
_MIN_T = 0.5 / (_TRADING_DAYS * 6.5)   # floor T at ~half a trading-hour to avoid blow-up


def _norm_pdf(x: float) -> float:
    return exp(-0.5 * x * x) / sqrt(2.0 * pi)


def _bs_gamma(spot: float, strike: float, iv_pct: float, t_years: float, r: float = 0.0) -> float:
    """Black-Scholes gamma. iv_pct is annualised IV in percent (e.g. 14.0 = 14%)."""
    sigma = iv_pct / 100.0
    if spot <= 0 or strike <= 0 or sigma <= 0 or t_years <= 0:
        return 0.0
    d1 = (log(spot / strike) + (r + 0.5 * sigma * sigma) * t_years) / (sigma * sqrt(t_years))
    return _norm_pdf(d1) / (spot * sigma * sqrt(t_years))


@dataclass
class GexState:
    net_gex: float = 0.0          # scaled net dealer gamma (sign is what matters)
    regime: str = NEUTRAL_GAMMA
    flip_strike: float | None = None   # zero-gamma level
    spot: float = 0.0
    distance_to_flip_pct: float | None = None   # (spot - flip)/spot * 100
    call_gex: float = 0.0
    put_gex: float = 0.0
    total_oi: float = 0.0
    strikes_used: int = 0
    notes: list[str] = field(default_factory=list)


def _f(v: Any, d: float = 0.0) -> float:
    try:
        return float(v) if v is not None else d
    except (TypeError, ValueError):
        return d


def compute_gex(
    rows: Iterable[dict[str, Any]],
    spot: float,
    t_years: float,
    *,
    strike_key: str = "strike",
    type_key: str = "option_type",
    iv_key: str = "iv",
    oi_key: str = "oi",
    contract_multiplier: float = 65.0,
) -> GexState:
    """Aggregate dealer GEX from a chain snapshot.

    rows: per-strike-per-type dicts (one row per option leg) with strike/type/iv/oi.
    Convention: dealers long call gamma, short put gamma (textbook). Net GEX per
    strike = (gamma_call*OI_call - gamma_put*OI_put) * spot^2 * mult * 0.01.
    Positive net => LONG gamma (pin); negative => SHORT gamma (trend).
    """
    st = GexState(spot=round(spot, 2))
    if spot <= 0:
        return st
    t = max(t_years, _MIN_T)

    # accumulate per-strike call/put gamma*OI, and the running net for the flip scan
    by_strike: dict[float, dict[str, float]] = {}
    for row in rows:
        k = _f(row.get(strike_key))
        if k <= 0:
            continue
        typ = str(row.get(type_key) or "").upper()
        iv = _f(row.get(iv_key))
        oi = _f(row.get(oi_key))
        if iv <= 0 or oi <= 0:
            continue
        g = _bs_gamma(spot, k, iv, t)
        slot = by_strike.setdefault(k, {"call": 0.0, "put": 0.0})
        if typ in ("CE", "CALL", "C"):
            slot["call"] += g * oi
            st.total_oi += oi
        elif typ in ("PE", "PUT", "P"):
            slot["put"] += g * oi
            st.total_oi += oi

    if not by_strike:
        return st

    scale = spot * spot * contract_multiplier * 0.01
    for k, slot in by_strike.items():
        st.call_gex += slot["call"] * scale
        st.put_gex += slot["put"] * scale
    st.net_gex = round(st.call_gex - st.put_gex, 2)
    st.call_gex = round(st.call_gex, 2)
    st.put_gex = round(st.put_gex, 2)
    st.strikes_used = len(by_strike)

    # Zero-gamma flip level: the strike at which cumulative net gamma flips sign,
    # scanning strikes low->high (dealer net gamma profile as spot rises).
    strikes = sorted(by_strike)
    cum = 0.0
    prev_k = None
    prev_cum = 0.0
    for k in strikes:
        slot = by_strike[k]
        step = (slot["call"] - slot["put"]) * scale
        new_cum = cum + step
        if prev_k is not None and (prev_cum < 0 <= new_cum or prev_cum > 0 >= new_cum) and (new_cum != prev_cum):
            # linear-interpolate the crossing strike
            frac = -prev_cum / (new_cum - prev_cum)
            st.flip_strike = round(prev_k + frac * (k - prev_k), 1)
            break
        prev_k, prev_cum = k, new_cum
        cum = new_cum

    if st.net_gex > 0:
        st.regime = LONG_GAMMA
        st.notes.append(f"net GEX +{st.net_gex:,.0f} → dealers LONG gamma (expect pin / mean-reversion)")
    elif st.net_gex < 0:
        st.regime = SHORT_GAMMA
        st.notes.append(f"net GEX {st.net_gex:,.0f} → dealers SHORT gamma (expect trend / acceleration)")

    if st.flip_strike is not None:
        st.distance_to_flip_pct = round((spot - st.flip_strike) / spot * 100.0, 2)
        st.notes.append(f"zero-gamma flip @ {st.flip_strike:.0f} (spot {st.distance_to_flip_pct:+.2f}% away)")
    return st
