"""
Decision Justification — the agent's reasoning, made explicit and auditable.

Every trade should answer: what does the OPTION CHAIN say? what does the CHART say?
do they AGREE? and therefore — why THIS strategy, THIS strike, THIS time, THIS exit?
The signals already exist across the codebase; this module assembles them into one
coherent thesis and grades the CONFLUENCE (how many independent dimensions point the
same way). Confluence is the honest proxy for conviction: a trade backed by option-
flow AND chart structure AND multi-timeframe levels is higher-quality than one backed
by a single signal — and the retrospective can then test whether high-confluence
trades actually fail less.

Pure/read-only: it interprets metadata, it does not place trades.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _f(m: dict, k: str, d: float = 0.0) -> float:
    v = m.get(k)
    try:
        return float(v) if v is not None else d
    except (TypeError, ValueError):
        return d


@dataclass
class DimensionRead:
    name: str
    bias: str            # BULLISH / BEARISH / NEUTRAL
    points: list[str] = field(default_factory=list)   # the evidence, human-readable


@dataclass
class TradeThesis:
    spot: float
    option_chain: DimensionRead = None
    chart: DimensionRead = None
    levels: DimensionRead = None
    net_bias: str = "NEUTRAL"
    confluence: int = 0            # how many of the 3 dimensions agree with net_bias
    conviction: float = 0.0        # 0..1
    strategy: str = ""
    strategy_why: str = ""
    entry_why: str = ""
    exit_why: str = ""

    def to_text(self) -> str:
        L = [f"TRADE THESIS @ spot {self.spot:.0f}  —  net {self.net_bias} (confluence {self.confluence}/3, conviction {self.conviction:.2f})"]
        for dim in (self.option_chain, self.chart, self.levels):
            if dim is None:
                continue
            L.append(f"  [{dim.name}] {dim.bias}: " + " | ".join(dim.points) if dim.points else f"  [{dim.name}] {dim.bias}: (no clear signal)")
        if self.strategy:
            L.append(f"  [STRATEGY] {self.strategy} — {self.strategy_why}")
        if self.entry_why:
            L.append(f"  [ENTRY] {self.entry_why}")
        if self.exit_why:
            L.append(f"  [EXIT] {self.exit_why}")
        return "\n".join(L)


# ── 1. Option-chain read: OI, ΔOI, PCR, smart money, gamma/pin, flow ────────
def read_option_chain(m: dict[str, Any], spot: float) -> DimensionRead:
    pts: list[str] = []
    score = 0  # + bullish, - bearish

    pcr = _f(m, "pcr_by_oi", 1.0)
    pcr_trend = str(m.get("pcr_trend") or "UNKNOWN")
    if pcr >= 1.15 or pcr_trend == "RISING":
        score += 1; pts.append(f"PCR {pcr:.2f} {pcr_trend.lower()} → put writing (downside support = bullish)")
    elif pcr <= 0.80 or pcr_trend == "FALLING":
        score -= 1; pts.append(f"PCR {pcr:.2f} {pcr_trend.lower()} → call writing / put unwind (overhead = bearish)")
    else:
        pts.append(f"PCR {pcr:.2f} balanced")

    cw = m.get("current_call_wall"); pw = m.get("current_put_wall")
    cvel = _f(m, "call_wall_oi_velocity"); pvel = _f(m, "put_wall_oi_velocity")
    if pw is not None:
        pts.append(f"put wall {float(pw):.0f} ({'building' if pvel > 0 else 'unwinding' if pvel < 0 else 'steady'})")
    if cw is not None:
        pts.append(f"call wall {float(cw):.0f} ({'building' if cvel > 0 else 'unwinding' if cvel < 0 else 'steady'})")

    smb = str(m.get("smart_money_bias") or "NEUTRAL")
    if smb == "BULLISH":
        score += 1; pts.append("smart-money OI flow BULLISH")
    elif smb == "BEARISH":
        score -= 1; pts.append("smart-money OI flow BEARISH")

    ofi = _f(m, "order_flow_imbalance")
    if ofi >= 0.15:
        score += 1; pts.append(f"order-flow +{ofi:.2f} (call-side aggression)")
    elif ofi <= -0.15:
        score -= 1; pts.append(f"order-flow {ofi:.2f} (put-side aggression)")

    if bool(m.get("pin_risk_active")):
        mgs = m.get("max_gamma_strike")
        pts.append(f"GAMMA PIN near {float(mgs):.0f} — expect pinning / suppressed range" if mgs else "gamma pin active")

    fii = str(m.get("fii_bias") or "NEUTRAL")
    if "BULL" in fii:
        score += 1; pts.append(f"FII {fii}")
    elif "BEAR" in fii:
        score -= 1; pts.append(f"FII {fii}")

    bias = "BULLISH" if score >= 1 else "BEARISH" if score <= -1 else "NEUTRAL"
    return DimensionRead("OPTION CHAIN", bias, pts)


# ── 2. Chart read: VWAP, structure (BOS/CHoCH), candlesticks, OR break ──────
def read_chart(m: dict[str, Any], spot: float) -> DimensionRead:
    pts: list[str] = []
    score = 0

    pvv = _f(m, "price_vs_vwap")
    if pvv > 8:
        score += 1; pts.append(f"price +{pvv:.0f} above VWAP")
    elif pvv < -8:
        score -= 1; pts.append(f"price {pvv:.0f} below VWAP")
    else:
        pts.append("price at VWAP")

    if m.get("bullish_bos"):
        score += 1; pts.append("bullish BOS (break of structure up)")
    if m.get("bearish_bos"):
        score -= 1; pts.append("bearish BOS (break of structure down)")
    if m.get("bullish_choch"):
        score += 1; pts.append("bullish CHoCH (change of character up)")
    if m.get("bearish_choch"):
        score -= 1; pts.append("bearish CHoCH")

    bcp = str(m.get("bullish_candle_pattern") or "NONE")
    xcp = str(m.get("bearish_candle_pattern") or "NONE")
    if bcp not in ("NONE", ""):
        score += 1; pts.append(f"bullish candle: {bcp}")
    if xcp not in ("NONE", ""):
        score -= 1; pts.append(f"bearish candle: {xcp}")

    orb = str(m.get("opening_range_break_state") or "NONE")
    if orb == "UP":
        score += 1; pts.append("broke opening range UP")
    elif orb in ("DOWN", "FAILED_UP"):
        score -= 1; pts.append(f"opening range {orb}")

    bias = "BULLISH" if score >= 1 else "BEARISH" if score <= -1 else "NEUTRAL"
    return DimensionRead("CHART", bias, pts)


# ── 3. Multi-timeframe levels: where is price vs daily/weekly S/R ───────────
def read_levels(m: dict[str, Any], spot: float) -> DimensionRead:
    pts: list[str] = []
    score = 0
    res = m.get("daily_swing_resistance") or m.get("weekly_high")
    sup = m.get("daily_swing_support") or m.get("weekly_low")
    if res is not None:
        d = float(res) - spot
        if 0 < d <= 60:
            score -= 1; pts.append(f"just under daily resistance {float(res):.0f} ({d:.0f}pts) — capped")
        elif d > 60:
            pts.append(f"room to resistance {float(res):.0f} ({d:.0f}pts up)")
    if sup is not None:
        d = spot - float(sup)
        if 0 < d <= 60:
            score += 1; pts.append(f"just above daily support {float(sup):.0f} ({d:.0f}pts) — cushioned")
        elif d > 60:
            pts.append(f"room to support {float(sup):.0f} ({d:.0f}pts down)")
    bias = "BULLISH" if score >= 1 else "BEARISH" if score <= -1 else "NEUTRAL"
    return DimensionRead("MTF LEVELS", bias, pts)


def build_thesis(metadata: dict[str, Any], spot: float, *, strategy: str = "", strategy_why: str = "",
                 entry_why: str = "", exit_why: str = "") -> TradeThesis:
    oc = read_option_chain(metadata, spot)
    ch = read_chart(metadata, spot)
    lv = read_levels(metadata, spot)

    # net bias = majority vote across the 3 dimensions
    votes = [d.bias for d in (oc, ch, lv)]
    bull = votes.count("BULLISH"); bear = votes.count("BEARISH")
    net = "BULLISH" if bull > bear else "BEARISH" if bear > bull else "NEUTRAL"
    confluence = max(bull, bear)  # how many dimensions agree with the winning side
    # conviction scales with confluence (1/3 dims → weak, 3/3 → strong)
    conviction = round({0: 0.30, 1: 0.45, 2: 0.65, 3: 0.85}.get(confluence, 0.3), 2)

    return TradeThesis(
        spot=spot, option_chain=oc, chart=ch, levels=lv,
        net_bias=net, confluence=confluence, conviction=conviction,
        strategy=strategy, strategy_why=strategy_why, entry_why=entry_why, exit_why=exit_why,
    )
