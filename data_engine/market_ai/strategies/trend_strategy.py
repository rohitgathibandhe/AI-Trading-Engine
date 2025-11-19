from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from typing import List, Optional
import math

from .models import (
    LegSide,
    OptionLeg,
    OptionType,
    StrategyType,
    TradeAction,
    TrendContext,
    TrendSide,
)


@dataclass
class TrendParams:
    rsi_trend_up: float = 55.0
    rsi_trend_down: float = 45.0
    atr_trend_threshold: float = 130.0
    lookback_swings: int = 10
    min_move_points: float = 80.0
    short_delta_min: float = 0.25
    short_delta_max: float = 0.30
    hedge_delta_min: float = 0.05
    hedge_delta_max: float = 0.10
    fallback_gap_short: int = 180
    fallback_gap_hedge: int = 320


class TrendStrategyEngine:
    def __init__(self, symbol: str, params: Optional[TrendParams] = None):
        self.symbol = symbol
        self.params = params or TrendParams()

    # -------- Regime detection --------
    def evaluate_market(self, candles_15m: List[dict], yesterday_high: float, yesterday_low: float) -> TrendContext:
        closes = [float(c["close"]) for c in candles_15m if "close" in c]
        highs = [float(c["high"]) for c in candles_15m if "high" in c]
        lows = [float(c["low"]) for c in candles_15m if "low" in c]
        rsi = _rsi(closes, period=14)
        atr = _atr(highs, lows, closes, period=14)
        swing_high = max(highs[-self.params.lookback_swings:], default=0)
        swing_low = min(lows[-self.params.lookback_swings:], default=0)
        last_close = closes[-1] if closes else 0
        prev_close = closes[-5] if len(closes) >= 5 else (closes[0] if closes else 0)

        breakout_up = last_close > swing_high or last_close > yesterday_high or (last_close - prev_close) >= self.params.min_move_points
        breakdown_down = last_close < swing_low or last_close < yesterday_low or (prev_close - last_close) >= self.params.min_move_points

        if atr >= self.params.atr_trend_threshold:
            if rsi >= self.params.rsi_trend_up and breakout_up:
                side = TrendSide.UP
            elif rsi <= self.params.rsi_trend_down and breakdown_down:
                side = TrendSide.DOWN
            else:
                side = TrendSide.SIDEWAYS
        else:
            side = TrendSide.SIDEWAYS

        return TrendContext(trend_side=side, rsi=rsi, atr=atr, breakout_up=breakout_up, breakdown_down=breakdown_down)

    # -------- Trade decision --------
    def decide_trend_trade(
        self,
        trend_ctx: TrendContext,
        spot: float,
        option_chain: List[dict],
        expiry,
        lot_size: int,
        now: datetime,
        last_entry_time: time,
        current_positions: List[OptionLeg],
        basket_mtm: float,
        tp: float,
        sl: float,
    ) -> TradeAction:
        if now.time() > last_entry_time:
            return TradeAction("HOLD", StrategyType.TREND_SPREAD, [], [], "Past entry cutoff")
        if trend_ctx.trend_side == TrendSide.SIDEWAYS:
            return TradeAction("HOLD", StrategyType.TREND_SPREAD, [], [], "Sideways regime; trend engine idle")

        existing = self._find_existing_spread(current_positions, expiry)
        if existing:
            if basket_mtm >= tp:
                return TradeAction("CLOSE_ALL", StrategyType.TREND_SPREAD, [], existing, "Basket TP hit")
            if basket_mtm <= sl:
                return TradeAction("CLOSE_ALL", StrategyType.TREND_SPREAD, [], existing, "Basket SL hit")
            return TradeAction("HOLD", StrategyType.TREND_SPREAD, [], [], "Trend spread open within TP/SL")

        if trend_ctx.trend_side == TrendSide.UP:
            short = self._pick_leg(option_chain, expiry, "PUT", spot, short=True)
            hedge = self._pick_leg(option_chain, expiry, "PUT", spot, short=False)
        else:
            short = self._pick_leg(option_chain, expiry, "CALL", spot, short=True)
            hedge = self._pick_leg(option_chain, expiry, "CALL", spot, short=False)

        if not short or not hedge:
            return TradeAction("HOLD", StrategyType.TREND_SPREAD, [], [], "No suitable spread strikes")

        legs = [
            self._as_leg(short, LegSide.SELL, lot_size),
            self._as_leg(hedge, LegSide.BUY, lot_size),
        ]
        return TradeAction("OPEN_SPREAD", StrategyType.TREND_SPREAD, legs_to_open=legs, legs_to_close=[], reason=f"{trend_ctx.trend_side} regime: opening credit spread")

    def _find_existing_spread(self, positions: List[OptionLeg], expiry) -> List[OptionLeg]:
        shorts = [p for p in positions if p.expiry == expiry and p.side == LegSide.SELL]
        hedges = [p for p in positions if p.expiry == expiry and p.side == LegSide.BUY]
        if not shorts or not hedges:
            return []
        return shorts + hedges

    def _pick_leg(self, chain: List[dict], expiry, option_type: str, spot: float, short: bool) -> Optional[dict]:
        params = self.params
        delta_band = (params.short_delta_min, params.short_delta_max) if short else (params.hedge_delta_min, params.hedge_delta_max)
        filt = []
        for row in chain:
            if row.get("expiry") != expiry:
                continue
            if str(row.get("option_type", "")).upper() != ("CE" if option_type == "CALL" else "PE"):
                continue
            d = row.get("delta")
            if d is None:
                continue
            if option_type == "CALL":
                if delta_band[0] <= float(d) <= delta_band[1]:
                    filt.append(row)
            else:
                if (-delta_band[1]) <= float(d) <= (-delta_band[0]):
                    filt.append(row)
        if filt:
            target_delta = sum(delta_band) / 2.0
            return sorted(filt, key=lambda r: abs(abs(float(r.get("delta") or 0)) - target_delta))[0]
        # fallback by distance
        gap_short = params.fallback_gap_short if short else params.fallback_gap_hedge
        target = spot - gap_short if option_type == "PUT" else spot + gap_short
        return self._closest_strike(chain, expiry, option_type, target)

    def _closest_strike(self, chain: List[dict], expiry, option_type: str, target: float) -> Optional[dict]:
        candidates = []
        for row in chain:
            if row.get("expiry") != expiry:
                continue
            if str(row.get("option_type", "")).upper() != ("CE" if option_type == "CALL" else "PE"):
                continue
            strike = float(row.get("strike", 0))
            candidates.append((abs(strike - target), row))
        if not candidates:
            return None
        return sorted(candidates, key=lambda x: x[0])[0][1]

    def _as_leg(self, row: dict, side: LegSide, qty: int) -> OptionLeg:
        return OptionLeg(
            symbol=self.symbol,
            expiry=row.get("expiry"),
            strike=float(row.get("strike", 0)),
            option_type=OptionType.CALL if str(row.get("option_type", "")).upper() == "CE" else OptionType.PUT,
            side=side,
            quantity=qty,
            entry_price=float(row.get("ltp", 0) or 0.0),
            security_id=str(row.get("security_id") or row.get("securityId") or ""),
        )


# -------- helpers --------
def _rsi(closes: List[float], period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    gains = []
    losses = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0))
        losses.append(abs(min(diff, 0)))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def _atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> float:
    if not highs or not lows or not closes or len(highs) != len(lows) or len(highs) != len(closes):
        return 0.0
    trs = []
    for i in range(len(highs)):
        prev_close = closes[i - 1] if i > 0 else closes[0]
        tr = max(highs[i] - lows[i], abs(highs[i] - prev_close), abs(lows[i] - prev_close))
        trs.append(tr)
    if len(trs) < period:
        return sum(trs) / len(trs)
    return sum(trs[-period:]) / period
