from __future__ import annotations

import logging
from typing import List, Optional
from datetime import datetime, timedelta

from .models import LegSide, OptionLeg, OptionType, StrategyType, TradeAction

LOG = logging.getLogger(__name__)

DEFAULT_DELTA_MIN = 0.12
DEFAULT_DELTA_MAX = 0.18
DEFAULT_BIASED_DELTA = 0.22
DEFAULT_HEDGE_GAP = 250
HEDGE_PREFERRED_MAX = 12.0
HEDGE_FALLBACK_MAX = 20.0
# Backward compat for imports in StrategySelector
HEDGE_MAX_PRICE = HEDGE_PREFERRED_MAX


def _pick_leg(
    option_chain: List[dict],
    expiry,
    option_type: str,
    target_delta: float,
    fallback_points: float,
    spot: float,
) -> Optional[dict]:
    candidates = []
    for row in option_chain:
        if row.get("expiry") != expiry:
            continue
        if str(row.get("option_type", "")).upper() != option_type:
            continue
        delta = row.get("delta")
        if delta is None:
            continue
        if option_type == "CE" and delta >= 0:
            candidates.append((abs(float(delta) - target_delta), row))
        elif option_type == "PE" and delta <= 0:
            candidates.append((abs(abs(float(delta)) - target_delta), row))
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    # fallback by distance
    target_strike = spot + fallback_points if option_type == "CE" else spot - fallback_points
    best: Optional[dict] = None
    best_diff = float("inf")
    for row in option_chain:
        if row.get("expiry") != expiry:
            continue
        if str(row.get("option_type", "")).upper() != option_type:
            continue
        diff = abs(float(row.get("strike", 0)) - target_strike)
        if diff < best_diff:
            best_diff = diff
            best = row
    return best


def _pick_hedge(
    option_chain: List[dict],
    expiry,
    option_type: str,
    short_strike: float,
    min_distance: float,
) -> Optional[dict]:
    def _find(max_price: float, distance: float) -> Optional[dict]:
        candidates: List[dict] = []
        for row in option_chain:
            if row.get("expiry") != expiry:
                continue
            if str(row.get("option_type") or "").upper() != option_type:
                continue
            strike = float(row.get("strike") or 0.0)
            ltp = float(row.get("ltp") or 0.0)
            if ltp <= 0 or ltp > max_price:
                continue
            if option_type == "CE":
                if strike < short_strike + distance:
                    continue
            else:
                if strike > short_strike - distance:
                    continue
            candidates.append(row)
        if not candidates:
            return None
        if option_type == "CE":
            return max(candidates, key=lambda r: float(r.get("strike") or 0.0))
        return min(candidates, key=lambda r: float(r.get("strike") or 0.0))

    hedge = _find(HEDGE_PREFERRED_MAX, min_distance)
    if hedge:
        return hedge
    # Fallback: allow higher hedge premium and widen distance to hunt deeper OTM strikes.
    hedge = _find(HEDGE_FALLBACK_MAX, min_distance + 50)
    if not hedge:
        LOG.warning(
            "Skipping hedge: no acceptable hedge with LTP <= %.2f (or fallback <= %.2f) found",
            HEDGE_PREFERRED_MAX,
            HEDGE_FALLBACK_MAX,
        )
    return hedge


class StrangleEngine:
    def __init__(self, symbol: str = "NIFTY", hedge_gap: int = DEFAULT_HEDGE_GAP):
        self.symbol = symbol
        self.hedge_gap = hedge_gap

    def build_neutral_strangle(
        self,
        spot: float,
        option_chain: List[dict],
        expiry,
        lot_size: int,
        target_delta: Optional[float] = None,
        fallback_points: Optional[float] = None,
    ) -> List[OptionLeg]:
        delta = target_delta if target_delta is not None else (DEFAULT_DELTA_MIN + DEFAULT_DELTA_MAX) / 2.0
        fallback = fallback_points if fallback_points is not None else 150.0
        ce = _pick_leg(option_chain, expiry, "CE", delta, fallback, spot)
        pe = _pick_leg(option_chain, expiry, "PE", delta, fallback, spot)
        if not ce or not pe:
            return []
        return self._build_strangle_from_legs(option_chain, ce, pe, expiry, lot_size, self.hedge_gap)

    def build_biased_strangle(
        self,
        spot: float,
        option_chain: List[dict],
        expiry,
        lot_size: int,
        bullish: bool,
        bias_delta: Optional[float] = None,
        fallback_points: Optional[float] = None,
    ) -> List[OptionLeg]:
        fallback = fallback_points if fallback_points is not None else 150.0
        if bullish:
            target = bias_delta if bias_delta is not None else DEFAULT_BIASED_DELTA
            pe = _pick_leg(option_chain, expiry, "PE", target, fallback, spot)
            ce = _pick_leg(option_chain, expiry, "CE", DEFAULT_DELTA_MIN, fallback + 50, spot)
        else:
            target = bias_delta if bias_delta is not None else DEFAULT_BIASED_DELTA
            ce = _pick_leg(option_chain, expiry, "CE", target, fallback, spot)
            pe = _pick_leg(option_chain, expiry, "PE", DEFAULT_DELTA_MIN, fallback + 50, spot)
        if not ce or not pe:
            return []
        return self._build_strangle_from_legs(option_chain, ce, pe, expiry, lot_size, self.hedge_gap)

    def _build_strangle_from_legs(
        self,
        option_chain: List[dict],
        ce: dict,
        pe: dict,
        expiry,
        lot_size: int,
        min_distance: float,
    ) -> List[OptionLeg]:
        hedge_ce = _pick_hedge(option_chain, expiry, "CE", float(ce["strike"]), min_distance)
        hedge_pe = _pick_hedge(option_chain, expiry, "PE", float(pe["strike"]), min_distance)
        if not hedge_ce or not hedge_pe:
            LOG.warning("Skipping strangle build due to missing hedge with LTP <= %.2f", HEDGE_MAX_PRICE)
            return []

        def _leg(row: dict, side: LegSide) -> OptionLeg:
            return OptionLeg(
                symbol=self.symbol,
                expiry=expiry,
                strike=float(row.get("strike")),
                option_type=OptionType.CALL if str(row.get("option_type")).upper() == "CE" else OptionType.PUT,
                side=side,
                quantity=lot_size,
                entry_price=float(row.get("ltp", 0.0) or 0.0),
                security_id=str(row.get("security_id")),
            )

        legs = [
            _leg(ce, LegSide.SELL),
            _leg(pe, LegSide.SELL),
            _leg(hedge_ce, LegSide.BUY),
            _leg(hedge_pe, LegSide.BUY),
        ]
        for leg in legs[:2]:
            leg.strategy_type = StrategyType.STRANGLE
        for leg in legs[2:]:
            leg.strategy_type = StrategyType.STRANGLE
        return legs

    def build_single_side(
        self,
        option_chain: List[dict],
        expiry,
        lot_size: int,
        option_type: OptionType,
        target_delta: float,
        fallback_points: float,
        require_hedge: bool,
        spot: float,
    ) -> List[OptionLeg]:
        side = "CE" if option_type == OptionType.CALL else "PE"
        row = _pick_leg(option_chain, expiry, side, target_delta, fallback_points, spot)
        if not row:
            return []
        short_leg = OptionLeg(
            symbol=self.symbol,
            expiry=expiry,
            strike=float(row.get("strike")),
            option_type=option_type,
            side=LegSide.SELL,
            quantity=lot_size,
            entry_price=float(row.get("ltp", 0.0) or 0.0),
            security_id=str(row.get("security_id")),
            strategy_type=StrategyType.STRANGLE,
        )
        legs = [short_leg]
        if require_hedge:
            hedge_row = _pick_hedge(option_chain, expiry, side, float(row.get("strike")), fallback_points)
            if not hedge_row:
                LOG.warning("Skipping re-entry: no acceptable hedge with LTP <= %.2f found", HEDGE_MAX_PRICE)
                return []
            hedge_leg = OptionLeg(
                symbol=self.symbol,
                expiry=expiry,
                strike=float(hedge_row.get("strike")),
                option_type=option_type,
                side=LegSide.BUY,
                quantity=lot_size,
                entry_price=float(hedge_row.get("ltp", 0.0) or 0.0),
                security_id=str(hedge_row.get("security_id")),
                strategy_type=StrategyType.STRANGLE,
            )
            legs.append(hedge_leg)
        return legs

    def manage_strangle_positions(
        self,
        current_positions: List[OptionLeg],
        basket_mtm: float,
        tp: float,
        sl: float,
        partial_tp: float,
        per_leg_sl: float,
        per_leg_tp: float,
        now,
        last_entry_time,
        max_hold_days: int = 2,
    ) -> TradeAction:
        per_leg_reason = self._per_leg_guard(current_positions, per_leg_sl, per_leg_tp)
        if per_leg_reason:
            return TradeAction("CLOSE_ALL", StrategyType.STRANGLE, [], current_positions, per_leg_reason)
        if partial_tp > 0 and basket_mtm >= partial_tp:
            return TradeAction("CLOSE_ALL", StrategyType.STRANGLE, [], current_positions, "Partial target booked")
        if basket_mtm >= tp:
            return TradeAction("CLOSE_ALL", StrategyType.STRANGLE, [], current_positions, "Basket TP hit")
        if basket_mtm <= sl:
            return TradeAction("CLOSE_ALL", StrategyType.STRANGLE, [], current_positions, "Basket SL hit")
        if max_hold_days and max_hold_days > 0:
            oldest_open = min((leg.opened_at for leg in current_positions if getattr(leg, "opened_at", None)), default=None)
            if oldest_open and isinstance(now, datetime) and now - oldest_open >= timedelta(days=max_hold_days):
                return TradeAction("CLOSE_ALL", StrategyType.STRANGLE, [], current_positions, "TIME_EXIT")
        if now.time() > last_entry_time:
            return TradeAction("HOLD", StrategyType.STRANGLE, [], [], "Past entry cutoff")
        return TradeAction("HOLD", StrategyType.STRANGLE, [], [], "Strangle open within TP/SL")

    def _per_leg_guard(self, positions: List[OptionLeg], sl_mult: float, tp_mult: float) -> Optional[str]:
        if not positions:
            return None
        for leg in positions:
            if getattr(leg, "side", None) != LegSide.SELL:
                continue
            entry = float(getattr(leg, "entry_price", 0.0) or 0.0)
            ltp = leg.current_ltp
            if entry <= 0 or ltp is None:
                continue
            if sl_mult > 0 and ltp >= entry * sl_mult:
                return f"Per-leg SL hit {leg.option_type.value} {leg.strike}"
            if tp_mult > 0 and ltp <= entry * tp_mult:
                return f"Per-leg TP hit {leg.option_type.value} {leg.strike}"
        return None
