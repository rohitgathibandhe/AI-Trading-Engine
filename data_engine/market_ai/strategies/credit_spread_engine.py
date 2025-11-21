from __future__ import annotations

import logging
from typing import List, Optional

from .models import LegSide, OptionLeg, OptionType, StrategyType, TradeAction

LOG = logging.getLogger(__name__)

SPREAD_SHORT_DELTA = 0.25
SPREAD_LONG_DELTA = 0.08
SPREAD_GAP_POINTS = 200
HEDGE_PREFERRED_MAX = 12.0
HEDGE_FALLBACK_MAX = 20.0
# Backward compat for any imports
HEDGE_MAX_PRICE = HEDGE_PREFERRED_MAX


def _select_hedge_leg(
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
    hedge = _find(HEDGE_FALLBACK_MAX, min_distance + 50)
    if not hedge:
        LOG.warning(
            "Skipping spread: no acceptable hedge with LTP <= %.2f (or fallback <= %.2f) found",
            HEDGE_PREFERRED_MAX,
            HEDGE_FALLBACK_MAX,
        )
    return hedge


def _select_leg(option_chain: List[dict], expiry, option_type: str, target_delta: float, fallback_offset: float, spot: float):
    best = None
    best_diff = float("inf")
    for row in option_chain:
        if row.get("expiry") != expiry:
            continue
        if str(row.get("option_type", "")).upper() != option_type:
            continue
        delta = row.get("delta")
        if delta is None:
            continue
        diff = abs(abs(float(delta)) - target_delta)
        if diff < best_diff:
            best_diff = diff
            best = row
    if best:
        return best
    # fallback by strike distance
    target = spot + fallback_offset if option_type == "CE" else spot - fallback_offset
    for row in option_chain:
        if row.get("expiry") != expiry:
            continue
        if str(row.get("option_type", "")).upper() != option_type:
            continue
        diff = abs(float(row.get("strike", 0)) - target)
        if diff < best_diff:
            best_diff = diff
            best = row
    return best


class CreditSpreadEngine:
    def __init__(self, symbol: str = "NIFTY"):
        self.symbol = symbol

    def build_bull_put_spread(
        self,
        spot: float,
        option_chain: List[dict],
        expiry,
        lot_size: int,
        short_delta: Optional[float] = None,
    ) -> List[OptionLeg]:
        delta = short_delta if short_delta is not None else SPREAD_SHORT_DELTA
        short_put = _select_leg(option_chain, expiry, "PE", delta, 150, spot)
        long_put = _select_hedge_leg(option_chain, expiry, "PE", float(short_put["strike"]) if short_put else 0.0, SPREAD_GAP_POINTS) if short_put else None
        if not short_put or not long_put:
            LOG.warning(
                "Skipping bull put spread due to missing hedge with LTP <= %.2f (fallback %.2f)",
                HEDGE_PREFERRED_MAX,
                HEDGE_FALLBACK_MAX,
            )
            return []
        return self._spread_from_rows(short_put, long_put, lot_size, expiry, is_put=True)

    def build_bear_call_spread(
        self,
        spot: float,
        option_chain: List[dict],
        expiry,
        lot_size: int,
        short_delta: Optional[float] = None,
    ) -> List[OptionLeg]:
        delta = short_delta if short_delta is not None else SPREAD_SHORT_DELTA
        short_call = _select_leg(option_chain, expiry, "CE", delta, 150, spot)
        long_call = _select_hedge_leg(option_chain, expiry, "CE", float(short_call["strike"]) if short_call else 0.0, SPREAD_GAP_POINTS) if short_call else None
        if not short_call or not long_call:
            LOG.warning(
                "Skipping bear call spread due to missing hedge with LTP <= %.2f (fallback %.2f)",
                HEDGE_PREFERRED_MAX,
                HEDGE_FALLBACK_MAX,
            )
            return []
        return self._spread_from_rows(short_call, long_call, lot_size, expiry, is_put=False)

    def _spread_from_rows(self, short_row: dict, long_row: dict, lot_size: int, expiry, is_put: bool) -> List[OptionLeg]:
        opt_type = OptionType.PUT if is_put else OptionType.CALL
        strategy = StrategyType.BULL_PUT_SPREAD if is_put else StrategyType.BEAR_CALL_SPREAD
        legs = [
            OptionLeg(
                symbol=self.symbol,
                expiry=expiry,
                strike=float(short_row.get("strike")),
                option_type=opt_type,
                side=LegSide.SELL,
                quantity=lot_size,
                entry_price=float(short_row.get("ltp", 0.0) or 0.0),
                security_id=str(short_row.get("security_id")),
                strategy_type=strategy,
            ),
            OptionLeg(
                symbol=self.symbol,
                expiry=expiry,
                strike=float(long_row.get("strike")),
                option_type=opt_type,
                side=LegSide.BUY,
                quantity=lot_size,
                entry_price=float(long_row.get("ltp", 0.0) or 0.0),
                security_id=str(long_row.get("security_id")),
                strategy_type=strategy,
            ),
        ]
        return legs

    def manage_spread_positions(
        self,
        current_positions: List[OptionLeg],
        basket_mtm: float,
        tp: float,
        sl: float,
        partial_tp: float,
        strategy: StrategyType,
        per_leg_sl: float,
        per_leg_tp: float,
    ) -> TradeAction:
        per_leg_reason = self._per_leg_guard(current_positions, per_leg_sl, per_leg_tp)
        if per_leg_reason:
            return TradeAction("CLOSE_ALL", strategy, [], current_positions, per_leg_reason)
        if partial_tp > 0 and basket_mtm >= partial_tp:
            return TradeAction("CLOSE_ALL", strategy, [], current_positions, "Partial target booked")
        if basket_mtm >= tp:
            return TradeAction("CLOSE_ALL", strategy, [], current_positions, "Spread TP hit")
        if basket_mtm <= sl:
            return TradeAction("CLOSE_ALL", strategy, [], current_positions, "Spread SL hit")
        return TradeAction("HOLD", strategy, [], [], "Spread open within TP/SL")

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
