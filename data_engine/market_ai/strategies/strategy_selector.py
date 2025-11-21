from __future__ import annotations

import logging
from datetime import time
from typing import List, Optional, TYPE_CHECKING

from .models import (
    LegSide,
    MarketSnapshot,
    OptionLeg,
    OptionType,
    ORBState,
    RiskConfig,
    StrategyType,
    TradeAction,
    TrendContext,
    TrendSide,
)
from .strangle_engine import StrangleEngine, HEDGE_MAX_PRICE
from .credit_spread_engine import CreditSpreadEngine

if TYPE_CHECKING:
    from .sr_levels import SRLevels

LOG = logging.getLogger(__name__)

VIX_HIGH_THRESHOLD = 18.0
EARLIEST_ENTRY_TIME = time(9, 25)
SIDEWAYS_OPEN_RANGE_PCT = 0.004  # 0.4% of spot
MIN_TREND_MOVE_PCT = 0.0025  # 0.25% move from previous close
MIN_RANGE_BUFFER_PCT = 0.003  # keep spreads away from opposite boundary


def _basket_trade_action(action: str, strategy: StrategyType, legs: List[OptionLeg], reason: str) -> TradeAction:
    return TradeAction(action, strategy, [], legs, reason)


class StrategySelector:
    def __init__(self, symbol: str = "NIFTY", lot_size: int = 75):
        self.symbol = symbol
        self.lot_size = lot_size
        self.strangle_engine = StrangleEngine(symbol)
        self.credit_spread_engine = CreditSpreadEngine(symbol)
        self.entry_start_time = EARLIEST_ENTRY_TIME

    def decide(
        self,
        market: MarketSnapshot,
        option_chain: List[dict],
        expiry,
        risk: RiskConfig,
        current_positions: List[OptionLeg],
        basket_mtm: float,
        trend_ctx: TrendContext,
        daily_trades: int = 0,
        day_mode: str = "NORMAL",
        entries_used: int = 0,
        legs_used: int = 0,
        orb_state: Optional["ORBState"] = None,
        sr_levels: Optional["SRLevels"] = None,
        reentry_buffer: float = 10.0,
    ) -> TradeAction:
        if basket_mtm >= risk.intraday_target:
            return _basket_trade_action("CLOSE_ALL", StrategyType.NONE, current_positions, "Global TP hit")
        if basket_mtm <= risk.max_intraday_loss:
            return _basket_trade_action("CLOSE_ALL", StrategyType.NONE, current_positions, "Global SL hit")
        if market.now.time() >= risk.force_exit_time:
            return _basket_trade_action("CLOSE_ALL", StrategyType.NONE, current_positions, "Force exit time")
        if market.india_vix > max(risk.vix_carry_threshold, VIX_HIGH_THRESHOLD):
            return TradeAction("HOLD", StrategyType.NONE, [], [], "High VIX; standing aside")
        if daily_trades >= risk.max_daily_trades:
            return TradeAction("HOLD", StrategyType.NONE, [], [], "Max daily trades reached")

        if day_mode == "LOCKED_RED":
            if current_positions:
                return _basket_trade_action("CLOSE_ALL", StrategyType.NONE, current_positions, "Daily max loss reached")
            return TradeAction("HOLD", StrategyType.NONE, [], [], "Daily max loss lock")
        if day_mode == "LOCKED_GREEN" and not current_positions:
            return TradeAction("HOLD", StrategyType.NONE, [], [], "Daily target achieved")

        allow_new_entries = (
            day_mode == "NORMAL"
            and entries_used < risk.max_entries_per_day
            and legs_used < risk.max_legs_per_day
        )

        if orb_state:
            orb_exit = self._maybe_exit_on_orb(orb_state, current_positions, market)
            if orb_exit:
                return orb_exit
            reentry_action = self._maybe_reenter_orb(
                orb_state,
                sr_levels,
                market,
                allow_new_entries,
                reentry_buffer,
                risk,
                option_chain,
                expiry,
                current_positions,
            )
            if reentry_action:
                return reentry_action

        if trend_ctx.trend_side == TrendSide.SIDEWAYS or trend_ctx.confidence < 2:
            return self._handle_strangle(
                market, option_chain, expiry, risk, current_positions, basket_mtm, allow_new_entries, orb_state, trend_ctx
            )
        if trend_ctx.trend_side == TrendSide.BULL:
            return self._handle_bullish(
                market, option_chain, expiry, risk, current_positions, basket_mtm, allow_new_entries
            )
        if trend_ctx.trend_side == TrendSide.BEAR:
            return self._handle_bearish(
                market, option_chain, expiry, risk, current_positions, basket_mtm, allow_new_entries, trend_ctx
            )
        return TradeAction("HOLD", StrategyType.NONE, [], [], "No trade regime")

    def _handle_strangle(self, market, option_chain, expiry, risk, current_positions, basket_mtm, allow_open, orb_state, trend_ctx):
        if market.now.time() < self.entry_start_time:
            return TradeAction("HOLD", StrategyType.STRANGLE, [], [], "Waiting for post-open window")
        if not self._spot_inside_prev_range(market):
            return TradeAction("HOLD", StrategyType.STRANGLE, [], [], "Breakout outside yesterday range")
        if not self._calm_open(market):
            return TradeAction("HOLD", StrategyType.STRANGLE, [], [], "Volatile open; skipping neutral strangle")
        if orb_state and orb_state.losing_side_exited:
            return TradeAction("HOLD", StrategyType.STRANGLE, [], [], "Awaiting ORB re-entry setup")
        if not allow_open:
            return TradeAction("HOLD", StrategyType.STRANGLE, [], [], "Daily limits reached")
        partial_tp = max(0.0, risk.intraday_target * getattr(risk, "partial_target_pct", 0.0))
        per_leg_sl = max(0.0, getattr(risk, "per_leg_sl_mult", 0.0))
        per_leg_tp = max(0.0, getattr(risk, "per_leg_tp_mult", 0.0))
        positions = current_positions
        if current_positions:
            tagged = [leg for leg in current_positions if getattr(leg, "strategy_type", None) == StrategyType.STRANGLE]
            if tagged:
                positions = tagged
        if positions:
            return self.strangle_engine.manage_strangle_positions(
                positions,
                basket_mtm,
                risk.intraday_target,
                risk.max_intraday_loss,
                partial_tp,
                per_leg_sl,
                per_leg_tp,
                market.now,
                risk.last_entry_time,
                max_hold_days=2,
            )
        if market.now.time() > risk.last_entry_time:
            return TradeAction("HOLD", StrategyType.STRANGLE, [], [], "Past entry cutoff")
        # Strangle gating (re-enabled with guardrails)
        gap_pct = 0.0
        if market.yesterday_close:
            gap_pct = abs((market.spot - market.yesterday_close) / market.yesterday_close * 100.0)
        if gap_pct >= 0.4:
            return TradeAction("HOLD", StrategyType.STRANGLE, [], [], "Gap too wide for neutral strangle")
        if orb_state and orb_state.breakout_signal:
            return TradeAction("HOLD", StrategyType.STRANGLE, [], [], "ORB breakout detected; skipping strangle")
        if max(trend_ctx.bull_score, trend_ctx.bear_score) >= 2.0:
            return TradeAction("HOLD", StrategyType.STRANGLE, [], [], "Trend conviction too high for strangle")
        vix_pct = getattr(market, "vix_percentile", None)
        if vix_pct is None and market.india_vix:
            # crude proxy using adaptive band
            span = max(1e-6, risk.vix_adaptive_high - risk.vix_adaptive_low)
            vix_pct = max(0.0, min(1.0, (market.india_vix - risk.vix_adaptive_low) / span))
        if vix_pct is None:
            vix_pct = 0.5  # default allow if unknown
        if vix_pct < 0.3:
            return TradeAction("HOLD", StrategyType.STRANGLE, [], [], "IV percentile too low for strangle")
        dte = (expiry - market.now.date()).days if hasattr(market.now, "date") else 0
        if dte < 2 or dte > 4:
            return TradeAction("HOLD", StrategyType.STRANGLE, [], [], "DTE not in 2-4 window for strangle")
        target_delta = self._adaptive_value(
            market.india_vix,
            risk.vix_adaptive_low,
            risk.vix_adaptive_high,
            risk.strangle_delta_low,
            risk.strangle_delta_high,
        )
        fallback_points = self._adaptive_value(
            market.india_vix,
            risk.vix_adaptive_low,
            risk.vix_adaptive_high,
            risk.strangle_offset_low,
            risk.strangle_offset_high,
        )
        legs = self.strangle_engine.build_neutral_strangle(
            market.spot,
            option_chain,
            expiry,
            self.lot_size,
            target_delta=target_delta,
            fallback_points=fallback_points,
        )
        if not legs:
            return TradeAction("HOLD", StrategyType.STRANGLE, [], [], "No suitable strikes")
        return TradeAction("OPEN", StrategyType.STRANGLE, legs_to_open=legs, legs_to_close=[], reason="SIDEWAYS regime strangle")

    def _handle_bullish(self, market, option_chain, expiry, risk, current_positions, basket_mtm, allow_open):
        if market.now.time() < self.entry_start_time:
            return TradeAction("HOLD", StrategyType.BULL_PUT_SPREAD, [], [], "Waiting for post-open window")
        if not self._trend_move_confirmed(market, bullish=True):
            return TradeAction("HOLD", StrategyType.BULL_PUT_SPREAD, [], [], "Bullish move not confirmed yet")
        if not self._pullback_confirmed(market, bullish=True):
            return TradeAction("HOLD", StrategyType.BULL_PUT_SPREAD, [], [], "Awaiting first pullback confirmation")
        if not allow_open and not current_positions:
            return TradeAction("HOLD", StrategyType.BULL_PUT_SPREAD, [], [], "Daily limits reached")
        partial_tp = max(0.0, risk.intraday_target * getattr(risk, "partial_target_pct", 0.0))
        per_leg_sl = max(0.0, getattr(risk, "per_leg_sl_mult", 0.0))
        per_leg_tp = max(0.0, getattr(risk, "per_leg_tp_mult", 0.0))
        positions = current_positions
        if current_positions:
            tagged = [leg for leg in current_positions if getattr(leg, "strategy_type", None) == StrategyType.BULL_PUT_SPREAD]
            if tagged:
                positions = tagged
        if positions:
            return self.credit_spread_engine.manage_spread_positions(
                positions,
                basket_mtm,
                risk.intraday_target,
                risk.max_intraday_loss,
                partial_tp,
                StrategyType.BULL_PUT_SPREAD,
                per_leg_sl,
                per_leg_tp,
            )
        if market.now.time() > risk.last_entry_time:
            return TradeAction("HOLD", StrategyType.BULL_PUT_SPREAD, [], [], "Past entry cutoff")
        short_delta = self._adaptive_value(
            market.india_vix,
            risk.vix_adaptive_low,
            risk.vix_adaptive_high,
            risk.spread_short_delta_low,
            risk.spread_short_delta_high,
        )
        legs = self.credit_spread_engine.build_bull_put_spread(
            market.spot, option_chain, expiry, self.lot_size, short_delta=short_delta
        )
        if not legs:
            return TradeAction("HOLD", StrategyType.BULL_PUT_SPREAD, [], [], "No suitable PUT strikes")
        return TradeAction("OPEN", StrategyType.BULL_PUT_SPREAD, legs_to_open=legs, legs_to_close=[], reason="Bullish regime spread")

    def _handle_bearish(self, market, option_chain, expiry, risk, current_positions, basket_mtm, allow_open, trend_ctx):
        if market.now.time() < self.entry_start_time:
            return TradeAction("HOLD", StrategyType.BEAR_CALL_SPREAD, [], [], "Waiting for post-open window")
        if trend_ctx.confidence < 0.7:
            return TradeAction("HOLD", StrategyType.BEAR_CALL_SPREAD, [], [], "Bear conviction too low")
        if not self._trend_move_confirmed(market, bullish=False):
            return TradeAction("HOLD", StrategyType.BEAR_CALL_SPREAD, [], [], "Bearish move not confirmed yet")
        if not self._pullback_confirmed(market, bullish=False):
            return TradeAction("HOLD", StrategyType.BEAR_CALL_SPREAD, [], [], "Awaiting first pullback confirmation")
        if not allow_open and not current_positions:
            return TradeAction("HOLD", StrategyType.BEAR_CALL_SPREAD, [], [], "Daily limits reached")
        partial_tp = max(0.0, risk.intraday_target * getattr(risk, "partial_target_pct", 0.0))
        per_leg_sl = max(0.0, getattr(risk, "per_leg_sl_mult", 0.0))
        per_leg_tp = max(0.0, getattr(risk, "per_leg_tp_mult", 0.0))
        positions = current_positions
        if current_positions:
            tagged = [leg for leg in current_positions if getattr(leg, "strategy_type", None) == StrategyType.BEAR_CALL_SPREAD]
            if tagged:
                positions = tagged
        if positions:
            return self.credit_spread_engine.manage_spread_positions(
                positions,
                basket_mtm,
                risk.intraday_target,
                risk.max_intraday_loss,
                partial_tp,
                StrategyType.BEAR_CALL_SPREAD,
                per_leg_sl,
                per_leg_tp,
            )
        if market.now.time() > risk.last_entry_time:
            return TradeAction("HOLD", StrategyType.BEAR_CALL_SPREAD, [], [], "Past entry cutoff")
        short_delta = self._adaptive_value(
            market.india_vix,
            risk.vix_adaptive_low,
            risk.vix_adaptive_high,
            risk.spread_short_delta_low,
            risk.spread_short_delta_high,
        )
        legs = self.credit_spread_engine.build_bear_call_spread(
            market.spot, option_chain, expiry, self.lot_size, short_delta=short_delta
        )
        if not legs:
            return TradeAction("HOLD", StrategyType.BEAR_CALL_SPREAD, [], [], "No suitable CALL strikes")
        return TradeAction("OPEN", StrategyType.BEAR_CALL_SPREAD, legs_to_open=legs, legs_to_close=[], reason="Bearish regime spread")

    def _maybe_exit_on_orb(self, orb_state, current_positions, market):
        if not orb_state or not orb_state.breakout_signal or orb_state.losing_side_exited:
            return None
        signal = orb_state.breakout_signal
        strangle_legs = [leg for leg in current_positions if getattr(leg, "strategy_type", None) == StrategyType.STRANGLE]
        if strangle_legs:
            losing_type = OptionType.CALL if signal.direction == TrendSide.BULL else OptionType.PUT
            legs_to_close = [
                leg
                for leg in strangle_legs
                if leg.side == LegSide.SELL and leg.option_type == losing_type
            ]
            if legs_to_close:
                orb_state.losing_side_exited = True
                orb_state.last_reentry_side = losing_type
                orb_state.last_reentry_time = market.now
                return TradeAction(
                    "CLOSE_LEGS",
                    StrategyType.STRANGLE,
                    [],
                    legs_to_close,
                    "ORB breakout: exiting losing side only, keeping hedges.",
                )

        if signal.direction == TrendSide.BULL:
            spread_type = StrategyType.BEAR_CALL_SPREAD
            reentry_side = OptionType.CALL
            reason = "ORB bullish breakout against bear call spread"
        else:
            spread_type = StrategyType.BULL_PUT_SPREAD
            reentry_side = OptionType.PUT
            reason = "ORB bearish breakout against bull put spread"
        spread_legs = [
            leg for leg in current_positions if getattr(leg, "strategy_type", None) == spread_type
        ]
        if spread_legs:
            orb_state.losing_side_exited = True
            orb_state.last_reentry_side = reentry_side
            orb_state.last_reentry_time = market.now
            return TradeAction("CLOSE_LEGS", spread_type, [], spread_legs, reason)
        return None

    def _maybe_reenter_orb(
        self,
        orb_state,
        sr_levels,
        market,
        allow_open,
        buffer_points,
        risk,
        option_chain,
        expiry,
        current_positions,
    ):
        if not orb_state or not orb_state.losing_side_exited or not allow_open:
            return None
        side = orb_state.last_reentry_side
        if not side or not sr_levels:
            return None
        targets = sr_levels.major_resistances if side == OptionType.CALL else sr_levels.major_supports
        if not targets:
            return None
        spot = market.spot
        ready = False
        for level in targets:
            if side == OptionType.CALL and spot >= level - buffer_points:
                ready = True
                break
            if side == OptionType.PUT and spot <= level + buffer_points:
                ready = True
                break
        if not ready:
            return None

        if side in (OptionType.CALL, OptionType.PUT):
            existing_hedge = next(
                (
                    leg
                    for leg in current_positions
                    if leg.side == LegSide.BUY
                    and leg.option_type == side
                    and getattr(leg, "strategy_type", None) == StrategyType.STRANGLE
                ),
                None,
            )
            hedge_cost = None
            if existing_hedge:
                hedge_cost = existing_hedge.current_ltp or existing_hedge.entry_price
            hedge_valid = hedge_cost is not None and hedge_cost <= HEDGE_MAX_PRICE
            target_delta = self._adaptive_value(
                market.india_vix,
                risk.vix_adaptive_low,
                risk.vix_adaptive_high,
                risk.strangle_delta_low,
                risk.strangle_delta_high,
            )
            fallback = self._adaptive_value(
                market.india_vix,
                risk.vix_adaptive_low,
                risk.vix_adaptive_high,
                risk.strangle_offset_low,
                risk.strangle_offset_high,
            )
            legs = self.strangle_engine.build_single_side(
                option_chain,
                expiry,
                self.lot_size,
                side,
                target_delta,
                fallback,
                require_hedge=not hedge_valid,
                spot=market.spot,
            )
            if not legs:
                return None
            orb_state.losing_side_exited = False
            orb_state.breakout_signal = None
            orb_state.last_reentry_time = market.now
            reason = (
                "ORB re-entry at resistance after bull breakout"
                if side == OptionType.CALL
                else "ORB re-entry at support after bear breakout"
            )
            return TradeAction("OPEN", StrategyType.STRANGLE, legs_to_open=legs, legs_to_close=[], reason=reason)

        short_delta = self._adaptive_value(
            market.india_vix,
            risk.vix_adaptive_low,
            risk.vix_adaptive_high,
            risk.spread_short_delta_low,
            risk.spread_short_delta_high,
        )
        if side == OptionType.CALL:
            legs = self.credit_spread_engine.build_bear_call_spread(
                market.spot, option_chain, expiry, self.lot_size, short_delta=short_delta
            )
            strategy = StrategyType.BEAR_CALL_SPREAD
            reason = "ORB re-entry at resistance after bull breakout"
        else:
            legs = self.credit_spread_engine.build_bull_put_spread(
                market.spot, option_chain, expiry, self.lot_size, short_delta=short_delta
            )
            strategy = StrategyType.BULL_PUT_SPREAD
            reason = "ORB re-entry at support after bear breakout"

        if not legs:
            LOG.warning("Skipping re-entry: no acceptable hedge with LTP <= %.2f found", HEDGE_MAX_PRICE)
            return None

        orb_state.losing_side_exited = False
        orb_state.breakout_signal = None
        orb_state.last_reentry_time = market.now
        return TradeAction("OPEN", strategy, legs_to_open=legs, legs_to_close=[], reason=reason)

    def _spot_inside_prev_range(self, market: MarketSnapshot) -> bool:
        if not market.yesterday_high or not market.yesterday_low:
            return True
        return market.yesterday_low <= market.spot <= market.yesterday_high

    def _calm_open(self, market: MarketSnapshot) -> bool:
        candles = market.candles_5m or []
        if not candles:
            return True
        spot = market.spot or max(market.yesterday_close, 1.0)
        first_high = self._candle_val(candles[0], "high")
        first_low = self._candle_val(candles[0], "low")
        if first_high is None or first_low is None:
            return True
        first_range = (first_high - first_low) / max(spot, 1.0)
        if first_range > SIDEWAYS_OPEN_RANGE_PCT:
            return False
        if len(candles) > 1:
            second_high = self._candle_val(candles[1], "high")
            second_low = self._candle_val(candles[1], "low")
            if second_high is None or second_low is None:
                return True
            second_range = (second_high - second_low) / max(spot, 1.0)
            if second_range > SIDEWAYS_OPEN_RANGE_PCT:
                return False
        return True

    def _trend_move_confirmed(self, market: MarketSnapshot, *, bullish: bool) -> bool:
        ref = market.yesterday_close or market.spot
        if not ref:
            return True
        move_pct = (market.spot - ref) / ref
        if bullish:
            if move_pct < MIN_TREND_MOVE_PCT and (not market.yesterday_high or market.spot < market.yesterday_high):
                return False
            if market.yesterday_low and (market.spot - market.yesterday_low) < market.spot * MIN_RANGE_BUFFER_PCT:
                return False
        else:
            if -move_pct < MIN_TREND_MOVE_PCT and (not market.yesterday_low or market.spot > market.yesterday_low):
                return False
            if market.yesterday_high and (market.yesterday_high - market.spot) < market.spot * MIN_RANGE_BUFFER_PCT:
                return False
        return True

    def _pullback_confirmed(self, market: MarketSnapshot, *, bullish: bool) -> bool:
        candles = market.candles_5m or []
        if len(candles) < 2:
            return False
        first = candles[0]
        second = candles[1]
        c1_close = self._candle_val(first, "close")
        if c1_close is None:
            return False
        if bullish:
            low2 = self._candle_val(second, "low")
            close2 = self._candle_val(second, "close")
            if low2 is None or close2 is None:
                return False
            return low2 <= c1_close and close2 >= c1_close
        high2 = self._candle_val(second, "high")
        close2 = self._candle_val(second, "close")
        if high2 is None or close2 is None:
            return False
        return high2 >= c1_close and close2 <= c1_close

    @staticmethod
    def _candle_val(candle: dict, key: str) -> Optional[float]:
        if not isinstance(candle, dict):
            return None
        val = candle.get(key)
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _adaptive_value(vix: float, low_thr: float, high_thr: float, low_val: float, high_val: float) -> float:
        try:
            v = float(vix)
        except Exception:
            v = 0.0
        lo = float(low_thr)
        hi = float(high_thr)
        if hi <= lo:
            return (low_val + high_val) / 2.0
        if v <= lo:
            return low_val
        if v >= hi:
            return high_val
        ratio = (v - lo) / (hi - lo)
        return low_val + ratio * (high_val - low_val)
