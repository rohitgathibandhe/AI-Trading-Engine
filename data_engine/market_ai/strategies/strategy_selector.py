from __future__ import annotations

from datetime import date
from typing import List

from .models import (
    MarketSnapshot,
    OptionLeg,
    RiskConfig,
    StrategyType,
    TradeAction,
    TrendSide,
)
from .strangle_strategy import StrangleStrategyEngine
from .trend_strategy import TrendStrategyEngine


class StrategySelector:
    def __init__(self, symbol: str = "NIFTY", lot_size: int = 75):
        self.symbol = symbol
        self.lot_size = lot_size
        self.trend_engine = TrendStrategyEngine(symbol)
        self.strangle_engine = StrangleStrategyEngine(symbol)

    def decide(
        self,
        market: MarketSnapshot,
        option_chain: List[dict],
        expiry: date,
        risk: RiskConfig,
        current_positions: List[OptionLeg],
        basket_mtm: float,
    ) -> TradeAction:
        # Global guards
        if basket_mtm >= risk.intraday_target:
            return TradeAction("CLOSE_ALL", StrategyType.NONE, [], current_positions, "Global TP hit")
        if basket_mtm <= risk.max_intraday_loss:
            return TradeAction("CLOSE_ALL", StrategyType.NONE, [], current_positions, "Global SL hit")
        if market.now.time() >= risk.force_exit_time:
            return TradeAction("CLOSE_ALL", StrategyType.NONE, [], current_positions, "Force exit time reached")
        if market.india_vix > 18.0:
            return TradeAction("HOLD", StrategyType.NONE, [], [], "High VIX; standing aside.")

        # Regime
        trend_ctx = self.trend_engine.evaluate_market(market.candles_15m, market.yesterday_high, market.yesterday_low)

        # Route
        if trend_ctx.trend_side == TrendSide.SIDEWAYS:
            return self.strangle_engine.decide_strangle_trade(
                spot=market.spot,
                option_chain=option_chain,
                expiry=expiry,
                lot_size=self.lot_size,
                now=market.now,
                last_entry_time=risk.last_entry_time,
                current_positions=current_positions,
                basket_mtm=basket_mtm,
                tp=risk.intraday_target,
                sl=risk.max_intraday_loss,
            )
        if trend_ctx.trend_side in (TrendSide.UP, TrendSide.DOWN):
            return self.trend_engine.decide_trend_trade(
                trend_ctx,
                spot=market.spot,
                option_chain=option_chain,
                expiry=expiry,
                lot_size=self.lot_size,
                now=market.now,
                last_entry_time=risk.last_entry_time,
                current_positions=current_positions,
                basket_mtm=basket_mtm,
                tp=risk.intraday_target,
                sl=risk.max_intraday_loss,
            )

        return TradeAction("HOLD", StrategyType.NONE, [], [], "No trade regime")
