from datetime import date, datetime, time

import market_ai.strategies.credit_spread_engine as cse
from market_ai.strategies.models import (
    LegSide,
    MarketSnapshot,
    OptionLeg,
    OptionType,
    RiskConfig,
    StrategyType,
    TrendContext,
    TrendSide,
)
from market_ai.strategies.strategy_selector import StrategySelector


def _market():
    return MarketSnapshot(
        symbol="NIFTY",
        spot=24000,
        candles_5m=[
            {"open": 23970, "high": 24010, "low": 23960, "close": 23990, "volume": 10000},
            {"open": 23990, "high": 24015, "low": 23970, "close": 23995, "volume": 9500},
        ],
        yesterday_high=24060,
        yesterday_low=23920,
        yesterday_close=23980,
        india_vix=12.0,
        now=datetime(2025, 11, 20, 10, 0, 0),
    )


def _risk():
    return RiskConfig(
        max_intraday_loss=-3000,
        intraday_target=4000,
        allow_carry_forward=False,
        max_carry_days=0,
        vix_carry_threshold=15.0,
        last_entry_time=time(14, 45),
        force_exit_time=time(15, 15),
        max_daily_loss_pct=0.03,
        max_daily_trades=5,
    )


def _chain(expiry):
    rows = []
    for i in range(-14, 15):
        strike = 24000 + i * 50
        ltp = 4.0
        rows.append({"expiry": expiry, "option_type": "CE", "strike": strike, "delta": 0.05 + i * 0.02, "ltp": ltp, "security_id": f"CE_{strike}"})
        rows.append({"expiry": expiry, "option_type": "PE", "strike": strike, "delta": -0.05 - i * 0.02, "ltp": ltp, "security_id": f"PE_{strike}"})
    return rows

cse.SPREAD_GAP_POINTS = 100


def _fake_spread(expiry, option_type):
    if option_type == OptionType.PUT:
        return [
            OptionLeg("NIFTY", expiry, 24000, OptionType.PUT, LegSide.SELL, 75, entry_price=20.0, strategy_type=StrategyType.BULL_PUT_SPREAD),
            OptionLeg("NIFTY", expiry, 23800, OptionType.PUT, LegSide.BUY, 75, entry_price=4.0, strategy_type=StrategyType.BULL_PUT_SPREAD),
        ]
    return [
        OptionLeg("NIFTY", expiry, 24000, OptionType.CALL, LegSide.SELL, 75, entry_price=20.0, strategy_type=StrategyType.BEAR_CALL_SPREAD),
        OptionLeg("NIFTY", expiry, 24200, OptionType.CALL, LegSide.BUY, 75, entry_price=4.0, strategy_type=StrategyType.BEAR_CALL_SPREAD),
    ]


def test_selector_opens_strangle_when_sideways():
    selector = StrategySelector(symbol="NIFTY", lot_size=75)
    expiry = date(2025, 11, 27)
    decision = selector.decide(
        market=_market(),
        option_chain=_chain(expiry),
        expiry=expiry,
        risk=_risk(),
        current_positions=[],
        basket_mtm=0.0,
        trend_ctx=TrendContext(TrendSide.SIDEWAYS, 1, 0, 1),
    )
    assert decision.action_type == "OPEN"
    assert decision.strategy_type.name == "STRANGLE"
    assert len(decision.legs_to_open) == 4


def test_selector_opens_bull_put_spread_for_bullish():
    selector = StrategySelector(symbol="NIFTY", lot_size=75)
    selector.credit_spread_engine.build_bull_put_spread = (
        lambda spot, chain, exp, lot, short_delta=None: _fake_spread(exp, OptionType.PUT)
    )
    expiry = date(2025, 11, 27)
    market = MarketSnapshot(
        symbol="NIFTY",
        spot=24150,
        candles_5m=[
            {"open": 23900, "high": 24080, "low": 23890, "close": 24060, "volume": 15000},
            {"open": 24060, "high": 24180, "low": 24010, "close": 24120, "volume": 14000},
        ],
        yesterday_high=24020,
        yesterday_low=23800,
        yesterday_close=23890,
        india_vix=12.0,
        now=datetime(2025, 11, 20, 10, 5, 0),
    )
    decision = selector.decide(
        market=market,
        option_chain=_chain(expiry),
        expiry=expiry,
        risk=_risk(),
        current_positions=[],
        basket_mtm=0.0,
        trend_ctx=TrendContext(TrendSide.BULL, 4, 1, 3),
    )
    assert decision.action_type == "OPEN"
    assert decision.strategy_type.name == "BULL_PUT_SPREAD"
    assert len(decision.legs_to_open) == 2
