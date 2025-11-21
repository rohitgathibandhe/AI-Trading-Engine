from datetime import date, datetime, time

from market_ai.strategies.models import (
    LegSide,
    MarketSnapshot,
    OptionLeg,
    OptionType,
    StrategyType,
    TrendContext,
    TrendSide,
)
from market_ai.strategies.trend_detector import detect_trend_from_open
from market_ai.strategies.strangle_engine import StrangleEngine
from market_ai.strategies.credit_spread_engine import CreditSpreadEngine


def _sample_chain(expiry: date):
    def row(strike, opt, delta, ltp=10.0):
        return {
            "expiry": expiry,
            "option_type": opt,
            "strike": strike,
            "delta": delta,
            "ltp": ltp,
            "security_id": f"{opt}_{strike}",
        }

    rows = []
    for i in range(-14, 15):
        strike = 24000 + i * 50
        ltp = 4.0
        rows.append(row(strike, "CE", 0.05 + i * 0.01, ltp=ltp))
        rows.append(row(strike, "PE", -0.05 - i * 0.01, ltp=ltp))
    return rows


def test_trend_detector_bullish_gap():
    now = datetime(2025, 11, 20, 9, 21)
    candles = [
        {"open": 24100, "close": 24180, "high": 24200, "low": 24100, "volume": 200000},
        {"open": 24180, "close": 24220, "high": 24250, "low": 24180, "volume": 180000},
    ]
    market = MarketSnapshot(
        symbol="NIFTY",
        spot=24200,
        candles_5m=candles,
        yesterday_high=24000,
        yesterday_low=23800,
        yesterday_close=23900,
        india_vix=12.0,
        now=now,
    )
    ctx = detect_trend_from_open(market, avg_5m_volume=100000, vwap=24150, pivot=24020)
    assert ctx.trend_side == TrendSide.BULL
    assert ctx.bull_score > ctx.bear_score


def test_trend_detector_sideways_when_scores_close():
    now = datetime(2025, 11, 20, 9, 22)
    candles = [
        {"open": 24000, "close": 24010, "high": 24020, "low": 23990, "volume": 90000},
    ]
    market = MarketSnapshot(
        symbol="NIFTY",
        spot=24005,
        candles_5m=candles,
        yesterday_high=24050,
        yesterday_low=23950,
        yesterday_close=24000,
        india_vix=11.0,
        now=now,
    )
    ctx = detect_trend_from_open(market, avg_5m_volume=120000, vwap=24010, pivot=24000)
    assert ctx.trend_side == TrendSide.SIDEWAYS


def test_strangle_engine_builds_legs():
    expiry = date(2025, 11, 27)
    engine = StrangleEngine(hedge_gap=150)
    legs = engine.build_neutral_strangle(spot=24000, option_chain=_sample_chain(expiry), expiry=expiry, lot_size=75)
    assert len(legs) == 4
    shorts = [leg for leg in legs if leg.side == LegSide.SELL]
    assert shorts[0].option_type != shorts[1].option_type


def test_credit_spread_engine_builds_bull_put():
    expiry = date(2025, 11, 27)
    engine = CreditSpreadEngine()
    legs = engine.build_bull_put_spread(spot=24000, option_chain=_sample_chain(expiry), expiry=expiry, lot_size=75)
    assert len(legs) == 2
    assert legs[0].side == LegSide.SELL and legs[0].option_type == OptionType.PUT
    assert legs[1].side == LegSide.BUY


def test_strangle_per_leg_sl_triggers_exit():
    engine = StrangleEngine()
    leg = OptionLeg(
        symbol="NIFTY",
        expiry=date(2025, 11, 27),
        strike=24000,
        option_type=OptionType.CALL,
        side=LegSide.SELL,
        quantity=75,
        entry_price=100.0,
        current_ltp=170.0,
        strategy_type=StrategyType.STRANGLE,
    )
    decision = engine.manage_strangle_positions(
        [leg],
        basket_mtm=0.0,
        tp=4000.0,
        sl=-3000.0,
        partial_tp=0.0,
        per_leg_sl=1.5,
        per_leg_tp=0.5,
        now=datetime(2025, 11, 20, 10, 0, 0),
        last_entry_time=time(14, 45),
    )
    assert decision.action_type == "CLOSE_ALL"
    assert "Per-leg SL hit" in decision.reason


def test_credit_spread_per_leg_tp_triggers_exit():
    engine = CreditSpreadEngine()
    legs = [
        OptionLeg(
            symbol="NIFTY",
            expiry=date(2025, 11, 27),
            strike=23800,
            option_type=OptionType.PUT,
            side=LegSide.SELL,
            quantity=75,
            entry_price=80.0,
            current_ltp=30.0,
            strategy_type=StrategyType.BULL_PUT_SPREAD,
        ),
        OptionLeg(
            symbol="NIFTY",
            expiry=date(2025, 11, 27),
            strike=23600,
            option_type=OptionType.PUT,
            side=LegSide.BUY,
            quantity=75,
            entry_price=30.0,
            current_ltp=10.0,
            strategy_type=StrategyType.BULL_PUT_SPREAD,
        ),
    ]
    decision = engine.manage_spread_positions(
        legs,
        basket_mtm=0.0,
        tp=4000.0,
        sl=-3000.0,
        partial_tp=0.0,
        strategy=StrategyType.BULL_PUT_SPREAD,
        per_leg_sl=1.5,
        per_leg_tp=0.4,
    )
    assert decision.action_type == "CLOSE_ALL"
    assert "Per-leg TP hit" in decision.reason
