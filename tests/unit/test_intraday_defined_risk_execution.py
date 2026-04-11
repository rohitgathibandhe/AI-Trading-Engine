from __future__ import annotations

from datetime import date, datetime, timedelta

from market_ai.intraday_defined_risk.data_models import (
    AccountRiskLimits,
    AccountState,
    MarketSnapshot,
    OhlcvBar,
    OhlcvSeries,
    OpenPosition,
    OptionType,
    OptionsChainSnapshot,
    OptionsContractQuote,
    RegimeLabel,
    RegimeState,
    StrategyLeg,
    StrategyType,
    TradeStructure,
)
from market_ai.intraday_defined_risk.execution import evaluate_exit


def test_evaluate_exit_uses_vwap_invalidation_for_bull_put() -> None:
    entry_structure = TradeStructure(
        strategy=StrategyType.BULL_PUT_CREDIT_SPREAD,
        legs=[
            StrategyLeg(
                action="SELL",
                option_type=OptionType.PUT,
                strike=95.0,
                quote=OptionsContractQuote(strike=95.0, option_type=OptionType.PUT, bid=11.0, ask=13.0, ltp=12.0, delta=-0.22),
            ),
            StrategyLeg(
                action="BUY",
                option_type=OptionType.PUT,
                strike=90.0,
                quote=OptionsContractQuote(strike=90.0, option_type=OptionType.PUT, bid=4.0, ask=6.0, ltp=5.0, delta=-0.08),
            ),
        ],
        credit_points=7.0,
        width_points=5.0,
    )
    position = OpenPosition(
        structure=entry_structure,
        lots=1,
        lot_size=65,
        entry_time=datetime(2026, 3, 30, 10, 0),
        entry_credit_points=7.0,
        target_value_points=2.45,
        stop_value_points=14.0,
        max_loss_rupees_per_lot=0.0,
        take_profit_capture_pct=0.65,
    )
    bars_5m = [
        OhlcvBar(timestamp=datetime(2026, 3, 30, 9, 55), open=100.5, high=101.0, low=99.7, close=99.8, volume=1000),
        OhlcvBar(timestamp=datetime(2026, 3, 30, 10, 0), open=99.8, high=100.0, low=99.2, close=99.3, volume=1100),
        OhlcvBar(timestamp=datetime(2026, 3, 30, 10, 5), open=99.3, high=99.4, low=98.8, close=99.0, volume=1200),
    ]
    bars_15m = [
        OhlcvBar(timestamp=datetime(2026, 3, 30, 9, 45), open=100.2, high=101.0, low=99.6, close=99.8, volume=3000),
        OhlcvBar(timestamp=datetime(2026, 3, 30, 10, 0), open=99.8, high=100.0, low=98.8, close=99.0, volume=3200),
    ]
    snapshot = MarketSnapshot(
        nifty_5m=OhlcvSeries(timeframe_minutes=5, bars=bars_5m),
        nifty_15m=OhlcvSeries(timeframe_minutes=15, bars=bars_15m),
        option_chain=OptionsChainSnapshot(
            timestamp=datetime(2026, 3, 30, 10, 5),
            expiry=date(2026, 4, 2),
            spot=99.0,
            quotes=[
                OptionsContractQuote(strike=95.0, option_type=OptionType.PUT, bid=15.0, ask=17.0, ltp=16.0, delta=-0.30),
                OptionsContractQuote(strike=90.0, option_type=OptionType.PUT, bid=7.0, ask=9.0, ltp=8.0, delta=-0.12),
            ],
        ),
        risk_limits=AccountRiskLimits(
            max_risk_rupees_per_trade=10000.0,
            max_margin_rupees=100000.0,
            max_daily_loss_rupees=5000.0,
        ),
        account_state=AccountState(),
        live_vwap=100.0,
        lot_size=65,
        slippage_points=0.5,
    )
    regime = RegimeState(
        regime=RegimeLabel.NO_TRADE,
        trend_15m="NEUTRAL",
        execution_5m="UNCONFIRMED",
        ema20_15m=None,
        ema20_slope_15m=None,
        rv30_pct=0.4,
        or_length_minutes=15,
        opening_range_high=101.0,
        opening_range_low=99.5,
        vwap=100.0,
        confidence=0.1,
        reasons=["VWAP broken to downside."],
    )

    exit_decision = evaluate_exit(
        position,
        current_snapshot=snapshot,
        current_regime=regime,
        now=snapshot.timestamp,
    )

    assert exit_decision.should_exit is True
    assert exit_decision.reason == "VWAP_INVALIDATION"
    assert exit_decision.current_value_points == 8.0
    assert exit_decision.pnl_rupees == (7.0 - 8.0) * 65


def test_evaluate_exit_uses_ema20_invalidation_for_bear_call() -> None:
    entry_structure = TradeStructure(
        strategy=StrategyType.BEAR_CALL_CREDIT_SPREAD,
        legs=[
            StrategyLeg(
                action="SELL",
                option_type=OptionType.CALL,
                strike=105.0,
                quote=OptionsContractQuote(strike=105.0, option_type=OptionType.CALL, bid=10.0, ask=12.0, ltp=11.0, delta=0.22),
            ),
            StrategyLeg(
                action="BUY",
                option_type=OptionType.CALL,
                strike=110.0,
                quote=OptionsContractQuote(strike=110.0, option_type=OptionType.CALL, bid=4.0, ask=6.0, ltp=5.0, delta=0.08),
            ),
        ],
        credit_points=6.0,
        width_points=5.0,
    )
    position = OpenPosition(
        structure=entry_structure,
        lots=1,
        lot_size=65,
        entry_time=datetime(2026, 3, 30, 13, 0),
        entry_credit_points=6.0,
        target_value_points=2.1,
        stop_value_points=12.0,
        max_loss_rupees_per_lot=0.0,
        take_profit_capture_pct=0.65,
    )
    start = datetime(2026, 3, 30, 11, 30)
    close_values = [
        100.0, 99.8, 99.6, 99.5, 99.4, 99.2, 99.1, 99.0, 98.9, 98.8,
        98.7, 98.8, 99.0, 99.2, 99.4, 99.7, 100.1, 100.4, 100.8, 101.2,
    ]
    bars_5m = []
    previous_close = close_values[0] + 0.1
    for idx, close in enumerate(close_values):
        open_ = previous_close
        bars_5m.append(
            OhlcvBar(
                timestamp=start + timedelta(minutes=5 * idx),
                open=open_,
                high=max(open_, close) + 0.2,
                low=min(open_, close) - 0.2,
                close=close,
                volume=1000 + idx,
            )
        )
        previous_close = close
    bars_15m = [
        OhlcvBar(timestamp=datetime(2026, 3, 30, 12, 30), open=99.4, high=99.6, low=98.8, close=99.0, volume=3000),
        OhlcvBar(timestamp=datetime(2026, 3, 30, 12, 45), open=99.0, high=99.4, low=98.7, close=98.9, volume=3200),
        OhlcvBar(timestamp=datetime(2026, 3, 30, 13, 0), open=98.9, high=101.3, low=98.8, close=101.2, volume=3500),
    ]
    snapshot = MarketSnapshot(
        nifty_5m=OhlcvSeries(timeframe_minutes=5, bars=bars_5m),
        nifty_15m=OhlcvSeries(timeframe_minutes=15, bars=bars_15m),
        option_chain=OptionsChainSnapshot(
            timestamp=datetime(2026, 3, 30, 13, 5),
            expiry=date(2026, 4, 2),
            spot=101.2,
            quotes=[
                OptionsContractQuote(strike=105.0, option_type=OptionType.CALL, bid=14.0, ask=16.0, ltp=15.0, delta=0.30),
                OptionsContractQuote(strike=110.0, option_type=OptionType.CALL, bid=6.0, ask=8.0, ltp=7.0, delta=0.12),
            ],
        ),
        risk_limits=AccountRiskLimits(
            max_risk_rupees_per_trade=10000.0,
            max_margin_rupees=100000.0,
            max_daily_loss_rupees=5000.0,
        ),
        account_state=AccountState(),
        live_vwap=100.0,
        lot_size=65,
        slippage_points=0.5,
    )
    regime = RegimeState(
        regime=RegimeLabel.DOWN_TREND,
        trend_15m="TREND_DOWN",
        execution_5m="DOWN_CONFIRMED",
        ema20_15m=99.5,
        ema20_slope_15m=-0.5,
        rv30_pct=0.5,
        or_length_minutes=30,
        opening_range_high=100.0,
        opening_range_low=98.5,
        vwap=100.0,
        confidence=0.9,
        reasons=["Bearish regime, and price reclaimed EMA20 and VWAP for two bars."],
    )

    exit_decision = evaluate_exit(
        position,
        current_snapshot=snapshot,
        current_regime=regime,
        now=snapshot.timestamp,
    )

    assert exit_decision.should_exit is True
    assert exit_decision.reason == "EMA20_INVALIDATION"


def test_bear_call_does_not_exit_on_single_ema20_close_reclaim() -> None:
    entry_structure = TradeStructure(
        strategy=StrategyType.BEAR_CALL_CREDIT_SPREAD,
        legs=[
            StrategyLeg(
                action="SELL",
                option_type=OptionType.CALL,
                strike=105.0,
                quote=OptionsContractQuote(strike=105.0, option_type=OptionType.CALL, bid=10.0, ask=12.0, ltp=11.0, delta=0.22),
            ),
            StrategyLeg(
                action="BUY",
                option_type=OptionType.CALL,
                strike=110.0,
                quote=OptionsContractQuote(strike=110.0, option_type=OptionType.CALL, bid=4.0, ask=6.0, ltp=5.0, delta=0.08),
            ),
        ],
        credit_points=6.0,
        width_points=5.0,
    )
    position = OpenPosition(
        structure=entry_structure,
        lots=1,
        lot_size=65,
        entry_time=datetime(2026, 3, 30, 13, 0),
        entry_credit_points=6.0,
        target_value_points=2.1,
        stop_value_points=12.0,
        max_loss_rupees_per_lot=0.0,
        take_profit_capture_pct=0.65,
    )
    start = datetime(2026, 3, 30, 12, 0)
    close_values = [99.2, 99.0, 98.9, 98.8, 98.7, 98.6, 98.8, 99.0, 99.2, 100.4]
    bars_5m = []
    previous_close = 99.3
    for idx, close in enumerate(close_values):
        open_ = previous_close
        bars_5m.append(
            OhlcvBar(
                timestamp=start + timedelta(minutes=5 * idx),
                open=open_,
                high=max(open_, close) + 0.2,
                low=min(open_, close) - 0.2,
                close=close,
                volume=1000 + idx,
            )
        )
        previous_close = close
    bars_15m = [
        OhlcvBar(timestamp=datetime(2026, 3, 30, 12, 30), open=98.9, high=99.4, low=98.7, close=98.8, volume=3000),
        OhlcvBar(timestamp=datetime(2026, 3, 30, 12, 45), open=98.8, high=100.6, low=98.7, close=100.4, volume=3200),
    ]
    snapshot = MarketSnapshot(
        nifty_5m=OhlcvSeries(timeframe_minutes=5, bars=bars_5m),
        nifty_15m=OhlcvSeries(timeframe_minutes=15, bars=bars_15m),
        option_chain=OptionsChainSnapshot(
            timestamp=datetime(2026, 3, 30, 12, 45),
            expiry=date(2026, 4, 2),
            spot=100.4,
            quotes=[
                OptionsContractQuote(strike=105.0, option_type=OptionType.CALL, bid=12.0, ask=14.0, ltp=13.0, delta=0.27),
                OptionsContractQuote(strike=110.0, option_type=OptionType.CALL, bid=5.0, ask=7.0, ltp=6.0, delta=0.10),
            ],
        ),
        risk_limits=AccountRiskLimits(
            max_risk_rupees_per_trade=10000.0,
            max_margin_rupees=100000.0,
            max_daily_loss_rupees=5000.0,
        ),
        account_state=AccountState(),
        live_vwap=100.0,
        lot_size=65,
        slippage_points=0.5,
    )
    regime = RegimeState(
        regime=RegimeLabel.DOWN_TREND,
        trend_15m="TREND_DOWN",
        execution_5m="DOWN_CONFIRMED",
        ema20_15m=99.5,
        ema20_slope_15m=-0.2,
        rv30_pct=0.5,
        or_length_minutes=30,
        opening_range_high=100.0,
        opening_range_low=98.5,
        vwap=100.0,
        confidence=0.8,
        reasons=["Single close reclaimed EMA20, but not enough confirmation."],
    )

    exit_decision = evaluate_exit(
        position,
        current_snapshot=snapshot,
        current_regime=regime,
        now=snapshot.timestamp,
    )

    assert exit_decision.should_exit is False
    assert exit_decision.reason == "HOLD"


def test_evaluate_exit_uses_daily_profit_trail_after_session_target() -> None:
    entry_structure = TradeStructure(
        strategy=StrategyType.BULL_PUT_CREDIT_SPREAD,
        legs=[
            StrategyLeg(
                action="SELL",
                option_type=OptionType.PUT,
                strike=95.0,
                quote=OptionsContractQuote(strike=95.0, option_type=OptionType.PUT, bid=11.0, ask=13.0, ltp=12.0, delta=-0.22),
            ),
            StrategyLeg(
                action="BUY",
                option_type=OptionType.PUT,
                strike=90.0,
                quote=OptionsContractQuote(strike=90.0, option_type=OptionType.PUT, bid=4.0, ask=6.0, ltp=5.0, delta=-0.08),
            ),
        ],
        credit_points=70.0,
        width_points=100.0,
    )
    position = OpenPosition(
        structure=entry_structure,
        lots=1,
        lot_size=65,
        entry_time=datetime(2026, 3, 30, 13, 0),
        entry_credit_points=70.0,
        target_value_points=24.5,
        stop_value_points=140.0,
        max_loss_rupees_per_lot=0.0,
        take_profit_capture_pct=0.65,
        metadata={"session_profit_peak_rupees": 5200.0},
    )
    bars_5m = [
        OhlcvBar(timestamp=datetime(2026, 3, 30, 13, 0), open=100.0, high=100.5, low=99.8, close=100.2, volume=1000),
        OhlcvBar(timestamp=datetime(2026, 3, 30, 13, 5), open=100.2, high=100.4, low=99.9, close=100.1, volume=1100),
    ]
    bars_15m = [
        OhlcvBar(timestamp=datetime(2026, 3, 30, 13, 0), open=100.0, high=100.5, low=99.8, close=100.1, volume=3000),
    ]
    snapshot = MarketSnapshot(
        nifty_5m=OhlcvSeries(timeframe_minutes=5, bars=bars_5m),
        nifty_15m=OhlcvSeries(timeframe_minutes=15, bars=bars_15m),
        option_chain=OptionsChainSnapshot(
            timestamp=datetime(2026, 3, 30, 13, 5),
            expiry=date(2026, 4, 2),
            spot=100.1,
            quotes=[
                    OptionsContractQuote(strike=95.0, option_type=OptionType.PUT, bid=69.0, ask=71.0, ltp=70.0, delta=-0.18),
                    OptionsContractQuote(strike=90.0, option_type=OptionType.PUT, bid=13.0, ask=15.0, ltp=14.0, delta=-0.06),
            ],
        ),
        risk_limits=AccountRiskLimits(
            max_risk_rupees_per_trade=10000.0,
            max_margin_rupees=100000.0,
            max_daily_loss_rupees=5000.0,
        ),
        account_state=AccountState(realised_pnl_rupees=3200.0, margin_used_rupees=0.0),
        live_vwap=100.0,
        lot_size=65,
        slippage_points=0.5,
    )
    regime = RegimeState(
        regime=RegimeLabel.UP_TREND,
        trend_15m="TREND_UP",
        execution_5m="UP_CONFIRMED",
        ema20_15m=99.8,
        ema20_slope_15m=0.3,
        rv30_pct=0.2,
        or_length_minutes=30,
        opening_range_high=100.1,
        opening_range_low=99.2,
        vwap=100.0,
        confidence=0.8,
        reasons=["Profit trail test."],
    )

    exit_decision = evaluate_exit(
        position,
        current_snapshot=snapshot,
        current_regime=regime,
        now=snapshot.timestamp,
    )

    assert exit_decision.should_exit is True
    assert exit_decision.reason == "DAILY_PROFIT_TRAIL"
