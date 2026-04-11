from __future__ import annotations

from datetime import date, datetime, timedelta

from market_ai.intraday_defined_risk.data_models import (
    AccountRiskLimits,
    AccountState,
    AdaptiveParameters,
    MarketSnapshot,
    OhlcvBar,
    OhlcvSeries,
    OptionType,
    ORLevels,
    OptionsChainSnapshot,
    OptionsContractQuote,
    RegimeLabel,
    RegimeState,
    StrategyType,
)
from market_ai.intraday_defined_risk.regime import classify_regime
from market_ai.intraday_defined_risk.risk import assess_trade_risk, compute_max_loss_rupees_per_lot, kill_switch_triggered
from market_ai.intraday_defined_risk.strikes import select_structure
from market_ai.intraday_defined_risk.features import (
    bearish_candle_context,
    bullish_candle_context,
    option_chain_oi_flow,
    option_chain_wall_migration,
)
import market_ai.intraday_defined_risk.strategy as strategy_module


def _build_15m_downtrend_bars() -> list[OhlcvBar]:
    start = datetime(2026, 4, 3, 9, 15)
    raw = [
        (22000.0, 22080.0, 21920.0, 21960.0),
        (21960.0, 22020.0, 21880.0, 21910.0),
        (21910.0, 22060.0, 21890.0, 21990.0),
        (21990.0, 22010.0, 21780.0, 21830.0),
        (21830.0, 22020.0, 21810.0, 21910.0),
        (21910.0, 21930.0, 21680.0, 21730.0),
        (21730.0, 21970.0, 21710.0, 21810.0),
        (21810.0, 21830.0, 21580.0, 21630.0),
        (21630.0, 21920.0, 21610.0, 21710.0),
        (21710.0, 21730.0, 21480.0, 21530.0),
    ]
    bars: list[OhlcvBar] = []
    for idx, (open_, high, low, close) in enumerate(raw):
        bars.append(OhlcvBar(timestamp=start + timedelta(minutes=15 * idx), open=open_, high=high, low=low, close=close, volume=1000 + idx))
    return bars


def _build_5m_downtrend_bars() -> list[OhlcvBar]:
    start = datetime(2026, 4, 3, 9, 15)
    raw = [
        (22000.0, 22040.0, 21970.0, 22020.0),
        (22020.0, 22060.0, 21990.0, 22040.0),
        (22040.0, 22050.0, 21980.0, 22000.0),
        (22000.0, 22010.0, 21920.0, 21930.0),
        (21930.0, 21950.0, 21880.0, 21900.0),
        (21900.0, 21910.0, 21840.0, 21860.0),
        (21860.0, 21880.0, 21810.0, 21830.0),
        (21830.0, 21840.0, 21790.0, 21800.0),
    ]
    bars: list[OhlcvBar] = []
    for idx, (open_, high, low, close) in enumerate(raw):
        bars.append(OhlcvBar(timestamp=start + timedelta(minutes=5 * idx), open=open_, high=high, low=low, close=close, volume=2000 + idx))
    return bars


def _build_15m_uptrend_bars() -> list[OhlcvBar]:
    start = datetime(2026, 4, 3, 9, 15)
    raw = [
        (22000.0, 22040.0, 21970.0, 22030.0),
        (22030.0, 22090.0, 22005.0, 22075.0),
        (22075.0, 22095.0, 22035.0, 22055.0),
        (22055.0, 22120.0, 22045.0, 22100.0),
        (22100.0, 22135.0, 22070.0, 22095.0),
        (22095.0, 22170.0, 22090.0, 22155.0),
        (22155.0, 22190.0, 22115.0, 22145.0),
        (22145.0, 22220.0, 22135.0, 22200.0),
        (22200.0, 22235.0, 22160.0, 22195.0),
        (22195.0, 22280.0, 22185.0, 22260.0),
    ]
    bars: list[OhlcvBar] = []
    for idx, (open_, high, low, close) in enumerate(raw):
        bars.append(OhlcvBar(timestamp=start + timedelta(minutes=15 * idx), open=open_, high=high, low=low, close=close, volume=1200 + idx))
    return bars


def _build_5m_uptrend_pullback_bars() -> list[OhlcvBar]:
    start = datetime(2026, 4, 3, 9, 15)
    raw = [
        (22000.0, 22025.0, 21995.0, 22020.0),
        (22020.0, 22045.0, 22010.0, 22040.0),
        (22040.0, 22065.0, 22030.0, 22060.0),
        (22060.0, 22085.0, 22050.0, 22080.0),
        (22080.0, 22105.0, 22070.0, 22100.0),
        (22100.0, 22110.0, 22088.0, 22092.0),
        (22092.0, 22100.0, 22078.0, 22084.0),
        (22084.0, 22118.0, 22080.0, 22112.0),
    ]
    bars: list[OhlcvBar] = []
    for idx, (open_, high, low, close) in enumerate(raw):
        bars.append(OhlcvBar(timestamp=start + timedelta(minutes=5 * idx), open=open_, high=high, low=low, close=close, volume=2200 + idx))
    return bars


def _build_option_chain(spot: float) -> OptionsChainSnapshot:
    ts = datetime(2026, 4, 3, 9, 50)
    quotes = [
        OptionsContractQuote(strike=spot + 100, option_type=OptionType.CALL, bid=38.0, ask=40.0, ltp=39.0, delta=0.31),
        OptionsContractQuote(strike=spot + 200, option_type=OptionType.CALL, bid=18.0, ask=19.0, ltp=18.5, delta=0.24),
        OptionsContractQuote(strike=spot + 300, option_type=OptionType.CALL, bid=31.0, ask=32.0, ltp=31.5, delta=0.22),
        OptionsContractQuote(strike=spot + 400, option_type=OptionType.CALL, bid=5.0, ask=5.5, ltp=5.2, delta=0.10),
        OptionsContractQuote(strike=spot - 400, option_type=OptionType.PUT, bid=3.4, ask=3.8, ltp=3.6, delta=-0.03),
        OptionsContractQuote(strike=spot - 300, option_type=OptionType.PUT, bid=5.0, ask=5.5, ltp=5.2, delta=-0.05),
        OptionsContractQuote(strike=spot - 200, option_type=OptionType.PUT, bid=31.0, ask=32.0, ltp=31.5, delta=-0.16),
        OptionsContractQuote(strike=spot - 100, option_type=OptionType.PUT, bid=39.0, ask=40.0, ltp=39.5, delta=-0.31),
    ]
    return OptionsChainSnapshot(timestamp=ts, expiry=date(2026, 4, 9), spot=spot, quotes=quotes, margin_estimate_per_lot=12000.0)


def _build_snapshot() -> MarketSnapshot:
    bars_5m = _build_5m_downtrend_bars()
    spot = bars_5m[-1].close
    return MarketSnapshot(
        nifty_5m=OhlcvSeries(timeframe_minutes=5, bars=bars_5m),
        nifty_15m=OhlcvSeries(timeframe_minutes=15, bars=_build_15m_downtrend_bars()),
        option_chain=_build_option_chain(spot=spot),
        risk_limits=AccountRiskLimits(
            max_risk_rupees_per_trade=10000.0,
            max_margin_rupees=100000.0,
            max_daily_loss_rupees=5000.0,
        ),
        account_state=AccountState(realised_pnl_rupees=0.0, margin_used_rupees=10000.0),
        live_vwap=21990.0,
        or_levels={45: ORLevels(length_minutes=45, high=22060.0, low=21980.0)},
        lot_size=65,
    )


def _build_bullish_option_chain(spot: float) -> OptionsChainSnapshot:
    ts = datetime(2026, 4, 3, 9, 50)
    quotes = [
        OptionsContractQuote(strike=spot + 100, option_type=OptionType.CALL, bid=34.0, ask=35.0, ltp=34.5, delta=0.18),
        OptionsContractQuote(strike=spot + 200, option_type=OptionType.CALL, bid=16.0, ask=16.8, ltp=16.4, delta=0.09),
        OptionsContractQuote(strike=spot - 50, option_type=OptionType.PUT, bid=33.0, ask=34.0, ltp=33.5, delta=-0.31),
        OptionsContractQuote(strike=spot - 100, option_type=OptionType.PUT, bid=23.0, ask=24.0, ltp=23.5, delta=-0.22),
        OptionsContractQuote(strike=spot - 200, option_type=OptionType.PUT, bid=6.0, ask=6.6, ltp=6.3, delta=-0.08),
        OptionsContractQuote(strike=spot - 300, option_type=OptionType.PUT, bid=3.8, ask=4.2, ltp=4.0, delta=-0.05),
    ]
    return OptionsChainSnapshot(timestamp=ts, expiry=date(2026, 4, 9), spot=spot, quotes=quotes, margin_estimate_per_lot=11500.0)


def _build_previous_bullish_option_chain(spot: float) -> OptionsChainSnapshot:
    ts = datetime(2026, 4, 3, 9, 45)
    quotes = [
        OptionsContractQuote(strike=spot + 100, option_type=OptionType.CALL, bid=36.0, ask=37.0, ltp=36.5, delta=0.18, oi=180000),
        OptionsContractQuote(strike=spot + 200, option_type=OptionType.CALL, bid=17.0, ask=17.8, ltp=17.4, delta=0.09, oi=140000),
        OptionsContractQuote(strike=spot - 50, option_type=OptionType.PUT, bid=35.0, ask=36.0, ltp=35.5, delta=-0.31, oi=160000),
        OptionsContractQuote(strike=spot - 100, option_type=OptionType.PUT, bid=24.0, ask=25.0, ltp=24.5, delta=-0.22, oi=200000),
        OptionsContractQuote(strike=spot - 200, option_type=OptionType.PUT, bid=6.5, ask=7.1, ltp=6.8, delta=-0.08, oi=120000),
        OptionsContractQuote(strike=spot - 300, option_type=OptionType.PUT, bid=4.0, ask=4.4, ltp=4.2, delta=-0.05, oi=90000),
    ]
    return OptionsChainSnapshot(timestamp=ts, expiry=date(2026, 4, 9), spot=spot - 10.0, quotes=quotes, margin_estimate_per_lot=11500.0)


def _build_bullish_snapshot() -> MarketSnapshot:
    bars_5m = _build_5m_uptrend_pullback_bars()
    spot = bars_5m[-1].close
    return MarketSnapshot(
        nifty_5m=OhlcvSeries(timeframe_minutes=5, bars=bars_5m),
        nifty_15m=OhlcvSeries(timeframe_minutes=15, bars=_build_15m_uptrend_bars()),
        option_chain=OptionsChainSnapshot(
            timestamp=datetime(2026, 4, 3, 9, 50),
            expiry=date(2026, 4, 9),
            spot=spot,
            quotes=[
                OptionsContractQuote(strike=spot + 100, option_type=OptionType.CALL, bid=34.0, ask=35.0, ltp=34.5, delta=0.18, oi=150000),
                OptionsContractQuote(strike=spot + 200, option_type=OptionType.CALL, bid=16.0, ask=16.8, ltp=16.4, delta=0.09, oi=120000),
                OptionsContractQuote(strike=spot - 50, option_type=OptionType.PUT, bid=33.0, ask=34.0, ltp=33.5, delta=-0.31, oi=220000),
                OptionsContractQuote(strike=spot - 100, option_type=OptionType.PUT, bid=23.0, ask=24.0, ltp=23.5, delta=-0.22, oi=280000),
                OptionsContractQuote(strike=spot - 200, option_type=OptionType.PUT, bid=6.0, ask=6.6, ltp=6.3, delta=-0.08, oi=180000),
                OptionsContractQuote(strike=spot - 300, option_type=OptionType.PUT, bid=3.8, ask=4.2, ltp=4.0, delta=-0.05, oi=130000),
            ],
            margin_estimate_per_lot=11500.0,
        ),
        risk_limits=AccountRiskLimits(
            max_risk_rupees_per_trade=10000.0,
            max_margin_rupees=100000.0,
            max_daily_loss_rupees=5000.0,
        ),
        account_state=AccountState(realised_pnl_rupees=0.0, margin_used_rupees=10000.0),
        live_vwap=22088.0,
        or_levels={30: ORLevels(length_minutes=30, high=22060.0, low=21995.0)},
        lot_size=65,
        previous_option_chain=_build_previous_bullish_option_chain(spot=spot),
        previous_session_close=21920.0,
    )


def _build_gap_up_bullish_snapshot() -> MarketSnapshot:
    start = datetime(2026, 4, 6, 9, 15)
    raw_5m = [
        (22120.0, 22142.0, 22112.0, 22135.0),
        (22135.0, 22158.0, 22126.0, 22150.0),
        (22150.0, 22174.0, 22144.0, 22166.0),
        (22166.0, 22184.0, 22158.0, 22178.0),
        (22178.0, 22186.0, 22160.0, 22168.0),
        (22168.0, 22195.0, 22162.0, 22188.0),
        (22188.0, 22208.0, 22182.0, 22202.0),
        (22202.0, 22222.0, 22196.0, 22218.0),
    ]
    bars_5m = [
        OhlcvBar(timestamp=start + timedelta(minutes=5 * idx), open=open_, high=high, low=low, close=close, volume=2600 + idx)
        for idx, (open_, high, low, close) in enumerate(raw_5m)
    ]
    raw_15m = [
        (22080.0, 22120.0, 22072.0, 22110.0),
        (22110.0, 22172.0, 22100.0, 22158.0),
        (22158.0, 22218.0, 22148.0, 22202.0),
        (22202.0, 22264.0, 22192.0, 22248.0),
        (22248.0, 22298.0, 22234.0, 22284.0),
        (22284.0, 22318.0, 22260.0, 22302.0),
    ]
    bars_15m = [
        OhlcvBar(timestamp=start + timedelta(minutes=15 * idx), open=open_, high=high, low=low, close=close, volume=1400 + idx)
        for idx, (open_, high, low, close) in enumerate(raw_15m)
    ]
    spot = bars_5m[-1].close
    current_quotes = [
        OptionsContractQuote(strike=spot + 100, option_type=OptionType.CALL, bid=33.0, ask=34.0, ltp=33.5, delta=0.19, oi=155000),
        OptionsContractQuote(strike=spot + 200, option_type=OptionType.CALL, bid=15.5, ask=16.2, ltp=15.8, delta=0.09, oi=118000),
        OptionsContractQuote(strike=spot - 50, option_type=OptionType.PUT, bid=32.0, ask=33.0, ltp=32.5, delta=-0.31, oi=235000),
        OptionsContractQuote(strike=spot - 100, option_type=OptionType.PUT, bid=22.5, ask=23.3, ltp=22.9, delta=-0.22, oi=295000),
        OptionsContractQuote(strike=spot - 200, option_type=OptionType.PUT, bid=6.1, ask=6.6, ltp=6.3, delta=-0.08, oi=185000),
        OptionsContractQuote(strike=spot - 300, option_type=OptionType.PUT, bid=3.8, ask=4.2, ltp=4.0, delta=-0.05, oi=132000),
    ]
    previous_quotes = [
        OptionsContractQuote(strike=spot + 100, option_type=OptionType.CALL, bid=34.0, ask=35.0, ltp=34.5, delta=0.19, oi=170000),
        OptionsContractQuote(strike=spot + 200, option_type=OptionType.CALL, bid=16.0, ask=16.8, ltp=16.4, delta=0.09, oi=130000),
        OptionsContractQuote(strike=spot - 50, option_type=OptionType.PUT, bid=33.0, ask=34.0, ltp=33.5, delta=-0.31, oi=190000),
        OptionsContractQuote(strike=spot - 100, option_type=OptionType.PUT, bid=23.0, ask=24.0, ltp=23.5, delta=-0.22, oi=250000),
        OptionsContractQuote(strike=spot - 200, option_type=OptionType.PUT, bid=6.3, ask=6.8, ltp=6.5, delta=-0.08, oi=150000),
        OptionsContractQuote(strike=spot - 300, option_type=OptionType.PUT, bid=4.0, ask=4.4, ltp=4.2, delta=-0.05, oi=110000),
    ]
    return MarketSnapshot(
        nifty_5m=OhlcvSeries(timeframe_minutes=5, bars=bars_5m),
        nifty_15m=OhlcvSeries(timeframe_minutes=15, bars=bars_15m),
        option_chain=OptionsChainSnapshot(
            timestamp=bars_5m[-1].timestamp,
            expiry=date(2026, 4, 9),
            spot=spot,
            quotes=current_quotes,
            margin_estimate_per_lot=12000.0,
        ),
        risk_limits=AccountRiskLimits(10000.0, 100000.0, 5000.0),
        account_state=AccountState(realised_pnl_rupees=0.0, margin_used_rupees=10000.0),
        live_vwap=22172.0,
        or_levels={15: ORLevels(length_minutes=15, high=22166.0, low=22112.0), 30: ORLevels(length_minutes=30, high=22178.0, low=22112.0)},
        lot_size=65,
        previous_option_chain=OptionsChainSnapshot(
            timestamp=bars_5m[-2].timestamp,
            expiry=date(2026, 4, 9),
            spot=spot - 8.0,
            quotes=previous_quotes,
            margin_estimate_per_lot=12000.0,
        ),
        previous_session_close=21940.0,
    )


def _build_gap_down_bearish_snapshot() -> MarketSnapshot:
    start = datetime(2026, 4, 7, 9, 15)
    raw_5m = [
        (21840.0, 21848.0, 21808.0, 21818.0),
        (21818.0, 21826.0, 21792.0, 21800.0),
        (21800.0, 21818.0, 21770.0, 21784.0),
        (21784.0, 21796.0, 21756.0, 21766.0),
        (21766.0, 21792.0, 21760.0, 21772.0),
        (21772.0, 21780.0, 21742.0, 21752.0),
        (21752.0, 21760.0, 21720.0, 21728.0),
        (21728.0, 21736.0, 21698.0, 21708.0),
    ]
    bars_5m = [
        OhlcvBar(timestamp=start + timedelta(minutes=5 * idx), open=open_, high=high, low=low, close=close, volume=2550 + idx)
        for idx, (open_, high, low, close) in enumerate(raw_5m)
    ]
    raw_15m = [
        (21920.0, 21932.0, 21852.0, 21860.0),
        (21860.0, 21874.0, 21796.0, 21810.0),
        (21810.0, 21826.0, 21738.0, 21754.0),
        (21754.0, 21776.0, 21698.0, 21718.0),
        (21718.0, 21732.0, 21662.0, 21686.0),
        (21686.0, 21704.0, 21630.0, 21648.0),
    ]
    bars_15m = [
        OhlcvBar(timestamp=start + timedelta(minutes=15 * idx), open=open_, high=high, low=low, close=close, volume=1450 + idx)
        for idx, (open_, high, low, close) in enumerate(raw_15m)
    ]
    spot = bars_5m[-1].close
    current_quotes = [
        OptionsContractQuote(strike=spot + 100, option_type=OptionType.CALL, bid=33.0, ask=34.0, ltp=33.5, delta=0.20, oi=310000),
        OptionsContractQuote(strike=spot + 200, option_type=OptionType.CALL, bid=14.6, ask=15.4, ltp=15.0, delta=0.09, oi=205000),
        OptionsContractQuote(strike=spot - 100, option_type=OptionType.PUT, bid=23.4, ask=24.2, ltp=23.8, delta=-0.22, oi=150000),
        OptionsContractQuote(strike=spot - 200, option_type=OptionType.PUT, bid=6.6, ask=7.1, ltp=6.9, delta=-0.08, oi=98000),
    ]
    previous_quotes = [
        OptionsContractQuote(strike=spot + 100, option_type=OptionType.CALL, bid=35.0, ask=36.0, ltp=35.5, delta=0.20, oi=260000),
        OptionsContractQuote(strike=spot + 200, option_type=OptionType.CALL, bid=15.6, ask=16.4, ltp=16.0, delta=0.09, oi=180000),
        OptionsContractQuote(strike=spot - 100, option_type=OptionType.PUT, bid=22.8, ask=23.6, ltp=23.1, delta=-0.22, oi=190000),
        OptionsContractQuote(strike=spot - 200, option_type=OptionType.PUT, bid=6.1, ask=6.6, ltp=6.3, delta=-0.08, oi=125000),
    ]
    return MarketSnapshot(
        nifty_5m=OhlcvSeries(timeframe_minutes=5, bars=bars_5m),
        nifty_15m=OhlcvSeries(timeframe_minutes=15, bars=bars_15m),
        option_chain=OptionsChainSnapshot(
            timestamp=bars_5m[-1].timestamp,
            expiry=date(2026, 4, 9),
            spot=spot,
            quotes=current_quotes,
            margin_estimate_per_lot=12000.0,
        ),
        risk_limits=AccountRiskLimits(10000.0, 100000.0, 5000.0),
        account_state=AccountState(realised_pnl_rupees=0.0, margin_used_rupees=10000.0),
        live_vwap=21778.0,
        or_levels={15: ORLevels(length_minutes=15, high=21848.0, low=21792.0), 30: ORLevels(length_minutes=30, high=21848.0, low=21756.0)},
        lot_size=65,
        previous_option_chain=OptionsChainSnapshot(
            timestamp=bars_5m[-2].timestamp,
            expiry=date(2026, 4, 9),
            spot=spot + 10.0,
            quotes=previous_quotes,
            margin_estimate_per_lot=12000.0,
        ),
        previous_session_close=22020.0,
    )


def test_classify_regime_detects_downtrend() -> None:
    snapshot = _build_snapshot()
    regime = classify_regime(snapshot)
    assert regime.regime.value == "DOWN_TREND"
    assert regime.trend_15m == "TREND_DOWN"
    assert regime.execution_5m == "DOWN_CONFIRMED"
    assert regime.or_length_minutes == 45
    assert regime.metadata["bearish_entry_ready"] is True


def test_classify_regime_detects_bullish_pullback_reclaim() -> None:
    snapshot = _build_bullish_snapshot()
    regime = classify_regime(snapshot)
    assert regime.regime.value == "UP_TREND"
    assert regime.execution_5m == "UP_CONFIRMED"
    assert regime.metadata["bullish_entry_ready"] is True
    assert regime.metadata["bullish_setup"] in {"OPEN_DRIVE_RECLAIM", "OPEN_DRIVE_CONTINUATION"}
    assert regime.metadata["smart_money_bias"] == "BULLISH"
    assert regime.metadata["bullish_bos"] is True
    assert regime.metadata["bullish_smc_alignment"] is True
    assert regime.metadata["bullish_candle_pattern"] in {"BULLISH_EXPANSION", "BULLISH_ENGULFING", "BULLISH_REJECTION_WICK", "BULLISH_BREAKDOWN_FAILURE", "BULLISH_MORNING_STAR", "BULLISH_HAMMER", "BULLISH_INSIDE_BAR_FAILURE", "NONE"}
    assert regime.metadata["bullish_confluence_score"] > 0.0
    assert regime.metadata["trade_plan"]["structure_signal"] == "BULLISH_BOS"
    assert regime.metadata["trade_plan"]["execution_plan"] == "SELL_BULL_PUT_ON_RECLAIM"
    assert regime.metadata["trade_plan"]["candle_pattern"] == regime.metadata["bullish_candle_pattern"]
    assert regime.metadata["trade_plan"]["confluence_score"] == regime.metadata["bullish_confluence_score"]
    assert regime.metadata["trade_plan"]["fvg_context"] in {"BULLISH_FVG_ACTIVE", "NONE"}


def test_classify_regime_detects_gap_up_bullish_continuation() -> None:
    snapshot = _build_gap_up_bullish_snapshot()
    regime = classify_regime(snapshot)
    assert regime.regime == RegimeLabel.UP_TREND
    assert regime.metadata["day_archetype"] == "GAP_UP_CONTINUATION"
    assert regime.metadata["playbook"] == "GAP_UP_BULLISH_CONTINUATION"
    assert regime.metadata["bullish_setup"] == "GAP_CONTINUATION"
    assert regime.metadata["gap_bullish_ready"] is True


def test_classify_regime_detects_gap_down_bearish_continuation() -> None:
    snapshot = _build_gap_down_bearish_snapshot()
    regime = classify_regime(snapshot)
    assert regime.regime == RegimeLabel.DOWN_TREND
    assert regime.metadata["day_archetype"] == "GAP_DOWN_CONTINUATION"
    assert regime.metadata["playbook"] == "GAP_DOWN_BEARISH_CONTINUATION"
    assert regime.metadata["bearish_setup"] == "GAP_CONTINUATION"
    assert regime.metadata["gap_bearish_ready"] is True
    assert regime.metadata["bearish_bos"] is True
    assert regime.metadata["bearish_candle_pattern"] in {"BEARISH_EXPANSION", "BEARISH_ENGULFING", "BEARISH_REJECTION_WICK", "BEARISH_BREAKOUT_FAILURE", "BEARISH_EVENING_STAR", "BEARISH_SHOOTING_STAR", "BEARISH_INSIDE_BAR_FAILURE", "NONE"}
    assert regime.metadata["bearish_confluence_score"] > 0.0
    assert regime.metadata["trade_plan"]["execution_plan"] == "SELL_BEAR_CALL_ON_REJECTION"
    assert regime.metadata["trade_plan"]["scenario"] == "GAP_DOWN_CONTINUATION"
    assert regime.metadata["trade_plan"]["candle_pattern"] == regime.metadata["bearish_candle_pattern"]
    assert regime.metadata["trade_plan"]["confluence_score"] == regime.metadata["bearish_confluence_score"]
    assert regime.metadata["trade_plan"]["order_block_context"] in {"BEARISH_ORDER_BLOCK_ACTIVE", "NONE"}


def test_option_chain_oi_flow_detects_bullish_support_build() -> None:
    snapshot = _build_bullish_snapshot()
    flow = option_chain_oi_flow(
        snapshot.option_chain.spot,
        snapshot.option_chain.quotes,
        snapshot.previous_option_chain.quotes if snapshot.previous_option_chain else None,
    )
    assert flow["smart_money_bias"] == "BULLISH"
    assert flow["bullish_flow_score"] > flow["bearish_flow_score"]
    assert flow["put_support_strike"] is not None


def test_option_chain_wall_migration_detects_bullish_shift() -> None:
    spot = 22050.0
    previous_quotes = [
        OptionsContractQuote(strike=22100.0, option_type=OptionType.CALL, bid=20.0, ask=21.0, ltp=20.5, delta=0.18, oi=300000),
        OptionsContractQuote(strike=22200.0, option_type=OptionType.CALL, bid=9.0, ask=10.0, ltp=9.5, delta=0.09, oi=180000),
        OptionsContractQuote(strike=21900.0, option_type=OptionType.PUT, bid=18.0, ask=19.0, ltp=18.5, delta=-0.18, oi=260000),
        OptionsContractQuote(strike=21800.0, option_type=OptionType.PUT, bid=8.0, ask=9.0, ltp=8.5, delta=-0.09, oi=150000),
    ]
    current_quotes = [
        OptionsContractQuote(strike=22200.0, option_type=OptionType.CALL, bid=18.0, ask=19.0, ltp=18.5, delta=0.18, oi=320000),
        OptionsContractQuote(strike=22300.0, option_type=OptionType.CALL, bid=8.0, ask=9.0, ltp=8.5, delta=0.09, oi=160000),
        OptionsContractQuote(strike=22000.0, option_type=OptionType.PUT, bid=20.0, ask=21.0, ltp=20.5, delta=-0.18, oi=310000),
        OptionsContractQuote(strike=21900.0, option_type=OptionType.PUT, bid=9.0, ask=10.0, ltp=9.5, delta=-0.09, oi=170000),
    ]
    migration = option_chain_wall_migration(
        spot,
        current_quotes,
        previous_quotes,
    )
    assert migration["wall_migration_bias"] == "BULLISH"
    assert migration["put_wall_shift"] >= 0


def test_bearish_candle_context_detects_breakout_failure() -> None:
    start = datetime(2026, 4, 3, 10, 0)
    bars = [
        OhlcvBar(timestamp=start, open=100.0, high=101.0, low=99.8, close=100.8, volume=1000),
        OhlcvBar(timestamp=start + timedelta(minutes=5), open=100.8, high=101.4, low=100.5, close=101.2, volume=1000),
        OhlcvBar(timestamp=start + timedelta(minutes=10), open=101.3, high=102.2, low=100.4, close=100.6, volume=1000),
    ]
    pattern = bearish_candle_context(bars)
    assert pattern["pattern"] == "BEARISH_BREAKOUT_FAILURE"
    assert float(pattern["quality_score"]) > 0.0


def test_bullish_candle_context_detects_breakdown_failure() -> None:
    start = datetime(2026, 4, 3, 10, 0)
    bars = [
        OhlcvBar(timestamp=start, open=100.0, high=100.2, low=99.0, close=99.2, volume=1000),
        OhlcvBar(timestamp=start + timedelta(minutes=5), open=99.2, high=99.5, low=98.8, close=99.0, volume=1000),
        OhlcvBar(timestamp=start + timedelta(minutes=10), open=98.9, high=100.1, low=98.2, close=99.9, volume=1000),
    ]
    pattern = bullish_candle_context(bars)
    assert pattern["pattern"] == "BULLISH_BREAKDOWN_FAILURE"
    assert float(pattern["quality_score"]) > 0.0


def test_select_structure_prefers_delta_based_bear_call_spread() -> None:
    snapshot = _build_snapshot()
    regime = classify_regime(snapshot)
    structure, reasons = select_structure(
        StrategyType.BEAR_CALL_CREDIT_SPREAD,
        snapshot,
        regime,
        AdaptiveParameters(directional_credit_width_ratio=0.25),
    )
    assert structure is not None, reasons
    assert structure.strategy == StrategyType.BEAR_CALL_CREDIT_SPREAD
    assert [leg.strike for leg in structure.legs] == [snapshot.option_chain.spot + 300, snapshot.option_chain.spot + 400]
    assert structure.metadata["selection_mode"] == "DELTA"
    assert structure.credit_points > 0


def test_bear_call_uses_structure_distance_fallback_when_delta_bands_do_not_match() -> None:
    snapshot = _build_snapshot()
    mismatched_quotes = []
    for quote in snapshot.option_chain.quotes:
        if quote.option_type == OptionType.CALL and quote.delta is not None:
            delta = 0.52 if quote.strike == snapshot.option_chain.spot + 300 else quote.delta
            if quote.strike == snapshot.option_chain.spot + 400:
                delta = 0.24
            mismatched_quotes.append(
                OptionsContractQuote(
                    strike=quote.strike,
                    option_type=quote.option_type,
                    bid=quote.bid,
                    ask=quote.ask,
                    ltp=quote.ltp,
                    delta=delta,
                )
            )
        else:
            mismatched_quotes.append(quote)
    snapshot = MarketSnapshot(
        nifty_5m=snapshot.nifty_5m,
        nifty_15m=snapshot.nifty_15m,
        option_chain=OptionsChainSnapshot(
            timestamp=snapshot.option_chain.timestamp,
            expiry=snapshot.option_chain.expiry,
            spot=snapshot.option_chain.spot,
            quotes=mismatched_quotes,
            margin_estimate_per_lot=snapshot.option_chain.margin_estimate_per_lot,
        ),
        risk_limits=snapshot.risk_limits,
        account_state=snapshot.account_state,
        live_vwap=snapshot.live_vwap,
        or_levels=snapshot.or_levels,
        lot_size=snapshot.lot_size,
    )
    regime = classify_regime(snapshot)
    structure, reasons = select_structure(
        StrategyType.BEAR_CALL_CREDIT_SPREAD,
        snapshot,
        regime,
        AdaptiveParameters(directional_credit_width_ratio=0.25),
    )
    assert structure is not None, reasons
    assert structure.metadata["selection_mode"] == "STRUCTURE_DISTANCE_FALLBACK"


def test_compute_max_loss_rupees_per_lot_for_vertical() -> None:
    snapshot = _build_snapshot()
    regime = classify_regime(snapshot)
    structure, reasons = select_structure(
        StrategyType.BEAR_CALL_CREDIT_SPREAD,
        snapshot,
        regime,
        AdaptiveParameters(directional_credit_width_ratio=0.25),
    )
    assert structure is not None, reasons
    max_loss = compute_max_loss_rupees_per_lot(structure, lot_size=snapshot.lot_size)
    expected = (structure.width_points - structure.credit_points) * snapshot.lot_size
    assert max_loss == expected


def test_kill_switch_and_risk_assessment_enforce_caps() -> None:
    snapshot = _build_snapshot()
    regime = classify_regime(snapshot)
    structure, reasons = select_structure(
        StrategyType.BEAR_CALL_CREDIT_SPREAD,
        snapshot,
        regime,
        AdaptiveParameters(directional_credit_width_ratio=0.25),
    )
    assert structure is not None, reasons

    triggered, trigger_reasons = kill_switch_triggered(
        AccountState(realised_pnl_rupees=-6000.0, margin_used_rupees=95000.0),
        snapshot.risk_limits,
    )
    assert triggered is True
    assert len(trigger_reasons) == 2

    risk = assess_trade_risk(
        structure=structure,
        lot_size=snapshot.lot_size,
        risk_limits=snapshot.risk_limits,
        account_state=AccountState(realised_pnl_rupees=0.0, margin_used_rupees=95000.0),
        margin_estimate_per_lot=structure.margin_estimate_per_lot,
    )
    assert risk.allowed is False
    assert risk.lots == 0


def test_defined_risk_structure_falls_back_to_max_loss_margin_proxy() -> None:
    snapshot = _build_snapshot()
    snapshot = MarketSnapshot(
        nifty_5m=snapshot.nifty_5m,
        nifty_15m=snapshot.nifty_15m,
        option_chain=OptionsChainSnapshot(
            timestamp=snapshot.option_chain.timestamp,
            expiry=snapshot.option_chain.expiry,
            spot=snapshot.option_chain.spot,
            quotes=snapshot.option_chain.quotes,
            margin_estimate_per_lot=None,
        ),
        risk_limits=snapshot.risk_limits,
        account_state=snapshot.account_state,
        live_vwap=snapshot.live_vwap,
        or_levels=snapshot.or_levels,
        lot_size=snapshot.lot_size,
    )
    regime = classify_regime(snapshot)
    structure, reasons = select_structure(
        StrategyType.BEAR_CALL_CREDIT_SPREAD,
        snapshot,
        regime,
        AdaptiveParameters(directional_credit_width_ratio=0.25),
    )
    assert structure is not None, reasons
    assert structure.margin_estimate_per_lot is not None
    assert structure.margin_estimate_per_lot > 0
    assert structure.metadata["margin_source"] == "MAX_LOSS_PROXY"


def test_opening_range_whipsaw_forces_no_trade() -> None:
    start = datetime(2026, 4, 3, 9, 15)
    bars_5m = [
        OhlcvBar(timestamp=start + timedelta(minutes=5 * idx), open=o, high=h, low=l, close=c, volume=1500 + idx)
        for idx, (o, h, l, c) in enumerate(
            [
                (22000.0, 22020.0, 21990.0, 22010.0),
                (22010.0, 22070.0, 22000.0, 22065.0),
                (22065.0, 22080.0, 21940.0, 21950.0),
                (21950.0, 22030.0, 21920.0, 22010.0),
                (22010.0, 22040.0, 21980.0, 22020.0),
                (22020.0, 22050.0, 22000.0, 22030.0),
                (22030.0, 22060.0, 22010.0, 22035.0),
                (22035.0, 22070.0, 22020.0, 22040.0),
                (22040.0, 22080.0, 22030.0, 22055.0),
                (22055.0, 22090.0, 22040.0, 22075.0),
            ]
        )
    ]
    chain = _build_option_chain(spot=22075.0)
    chain = OptionsChainSnapshot(
        timestamp=datetime(2026, 4, 3, 10, 5),
        expiry=chain.expiry,
        spot=chain.spot,
        quotes=chain.quotes,
        margin_estimate_per_lot=chain.margin_estimate_per_lot,
    )
    snapshot = MarketSnapshot(
        nifty_5m=OhlcvSeries(timeframe_minutes=5, bars=bars_5m),
        nifty_15m=OhlcvSeries(timeframe_minutes=15, bars=_build_15m_downtrend_bars()),
        option_chain=chain,
        risk_limits=AccountRiskLimits(
            max_risk_rupees_per_trade=10000.0,
            max_margin_rupees=100000.0,
            max_daily_loss_rupees=5000.0,
        ),
        account_state=AccountState(realised_pnl_rupees=0.0, margin_used_rupees=0.0),
        live_vwap=22010.0,
        or_levels={
            15: ORLevels(length_minutes=15, high=22060.0, low=21990.0),
            30: ORLevels(length_minutes=30, high=22060.0, low=21990.0),
            45: ORLevels(length_minutes=45, high=22060.0, low=21990.0),
        },
        lot_size=65,
    )
    regime = classify_regime(snapshot)
    assert regime.regime.value == "NO_TRADE"
    assert "event risk" in " ".join(regime.reasons)


def test_bear_calls_require_late_session_timing() -> None:
    original_flag = strategy_module.ENABLE_BEAR_CALL_ACTIVE
    original_sideways_flag = strategy_module.ENABLE_SIDEWAYS_BEARISH_REJECTION_ACTIVE
    strategy_module.ENABLE_BEAR_CALL_ACTIVE = False
    strategy_module.ENABLE_SIDEWAYS_BEARISH_REJECTION_ACTIVE = True
    try:
        regime = RegimeState(
            regime=RegimeLabel.DOWN_TREND,
            trend_15m="TREND_DOWN",
            execution_5m="DOWN_CONFIRMED",
            ema20_15m=22000.0,
            ema20_slope_15m=-5.0,
            rv30_pct=0.42,
            or_length_minutes=30,
            opening_range_high=22050.0,
            opening_range_low=21950.0,
            vwap=22010.0,
            confidence=0.90,
            reasons=["Bearish regime detected."],
            metadata={
                "day_archetype": "SIDEWAYS_TO_BEARISH",
                "bearish_entry_ready": True,
                "bearish_setup": "PULLBACK_REJECTION",
                "playbook": "SIDEWAYS_TO_BEARISH_REJECTION",
            },
        )
        strategy, reasons = strategy_module.select_strategy(regime, datetime(2026, 4, 3, 10, 0).time())
        assert strategy == StrategyType.NO_TRADE
        assert any("require time >= 11:30 IST" in reason for reason in reasons)
    finally:
        strategy_module.ENABLE_BEAR_CALL_ACTIVE = original_flag
        strategy_module.ENABLE_SIDEWAYS_BEARISH_REJECTION_ACTIVE = original_sideways_flag


def test_early_balance_bearish_failed_reclaim_requires_1010_start() -> None:
    original_flag = strategy_module.ENABLE_EARLY_BALANCE_BEARISH_ACTIVE
    try:
        strategy_module.ENABLE_EARLY_BALANCE_BEARISH_ACTIVE = True
        regime = RegimeState(
            regime=RegimeLabel.DOWN_TREND,
            trend_15m="TREND_DOWN",
            execution_5m="DOWN_CONFIRMED",
            ema20_15m=22000.0,
            ema20_slope_15m=-5.0,
            rv30_pct=0.28,
            or_length_minutes=30,
            opening_range_high=22050.0,
            opening_range_low=21950.0,
            vwap=22010.0,
            confidence=0.90,
            reasons=["Bearish regime detected."],
            metadata={"day_archetype": "EARLY_BALANCE_TO_BEARISH", "playbook": "EARLY_BALANCE_BEARISH_FAILED_RECLAIM", "bearish_entry_ready": True, "bearish_setup": "FAILED_RECLAIM"},
        )
        strategy, reasons = strategy_module.select_strategy(regime, datetime(2026, 4, 3, 10, 0).time())
        assert strategy == StrategyType.NO_TRADE
        assert any("require time >= 10:10 IST" in reason for reason in reasons)
    finally:
        strategy_module.ENABLE_EARLY_BALANCE_BEARISH_ACTIVE = original_flag


def test_early_balance_bearish_failed_reclaim_is_allowed_after_1010() -> None:
    original_flag = strategy_module.ENABLE_EARLY_BALANCE_BEARISH_ACTIVE
    try:
        strategy_module.ENABLE_EARLY_BALANCE_BEARISH_ACTIVE = True
        regime = RegimeState(
            regime=RegimeLabel.DOWN_TREND,
            trend_15m="TREND_DOWN",
            execution_5m="DOWN_CONFIRMED",
            ema20_15m=22000.0,
            ema20_slope_15m=-5.0,
            rv30_pct=0.28,
            or_length_minutes=30,
            opening_range_high=22050.0,
            opening_range_low=21950.0,
            vwap=22010.0,
            confidence=0.90,
            reasons=["Bearish regime detected."],
            metadata={"day_archetype": "EARLY_BALANCE_TO_BEARISH", "playbook": "EARLY_BALANCE_BEARISH_FAILED_RECLAIM", "bearish_entry_ready": True, "bearish_setup": "FAILED_RECLAIM"},
        )
        strategy, reasons = strategy_module.select_strategy(regime, datetime(2026, 4, 3, 10, 15).time())
        assert strategy == StrategyType.BEAR_CALL_CREDIT_SPREAD
        assert any("Bear Call Credit Spread" in reason for reason in reasons)
    finally:
        strategy_module.ENABLE_EARLY_BALANCE_BEARISH_ACTIVE = original_flag


def test_high_confluence_bearish_continuation_is_allowed_after_1000() -> None:
    original_flag = strategy_module.ENABLE_HIGH_CONFLUENCE_BEARISH_ACTIVE
    try:
        strategy_module.ENABLE_HIGH_CONFLUENCE_BEARISH_ACTIVE = True
        regime = RegimeState(
            regime=RegimeLabel.DOWN_TREND,
            trend_15m="NEUTRAL",
            execution_5m="DOWN_CONFIRMED",
            ema20_15m=22000.0,
            ema20_slope_15m=-1.0,
            rv30_pct=0.28,
            or_length_minutes=30,
            opening_range_high=22050.0,
            opening_range_low=21950.0,
            vwap=22010.0,
            confidence=0.92,
            reasons=["Bearish regime detected."],
            metadata={
                "day_archetype": "HIGH_CONFLUENCE_BEARISH",
                "playbook": "HIGH_CONFLUENCE_BEARISH_CONTINUATION",
                "bearish_entry_ready": True,
                "bearish_setup": "HIGH_CONFLUENCE_CONTINUATION",
            },
        )
        strategy, reasons = strategy_module.select_strategy(regime, datetime(2026, 4, 3, 10, 5).time())
        assert strategy == StrategyType.BEAR_CALL_CREDIT_SPREAD
        assert any("Bear Call Credit Spread" in reason for reason in reasons)
    finally:
        strategy_module.ENABLE_HIGH_CONFLUENCE_BEARISH_ACTIVE = original_flag


def test_high_confluence_bullish_continuation_is_allowed_after_1000() -> None:
    original_flag = strategy_module.ENABLE_HIGH_CONFLUENCE_BULLISH_ACTIVE
    try:
        strategy_module.ENABLE_HIGH_CONFLUENCE_BULLISH_ACTIVE = True
        regime = RegimeState(
            regime=RegimeLabel.UP_TREND,
            trend_15m="NEUTRAL",
            execution_5m="UP_CONFIRMED",
            ema20_15m=22000.0,
            ema20_slope_15m=1.0,
            rv30_pct=0.28,
            or_length_minutes=30,
            opening_range_high=22050.0,
            opening_range_low=21950.0,
            vwap=22010.0,
            confidence=0.92,
            reasons=["Bullish regime detected."],
            metadata={
                "day_archetype": "HIGH_CONFLUENCE_BULLISH",
                "playbook": "HIGH_CONFLUENCE_BULLISH_CONTINUATION",
                "bullish_entry_ready": True,
                "bullish_setup": "HIGH_CONFLUENCE_CONTINUATION",
            },
        )
        strategy, reasons = strategy_module.select_strategy(regime, datetime(2026, 4, 3, 10, 5).time())
        assert strategy == StrategyType.BULL_PUT_CREDIT_SPREAD
        assert any("Bull Put Credit Spread" in reason for reason in reasons)
    finally:
        strategy_module.ENABLE_HIGH_CONFLUENCE_BULLISH_ACTIVE = original_flag


def test_bull_put_requires_minimum_conviction() -> None:
    original_flag = strategy_module.ENABLE_OPEN_DRIVE_BULLISH_ACTIVE
    strategy_module.ENABLE_OPEN_DRIVE_BULLISH_ACTIVE = True
    try:
        regime = RegimeState(
            regime=RegimeLabel.UP_TREND,
            trend_15m="TREND_UP",
            execution_5m="UP_CONFIRMED",
            ema20_15m=22000.0,
            ema20_slope_15m=5.0,
            rv30_pct=0.42,
            or_length_minutes=30,
            opening_range_high=22050.0,
            opening_range_low=21950.0,
            vwap=22010.0,
            confidence=0.50,
            reasons=["Bullish regime detected."],
            metadata={"day_archetype": "OPEN_DRIVE_BULLISH", "playbook": "OPEN_DRIVE_BULLISH", "bullish_entry_ready": True, "bullish_setup": "OPEN_DRIVE_RECLAIM"},
        )
        strategy, reasons = strategy_module.select_strategy(regime, datetime(2026, 4, 3, 10, 30).time())
        assert strategy == StrategyType.NO_TRADE
        assert any("Bullish setup detected" in reason for reason in reasons)
    finally:
        strategy_module.ENABLE_OPEN_DRIVE_BULLISH_ACTIVE = original_flag


def test_high_conviction_bull_put_is_allowed() -> None:
    original_flag = strategy_module.ENABLE_OPEN_DRIVE_BULLISH_ACTIVE
    strategy_module.ENABLE_OPEN_DRIVE_BULLISH_ACTIVE = True
    try:
        regime = RegimeState(
            regime=RegimeLabel.UP_TREND,
            trend_15m="TREND_UP",
            execution_5m="UP_CONFIRMED",
            ema20_15m=22000.0,
            ema20_slope_15m=5.0,
            rv30_pct=0.42,
            or_length_minutes=30,
            opening_range_high=22050.0,
            opening_range_low=21950.0,
            vwap=22010.0,
            confidence=0.70,
            reasons=["Bullish regime detected."],
            metadata={"day_archetype": "OPEN_DRIVE_BULLISH", "playbook": "OPEN_DRIVE_BULLISH", "bullish_entry_ready": True, "bullish_setup": "OPEN_DRIVE_RECLAIM"},
        )
        strategy, reasons = strategy_module.select_strategy(regime, datetime(2026, 4, 3, 10, 30).time())
        assert strategy == StrategyType.BULL_PUT_CREDIT_SPREAD
        assert any("Bull Put Credit Spread" in reason for reason in reasons)
    finally:
        strategy_module.ENABLE_OPEN_DRIVE_BULLISH_ACTIVE = original_flag


def test_sideways_bullish_reclaim_requires_later_start() -> None:
    original_flag = strategy_module.ENABLE_SIDEWAYS_BULLISH_RECLAIM_ACTIVE
    strategy_module.ENABLE_SIDEWAYS_BULLISH_RECLAIM_ACTIVE = True
    try:
        regime = RegimeState(
            regime=RegimeLabel.UP_TREND,
            trend_15m="TREND_UP",
            execution_5m="UP_CONFIRMED",
            ema20_15m=22000.0,
            ema20_slope_15m=5.0,
            rv30_pct=0.28,
            or_length_minutes=30,
            opening_range_high=22050.0,
            opening_range_low=21950.0,
            vwap=22010.0,
            confidence=0.70,
            reasons=["Bullish regime detected."],
            metadata={"day_archetype": "SIDEWAYS_TO_BULLISH", "playbook": "SIDEWAYS_TO_BULLISH_RECLAIM", "bullish_entry_ready": True, "bullish_setup": "PULLBACK_RECLAIM"},
        )
        strategy, reasons = strategy_module.select_strategy(regime, datetime(2026, 4, 3, 10, 30).time())
        assert strategy == StrategyType.NO_TRADE
        assert any("require time >= 11:30 IST" in reason for reason in reasons)
    finally:
        strategy_module.ENABLE_SIDEWAYS_BULLISH_RECLAIM_ACTIVE = original_flag


def test_early_balance_bullish_reclaim_requires_1010_start() -> None:
    original_flag = strategy_module.ENABLE_EARLY_BALANCE_BULLISH_ACTIVE
    strategy_module.ENABLE_EARLY_BALANCE_BULLISH_ACTIVE = True
    try:
        regime = RegimeState(
            regime=RegimeLabel.UP_TREND,
            trend_15m="TREND_UP",
            execution_5m="UP_CONFIRMED",
            ema20_15m=22000.0,
            ema20_slope_15m=5.0,
            rv30_pct=0.28,
            or_length_minutes=30,
            opening_range_high=22050.0,
            opening_range_low=21950.0,
            vwap=22010.0,
            confidence=0.80,
            reasons=["Bullish regime detected."],
            metadata={"day_archetype": "SIDEWAYS_TO_BULLISH", "playbook": "EARLY_BALANCE_BULLISH_RECLAIM", "bullish_entry_ready": True, "bullish_setup": "EARLY_BALANCE_RECLAIM"},
        )
        strategy, reasons = strategy_module.select_strategy(regime, datetime(2026, 4, 3, 10, 0).time())
        assert strategy == StrategyType.NO_TRADE
        assert any("require time >= 10:10 IST" in reason for reason in reasons)
    finally:
        strategy_module.ENABLE_EARLY_BALANCE_BULLISH_ACTIVE = original_flag


def test_early_balance_bullish_reclaim_is_allowed_after_1010() -> None:
    original_flag = strategy_module.ENABLE_EARLY_BALANCE_BULLISH_ACTIVE
    strategy_module.ENABLE_EARLY_BALANCE_BULLISH_ACTIVE = True
    try:
        regime = RegimeState(
            regime=RegimeLabel.UP_TREND,
            trend_15m="TREND_UP",
            execution_5m="UP_CONFIRMED",
            ema20_15m=22000.0,
            ema20_slope_15m=5.0,
            rv30_pct=0.28,
            or_length_minutes=30,
            opening_range_high=22050.0,
            opening_range_low=21950.0,
            vwap=22010.0,
            confidence=0.80,
            reasons=["Bullish regime detected."],
            metadata={"day_archetype": "SIDEWAYS_TO_BULLISH", "playbook": "EARLY_BALANCE_BULLISH_RECLAIM", "bullish_entry_ready": True, "bullish_setup": "EARLY_BALANCE_RECLAIM"},
        )
        strategy, reasons = strategy_module.select_strategy(regime, datetime(2026, 4, 3, 10, 15).time())
        assert strategy == StrategyType.BULL_PUT_CREDIT_SPREAD
        assert any("Bull Put Credit Spread" in reason for reason in reasons)
    finally:
        strategy_module.ENABLE_EARLY_BALANCE_BULLISH_ACTIVE = original_flag


def test_afternoon_trend_hold_bullish_requires_afternoon_start() -> None:
    original_flag = strategy_module.ENABLE_AFTERNOON_TREND_BULLISH_ACTIVE
    strategy_module.ENABLE_AFTERNOON_TREND_BULLISH_ACTIVE = True
    try:
        regime = RegimeState(
            regime=RegimeLabel.UP_TREND,
            trend_15m="TREND_UP",
            execution_5m="UP_CONFIRMED",
            ema20_15m=22000.0,
            ema20_slope_15m=5.0,
            rv30_pct=0.24,
            or_length_minutes=30,
            opening_range_high=22050.0,
            opening_range_low=21950.0,
            vwap=22010.0,
            confidence=0.80,
            reasons=["Bullish regime detected."],
            metadata={"day_archetype": "TREND_BULLISH", "playbook": "AFTERNOON_TREND_HOLD_BULLISH", "bullish_entry_ready": True, "bullish_setup": "AFTERNOON_TREND_HOLD"},
        )
        strategy, reasons = strategy_module.select_strategy(regime, datetime(2026, 4, 3, 12, 30).time())
        assert strategy == StrategyType.NO_TRADE
        assert any("require time >= 13:00 IST" in reason for reason in reasons)
    finally:
        strategy_module.ENABLE_AFTERNOON_TREND_BULLISH_ACTIVE = original_flag


def test_afternoon_trend_hold_bullish_is_allowed_after_1300() -> None:
    original_flag = strategy_module.ENABLE_AFTERNOON_TREND_BULLISH_ACTIVE
    strategy_module.ENABLE_AFTERNOON_TREND_BULLISH_ACTIVE = True
    try:
        regime = RegimeState(
            regime=RegimeLabel.UP_TREND,
            trend_15m="TREND_UP",
            execution_5m="UP_CONFIRMED",
            ema20_15m=22000.0,
            ema20_slope_15m=5.0,
            rv30_pct=0.24,
            or_length_minutes=30,
            opening_range_high=22050.0,
            opening_range_low=21950.0,
            vwap=22010.0,
            confidence=0.80,
            reasons=["Bullish regime detected."],
            metadata={"day_archetype": "TREND_BULLISH", "playbook": "AFTERNOON_TREND_HOLD_BULLISH", "bullish_entry_ready": True, "bullish_setup": "AFTERNOON_TREND_HOLD"},
        )
        strategy, reasons = strategy_module.select_strategy(regime, datetime(2026, 4, 3, 13, 10).time())
        assert strategy == StrategyType.BULL_PUT_CREDIT_SPREAD
        assert any("Bull Put Credit Spread" in reason for reason in reasons)
    finally:
        strategy_module.ENABLE_AFTERNOON_TREND_BULLISH_ACTIVE = original_flag


def test_range_balanced_condor_requires_entry_window() -> None:
    regime = RegimeState(
        regime=RegimeLabel.RANGE,
        trend_15m="NEUTRAL",
        execution_5m="RANGE_CONFIRMED",
        ema20_15m=22000.0,
        ema20_slope_15m=0.0,
        rv30_pct=0.18,
        or_length_minutes=15,
        opening_range_high=22050.0,
        opening_range_low=21950.0,
        vwap=22000.0,
        confidence=0.68,
        reasons=["Range regime detected."],
        metadata={"playbook": "RANGE_BALANCED_CONDOR", "range_entry_ready": True, "range_balance_score": 3.5},
    )
    strategy, reasons = strategy_module.select_strategy(regime, datetime(2026, 4, 3, 11, 0).time())
    assert strategy == StrategyType.NO_TRADE
    assert any("allowed only after 11:30 IST" in reason for reason in reasons)
