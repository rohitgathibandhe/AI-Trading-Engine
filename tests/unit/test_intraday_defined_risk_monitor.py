from __future__ import annotations

from datetime import datetime

import market_ai.intraday_defined_risk.monitor as monitor_module
from market_ai.intraday_defined_risk.data_models import (
    AccountRiskLimits,
    AccountState,
    DecisionOutput,
    MarketSnapshot,
    OhlcvBar,
    OhlcvSeries,
    OptionType,
    OptionsChainSnapshot,
    OptionsContractQuote,
    RegimeLabel,
    RegimeState,
    StrategyLeg,
    StrategyType,
    TradeStructure,
)
from market_ai.intraday_defined_risk.monitor import IntradayDefinedRiskAgent
from market_ai.intraday_defined_risk.learning import LearningStore


def _snapshot() -> MarketSnapshot:
    ts = datetime(2026, 4, 3, 10, 0)
    bars_5m = [
        OhlcvBar(timestamp=ts.replace(hour=9, minute=15), open=22000.0, high=22020.0, low=21990.0, close=22010.0, volume=1000.0),
        OhlcvBar(timestamp=ts.replace(hour=9, minute=20), open=22010.0, high=22040.0, low=22000.0, close=22030.0, volume=1100.0),
    ]
    bars_15m = [
        OhlcvBar(timestamp=ts.replace(hour=9, minute=15), open=22000.0, high=22040.0, low=21980.0, close=22030.0, volume=2000.0),
    ]
    quotes = [
        OptionsContractQuote(strike=21900.0, option_type=OptionType.PUT, bid=20.0, ask=21.0, ltp=20.5, delta=-0.22),
        OptionsContractQuote(strike=21800.0, option_type=OptionType.PUT, bid=6.0, ask=6.5, ltp=6.25, delta=-0.08),
    ]
    return MarketSnapshot(
        nifty_5m=OhlcvSeries(timeframe_minutes=5, bars=bars_5m),
        nifty_15m=OhlcvSeries(timeframe_minutes=15, bars=bars_15m),
        option_chain=OptionsChainSnapshot(timestamp=ts, expiry=ts.date(), spot=22030.0, quotes=quotes, margin_estimate_per_lot=10000.0),
        risk_limits=AccountRiskLimits(max_risk_rupees_per_trade=10000.0, max_margin_rupees=100000.0, max_daily_loss_rupees=5000.0),
        account_state=AccountState(realised_pnl_rupees=0.0, margin_used_rupees=0.0),
        live_vwap=22010.0,
        lot_size=65,
    )


def test_agent_blocks_third_repeat_strategy_entry_in_same_session(monkeypatch) -> None:
    snapshot = _snapshot()
    agent = IntradayDefinedRiskAgent(
        learning_store=LearningStore("/tmp/test_intraday_defined_risk_monitor.sqlite3"),
        use_regime_tradability_layer=False,
    )
    regime = RegimeState(
        regime=RegimeLabel.UP_TREND,
        trend_15m="TREND_UP",
        execution_5m="UP_CONFIRMED",
        ema20_15m=22010.0,
        ema20_slope_15m=5.0,
        rv30_pct=0.45,
        or_length_minutes=30,
        opening_range_high=22000.0,
        opening_range_low=21950.0,
        vwap=22010.0,
        confidence=0.75,
        reasons=["Bullish setup detected."],
        metadata={"bullish_entry_ready": True, "bullish_setup": "PULLBACK_RECLAIM"},
    )
    structure = TradeStructure(
        strategy=StrategyType.BULL_PUT_CREDIT_SPREAD,
        legs=[
            StrategyLeg(action="SELL", option_type=OptionType.PUT, strike=21900.0, quote=snapshot.option_chain.quotes[0]),
            StrategyLeg(action="BUY", option_type=OptionType.PUT, strike=21800.0, quote=snapshot.option_chain.quotes[1]),
        ],
        credit_points=14.0,
        width_points=100.0,
        put_width_points=100.0,
        margin_estimate_per_lot=10000.0,
    )

    monkeypatch.setattr(monitor_module, "classify_regime", lambda *_args, **_kwargs: regime)
    monkeypatch.setattr(
        monitor_module,
        "select_strategy",
        lambda *_args, **_kwargs: (StrategyType.BULL_PUT_CREDIT_SPREAD, ["Bullish pullback reclaim confirmed."]),
    )
    monkeypatch.setattr(monitor_module, "validate_entry_time", lambda *_args, **_kwargs: (True, None))
    monkeypatch.setattr(
        monitor_module,
        "select_best_structure",
        lambda *_args, **_kwargs: (
            structure,
            ["Structure selected."],
            {
                "passed_spread_construction": True,
                "passed_liquidity": True,
                "passed_credit_width": True,
                "passed_delta_band": True,
                "passed_anchor_distance": True,
                "monetization_score": 1.0,
                "final_trade_score": 1.0,
                "best_candidate": None,
                "best_failed_candidate": None,
                "candidate_evaluations": [],
            },
        ),
    )
    monkeypatch.setattr(
        monitor_module,
        "assess_trade_risk",
        lambda *_args, **_kwargs: type(
            "RiskAssessmentStub",
            (),
            {
                "allowed": True,
                "reasons": [],
                "max_loss_rupees_per_lot": 5590.0,
                "lots": 1,
                "projected_margin_rupees": 10000.0,
                "projected_margin_utilisation": 0.1,
            },
        )(),
    )

    first = agent.evaluate(snapshot)
    assert first.action == "TRADE"
    agent.start_position(snapshot, first)
    agent.open_position = None

    second = agent.evaluate(snapshot)
    assert second.action == "TRADE"
    agent.start_position(snapshot, second)
    agent.open_position = None

    third = agent.evaluate(snapshot)
    assert third.action == "NO_TRADE"
    assert any("already traded twice this session" in reason for reason in third.rationale)


def test_agent_blocks_trade_when_expected_net_edge_is_too_small(monkeypatch) -> None:
    snapshot = _snapshot()
    agent = IntradayDefinedRiskAgent(
        learning_store=LearningStore("/tmp/test_intraday_defined_risk_monitor_netedge.sqlite3"),
        use_regime_tradability_layer=False,
    )
    regime = RegimeState(
        regime=RegimeLabel.UP_TREND,
        trend_15m="TREND_UP",
        execution_5m="UP_CONFIRMED",
        ema20_15m=22010.0,
        ema20_slope_15m=5.0,
        rv30_pct=0.45,
        or_length_minutes=30,
        opening_range_high=22000.0,
        opening_range_low=21950.0,
        vwap=22010.0,
        confidence=0.75,
        reasons=["Bullish setup detected."],
        metadata={"bullish_entry_ready": True, "bullish_setup": "PULLBACK_RECLAIM", "playbook": "BULLISH_CONTINUATION"},
    )
    structure = TradeStructure(
        strategy=StrategyType.BULL_PUT_CREDIT_SPREAD,
        legs=[
            StrategyLeg(action="SELL", option_type=OptionType.PUT, strike=21900.0, quote=snapshot.option_chain.quotes[0]),
            StrategyLeg(action="BUY", option_type=OptionType.PUT, strike=21800.0, quote=snapshot.option_chain.quotes[1]),
        ],
        credit_points=4.0,
        width_points=100.0,
        put_width_points=100.0,
        margin_estimate_per_lot=10000.0,
    )

    monkeypatch.setattr(monitor_module, "classify_regime", lambda *_args, **_kwargs: regime)
    monkeypatch.setattr(
        monitor_module,
        "select_strategy",
        lambda *_args, **_kwargs: (StrategyType.BULL_PUT_CREDIT_SPREAD, ["Bullish pullback reclaim confirmed."]),
    )
    monkeypatch.setattr(monitor_module, "validate_entry_time", lambda *_args, **_kwargs: (True, None))
    monkeypatch.setattr(
        monitor_module,
        "select_best_structure",
        lambda *_args, **_kwargs: (
            structure,
            ["Structure selected."],
            {
                "passed_spread_construction": True,
                "passed_liquidity": True,
                "passed_credit_width": True,
                "passed_delta_band": True,
                "passed_anchor_distance": True,
                "monetization_score": 1.0,
                "final_trade_score": 1.0,
                "best_candidate": None,
                "best_failed_candidate": None,
                "candidate_evaluations": [],
            },
        ),
    )
    monkeypatch.setattr(
        monitor_module,
        "assess_trade_risk",
        lambda *_args, **_kwargs: type(
            "RiskAssessmentStub",
            (),
            {
                "allowed": True,
                "reasons": [],
                "max_loss_rupees_per_lot": 6240.0,
                "lots": 1,
                "projected_margin_rupees": 10000.0,
                "projected_margin_utilisation": 0.1,
            },
        )(),
    )

    decision = agent.evaluate(snapshot)
    assert decision.action == "NO_TRADE"
    assert any("Expected net edge" in reason for reason in decision.rationale)
