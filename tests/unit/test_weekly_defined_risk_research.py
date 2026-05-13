from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pandas as pd

from market_ai.utils.nifty_expiry_calendar import infer_nifty_weekly_expiry
from market_ai.weekly_defined_risk.research import (
    StrategyCandidate,
    WeekContext,
    WeeklyResearchConfig,
    _classify_weekly_regime,
    _credit_spread_candidate,
    _load_option_chain,
    _repair_weekly_option_chain_expiry_history,
    generate_weekly_candidates,
    _max_profit_and_loss_rupees,
)


def _context(**overrides) -> WeekContext:
    base = dict(
        week_id="2025-03-17_2025-03-20",
        deployment_date=date(2025, 3, 17),
        deployment_weekday="Monday",
        entry_timestamp=datetime(2025, 3, 17, 9, 45),
        expected_entry_time="09:45",
        next_weekly_expiry=date(2025, 3, 20),
        observed_historical_expiry_weekday="Thursday",
        preferred_expiry_weekday="Tuesday",
        expiry_alignment_ok=False,
        regime="BULLISH_CONTINUATION",
        daily_trend_state="BULLISH",
        trend_60m_state="BULLISH",
        monday_gap_pct=0.001,
        weekly_atr=220.0,
        daily_atr=95.0,
        india_vix=8.0,
        atm_iv=8.0,
        call_wall=25400.0,
        put_wall=24900.0,
        pcr=1.02,
        oi_change_pressure=0.10,
        support_level=24880.0,
        resistance_level=25450.0,
        previous_week_high=25450.0,
        previous_week_low=24880.0,
        previous_week_close=25220.0,
        spot=25200.0,
        weekly_range_position=0.56,
        event_flag=False,
        event_labels=[],
        option_chain_snapshot={"timestamp": "2025-03-17T09:45:00", "expiry": "2025-03-20", "row_count": 8},
        missing_data_flags={},
        trade_ready=True,
        no_trade_reason=None,
        actual_expiry=date(2025, 3, 20),
        preferred_expiry=date(2025, 3, 18),
        metadata={},
    )
    base.update(overrides)
    return WeekContext(**base)


def test_classify_weekly_regime_bearish_gap_up_failure() -> None:
    regime = _classify_weekly_regime(
        monday_gap_pct=0.006,
        daily_trend="BEARISH",
        trend_60m="BEARISH",
        weekly_range_position=0.35,
        atm_iv=14.0,
        call_wall=25300.0,
        put_wall=24800.0,
        pcr=0.88,
        oi_pressure=-0.25,
        support_level=24850.0,
        resistance_level=25240.0,
        spot=25180.0,
        event_flag=False,
    )
    assert regime == "GAP_UP_FAILURE"


def test_max_profit_and_loss_for_credit_spread() -> None:
    legs = [
        {"option_type": "CALL", "strike": 22750.0, "qty": -1},
        {"option_type": "CALL", "strike": 22850.0, "qty": 1},
    ]
    max_profit, max_loss = _max_profit_and_loss_rupees(legs, entry_cashflow_points=35.0, lot_size=75)
    assert round(max_profit, 2) == 2625.0
    assert round(max_loss, 2) == 4875.0


def test_credit_spread_search_uses_later_valid_candidate() -> None:
    config = WeeklyResearchConfig()
    context = _context()
    chain = pd.DataFrame(
        [
            {"timestamp": context.entry_timestamp, "expiry": pd.Timestamp("2025-03-20"), "spot": 25200.0, "strike": 24900.0, "option_type": "PUT", "bid": 52.0, "ask": 54.0, "ltp": 53.0, "delta": -0.22, "iv": 13.5, "oi": 1000, "oi_change": 10.0},
            {"timestamp": context.entry_timestamp, "expiry": pd.Timestamp("2025-03-20"), "spot": 25200.0, "strike": 24800.0, "option_type": "PUT", "bid": 32.0, "ask": 34.0, "ltp": 33.0, "delta": -0.15, "iv": 13.2, "oi": 1000, "oi_change": 10.0},
            {"timestamp": context.entry_timestamp, "expiry": pd.Timestamp("2025-03-20"), "spot": 25200.0, "strike": 25000.0, "option_type": "PUT", "bid": 70.0, "ask": 72.0, "ltp": 71.0, "delta": -0.27, "iv": 13.8, "oi": 1200, "oi_change": 12.0},
            {"timestamp": context.entry_timestamp, "expiry": pd.Timestamp("2025-03-20"), "spot": 25200.0, "strike": 24900.0, "option_type": "PUT", "bid": 52.0, "ask": 54.0, "ltp": 53.0, "delta": -0.22, "iv": 13.5, "oi": 1000, "oi_change": 10.0},
        ]
    )
    candidate = _credit_spread_candidate(
        context=context,
        entry_chain=chain,
        option_type="PUT",
        width_request=100,
        config=config,
    )
    assert isinstance(candidate, StrategyCandidate)
    assert candidate is not None
    assert candidate.is_valid is True
    assert candidate.strategy == "BULL_PUT_CREDIT_SPREAD"
    assert candidate.legs[0]["strike"] == 25000.0
    assert candidate.legs[1]["strike"] == 24900.0


def test_generate_weekly_candidates_respects_strategy_and_regime_filters() -> None:
    config = WeeklyResearchConfig(
        allowed_strategies=("BEAR_CALL_CREDIT_SPREAD",),
        allowed_regimes=("BEARISH_CONTINUATION",),
    )
    context = _context(
        regime="BEARISH_CONTINUATION",
        daily_trend_state="BEARISH",
        trend_60m_state="BEARISH",
        spot=25050.0,
        atm_iv=13.5,
        india_vix=13.5,
        resistance_level=25100.0,
    )
    chain = pd.DataFrame(
        [
            {"timestamp": context.entry_timestamp, "expiry": pd.Timestamp("2025-03-20"), "date": context.deployment_date, "spot": 25050.0, "strike": 25300.0, "option_type": "CALL", "bid": 59.0, "ask": 61.0, "ltp": 60.0, "delta": 0.24, "iv": 13.5, "oi": 1000, "oi_change": 10.0},
            {"timestamp": context.entry_timestamp, "expiry": pd.Timestamp("2025-03-20"), "date": context.deployment_date, "spot": 25050.0, "strike": 25400.0, "option_type": "CALL", "bid": 35.0, "ask": 37.0, "ltp": 36.0, "delta": 0.14, "iv": 13.2, "oi": 1000, "oi_change": 10.0},
            {"timestamp": context.entry_timestamp, "expiry": pd.Timestamp("2025-03-20"), "date": context.deployment_date, "spot": 25050.0, "strike": 24950.0, "option_type": "PUT", "bid": 58.0, "ask": 60.0, "ltp": 59.0, "delta": -0.24, "iv": 13.5, "oi": 1000, "oi_change": 10.0},
            {"timestamp": context.entry_timestamp, "expiry": pd.Timestamp("2025-03-20"), "date": context.deployment_date, "spot": 25050.0, "strike": 24850.0, "option_type": "PUT", "bid": 34.0, "ask": 36.0, "ltp": 35.0, "delta": -0.14, "iv": 13.2, "oi": 1000, "oi_change": 10.0},
        ]
    )
    frames = {"option_chain": chain}
    candidates = generate_weekly_candidates(context, frames, config)
    assert candidates
    assert all(candidate.strategy == "BEAR_CALL_CREDIT_SPREAD" for candidate in candidates)


def test_repair_weekly_option_chain_expiry_history_fixes_post_switch_thursday_rows() -> None:
    option_chain = pd.DataFrame(
        [
            {"timestamp": pd.Timestamp("2025-08-25T09:45:00"), "previous_timestamp": pd.Timestamp("2025-08-25T09:40:00"), "expiry": pd.Timestamp("2025-08-28"), "strike": 24800, "option_type": "PUT"},
            {"timestamp": pd.Timestamp("2025-09-01T09:45:00"), "previous_timestamp": pd.Timestamp("2025-09-01T09:40:00"), "expiry": pd.Timestamp("2025-09-04"), "strike": 24900, "option_type": "PUT"},
            {"timestamp": pd.Timestamp("2025-09-08T09:45:00"), "previous_timestamp": pd.Timestamp("2025-09-08T09:40:00"), "expiry": pd.Timestamp("2025-09-11"), "strike": 25000, "option_type": "CALL"},
        ]
    )
    repaired, report = _repair_weekly_option_chain_expiry_history(
        option_chain,
        trading_days={
            date(2025, 8, 25),
            date(2025, 8, 26),
            date(2025, 8, 27),
            date(2025, 8, 28),
            date(2025, 8, 29),
            date(2025, 9, 1),
            date(2025, 9, 2),
            date(2025, 9, 3),
            date(2025, 9, 4),
            date(2025, 9, 5),
            date(2025, 9, 8),
            date(2025, 9, 9),
        },
    )

    assert repaired.loc[repaired["timestamp"].dt.date == date(2025, 8, 25), "expiry"].iloc[0].date() == date(2025, 8, 28)
    assert repaired.loc[repaired["timestamp"].dt.date == date(2025, 9, 1), "expiry"].iloc[0].date() == date(2025, 9, 2)
    assert repaired.loc[repaired["timestamp"].dt.date == date(2025, 9, 8), "expiry"].iloc[0].date() == date(2025, 9, 9)
    assert report["repair_applied"] is True
    assert report["reason_counts"]["REGIME_MISMATCH"] == 2


def test_load_option_chain_tolerates_missing_previous_timestamp(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    pd.DataFrame(
        [
            {
                "timestamp": "2025-09-01T09:45:00",
                "expiry": "2025-09-02",
                "spot": 24850.0,
                "strike": 24900,
                "option_type": "CALL",
                "bid": 20.0,
                "ask": 22.0,
                "ltp": 21.0,
                "delta": 0.21,
                "iv": 13.0,
                "oi": 1000,
            }
        ]
    ).to_csv(dataset_root / "options_chain.csv", index=False)
    config = WeeklyResearchConfig(dataset_root=dataset_root)
    chain = _load_option_chain(config)
    assert len(chain) == 1
    assert chain.iloc[0]["date"] == date(2025, 9, 1)


def test_infer_nifty_weekly_expiry_shifts_to_previous_trading_day_on_holiday() -> None:
    expiry = infer_nifty_weekly_expiry(
        date(2026, 3, 30),
        trading_days={date(2026, 3, 30), date(2026, 4, 1)},
    )
    assert expiry == date(2026, 3, 30)
