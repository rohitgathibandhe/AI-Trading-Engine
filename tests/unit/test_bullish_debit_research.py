from market_ai.intraday_defined_risk.bullish_debit_research import (
    _expansion_recommendation,
    _expansion_rejection_reason,
)


def test_expansion_rejection_reason_orders_constraints_by_severity() -> None:
    reason = _expansion_rejection_reason(
        passed_liquidity=True,
        passed_delta=True,
        passed_debit=False,
        passed_reward_risk=False,
        passed_width_vs_move=False,
        passed_time=False,
        passed_extension=False,
        passed_vwap_distance=False,
        passed_resistance=False,
    )
    assert reason == "DEBIT_TOO_HIGH"


def test_expansion_recommendation_requires_positive_clean_sample() -> None:
    report = {
        "by_playbook": {
            "OPEN_DRIVE_BULLISH": {
                "executed_trades": 3,
                "realized_pnl_rupees": 4500.0,
                "losses": 0,
            },
            "GAP_UP_BULLISH_CONTINUATION": {
                "executed_trades": 2,
                "realized_pnl_rupees": 900.0,
                "losses": 0,
            },
            "HIGH_CONFLUENCE_BULLISH_CONTINUATION": {
                "executed_trades": 0,
                "realized_pnl_rupees": 0.0,
                "losses": 0,
            },
        }
    }
    assert _expansion_recommendation(report) == "promote_to_tier_a_trial:OPEN_DRIVE_BULLISH"
