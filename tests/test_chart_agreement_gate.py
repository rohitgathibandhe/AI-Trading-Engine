from market_ai.intraday_defined_risk.decision_justification import (
    DimensionRead, TradeThesis, build_thesis, chart_opposes,
)


def _thesis(chart_bias: str) -> TradeThesis:
    return TradeThesis(spot=22500.0, chart=DimensionRead("CHART", chart_bias, []))


def test_bearish_structure_against_bullish_chart_is_blocked():
    assert chart_opposes("PUT_DEBIT_SPREAD", _thesis("BULLISH"))
    assert chart_opposes("BEAR_CALL_CREDIT_SPREAD", _thesis("BULLISH"))


def test_bullish_structure_against_bearish_chart_is_blocked():
    assert chart_opposes("BULL_PUT_CREDIT_SPREAD", _thesis("BEARISH"))
    assert chart_opposes("CALL_DEBIT_SPREAD", _thesis("BEARISH"))


def test_aligned_or_neutral_chart_passes():
    assert not chart_opposes("PUT_DEBIT_SPREAD", _thesis("BEARISH"))
    assert not chart_opposes("PUT_DEBIT_SPREAD", _thesis("NEUTRAL"))
    assert not chart_opposes("BULL_PUT_CREDIT_SPREAD", _thesis("BULLISH"))


def test_non_directional_or_missing_thesis_never_blocks():
    assert not chart_opposes("IRON_CONDOR", _thesis("BULLISH"))
    assert not chart_opposes("PUT_DEBIT_SPREAD", None)


def test_2026_09_29_entry_would_be_blocked():
    # The -3,146 put-debit: price +15 above VWAP + BULLISH_EXPANSION candle -> chart BULLISH.
    th = build_thesis({"price_vs_vwap": 15.0, "bullish_candle_pattern": "BULLISH_EXPANSION"}, 22621.3)
    assert th.chart.bias == "BULLISH"
    assert chart_opposes("PUT_DEBIT_SPREAD", th)
