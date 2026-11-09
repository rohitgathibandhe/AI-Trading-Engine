from market_ai.intraday_defined_risk import strategy_selector as ss


def _choice(struct="CALL_DEBIT_SPREAD"):
    return ss.StrategyChoice(family="DIRECTIONAL_DEBIT", structures=[struct], condition="BREAKOUT_UP", rationale="x")


def test_blocks_debit_when_candles_overlap(monkeypatch):
    monkeypatch.setenv("SEL_MAX_OVERLAP", "0.55")
    md = {"candle_overlap_ratio": 0.58}
    c = ss._chop_overlap_gate(_choice("PUT_DEBIT_SPREAD"), md)
    assert c.family == ss.FAM_STAND_ASIDE and c.structures == [] and md["chop_overlap_skip"] == 0.58


def test_clean_candles_pass(monkeypatch):
    monkeypatch.setenv("SEL_MAX_OVERLAP", "0.55")
    assert ss._chop_overlap_gate(_choice(), {"candle_overlap_ratio": 0.48}).structures == ["CALL_DEBIT_SPREAD"]


def test_off_by_default_and_other_structures_untouched(monkeypatch):
    monkeypatch.delenv("SEL_MAX_OVERLAP", raising=False)
    assert ss._chop_overlap_gate(_choice(), {"candle_overlap_ratio": 0.9}).structures == ["CALL_DEBIT_SPREAD"]
    monkeypatch.setenv("SEL_MAX_OVERLAP", "0.55")
    assert ss._chop_overlap_gate(_choice("BULL_PUT_CREDIT_SPREAD"), {"candle_overlap_ratio": 0.9}).structures
