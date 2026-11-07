from market_ai.intraday_defined_risk import strategy_selector as ss


def _choice(struct="CALL_DEBIT_SPREAD"):
    return ss.StrategyChoice(condition="BREAKOUT_UP", family=ss.FAM_DIRECTIONAL_DEBIT, structures=[struct],
                             rationale="", executable_today=True)


def _meta(close, thesis="TREND_UP"):
    return {"day_thesis_type": thesis,
            "trend_day": {"swings": {"day_high": 24700.0, "day_low": 24500.0}, "last_bar": {"o": 0, "h": 0, "l": 0, "c": close}}}


def test_filters_off_by_default(monkeypatch):
    monkeypatch.delenv("SEL_NO_CHASE", raising=False)
    monkeypatch.delenv("SEL_THESIS_RANGE_BLOCK", raising=False)
    assert ss._entry_filters(_choice(), _meta(24695, "RANGE_PREMIUM")).structures == ["CALL_DEBIT_SPREAD"]


def test_no_chase_blocks_a_call_at_the_high_not_a_put(monkeypatch):
    monkeypatch.setenv("SEL_NO_CHASE", "0.85")
    m = _meta(24690)
    assert ss._entry_filters(_choice(), m).structures == [] and "chasing" in m["entry_filter_skip"]
    assert ss._entry_filters(_choice("PUT_DEBIT_SPREAD"), _meta(24690)).structures == ["PUT_DEBIT_SPREAD"]
    assert ss._entry_filters(_choice("PUT_DEBIT_SPREAD"), _meta(24510)).structures == []
    assert ss._entry_filters(_choice(), _meta(24600)).structures == ["CALL_DEBIT_SPREAD"]


def test_thesis_block_only_named_types(monkeypatch):
    monkeypatch.setenv("SEL_THESIS_RANGE_BLOCK", "RANGE_PREMIUM")
    assert ss._entry_filters(_choice(), _meta(24600, "RANGE_PREMIUM")).structures == []
    assert ss._entry_filters(_choice(), _meta(24600, "UNCERTAIN")).structures == ["CALL_DEBIT_SPREAD"]
