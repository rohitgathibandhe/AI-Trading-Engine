import inspect

from market_ai.intraday_defined_risk import monitor


def test_selector_path_exposes_execution_5m_for_the_ph21_gate():
    src = inspect.getsource(monitor.IntradayDefinedRiskAgent._evaluate_via_selector)
    assert 'meta["execution_5m"] = regime_state.execution_5m' in src
