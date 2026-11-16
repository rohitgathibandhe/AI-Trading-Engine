from market_ai.intraday_defined_risk.ops_runtime import RuntimeRiskGovernance


def test_one_loss_stop_env_overrides_file(monkeypatch):
    monkeypatch.setenv("ONE_LOSS_STOP", "1")
    assert RuntimeRiskGovernance.from_payload({"second_trade_requires_first_trade_profit": False}).second_trade_requires_first_trade_profit


def test_file_value_kept_without_env(monkeypatch):
    monkeypatch.delenv("ONE_LOSS_STOP", raising=False)
    assert not RuntimeRiskGovernance.from_payload({"second_trade_requires_first_trade_profit": False}).second_trade_requires_first_trade_profit
