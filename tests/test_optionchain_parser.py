import json
from pathlib import Path

import pytest

from market_ai.modules.strategies.monthly_strangle_with_weekly_hedge import (
    _infer_tick,
    _collect_option_chain_sources,
    _extract_leg_from_chain_sources,
    _safe_delta,
    _safe_ltp,
)

pytestmark = pytest.mark.unit


def test_safe_ltp_handles_string_prices():
    opt = {"last_price": "12.35"}
    assert _safe_ltp(opt) == pytest.approx(12.35)


def test_safe_delta_prefers_nested_greeks():
    opt = {"greeks": {"delta": "-0.18"}}
    assert _safe_delta(opt) == pytest.approx(-0.18)


def test_infer_tick_detects_50pt_spacing():
    strikes = [24800.0, 24850.0, 24900.0]
    assert _infer_tick(strikes, default=25.0) == pytest.approx(50.0)


def test_chain_lookup_handles_decimal_keys(monkeypatch):
    payload = json.loads(Path("data_engine/market_ai/state/option_chain_sample.json").read_text())
    data = payload["data"]["data"]
    sources = _collect_option_chain_sources(data)

    captured = {}

    def fake_resolver(symbol, expiry, strike, opt_type):
        captured["args"] = (symbol, expiry, strike, opt_type)
        return 424242 if opt_type == "CALL" else 434343

    monkeypatch.setattr(
        "market_ai.modules.strategies.monthly_strangle_with_weekly_hedge.resolve_option_security_id",
        fake_resolver,
    )

    strike = 19200
    result = _extract_leg_from_chain_sources(sources, strike, "CALL", "NIFTY", "2025-11-25")
    assert result == (424242, 0.0)
    assert captured["args"] == ("NIFTY", "2025-11-25", float(strike), "CALL")
