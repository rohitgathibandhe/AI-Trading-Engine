import pytest

from market_ai.modules.strategies.monthly_strangle_with_weekly_hedge import (
    _infer_tick,
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
