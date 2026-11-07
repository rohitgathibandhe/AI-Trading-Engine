from market_ai.intraday_defined_risk.data_models import OptionsContractQuote, OptionType
from market_ai.intraday_defined_risk.replay_provider import _shift_quote


def _q(strike, ot, px, delta):
    return OptionsContractQuote(strike=strike, option_type=ot, bid=px - 0.05, ask=px + 0.05, ltp=px, delta=delta)


def test_put_gains_when_spot_falls():
    # 2025-05-08: chain at 24,318 (24350P 34.8, delta -0.82); closed-candle spot 24,287 -> intrinsic 63 floor (real 15:05 chain: 64.4)
    q = _shift_quote(_q(24350, OptionType.PUT, 34.8, -0.82), 24287 - 24318, 24287)
    assert 62.9 < q.ltp < 64.5 and q.bid <= q.ask


def test_never_below_intrinsic_or_tick():
    q = _shift_quote(_q(24350, OptionType.PUT, 5.0, -0.2), 300, 24650)
    assert q.ltp == 0.05
    q = _shift_quote(_q(24000, OptionType.CALL, 10.0, 0.1), 200, 24200)
    assert q.ltp >= 200


def test_no_delta_or_no_move_is_unchanged():
    q = _q(24000, OptionType.CALL, 10.0, None)
    assert _shift_quote(q, 50, 24050) is q
    q2 = _q(24000, OptionType.CALL, 10.0, 0.5)
    assert _shift_quote(q2, 0, 24000) is q2
