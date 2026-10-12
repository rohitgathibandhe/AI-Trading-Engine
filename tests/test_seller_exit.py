from types import SimpleNamespace as NS

from market_ai.intraday_defined_risk.data_models import OptionType
from market_ai.intraday_defined_risk.execution import _evaluate_seller_exit


def _pos(credit=20.0, short=22600.0):
    legs = [NS(action="SELL", option_type=OptionType.CALL, strike=short),
            NS(action="BUY", option_type=OptionType.CALL, strike=short + 200)]
    return NS(entry_credit_points=credit, lot_size=65, lots=2, structure=NS(legs=legs), metadata={"seller_exit": True})


def _snap(spot):
    return NS(option_chain=NS(spot=spot))


def test_take_profit_at_half_the_credit():
    assert _evaluate_seller_exit(_pos(), _snap(22400), current_value_points=10.0).reason == "SELLER_TAKE_PROFIT"


def test_stop_at_twice_the_credit():
    assert _evaluate_seller_exit(_pos(), _snap(22450), current_value_points=60.0).reason == "SELLER_STOP_CREDIT"


def test_strike_buffer_stop_when_spot_nears_the_short_call():
    assert _evaluate_seller_exit(_pos(), _snap(22555), current_value_points=30.0).reason == "SELLER_STRIKE_BUFFER"


def test_holds_through_a_pop_that_never_threatens_the_short():
    # the 08-05 failure mode: a 68pt pop with a wide buffer must NOT exit the seller
    assert _evaluate_seller_exit(_pos(), _snap(22470), current_value_points=26.0) is None
