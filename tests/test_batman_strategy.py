from types import SimpleNamespace

import pandas as pd
import pytest

from market_ai.modules.strategies.monthly_strangle_with_weekly_hedge import (
    BatmanConfig,
    BatmanStructure,
    LiveConfig,
    _ensure_batman_hedges,
    _select_replacement,
)

pytestmark = pytest.mark.unit


def _sample_chain() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"strike": 25500.0, "ltp_ce": 8.0, "delta_ce": 0.10, "ltp_pe": 10.0, "delta_pe": -0.12},
            {"strike": 25700.0, "ltp_ce": 5.0, "delta_ce": 0.06, "ltp_pe": 14.0, "delta_pe": -0.16},
            {"strike": 25900.0, "ltp_ce": 3.0, "delta_ce": 0.04, "ltp_pe": 20.0, "delta_pe": -0.20},
            {"strike": 26100.0, "ltp_ce": 6.0, "delta_ce": 0.05, "ltp_pe": 24.0, "delta_pe": -0.25},
        ]
    )


def test_select_replacement_picks_first_valid_call():
    df = _sample_chain()
    row = _select_replacement(df, target_strike=25600.0, opt_type="CALL", direction="UP")
    assert row is not None
    assert row["strike"] >= 25600.0


def test_ensure_batman_hedges_places_order_when_short_unhedged(monkeypatch):
    chain_df = _sample_chain()
    captured = []

    def fake_place_option_order(dw, cfg, side, quantity, row_dict, opt_type, context):
        captured.append((side, quantity, row_dict["strike"], context["tag"]))
        return True

    from market_ai.modules.strategies import monthly_strangle_with_weekly_hedge as strat_mod

    monkeypatch.setattr(strat_mod, "_place_option_order", fake_place_option_order)

    structure = BatmanStructure(
        expiry="2025-11-25",
        exchange_seg="NSE_FNO",
        short_call={"netQty": -150, "costPrice": 50.0},
        short_call_strike=25900.0,
        short_put={"netQty": -150, "costPrice": 40.0},
        short_put_strike=25500.0,
        long_call={"netQty": 0},
        long_call_strike=None,
        long_put={"netQty": 0},
        long_put_strike=None,
    )

    cfg = LiveConfig(lot_size=150)
    bat_cfg = BatmanConfig(hedge_price_max=10.0, hedge_delta_max=0.2)

    _ensure_batman_hedges(
        dw=SimpleNamespace(),
        cfg=cfg,
        bat_cfg=bat_cfg,
        structure=structure,
        chain_df=chain_df,
        spot=25600.0,
    )

    assert captured, "expected at least one hedge order"
    assert captured[0][0] == "BUY"
