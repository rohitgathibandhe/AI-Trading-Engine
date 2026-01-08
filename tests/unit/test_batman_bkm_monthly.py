from datetime import date

from market_ai.modules.strategies.batman_bkm_monthly import BatmanBKMConfig, BatmanBKMStrategy


def _stub_chain(strikes, price=50.0):
    chain = []
    for opt in ("CE", "PE"):
        for k in strikes:
            chain.append({"option_type": opt, "strike": k, "ltp": price, "security_id": f"{opt}{k}"})
    return chain


def test_strike_builder():
    cfg = BatmanBKMConfig(base_distance_points=400, inner_step_points=200, outer_step_points=800, strike_rounding=50)
    strat = BatmanBKMStrategy(cfg)
    strikes = strat._build_strikes(20050, cfg.base_distance_points)
    assert strikes["ce_buy"] == 20050 + 400
    assert strikes["ce_sell"] == 20050 + 400 + 200
    assert strikes["ce_hedge"] == strikes["ce_sell"] + 800
    assert strikes["pe_buy"] == 20050 - 400
    assert strikes["pe_sell"] == strikes["pe_buy"] - 200
    assert strikes["pe_hedge"] == strikes["pe_sell"] - 800


def test_widening_stops_after_iterations():
    cfg = BatmanBKMConfig(max_credit_pct=1.0, estimated_margin=1.0, max_widen_iterations=1, base_distance_points=400)
    strat = BatmanBKMStrategy(cfg)
    strikes = [19650, 19850, 20450, 20650, 21450]
    chain = _stub_chain(strikes, price=100.0)
    basket, reason = strat.maybe_enter(20050, chain, date(2026, 1, 27))
    assert basket is None
    assert reason == "CREDIT_TOO_HIGH_AFTER_WIDEN"


def test_balance_direction_increases_hedge_qty(monkeypatch):
    cfg = BatmanBKMConfig()
    strat = BatmanBKMStrategy(cfg)
    strikes = strat._build_strikes(20050, cfg.base_distance_points)
    chain = _stub_chain(strikes.values(), price=10.0)
    legs = strat._build_legs(strikes, chain, hedge_q_ce=2, hedge_q_pe=2)
    # force imbalance: make upside loss worse so CE hedge increments
    strat._max_losses = lambda legs, atm: (-20000.0, -5000.0)  # type: ignore
    h_ce, h_pe, balanced = strat._balance_hedges(legs, strikes["atm"], 2, 2)
    assert balanced
    assert h_ce > 2
