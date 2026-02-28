from datetime import date

from market_ai.modules.strategies.batman_bkm_monthly import BatmanBKMConfig, BatmanBKMStrategy, Leg


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
    # Include all strikes required for base distance (400) and one widened pass (+100).
    strikes = [
        18650, 19450, 19650, 20450, 20650, 21450,  # base 400 structure
        18550, 19350, 19550, 20550, 20750, 21550,  # widened (+100) structure
    ]
    chain = _stub_chain(strikes, price=20.0)
    # Force very high credit: make short strikes expensive and long/hedge strikes cheap.
    short_strikes = {19450, 20650, 19350, 20750}
    for row in chain:
        row["ltp"] = 220.0 if row["strike"] in short_strikes else 10.0
    basket, reason = strat.maybe_enter(20050, chain, date(2026, 1, 27))
    assert basket is None
    assert reason == "CREDIT_TOO_HIGH_AFTER_WIDEN"


def test_balance_direction_increases_hedge_qty(monkeypatch):
    cfg = BatmanBKMConfig()
    strat = BatmanBKMStrategy(cfg)
    strikes = strat._build_strikes(20050, cfg.base_distance_points)
    chain = _stub_chain(strikes.values(), price=10.0)
    legs = strat._build_legs(strikes, chain, hedge_q_ce=2, hedge_q_pe=2)
    # force imbalance and converge only after CE hedge quantity increases.
    def _fake_max_losses(current_legs, _atm):
        ce_hedge_qty = max(
            l.qty for l in current_legs if str(l.option_type).upper() == "CE" and str(l.side).upper() == "BUY"
        )
        ce_hedge_lots = ce_hedge_qty / cfg.lot_size
        return (-20000.0 + max(0.0, ce_hedge_lots - 2.0) * 6000.0, -5000.0)

    strat._max_losses = _fake_max_losses  # type: ignore
    h_ce, h_pe, balanced = strat._balance_hedges(legs, strikes["atm"], 2, 2)
    assert balanced
    assert h_ce > 2


def test_quality_blocks_extreme_tail_asymmetry():
    cfg = BatmanBKMConfig()
    strat = BatmanBKMStrategy(cfg)
    legs = [
        Leg(option_type="PE", side="BUY", strike=24750.0, qty=65, entry=102.6),
        Leg(option_type="PE", side="SELL", strike=24500.0, qty=195, entry=73.4),
        Leg(option_type="PE", side="BUY", strike=23000.0, qty=130, entry=15.9),
        Leg(option_type="CE", side="BUY", strike=26300.0, qty=65, entry=86.8),
        Leg(option_type="CE", side="SELL", strike=26500.0, qty=195, entry=53.05),
        Leg(option_type="CE", side="BUY", strike=27200.0, qty=130, entry=12.05),
    ]
    quality = strat.assess_legs_quality(
        legs=legs,
        spot=25446.35,
        margin_required=1_000_000.0,
        net_credit=8713.25,
    )
    assert quality["ok"] is False
    assert "OUTER_WING_ASYMMETRY_HIGH" in quality["reasons"]
    assert "TAIL_RISK_ASYMMETRY_HIGH" in quality["reasons"]


def test_maybe_enter_quality_fail_reason_propagates():
    cfg = BatmanBKMConfig(
        max_credit_pct=100.0,
        min_credit_pct=0.0,
        max_widen_iterations=0,
        min_short_distance_points=5000.0,
        quality_block_on_fail=True,
    )
    strat = BatmanBKMStrategy(cfg)
    strikes = [18650, 19450, 19650, 20450, 20650, 21450]
    chain = _stub_chain(strikes, price=20.0)
    basket, reason = strat.maybe_enter(20050, chain, date(2026, 1, 27))
    assert basket is None
    assert reason == "SHORT_DISTANCE_TOO_TIGHT"


def test_maybe_enter_sets_quality_on_basket():
    cfg = BatmanBKMConfig(
        max_credit_pct=100.0,
        min_credit_pct=0.0,
        min_short_distance_points=0.0,
        min_short_width_points=0.0,
        max_center_offset_points=1_000_000.0,
        max_outer_distance_ratio=99.0,
        max_tail_loss_imbalance_abs=1_000_000_000.0,
        max_tail_loss_imbalance_ratio=1.0,
        max_worst_loss_to_credit_ratio=1_000_000.0,
        max_widen_iterations=0,
    )
    strat = BatmanBKMStrategy(cfg)
    strikes = [18650, 19450, 19650, 20450, 20650, 21450]
    chain = _stub_chain(strikes, price=20.0)
    basket, reason = strat.maybe_enter(20050, chain, date(2026, 1, 27))
    assert reason == "ENTER"
    assert basket is not None
    assert basket.quality_status == "PASS"
    assert basket.quality_score > 0
    assert isinstance(basket.quality_metrics, dict)
