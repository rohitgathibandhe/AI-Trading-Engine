#!/usr/bin/env python3
from __future__ import annotations

import os
import json
from datetime import datetime

from market_ai.dhan_wrapper import DhanWrapper
from market_ai.modules.strategies.batman_bkm_monthly import BatmanBKMConfig, BatmanBKMStrategy
from data_engine.market_ai.start_agent import _fetch_monthly_expiry, _map_chain, INDEX_SECURITY_ID, INDEX_EXCHANGE_SEG


def main():
    dw = DhanWrapper(logger=None)
    expiry = _fetch_monthly_expiry(dw)
    if not expiry:
        print("No expiry found")
        return
    spot = float(dw.get_ltp_once(INDEX_EXCHANGE_SEG, INDEX_SECURITY_ID) or 0.0)
    chain_raw = dw.get_option_chain(INDEX_SECURITY_ID, INDEX_EXCHANGE_SEG, expiry.isoformat())
    chain = _map_chain(chain_raw, expiry, symbol="NIFTY", spot=spot)

    cfg = BatmanBKMConfig()
    strat = BatmanBKMStrategy(cfg)
    basket, reason = strat.maybe_enter(spot, chain, expiry)
    print(f"Attempt reason={reason} expiry={expiry}")
    if not basket:
        return
    max_up, max_down = strat._max_losses(basket.legs, strat._build_strikes(spot, cfg.base_distance_points)["atm"])
    out = {
        "expiry": expiry.isoformat(),
        "spot": spot,
        "net_credit": basket.net_credit,
        "margin": basket.margin_required,
        "credit_pct": basket.credit_pct,
        "hedge_qty_call": basket.hedge_qty_call,
        "hedge_qty_put": basket.hedge_qty_put,
        "max_loss_up": max_up,
        "max_loss_down": max_down,
        "legs": [
            {
                "opt": leg.option_type,
                "side": leg.side,
                "strike": leg.strike,
                "qty": leg.qty,
                "entry": leg.entry,
                "ltp": leg.ltp,
                "sec_id": leg.security_id,
            }
            for leg in basket.legs
        ],
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
