# market_ai/ai_strategy_backtest.py
from __future__ import annotations
import argparse
import logging
import os
import sys
from typing import Any, Dict

import pandas as pd

from market_ai.modules.backtest_engine_adapter import run_backtest as adapter_run
from market_ai.modules.data_fetch.rolling_expired_options import RollingExpiredOptionsMarket

market = RollingExpiredOptionsMarket(
    client_id=os.environ["DHAN_CLIENT_ID"],
    access_token=os.environ["DHAN_ACCESS_TOKEN"],
    # add these overrides:
    throttle_s=0.35,
    max_requests=8,
    neighborhood_span=1,
    seed_otm_pct=0.035,
    min_premium=10.0,
    iv_fallback=0.20,
)


def _build_market(market_kind: str, underlying_seg: str):
    """
    Build a market object exposing: get_option_chain(underlying_id, expiry, **kwargs)

    Choices:
      - "historical": reads snapshots from SQLite (oc_history.sqlite) via HistoricalOptionChain.
                      Use when you've stored past option-chain rows.
      - "live":       DHAN live OptionChain (current/future expiries only).
      - "rolling":    DHAN expired-options rolling charts API (minute OHLC/IV/OI for past 5y).
                      Best for true backtests without your own history DB.
    """
    if market_kind == "historical":
        from market_ai.modules.data_fetch.optionchain_repo import OptionChainRepo
        from market_ai.modules.data_fetch.historical_option_chain import HistoricalOptionChain
        db_path = os.environ.get("OC_HISTORY_DB", "oc_history.sqlite")
        repo = OptionChainRepo(db_path)
        return HistoricalOptionChain(repo, default_seg=underlying_seg)

    if market_kind == "rolling":
        # Uses DHAN expired options rolling API.
        # Requires DHAN_CLIENT_ID and DHAN_ACCESS_TOKEN in env.
        cid = os.environ.get("DHAN_CLIENT_ID")
        tok = os.environ.get("DHAN_ACCESS_TOKEN")
        if not cid or not tok:
            raise RuntimeError("rolling market selected but DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN not set.")
        from market_ai.modules.data_fetch.rolling_expired_options import RollingExpiredOptionsMarket
        # For indices, DHAN examples use NSE_FNO + OPTIDX.
        # You can change exchange_segment/instrument if needed.
        return RollingExpiredOptionsMarket(
            client_id=cid,
            access_token=tok,
            exchange_segment=underlying_seg if underlying_seg else "NSE_FNO",
            instrument="OPTIDX",
            expiry_flag="MONTH",   # we roll monthlies
            interval="5",          # 5-min candles; balance between size and fidelity
            risk_free_rate=0.06,
        )

    # default: live
    from market_ai.modules.data_fetch.dhan_api import make_client
    from market_ai.modules.data_fetch.dhan_option_chain import OptionChain
    client = make_client()
    return OptionChain(client, default_seg=underlying_seg)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="ai_backtest")
    parser.add_argument("--input", dest="predictions_csv", required=True,
                        help="CSV with at least a 'date' column (and ideally 'close').")
    parser.add_argument("--capital", type=int, default=500000)
    parser.add_argument("--strategy", default="monthly_strangle_with_weekly_hedge")
    parser.add_argument("--underlying-id", type=int, default=13)
    parser.add_argument("--underlying-seg", default="NSE_FNO",
                        help="Exchange segment (e.g., NSE_FNO or IDX_I). Rolling API typically uses NSE_FNO.")
    parser.add_argument("--lot-size", type=int, default=50)
    parser.add_argument("--market", choices=["historical", "live", "rolling"], default="rolling",
                        help="historical: SQLite snapshots | live: DHAN live OC | rolling: DHAN expired options API")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    log = logging.getLogger("ai_backtest")

    # 1) Load input CSV
    try:
        df = pd.read_csv(args.predictions_csv)
    except Exception as e:
        log.error("Failed to read --input CSV '%s': %s", args.predictions_csv, e)
        return 2
    if "date" not in df.columns:
        log.error("Input CSV must contain a 'date' column.")
        return 2

    # 2) Build market adapter
    try:
        market = _build_market(args.market, args.underlying_seg)
    except Exception as e:
        log.error("Failed to build market adapter (%s): %s", args.market, e, exc_info=args.debug)
        return 2

    # 3) Strategy config
    cfg: Dict[str, Any] = {
        "strategy": args.strategy,
        "capital": args.capital,
        "underlying_id": args.underlying_id,
        "underlying_seg": args.underlying_seg,
        "lot_size": args.lot_size,
        "plot": args.plot,
        "debug": args.debug,
        # The strategy requires this:
        "market": market,
    }

    # 4) Run backtest
    try:
        log.debug("Invoking adapter run_backtest with strategy=%s market=%s",
                  args.strategy, args.market)
        trades_df, metrics_df, summary = adapter_run(df, cfg)
    except Exception as e:
        log.error("Backtest failed: %s", e, exc_info=args.debug)
        return 1

    # 5) Report
    total_pnl = float(summary.get("net_credit", 0.0))
    log.info("Backtest complete. Market=%s | Monthly cycles: %s | Trades: %s | Total PNL: %.2f",
             args.market,
             summary.get("monthly_cycles", summary.get("n_trades", "NA")),
             summary.get("n_trades", "NA"),
             total_pnl)

    # 6) Save outputs
    try:
        trades_path = "backtest_trades.csv"
        trades_df.to_csv(trades_path, index=False)
        log.info("Trades saved to %s", trades_path)

        summary_path = "backtest_summary.csv"
        pd.DataFrame([summary]).to_csv(summary_path, index=False)
        log.info("Summary saved to %s", summary_path)
    except Exception as e:
        log.warning("Failed to write output CSVs: %s", e, exc_info=args.debug)

    print(f"\n=== TOTAL PNL (cashflow) : {total_pnl:.2f} ===\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
