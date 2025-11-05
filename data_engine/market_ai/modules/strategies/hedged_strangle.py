"""
Hedged strangle strategy (example simulation).

This implementation is deterministic and simple:
 - picks SELL signals from the predictions CSV using pred/prob threshold
 - opens a trade on the 'date' and exits after hold_days (or at last available date)
 - collects an entry premium proportional to 'prob'
 - computes PnL using a simple underlying-move model
 - respects skip_sells flag (no trades when skip_sells=True)
"""
from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import numpy as np

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


@dataclass
class Strategy:
    config: Dict[str, Any]
    capital: float = 0.0

    def run_backtest(self, df=None, cfg=None, *args, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        # normalize incoming args
        if cfg is None:
            cfg = self.config or {}
        else:
            # overlay provided cfg on inst config
            merged = dict(self.config or {})
            merged.update(cfg)
            cfg = merged

        if df is None:
            # attempt to load from cfg
            csv = cfg.get("predictions_csv")
            if not csv:
                raise ValueError("No input df or cfg['predictions_csv'] provided")
            df = pd.read_csv(csv, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)

        # normalize column names
        if "pred_signal" in df.columns and "pred" not in df.columns:
            df = df.rename(columns={"pred_signal": "pred"})
        # detect prob column
        prob_col = None
        for candidate in ("prob", "prob_pos", "prob_score"):
            if candidate in df.columns:
                prob_col = candidate
                break
        if prob_col and prob_col != "prob":
            df = df.rename(columns={prob_col: "prob"})

        # ensure close column
        if "close" not in df.columns:
            # try last_price or ltp
            for cand in ("last_price", "ltp", "close_price"):
                if cand in df.columns:
                    df = df.rename(columns={cand: "close"})
                    break
        if "close" not in df.columns:
            raise ValueError("Predictions CSV must include 'close' column")

        prob_threshold = float(cfg.get("prob_threshold", 0.6))
        hold_days = int(cfg.get("hold_days", 7))
        profit_take_pct = float(cfg.get("profit_take_pct", 0.6))
        min_credit = float(cfg.get("core", {}).get("min_total_credit_per_lot", cfg.get("min_total_credit_per_lot", 3000.0)))
        max_lots = int(cfg.get("max_lots", 1))
        lot_size = int(cfg.get("lot_size", cfg.get("lot_size", 1)))

        skip_sells = bool(cfg.get("skip_sells", kwargs.get("skip_sells", False)))
        vix_used = kwargs.get("vix_used", cfg.get("vix_used", None))

        if skip_sells:
            log.info("skip_sells set -> no sells performed (hedged_strangle)")
            return pd.DataFrame(), pd.DataFrame(), {"initial_capital": self.capital, "final_equity": self.capital, "n_trades": 0, "total_net_profit": 0.0}

        # find sell signals (accept both 'SELL' string or numeric 1)
        sells = df[(df.get("pred").astype(str).str.upper() == "SELL") & (df.get("prob", 0) >= prob_threshold)]
        trades = []
        equity = float(self.capital or cfg.get("capital", 0.0))
        equity_timeline = []
        equity_dates = sorted(df["date"].unique())

        # map date -> row index for quick lookups
        date_to_idx = {d: i for i, d in enumerate(sorted(df["date"].unique()))}
        # convert df index to unique dates
        rows = df

        for _, row in sells.iterrows():
            entry_date = row["date"]
            entry_idx = rows.index.get_loc(row.name)
            # find exit index by date offset
            # use unique date array to step forward hold_days
            unique_dates = sorted(rows["date"].unique())
            try:
                pos = unique_dates.index(pd.Timestamp(entry_date))
            except ValueError:
                pos = 0
            exit_pos = min(pos + hold_days, len(unique_dates) - 1)
            exit_date = unique_dates[exit_pos]

            # find close prices
            entry_close = float(row["close"])
            exit_rows = rows[rows["date"] == exit_date]
            if exit_rows.empty:
                exit_close = entry_close
            else:
                exit_close = float(exit_rows.iloc[0]["close"])

            prob = float(row.get("prob", 0.6))
            premium = min_credit * prob  # example: credit proportional to prob
            # compute underlying movement
            underlying_move = abs((exit_close - entry_close) / entry_close)

            # simple risk model:
            # - if underlying_move <= 3% -> keep profit_take_pct of premium
            # - else loss proportional to movement capped at premium*10
            if underlying_move <= 0.03:
                pnl = premium * profit_take_pct
            else:
                loss = min(premium * 10.0, underlying_move * (self.capital or cfg.get("capital", 0.0)))
                pnl = -loss

            lots = max(1, min(max_lots, int(cfg.get("max_lots", 1))))
            pnl_total = pnl * lots

            trades.append({
                "entry_date": pd.Timestamp(entry_date),
                "exit_date": pd.Timestamp(exit_date),
                "entry_close": entry_close,
                "exit_close": exit_close,
                "premium_collected": premium,
                "pnl": pnl_total,
                "lots": lots,
                "prob": prob,
            })

        trades_df = pd.DataFrame(trades)
        # Build returns timeline: start with initial equity, update on exit dates
        equity_by_date = {d: equity for d in equity_dates}
        for t in trades:
            ed = pd.Timestamp(t["exit_date"])
            equity_by_date[ed] = equity_by_date.get(ed, equity) + t["pnl"]
            # ensure later dates carry forward cumulative
            equity = equity_by_date[ed]

        # Build ordered returns dataframe
        cum = float(self.capital or cfg.get("capital", 0.0))
        rows_out = []
        for d in equity_dates:
            if pd.Timestamp(d) in equity_by_date:
                cum = equity_by_date[pd.Timestamp(d)]
            rows_out.append({"date": pd.Timestamp(d), "equity": cum})
        returns_df = pd.DataFrame(rows_out)

        total_net = float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0
        summary = {
            "initial_capital": float(self.capital or cfg.get("capital", 0.0)),
            "final_equity": float((self.capital or cfg.get("capital", 0.0)) + total_net),
            "n_trades": len(trades_df),
            "total_net_profit": float(total_net),
            "vix_used": vix_used,
            "skip_sells": skip_sells,
        }
        return trades_df, returns_df, summary
