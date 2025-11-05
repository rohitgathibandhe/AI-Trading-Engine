"""
Iron condor simplified backtest simulation.

This example checks SELL signals and simulates short iron condor PnL
with a conservative model (keeps small premium when move small, large loss when move big).
"""
from __future__ import annotations
from dataclasses import dataclass
import logging
from typing import Any, Dict, Tuple
import pandas as pd

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


@dataclass
class Strategy:
    config: Dict[str, Any]
    capital: float = 0.0

    def run_backtest(self, df=None, cfg=None, *args, **kwargs) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        # merge config
        if cfg is None:
            cfg = dict(self.config or {})
        else:
            merged = dict(self.config or {})
            merged.update(cfg)
            cfg = merged

        if df is None:
            csv = cfg.get("predictions_csv")
            if not csv:
                raise ValueError("No input df or cfg['predictions_csv'] provided")
            df = pd.read_csv(csv, parse_dates=["date"])
        df = df.sort_values("date").reset_index(drop=True)

        if "pred_signal" in df.columns and "pred" not in df.columns:
            df = df.rename(columns={"pred_signal": "pred"})
        # unify prob
        for c in ("prob", "prob_pos", "prob_score"):
            if c in df.columns:
                df = df.rename(columns={c: "prob"})

        if "close" not in df.columns:
            for cand in ("last_price", "ltp"):
                if cand in df.columns:
                    df = df.rename(columns={cand: "close"})
                    break
        if "close" not in df.columns:
            raise ValueError("Missing 'close' column")

        prob_threshold = float(cfg.get("prob_threshold", 0.6))
        hold_days = int(cfg.get("hold_days", 7))
        min_credit = float(cfg.get("core", {}).get("min_total_credit_per_lot", cfg.get("min_total_credit_per_lot", 2500.0)))

        skip_sells = bool(cfg.get("skip_sells", kwargs.get("skip_sells", False)))
        vix_used = kwargs.get("vix_used", cfg.get("vix_used", None))

        if skip_sells:
            return pd.DataFrame(), pd.DataFrame(), {"initial_capital": self.capital, "final_equity": self.capital, "n_trades": 0, "total_net_profit": 0.0}

        sells = df[(df.get("pred").astype(str).str.upper() == "SELL") & (df.get("prob", 0) >= prob_threshold)]
        trades = []
        equity = float(self.capital or cfg.get("capital", 0.0))
        equity_by_date = {d: equity for d in sorted(df["date"].unique())}

        unique_dates = sorted(df["date"].unique())
        for _, r in sells.iterrows():
            entry_date = r["date"]
            entry_idx = unique_dates.index(pd.Timestamp(entry_date))
            exit_idx = min(entry_idx + hold_days, len(unique_dates)-1)
            exit_date = unique_dates[exit_idx]
            entry_close = float(r["close"])
            exit_close = float(df[df["date"] == exit_date].iloc[0]["close"])
            prob = float(r.get("prob", 0.6))
            premium = min_credit * prob  # simple mapping
            move = abs((exit_close - entry_close) / entry_close)
            # iron condor: small move = keep more, big move = heavy loss (capped)
            if move <= 0.03:
                pnl = premium * 0.7
            else:
                pnl = -min(premium * 12.0, move * (self.capital or cfg.get("capital", 1e5)))
            trades.append({
                "entry_date": entry_date,
                "exit_date": exit_date,
                "entry_close": entry_close,
                "exit_close": exit_close,
                "premium": premium,
                "pnl": pnl,
                "prob": prob,
            })
            equity_by_date[exit_date] = equity_by_date.get(exit_date, equity) + pnl
            equity = equity_by_date[exit_date]

        returns = []
        cum = float(self.capital or cfg.get("capital", 0.0))
        for d in sorted(df["date"].unique()):
            if d in equity_by_date:
                cum = equity_by_date[d]
            returns.append({"date": d, "equity": cum})

        trades_df = pd.DataFrame(trades)
        returns_df = pd.DataFrame(returns)
        summary = {
            "initial_capital": float(self.capital or cfg.get("capital", 0.0)),
            "final_equity": float(returns_df["equity"].iat[-1] if not returns_df.empty else self.capital),
            "n_trades": len(trades_df),
            "total_net_profit": float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0,
            "vix_used": vix_used,
        }
        return trades_df, returns_df, summary
