# market_ai/modules/strategies/example_strategy.py
"""Minimal example strategy module used as fallback.
   Production-style, spaces-only indentation.
"""

from typing import Any, Dict, Optional
import logging
import pandas as pd

log = logging.getLogger("market_ai.modules.strategies.example_strategy")
log.addHandler(logging.NullHandler())


class Strategy:
    """Minimal pluggable strategy object expected by the backtest engine."""

    def __init__(self, config: Dict[str, Any], capital: float):
        """
        config: the strategy-specific config dict loaded from config/strategies/*
        capital: allocated capital for this strategy (float)
        """
        self.config = config or {}

        try:
            self.capital = float(capital)
        except Exception:
            # fallback if something weird gets passed
            logging.warning(
                "[example_strategy] capital arg not numeric: %s -> defaulting to 0.0",
                type(capital),
            )
            self.capital = 0.0

        log.info("ExampleStrategy initialized (capital=%s)", self.capital)

    def run_backtest(self, start: str = "2024-01-01", end: str = "2024-12-31"):
        """
        Minimal backtest runner that the engine can call.
        It reads the 'predictions' CSV path from config if present, otherwise does nothing.
        Returns a small result tuple (trades_df, returns_df, summary_dict) to match engine expectations.
        """
        pred_path = self.config.get("predictions_csv", "predictions_latest.csv")
        log.info("ExampleStrategy.run_backtest start=%s end=%s pred=%s", start, end, pred_path)

        try:
            df = pd.read_csv(pred_path, parse_dates=["date"]) if pred_path else pd.DataFrame()
        except Exception as e:
            log.exception("Failed to load predictions CSV '%s': %s", pred_path, e)
            df = pd.DataFrame()

        # no trading logic here — return zero-trade structures so engine can continue
        trades_df = pd.DataFrame(
            columns=[
                "signal_date",
                "entry_date",
                "exit_date",
                "signal",
                "entry",
                "exit",
                "qty",
                "lots",
                "raw_ret",
                "net_ret",
                "profit",
            ]
        )
        returns_df = pd.DataFrame(columns=["date", "daily_pct_return", "equity"])
        summary = {
            "initial_capital": float(self.capital),
            "final_equity": float(self.capital),
            "n_trades": int(0),
            "total_net_profit": float(0.0),
        }

        log.info("ExampleStrategy.run_backtest completed (no trades).")
        return trades_df, returns_df, summary
